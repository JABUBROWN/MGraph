import torch
import torch.nn as nn
import torch.nn.functional as F
from model.MGraph.attention import SelfAttentionLayer  # Import the attention layer

class SkipConnection(nn.Module):
    def __init__(self, c_in, c_out):
        super(SkipConnection, self).__init__()
        self.c_out = c_out
        self.c_in = c_in
        if self.c_in > self.c_out:
            self.mlp = nn.Linear(c_in, c_out)

    def forward(self, x):
        B, N, C = x.shape
        assert self.c_in == C
        if self.c_in < self.c_out:
            tmp = torch.cat([x, torch.zeros([B, N, self.c_out - self.c_in])], dim=-1)
        elif self.c_in > self.c_out:
            tmp = self.mlp(x)
        else:
            tmp = x
        return tmp


class gc_operation(nn.Module):
    def __init__(self, in_channels, out_channels, activation='GLU', use_bias=True):
        super(gc_operation, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bias = use_bias
        assert activation == 'GLU'
        units = out_channels * 2
        self.kernel = nn.Parameter(torch.Tensor(in_channels, units))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(units))

        self.activation = activation
        self.use_bias = use_bias

    def forward(self, features, adj):
        if len(adj.shape) == 3:
            supports = torch.einsum("lij,ljk->lik", adj, features)
        else:
            supports = torch.einsum("ij,ljk->lik", adj, features)
        output = torch.einsum("lij,jk->lik", supports, self.kernel)
        if self.use_bias:
            output += self.bias
        lhs = output[:, :, 0:self.out_channels]
        rhs = output[:, :, -self.out_channels:]
        output = lhs * torch.sigmoid(rhs)

        return output


class GCmodule(nn.Module):
    def __init__(self, in_channels, filters, num_nodes, activation, conv_channels=1):
        super(GCmodule, self).__init__()
        self.filters = filters
        self.num_nodes = num_nodes
        self.activation = activation
        self.conv_channels = conv_channels
        self.gconvs = nn.ModuleList(
            [gc_operation(in_channels=in_channels, out_channels=self.filters[i], activation=activation) for i in
             range(len(filters))])
        self.skip_connect = SkipConnection(in_channels, self.filters[0] * self.conv_channels)

    def forward(self, data, adj):
        data_indentity = data.clone()
        for i in range(len(self.filters)):
            data_gconv = self.gconvs[i](data_indentity, adj)
            data_gconv = data_gconv + self.skip_connect(data_indentity)
            data_indentity = data_gconv
        output = torch.unsqueeze(data_indentity, dim=1)
        return output


class AdjGenerator(nn.Module):
    def __init__(self, in_dim, num_of_vertices, d=10, adj_dropout=0.8):
        super(AdjGenerator, self).__init__()
        self.in_dim = in_dim
        self.num_nodes = num_of_vertices
        self.dense1 = nn.Linear(in_dim, d)
        self.dense2 = nn.Linear(d, num_of_vertices * num_of_vertices)
        self.dropout = nn.Dropout(adj_dropout)

    def forward(self, x):
        if len(x.shape) > 2:
            B, _, N, C = x.shape
            x = x.reshape((B, -1))
        else:
            B, _ = x.shape
            N = self.num_nodes
        h1 = self.dense1(x)
        adj = self.dense2(h1)
        h2 = torch.tanh(adj)
        h2 = self.dropout(h2)
        h2 = h2.reshape((B, N, N))
        return h2


class output_module(nn.Module):

    def __init__(self, num_of_vertices, input_length, num_of_features,
                 num_of_filters=128, predict_length=1):
        super(output_module, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.input_length = input_length
        self.num_of_features = num_of_features
        self.predict_length = predict_length
        self.dense1 = nn.Linear(input_length * num_of_features, num_of_filters)
        self.dense2 = nn.Linear(num_of_filters, predict_length)

    def forward(self, data):
        B, N, T, C = data.shape
        data = data.reshape((-1, N, T * C))
        data = torch.relu(self.dense1(data))
        data = self.dense2(data)
        data = data.transpose(1, 2)
        data = data.unsqueeze(3)
        return data


class MGraph(nn.Module):

    def __init__(self, input_length, predict_length, num_of_vertices, num_of_features,
                 hid_dim=64, tcn_kernel_size=3, emb_length=10, out_hid_dim=128, adj_dropout=.8,
                 TimeEncodingType=3, addLatestX=True, hasCross=True,
                 num_heads=4, dropout=0.1, ff_dim=256):  # Add attention hyperparameters
        super(MGraph, self).__init__()
        self.input_length = input_length
        self.predict_length = predict_length
        self.num_of_vertices = num_of_vertices
        self.hid_dim = hid_dim
        self.addLatestX = addLatestX
        self.hasCross = hasCross
        gcn_filters = [hid_dim]
        self.num_heads = num_heads
        self.dropout = dropout
        self.ff_dim = ff_dim

        if hasCross:
            self.data_dense = nn.Linear(num_of_features * 2, hid_dim)
        else:
            self.data_dense = nn.Linear(num_of_features, hid_dim)

        self.data_dense1 = nn.Linear(hid_dim, hid_dim)

        self.TimeEncodingType = TimeEncodingType
        if TimeEncodingType == 3:
            self.hour_dense = nn.Linear(288, hid_dim)
            self.day_dense = nn.Linear(7, hid_dim)
        elif TimeEncodingType == 2:
            self.day_dense = nn.Linear(7, hid_dim)
        elif TimeEncodingType == 1:
            self.hour_dense = nn.Linear(288, hid_dim)

        tcn_pad_l = 0 if tcn_kernel_size == 2 else 1
        tcn_pad_r = 1
        self.padding = nn.ReplicationPad2d((tcn_pad_l, tcn_pad_r, 0, 0))
        self.tcn1 = nn.Conv2d(hid_dim, hid_dim, (1, tcn_kernel_size))

        if hasCross:
            self.feat_aml_modules = nn.ModuleList(
                [AdjGenerator(num_of_vertices * num_of_features * 2, num_of_vertices, d=emb_length,
                              adj_dropout=adj_dropout) for _ in range(input_length - 1)])
        else:
            self.feat_aml_modules = nn.ModuleList(
                [AdjGenerator(num_of_vertices * num_of_features, num_of_vertices, d=emb_length, adj_dropout=adj_dropout)
                 for _ in range(input_length - 1)])

        self.feat_gc_list = nn.ModuleList(
            [GCmodule(hid_dim, gcn_filters, num_of_vertices, activation='GLU') for _ in range(input_length - 1)])
        self.feat_aml_last = AdjGenerator(num_of_vertices * num_of_features, num_of_vertices, adj_dropout=adj_dropout)
        self.feat_gc_last = GCmodule(hid_dim, gcn_filters, num_of_vertices, activation='GLU')

        # Multi-head self-attention layer
        self.temporal_attention = SelfAttentionLayer(hid_dim, self.ff_dim, self.num_heads, self.dropout)

        self.output_module_list = nn.ModuleList(
            [output_module(num_of_vertices, input_length, hid_dim, num_of_filters=out_hid_dim) for _ in
             range(predict_length)])

    def forward(self, input):
        x, t_hour, t_day = input
        latestX = x[:, -1:, :, 0:1].repeat([1, self.input_length, 1, 1])
        if self.hasCross:
            acrossx = torch.cat([x[:, :, :, 0:1], latestX], dim=-1)
            data = torch.relu(self.data_dense(acrossx))
        else:
            data = torch.relu(self.data_dense(x))

        data = self.data_dense1(data)

        if self.TimeEncodingType == 3:
            hour_data = torch.relu(self.hour_dense(t_hour)).unsqueeze(2).repeat([1, 1, self.num_of_vertices, 1])
            day_data = torch.relu(self.day_dense(t_day)).unsqueeze(2).repeat([1, 1, self.num_of_vertices, 1])
            data = data + hour_data + day_data
        elif self.TimeEncodingType == 2:
            day_data = torch.relu(self.day_dense(t_day)).unsqueeze(2).repeat([1, 1, self.num_of_vertices, 1])
            data = data + day_data
        elif self.TimeEncodingType == 1:
            hour_data = torch.relu(self.hour_dense(t_hour)).unsqueeze(2).repeat([1, 1, self.num_of_vertices, 1])
            data = data + hour_data

        data = data.transpose(1, 3)
        data = self.padding(data)
        data = self.tcn1(data)
        data = data.transpose(1, 3)
        assert data.shape[1] == self.input_length

        # Apply multi-head self-attention
        B, T, N, C = data.shape
        # Reshape for temporal attention (B * N, T, C)
        attn_input = data.permute(0, 2, 1, 3).reshape(-1, T, C)
        attn_output = self.temporal_attention(attn_input)
        # Reshape back to (B, T, N, C)
        data = attn_output.view(B, N, T, C).permute(0, 2, 1, 3)

        need_concat = []

        for i in range(self.input_length - 1):
            if self.hasCross:
                feat_adj = self.feat_aml_modules[i](acrossx[:, i:i + 1])
            else:
                feat_adj = self.feat_aml_modules[i](x[:, i:i + 1])
            feat_data = self.feat_gc_list[i](data[:, i, :, :],
                                             feat_adj)
            need_concat.append(feat_data)

        feat_adj_last = self.feat_aml_last(x[:, -1:, :, 0:1])
        feat_data_last = self.feat_gc_last(data[:, -1, :, :], feat_adj_last)
        need_concat.append(feat_data_last)

        data = torch.cat(need_concat, dim=1)

        data = data.transpose(1, 2)

        need_concat = []
        for i in range(self.predict_length):
            need_concat.append(self.output_module_list[i](data))

        if self.predict_length > 1:
            main_output = torch.cat(need_concat, dim=1)
        else:
            main_output = need_concat[0]

        if self.addLatestX:
            if self.input_length == self.predict_length:
                main_output += latestX
            else:
                main_output += latestX[:, 0:self.predict_length, :, :]

        return main_output