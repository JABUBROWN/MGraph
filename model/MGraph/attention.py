import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self, in_dim, n_heads):
        super().__init__()
        self.d = in_dim // n_heads  # Dimension per head
        self.n_heads = n_heads
        self.W_Q = nn.Linear(in_dim, in_dim)
        self.W_K = nn.Linear(in_dim, in_dim)
        self.W_V = nn.Linear(in_dim, in_dim)
        self.W_out = nn.Linear(in_dim, in_dim)

    def forward(self, Q, K, V):
        batch_size = Q.size(0)

        Q = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.n_heads, self.d).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.n_heads, self.d).transpose(1, 2)

        K_T = K.transpose(-1, -2)
        attention_score = (Q @ K_T) / math.sqrt(self.d)
        attention_score = torch.softmax(attention_score, dim=-1)
        out = (attention_score @ V).transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d)
        out = self.W_out(out)
        return out

class SelfAttentionLayer(nn.Module):
    def __init__(self, model_dim, feed_forward_dim, num_heads, dropout):
        super().__init__()
        self.attn = AttentionLayer(model_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.attn(x, x, x)
        out = self.dropout(out)
        out = self.layer_norm(residual + out)

        residual = out
        out = self.feed_forward(out)
        out = self.dropout(out)
        out = self.layer_norm(residual + out)
        return out