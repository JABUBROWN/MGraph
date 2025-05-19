import os
import csv

import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

from util.scaler import NormalScaler
from util.gen_data import gen_data

def get_npz_dataset(root_path, args):
    data_path = os.path.join(root_path, args.data_file)  # MGraph/dataset/PeMS0X/PEMS0X.npz
    data = np.load(data_path)["data"][:, :, 0:1]  # (T, N, 1)
    t_idx = np.tile(np.arange(data.shape[0]).reshape(-1, 1, 1), (1, data.shape[1], 1))
    n_idx = np.tile(np.arange(data.shape[1]).reshape(1, -1, 1), (data.shape[0], 1, 1))
    return np.concatenate((data, t_idx, n_idx), axis=-1)

def get_h5_dataset(root_path, args):
    file_name = os.path.join(root_path, args.data_file)  # MGraph/dataset/xxx/xxx.h5
    df = pd.read_hdf(file_name)  # (T, N)
    data = np.expand_dims(df.values, axis=-1)  # Convert DataFrame to Numpy
    t_idx = np.tile(np.arange(data.shape[0]).reshape(-1, 1, 1), (1, data.shape[1], 1))
    n_idx = np.tile(np.arange(data.shape[1]).reshape(1, -1, 1), (data.shape[0], 1, 1))
    return np.concatenate((data, t_idx, n_idx), axis=-1)

def get_csv_dataset(root_path, args):
    file_name = os.path.join(root_path, args.data_file)
    rows = []
    with open(file_name, 'r') as f:
        f.readline()
        reader = csv.reader(f, delimiter=args.adj_file_delimiter)
        line = 0
        for row in reader:
            line += 1
            if line == 1:
                continue
            rows.append([float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7])])
    arr = np.array(rows, dtype=float)
    arr = np.expand_dims(arr, axis=-1)
    t_idx = np.tile(np.arange(arr.shape[0]).reshape(-1, 1, 1), (1, arr.shape[1], 1))
    n_idx = np.tile(np.arange(arr.shape[1]).reshape(1, -1, 1), (data.shape[0], 1, 1))
    return np.concatenate((arr, t_idx, n_idx), axis=-1)

def split_dataset(args, data, begin_weekday=0):
    total_length = data.shape[0]

    if args.model_name == "MGraph":
        tid = np.tile(np.arange(0, args.daily_time_step, 1), total_length // args.daily_time_step + 1)[:data.shape[0]]
        tid = tid.reshape((-1, 1, 1)).repeat(args.node_num, axis=1)
        dow = np.array([i % 7 for i in range(7)]).repeat(args.daily_time_step)
        dow = np.array([dow[i % len(dow)] for i in range(total_length)])
        dow = dow.reshape((-1, 1, 1)).repeat(args.node_num, axis=1)
        data = np.concatenate((data, tid, dow), axis=-1)
        n_pos = np.arange(args.node_num)
        n_pos = n_pos.reshape((1, -1, 1))
        n_pos = np.repeat(n_pos, data.shape[0], axis=0)
        daily = np.tile(np.arange(0, args.daily_time_step, 1), total_length // args.daily_time_step + 1)[:data.shape[0]]
        daily = daily.reshape((-1, 1, 1)).repeat(args.node_num, axis=1)
        weekly = np.array([(i + begin_weekday) % 7 for i in range(7)]).repeat(args.daily_time_step)
        weekly = np.array([weekly[i % len(weekly)] for i in range(total_length)])
        weekly = weekly.reshape((-1, 1, 1)).repeat(args.node_num, axis=1)
        data = np.concatenate((data, n_pos, daily, weekly), axis=-1)

    idx_1 = int(total_length * args.train_ratio)
    idx_2 = int(total_length * args.val_ratio) + idx_1
    train_data = data[:idx_1, ...]
    val_data = data[idx_1: idx_2, ...]
    test_data = data[idx_2:, ...]
    return train_data, val_data, test_data

def scale_dataset(args, train_data, val_data, test_data):
    scaler = eval(args.scaler)(train_data)
    train_data = scaler.scale(train_data)
    val_data = scaler.scale(val_data)
    test_data = scaler.scale(test_data)
    return train_data, val_data, test_data, scaler

def generate_window(args, data):
    window_length = args.given_time_step + args.predict_time_step
    given_data_list = []
    predict_data_list = []
    for t in range(len(data) - window_length + 1):
        given_data = data[t: t + args.given_time_step, ...]
        predict_data = data[t + args.given_time_step: t + window_length, ...]
        given_data_list.append(given_data)
        predict_data_list.append(predict_data)
    return np.array(given_data_list), np.array(predict_data_list)

def convert_to_tensor(args, X, y, shuffle, drop_last):
    X = torch.tensor(X, dtype=torch.float32, device=args.device)
    y = torch.tensor(y, dtype=torch.float32, device=args.device)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader

def load_dataset(root_path, args):
    if args.data_file.endswith('npz'):
        data = get_npz_dataset(root_path, args)
    elif args.data_file.endswith('csv'):
        data = get_csv_dataset(root_path, args)
    elif args.data_file.endswith('h5'):
        data = get_h5_dataset(root_path, args)
    else:
        raise NotImplementedError("Unsupported datafile type.")

    if args.model_name == "MGraph":
        print("Applying the weekly embeddings.")
        data, begin_weekday = gen_data(args, data)
    else:
        begin_weekday = 0

    train_data, val_data, test_data = split_dataset(args, data, begin_weekday)
    train_data, val_data, test_data, scaler = scale_dataset(args, train_data, val_data, test_data)
    train_X, train_y = generate_window(args, train_data)
    val_X, val_y = generate_window(args, val_data)
    test_X, test_y = generate_window(args, test_data)
    train_dataloader = convert_to_tensor(args, train_X, train_y, shuffle=True, drop_last=False)
    val_dataloader = convert_to_tensor(args, val_X, val_y, shuffle=False, drop_last=False)
    test_dataloader = convert_to_tensor(args, test_X, test_y, shuffle=False, drop_last=False)
    return train_dataloader, val_dataloader, test_dataloader, scaler