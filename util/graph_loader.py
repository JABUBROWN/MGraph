import csv
import os.path

import numpy as np


def load_adj(root_path, args):

    if args.data_name == "ETTh1" or args.data_name == "ETTh2" or args.data_name == "ETTm1" or args.data_name == "ETTm2":
        return None

    # For NYCBike, NYCTaxi and BJTaxi
    if args.adj_file.endswith('npz'):
        adj_file = os.path.join(root_path, args.adj_file)
        return np.load(adj_file)['adj_mx'].astype(np.float32)

    # For PeMS0X, PEMS-BAY and META-LR
    A = np.zeros((args.node_num, args.node_num), dtype=np.float32)
    id_dict = {}  # Default dict is 0->0, 1->1, 2->2. For PeMS03, it's 313344->0, 313349->1...

    # Load id_file if exists. ( Only for PeMS03 )
    if args.id_file:
        id_file = os.path.join(root_path, args.id_file)
        with open(id_file, 'r') as f:
            for idx, i in enumerate(f.read().strip().split('\n')):
                id_dict[int(i)] = idx
    else:
        for idx in range(args.node_num):
            id_dict[idx] = idx

    # Fill the Adj matrix
    adj_file = os.path.join(root_path, args.adj_file)
    with open(adj_file, 'r') as f:
        f.readline()
        reader = csv.reader(f, delimiter=args.adj_file_delimiter)  # Comma
        for row in reader:
            if len(row) != 3:
                continue  # Skip blank lines.
            i, j, dist = id_dict[int(row[0])], id_dict[int(row[1])], float(row[2])  # from, to, distance
            A[i, j] = A[j, i] = dist

    # 0-1 Scaler
    A /= np.max(A)

    # diagonal all one
    for i in range(args.node_num):
        A[i, i] = 1

    return A
