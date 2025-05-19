import argparse
import os.path
import numpy as np
from execute.base_trainer import BaseTrainer
from execute.mgraph_trainer import MGraphTrainer
from util.check_cuda import check_cuda
from util.config_loader import load_config
from util.dataset_loader import load_dataset
from util.graph_loader import load_adj
from util.model_initializer import model_initialize
from util.seed_loader import set_seed

# Specify a config file in ./config/ directory.
DATA_CONFIG_FILE = "PeMS03.conf"
MODEL_CONFIG_FILE = "MGraph.conf"

# Get the working directory of main.py.
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

# system args
parser = argparse.ArgumentParser()

parser.add_argument("--device", default="0", type=str, choices=['0', '1', '2', '3', 'cpu'])
parser.add_argument("--stage", default="train", type=str, choices=["train", "test"])
parser.add_argument("--epoch", default=100, type=int, help="Maximum training epoch.")
parser.add_argument("--seed", default=1, type=int, help="Random Seed (usually from 1 to 10).")
parser.add_argument("--data_config_file", default=DATA_CONFIG_FILE, type=str)
parser.add_argument("--model_config_file", default=MODEL_CONFIG_FILE, type=str)

# Load the config args into the parser.
args = load_config(ROOT_PATH, parser)

# Check the availability of CUDA.
check_cuda(args)

# Set seed.
set_seed(args.seed)

# Get graph (Adjacent Matrix).
A = load_adj(ROOT_PATH, args)

# Get dataset.
train_dataloader, val_dataloader, test_dataloader, scaler = load_dataset(ROOT_PATH, args)

# Initialize model.
model, optimizer, lr_scheduler, loss, logger = model_initialize(args, A, print_params=False)

# Initialize trainer.
trainer = MGraphTrainer(args, model, optimizer, lr_scheduler, loss, train_dataloader, val_dataloader, test_dataloader, scaler, logger)

# Train model
trainer.train()
