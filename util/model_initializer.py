import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR
from torch.nn import L1Loss
from util.logger import Logger
from model.MGraph.model import MGraph

def model_initialize(args, A, print_params=False):

    # Init model
    if args.model_name == "MGraph":
        model = MGraph(args.given_time_step, args.predict_time_step, args.node_num, args.input_dim, args.hidden_dim, 3, args.embed_dim, 128, 0.8, 3, True, True).to(args.device)
    else:
        raise NotImplementedError("Unsupported Model.")

    # Init model values
    if args.xavier_init:
        for p in model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.uniform_(p)

    # Print params
    if print_params:
        for name, param in model.named_parameters():
            print(name, param.shape, param.requires_grad)

    # Count params number
    total_num = sum([param.nelement() for param in model.parameters()])

    # Init Optimizer and Loss
    optimizer = eval(args.optimizer)(params=model.parameters(), lr=args.lr_init, weight_decay=args.weight_decay)

    # Init Scheduler
    scheduler_params = args.lr_scheduler.split(',')
    if args.lr_scheduler_name == "MultiStepLR":
        lr_decay_steps = [int(i) for i in scheduler_params[2:]]
        lr_scheduler = MultiStepLR(optimizer=optimizer, milestones=lr_decay_steps, gamma=args.gamma)
    elif args.lr_scheduler_name == "ExponentialLR":
        lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=args.gamma)

    # Init Loss
    loss = eval(args.loss)().to(args.device)  # L1Loss

    # Init Logger
    logger = Logger(args)

    # Print Intro
    logger.log_and_print('Model: {}, Dataset: {}, Seed: {}, Device: {}'.format(args.model_name, args.data_name, args.seed, args.device))
    logger.log_and_print('Total params num: {}'.format(total_num))

    return model, optimizer, lr_scheduler, loss, logger