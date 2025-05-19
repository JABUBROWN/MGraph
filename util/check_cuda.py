import torch


def check_cuda(args):
    if args.device != "cpu" and torch.cuda.is_available():
        torch.cuda.set_device(int(args.device))
        args.device = "cuda:{}".format(args.device)
    else:
        args.device = "cpu"
