## From fastai
import argparse

import torch
import numpy as np
import random

def set_seed(s, reproducible=False):
    "Set random seed for `random`, `torch`, and `numpy` (where available)"
    try: torch.manual_seed(s)
    except NameError: pass
    try: torch.cuda.manual_seed_all(s)
    except NameError: pass
    try: np.random.seed(s%(2**32-1))
    except NameError: pass
    random.seed(s)
    if reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def to_device(t, device):
    if isinstance(t, (tuple, list)):
        return [_t.to(device) for _t in t]
    elif isinstance(t, torch.Tensor):
        return t.to(device)
    else:
        raise("Not a Tensor or list of Tensors")
    return t    

def model_size(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def parse_args(config):
    parser = argparse.ArgumentParser(description='Run training baseline')
    for k,v in config.__dict__.items():
        parser.add_argument('--'+k, type=type(v), default=v)
    args = vars(parser.parse_args())
    
    # update config with parsed args
    for k, v in args.items():
        setattr(config, k, v)