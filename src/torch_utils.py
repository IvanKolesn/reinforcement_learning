"""
Useful functions for PyTorch
"""

import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Используемое устройство: {device}")
    return device


def init_weights(module):
    """
    Xavier uniform initialization for Linear layers
    """
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.xavier_uniform(module.weight)
        module.bias.data.fill_(0.01)
