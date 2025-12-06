"""
Useful functions for PyTorch
"""

import torch
import numpy as np


def get_device():
    """
    Determine which device is available
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
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


def preprocess_state(state):
    """
    normalize
    """
    # Normalize to [0, 1]
    state = state / 255.0
    return state


def transform_state_to_tensor(state: np.array, device):
    """
    Multiframe Atari state to torch tensor
    """
    state = torch.tensor(state, dtype=torch.float32, device=device)
    # N frames, C channels, H px Height, W px Width
    state = state.permute(0, 3, 1, 2)
    state = state.unsqueeze(0)  # add batch layer
    return state
