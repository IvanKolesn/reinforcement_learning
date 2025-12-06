"""
Advantage Actor Critic algorithm
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as td


def compute_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    terminated: torch.Tensor,
    device,
) -> torch.Tensor:
    returns = []
    next_value = 0

    for t in reversed(range(len(rewards))):
        # If terminated, next value is 0
        if terminated[t] or t == len(rewards) - 1:
            returns_t = rewards[t]
            next_value = 0
        else:
            returns_t = rewards[t] + gamma * next_value
        returns.insert(0, returns_t)
        next_value = returns_t

    return torch.stack(returns).to(device)


class ActorNet(nn.Module):
    """
    Predicts next action for the car
    """

    def __init__(self):
        super().__init__()
        self.conv_model = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.linear_model = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6),  # 3 means + 3 log_stds
        )

    def forward(self, state: torch.Tensor):
        """
        Get model prediction
        """
        batch_size, frames, channels, height, width = state.size()
        state = state.reshape(batch_size, frames * channels, height, width)
        features = self.conv_model(state)
        out = self.linear_model(features)
        means, log_stds = out.chunk(2, dim=-1)
        log_stds = torch.clamp(log_stds, -5, 2)
        return means, log_stds

    def get_actions(self, state):
        """
        Transform prediction into actions using gaussian policy
        """
        means, log_stds = self.forward(state)
        stds = torch.exp(log_stds)
        # action sample from normal distribution
        base_dist = td.Normal(means, stds)
        raw_actions = base_dist.sample()

        # Apply different transforms to each action dimension
        transformed_actions = torch.zeros_like(raw_actions)

        # Action 1 (Steering): tanh transform (range [-1, 1])
        transformed_actions[:, 0] = torch.tanh(raw_actions[:, 0])

        # Action 2 (Acceleration): sigmoid transform (range [0, 1])
        transformed_actions[:, 1] = torch.sigmoid(raw_actions[:, 1])

        # Action 3 (Breaking): sigmoid transform (range [0, 1])
        transformed_actions[:, 2] = torch.sigmoid(raw_actions[:, 2])

        return raw_actions, transformed_actions

    def get_log_prob_given_actions(self, state, raw_actions):
        """
        Transform prediction into actions using gaussian policy
        """
        means, log_stds = self.forward(state)
        stds = torch.exp(log_stds)
        # action sample from normal distribution
        base_dist = td.Normal(means, stds)

        # Apply different transforms to each action dimension
        transformed_actions = torch.zeros_like(raw_actions)

        # Action 1 (Steering): tanh transform (range [-1, 1])
        transformed_actions[:, 0] = torch.tanh(raw_actions[:, 0])

        # Action 2 (Acceleration): sigmoid transform (range [0, 1])
        transformed_actions[:, 1] = torch.sigmoid(raw_actions[:, 1])

        # Action 3 (Breaking): sigmoid transform (range [0, 1])
        transformed_actions[:, 2] = torch.sigmoid(raw_actions[:, 2])

        # Compute log probability with Jacobian correction
        log_prob = base_dist.log_prob(raw_actions).sum(dim=-1)

        # Add Jacobian correction for each transform
        # tanh correction: log(1 - tanh^2(x))
        tanh_correction = torch.log(1 - transformed_actions[:, 0].pow(2) + 1e-6)

        # sigmoid correction: -log(sigmoid(x)) - log(1 - sigmoid(x))
        # Actually: sigmoid(x) = 1/(1+exp(-x)), derivative = sigmoid(x)*(1-sigmoid(x))
        # So correction = -log(sigmoid(x)) - log(1-sigmoid(x))
        sigmoid_correction1 = -torch.log(transformed_actions[:, 1] + 1e-6) - torch.log(
            1 - transformed_actions[:, 1] + 1e-6
        )
        sigmoid_correction2 = -torch.log(transformed_actions[:, 2] + 1e-6) - torch.log(
            1 - transformed_actions[:, 2] + 1e-6
        )
        # Subtract Jacobian corrections (log|det(J)|)
        log_prob = log_prob - (
            tanh_correction + sigmoid_correction1 + sigmoid_correction2
        )

        return log_prob


class ValueNet(nn.Module):
    """
    Predicts next action's value
    """

    def __init__(self):
        super().__init__()
        self.conv_model = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.linear_model = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state: torch.Tensor):
        """
        Get model prediction
        """
        batch_size, frames, channels, height, width = state.size()
        state = state.reshape(batch_size, frames * channels, height, width)
        features = self.conv_model(state)
        out = self.linear_model(features)
        return out
