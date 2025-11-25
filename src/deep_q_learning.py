"""
Deep Q-Learning (DQN) algorithm
"""

import torch
import random

from torch import nn
from collections import namedtuple, deque

from src.torch_utils import init_weights

Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "done")
)


class ReplayMemory(object):
    """
    https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    """

    def __init__(self, capacity: int, batch_size: int):
        if batch_size > capacity:
            raise ValueError("Batch size > capacity")
        self.batch_size = batch_size
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """
        Save a transition
        """
        self.memory.append(Transition(*args))

    def sample(self):
        """
        Get a batch sample
        """
        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    """
    Deep Q-Learning module
    """

    def __init__(self, n_observations: int, n_actions: int):
        super(DQN, self).__init__()
        self._initialize_weights()
        # Simpler architecture is suggested for classic control tasks]
        self.model = nn.Sequential(
            nn.Linear(n_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def _initialize_weights(self):
        for module in self.modules():
            init_weights(module)

    def forward(self, x):
        return self.model(x)


def soft_update(target_net: DQN, policy_net: DQN, tau: float = 0.005):
    """
    Update target_net a little bit every step
    """
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()

    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[
            key
        ] * tau + target_net_state_dict[key] * (1 - tau)
    target_net.load_state_dict(target_net_state_dict)


def optimize_model(
    optimizer,
    criterion,
    device,
    gamma: float,
    train_sample: list[torch.Tensor],
    policy_net: DQN,
    target_net: DQN,
    verbose: bool = False,
) -> None:
    optimizer.zero_grad()

    done_mask = torch.stack([x.done for x in train_sample]).squeeze()
    next_state_tensor = torch.stack([x.next_state for x in train_sample])
    state_tensor = torch.stack([x.state for x in train_sample])
    action_batch = torch.stack([x.action for x in train_sample])
    reward_batch = torch.stack([x.reward for x in train_sample]).squeeze()

    # Current Q values for taken actions
    current_q_values = policy_net(state_tensor).gather(1, action_batch.unsqueeze(1))

    # Next Q values from target network
    with torch.no_grad():
        next_q_values = target_net(next_state_tensor).max(1)[0]
        expected_q_values = reward_batch + gamma * next_q_values * (1 - done_mask)

    loss = criterion(current_q_values.squeeze(), expected_q_values)

    if verbose:
        print(f"Loss function value: {loss.item()}")

    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
    optimizer.step()


def select_action(
    n_actions: int,
    epsilon: float,
    state: torch.Tensor,
    policy_net: torch.nn.Module,
    device: torch.device,
    exploratory_period: bool = False,
) -> torch.Tensor:
    """
    Select action based on current state
    """
    if exploratory_period or torch.rand(1) < epsilon:
        # random action
        return torch.randint(0, n_actions, (1,)).max().to(device)

    with torch.no_grad():
        # best action
        return policy_net(state).argmax().to(device)
