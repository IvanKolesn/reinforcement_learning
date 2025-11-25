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
        # Simpler architecture is suggested for classic control tasks
        self.model = nn.Sequential(
            nn.Linear(n_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def _initialize_weights(self):
        for m in self.modules():
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
    gamma: float,
    train_sample: list[torch.Tensor()],
    policy_net: DQN,
    target_net: DQN,
    verbose: bool = False,
) -> None:
    """
    Optimize policy network
    """
    optimizer.zero_grad()

    done_mask = torch.tensor([float(x.done) for x in train_sample])
    next_state_tensor = torch.stack([x.next_state for x in train_sample])
    state_tensor = torch.stack([x.state for x in train_sample])
    action_batch = torch.stack([x.action for x in train_sample])
    reward_batch = torch.stack([x.reward for x in train_sample]).squeeze()

    # Actual q values
    current_q_tensor = policy_net(state_tensor)
    next_state_action_tensor = current_q_tensor.gather(1, action_batch.view(1, -1)).to(
        torch.float32
    )

    # Expected q values
    with torch.no_grad():
        exp_q_tensor = target_net(next_state_tensor)
        next_state_expected_tensor = (
            gamma * (1 - done_mask) * exp_q_tensor.gather(1, action_batch.view(1, -1))
            + reward_batch
        ).to(torch.float32)

    loss = criterion(next_state_action_tensor, next_state_expected_tensor)

    if verbose:
        print(f"Loss function value: {loss}")

    # Optimize the model
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
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
        return torch.randint(0, n_actions, (1,), device=device).max()

    with torch.no_grad():
        # best action
        return policy_net(state).argmax().to(device)
