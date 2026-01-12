"""
Model free algorithms:

Q learning algorithm
"""

import gymnasium as gym
import numpy as np

from tqdm import trange


def train_with_q_learning(
    env: gym.Env,
    alpha: float,
    gamma: float,
    epsilon: float,
    exploratory_period: int = 100,
    n_epochs: int = 1000,
    seed: int = 42,
    print_every: int = 1000,
    silent_mode: bool = False,
) -> tuple[np.array]:
    """
    Train the model
    """
    # Инциализируем Q-table

    rewards = []

    np.random.seed(seed)

    n_actions = env.action_space.n
    n_states = env.observation_space.n
    q_table = np.zeros((n_states, n_actions))

    for episode in trange(n_epochs, disable=silent_mode):

        state, _ = env.reset()
        done = False
        total_reward = 0
        total_penalties = 0
        step = 0

        while not done:
            # For the first n episodes we select completely random actions to have enough exploration
            if np.random.uniform(0, 1) < epsilon or episode < exploratory_period:
                # Take a random action (explore)
                action = env.action_space.sample()
            else:
                # Choose a q-maximizing action (exploit)
                action = np.argmax(q_table[state, :])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Update q function
            future_state = float(not done) * max(q_table[next_state])

            q_table[state, action] = q_table[state, action] + alpha * (
                reward + gamma * future_state - q_table[state, action]
            )

            total_reward += reward
            if reward == -10:
                total_penalties += 1
            step += 1
            state = next_state

        rewards.append([total_reward, total_penalties, step])

        # after 70% of episodes, we slowly start to decrease the epsilon parameter
        if episode > 0.7 * n_epochs:
            epsilon *= 0.99

        if episode % print_every == 0 and not silent_mode:
            print(
                f"{episode=} reward {total_reward},  # penalties {total_penalties}, # steps {step}"
            )

    return q_table, rewards
