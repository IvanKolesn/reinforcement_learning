"""
Dyna-Q+ Reinforcement Learning Agent.

Combines direct reinforcement learning with model-based planning.
Dyna-Q+ adds exploration bonuses to encourage revisiting infrequently
visited state-action pairs, which is useful in stochastic environments.
"""

import numpy as np
import random
from collections import deque


class DynaQModel:
    """
    Dyna-Q+ Reinforcement Learning Agent.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        n_planning: int = 5,
        kappa: float = 0.001,
        initial_q: float = 0.0,
    ):
        """
        Initialize Dyna-Q+ agent.

        Args:
            state_size: Number of possible states in the environment
            action_size: Number of possible actions
            alpha: Learning rate (0 < alpha ≤ 1)
            gamma: Discount factor (0 ≤ gamma < 1)
            epsilon: Initial exploration probability for ε-greedy
            epsilon_decay: Multiplicative decay factor per episode
            epsilon_min: Minimum exploration probability
            n_planning: Number of planning steps per real step
            kappa: Exploration bonus scaling factor
            initial_q: Initial Q-value for all state-action pairs
        """
        self.state_size = state_size
        self.action_size = action_size

        # Learning parameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.n_planning = n_planning
        self.kappa = kappa

        # Initialize Q-table with optimistic values
        self.q_table = np.full((state_size, action_size), initial_q, dtype=np.float32)

        # Environment model: (state, action) -> (reward, next_state, done)
        self.model = {}

        # Track time since last visit for exploration bonus
        self.time_since_last_visit = np.zeros(
            (state_size, action_size), dtype=np.float32
        )

        # Set of visited state-action pairs for planning
        self.visited_sa = set()

        # Episode statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.successes = []

        # Global time counter
        self.global_time = 0

    def select_action(self, state: int, training: bool = True) -> int:
        """
        Select action using ε-greedy policy.

        Args:
            state: Current state index
            training: If True, use exploration; if False, use greedy policy

        Returns:
            Selected action index
        """
        if training and random.random() < self.epsilon:
            # Exploration: random action
            return random.randint(0, self.action_size - 1)

        # Exploitation: best action (break ties randomly)
        q_values = self.q_table[state]
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        return np.random.choice(best_actions)

    def _get_exploration_bonus(self, state: int, action: int) -> float:
        """
        Calculate exploration bonus based on time since last visit.

        Args:
            state: State index
            action: Action index

        Returns:
            Exploration bonus
        """
        return self.kappa * np.sqrt(self.time_since_last_visit[state, action])

    def _update_q_value(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
        bonus: float = 0.0,
    ) -> float:
        """
        Update Q-value using Q-learning with optional exploration bonus.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
            bonus: Exploration bonus to add to reward

        Returns:
            Temporal difference error
        """
        # Add exploration bonus to reward
        reward_with_bonus = reward + bonus

        # Current Q-value
        current_q = self.q_table[state, action]

        # Maximum Q-value for next state (0 if terminal)
        if done:
            next_max_q = 0.0
        else:
            next_max_q = np.max(self.q_table[next_state])

        # Q-learning update
        td_target = reward_with_bonus + self.gamma * next_max_q
        td_error = td_target - current_q
        self.q_table[state, action] = current_q + self.alpha * td_error

        return td_error

    def _update_model(
        self, state: int, action: int, reward: float, next_state: int, done: bool
    ) -> None:
        """
        Update environment model with new experience.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        """
        # Store transition in model
        self.model[(state, action)] = (reward, next_state, done)

        # Mark state-action pair as visited
        self.visited_sa.add((state, action))

        # Reset time since last visit for this pair
        self.time_since_last_visit[state, action] = 0.0

    def _planning_step(self) -> None:
        """Perform planning using learned model."""
        if not self.visited_sa:
            return

        visited_list = list(self.visited_sa)

        for _ in range(self.n_planning):
            # Randomly sample a previously experienced state-action pair
            state, action = random.choice(visited_list)

            # Skip if not in model (safety check)
            if (state, action) not in self.model:
                continue

            # Get simulated experience from model
            reward, next_state, done = self.model[(state, action)]

            # Calculate exploration bonus for simulated experience
            bonus = self._get_exploration_bonus(state, action)

            # Update Q-value using simulated experience
            self._update_q_value(state, action, reward, next_state, done, bonus)

    def train_episode(self, env, max_steps: int = 1000) -> tuple:
        """
        Train for a single episode.

        Args:
            env: OpenAI Gym environment
            max_steps: Maximum steps per episode

        Returns:
            Tuple of (total_reward, episode_length, success)
        """
        state = env.reset()
        # not slippery
        state = state[0]
        total_reward = 0.0
        done = False
        step = 0

        while not done and step < max_steps:
            # Increment global time
            self.global_time += 1

            # Increment time since last visit for all state-action pairs
            self.time_since_last_visit += 1.0

            # Select action
            action = self.select_action(state, training=True)

            # Take action in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            # slippery = False
            done = terminated or truncated

            # Calculate exploration bonus for real experience
            bonus = self._get_exploration_bonus(state, action)

            # Update Q-value with real experience
            self._update_q_value(state, action, reward, next_state, done, bonus)

            # Update model with real experience
            self._update_model(state, action, reward, next_state, done)

            # Perform planning
            self._planning_step()

            # Move to next state
            state = next_state
            total_reward += reward
            step += 1

        # Decay exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Record episode statistics
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(step)
        self.successes.append(reward > 0)  # Success if positive reward at end

        return total_reward, step, (reward > 0)

    def train(
        self,
        env,
        n_episodes: int = 1000,
        max_steps: int = 1000,
        log_interval: int = 100,
    ) -> dict:
        """
        Train agent for multiple episodes.

        Args:
            env: OpenAI Gym environment
            n_episodes: Number of training episodes
            max_steps: Maximum steps per episode
            log_interval: Interval for logging progress

        Returns:
            Dictionary with training statistics
        """
        recent_rewards = deque(maxlen=100)
        recent_successes = deque(maxlen=100)

        for episode in range(n_episodes):
            total_reward, steps, success = self.train_episode(env, max_steps)
            recent_rewards.append(total_reward)
            recent_successes.append(success)

            # Log progress
            if (episode + 1) % log_interval == 0:
                avg_reward = np.mean(recent_rewards)
                success_rate = np.mean(recent_successes)
                avg_steps = np.mean(self.episode_lengths[-log_interval:])

                print(
                    f"Episode {episode + 1:4d} | "
                    f"Avg Reward: {avg_reward:7.2f} | "
                    f"Success Rate: {success_rate:6.1%} | "
                    f"Avg Steps: {avg_steps:6.1f} | "
                    f"Epsilon: {self.epsilon:.3f} | "
                    f"Visited SA: {len(self.visited_sa):5d}"
                )

        # Compile final statistics
        stats = {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "successes": self.successes,
            "final_epsilon": self.epsilon,
            "visited_sa_pairs": len(self.visited_sa),
            "model_size": len(self.model),
        }

        return stats

    def evaluate(self, env, n_episodes: int = 100, max_steps: int = 1000) -> dict:
        """
        Evaluate the learned policy without exploration.

        Args:
            env: OpenAI Gym environment
            n_episodes: Number of evaluation episodes
            max_steps: Maximum steps per episode

        Returns:
            Dictionary with evaluation statistics
        """
        eval_rewards = []
        eval_lengths = []
        eval_successes = []

        for _ in range(n_episodes):
            state = env.reset()
            total_reward = 0.0
            done = False
            steps = 0

            while not done and steps < max_steps:
                # Use greedy policy (no exploration)
                action = np.argmax(self.q_table[state])
                next_state, reward, terminated, truncated, info = env.step(action)
                next_state = next_state[0]
                done = terminated or truncated

                total_reward += reward
                steps += 1

            eval_rewards.append(total_reward)
            eval_lengths.append(steps)
            eval_successes.append(reward > 0)

        stats = {
            "mean_reward": np.mean(eval_rewards),
            "std_reward": np.std(eval_rewards),
            "mean_length": np.mean(eval_lengths),
            "success_rate": np.mean(eval_successes),
        }

        return stats

    def get_policy(self) -> np.ndarray:
        """
        Extract greedy policy from Q-table.

        Returns:
            Array where policy[state] = best_action
        """
        return np.argmax(self.q_table, axis=1)
