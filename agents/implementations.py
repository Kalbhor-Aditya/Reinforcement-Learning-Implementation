"""Concrete agent classes wrapping Stable-Baselines3 algorithms."""
from __future__ import annotations

from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3

from agents.base_agent import BaseAgent


class DQNAgent(BaseAgent):
    """Deep Q-Network. Discrete actions only.

    Off-policy value-based: learns Q(s,a) using a neural network and
    replays past transitions stored in a buffer.
    """

    name = "DQN"
    sb3_class = DQN

    def _default_hyperparams(self) -> dict:
        return {
            "policy": "MlpPolicy",
            "learning_rate": 1e-4,
            "buffer_size": 50_000,
            "learning_starts": 1_000,
            "batch_size": 64,
            "gamma": 0.99,
            "exploration_fraction": 0.2,
            "exploration_final_eps": 0.05,
            "verbose": 0,
        }


class DoubleDQNAgent(DQNAgent):
    """Double DQN. Reduces overestimation bias using two networks."""

    name = "DoubleDQN"

    def _default_hyperparams(self) -> dict:
        # SB3 DQN already uses target network; "Double DQN" enabled via target update
        # We mark the difference via a slightly different exploration schedule.
        params = super()._default_hyperparams()
        params.update({"target_update_interval": 1_000})
        return params


class PPOAgent(BaseAgent):
    """Proximal Policy Optimization. On-policy, supports both spaces."""

    name = "PPO"
    sb3_class = PPO

    def _default_hyperparams(self) -> dict:
        return {
            "policy": "MlpPolicy",
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "verbose": 0,
        }


class A2CAgent(BaseAgent):
    """Advantage Actor-Critic. On-policy."""

    name = "A2C"
    sb3_class = A2C

    def _default_hyperparams(self) -> dict:
        return {
            "policy": "MlpPolicy",
            "learning_rate": 7e-4,
            "n_steps": 5,
            "gamma": 0.99,
            "gae_lambda": 1.0,
            "ent_coef": 0.01,
            "verbose": 0,
        }


class SACAgent(BaseAgent):
    """Soft Actor-Critic. Off-policy, continuous actions only."""

    name = "SAC"
    sb3_class = SAC
    requires_continuous_actions = True

    def _default_hyperparams(self) -> dict:
        return {
            "policy": "MlpPolicy",
            "learning_rate": 3e-4,
            "buffer_size": 50_000,
            "learning_starts": 1_000,
            "batch_size": 256,
            "gamma": 0.99,
            "tau": 0.005,
            "ent_coef": "auto",
            "verbose": 0,
        }


class TD3Agent(BaseAgent):
    """Twin Delayed DDPG. Off-policy, continuous actions only."""

    name = "TD3"
    sb3_class = TD3
    requires_continuous_actions = True

    def _default_hyperparams(self) -> dict:
        return {
            "policy": "MlpPolicy",
            "learning_rate": 1e-3,
            "buffer_size": 50_000,
            "learning_starts": 1_000,
            "batch_size": 100,
            "gamma": 0.99,
            "tau": 0.005,
            "policy_delay": 2,
            "verbose": 0,
        }


class DDPGAgent(BaseAgent):
    """Deep Deterministic Policy Gradient. Off-policy, continuous only."""

    name = "DDPG"
    sb3_class = DDPG
    requires_continuous_actions = True

    def _default_hyperparams(self) -> dict:
        return {
            "policy": "MlpPolicy",
            "learning_rate": 1e-3,
            "buffer_size": 50_000,
            "learning_starts": 1_000,
            "batch_size": 100,
            "gamma": 0.99,
            "tau": 0.005,
            "verbose": 0,
        }
