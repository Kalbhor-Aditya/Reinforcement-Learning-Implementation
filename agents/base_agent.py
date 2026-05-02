"""Common base class for all RL agents.

Wraps Stable-Baselines3 algorithms with a uniform interface so the rest
of the codebase (training, evaluation, dashboard) can treat all algorithms
the same way.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Type

import gymnasium as gym
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm

from utils.logger import get_logger

logger = get_logger(__name__)


class BaseAgent(ABC):
    """Abstract base class for RL agents."""

    name: str = "BaseAgent"
    requires_continuous_actions: bool = False
    sb3_class: Type[BaseAlgorithm]

    def __init__(self, env: gym.Env, **kwargs: Any) -> None:
        self.env = env
        self.kwargs = self._default_hyperparams() | kwargs
        self.model: Optional[BaseAlgorithm] = None

    @abstractmethod
    def _default_hyperparams(self) -> dict:
        """Default hyperparameters for the algorithm."""

    def build(self) -> BaseAlgorithm:
        """Instantiate the underlying SB3 model."""
        logger.info("Building %s with hyperparams: %s", self.name, self.kwargs)
        self.model = self.sb3_class(env=self.env, **self.kwargs)
        return self.model

    def train(self, total_timesteps: int, callback: Any = None) -> None:
        """Train the agent for `total_timesteps` interactions."""
        if self.model is None:
            self.build()
        logger.info("Training %s for %d timesteps", self.name, total_timesteps)
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True,
        )

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        """Predict action(s) from observation."""
        if self.model is None:
            raise RuntimeError("Agent not trained or loaded yet.")
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Nothing to save - model is None.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        logger.info("Saved %s model to %s", self.name, path)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        logger.info("Loading %s model from %s", self.name, path)
        self.model = self.sb3_class.load(str(path), env=self.env)
