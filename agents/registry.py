"""Agent registry for lookup-by-name."""
from __future__ import annotations

from typing import Dict, Type

import gymnasium as gym

from agents.base_agent import BaseAgent
from agents.implementations import (
    A2CAgent,
    DDPGAgent,
    DQNAgent,
    DoubleDQNAgent,
    PPOAgent,
    SACAgent,
    TD3Agent,
)

AGENT_REGISTRY: Dict[str, Type[BaseAgent]] = {
    "DQN": DQNAgent,
    "DoubleDQN": DoubleDQNAgent,
    "PPO": PPOAgent,
    "A2C": A2CAgent,
    "SAC": SACAgent,
    "TD3": TD3Agent,
    "DDPG": DDPGAgent,
}


def build_agent(name: str, env: gym.Env, **kwargs) -> BaseAgent:
    """Construct an agent by name."""
    if name not in AGENT_REGISTRY:
        raise ValueError(
            f"Unknown agent '{name}'. Available: {list(AGENT_REGISTRY)}"
        )
    cls = AGENT_REGISTRY[name]
    return cls(env=env, **kwargs)


def is_continuous(name: str) -> bool:
    return AGENT_REGISTRY[name].requires_continuous_actions
