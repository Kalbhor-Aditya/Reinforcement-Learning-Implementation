"""Reward function strategies for the trading environment.

Each strategy takes (prev_value, curr_value, info) and returns a scalar reward.
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np


def simple_return(prev_value: float, curr_value: float, info: dict) -> float:
    """Percentage change of portfolio value (most basic reward)."""
    if prev_value <= 0:
        return 0.0
    return (curr_value - prev_value) / prev_value


def log_return(prev_value: float, curr_value: float, info: dict) -> float:
    """Log return - smoother gradient than simple return."""
    if prev_value <= 0 or curr_value <= 0:
        return 0.0
    return float(np.log(curr_value / prev_value))


def penalty_adjusted(prev_value: float, curr_value: float, info: dict) -> float:
    """Return minus a small penalty for trading (encourages selectivity)."""
    base = simple_return(prev_value, curr_value, info)
    cost = info.get("transaction_cost", 0.0)
    return base - cost / max(prev_value, 1.0)


REWARD_FUNCTIONS: Dict[str, Callable[[float, float, dict], float]] = {
    "simple": simple_return,
    "log": log_return,
    "penalty": penalty_adjusted,
}


def get_reward_function(name: str) -> Callable[[float, float, dict], float]:
    if name not in REWARD_FUNCTIONS:
        raise ValueError(
            f"Unknown reward '{name}'. Choose from {list(REWARD_FUNCTIONS)}"
        )
    return REWARD_FUNCTIONS[name]
