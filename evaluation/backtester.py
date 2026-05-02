"""Run a trained agent through the test environment and collect results."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from agents.base_agent import BaseAgent
from environments.stock_trading_env import StockTradingEnv
from evaluation.metrics import compute_metrics


def backtest(
    agent: BaseAgent,
    env: StockTradingEnv,
    deterministic: bool = True,
) -> Tuple[Dict[str, float], List[float], List[int]]:
    """Run agent through env; return (metrics, portfolio_history, actions)."""
    obs, _ = env.reset()
    done = False
    actions: List[int] = []
    while not done:
        action = agent.predict(obs, deterministic=deterministic)
        actions.append(int(np.asarray(action).flatten()[0]))
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    metrics = compute_metrics(env._portfolio_history, env.initial_balance)
    return metrics, list(env._portfolio_history), actions


def buy_and_hold(env: StockTradingEnv) -> Tuple[Dict[str, float], List[float]]:
    """Baseline: buy max shares at start, sell at end."""
    obs, _ = env.reset()
    # Force a buy on step 1, then hold
    if env.continuous_actions:
        action = np.array([1.0], dtype=np.float32)
    else:
        action = 1  # BUY

    obs, _, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    hold = 0 if not env.continuous_actions else np.array([1.0], dtype=np.float32)
    while not done:
        obs, _, terminated, truncated, _ = env.step(hold)
        done = terminated or truncated
    metrics = compute_metrics(env._portfolio_history, env.initial_balance)
    return metrics, list(env._portfolio_history)
