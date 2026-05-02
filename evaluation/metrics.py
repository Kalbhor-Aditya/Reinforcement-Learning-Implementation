"""Performance metrics for evaluating trading strategies."""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def compute_metrics(
    portfolio_history: List[float],
    initial_capital: float,
    risk_free_rate: float = 0.05,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    """Compute standard trading performance metrics.

    Args:
        portfolio_history: List of portfolio values per step (daily).
        initial_capital: Starting capital.
        risk_free_rate: Annual risk-free rate (default 5%).
        periods_per_year: 252 for daily, 12 for monthly.
    """
    arr = np.asarray(portfolio_history, dtype=np.float64)
    if len(arr) < 2:
        return {
            "total_return_pct": 0.0,
            "annualized_return_pct": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "volatility_pct": 0.0,
            "final_value": float(initial_capital),
        }

    final_value = float(arr[-1])
    total_return = (final_value - initial_capital) / initial_capital

    daily_returns = np.diff(arr) / arr[:-1]
    daily_returns = np.where(np.isfinite(daily_returns), daily_returns, 0.0)

    n_periods = len(daily_returns)
    annualized_return = (
        (1 + total_return) ** (periods_per_year / max(n_periods, 1)) - 1
    )
    volatility = daily_returns.std() * np.sqrt(periods_per_year)
    daily_rf = risk_free_rate / periods_per_year
    excess = daily_returns - daily_rf

    sharpe = (
        excess.mean() / (daily_returns.std() + 1e-8) * np.sqrt(periods_per_year)
    )
    downside = daily_returns[daily_returns < 0]
    sortino = (
        excess.mean() / (downside.std() + 1e-8) * np.sqrt(periods_per_year)
        if len(downside) > 0
        else 0.0
    )

    cum = np.maximum.accumulate(arr)
    drawdown = (arr - cum) / cum
    max_dd = float(drawdown.min())

    return {
        "total_return_pct": float(total_return * 100),
        "annualized_return_pct": float(annualized_return * 100),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown_pct": float(max_dd * 100),
        "volatility_pct": float(volatility * 100),
        "final_value": final_value,
    }
