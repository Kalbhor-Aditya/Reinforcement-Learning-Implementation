"""Convert backtest metrics into a human-readable narrative."""
from __future__ import annotations

from typing import Dict

from insights.llm_client import get_llm_client

SUMMARY_SYSTEM = (
    "You are a portfolio analyst. Given the performance metrics of a trading "
    "strategy, write a clear, beginner-friendly 4-6 sentence summary. "
    "Highlight: total return, risk-adjusted performance (Sharpe), max "
    "drawdown, and one concrete suggestion for improvement."
)


def summarize_performance(algo: str, ticker: str, metrics: Dict[str, float]) -> str:
    metric_str = "\n".join(f"  {k}: {v:.4f}" for k, v in metrics.items())
    user = (
        f"Algorithm: {algo}\nTicker: {ticker}\nMetrics:\n{metric_str}"
    )
    return get_llm_client().ask(SUMMARY_SYSTEM, user, temperature=0.4)
