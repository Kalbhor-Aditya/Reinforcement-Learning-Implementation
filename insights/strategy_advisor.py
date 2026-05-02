"""High-level trading strategy recommendations from the LLM."""
from __future__ import annotations

from typing import Dict

from insights.llm_client import get_llm_client


STRATEGY_SYSTEM = (
    "You are an experienced trading strategist. Given current market "
    "conditions for an Indian stock, suggest 2-3 actionable trading "
    "strategies (e.g., trend-following, mean-reversion, breakout). Be "
    "concise, beginner-friendly, and include risk warnings. End with a "
    "disclaimer: 'This is educational, not financial advice.'"
)


def suggest_strategies(ticker: str, snapshot: Dict[str, float]) -> str:
    """snapshot: dict of recent indicator values."""
    state_str = "\n".join(f"  {k}: {v}" for k, v in snapshot.items())
    user = f"Stock: {ticker}\nCurrent indicators:\n{state_str}"
    return get_llm_client().ask(STRATEGY_SYSTEM, user, temperature=0.5, max_tokens=800)
