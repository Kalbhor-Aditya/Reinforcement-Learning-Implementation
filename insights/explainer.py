"""Explain agent decisions in natural language."""
from __future__ import annotations

from typing import Dict

from insights.llm_client import get_llm_client


EXPLAIN_SYSTEM = (
    "You are an RL trading assistant. The user will give you the current "
    "market state (technical indicators) and the action an RL agent took. "
    "Explain in 3-5 sentences, in simple language a beginner can understand, "
    "WHY the agent might have made this decision, citing 2-3 specific "
    "indicator values. Avoid jargon."
)

ACTION_NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}


def explain_decision(state: Dict[str, float], action, algo: str) -> str:
    """Return a plain-English explanation."""
    try:
        action_int = int(action)
        action_str = ACTION_NAMES.get(action_int, f"action={action_int}")
    except (TypeError, ValueError):
        action_str = f"position_target={action}"

    state_str = "\n".join(f"  {k}: {v:.4f}" for k, v in state.items())
    user = (
        f"Algorithm: {algo}\n"
        f"Action taken: {action_str}\n"
        f"Market state:\n{state_str}\n\n"
        "Explain why the agent likely chose this action."
    )
    return get_llm_client().ask(EXPLAIN_SYSTEM, user, temperature=0.4)
