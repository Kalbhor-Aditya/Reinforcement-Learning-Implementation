"""Market sentiment analysis via LLM."""
from __future__ import annotations

from typing import List

from insights.llm_client import get_llm_client


SENTIMENT_SYSTEM = (
    "You are a financial sentiment analyst. Given a list of news headlines "
    "about a stock or market, classify the OVERALL sentiment as one of: "
    "BULLISH, BEARISH, or NEUTRAL. Then give a 2-3 sentence explanation in "
    "simple language. Format:\n"
    "Sentiment: <label>\nExplanation: <text>"
)


def analyze_headlines(ticker: str, headlines: List[str]) -> str:
    """Return sentiment summary string."""
    if not headlines:
        return "Sentiment: NEUTRAL\nExplanation: No headlines provided."
    user = f"Ticker: {ticker}\nHeadlines:\n- " + "\n- ".join(headlines[:20])
    return get_llm_client().ask(SENTIMENT_SYSTEM, user, temperature=0.2)
