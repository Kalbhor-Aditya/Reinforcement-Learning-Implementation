"""AI-powered insights via Azure OpenAI / Groq."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from config import settings
from dashboard.theme import ai_response, apply_theme, page_header, status_badge
from data.fetch_data import load_ticker
from features.technical_indicators import FEATURE_COLUMNS, add_indicators
from insights.explainer import explain_decision
from insights.llm_client import get_llm_client
from insights.portfolio_summarizer import summarize_performance
from insights.sentiment import analyze_headlines
from insights.strategy_advisor import suggest_strategies

st.set_page_config(page_title="AI Insights", page_icon="🤖", layout="wide")
apply_theme()

page_header(
    title="AI Insights",
    subtitle="Use Azure OpenAI (with Groq fallback) to interpret market sentiment, "
             "explain agent decisions and suggest strategies — all in plain English.",
    icon="🤖",
)

llm = get_llm_client()
if llm.provider == "azure":
    st.markdown(
        f"<div style='margin-bottom:1rem'>"
        f"{status_badge(f'Azure OpenAI · {settings.azure_openai_deployment}', 'green')}"
        f"{status_badge('Auto-fallback to Groq enabled', 'blue')}</div>",
        unsafe_allow_html=True,
    )
elif llm.provider == "groq":
    st.markdown(
        f"<div style='margin-bottom:1rem'>{status_badge(f'Groq · {settings.groq_model}', 'blue')}</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"<div style='margin-bottom:1rem'>{status_badge('LLM Offline — placeholder responses', 'amber')}</div>",
        unsafe_allow_html=True,
    )

tab1, tab2, tab3, tab4 = st.tabs(
    ["📰 Sentiment", "🧠 Explain Decision", "🎯 Strategy", "📈 Portfolio Summary"]
)

# ---- Sentiment ----
with tab1:
    st.subheader("Market Sentiment Analyzer")
    st.caption("Paste recent news headlines — the LLM will tag the overall sentiment.")

    c1, c2 = st.columns([1, 2])
    with c1:
        ticker = st.selectbox("Ticker", options=settings.default_tickers, key="sent_ticker")
    with c2:
        headlines = st.text_area(
            "Headlines (one per line)",
            value="Reliance Q3 profits beat estimates\n"
                  "Jio user base hits record high\n"
                  "Oil prices weigh on margins",
            height=140,
        )
    if st.button("Analyze Sentiment", key="sent_btn", type="primary"):
        with st.spinner("Calling LLM..."):
            result = analyze_headlines(
                ticker, [h for h in headlines.splitlines() if h.strip()]
            )
        ai_response(result.replace("\n", "<br>"))

# ---- Explain ----
with tab2:
    st.subheader("Decision Explainer")
    st.caption("Pick a market state + an action; the LLM will explain why an RL agent might choose it.")

    c1, c2, c3 = st.columns(3)
    with c1:
        ticker_e = st.selectbox("Ticker", options=settings.default_tickers, key="exp_ticker")
    with c2:
        algo_e = st.selectbox("Algorithm",
                              options=["PPO", "DQN", "A2C", "SAC", "TD3", "DDPG"],
                              key="exp_algo")
    with c3:
        action_e = st.selectbox("Action",
                                options=[("HOLD", 0), ("BUY", 1), ("SELL", 2)],
                                format_func=lambda x: x[0])

    if st.button("Explain", key="exp_btn", type="primary"):
        try:
            df = add_indicators(load_ticker(ticker_e))
            latest = df.iloc[-1][FEATURE_COLUMNS].to_dict()
            with st.expander("🔬 Market state used as context"):
                st.json({k: round(v, 4) for k, v in latest.items()})
            with st.spinner("Calling LLM..."):
                expl = explain_decision(latest, action_e[1], algo_e)
            ai_response(expl.replace("\n", "<br>"))
        except FileNotFoundError as e:
            st.error(f"📂 {e}")

# ---- Strategy ----
with tab3:
    st.subheader("Strategy Advisor")
    st.caption("Get 2–3 actionable strategies tailored to current technical conditions.")

    ticker_s = st.selectbox("Ticker", options=settings.default_tickers, key="strat_ticker")
    if st.button("Suggest Strategies", key="strat_btn", type="primary"):
        try:
            df = add_indicators(load_ticker(ticker_s))
            snapshot = df.iloc[-1][FEATURE_COLUMNS].round(4).to_dict()
            with st.expander("📋 Snapshot fed to the LLM"):
                st.json(snapshot)
            with st.spinner("Calling LLM..."):
                recs = suggest_strategies(ticker_s, snapshot)
            ai_response(recs.replace("\n", "<br>"))
        except FileNotFoundError as e:
            st.error(str(e))

# ---- Portfolio Summary ----
with tab4:
    st.subheader("Portfolio Performance Summary")
    st.caption("Convert raw backtest metrics into a beginner-friendly narrative.")

    c1, c2 = st.columns(2)
    with c1:
        algo_p = st.text_input("Algorithm", value="PPO", key="port_algo")
    with c2:
        ticker_p = st.selectbox("Ticker", options=settings.default_tickers, key="port_ticker")

    st.markdown("**Performance metrics:**")
    m1, m2, m3 = st.columns(3)
    with m1:
        total_ret = st.number_input("Total Return %", value=15.5)
        sharpe = st.number_input("Sharpe Ratio", value=1.2)
    with m2:
        max_dd = st.number_input("Max Drawdown %", value=-12.0)
        sortino = st.number_input("Sortino Ratio", value=1.5)
    with m3:
        vol = st.number_input("Volatility %", value=22.0)
        final_val = st.number_input("Final Value ₹", value=115_500.0)

    if st.button("Summarize", key="port_btn", type="primary"):
        metrics = {
            "total_return_pct": total_ret,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_dd,
            "sortino_ratio": sortino,
            "volatility_pct": vol,
            "final_value": final_val,
        }
        with st.spinner("Calling LLM..."):
            summary = summarize_performance(algo_p, ticker_p, metrics)
        ai_response(summary.replace("\n", "<br>"))
