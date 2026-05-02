"""Streamlit dashboard - main entry point.

Run with:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from config import settings
from dashboard.theme import apply_theme, feature_card, page_header, status_badge
from insights.llm_client import get_llm_client


st.set_page_config(
    page_title="Quant RL — Indian Markets",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()

page_header(
    title="Quant RL Trading",
    subtitle="An end-to-end reinforcement-learning workbench for Indian equities (NSE / BSE). "
             "Train, compare and explain RL agents — all in one place.",
    icon="📈",
)

# --------------------------------------------------------------------
# Status badges
# --------------------------------------------------------------------
llm = get_llm_client()
if llm.provider == "azure":
    llm_badge = status_badge(f"Azure OpenAI · {settings.azure_openai_deployment}", "green")
elif llm.provider == "groq":
    llm_badge = status_badge(f"Groq · {settings.groq_model}", "blue")
else:
    llm_badge = status_badge("LLM Offline", "amber")

st.markdown(
    f"""
    <div style="margin-bottom: 1.5rem;">
        {status_badge('NSE / BSE', 'blue')}
        {status_badge('Stable-Baselines3', 'blue')}
        {status_badge('MLflow tracked', 'blue')}
        {llm_badge}
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------
# KPI metrics
# --------------------------------------------------------------------
st.subheader("⚙️ Configuration Snapshot")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Initial Capital", f"₹{settings.initial_capital:,.0f}")
c2.metric("Transaction Cost", f"{settings.transaction_cost_pct * 100:.2f}%")
c3.metric("Default Tickers", len(settings.default_tickers))
c4.metric("Train Timesteps", f"{settings.total_timesteps:,}")

st.caption("Tickers: " + " • ".join(settings.default_tickers))

# --------------------------------------------------------------------
# Feature cards
# --------------------------------------------------------------------
st.subheader("🚀 What you can do here")

f1, f2, f3, f4 = st.columns(4)
f1.markdown(
    feature_card("📊", "Explore Markets",
                 "Interactive candlestick charts with RSI, MACD and Bollinger Bands."),
    unsafe_allow_html=True,
)
f2.markdown(
    feature_card("🏋️", "Train Agents",
                 "DQN, PPO, A2C, SAC, TD3, DDPG — all from a single UI."),
    unsafe_allow_html=True,
)
f3.markdown(
    feature_card("🏆", "Benchmark",
                 "Race every algorithm head-to-head against Buy & Hold."),
    unsafe_allow_html=True,
)
f4.markdown(
    feature_card("🤖", "AI Insights",
                 "LLM-powered sentiment, decision explanations and strategies."),
    unsafe_allow_html=True,
)

st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

# --------------------------------------------------------------------
# Quick start + Resources
# --------------------------------------------------------------------
left, right = st.columns([1.2, 1])

with left:
    st.subheader("⚡ Quick Start")
    st.code(
        """# 1. Fetch historical data
python -m data.fetch_data --tickers RELIANCE.NS,TCS.NS

# 2. Train one agent
python -m training.trainer --algo PPO --ticker RELIANCE.NS --timesteps 50000

# 3. Compare every algorithm
python -m evaluation.comparison --ticker RELIANCE.NS --timesteps 20000

# 4. Open MLflow tracking UI
mlflow ui --backend-store-uri ./mlruns
""",
        language="powershell",
    )

with right:
    st.subheader("📚 Resources")
    st.markdown(
        """
        - **[Architecture](../.github/ARCHITECTURE.md)** — system design overview
        - **[Algorithms](../.github/ALGORITHMS.md)** — every RL algorithm explained
        - **[Setup](../.github/SETUP.md)** — installation guide
        - **[Learning Guide](../.github/LEARNING_GUIDE.md)** — 7-week curriculum
        - **[Contributing](../.github/CONTRIBUTING.md)** — extension patterns
        """
    )

st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
st.caption(
    "⚠️ Educational project only. Do **not** use this code for live trading with real money. "
    "Past performance on historical data does not guarantee future results."
)
