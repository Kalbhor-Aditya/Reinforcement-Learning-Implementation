"""Training page - launch RL training and view results."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from agents import AGENT_REGISTRY
from agents.registry import is_continuous
from config import settings
from dashboard.theme import apply_theme, page_header, status_badge
from training.trainer import train_agent

st.set_page_config(page_title="Train Agent", page_icon="🏋️", layout="wide")
apply_theme()

page_header(
    title="Train an RL Agent",
    subtitle="Pick an algorithm and a stock — the trainer will collect interactions, "
             "optimize the policy and log everything to MLflow.",
    icon="🏋️",
)

ALGO_INFO = {
    "DQN": ("Value-based · Discrete actions",
            "Classic Q-learning with neural networks. Great first baseline."),
    "DoubleDQN": ("Value-based · Discrete actions",
                  "Reduces overestimation bias from vanilla DQN."),
    "PPO": ("Policy gradient · On-policy",
            "Industry default. Stable, robust, easy to tune."),
    "A2C": ("Actor-Critic · On-policy",
            "Simpler and faster than PPO; needs more samples."),
    "SAC": ("Actor-Critic · Continuous only",
            "Adds an entropy bonus for natural exploration."),
    "TD3": ("Actor-Critic · Continuous only",
            "Twin critics + delayed updates for stability."),
    "DDPG": ("Actor-Critic · Continuous only",
            "First continuous-action algorithm; less stable."),
}

c1, c2, c3 = st.columns([1.2, 1.2, 1])
with c1:
    algo = st.selectbox("Algorithm", options=list(AGENT_REGISTRY))
with c2:
    ticker = st.selectbox("Ticker", options=settings.default_tickers)
with c3:
    timesteps = st.number_input("Total timesteps", 1_000, 500_000,
                                value=20_000, step=5_000)

sub, desc = ALGO_INFO.get(algo, ("", ""))
action_kind = "Continuous" if is_continuous(algo) else "Discrete"
st.markdown(
    f"""
    <div class="hero-card" style="margin-top: 1rem;">
        <div style="font-size: 1.4rem; font-weight: 700; color: #e6edf3;">
            {algo} <span style="font-size: 0.85rem; color: #8b949e; font-weight: 500;">— {sub}</span>
        </div>
        <div style="margin-top: 0.5rem; color: #8b949e;">{desc}</div>
        <div style="margin-top: 0.8rem;">
            {status_badge(action_kind + ' actions', 'blue')}
            {status_badge(f'{int(timesteps):,} steps', 'amber')}
            {status_badge(ticker, 'green')}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("💡 Training tips"):
    st.markdown(
        """
        - **Start small.** 10–20k timesteps gives a quick sanity check (~2 min).
        - **PPO** usually trains best out of the box. Try it first.
        - **SAC/TD3/DDPG** require continuous actions — the env auto-switches.
        - All runs are logged to **MLflow** at `./mlruns` (open with `mlflow ui`).
        - Trained models are saved to `models/{ALGO}_{TICKER}.zip`.
        """
    )

if st.button("🚀 Start Training", type="primary"):
    with st.status(f"Training {algo} on {ticker} for {int(timesteps):,} steps...",
                   expanded=True) as status:
        st.write("⏳ Loading & feature-engineering market data...")
        try:
            path = train_agent(algo, ticker, total_timesteps=int(timesteps))
            status.update(label=f"✅ Training complete: {algo} on {ticker}",
                          state="complete")
            st.success(f"Model saved to `{path}`")
            st.balloons()
        except Exception as exc:  # noqa: BLE001
            status.update(label="❌ Training failed", state="error")
            st.exception(exc)
