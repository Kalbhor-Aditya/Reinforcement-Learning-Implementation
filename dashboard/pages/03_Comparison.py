"""Compare all RL algorithms side-by-side."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from agents import AGENT_REGISTRY
from config import settings
from dashboard.theme import apply_theme, page_header, status_badge
from evaluation.comparison import compare_all

st.set_page_config(page_title="Comparison", page_icon="🏆", layout="wide")
apply_theme()

page_header(
    title="Algorithm Showdown",
    subtitle="Train every selected agent on the same data, then rank them by Sharpe ratio "
             "against a Buy & Hold benchmark.",
    icon="🏆",
)

c1, c2, c3 = st.columns([1.2, 1.2, 2])
with c1:
    ticker = st.selectbox("Ticker", options=settings.default_tickers)
with c2:
    timesteps = st.number_input("Timesteps / agent", 1_000, 200_000, 10_000, 5_000)
with c3:
    algos = st.multiselect(
        "Algorithms",
        options=list(AGENT_REGISTRY),
        default=["PPO", "A2C", "DQN", "SAC"],
    )

comparison_csv = settings.logs_dir / "comparisons" / f"{ticker.replace('.', '_')}_comparison.csv"
histories_json = settings.logs_dir / "comparisons" / f"{ticker.replace('.', '_')}_histories.json"

if st.button("🏁 Run Showdown", type="primary"):
    if not algos:
        st.error("Please select at least one algorithm.")
    else:
        with st.status(f"Training {len(algos)} agents on {ticker}...",
                       expanded=True) as status:
            try:
                compare_all(ticker, total_timesteps=int(timesteps), algos=algos)
                status.update(label="✅ Showdown complete!", state="complete")
            except Exception as exc:  # noqa: BLE001
                status.update(label="❌ Failed", state="error")
                st.exception(exc)

if comparison_csv.exists():
    df = pd.read_csv(comparison_csv)

    if len(df) > 0:
        winner = df.iloc[0]
        ret_color = "green" if winner['total_return_pct'] >= 0 else "red"
        st.markdown(
            f"""
            <div class="hero-card" style="border-color: #ffa726;">
                <div style="font-size: 1rem; color: #ffa726; font-weight: 600;">🥇 BEST PERFORMER (by Sharpe)</div>
                <div style="font-size: 2rem; font-weight: 800; color: #e6edf3; margin: 0.3rem 0;">
                    {winner['algo']}
                </div>
                <div style="margin-top: 0.6rem;">
                    {status_badge(f"Sharpe {winner['sharpe_ratio']:.2f}", 'blue')}
                    {status_badge(f"Return {winner['total_return_pct']:+.2f}%", ret_color)}
                    {status_badge(f"Max DD {winner['max_drawdown_pct']:.2f}%", 'red')}
                    {status_badge(f"Final ₹{winner['final_value']:,.0f}", 'green')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("📊 Full Leaderboard")
    styled = df.style.format(
        {
            "total_return_pct": "{:+.2f}%",
            "annualized_return_pct": "{:+.2f}%",
            "sharpe_ratio": "{:.3f}",
            "sortino_ratio": "{:.3f}",
            "max_drawdown_pct": "{:.2f}%",
            "volatility_pct": "{:.2f}%",
            "final_value": "₹{:,.0f}",
        }
    ).background_gradient(subset=["sharpe_ratio"], cmap="RdYlGn")
    st.dataframe(styled, use_container_width=True)

if histories_json.exists():
    with open(histories_json) as f:
        histories = json.load(f)

    st.subheader("📈 Equity Curves")

    palette = ["#4f9eff", "#9b6dff", "#3fb950", "#ffa726", "#f85149",
               "#56d4dd", "#ff79c6", "#bd93f9"]

    fig = go.Figure()
    for i, (name, hist) in enumerate(histories.items()):
        is_baseline = name == "Buy&Hold"
        fig.add_trace(
            go.Scatter(
                y=hist,
                name=name,
                mode="lines",
                line=dict(
                    color=palette[i % len(palette)],
                    width=3 if is_baseline else 2,
                    dash="dash" if is_baseline else "solid",
                ),
            )
        )
    fig.update_layout(
        title=f"Portfolio Value over Test Period — {ticker}",
        xaxis_title="Step",
        yaxis_title="Portfolio Value (₹)",
        height=520,
        template="plotly_dark",
        plot_bgcolor="rgba(28,35,51,0.5)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6edf3"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
