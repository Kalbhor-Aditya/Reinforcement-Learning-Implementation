"""Data Explorer - view stock prices and technical indicators."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from config import settings
from dashboard.theme import apply_theme, page_header, status_badge
from data.fetch_data import download_ticker, load_ticker
from features.technical_indicators import add_indicators


st.set_page_config(page_title="Data Explorer", page_icon="📊", layout="wide")
apply_theme()

page_header(
    title="Data Explorer",
    subtitle="Visualize Indian stock prices alongside the technical indicators that feed the RL agents.",
    icon="📊",
)

# Controls
ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 1.2, 1.2, 1])
with ctrl1:
    ticker = st.selectbox("Ticker", options=settings.default_tickers)
with ctrl2:
    start = st.date_input("Start date", value=None)
with ctrl3:
    end = st.date_input("End date", value=None)
with ctrl4:
    st.write("")
    st.write("")
    go_btn = st.button("Load 📈", use_container_width=True)

if go_btn:
    with st.spinner(f"Loading {ticker}..."):
        try:
            df = load_ticker(ticker)
        except FileNotFoundError:
            df = download_ticker(
                ticker,
                str(start) if start else settings.data_start_date,
                str(end) if end else settings.data_end_date,
            )

        if df.empty:
            st.error("No data returned.")
            st.stop()

        feat = add_indicators(df)

    # ---------- KPI cards ----------
    latest = feat.iloc[-1]
    prev = feat.iloc[-2] if len(feat) > 1 else latest
    pct_chg = ((latest["Close"] - prev["Close"]) / prev["Close"]) * 100

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Last Close", f"₹{latest['Close']:,.2f}", f"{pct_chg:+.2f}%")
    k2.metric("RSI (14)", f"{latest['rsi_14']:.1f}",
              "Overbought" if latest['rsi_14'] > 70
              else "Oversold" if latest['rsi_14'] < 30 else "Neutral")
    k3.metric("MACD", f"{latest['macd']:.3f}",
              f"Signal {latest['macd_signal']:.3f}")
    k4.metric("ATR (14)", f"{latest['atr_14']:.2f}")
    k5.metric("Rows", f"{len(feat):,}")

    rsi_badge = status_badge(
        "RSI Overbought" if latest['rsi_14'] > 70
        else "RSI Oversold" if latest['rsi_14'] < 30 else "RSI Neutral",
        "red" if latest['rsi_14'] > 70
        else "green" if latest['rsi_14'] < 30 else "blue",
    )
    trend_badge = status_badge(
        "Above 50-SMA" if latest['Close'] > latest['sma_50'] else "Below 50-SMA",
        "green" if latest['Close'] > latest['sma_50'] else "red",
    )
    st.markdown(f"<div style='margin: 1rem 0;'>{rsi_badge}{trend_badge}</div>",
                unsafe_allow_html=True)

    # ---------- Chart ----------
    st.subheader("📉 Price + Indicators")
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.225, 0.225],
        vertical_spacing=0.04,
        subplot_titles=("Price · SMA · Bollinger Bands", "RSI (14)", "MACD"),
    )

    fig.add_trace(go.Scatter(x=feat["Date"], y=feat["bb_high"], name="BB Upper",
                             line=dict(color="rgba(155,109,255,0.4)", width=1, dash="dot")),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=feat["Date"], y=feat["bb_low"], name="BB Lower",
                             line=dict(color="rgba(155,109,255,0.4)", width=1, dash="dot"),
                             fill="tonexty", fillcolor="rgba(155,109,255,0.05)"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=feat["Date"], y=feat["Close"], name="Close",
                             line=dict(color="#4f9eff", width=2)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=feat["Date"], y=feat["sma_20"], name="SMA 20",
                             line=dict(color="#3fb950", width=1.2)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=feat["Date"], y=feat["sma_50"], name="SMA 50",
                             line=dict(color="#ffa726", width=1.2)),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=feat["Date"], y=feat["rsi_14"], name="RSI",
                             line=dict(color="#9b6dff", width=1.5)),
                  row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#f85149",
                  opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#3fb950",
                  opacity=0.5, row=2, col=1)

    fig.add_trace(go.Bar(x=feat["Date"], y=feat["macd_diff"], name="Histogram",
                         marker_color=["#3fb950" if v > 0 else "#f85149"
                                       for v in feat["macd_diff"]]),
                  row=3, col=1)
    fig.add_trace(go.Scatter(x=feat["Date"], y=feat["macd"], name="MACD",
                             line=dict(color="#4f9eff", width=1.5)),
                  row=3, col=1)
    fig.add_trace(go.Scatter(x=feat["Date"], y=feat["macd_signal"], name="Signal",
                             line=dict(color="#ffa726", width=1.5)),
                  row=3, col=1)

    fig.update_layout(
        height=780,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        template="plotly_dark",
        plot_bgcolor="rgba(28,35,51,0.5)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(color="#e6edf3"),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔍 Raw data preview (last 10 rows)"):
        st.dataframe(feat.tail(10), use_container_width=True)
