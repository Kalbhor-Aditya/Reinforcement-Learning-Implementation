"""Technical-indicator feature engineering using the `ta` library.

The output of `add_indicators(df)` is a DataFrame containing the original
OHLCV columns plus the engineered indicator columns listed in
`FEATURE_COLUMNS`. NaN rows from indicator warm-up are dropped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, ADXIndicator, EMAIndicator, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator

# Order matters - this is the order features are exposed to the agent.
FEATURE_COLUMNS = [
    "Close",
    "Volume",
    "sma_20",
    "sma_50",
    "ema_12",
    "ema_26",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_diff",
    "bb_high",
    "bb_low",
    "bb_mid",
    "atr_14",
    "adx_14",
    "stoch_k",
    "stoch_d",
    "obv",
    "return_1d",
    "return_5d",
]


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to an OHLCV DataFrame.

    Args:
        df: DataFrame with Open, High, Low, Close, Volume columns.

    Returns:
        DataFrame with FEATURE_COLUMNS appended; NaN warm-up rows dropped.
    """
    out = df.copy()
    close = out["Close"]
    high = out["High"]
    low = out["Low"]
    volume = out["Volume"]

    # Trend
    out["sma_20"] = SMAIndicator(close, window=20).sma_indicator()
    out["sma_50"] = SMAIndicator(close, window=50).sma_indicator()
    out["ema_12"] = EMAIndicator(close, window=12).ema_indicator()
    out["ema_26"] = EMAIndicator(close, window=26).ema_indicator()

    # Momentum
    out["rsi_14"] = RSIIndicator(close, window=14).rsi()
    macd = MACD(close)
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_diff"] = macd.macd_diff()

    # Volatility
    bb = BollingerBands(close, window=20, window_dev=2)
    out["bb_high"] = bb.bollinger_hband()
    out["bb_low"] = bb.bollinger_lband()
    out["bb_mid"] = bb.bollinger_mavg()
    out["atr_14"] = AverageTrueRange(high, low, close, window=14).average_true_range()

    # Trend strength
    out["adx_14"] = ADXIndicator(high, low, close, window=14).adx()

    # Stochastic
    stoch = StochasticOscillator(high, low, close, window=14, smooth_window=3)
    out["stoch_k"] = stoch.stoch()
    out["stoch_d"] = stoch.stoch_signal()

    # Volume
    out["obv"] = OnBalanceVolumeIndicator(close, volume).on_balance_volume()

    # Returns
    out["return_1d"] = close.pct_change(1)
    out["return_5d"] = close.pct_change(5)

    out = out.replace([np.inf, -np.inf], np.nan).dropna().reset_index()
    return out
