"""Helpers to build train/test environments from a ticker name."""
from __future__ import annotations

from typing import Tuple

import pandas as pd

from config import settings
from data.fetch_data import download_ticker, load_ticker
from environments.stock_trading_env import StockTradingEnv
from features.technical_indicators import add_indicators
from utils.logger import get_logger

logger = get_logger(__name__)


def load_or_download(ticker: str) -> pd.DataFrame:
    """Load a ticker CSV; download if not present."""
    try:
        return load_ticker(ticker)
    except FileNotFoundError:
        logger.info("Data not found locally for %s - downloading.", ticker)
        return download_ticker(
            ticker, settings.data_start_date, settings.data_end_date
        )


def split_train_test(
    df: pd.DataFrame, train_pct: float = 0.85
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Time-series split (no shuffle)."""
    n = len(df)
    cut = int(n * train_pct)
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


def make_envs(
    ticker: str,
    continuous: bool = False,
    train_pct: float = 0.85,
) -> Tuple[StockTradingEnv, StockTradingEnv]:
    """Return (train_env, test_env) for a ticker."""
    raw = load_or_download(ticker)
    feat = add_indicators(raw)
    train_df, test_df = split_train_test(feat, train_pct)

    common = dict(
        initial_balance=settings.initial_capital,
        transaction_cost_pct=settings.transaction_cost_pct,
        continuous_actions=continuous,
        reward_strategy="log",
    )
    train_env = StockTradingEnv(train_df, **common)
    test_env = StockTradingEnv(test_df, **common)
    return train_env, test_env
