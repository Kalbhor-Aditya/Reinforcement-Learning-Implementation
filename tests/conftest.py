"""Synthetic OHLCV fixture for fast tests (no network)."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def ohlcv_df():
    rng = np.random.default_rng(42)
    n = 300
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    price = 100 + np.cumsum(rng.normal(0, 1, n))
    price = np.clip(price, 10, None)
    df = pd.DataFrame(
        {
            "Open": price + rng.normal(0, 0.5, n),
            "High": price + np.abs(rng.normal(0, 1, n)),
            "Low": price - np.abs(rng.normal(0, 1, n)),
            "Close": price,
            "Volume": rng.integers(1_000, 100_000, n),
        },
        index=dates,
    )
    df.index.name = "Date"
    return df
