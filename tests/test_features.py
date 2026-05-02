"""Tests for technical indicators."""
from features.technical_indicators import FEATURE_COLUMNS, add_indicators


def test_add_indicators_columns(ohlcv_df):
    out = add_indicators(ohlcv_df)
    for col in FEATURE_COLUMNS:
        assert col in out.columns


def test_add_indicators_no_nan(ohlcv_df):
    out = add_indicators(ohlcv_df)
    assert out[FEATURE_COLUMNS].isna().sum().sum() == 0


def test_add_indicators_length(ohlcv_df):
    out = add_indicators(ohlcv_df)
    assert len(out) > 0
    assert len(out) < len(ohlcv_df)  # warm-up rows dropped
