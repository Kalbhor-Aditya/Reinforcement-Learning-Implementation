"""Tests for evaluation metrics."""
import numpy as np

from evaluation.metrics import compute_metrics


def test_metrics_basic():
    history = [100_000, 101_000, 102_000, 100_000, 105_000]
    m = compute_metrics(history, initial_capital=100_000)
    assert m["final_value"] == 105_000
    assert m["total_return_pct"] == 5.0
    assert "sharpe_ratio" in m
    assert "max_drawdown_pct" in m


def test_metrics_short_history():
    m = compute_metrics([100_000], initial_capital=100_000)
    assert m["final_value"] == 100_000
    assert m["sharpe_ratio"] == 0.0


def test_metrics_drawdown_negative():
    history = [100, 120, 80, 90]
    m = compute_metrics(history, initial_capital=100)
    assert m["max_drawdown_pct"] < 0
