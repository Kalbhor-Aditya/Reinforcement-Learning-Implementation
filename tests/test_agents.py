"""Tests for agent registry & build."""
import pytest

from agents import AGENT_REGISTRY, build_agent
from agents.registry import is_continuous
from environments.stock_trading_env import StockTradingEnv
from features.technical_indicators import add_indicators


@pytest.mark.parametrize("algo", list(AGENT_REGISTRY))
def test_agent_can_be_built(algo, ohlcv_df):
    df = add_indicators(ohlcv_df)
    env = StockTradingEnv(df, continuous_actions=is_continuous(algo))
    agent = build_agent(algo, env=env)
    model = agent.build()
    assert model is not None
    assert agent.name == algo


def test_unknown_agent_raises(ohlcv_df):
    df = add_indicators(ohlcv_df)
    env = StockTradingEnv(df)
    with pytest.raises(ValueError):
        build_agent("NOTAREALALGO", env=env)
