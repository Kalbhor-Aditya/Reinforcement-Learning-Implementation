"""Tests for the trading environment."""
import numpy as np
from gymnasium.utils.env_checker import check_env

from environments.stock_trading_env import StockTradingEnv
from features.technical_indicators import add_indicators


def test_env_passes_gym_check(ohlcv_df):
    df = add_indicators(ohlcv_df)
    env = StockTradingEnv(df, initial_balance=10_000)
    check_env(env, skip_render_check=True)


def test_env_reset_and_step(ohlcv_df):
    df = add_indicators(ohlcv_df)
    env = StockTradingEnv(df, initial_balance=10_000)
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    obs, reward, term, trunc, info = env.step(env.action_space.sample())
    assert isinstance(reward, float)
    assert isinstance(info, dict)


def test_env_full_episode(ohlcv_df):
    df = add_indicators(ohlcv_df)
    env = StockTradingEnv(df, initial_balance=10_000)
    obs, _ = env.reset()
    done = False
    steps = 0
    while not done and steps < len(df) + 5:
        obs, _, term, trunc, _ = env.step(0)  # always HOLD
        done = term or trunc
        steps += 1
    assert done


def test_continuous_action_space(ohlcv_df):
    df = add_indicators(ohlcv_df)
    env = StockTradingEnv(df, continuous_actions=True, initial_balance=10_000)
    obs, _ = env.reset()
    obs, _, _, _, _ = env.step(np.array([0.5], dtype=np.float32))
    assert obs.shape == env.observation_space.shape
