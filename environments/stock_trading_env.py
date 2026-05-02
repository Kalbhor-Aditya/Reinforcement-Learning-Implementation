"""Custom Gymnasium environment for single-stock trading.

Observation space (Box, float32):
    Concatenation of normalized technical indicators (FEATURE_COLUMNS)
    + 3 portfolio features: [balance_pct, shares_held_norm, position_value_pct]

Action space:
    - Discrete (default): 0 = HOLD, 1 = BUY, 2 = SELL
    - Continuous (for SAC/TD3/DDPG): Box([-1], [1]) representing target
      position fraction (-1 = all cash, +1 = all in stock).

Reward:
    Configurable via `reward_strategy` (see environments/rewards.py).
"""
from __future__ import annotations

from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from environments.rewards import get_reward_function
from features.technical_indicators import FEATURE_COLUMNS


HOLD, BUY, SELL = 0, 1, 2


class StockTradingEnv(gym.Env):
    """A simple single-stock trading environment."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 100_000.0,
        transaction_cost_pct: float = 0.001,
        window_size: int = 1,
        continuous_actions: bool = False,
        reward_strategy: str = "log",
        max_shares_per_trade: Optional[int] = None,
    ) -> None:
        super().__init__()

        # Ensure required columns
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"DataFrame missing feature columns: {missing}. "
                "Pass through features.add_indicators() first."
            )

        self.df = df.reset_index(drop=True)
        self.initial_balance = float(initial_balance)
        self.transaction_cost_pct = float(transaction_cost_pct)
        self.window_size = int(window_size)
        self.continuous_actions = bool(continuous_actions)
        self.max_shares_per_trade = max_shares_per_trade
        self._reward_fn = get_reward_function(reward_strategy)

        self.n_features = len(FEATURE_COLUMNS) + 3  # +balance, +shares, +position

        # Action space
        if self.continuous_actions:
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )
        else:
            self.action_space = spaces.Discrete(3)

        # Observation space - using -inf/inf for normalized features
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_features,),
            dtype=np.float32,
        )

        # Compute per-feature normalization (z-score) for stability
        feat_data = self.df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        self._feat_mean = feat_data.mean(axis=0)
        self._feat_std = feat_data.std(axis=0) + 1e-8

        # Episode state
        self._current_step = 0
        self._balance = self.initial_balance
        self._shares_held = 0
        self._cost_basis = 0.0
        self._total_trades = 0
        self._portfolio_history: list[float] = []

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._current_step = self.window_size
        self._balance = self.initial_balance
        self._shares_held = 0
        self._cost_basis = 0.0
        self._total_trades = 0
        self._portfolio_history = [self.initial_balance]
        return self._get_observation(), self._get_info()

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        prev_value = self._portfolio_value()

        cost = self._execute_action(action)

        self._current_step += 1
        terminated = self._current_step >= len(self.df) - 1
        truncated = False

        # Bankruptcy check
        if self._portfolio_value() <= 0:
            terminated = True

        curr_value = self._portfolio_value()
        self._portfolio_history.append(curr_value)

        info = self._get_info()
        info["transaction_cost"] = cost
        reward = float(self._reward_fn(prev_value, curr_value, info))

        return self._get_observation(), reward, terminated, truncated, info

    def render(self) -> None:
        print(
            f"Step {self._current_step} | "
            f"Price ₹{self._current_price():.2f} | "
            f"Balance ₹{self._balance:.2f} | "
            f"Shares {self._shares_held} | "
            f"Portfolio ₹{self._portfolio_value():.2f}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _current_price(self) -> float:
        return float(self.df.iloc[self._current_step]["Close"])

    def _portfolio_value(self) -> float:
        return self._balance + self._shares_held * self._current_price()

    def _get_observation(self) -> np.ndarray:
        row = self.df.iloc[self._current_step][FEATURE_COLUMNS].to_numpy(
            dtype=np.float64
        )
        norm = (row - self._feat_mean) / self._feat_std

        portfolio_value = self._portfolio_value()
        balance_pct = self._balance / max(portfolio_value, 1e-8)
        shares_norm = (self._shares_held * self._current_price()) / max(
            portfolio_value, 1e-8
        )
        pnl_pct = (portfolio_value - self.initial_balance) / self.initial_balance

        portfolio_obs = np.array([balance_pct, shares_norm, pnl_pct], dtype=np.float64)
        obs = np.concatenate([norm, portfolio_obs]).astype(np.float32)
        return obs

    def _get_info(self) -> dict:
        return {
            "step": self._current_step,
            "balance": self._balance,
            "shares_held": self._shares_held,
            "portfolio_value": self._portfolio_value(),
            "total_trades": self._total_trades,
            "price": self._current_price(),
        }

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------
    def _execute_action(self, action) -> float:
        """Execute action; return transaction cost incurred."""
        price = self._current_price()
        if price <= 0:
            return 0.0

        if self.continuous_actions:
            return self._execute_continuous(float(np.asarray(action).flatten()[0]), price)
        return self._execute_discrete(int(action), price)

    def _execute_discrete(self, action: int, price: float) -> float:
        cost = 0.0
        if action == BUY:
            max_shares = int(self._balance // (price * (1 + self.transaction_cost_pct)))
            if self.max_shares_per_trade:
                max_shares = min(max_shares, self.max_shares_per_trade)
            if max_shares > 0:
                gross = max_shares * price
                fee = gross * self.transaction_cost_pct
                self._balance -= gross + fee
                self._shares_held += max_shares
                self._cost_basis = price
                self._total_trades += 1
                cost = fee
        elif action == SELL and self._shares_held > 0:
            gross = self._shares_held * price
            fee = gross * self.transaction_cost_pct
            self._balance += gross - fee
            self._shares_held = 0
            self._total_trades += 1
            cost = fee
        return cost

    def _execute_continuous(self, target_position: float, price: float) -> float:
        """target_position in [-1, 1] -> fraction of portfolio in stock.

        Negative values are clipped to 0 (no shorting in this simple env).
        """
        target_position = float(np.clip(target_position, 0.0, 1.0))
        portfolio_value = self._portfolio_value()
        target_value = target_position * portfolio_value
        target_shares = int(target_value // price)
        delta = target_shares - self._shares_held

        cost = 0.0
        if delta > 0:  # buy
            gross = delta * price
            fee = gross * self.transaction_cost_pct
            total = gross + fee
            if total <= self._balance:
                self._balance -= total
                self._shares_held += delta
                self._total_trades += 1
                cost = fee
        elif delta < 0:  # sell
            qty = -delta
            gross = qty * price
            fee = gross * self.transaction_cost_pct
            self._balance += gross - fee
            self._shares_held -= qty
            self._total_trades += 1
            cost = fee
        return cost
