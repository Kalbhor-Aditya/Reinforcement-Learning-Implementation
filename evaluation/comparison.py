"""Train all available agents on the same data and compare results.

CLI:
    python -m evaluation.comparison --ticker RELIANCE.NS --timesteps 20000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from agents import AGENT_REGISTRY, build_agent
from agents.registry import is_continuous
from config import settings
from evaluation.backtester import backtest, buy_and_hold
from training.env_builder import make_envs
from utils.logger import get_logger

logger = get_logger(__name__)


def compare_all(
    ticker: str,
    total_timesteps: int = 20_000,
    algos: List[str] | None = None,
) -> pd.DataFrame:
    """Train each algorithm and return a metrics comparison DataFrame."""
    algos = algos or list(AGENT_REGISTRY)
    rows: List[Dict] = []

    # Buy-and-hold baseline (use any env)
    _, test_env = make_envs(ticker, continuous=False)
    bh_metrics, bh_history = buy_and_hold(test_env)
    bh_metrics["algo"] = "Buy&Hold"
    rows.append(bh_metrics)

    histories: Dict[str, List[float]] = {"Buy&Hold": bh_history}

    for algo in algos:
        try:
            logger.info("=== Training %s ===", algo)
            continuous = is_continuous(algo)
            train_env, test_env = make_envs(ticker, continuous=continuous)
            agent = build_agent(algo, env=train_env, seed=settings.seed)
            agent.train(total_timesteps=total_timesteps)

            metrics, history, _actions = backtest(agent, test_env)
            metrics["algo"] = algo
            rows.append(metrics)
            histories[algo] = history

            save_path = settings.models_dir / f"{algo}_{ticker.replace('.', '_')}.zip"
            agent.save(save_path)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Training %s failed: %s", algo, exc)

    df = pd.DataFrame(rows)
    cols = [
        "algo",
        "total_return_pct",
        "annualized_return_pct",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown_pct",
        "volatility_pct",
        "final_value",
    ]
    df = df[[c for c in cols if c in df.columns]]
    df = df.sort_values("sharpe_ratio", ascending=False).reset_index(drop=True)

    out_dir = settings.logs_dir / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{ticker.replace('.', '_')}_comparison.csv", index=False)

    with open(out_dir / f"{ticker.replace('.', '_')}_histories.json", "w") as f:
        json.dump(histories, f)

    logger.info("\n%s", df.to_string(index=False))
    return df


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare all RL algorithms.")
    p.add_argument("--ticker", type=str, default="RELIANCE.NS")
    p.add_argument("--timesteps", type=int, default=20_000)
    p.add_argument(
        "--algos",
        type=str,
        default=None,
        help="Comma-separated subset (default: all registered agents)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    algos = [a.strip() for a in args.algos.split(",")] if args.algos else None
    compare_all(args.ticker, args.timesteps, algos)


if __name__ == "__main__":
    main()
