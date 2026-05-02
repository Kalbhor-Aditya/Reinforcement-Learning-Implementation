"""End-to-end training pipeline with MLflow logging.

CLI:
    python -m training.trainer --algo PPO --ticker RELIANCE.NS --timesteps 50000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mlflow

from agents import AGENT_REGISTRY, build_agent
from agents.registry import is_continuous
from config import settings
from training.env_builder import make_envs
from utils.logger import get_logger

logger = get_logger(__name__)


def train_agent(
    algo: str,
    ticker: str,
    total_timesteps: int = 50_000,
    save_model: bool = True,
) -> Path:
    """Train one agent end-to-end and save the model."""
    if algo not in AGENT_REGISTRY:
        raise ValueError(f"Unknown algo '{algo}'. Choices: {list(AGENT_REGISTRY)}")

    continuous = is_continuous(algo)
    train_env, _ = make_envs(ticker, continuous=continuous)

    mlflow.set_tracking_uri(f"file:///{settings.mlruns_dir.as_posix()}")
    mlflow.set_experiment(f"rl_trading_{ticker}")

    with mlflow.start_run(run_name=f"{algo}_{ticker}") as run:
        mlflow.log_params(
            {
                "algo": algo,
                "ticker": ticker,
                "total_timesteps": total_timesteps,
                "initial_capital": settings.initial_capital,
                "continuous_actions": continuous,
                "seed": settings.seed,
            }
        )

        agent = build_agent(algo, env=train_env, seed=settings.seed)
        agent.train(total_timesteps=total_timesteps)

        save_path = settings.models_dir / f"{algo}_{ticker.replace('.', '_')}.zip"
        if save_model:
            agent.save(save_path)
            mlflow.log_artifact(str(save_path))

        logger.info("Training complete: run_id=%s", run.info.run_id)
        return save_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train an RL trading agent.")
    p.add_argument("--algo", type=str, required=True, choices=list(AGENT_REGISTRY))
    p.add_argument("--ticker", type=str, default="RELIANCE.NS")
    p.add_argument("--timesteps", type=int, default=settings.total_timesteps)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    train_agent(args.algo, args.ticker, args.timesteps)


if __name__ == "__main__":
    main()
