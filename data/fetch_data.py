"""Download historical OHLCV data for Indian (NSE/BSE) stocks via yfinance.

Use suffixes:
- .NS  -> NSE (e.g., RELIANCE.NS)
- .BO  -> BSE (e.g., RELIANCE.BO)

Run as a script:
    python -m data.fetch_data
    python -m data.fetch_data --tickers RELIANCE.NS,TCS.NS --start 2020-01-01
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yfinance as yf

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def download_ticker(
    ticker: str,
    start: str,
    end: str,
    save_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Download OHLCV data for a single ticker and save as CSV.

    Returns a DataFrame indexed by Date with columns: Open, High, Low, Close,
    Volume. If yfinance returns multi-level columns (when there's only one
    ticker but auto_adjust=True), they are flattened.
    """
    save_dir = save_dir or settings.data_raw_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading %s from %s to %s", ticker, start, end)
    df = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
    )

    if df is None or df.empty:
        logger.warning("No data returned for %s", ticker)
        return pd.DataFrame()

    # Flatten possibly multi-level columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    # Keep standard OHLCV
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].copy()
    df.index.name = "Date"

    out = save_dir / f"{ticker.replace('.', '_')}.csv"
    df.to_csv(out)
    logger.info("Saved %d rows to %s", len(df), out)
    return df


def download_many(
    tickers: List[str],
    start: str,
    end: str,
) -> dict[str, pd.DataFrame]:
    """Download multiple tickers sequentially."""
    return {t: download_ticker(t, start, end) for t in tickers}


def load_ticker(ticker: str) -> pd.DataFrame:
    """Load an already-downloaded ticker CSV from data/raw/."""
    path = settings.data_raw_dir / f"{ticker.replace('.', '_')}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No data file for {ticker}. Run: python -m data.fetch_data --tickers {ticker}"
        )
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch Indian stock OHLCV data.")
    p.add_argument(
        "--tickers",
        type=str,
        default=",".join(settings.default_tickers),
        help="Comma-separated tickers (e.g., RELIANCE.NS,TCS.NS).",
    )
    p.add_argument("--start", type=str, default=settings.data_start_date)
    p.add_argument("--end", type=str, default=settings.data_end_date)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    download_many(tickers, args.start, args.end)
    logger.info("Done. Files saved to %s", settings.data_raw_dir)


if __name__ == "__main__":
    main()
