"""Centralized application configuration.

All configuration is loaded from environment variables (.env file) using
Pydantic settings. The LLM section supports a primary (Azure OpenAI) +
fallback (Groq) pattern - if Azure OpenAI credentials are missing or
invalid, the app automatically uses Groq.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from typing_extensions import Annotated


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # LLM - Azure OpenAI (primary)
    # ------------------------------------------------------------------
    azure_openai_endpoint: Optional[str] = Field(default=None)
    azure_openai_api_key: Optional[str] = Field(default=None)
    azure_openai_deployment: Optional[str] = Field(default=None)
    azure_openai_api_version: str = Field(default="2024-08-01-preview")

    # ------------------------------------------------------------------
    # LLM - Groq (fallback)
    # ------------------------------------------------------------------
    groq_api_key: Optional[str] = Field(default=None)
    groq_model: str = Field(default="llama-3.3-70b-versatile")

    # ------------------------------------------------------------------
    # Trading config
    # ------------------------------------------------------------------
    default_tickers: Annotated[List[str], NoDecode] = Field(
        default_factory=lambda: ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]
    )
    data_start_date: str = Field(default="2018-01-01")
    data_end_date: str = Field(default="2024-12-31")
    initial_capital: float = Field(default=100_000.0)
    transaction_cost_pct: float = Field(default=0.001)

    # ------------------------------------------------------------------
    # Training config
    # ------------------------------------------------------------------
    total_timesteps: int = Field(default=50_000)
    seed: int = Field(default=42)
    log_level: str = Field(default="INFO")

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    project_root: Path = Field(default=PROJECT_ROOT)

    @property
    def data_raw_dir(self) -> Path:
        return self.project_root / "data" / "raw"

    @property
    def data_processed_dir(self) -> Path:
        return self.project_root / "data" / "processed"

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"

    @property
    def logs_dir(self) -> Path:
        return self.project_root / "logs"

    @property
    def mlruns_dir(self) -> Path:
        return self.project_root / "mlruns"

    @field_validator("default_tickers", mode="before")
    @classmethod
    def _split_tickers(cls, v):
        if isinstance(v, str):
            return [t.strip() for t in v.split(",") if t.strip()]
        return v

    # ------------------------------------------------------------------
    # LLM availability helpers
    # ------------------------------------------------------------------
    def has_azure_openai(self) -> bool:
        """Check if Azure OpenAI is fully configured."""
        return all(
            [
                self.azure_openai_endpoint,
                self.azure_openai_api_key,
                self.azure_openai_deployment,
                self.azure_openai_endpoint and "your-" not in self.azure_openai_endpoint,
                self.azure_openai_api_key and "your-" not in self.azure_openai_api_key,
            ]
        )

    def has_groq(self) -> bool:
        """Check if Groq fallback is configured."""
        return bool(
            self.groq_api_key and "your-" not in (self.groq_api_key or "")
        )

    def ensure_dirs(self) -> None:
        """Create all required project directories."""
        for d in (
            self.data_raw_dir,
            self.data_processed_dir,
            self.models_dir,
            self.logs_dir,
            self.mlruns_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
