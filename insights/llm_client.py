"""Unified LLM client with automatic fallback.

Primary: Azure OpenAI
Fallback: Groq (if Azure credentials are missing or fail)

If neither is configured, methods return a friendly placeholder so the rest
of the app continues to work in offline mode.
"""
from __future__ import annotations

from typing import List, Optional

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class LLMClient:
    """Singleton-style LLM client with provider fallback."""

    def __init__(self) -> None:
        self.provider: Optional[str] = None
        self._azure = None
        self._groq = None
        self._init_clients()

    def _init_clients(self) -> None:
        if settings.has_azure_openai():
            try:
                from openai import AzureOpenAI

                self._azure = AzureOpenAI(
                    api_key=settings.azure_openai_api_key,
                    api_version=settings.azure_openai_api_version,
                    azure_endpoint=settings.azure_openai_endpoint,
                )
                self.provider = "azure"
                logger.info("LLM provider: Azure OpenAI (%s)", settings.azure_openai_deployment)
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("Azure OpenAI init failed: %s. Falling back.", exc)

        if settings.has_groq():
            try:
                from groq import Groq

                self._groq = Groq(api_key=settings.groq_api_key)
                self.provider = "groq"
                logger.info("LLM provider: Groq (%s)", settings.groq_model)
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("Groq init failed: %s.", exc)

        logger.warning(
            "No LLM provider configured. AI insights will return placeholders."
        )
        self.provider = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def chat(
        self,
        messages: List[dict],
        temperature: float = 0.3,
        max_tokens: int = 600,
    ) -> str:
        """Send chat messages, return assistant text. Falls back automatically."""
        if self.provider == "azure":
            try:
                return self._chat_azure(messages, temperature, max_tokens)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Azure call failed: %s. Trying Groq.", exc)
                if self._groq is None and settings.has_groq():
                    try:
                        from groq import Groq

                        self._groq = Groq(api_key=settings.groq_api_key)
                    except Exception:  # noqa: BLE001
                        pass
                if self._groq is not None:
                    return self._chat_groq(messages, temperature, max_tokens)

        if self.provider == "groq":
            try:
                return self._chat_groq(messages, temperature, max_tokens)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Groq call failed: %s", exc)

        return self._offline_placeholder(messages)

    def ask(self, system: str, user: str, **kwargs) -> str:
        """Convenience wrapper."""
        return self.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Provider impls
    # ------------------------------------------------------------------
    @staticmethod
    def _is_reasoning_model(name: str) -> bool:
        """o1/o3/o4 family models have different parameter requirements."""
        n = (name or "").lower()
        return any(n.startswith(p) or f"-{p}" in n or f"/{p}" in n
                   for p in ("o1", "o3", "o4"))

    def _chat_azure(self, messages, temperature, max_tokens) -> str:
        deployment = settings.azure_openai_deployment
        kwargs = {"model": deployment, "messages": messages}

        if self._is_reasoning_model(deployment):
            # Reasoning models: only max_completion_tokens, no custom temperature.
            # Reasoning models consume tokens internally - give them headroom.
            kwargs["max_completion_tokens"] = max(max_tokens * 8, 4000)
        else:
            kwargs["temperature"] = temperature
            kwargs["max_tokens"] = max_tokens

        resp = self._azure.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def _chat_groq(self, messages, temperature, max_tokens) -> str:
        resp = self._groq.chat.completions.create(
            model=settings.groq_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    @staticmethod
    def _offline_placeholder(messages) -> str:
        last = messages[-1].get("content", "") if messages else ""
        return (
            "[LLM OFFLINE] No Azure OpenAI or Groq credentials found in .env. "
            "Configure either AZURE_OPENAI_* or GROQ_API_KEY to enable AI insights. "
            f"(Prompt was: {last[:120]}...)"
        )


# Module-level singleton
_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
