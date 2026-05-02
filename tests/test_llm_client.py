"""Tests for LLM client (verifies offline placeholder works without keys)."""
from insights.llm_client import LLMClient


def test_llm_client_offline_placeholder(monkeypatch):
    # Force no provider
    client = LLMClient.__new__(LLMClient)
    client.provider = None
    client._azure = None
    client._groq = None
    out = client.chat([{"role": "user", "content": "hi"}])
    assert "[LLM OFFLINE]" in out


def test_reasoning_model_detection():
    assert LLMClient._is_reasoning_model("o1")
    assert LLMClient._is_reasoning_model("o1-mini")
    assert LLMClient._is_reasoning_model("o3")
    assert LLMClient._is_reasoning_model("o4-mini")
    assert not LLMClient._is_reasoning_model("gpt-4o")
    assert not LLMClient._is_reasoning_model("gpt-4o-mini")
    assert not LLMClient._is_reasoning_model("gpt-3.5-turbo")

