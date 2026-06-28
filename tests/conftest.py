"""Shared fixtures for thai-uni-qa-hybrid tests."""
from __future__ import annotations
import os
import pytest

@pytest.fixture
def tmp_db(tmp_path) -> str:
    """A fresh DB path under pytest's tmp dir."""
    return str(tmp_path / "test.db")

@pytest.fixture
def tmp_db_path(tmp_path) -> str:
    """Alias used by ported ingest tests."""
    return str(tmp_path / "test.db")

@pytest.fixture
def mock_llm_api_key(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-test-fake")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("LLM_MODEL", "fake-model")

@pytest.fixture
def monkeypatch_env(monkeypatch):
    """Default env vars so LLM client can be imported without a real .env."""
    monkeypatch.setenv("LLM_BASE_URL", "http://test-llm.local/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    # Force the OpenAI-compatible default; .env may override this for real runs.
    monkeypatch.setenv("LLM_ENDPOINT_PATH", "/chat/completions")