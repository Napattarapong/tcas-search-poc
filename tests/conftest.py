"""Shared fixtures for thai-uni-qa-hybrid tests."""
from __future__ import annotations
import os
import pytest

@pytest.fixture
def tmp_db(tmp_path) -> str:
    """A fresh DB path under pytest's tmp dir."""
    return str(tmp_path / "test.db")

@pytest.fixture
def mock_llm_api_key(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-test-fake")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("LLM_MODEL", "fake-model")