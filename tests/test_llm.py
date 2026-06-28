"""Tests for LLM client. Uses a fake HTTP backend."""
import json
import pytest
from src.llm import chat, LLMConfig, LLMError

class FakeResponse:
    def __init__(self, payload):
        self._payload = payload
    def raise_for_status(self):
        return None
    def json(self):
        return self._payload

def test_chat_sends_messages_and_returns_text(monkeypatch, monkeypatch_env):
    from src import llm
    captured = {}
    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return FakeResponse({
            "choices": [{"message": {"content": "สวัสดี"}}]
        })
    monkeypatch.setattr(llm.requests, "post", fake_post)
    text = chat(
        messages=[{"role": "user", "content": "hi"}],
        model="test-model",
        temperature=0.0,
    )
    assert text == "สวัสดี"
    assert captured["url"].endswith("/chat/completions")
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["json"]["model"] == "test-model"
    assert captured["json"]["temperature"] == 0.0
    assert captured["json"]["messages"][0]["content"] == "hi"

def test_chat_supports_json_response_format(monkeypatch, monkeypatch_env):
    from src import llm
    def fake_post(url, json=None, headers=None, timeout=None):
        return FakeResponse({
            "choices": [{"message": {"content": '{"sql":"SELECT 1"}'}}]
        })
    monkeypatch.setattr(llm.requests, "post", fake_post)
    text = chat(
        messages=[{"role": "user", "content": "x"}],
        model="m", temperature=0.0,
        response_format={"type": "json_object"},
    )
    assert text == '{"sql":"SELECT 1"}'

def test_chat_raises_on_http_error(monkeypatch, monkeypatch_env):
    from src import llm
    class BadResponse(FakeResponse):
        def raise_for_status(self):
            raise llm.requests.HTTPError("500 Server Error")
    monkeypatch.setattr(llm.requests, "post", lambda *a, **kw: BadResponse({}))
    with pytest.raises(LLMError):
        chat(messages=[{"role":"user","content":"x"}], model="m", temperature=0.0)