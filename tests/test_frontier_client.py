"""Tests for baselines/frontier_client.py. No test ever holds a real key or
makes a real network call -- the Anthropic SDK itself is faked via sys.modules."""
from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest


def _reload_with_env(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import baselines.frontier_client as fc
    importlib.reload(fc)
    return fc


def _install_fake_anthropic(monkeypatch, captured: dict):
    class _FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(content=[SimpleNamespace(text="mocked answer")])

    class _FakeAnthropic:
        def __init__(self, api_key):
            captured["api_key"] = api_key
            self.messages = _FakeMessages()

    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=_FakeAnthropic))


def test_complete_frontier_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    fc = _reload_with_env(monkeypatch, FRONTIER_PROVIDER="anthropic")
    with pytest.raises(RuntimeError):
        fc.complete_frontier("question")


def test_complete_frontier_calls_anthropic_sdk_with_expected_shape(monkeypatch):
    fc = _reload_with_env(monkeypatch, FRONTIER_PROVIDER="anthropic",
                          ANTHROPIC_API_KEY="test-key", FRONTIER_MODEL="claude-test")
    captured: dict = {}
    _install_fake_anthropic(monkeypatch, captured)

    answer = fc.complete_frontier("question", system="be terse")

    assert answer == "mocked answer"
    assert captured["api_key"] == "test-key"
    assert captured["model"] == "claude-test"
    assert captured["system"] == "be terse"
    assert captured["messages"] == [{"role": "user", "content": "question"}]


def test_unsupported_provider_raises(monkeypatch):
    fc = _reload_with_env(monkeypatch, FRONTIER_PROVIDER="not-a-real-provider",
                          ANTHROPIC_API_KEY="test-key")
    with pytest.raises(ValueError):
        fc.complete_frontier("question")


def test_frontier_context_never_lands_in_system_role(monkeypatch):
    """I4: fenced retrieved-document context must be in the user message,
    never the system role."""
    fc = _reload_with_env(monkeypatch, FRONTIER_PROVIDER="anthropic",
                          ANTHROPIC_API_KEY="test-key", FRONTIER_MODEL="claude-test")
    captured: dict = {}
    _install_fake_anthropic(monkeypatch, captured)

    prompt = ("Question: q\n\n<untrusted_document_content>\nSOME DOC TEXT"
             "\n</untrusted_document_content>")
    fc.complete_frontier(prompt, system="You are an Indian legal assistant.")

    assert "SOME DOC TEXT" not in captured["system"]
    assert "<untrusted_document_content>" not in captured["system"]
    assert captured["messages"][0]["content"] == prompt
    assert "<untrusted_document_content>" in captured["messages"][0]["content"]
