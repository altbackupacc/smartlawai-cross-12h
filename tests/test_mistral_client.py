"""Tests for smartlawai.core.mistral_client -- previously untested. Written
while integrating M7's baselines/generators.py, which calls complete() with
base_url/model/api_key overrides for the SaulLM-7B baseline (served the same
OpenAI-compatible shape as Mistral, just a different endpoint/model).

The gap this closes: M7's own test suite verifies SaulLMGenerator passes the
right override kwargs to a *mocked* mistral_client.complete, but never
exercises the real function -- so a signature mismatch (e.g. the override
kwargs not existing yet) would pass every M7 test while still breaking at
runtime. These tests call the real complete() with requests.post mocked at
the HTTP boundary instead, so the actual kwarg-plumbing is exercised."""
from __future__ import annotations

from unittest import mock

from smartlawai.core import mistral_client


def _fake_response(content: str):
    resp = mock.Mock()
    resp.raise_for_status = lambda: None
    resp.json = lambda: {"choices": [{"message": {"content": content}}]}
    return resp


def test_complete_uses_module_defaults_when_no_override_given():
    with mock.patch.object(mistral_client.requests, "post",
                           return_value=_fake_response("default answer")) as post:
        result = mistral_client.complete("hello", system="be terse")

    assert result == "default answer"
    assert post.call_args[0][0] == f"{mistral_client.BASE_URL}/chat/completions"
    assert post.call_args[1]["headers"]["Authorization"] == f"Bearer {mistral_client.API_KEY}"
    assert post.call_args[1]["json"]["model"] == mistral_client.MODEL
    assert post.call_args[1]["json"]["messages"] == [
        {"role": "system", "content": "be terse"}, {"role": "user", "content": "hello"}]


def test_complete_accepts_base_url_model_api_key_overrides():
    """The real integration point M7's SaulLMGenerator depends on -- these
    three keyword-only params must exist and actually be used, not just
    accepted and silently ignored."""
    with mock.patch.object(mistral_client.requests, "post",
                           return_value=_fake_response("saullm answer")) as post:
        result = mistral_client.complete(
            "question", base_url="http://saullm:9999/v1",
            model="saullm-7b-test", api_key="saullm-key")

    assert result == "saullm answer"
    assert post.call_args[0][0] == "http://saullm:9999/v1/chat/completions"
    assert post.call_args[1]["headers"]["Authorization"] == "Bearer saullm-key"
    assert post.call_args[1]["json"]["model"] == "saullm-7b-test"


def test_complete_with_no_system_omits_system_message():
    with mock.patch.object(mistral_client.requests, "post",
                           return_value=_fake_response("answer")) as post:
        mistral_client.complete("just a question")

    assert post.call_args[1]["json"]["messages"] == [
        {"role": "user", "content": "just a question"}]
