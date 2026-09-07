"""Tests for baselines/generators.py. No test makes a real network call --
core.mistral_client.complete and frontier_client.complete_frontier are
monkeypatched at the point generators.py calls them."""
from __future__ import annotations

from smartlawai.adapters.base import Chunk, RetrievedChunk


def _rc(chunk_id, text, doc_id="doc-1"):
    return RetrievedChunk(
        chunk=Chunk(chunk_id=chunk_id, doc_id=doc_id, chunk_index=0,
                   chunk_text=text, char_start=0, char_end=len(text)),
        score=1.0)


def test_mistral_zero_shot_sends_bare_question_no_context_fence(monkeypatch):
    import baselines.generators as gens

    calls = {}

    def fake_complete(prompt, system="", **kwargs):
        calls["prompt"] = prompt
        calls["system"] = system
        calls["kwargs"] = kwargs
        return "an answer"

    monkeypatch.setattr(gens.mistral_client, "complete", fake_complete)
    result = gens.MistralGenerator().generate("What is Section 302 IPC?", [])

    assert calls["prompt"] == "What is Section 302 IPC?"
    assert "<untrusted_document_content>" not in calls["prompt"]
    assert calls["kwargs"] == {}  # no base_url/model/api_key override for Mistral
    assert result.model_id == gens.config.MISTRAL_MODEL_ID
    assert result.claims[0].text == "an answer"
    assert result.claims[0].passage_ids == []


def test_mistral_rag_fences_context_in_user_role(monkeypatch):
    import baselines.generators as gens

    calls = {}

    def fake_complete(prompt, system="", **kwargs):
        calls["prompt"] = prompt
        calls["system"] = system
        return "an answer citing Section 74 of the Indian Contract Act, 1872"

    monkeypatch.setattr(gens.mistral_client, "complete", fake_complete)
    passages = [_rc("c1", "Section 74 covers damages.")]
    result = gens.MistralGenerator().generate("What about damages?", passages)

    assert "<untrusted_document_content>" in calls["prompt"]
    assert "Section 74 covers damages." in calls["prompt"]
    assert "<untrusted_document_content>" not in calls["system"]
    assert result.claims[0].passage_ids == ["c1"]
    assert len(result.claims[0].citations) >= 1


def test_saullm_generate_uses_override_endpoint(monkeypatch):
    import baselines.generators as gens

    calls = {}

    def fake_complete(prompt, system="", **kwargs):
        calls.update(kwargs)
        return "saullm answer"

    monkeypatch.setattr(gens.mistral_client, "complete", fake_complete)
    result = gens.SaulLMGenerator().generate("question", [])

    assert calls["base_url"] == gens.config.SAULLM_BASE_URL
    assert calls["model"] == gens.config.SAULLM_MODEL_ID
    assert calls["api_key"] == gens.config.SAULLM_API_KEY
    assert result.model_id == gens.config.SAULLM_MODEL_ID


def test_frontier_generator_zero_shot(monkeypatch):
    import baselines.generators as gens

    monkeypatch.setattr(gens, "complete_frontier",
                        lambda prompt, system="", **kw: "frontier answer")
    result = gens.FrontierGenerator().generate("question", [])
    assert result.claims[0].text == "frontier answer"


def test_frontier_generator_rag_fences_context(monkeypatch):
    import baselines.generators as gens

    calls = {}

    def fake(prompt, system="", **kw):
        calls["prompt"] = prompt
        calls["system"] = system
        return "frontier rag answer"

    monkeypatch.setattr(gens, "complete_frontier", fake)
    passages = [_rc("c1", "Some passage text.")]
    gens.FrontierGenerator().generate("question", passages)
    assert "<untrusted_document_content>" in calls["prompt"]
    assert "<untrusted_document_content>" not in calls["system"]


def test_empty_generation_produces_no_claims(monkeypatch):
    import baselines.generators as gens

    monkeypatch.setattr(gens.mistral_client, "complete", lambda *a, **k: "   ")
    result = gens.MistralGenerator().generate("q", [])
    assert result.claims == []
    assert "empty_generation" in result.unanswerable_aspects
