"""HTTP client for a frontier-model API (baselines only -- never used in
serving). Provider is env-selected and dispatched so adding a second provider
later is additive, not a rewrite, mirroring core/mistral_client.py's own
env-driven design. Anthropic is the only provider wired for now (M7 decision).

I7: the API key is read from env only, never passed in a chat message or
hardcoded. I4: the fenced <untrusted_document_content> context is built by
callers (see baselines/generators.py) and always passed as the user message,
never as `system` -- this module doesn't itself insert document text anywhere,
so there is no path for it to land in the system role."""
from __future__ import annotations

import os

PROVIDER = os.environ.get("FRONTIER_PROVIDER", "anthropic")
MODEL = os.environ.get("FRONTIER_MODEL", "")
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


def complete_frontier(prompt: str, system: str = "", max_tokens: int = 512,
                       temperature: float = 0.2) -> str:
    if not API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set -- put it in .env, never in code or chat (I7)")
    if PROVIDER == "anthropic":
        return _complete_anthropic(prompt, system, max_tokens, temperature)
    raise ValueError(f"Unsupported FRONTIER_PROVIDER={PROVIDER!r}")


def _complete_anthropic(prompt: str, system: str, max_tokens: int,
                         temperature: float) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=API_KEY)
    msg = client.messages.create(
        model=MODEL, max_tokens=max_tokens, temperature=temperature,
        system=system, messages=[{"role": "user", "content": prompt}])
    return msg.content[0].text
