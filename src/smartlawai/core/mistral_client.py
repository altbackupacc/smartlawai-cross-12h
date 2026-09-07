"""HTTP client for an OpenAI-compatible Mistral endpoint (Ollama, vLLM, TGI,
Cloud Run GPU). URL is env-driven, so the same code works everywhere."""
from __future__ import annotations

import os

import requests

BASE_URL = os.environ.get("MISTRAL_BASE_URL", "http://localhost:11434/v1")
MODEL = os.environ.get("MISTRAL_MODEL", "mistral:7b-instruct")
API_KEY = os.environ.get("MISTRAL_API_KEY", "not-needed")


def complete(prompt: str, system: str = "", max_tokens: int = 512,
             temperature: float = 0.2, *,
             base_url: str | None = None, model: str | None = None,
             api_key: str | None = None) -> str:
    """base_url/model/api_key default to this module's env-driven constants
    (the Mistral endpoint) but can be overridden to hit any other
    OpenAI-compatible vLLM/TGI/Ollama deployment -- e.g. M7's SaulLM-7B
    baseline, served the same way Mistral is, just a different model."""
    base_url = base_url or BASE_URL
    model = model or MODEL
    api_key = api_key or API_KEY
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": prompt}]
    resp = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "messages": msgs,
              "max_tokens": max_tokens, "temperature": temperature},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()
