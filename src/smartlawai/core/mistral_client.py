"""HTTP client for an OpenAI-compatible Mistral endpoint (Ollama, vLLM, TGI,
Cloud Run GPU). URL is env-driven, so the same code works everywhere."""
from __future__ import annotations

import os

import requests

BASE_URL = os.environ.get("MISTRAL_BASE_URL", "http://localhost:11434/v1")
MODEL = os.environ.get("MISTRAL_MODEL", "mistral:7b-instruct")
API_KEY = os.environ.get("MISTRAL_API_KEY", "not-needed")


def complete(prompt: str, system: str = "", max_tokens: int = 512,
             temperature: float = 0.2) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": prompt}]
    resp = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"model": MODEL, "messages": msgs,
              "max_tokens": max_tokens, "temperature": temperature},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()
