"""Structured generation for Indian legal RAG (PLAN.md M3).

Conforms to protocols.Generator.
Generates atomic claims mapped to retrieved passage IDs and statutory citations
rather than amorphous prose, enabling deterministic claim-by-claim verification (M5).

Invariants:
- I2: Safety components NEVER fail open. Malformed generation returns empty claims,
      triggering deterministic refusal in pipeline._decide.
- I3: Refusal decision is application code, never the LLM.
- I4: Untrusted retrieved text is fenced in <untrusted_document_content> in user role.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from smartlawai import config
from smartlawai.adapters.base import RetrievedChunk
from smartlawai.core import mistral_client
from smartlawai.core.ner import extract_entities
from smartlawai.protocols import Claim, GenerationResult

logger = logging.getLogger(__name__)


def _extract_outermost_json(text: str) -> str:
    """Extract outermost JSON object from raw LLM text, stripping code fences if present."""
    trimmed = text.strip()
    # Check for markdown code fences
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", trimmed, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    # Search for first '{' and last '}'
    first_brace = trimmed.find("{")
    last_brace = trimmed.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return trimmed[first_brace:last_brace + 1].strip()

    return trimmed


def _build_passage_maps(
    passages: list[RetrievedChunk],
) -> tuple[list[str], dict[str, str]]:
    """Format passages into [p1], [p2] blocks and build fuzzy resolution lookup table."""
    context_blocks: list[str] = []
    fuzzy_map: dict[str, str] = {}

    for i, rc in enumerate(passages):
        pid = f"p{i + 1}"
        num_str = str(i + 1)
        cid = rc.chunk.chunk_id

        context_blocks.append(f"[{pid}] {rc.chunk.chunk_text}")

        # Support common variations output by LLMs
        fuzzy_map[pid] = cid
        fuzzy_map[pid.lower()] = cid
        fuzzy_map[pid.upper()] = cid
        fuzzy_map[num_str] = cid
        fuzzy_map[f"[{pid}]"] = cid
        fuzzy_map[f"[{pid.lower()}]"] = cid
        fuzzy_map[f"[{num_str}]"] = cid
        fuzzy_map[f"passage_{pid}"] = cid
        fuzzy_map[f"passage_{num_str}"] = cid
        fuzzy_map[cid] = cid

    return context_blocks, fuzzy_map


class StructuredMistralGenerator:
    """Production generator returning atomic claims with passage IDs and citations.

    Talks to an OpenAI-compatible endpoint (GCP Cloud Run GPU / vLLM / A100)
    using core.mistral_client.complete.
    """

    def __init__(
        self,
        model_id: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.model_id = model_id or config.MISTRAL_GENERATOR_MODEL_ID
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature if temperature is not None else config.GENERATOR_TEMPERATURE
        self.max_tokens = max_tokens if max_tokens is not None else config.GENERATOR_MAX_TOKENS

    def generate(self, question: str, passages: list[RetrievedChunk]) -> GenerationResult:
        """Generate structured claims grounded in the provided passages.

        If passages is empty, returns empty claims and records question as
        unanswerable without making an unnecessary LLM call.
        """
        if not passages:
            return GenerationResult(
                claims=[],
                unanswerable_aspects=[question],
                model_id=self.model_id,
            )

        context_blocks, fuzzy_map = _build_passage_maps(passages)
        context_str = "\n\n".join(context_blocks)

        # I4: Untrusted document text is fenced in user role with explicit hierarchy preamble
        user_prompt = config.GENERATOR_USER_TEMPLATE.format(
            question=question,
            context=context_str,
        )
        system_prompt = config.GENERATOR_SYSTEM_PROMPT

        kwargs: dict[str, Any] = {}
        if self.base_url is not None:
            kwargs["base_url"] = self.base_url
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key

        try:
            raw_response = mistral_client.complete(
                prompt=user_prompt,
                system=system_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                model=self.model_id,
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001 - defensive catch so network/endpoint failures never crash pipeline
            logger.error("Generator endpoint call failed: %s", exc)
            # I2: Safety components never fail open
            return GenerationResult(
                claims=[],
                unanswerable_aspects=[f"generation_failed: {exc}"],
                model_id=self.model_id,
            )

        claims, unanswerable = self._parse_structured_output(
            raw_response=raw_response,
            fuzzy_map=fuzzy_map,
            fallback_question=question,
        )

        return GenerationResult(
            claims=claims,
            unanswerable_aspects=unanswerable,
            model_id=self.model_id,
        )

    def _parse_structured_output(
        self,
        raw_response: str,
        fuzzy_map: dict[str, str],
        fallback_question: str,
    ) -> tuple[list[Claim], list[str]]:
        """Parse and validate JSON response into atomic Claim instances."""
        json_str = _extract_outermost_json(raw_response)
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("Failed to decode generator output as JSON: %s", raw_response)
            # I2: Never fail open on malformed output
            return [], [fallback_question]

        if not isinstance(parsed, dict):
            logger.warning("Generator JSON root is not an object: %s", raw_response)
            return [], [fallback_question]

        raw_claims = parsed.get("claims", [])
        raw_unanswerable = parsed.get("unanswerable_aspects", [])

        unanswerable: list[str] = [
            str(u).strip() for u in raw_unanswerable if str(u).strip()
        ]

        claims: list[Claim] = []
        if isinstance(raw_claims, list):
            for item in raw_claims:
                if not isinstance(item, dict):
                    continue
                claim_text = str(item.get("text", "")).strip()
                if not claim_text:
                    continue

                # Resolve passage IDs
                item_pids = item.get("passage_ids", [])
                if not isinstance(item_pids, list):
                    item_pids = [item_pids]

                resolved_chunk_ids: list[str] = []
                for pid in item_pids:
                    key = str(pid).strip()
                    if key in fuzzy_map:
                        resolved_chunk_ids.append(fuzzy_map[key])
                    elif key.lower() in fuzzy_map:
                        resolved_chunk_ids.append(fuzzy_map[key.lower()])

                # Deduplicate passage IDs preserving order
                deduped_passage_ids = list(dict.fromkeys(resolved_chunk_ids))

                # Collect citations (model output + regex NER fallback)
                item_citations = item.get("citations", [])
                if not isinstance(item_citations, list):
                    item_citations = [item_citations]

                citations_set: set[str] = {
                    str(c).strip() for c in item_citations if str(c).strip()
                }

                # Citation enrichment via legal NER
                ner_res = extract_entities("claim", claim_text)
                for st in ner_res.statutes:
                    citations_set.add(st)
                for prov in ner_res.legal_provisions:
                    citations_set.add(prov)

                claims.append(
                    Claim(
                        text=claim_text,
                        passage_ids=deduped_passage_ids,
                        citations=sorted(citations_set),
                    )
                )

        if not claims and not unanswerable:
            unanswerable = [fallback_question]

        return claims, unanswerable
