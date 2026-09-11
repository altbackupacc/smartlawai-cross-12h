"""Entailment verification: dual-scorer architecture (HHEM-2.1 and InLegalNLI).

Implements Invariant I2 (never fail open, never substitute default scores)
and Invariant I11 (dual scorer: HHEM-2.1 + InLegalNLI slot, both recorded in trace).
Vendored loader explicitly disables trust_remote_code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from smartlawai import config
from smartlawai.adapters.base import RetrievedChunk
from smartlawai.protocols import Claim

logger = logging.getLogger(__name__)


@dataclass
class EntailmentScore:
    """Score for a single verifier component."""

    name: str
    value: float | None
    status: str  # "ok" | "unavailable"
    reason: str = ""


@dataclass
class ClaimEntailmentResult:
    """Combined entailment evaluation for a single claim."""

    claim_text: str
    passage_id: str
    hhem_score: EntailmentScore
    inlegalnli_score: EntailmentScore
    is_entailed: bool
    status: str  # "ok" | "unavailable"
    reason: str = ""


class HHEMScorer:
    """Wrapper for Vectara HHEM-2.1-Open without trust_remote_code=True."""

    def __init__(self, model_id: str = "vectara/hallucination_evaluation_model", device: str | None = None) -> None:
        self.model_id = model_id
        self.device = device
        self._model = None
        self._tokenizer = None
        self._load_failed = False
        self._load_error = ""

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if self._load_failed:
            return False

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            # Explicitly disallow remote code execution (PLAN.md M5.2)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=False)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id, trust_remote_code=False
            )
            if self.device:
                self._model.to(torch.device(self.device))
            self._model.eval()
            return True
        except Exception as e:  # noqa: BLE001
            self._load_failed = True
            self._load_error = str(e)
            logger.warning("HHEM model failed to load: %s", e)
            return False

    def score_pair(self, passage: str, claim: str) -> EntailmentScore:
        """Score (passage, claim) pair.

        Returns status="unavailable" on any failure (I2: never return a fake score).
        """
        if not self._ensure_loaded():
            return EntailmentScore(
                name="hhem-2.1",
                value=None,
                status="unavailable",
                reason=f"hhem_unavailable: {self._load_error}",
            )

        try:
            import torch

            inputs = self._tokenizer(
                passage,
                claim,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            if self.device:
                inputs = {k: v.to(torch.device(self.device)) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits
                # Classification output: probability of entailment / factuality
                probs = torch.softmax(logits, dim=-1)
                # Typically index 1 is factual / entailed in 2-class HHEM
                score_val = float(probs[0, 1].item())

            return EntailmentScore(
                name="hhem-2.1",
                value=score_val,
                status="ok",
                reason="scored",
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("HHEM inference error: %s", e)
            return EntailmentScore(
                name="hhem-2.1",
                value=None,
                status="unavailable",
                reason=f"hhem_inference_failed: {e}",
            )


class InLegalNLIScorer:
    """Slot for InLegalNLI (fine-tuned in M8a).

    Returns status="unavailable" until M8a model weights are trained (PLAN.md M5.2).
    """

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or getattr(config, "INLEGALNLI_MODEL_ID", "smartlawai/InLegalNLI")

    def score_pair(self, passage: str, claim: str) -> EntailmentScore:
        # Per PLAN.md M5.2: Until M8a lands, run HHEM-only and leave InLegalNLI slot returning unavailable
        return EntailmentScore(
            name="inlegalnli",
            value=None,
            status="unavailable",
            reason="inlegalnli_pending_m8a_training",
        )


class DualEntailmentVerifier:
    """Dual-scorer entailment verifier coordinating HHEM-2.1 and InLegalNLI."""

    def __init__(self, hhem_scorer: HHEMScorer | None = None, nli_scorer: InLegalNLIScorer | None = None) -> None:
        self.hhem = hhem_scorer or HHEMScorer()
        self.nli = nli_scorer or InLegalNLIScorer()
        self.threshold = getattr(config, "CLAIM_ENTAILMENT_THRESHOLD", 0.5)

    def verify_claim(
        self, claim: Claim, passages_by_id: dict[str, RetrievedChunk]
    ) -> ClaimEntailmentResult:
        """Verify entailment of a claim against its cited passages."""
        if not claim.passage_ids:
            return ClaimEntailmentResult(
                claim_text=claim.text,
                passage_id="",
                hhem_score=EntailmentScore(name="hhem-2.1", value=None, status="unavailable", reason="no_passage_id"),
                inlegalnli_score=EntailmentScore(name="inlegalnli", value=None, status="unavailable", reason="no_passage_id"),
                is_entailed=False,
                status="unavailable",
                reason="no_passage_ids_provided",
            )

        # Combine text of cited passages
        passage_texts = []
        for pid in claim.passage_ids:
            chunk_obj = passages_by_id.get(pid)
            if chunk_obj:
                raw_chunk = getattr(chunk_obj, "chunk", chunk_obj)
                txt = getattr(raw_chunk, "chunk_text", getattr(raw_chunk, "text", ""))
                if txt:
                    passage_texts.append(txt)

        combined_passage = " ".join(passage_texts).strip()
        if not combined_passage:
            return ClaimEntailmentResult(
                claim_text=claim.text,
                passage_id=",".join(claim.passage_ids),
                hhem_score=EntailmentScore(name="hhem-2.1", value=None, status="unavailable", reason="passage_text_empty"),
                inlegalnli_score=EntailmentScore(name="inlegalnli", value=None, status="unavailable", reason="passage_text_empty"),
                is_entailed=False,
                status="unavailable",
                reason="passage_text_missing",
            )

        # Dual scoring (Invariant I11)
        hhem_res = self.hhem.score_pair(combined_passage, claim.text)
        nli_res = self.nli.score_pair(combined_passage, claim.text)

        # Determine verdict based on operational scorer (HHEM until M8a)
        if hhem_res.status == "unavailable":
            # Invariant I2: Never fail open on verifier failure
            return ClaimEntailmentResult(
                claim_text=claim.text,
                passage_id=claim.passage_ids[0],
                hhem_score=hhem_res,
                inlegalnli_score=nli_res,
                is_entailed=False,
                status="unavailable",
                reason="hhem_scorer_unavailable",
            )

        is_entailed = (hhem_res.value is not None) and (hhem_res.value >= self.threshold)
        return ClaimEntailmentResult(
            claim_text=claim.text,
            passage_id=claim.passage_ids[0],
            hhem_score=hhem_res,
            inlegalnli_score=nli_res,
            is_entailed=is_entailed,
            status="ok",
            reason="entailed" if is_entailed else "not_entailed",
        )
