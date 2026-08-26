"""Clause extraction as QA-span extraction (CUAD approach) + 3-tier risk flagging.
Each clause type is a question; the QA model returns the answer span. Risk via
zero-shot NLI. torch/transformers imported lazily."""
from __future__ import annotations

import uuid

from smartlawai.adapters.base import Clause

CUAD_QA_MODEL = "Rakib/roberta-base-on-cuad"
NLI_MODEL = "facebook/bart-large-mnli"
ANSWER_THRESHOLD = 0.15

# 12 Indian-specific clause types as CUAD-style questions
INDIAN_CLAUSE_QUESTIONS: dict[str, tuple[str, str, str]] = {
    "ARBITRATION_IN":       ("Does the contract contain an arbitration clause?", "Arbitration and Conciliation Act, 1996", "MEDIUM"),
    "JURISDICTION_IN":      ("Which courts have jurisdiction over disputes?", "CPC, 1908 (S.20)", "MEDIUM"),
    "FORCE_MAJEURE_IN":     ("Is there a force majeure clause?", "Indian Contract Act, 1872 (S.32/56)", "MEDIUM"),
    "LIQUIDATED_DAMAGES_IN":("What are the liquidated damages or penalty provisions?", "Indian Contract Act, 1872 (S.74)", "HIGH"),
    "STAMP_DUTY_IN":        ("Is stamp duty addressed?", "Indian Stamp Act, 1899", "LOW"),
    "GST_TAX_IN":           ("How are GST or taxes handled?", "CGST/SGST Act, 2017", "LOW"),
    "NON_COMPETE_IN":       ("Is there a non-compete or restraint of trade clause?", "Indian Contract Act, 1872 (S.27)", "HIGH"),
    "INDEMNITY_IN":         ("Is there an indemnity clause?", "Indian Contract Act, 1872 (S.124)", "MEDIUM"),
    "TERMINATION_IN":       ("What are the termination conditions?", "Indian Contract Act, 1872", "MEDIUM"),
    "CONFIDENTIALITY_IN":   ("Is there a confidentiality or NDA clause?", "Indian Contract Act, 1872", "LOW"),
    "GOVERNING_LAW_IN":     ("What is the governing law?", "General", "LOW"),
    "DISPUTE_RESOLUTION_IN":("How are disputes resolved?", "Arbitration and Conciliation Act, 1996", "MEDIUM"),
}
RISK_LABELS = ["fair standard clause", "ambiguous clause needing review",
               "unfair or high-risk clause"]
_LABEL_TO_TIER = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}


class ClauseExtractor:
    def __init__(self) -> None:
        import torch
        from transformers import pipeline
        dev = 0 if torch.cuda.is_available() else -1
        self.qa = pipeline("question-answering", model=CUAD_QA_MODEL, device=dev)
        self.nli = pipeline("zero-shot-classification", model=NLI_MODEL, device=dev)

    def _risk(self, clause_text: str, default_tier: str) -> tuple[str, float]:
        if not clause_text.strip():
            return default_tier, 0.5
        res = self.nli(clause_text[:1000], candidate_labels=RISK_LABELS)
        top = RISK_LABELS.index(res["labels"][0])
        return _LABEL_TO_TIER[top], float(res["scores"][0])

    def extract(self, doc_id: str, text: str) -> list[Clause]:
        """Extract clauses using overlapping sliding windows over the full document.

        Windows of WINDOW_SIZE chars with WINDOW_OVERLAP overlap ensure clauses
        in later sections of long documents are found. Per clause type, only
        the highest-confidence extraction is kept.
        """
        WINDOW_SIZE = 4000
        WINDOW_OVERLAP = 500

        # Build list of (window_start, window_text) pairs
        windows: list[tuple[int, str]] = []
        if len(text) <= WINDOW_SIZE:
            windows.append((0, text))
        else:
            step = WINDOW_SIZE - WINDOW_OVERLAP
            for start in range(0, len(text), step):
                win = text[start:start + WINDOW_SIZE]
                if win.strip():
                    windows.append((start, win))
                if start + WINDOW_SIZE >= len(text):
                    break

        # For each clause type, track best hit across all windows
        best: dict[str, tuple[float, Clause]] = {}

        for win_start, win_text in windows:
            for code, (question, statute, default_tier) in INDIAN_CLAUSE_QUESTIONS.items():
                ans = self.qa(question=question, context=win_text,
                              handle_impossible_answer=True)
                if ans["score"] < ANSWER_THRESHOLD or not ans["answer"].strip():
                    continue
                # Keep only the highest-confidence hit per clause type
                if code in best and best[code][0] >= ans["score"]:
                    continue
                tier, risk_score = self._risk(ans["answer"], default_tier)
                clause = Clause(
                    clause_id=f"cl-{uuid.uuid4().hex[:10]}", doc_id=doc_id,
                    chunk_id=None, clause_type=code, clause_text=ans["answer"],
                    span_start=win_start + ans["start"],
                    span_end=win_start + ans["end"],
                    risk_tier=tier, risk_score=round(risk_score, 4),
                    statute_ref=statute)
                best[code] = (ans["score"], clause)

        return [clause for _, clause in best.values()]
