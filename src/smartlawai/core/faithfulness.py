"""Faithfulness layer: HHEM-2.1 (Vectara) + LLM-as-judge. Answers below
THRESHOLD are refused, not shown (non-optional safety layer)."""
from __future__ import annotations

import json

from smartlawai.core.mistral_client import complete

HHEM_MODEL = "vectara/hallucination_evaluation_model"
THRESHOLD = 0.70

_JUDGE = ("Score 0.0-1.0 how fully the ANSWER is supported by the CONTEXT. "
          "Reply ONLY JSON: {{\"score\": float, \"reason\": str}}.\n\n"
          "CONTEXT:\n{ctx}\n\nANSWER:\n{ans}")


class FaithfulnessEvaluator:
    def __init__(self, use_hhem: bool = True) -> None:
        self.hhem = None
        if use_hhem:
            try:
                from transformers import AutoModelForSequenceClassification
                self.hhem = AutoModelForSequenceClassification.from_pretrained(
                    HHEM_MODEL, trust_remote_code=True)
            except Exception:
                self.hhem = None  # fall back to judge-only

    def hhem_score(self, context: str, answer: str) -> float:
        if self.hhem is None:
            return 0.5
        try:
            return float(self.hhem.predict([(context, answer)])[0])
        except Exception:
            return 0.5

    def judge_score(self, context: str, answer: str) -> tuple[float, str]:
        try:
            raw = complete(_JUDGE.format(ctx=context[:3000], ans=answer),
                           max_tokens=150, temperature=0.0)
            data = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
            return float(data["score"]), data.get("reason", "")
        except Exception as e:  # noqa: BLE001
            return 0.5, f"judge_parse_error: {e}"

    def evaluate(self, context: str, answer: str) -> dict:
        h = self.hhem_score(context, answer)
        j, reason = self.judge_score(context, answer)
        combined = round(0.6 * h + 0.4 * j, 4)
        return {"hhem": round(h, 4), "judge": round(j, 4), "score": combined,
                "passed": combined >= THRESHOLD, "reason": reason}
