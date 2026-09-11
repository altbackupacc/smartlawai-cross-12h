"""Out-of-Scope (OOS) detector and Platt calibration (PLAN.md M5.6).

Replaces the arbitrary raw-logit floor (RERANK_FLOOR = -10.0) with calibrated
in-scope probability via Platt scaling:
    P(in_scope | logit) = 1 / (1 + exp(-(A * logit + B)))
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class PlattScaler:
    """Calibrated logistic regression parameters over cross-encoder logits."""

    a: float = 1.0
    b: float = 0.0
    threshold: float = 0.5  # In-scope probability threshold

    def predict_proba(self, logit: float) -> float:
        """Compute P(in_scope | logit) after Platt scaling."""
        z = self.a * logit + self.b
        # Numerically stable sigmoid
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        else:
            ez = math.exp(z)
            return ez / (1.0 + ez)

    def is_oos(self, logit: float) -> bool:
        """True if probability of being in-scope falls below threshold."""
        return self.predict_proba(logit) < self.threshold

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> PlattScaler:
        return cls(
            a=float(data.get("a", 1.0)),
            b=float(data.get("b", 0.0)),
            threshold=float(data.get("threshold", 0.5)),
        )

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> PlattScaler:
        p = Path(path)
        if not p.exists():
            return cls()  # Default scaler
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls.from_dict(data)


def fit_platt_scaling(
    logits: list[float],
    labels: list[int],
    learning_rate: float = 0.05,
    max_iter: int = 100,
) -> PlattScaler:
    """Fit logistic sigmoid parameters (A, B) via gradient descent on binary cross-entropy.

    labels: 1 for in_scope, 0 for out_of_scope.
    """
    if not logits or len(logits) != len(labels):
        return PlattScaler()

    # Initialize A > 0 and B
    a = 1.0
    b = 0.0

    n = len(logits)
    for _ in range(max_iter):
        grad_a = 0.0
        grad_b = 0.0
        for x, y in zip(logits, labels):
            # Sigmoid(a*x + b)
            z = a * x + b
            prob = 1.0 / (1.0 + math.exp(-max(min(z, 30.0), -30.0)))
            err = prob - y
            grad_a += err * x
            grad_b += err

        a -= (learning_rate / n) * grad_a
        b -= (learning_rate / n) * grad_b

    return PlattScaler(a=a, b=b, threshold=0.5)
