"""The human-annotation half of M6 (PLAN.md M6: "build this first -- everything
else depends on it").

    guidelines.py  rubrics and rating scales as versioned data
    store.py       append-only annotation log + export to eval/gold/*.jsonl
    agreement.py   Cohen's kappa / Krippendorff's alpha, live and standalone
    pooling.py     TREC pooling for the retrieval set's candidate lists
    app.py         the Streamlit tool raters actually use

Nothing here imports `smartlawai` except app.py's optional smoke-input path.
"""
from __future__ import annotations

__all__ = ["agreement", "guidelines", "pooling", "store"]
