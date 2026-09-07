"""The M6 annotation tool.

    streamlit run eval/annotate/app.py

One item per screen, keyboard-driven. PLAN.md M6 calls this the ~4h build that
everything else depends on, and lists what it must enforce. Those are not
preferences -- they are what makes the resulting labels usable in a paper -- so
each is implemented in code and named here with where it lives:

  Randomised order + blinded system labels
      store.randomised_order() (deterministic per rater, so a resumed session
      keeps its queue) and store.blind_item() (strips every system-identifying
      field, recursively, before anything is rendered). No model or system name
      ever reaches the screen.
  Correction mode vs cold mode
      store.assign_mode() puts ~10% of items in cold mode by a hash of the item
      id, so every rater sees the same item in the same mode and the cold subset
      is comparable across raters (RESEARCH.md §7.1.1's anchoring control).
  Logs rater id, timestamp, time-per-item
      store.record(); the timer starts when an item is rendered, not when the
      app loads, so the number means what §7.3 needs it to mean.
  Live Cohen's kappa / Krippendorff's alpha
      agreement.agreement() recomputed after every submission and shown in the
      sidebar as items come in, not at the end.
  Exports straight to eval/gold/*.jsonl
      store.export_gold_set() -- no spreadsheet-to-JSON step.

Annotator-pool-agnostic by construction (PLAN.md M6): rater ids are opaque
strings and nothing in the UI or the export schema knows whether a rater is a
student or a practising advocate. Recruiting advocates later is a config change.

This is the ONLY module in eval/annotate that touches Streamlit; every piece of
logic worth testing lives in store.py, agreement.py and guidelines.py, which are
importable and tested without a browser.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import streamlit as st
except ImportError:  # pragma: no cover - a clear message beats a stack trace
    raise SystemExit(
        "streamlit is required for the annotation app. Install the ui extra:\n"
        '    pip install -e ".[eval,ui,dev]"\n'
        "then run:  streamlit run eval/annotate/app.py")

from eval import config
from eval.annotate.guidelines import rubric_for
from eval.annotate.store import (
    AnnotationStore,
    assign_mode,
    blind_item,
    randomised_order,
)
from eval.gold_sets import GoldSetMissing, load_gold_set
from eval.schemas import GOLD_SET_NAMES, GOLD_SET_SPECS


# --------------------------------------------------------------------------- #
# Optional GCS credentials bridge (persistent annotation storage)
# --------------------------------------------------------------------------- #
def _bridge_gcp_credentials_from_secrets() -> None:
    """Let SMARTLAW_ANNOTATIONS_BACKEND=gcs (eval/annotate/cloud_storage.py)
    authenticate on Streamlit Community Cloud.

    Cloud has no persistent disk to point GOOGLE_APPLICATION_CREDENTIALS at
    ahead of time, so the standard pattern is: put the service-account JSON in
    Streamlit's own encrypted Secrets manager under the `gcp_service_account`
    key, and at startup write it to a temp file and point the env var there --
    the same Application Default Credentials path `adapters/gcloud.py`
    already relies on via a bare `storage.Client()`. The credential value
    itself is never in this file or in git (CLAUDE.md I7).

    Deliberately inert everywhere else: a local run with no secrets.toml, or
    a deployment not using GCS at all, hits the broad except below and simply
    does nothing -- this must never be the reason the app fails to start.
    Never overrides an already-set GOOGLE_APPLICATION_CREDENTIALS (e.g. a
    developer's own `gcloud auth application-default login`).
    """
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return
    try:
        service_account_info = dict(st.secrets["gcp_service_account"])
    except Exception:  # noqa: BLE001 -- deliberately broad, see docstring above
        # No secrets.toml, no such key, or a secrets backend error -- all mean
        # "GCS credentials aren't configured this way," not "crash the app."
        return

    credentials_path = Path(tempfile.gettempdir()) / "smartlawai_gcp_service_account.json"
    credentials_path.write_text(json.dumps(service_account_info), encoding="utf-8")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path)


# --------------------------------------------------------------------------- #
# Queue loading
# --------------------------------------------------------------------------- #
def load_queue(task: str, queue_path: Path | None) -> tuple[dict[str, Any], str]:
    """Load the items to be annotated.

    Two sources, in order:
      1. an explicit queue file (unlabelled items prepared for a session), or
      2. the existing gold set, for re-annotation / IAA duplication passes.

    Returns ({item_id: item}, source description). Never invents items.
    """
    spec = GOLD_SET_SPECS[task]
    if queue_path and queue_path.exists():
        items: dict[str, Any] = {}
        with queue_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                # A queue row may legitimately carry no label yet; fill the
                # schema's placeholder so it validates, then let the rater set it.
                row.setdefault("item_id", f"{task}-{len(items):05d}")
                for label_field in ("label", "in_force_label"):
                    if hasattr(spec.schema, "__dataclass_fields__") and \
                            label_field in spec.schema.__dataclass_fields__:
                        row.setdefault(label_field, spec.labels[0])
                item = spec.schema(**{k: v for k, v in row.items()
                                      if k in spec.schema.__dataclass_fields__})
                items[item.item_id] = item
        return items, f"queue file {queue_path}"

    try:
        gold = load_gold_set(task)
    except GoldSetMissing as e:
        return {}, str(e)
    return {i.item_id: i for i in gold}, f"existing gold set {task}.jsonl"


def model_proposal(task: str, item: Any) -> Any:
    """The pre-filled label shown in correction mode.

    RESEARCH.md §7.1.1's model-assisted pre-labelling assumes a frontier model
    proposed the label. This app does not call one: M6 has no model budget and
    calling one from inside the annotation loop would put an unlogged,
    unversioned dependency in the middle of the evidence chain. Instead, the
    proposal is whatever label the queue row already carries -- which is exactly
    where a batch pre-labelling job writes it.

    Returns None when the row carries no proposal, and the app then shows the
    item cold and records `mode="cold"` honestly rather than pretending an
    assisted pass happened.
    """
    if task == "retrieval":
        return dict(getattr(item, "judgments", {})) or None
    for attr in ("label", "in_force_label"):
        value = getattr(item, attr, None)
        if value:
            return value
    return None


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_item(task: str, item: Any) -> None:
    """Show the item, blinded. Never shows a system/model identifier."""
    rubric = rubric_for(task)
    blinded = blind_item(item.to_row())
    for field_name in rubric.fields_shown:
        value = blinded.get(field_name)
        if not value:
            continue
        st.markdown(f"**{field_name.replace('_', ' ').title()}**")
        if field_name in ("query", "question", "claim_text", "citation", "as_of_date"):
            st.info(value)
        else:
            st.text_area(field_name, value, height=220, disabled=True,
                         label_visibility="collapsed")

    if task == "retrieval":
        st.markdown("**Pooled candidates** (judge each; unshown chunks are "
                    "unjudged, not irrelevant)")


def render_guidelines(task: str) -> None:
    rubric = rubric_for(task)
    with st.expander(f"Guidelines & rating scale — {rubric.task} ({rubric.version})",
                     expanded=False):
        st.markdown(f"**{rubric.question}**")
        st.write(rubric.instructions)
        for option in rubric.options:
            st.markdown(f"- **[{option.hotkey}] {option.title}** (`{option.value}`) "
                        f"— {option.description}")
        if rubric.scale_note:
            st.caption(rubric.scale_note)
        if rubric.pitfalls:
            st.markdown("**Common mistakes**")
            for pitfall in rubric.pitfalls:
                st.markdown(f"- {pitfall}")


def render_agreement(store: AnnotationStore) -> None:
    """Live agreement + throughput, recomputed as items come in (PLAN.md M6)."""
    result = store.agreement()
    st.sidebar.markdown("### Live agreement")
    if result.value is None:
        st.sidebar.caption(f"{result.statistic}: unavailable — {result.reason}")
    else:
        st.sidebar.metric(result.statistic, f"{result.value:.3f}",
                          help=f"{result.n_items} items, {result.n_raters} raters")

    throughput = store.throughput()
    st.sidebar.markdown("### Throughput")
    if throughput.items_per_hour is None:
        st.sidebar.caption("no timed annotations yet")
    else:
        st.sidebar.metric("items / hour", f"{throughput.items_per_hour:.0f}",
                          help=f"median {throughput.median_seconds:.1f}s per item — "
                               "this number sizes every remaining gold set "
                               "(RESEARCH.md §7.3)")
        for mode, rate in throughput.by_mode.items():
            if rate is not None:
                st.sidebar.caption(f"{mode}: {rate:.0f}/h")
    st.sidebar.caption(f"{throughput.n_annotations} annotations submitted")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    _bridge_gcp_credentials_from_secrets()
    st.set_page_config(page_title="SmartLawAI — M6 annotation", layout="wide")
    st.title("SmartLawAI — annotation")

    st.sidebar.header("Session")
    task = st.sidebar.selectbox("Gold set", GOLD_SET_NAMES)
    rater_id = st.sidebar.text_input(
        "Rater ID", value="",
        help="Any opaque identifier. Deliberately not tied to a rater pool — "
             "students and advocates are the same field here.")
    queue_file = st.sidebar.text_input("Queue file (optional)", value="")
    seed = st.sidebar.number_input("Order seed", value=config.DEFAULT_SEED, step=1)

    spec = GOLD_SET_SPECS[task]
    st.sidebar.caption(f"{spec.purpose}")
    st.sidebar.caption(f"IAA duplication target: {spec.iaa_duplication_rate:.0%} · "
                       f"target n: {spec.target_n}")
    if spec.never_train_on:
        st.sidebar.warning("Held-out set — never used to train the judge "
                           "(RESEARCH.md §2.5).")

    if not rater_id.strip():
        st.info("Enter a Rater ID in the sidebar to begin.")
        return

    items, source = load_queue(task, Path(queue_file) if queue_file.strip() else None)
    st.caption(f"Source: {source}")
    if not items:
        st.warning("No items to annotate. Provide a queue file, or collect the "
                   "gold set's items first.")
        return

    store = AnnotationStore(task)
    rubric = rubric_for(task)
    render_guidelines(task)

    done = store.rated_item_ids(rater_id)
    order = [i for i in randomised_order(items, rater_id, int(seed)) if i not in done]

    st.progress(len(done) / len(items) if items else 0.0,
                text=f"{len(done)} / {len(items)} items done by this rater")
    if not order:
        st.success("This rater has annotated every item in the queue.")
        _render_export(store, items)
        render_agreement(store)
        return

    item_id = order[0]
    item = items[item_id]
    mode = assign_mode(item_id, config.COLD_SUBSET_FRACTION, int(seed))
    proposal = model_proposal(task, item) if mode == "correction" else None
    if proposal is None:
        mode = "cold"  # no proposal exists -> record it honestly as a cold label

    # Timer starts when THIS item is first rendered, not at app load.
    timer_key = f"t0:{task}:{rater_id}:{item_id}"
    if timer_key not in st.session_state:
        st.session_state[timer_key] = time.perf_counter()

    st.subheader(rubric.question)
    st.caption(f"Item {item_id} · mode: {mode}"
               + ("  (cold subset — anchoring control)" if mode == "cold" else ""))
    render_item(task, item)

    with st.form(key=f"form:{item_id}"):
        if task == "retrieval":
            label: Any = {}
            existing = getattr(item, "judgments", {}) or {}
            candidates = list(existing) or list(getattr(item, "pooled_from", []))
            for chunk_id in candidates:
                default = str(existing.get(chunk_id, 0)) if mode == "correction" else "0"
                label[chunk_id] = int(st.radio(
                    f"`{chunk_id}`", [o.value for o in rubric.options],
                    index=[o.value for o in rubric.options].index(default),
                    horizontal=True, key=f"r:{item_id}:{chunk_id}",
                    format_func=lambda v: f"{v} — {rubric.option(v).title}"))
        else:
            values = list(rubric.values())
            default_index = (values.index(proposal)
                             if mode == "correction" and proposal in values else 0)
            label = st.radio(
                "Label", values, index=default_index,
                format_func=lambda v: f"[{rubric.option(v).hotkey}] "
                                      f"{rubric.option(v).title}",
                key=f"label:{item_id}")

        confidence = st.slider("Your confidence", 1, 5, 3, key=f"conf:{item_id}")
        comment = st.text_input("Comment (optional)", key=f"c:{item_id}")
        submitted = st.form_submit_button("Submit and next  (Ctrl+Enter)",
                                          type="primary")

    if submitted:
        elapsed = time.perf_counter() - st.session_state.pop(timer_key,
                                                             time.perf_counter())
        store.record(item_id=item_id, rater_id=rater_id, label=label, mode=mode,
                     seconds_on_item=elapsed, proposed_label=proposal,
                     comment=comment, confidence=int(confidence),
                     rubric_version=rubric.version, session_id=f"{rater_id}:{seed}")
        st.rerun()

    _render_export(store, items)
    render_agreement(store)


def _render_export(store: AnnotationStore, items: dict[str, Any]) -> None:
    st.sidebar.markdown("### Export")
    st.sidebar.caption("Writes eval/gold/<task>.jsonl directly — no spreadsheet step.")
    min_annotations = st.sidebar.number_input("Min annotations per item", 1, 5, 1)
    if st.sidebar.button("Export to gold set"):
        exported = store.export_gold_set(items, min_annotations=int(min_annotations))
        st.sidebar.success(f"Exported {len(exported)} items to "
                           f"eval/gold/{store.task}.jsonl")


if __name__ == "__main__":
    main()
