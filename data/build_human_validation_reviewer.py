"""Generates a self-contained, offline HTML review page for one rater's
200-item human validation CSV (data/sample_human_validation.py's output) --
the raw CSV is unreadable in a spreadsheet (two long free-text columns
overflow with no wrapping across 200 rows). This produces one clean,
one-item-at-a-time review page instead: no server, no new Python
dependency, no data leaves the machine -- just double-click the generated
HTML file to open it in a normal browser.

Exports back to the exact same CSV schema data/score_human_validation.py
already expects, so nothing downstream needs to change.

Usage:
    python data/build_human_validation_reviewer.py \
        --input data/human_validation/rater1_sample.csv \
        --output data/human_validation/rater1_review.html
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

CSV_COLUMNS = ["item_id", "doc_id", "section", "source_headnote",
               "generated_text", "rater_verdict", "rater_notes"]


def load_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_html(rows: list[dict], storage_key: str) -> str:
    data_json = json.dumps(rows)
    return HTML_TEMPLATE.replace("__DATA_JSON__", data_json).replace(
        "__STORAGE_KEY__", json.dumps(storage_key))


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Human Validation Review</title>
<style>
  :root { color-scheme: light; }
  body { font-family: Georgia, 'Times New Roman', serif; max-width: 900px;
         margin: 0 auto; padding: 24px 32px 80px; background: #faf9f6; color: #1a1a1a; }
  h1 { font-size: 20px; font-family: Arial, sans-serif; }
  .progress-bar { height: 8px; background: #ddd; border-radius: 4px; overflow: hidden; margin: 8px 0 20px; }
  .progress-fill { height: 100%; background: #2c6e49; transition: width 0.2s; }
  .meta { font-family: Arial, sans-serif; font-size: 13px; color: #666; margin-bottom: 16px; }
  .block { margin-bottom: 20px; }
  .block-label { font-family: Arial, sans-serif; font-size: 12px; font-weight: bold;
                 text-transform: uppercase; letter-spacing: 0.05em; color: #555; margin-bottom: 6px; }
  .block-text { background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 16px;
                font-size: 17px; line-height: 1.6; white-space: pre-wrap; }
  .source .block-text { border-left: 4px solid #8a8a8a; }
  .generated .block-text { border-left: 4px solid #2c6e49; }
  .verdict-buttons { display: flex; gap: 12px; margin: 20px 0; }
  .verdict-buttons button { flex: 1; padding: 16px; font-size: 16px; font-family: Arial, sans-serif;
                             font-weight: bold; border: 2px solid #333; border-radius: 8px;
                             background: #fff; cursor: pointer; }
  .verdict-buttons button:hover { background: #eee; }
  .verdict-buttons button.active-accept { background: #2c6e49; color: #fff; border-color: #2c6e49; }
  .verdict-buttons button.active-revise { background: #c9891a; color: #fff; border-color: #c9891a; }
  .verdict-buttons button.active-reject { background: #a4302f; color: #fff; border-color: #a4302f; }
  textarea { width: 100%; box-sizing: border-box; font-family: Arial, sans-serif; font-size: 14px;
             padding: 10px; border: 1px solid #ccc; border-radius: 6px; min-height: 50px; }
  .nav { display: flex; justify-content: space-between; align-items: center; margin-top: 24px;
         font-family: Arial, sans-serif; }
  .nav button { padding: 10px 18px; font-size: 14px; border: 1px solid #333; border-radius: 6px;
                background: #fff; cursor: pointer; }
  .nav button:disabled { opacity: 0.4; cursor: default; }
  .hint { font-family: Arial, sans-serif; font-size: 12px; color: #888; text-align: center; margin-top: 6px; }
  .footer { position: fixed; bottom: 0; left: 0; right: 0; background: #fff; border-top: 1px solid #ddd;
            padding: 10px 32px; font-family: Arial, sans-serif; font-size: 13px; display: flex;
            justify-content: space-between; align-items: center; }
  .footer button { padding: 8px 16px; font-size: 14px; border-radius: 6px; border: 1px solid #333;
                    background: #1a1a1a; color: #fff; cursor: pointer; }
</style>
</head>
<body>
  <h1>Human Validation Review</h1>
  <div class="meta" id="itemMeta"></div>
  <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>

  <div class="block source">
    <div class="block-label">Source</div>
    <div class="block-text" id="sourceText"></div>
  </div>
  <div class="block generated">
    <div class="block-label">Generated Summary</div>
    <div class="block-text" id="generatedText"></div>
  </div>

  <div class="verdict-buttons">
    <button id="btnAccept" onclick="setVerdict('accept')">Accept (A)</button>
    <button id="btnRevise" onclick="setVerdict('revise')">Revise (S)</button>
    <button id="btnReject" onclick="setVerdict('reject')">Reject (D)</button>
  </div>
  <textarea id="notes" placeholder="Optional notes..." oninput="saveNotes()"></textarea>
  <div class="hint">Keyboard: A = Accept, S = Revise, D = Reject, &larr;/&rarr; = Prev/Next</div>

  <div class="nav">
    <button id="btnPrev" onclick="go(-1)">&larr; Previous</button>
    <span id="navPos"></span>
    <button id="btnNext" onclick="go(1)">Next &rarr;</button>
  </div>

  <div class="footer">
    <span id="answeredCount"></span>
    <button onclick="exportCsv()">Export CSV</button>
  </div>

<script>
const DATA = __DATA_JSON__;
const STORAGE_KEY = __STORAGE_KEY__;
let answers = {};
try {
  answers = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
} catch (e) { answers = {}; }

// Seed from any verdicts already present in the input CSV (e.g. resuming
// a partially-completed export), without overwriting anything already
// saved in localStorage.
DATA.forEach(row => {
  if (!(row.item_id in answers) && (row.rater_verdict || row.rater_notes)) {
    answers[row.item_id] = { verdict: (row.rater_verdict || "").toLowerCase(), notes: row.rater_notes || "" };
  }
});

let idx = 0;

function currentAnswer() {
  const id = DATA[idx].item_id;
  return answers[id] || { verdict: "", notes: "" };
}

function render() {
  const row = DATA[idx];
  document.getElementById("itemMeta").textContent =
    `Item ${idx + 1} of ${DATA.length}  —  doc_id ${row.doc_id}, section: ${row.section}`;
  document.getElementById("sourceText").textContent = row.source_headnote;
  document.getElementById("generatedText").textContent = row.generated_text;
  const ans = currentAnswer();
  document.getElementById("notes").value = ans.notes;
  ["accept", "revise", "reject"].forEach(v => {
    document.getElementById("btn" + v.charAt(0).toUpperCase() + v.slice(1))
      .className = (ans.verdict === v) ? "active-" + v : "";
  });
  document.getElementById("progressFill").style.width =
    (Object.keys(answers).length / DATA.length * 100) + "%";
  document.getElementById("navPos").textContent = `${idx + 1} / ${DATA.length}`;
  document.getElementById("btnPrev").disabled = idx === 0;
  document.getElementById("btnNext").disabled = idx === DATA.length - 1;
  const answeredCount = DATA.filter(r => answers[r.item_id] && answers[r.item_id].verdict).length;
  document.getElementById("answeredCount").textContent = `Answered: ${answeredCount} / ${DATA.length}`;
}

function save() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(answers));
}

function setVerdict(v) {
  const id = DATA[idx].item_id;
  const existing = answers[id] || { verdict: "", notes: "" };
  answers[id] = { verdict: v, notes: existing.notes };
  save();
  render();
  if (idx < DATA.length - 1) {
    setTimeout(() => { go(1); }, 150);
  }
}

function saveNotes() {
  const id = DATA[idx].item_id;
  const existing = answers[id] || { verdict: "", notes: "" };
  existing.notes = document.getElementById("notes").value;
  answers[id] = existing;
  save();
}

function go(delta) {
  const newIdx = idx + delta;
  if (newIdx < 0 || newIdx >= DATA.length) return;
  idx = newIdx;
  render();
}

document.addEventListener("keydown", (e) => {
  if (document.activeElement.tagName === "TEXTAREA") return;
  if (e.key === "a" || e.key === "A") setVerdict("accept");
  else if (e.key === "s" || e.key === "S") setVerdict("revise");
  else if (e.key === "d" || e.key === "D") setVerdict("reject");
  else if (e.key === "ArrowLeft") go(-1);
  else if (e.key === "ArrowRight") go(1);
});

function csvEscape(field) {
  return '"' + String(field == null ? "" : field).replace(/"/g, '""') + '"';
}

function exportCsv() {
  const answeredCount = DATA.filter(r => answers[r.item_id] && answers[r.item_id].verdict).length;
  if (answeredCount < DATA.length) {
    if (!confirm(`Only ${answeredCount}/${DATA.length} items answered. Export anyway?`)) return;
  }
  const header = ["item_id", "doc_id", "section", "source_headnote",
                   "generated_text", "rater_verdict", "rater_notes"];
  const lines = [header.map(csvEscape).join(",")];
  DATA.forEach(row => {
    const ans = answers[row.item_id] || { verdict: "", notes: "" };
    const verdict = ans.verdict ? ans.verdict.charAt(0).toUpperCase() + ans.verdict.slice(1) : "";
    lines.push([row.item_id, row.doc_id, row.section, row.source_headnote,
                row.generated_text, verdict, ans.notes || ""].map(csvEscape).join(","));
  });
  const blob = new Blob([lines.join("\r\n")], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "rater_sample_completed.csv";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

render();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = load_rows(args.input)
    if not rows:
        raise SystemExit(f"{args.input} has no rows.")
    missing_cols = set(CSV_COLUMNS) - set(rows[0].keys())
    if missing_cols:
        raise SystemExit(f"{args.input} is missing expected column(s): {missing_cols}")

    storage_key = f"human_validation::{args.input.stem}"
    html = build_html(rows, storage_key)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"Wrote {args.output} ({len(rows)} items).")
    print("Open it directly in a browser (double-click the file) to start reviewing.")
    print("Progress autosaves to this browser's local storage as you go.")
    print(f"When done, click 'Export CSV' and save the download over "
          f"{args.input} (or point score_human_validation.py at the download directly).")


if __name__ == "__main__":
    main()
