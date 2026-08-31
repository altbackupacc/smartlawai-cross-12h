"""Automates PLAN.md's literal M0 done-when text: `smartlaw ingest <path>` then
`smartlaw ask --doc-id X --question "..."` returns a structured answer with a
complete trace."""
from __future__ import annotations

import json
import os
import re
import tempfile

from click.testing import CliRunner


def test_ingest_then_ask_end_to_end():
    os.environ["SMARTLAW_LOCAL_DIR"] = tempfile.mkdtemp()

    from smartlawai.cli import cli

    runner = CliRunner()
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("1. First clause. WHEREAS the parties agree. 2. Second clause.")
        path = f.name

    ingest_result = runner.invoke(cli, ["ingest", path, "--owner-id", "alice"])
    assert ingest_result.exit_code == 0, ingest_result.output
    m = re.search(r"doc_id: (\S+)", ingest_result.output)
    assert m, ingest_result.output
    doc_id = m.group(1)

    ask_result = runner.invoke(cli, ["ask", "--doc-id", doc_id, "--owner-id", "alice",
                                     "--question", "What does clause 1 say?", "--json"])
    assert ask_result.exit_code == 0, ask_result.output
    payload = json.loads(ask_result.output)
    assert payload["decision"] in ("ANSWER", "REFUSE")
    stage_names = {s["name"] for s in payload["trace"]["stages"]}
    assert stage_names == {"retrieve", "rerank", "generate", "verify", "decide"}
