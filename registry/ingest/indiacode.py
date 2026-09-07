"""Ingest section lists from India Code (indiacode.gov.in).

India Code runs DSpace 7 and exposes a REST API. Acts are items with
`dc.identifier.collection = 'ACT'`; each section is its own item with
`collection = 'SECTION'` and the parent's `act_id`. This gives an authoritative,
uniform section list per statute -- one ingestion path for every act, rather
than bespoke parsing of each act's PDF layout.

**Known limitation, and it matters.** India Code does not index the repealed
IPC 1860, CrPC 1973 or IEA 1872 -- neither as ACT items nor as SECTION items.
Those three are precisely the statutes the paper's natural experiment depends
on. Their section numbers come from the MHA correspondence tables instead (see
`ingest/mha_tables.py`); their headings are recoverable from the same tables'
old-code column. Recorded in `data/mha/PROVENANCE.md`.

Note also that the site migrated from `indiacode.nic.in` to `indiacode.gov.in`;
the old domain now serves only a redirect notice.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from html import unescape

BASE = "https://indiacode.gov.in/server/api"
SEARCH = BASE + "/discover/search/objects"
UA = {"Accept": "application/json", "User-Agent": "smartlawai-m4-registry/0.1"}


def _get(url: str, retries: int = 3) -> dict:
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} tries: {url} ({last})")


def _md1(md: dict, key: str):
    v = md.get(key)
    return v[0].get("value") if v else None


_BLOCK = re.compile(r"<\s*(br|/p|/div|hr)\s*/?\s*>", re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")


def html_to_text(html: str | None) -> str | None:
    """India Code stores section text as HTML in `section_page_note`.

    Block-level tags become newlines so sub-section structure survives; the rest
    are stripped and entities unescaped. Nothing is reflowed -- the registry
    stores the provision as published, not a paraphrase.
    """
    if not html:
        return None
    s = _BLOCK.sub("\n", html)
    s = _TAG.sub("", s)
    s = unescape(s)
    s = s.replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s*\n\s*\n+", "\n\n", s)
    s = "\n".join(ln.strip() for ln in s.splitlines())
    s = s.strip()
    return s or None


def fetch_sections(act_id: str, page_size: int = 100) -> list[dict]:
    """Every SECTION item belonging to `act_id`, in statutory order."""
    out: list[dict] = []
    page = 0
    total = None
    while True:
        url = (f"{SEARCH}?sort=dc.identifier.order_number,ASC&page={page}&size={page_size}"
               f"&f.identifier_collection=SECTION,equals&f.act_id={urllib.parse.quote(act_id)},equals")
        d = _get(url)
        sr = d.get("_embedded", {}).get("searchResult", {})
        if total is None:
            total = sr.get("page", {}).get("totalElements", 0)
        objs = sr.get("_embedded", {}).get("objects", [])
        if not objs:
            break
        for o in objs:
            it = o.get("_embedded", {}).get("indexableObject", {})
            md = it.get("metadata", {})
            num = _md1(md, "dc.identifier.section_number")
            if not num:
                continue
            out.append({
                "number": str(num).strip(),
                "heading": (it.get("name") or "").strip() or None,
                "text": html_to_text(_md1(md, "dc.identifier.section_page_note")),
                "footnote": html_to_text(_md1(md, "dc.identifier.section_footnote")),
                "order": _md1(md, "dc.identifier.order_number"),
                "repealed": str(_md1(md, "dc.identifier.repealed")).lower() == "true",
                "uuid": it.get("uuid"),
            })
        page += 1
        if page * page_size >= (total or 0):
            break
    return out
