"""Resolve India Code act_ids for the seed statutes.

India Code runs DSpace 7 with a REST API. Act items carry
`dc.identifier.collection = 'ACT'`; their sections are separate items with
`collection = 'SECTION'` and the same `act_id`. Free-text search matches the
full text of every act, so it is far too noisy -- we page through ACT-collection
hits and keep only exact title matches on central (`AC_CEN_`) acts.
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://indiacode.gov.in/server/api/discover/search/objects"
OUT = Path(__file__).resolve().parents[1] / "out" / "act_ids.json"

# our registry id -> exact India Code title
WANT = {
    "ipc-1860": "The Indian Penal Code, 1860",
    "crpc-1973": "The Code of Criminal Procedure, 1973",
    "iea-1872": "The Indian Evidence Act, 1872",
    "bns-2023": "The Bharatiya Nyaya Sanhita, 2023",
    "bnss-2023": "The Bharatiya Nagarik Suraksha Sanhita, 2023",
    "bsa-2023": "The Bharatiya Sakshya Adhiniyam, 2023",
    "contract-1872": "The Indian Contract Act, 1872",
    "arbitration-1996": "The Arbitration and Conciliation Act, 1996",
    "consumer-2019": "The Consumer Protection Act, 2019",
    "cpc-1908": "The Code of Civil Procedure, 1908",
    "ni-1881": "The Negotiable Instruments Act, 1881",
    "it-2000": "The Information Technology Act, 2000",
    "stamp-1899": "The Indian Stamp Act, 1899",
}


def get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read().decode("utf-8"))


def md1(md: dict, key: str):
    v = md.get(key)
    return v[0].get("value") if v else None


def search_acts(title: str, pages: int = 4, size: int = 100):
    """Yield central ACT items matching the title query."""
    q = urllib.parse.quote('"{}"'.format(title.replace("The ", "", 1)))
    for page in range(pages):
        url = ("%s?query=%s&f.identifier_collection=ACT,equals&page=%d&size=%d"
               % (BASE, q, page, size))
        try:
            d = get(url)
        except Exception as e:
            print(f"   ! {e}")
            return
        sr = d.get("_embedded", {}).get("searchResult", {})
        objs = sr.get("_embedded", {}).get("objects", [])
        if not objs:
            return
        for o in objs:
            it = o.get("_embedded", {}).get("indexableObject", {})
            md = it.get("metadata", {})
            aid = md1(md, "dc.identifier.act_id") or ""
            if aid.startswith("AC_CEN"):
                yield it, md, aid
        if (page + 1) * size >= sr.get("page", {}).get("totalElements", 0):
            return


def norm(s: str) -> str:
    return " ".join((s or "").lower().replace(",", " ").split())


def main() -> int:
    found = {}
    for reg_id, title in WANT.items():
        print("== %-18s %s" % (reg_id, title))
        best = None
        for it, md, aid in search_acts(title):
            if norm(it.get("name")) == norm(title):
                best = {
                    "registry_id": reg_id,
                    "act_id": aid,
                    "uuid": it.get("uuid"),
                    "name": it.get("name"),
                    "act_number": md1(md, "dc.identifier.act_number"),
                    "act_year": md1(md, "dc.date.act_year"),
                    "repealed": md1(md, "dc.identifier.repealed"),
                    "enact_date": md1(md, "dc.date.enact_date"),
                    "enforcement_date": md1(md, "dc.date.enforcement_date"),
                    "long_title": md1(md, "dc.title.long_title"),
                    "ministry": md1(md, "dc.identifier.ministry_name"),
                }
                break
        if best:
            found[reg_id] = best
            print("   FOUND {}  no={} yr={} repealed={} enf={}".format(best["act_id"], best["act_number"], best["act_year"],
                     best["repealed"], best["enforcement_date"]))
        else:
            print("   NOT FOUND")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(found, indent=2), encoding="utf-8")
    print("\n%d/%d resolved -> %s" % (len(found), len(WANT), OUT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
