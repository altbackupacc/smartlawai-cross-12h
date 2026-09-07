# India Code — section-list provenance

Source: **https://indiacode.gov.in** (Ministry of Law and Justice, Legislative
Department). Retrieved 2026-09-05.

> **The site has migrated.** `indiacode.nic.in` — the URL cited in `PLAN.md`
> and `M4_ONBOARDING.md` — now serves only a redirect notice. The live host is
> `indiacode.gov.in`. Seed rows written before this was discovered carry the
> old URL and should be updated.

## How it is accessed

India Code runs **DSpace 7** and exposes a REST API. No key, no auth, no
scraping of rendered HTML:

```
GET /server/api/discover/search/objects
    ?sort=dc.identifier.order_number,ASC
    &page=0&size=100
    &f.identifier_collection=SECTION,equals
    &f.act_id=<ACT_ID>,equals
```

Acts are items with `dc.identifier.collection = 'ACT'`; each section is its own
item with `collection = 'SECTION'` and the parent's `act_id`. Central acts have
an `AC_CEN_` prefixed `act_id`. Corpus size: 11,274 ACT items and 74,780
SECTION items.

Per-section metadata includes `dc.identifier.section_number`, the heading (as
the item `name`), `dc.identifier.order_number`, and `dc.identifier.repealed`.
Act items additionally carry `dc.date.enforcement_date`, `dc.date.enact_date`,
`dc.title.long_title` and `dc.identifier.repealed`.

Free-text search matches the full text of every act and is far too noisy to
identify an act by name; `scripts/find_act_ids.py` pages the ACT collection and
keeps only exact title matches on `AC_CEN_` items. Resolved ids are committed
in `M4/out/act_ids.json` so the lookup is done once, not per run.

## Ingested — 2,088 sections across 10 statutes

| Statute | Sections | Enforcement date (official) |
|---|---|---|
| bns-2023 | 358 | 2024-07-01 |
| bnss-2023 | 531 | 2024-07-01 |
| bsa-2023 | 170 | 2024-07-01 |
| contract-1872 | 268 | 1872-09-01 |
| arbitration-1996 | 106 | 1996-08-22 |
| consumer-2019 | 107 | *(staged commencement — see below)* |
| cpc-1908 | 171 | 1909-01-01 |
| ni-1881 | 155 | 1882-03-01 |
| it-2000 | 125 | 2000-10-17 |
| stamp-1899 | 97 | 1899-07-01 |

`enforcement_date` independently **confirms seven of the hand-seeded
commencement dates** in `scripts/seed_statutes.py`.

**One discrepancy to resolve:** the Consumer Protection Act 2019 was seeded as
`2020-07-20`, but India Code records a staged commencement — a block of
provisions on **24 July 2020** by notification S.O. 2421(E). Neither date is
wrong for the whole Act; the Act commenced in stages. Flag for the verification
pass.

## The gap that matters — IPC, CrPC and IEA are NOT in India Code

**India Code does not index the repealed Indian Penal Code 1860, Code of
Criminal Procedure 1973, or Indian Evidence Act 1872** — not as ACT items, not
as SECTION items. Confirmed three ways: exact-title search over the ACT
collection, `act_number`+`act_year` query, and a SECTION search for a
distinctive IPC heading ("Punishment for murder") which returns only BNS 104.
There is no `REPEALED_ACT` collection; the facet list holds no such value.

These are exactly the three statutes the paper's natural experiment depends on.

Consequence for the registry as it stands:

| Statute | Section numbers | Headings |
|---|---|---|
| IPC / CrPC / IEA | from MHA correspondence tables | **missing** |
| all other 10 | India Code (authoritative) | India Code |

The headings *are* recoverable — the correspondence tables' old-code column
carries them ("174A. Non-appearance in response to a proclamation…"). That is
the next ingestion step, and it needs no new source.

## Cross-validation — the payoff of holding two sources

For BNS/BNSS/BSA the correspondence tables and India Code are independent.
`scripts/validate_sections.py` compares them:

```
bns-2023    india code 358   registry 358   spurious 0   missing 0
bnss-2023   india code 531   registry 531   spurious 0   missing 0
bsa-2023    india code 170   registry 170   spurious 0   missing 0
```

**1,059 of 1,059 sections agree.** The check earned its keep immediately: on
first run it flagged one spurious BNS section, `2023`, traced to this row —

```
L: 209. Non-appearance ... under section 84 of Bharatiya Nagarik Suraksha Sanhita, 2023
R: 174A. Non-appearance ... under section 82 of Act 2 of 1974.
```

— where the parser read the *years* in an inline cross-reference as section
numbers and formed a cross-product of fabricated mappings. Fixed in
`ingest/mha_tables.py` by rejecting year-range numbers and numbers preceded by
a cross-reference marker ("section N", "Act N"). That single fix removed 36
false supersession rows.

The same machinery will validate IPC/CrPC/IEA the moment an authoritative
section list for them is sourced. Until then their numbers rest on a single
source and must be treated as such.

## Still missing

**The Constitution of India is not ingested.** It is not an "Act" in India
Code's model and does not appear in the ACT collection. It is the most-cited
authority in the corpus (12,394 citations), so it is the largest remaining
coverage hole — see the `xfail` in `M4/tests/test_registry.py`. Likely source:
legislative.gov.in.
