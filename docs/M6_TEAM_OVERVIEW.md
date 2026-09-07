
---

## 1. What is M6?

SmartLawAI is an AI system that reads legal documents (contracts, court
judgments) and answers questions about them, with citations. M6 is not part of
that system — **M6 is the scoring system that tells us whether SmartLawAI is
actually any good.**

Here's the problem M6 solves: if we build the AI pipeline and just eyeball a
few answers to see if they "look right," we have no real evidence for a paper,
no way to know if a change made things better or worse, and no way to catch a
model that sounds confident while being wrong. "Sounds confident while being
wrong" — usually called **hallucination** in AI — is the single biggest risk
in a legal product, because a wrong-sounding-right answer about the law can
genuinely hurt someone.

So we cannot just build the AI and ship it. We need:
- **Test questions with known correct answers** (called **gold test sets** —
  "gold" just means "we trust this answer, usually because a human checked it").
- **A way to automatically run those questions through the system and score
  the answers.**
- **A way for humans to double-check the automatic scores**, because some
  judgments (is this legal claim actually true?) are hard for software alone
  to get right.
- **Honest reporting** — if we can't measure something yet, the system should
  say "we don't know" instead of making up a number.

That's the whole job of M6.

---

## 2. What We Built

### Gold test sets
**What it is:** Five separate files of test questions/claims, each with a
human-checked correct answer.
**Why we need it:** Without a known-correct answer to compare against, there's
nothing to score.
**What we built:** Five files (`retrieval`, `oos`, `groundedness`, `repealed`,
`entailment` — explained one by one in section 4), 156 test items total. These
are currently small **seed** sets built by one person, not the full 300–400
item, multiple-rater study the research plan calls for. That's honestly stated
everywhere they're used, not hidden.

### Evaluation runner
**What it is:** A script that automatically takes every test question, runs it
through the real SmartLawAI system, and records what happened.
**Why we need it:** Doing this by hand for hundreds of questions isn't
realistic, and it wouldn't be repeatable.
**What we built:** `eval/run_eval.py` — runs all five gold sets against the
real (currently stub) pipeline, computes every metric that can honestly be
computed, and writes a report. Confirmed working: a real run processes all 42
directly-runnable items with **zero failures**.

### Metrics system
**What it is:** The actual scoring formulas — how "good" gets turned into a
number.
**Why we need it:** "It seems fine" isn't a score. We need the same, standard
way of measuring things every time.
**What we built:** `eval/metrics.py` — covers retrieval quality, text
similarity (ROUGE), whether an answer is grounded in real evidence, refusal
behavior, citation checking, and more (full list in section 7). Uses
well-known published scoring methods (like ROUGE) rather than inventing our
own where a standard one already exists.

### Statistical analysis
**What it is:** Tools that tell us whether a difference between two versions
of the system is a *real* improvement or just random noise.
**Why we need it:** If one model scores 71% and another scores 73%, is that
meaningful, or could it happen by chance? Without statistics, we can't tell.
**What we built:** `eval/stats.py` — a proper resampling technique (bootstrap,
10,000 resamples) for comparing two systems fairly, a calibration check (does
the system's confidence match its actual accuracy?), and correction for
running many comparisons at once.

### Extractiveness analysis
**What it is:** A check for whether the AI is just copy-pasting the source
text instead of actually reasoning about it.
**Why we need it:** A model that just repeats the input word-for-word can look
"very faithful to the source" while being a useless summarizer. This catches
that.
**What we built:** `eval/extractiveness.py` — measures how much of a generated
answer is copied verbatim vs. genuinely composed.

### Human annotation system
**What it is:** The behind-the-scenes storage and rules for how a person's
judgment gets recorded, kept honest, and turned into test data.
**Why we need it:** Some things (is this legal claim actually supported by
this passage?) need a person to judge, not just a formula. We need to record
who judged what, when, and how confidently — and make sure a rater who knows
which answer is "ours" isn't unconsciously biased toward liking it.
**What we built:** `eval/annotate/store.py` — records every rating with a
timestamp and time taken, hides which system produced an answer before
showing it to the rater (called "blinding"), and randomizes the order items
are shown in.

### Streamlit annotation app
**What it is:** The actual clickable web page a human rater uses.
**Why we need it:** A researcher can't paste JSON files by hand — they need a
simple screen: read this, pick a rating, click next.
**What we built:** `eval/annotate/app.py` — one item per screen, keyboard
shortcuts, a live agreement score updating in the sidebar as ratings come in.
Verified to start and run correctly.

### Inter-rater agreement
**What it is:** A statistic that measures whether two different human raters
agree with each other.
**Why we need it:** If two people rate the same item completely differently,
that tells us the question is ambiguous — the rating itself isn't trustworthy
yet. This is standard practice in any research that uses human judgments.
**What we built:** `eval/annotate/agreement.py` — computes Cohen's kappa (2
raters) and Krippendorff's alpha (3+ raters), the two standard statistics for
this.

### TREC pooling
**What it is:** A technique for judging retrieval quality without needing to
manually check every possible document against every possible question.
**Why we need it:** If we have hundreds of documents and dozens of questions,
checking every combination by hand is impossible. Pooling picks the ~15 most
promising candidates per question instead of thousands.
**What we built:** `eval/annotate/pooling.py` — combines results from
multiple retrieval methods into one manageable list for a human to check.

### Perturbation / robustness testing
**What it is:** A tool that deliberately creates *wrong* versions of correct
legal claims (e.g., swapping a section number, negating "shall" to "shall
not") to train and test a future AI judge on catching these specific mistakes.
**Why we need it:** A generic error-detector might miss legal-specific
mistakes like citing the wrong law section. This builds targeted training
data for exactly those mistakes.
**What we built:** `data/perturbations/generate_perturbations.py` — creates
seven distinct types of realistic legal errors from real case documents.

### Reporting system
**What it is:** The part that turns all the computed numbers into something a
person can actually read.
**Why we need it:** A pile of numbers in a database isn't useful in a
meeting or a paper.
**What we built:** `eval/report.py` — produces both a machine-readable file
(JSON) and a plain-English Markdown report with tables, a clear "Not
measured" section, and failure listings.

### Automated tests
**What it is:** Code that checks the evaluation code itself is correct.
**Why we need it:** If the scoring system has a bug, every result downstream
is wrong — and we might not notice for months.
**What we built:** Over 300 automated checks across the M6 code, verified
passing (see section 9 for exact numbers).

### CI regression workflow
**What it is:** An automatic check that runs every time someone proposes a
code change, comparing new results against the last known-good results.
**Why we need it:** To catch it immediately if someone accidentally breaks
the scoring system, instead of finding out weeks later.
**What we built:** `.github/workflows/eval-regression.yml` — runs the
evaluation automatically on relevant code changes and fails the check if a
score drops unexpectedly (deliberately ignoring normal timing noise, which
isn't a real problem).

---

## 3. Simple M6 Architecture

**Automatic evaluation:**

```
SmartLawAI Pipeline
        ↓
Gold Test Sets  (the known-correct answers)
        ↓
Evaluation Runner  (eval/run_eval.py)
        ↓
Metrics + Statistical Analysis
        ↓
JSON + Markdown Reports
```

**Human evaluation, feeding into the same gold test sets:**

```
Human Rater
        ↓
Streamlit Annotation App
        ↓
Review Question / Passage / Answer
        ↓
Submit Rating
        ↓
Annotation Storage
        ↓
Agreement Analysis
```

The human side and the automatic side connect at the Gold Test Sets box — the
annotation app is literally how those files get written and updated.

---

## 4. The Five Gold Test Sets

| Gold Set | What It Tests | Current Seed Size | Why It Matters |
|---|---:|---:|---|
| **Retrieval** | Does the system find the right document passages for a question? | 10 questions | If retrieval is wrong, everything downstream is wrong too — it's the foundation |
| **OOS** (Out-of-scope) | Does the system correctly refuse questions it shouldn't answer (e.g., "should I sue?" needs a lawyer, not an AI) | 32 questions | A system that confidently answers things it shouldn't is dangerous in a legal context |
| **Groundedness** | Is a generated claim actually supported by the passage it cites? | 40 claim/passage pairs | This is the core hallucination check |
| **Repealed / Legal Authority** | Does the system know when a law has been replaced (e.g., old Indian Penal Code sections replaced in 2024)? | 34 questions | This is the project's central research idea — citing outdated law with total confidence is a real, current risk |
| **Entailment** | A separate, independent set used to judge how good our *judge* (the AI that checks other AI) is | 40 claim/passage pairs | Keeps us from grading our own homework — the judge is tested on data it never trained on |

**Important honesty note:** these numbers (10–40 items each) are **seed
data** — small starter sets built by one person to prove the whole pipeline
works end-to-end. The research plan calls for **300–400 items per set**,
built by multiple independent human raters with measured agreement between
them. That full-scale study has not happened yet — it needs real people
spending real time, which is a separate, future effort (see section 12). None
of the current results should be read as final, trustworthy scores — they
prove the *machinery* works, not that the *system* is good.

---

## 5. How M6 Evaluation Works

Step by step, what actually happens when you run an evaluation:

1. **Select the evaluation split** — e.g. "dev" (a development/testing subset,
   as opposed to a final "test" split saved for the very end).
2. **Load the required gold datasets** — reads all five `.jsonl` files.
3. **Run the SmartLawAI pipeline** — sends each question through the real
   system (currently, all internal AI components are placeholder "stub"
   versions, since the real ones — M2, M3, M4, M5 — aren't built yet).
4. **Collect system outputs** — what did the pipeline answer, what did it
   cite, did it answer or refuse?
5. **Calculate applicable metrics** — scores whatever can honestly be scored
   right now.
6. **Perform statistical analysis where data supports it** — e.g. checking
   whether the system's confidence matches its real accuracy.
7. **Handle unavailable metrics honestly** — if something can't be measured
   yet (e.g., no legal-authority database exists yet), the report says
   "unavailable" and explains why, instead of guessing or defaulting to zero.
8. **Generate JSON results** — a permanent, machine-readable record.
9. **Generate a Markdown report** — a plain-English summary a person can read.

**The actual command:**
```bash
python -m eval.run_eval --split dev
```

**What happens when you run it** (verified in a real run): it processed 42
test items with **zero failures**, computed **31 real metrics**, and
correctly marked **17 metrics as unavailable** with a clear reason for each
(e.g., "no legal-authority registry exists yet"). It writes two files —
`eval/results/dev_<timestamp>.json` and the matching `.md` report — which are
kept permanently as a record.

---

## 6. Human Evaluation and Streamlit App

Automatic metrics can check things like "does this text overlap with the
source text," but they **cannot** reliably judge things like "is this legal
claim actually true and properly supported." Some judgments genuinely need a
human with legal understanding. That's why M6 includes a human-rating system,
not just automated scoring.

**Streamlit** is simply the tool we used to build a quick, simple web page for
raters — no separate app to install, just run one command and it opens in a
browser.

**The actual command:**
```bash
streamlit run eval/annotate/app.py
```

**What a rater does:**
- **Gold set selection** — pick which of the five test sets to work on.
- **Rater ID** — just a name or ID typed in; it's not tied to "student" vs.
  "expert," so recruiting different types of raters later is easy.
- **Annotation items** — one question/claim at a time, shown with instructions
  and rating options, keyboard shortcuts included for speed.
- **Rating workflow** — read it, pick a rating, optionally leave a comment,
  click submit, move to the next one automatically.
- **Annotation progress** — a progress bar shows how many items are done; the
  sidebar shows live agreement stats and how many items/hour the rater is
  managing.
- **Storage/export** — every rating is saved immediately with a timestamp; a
  button exports finished ratings straight into the official gold test set
  files.

**Important:** this app is a research tool for evaluators, not part of the
SmartLawAI product an end-user would ever see.

---

## 7. Important Metrics

Only metrics that actually exist in the code today:

| Metric | What It Measures | Why It Matters |
|---|---|---|
| Recall / Precision / MRR / nDCG | Did retrieval find the right passages, and rank them near the top? | Foundation of the whole system — currently **unavailable** (see limitations) |
| ROUGE-1/2/L | Text overlap between a generated answer and a reference | Standard summarization quality check |
| Extractiveness | How much of the answer is copy-pasted vs. original | Catches a model that "cheats" by just quoting the source |
| Groundedness rate / Hallucination rate | Is a claim actually supported by its cited passage? | The direct hallucination measurement |
| Judge agreement (kappa) | Does our automatic checker agree with human judgment? | Tells us if the automatic checker can be trusted |
| Citation validity / Repealed-citation rate | Is a cited law real, and still in force? | Central to the project's legal-safety goal — currently **unavailable** (needs the legal-authority database, not built yet) |
| Refusal precision / recall | Does the system correctly refuse questions it should refuse? | Safety check against overconfident wrong answers |
| Calibration (ECE) | When the system says "I'm 90% sure," is it right 90% of the time? | Users need to be able to trust a stated confidence level — currently **unavailable** (needs a piece of data we don't have yet) |
| Latency (p50/p95) | How fast does each step of the pipeline run? | Performance tracking |

**Currently unavailable, and why that's honest, not a bug:** retrieval
metrics, citation/legal-authority metrics, and calibration all require pieces
of the system (M2 retrieval, M4 legal database, M5 verification) that haven't
been built yet. The evaluation code correctly says "unavailable" for these
instead of making up a fake score.

---

## 8. Simplified M6 Folder Structure

```
eval/
├── gold/            The five test-question files, with a written explanation
│                    of exactly how each one was made
├── annotate/         Everything for human rating: the app, storage, agreement
│                    stats, and the rulebook (guidelines) raters follow
├── config.py         All the settings and thresholds in one place
├── schemas.py         The exact required format for each gold-set file
├── metrics.py         All the scoring formulas
├── stats.py           The statistical comparison tools
├── extractiveness.py Copy-paste detection
├── gold_sets.py       Loads and checks the gold-set files
├── run_eval.py         The main script that runs everything end-to-end
├── report.py           Turns results into readable reports
└── results/          Every past evaluation run, kept permanently
```

---

## 9. What We Successfully Tested

Actual results from running the code, not estimates:

| Check | Result |
|---|---|
| Environment setup (Python, dependencies, imports) | **PASS** — all required packages installed and importing correctly |
| M6-specific automated tests | **356 passed**, 2 honestly skipped (see below) |
| Full project test suite (all modules, not just M6) | **398 passed**, 2 skipped |
| Evaluation runner (`run_eval.py`) | **PASS** — 42 items processed, **0 failures** |
| Report generation | **PASS** — JSON and Markdown files produced correctly |
| Streamlit annotation app | **PASS** — starts, serves in browser, shuts down cleanly |
| Integration tests (does everything connect correctly?) | **PASS** |

**The 2 skipped tests are honest, not hidden failures:** one skips because an
optional library (`bert-score`) isn't installed in this environment, one
skips because a data file from an earlier project stage (`data/splits/`)
doesn't exist yet — both clearly labeled with the reason, not silently
ignored.

**One unrelated known issue for transparency:** there's a pre-existing flaky
test in an unrelated part of the codebase (a random-number-based search test
that isn't part of M6 and sometimes fails by chance). It has nothing to do
with the evaluation system and was already present before M6 started.

---

## 10. Current Limitations

Being fully honest here, as required:

- **The pipeline is entirely stub/placeholder right now.** M2 (real
  retrieval), M3 (real answer generation), M4 (legal authority database), and
  M5 (verification) haven't been built yet. So M6's scores today measure "does
  the scoring system work," not "is SmartLawAI good."
- **Gold test sets are small seed data**, not the full research-grade
  dataset. 10–40 items per set today vs. a target of 300–400.
  Single-labeled by one person, not yet cross-checked by multiple independent
  raters.
- **Several important metrics are unavailable, honestly:**
  - Retrieval metrics — because document/passage IDs aren't stable yet
    (they change every run until M2 is built).
  - Citation validity and legal-authority checking — because the legal
    database (M4) doesn't exist yet.
  - Confidence calibration — because we don't yet have an independent way to
    check if the system's stated confidence is accurate.
- **Human annotation hasn't happened yet.** The app and storage system are
  built and tested, but real annotation sessions with real raters (and the
  agreement statistics that come from them) are still ahead.
- **A larger multi-rater study is required** before any evaluation number can
  be treated as a scientifically defensible result for the paper.
- **One data-quality item found during review:** the "repealed law" test
  set's mapping from old law sections to new law sections was built from the
  new law's own text, but has **not yet been individually double-checked**
  against the official government record. This is flagged directly in the
  gold-set documentation and must be verified before it's used for a real
  reported result.

None of these are secrets — they're documented in the repository itself
(`eval/gold/PROVENANCE.md` and the compliance reports) exactly as found above.

---

## 11. How M6 Connects to Other SmartLawAI Modules

Verified against `PLAN.md`'s actual milestone plan:

```
M0–M5
Build the SmartLawAI system
(skeleton → data → retrieval → answer generation →
 legal-authority database → verification/safety gate)
        ↓
M6  ← we are here
Evaluate and measure system quality
        ↓
M7
Compare against simple baseline approaches
(so we know if our fancy pipeline actually beats something simple)
        ↓
M8
Train and improve specialist AI models
        ↓
M9
Final experiments, analysis, and the research paper
```

The project's own rule (from `OPS.md`) is blunt about this ordering: **"Nobody
submits a training job until M6 can measure it."** In other words, M6 isn't
optional infrastructure — it's the gate that everything after it depends on.
A trained model nobody can measure isn't progress. That's also why M6 was
deliberately built to work *now*, against the placeholder pipeline, rather
than waiting for M2–M5 to finish first.

---

## 12. What Happens Next?

### Immediate
- Review this overview and the actual reports (`eval/results/*.md`) as a
  team, so everyone has seen a real evaluation output at least once.
- Decide who will act as human raters and schedule a first small trial
  ("pilot") session using the Streamlit app — the plan calls for piloting on
  20 items before scaling up any test set.
- Double-check the "repealed law" mapping against the official government
  record before anyone treats that set's numbers as final.

### After M4/M5 Are More Complete
- Retrieval metrics become measurable once M2 gives documents stable IDs.
- Citation-validity and repealed-citation metrics become measurable once M4's
  legal-authority database exists.
- Calibration (confidence-accuracy matching) becomes measurable once M5's
  real verification component exists.

### Future Research Work
- Scale each gold test set from 10–40 items up to the target 300–400.
- Recruit multiple independent human raters (not just one person) and measure
  real agreement between them.
- Run the full evaluation against real (non-stub) models once they exist, and
  compare against the M7 baseline approaches.
- Consider recruiting practising lawyers (not only students) as raters for a
  stronger, more credible evaluation — this is explicitly planned as a
  possible later addition.

---

## 13. Two-Minute Team Presentation

*A script you can read almost word-for-word.*

> "So M6 is done — but M6 isn't part of the AI system itself. M6 is how we
> check whether the AI system is actually good.
>
> Here's why that matters: if we build SmartLawAI and just eyeball a few
> answers, we have no real evidence it works, and no way to catch it
> confidently citing a law that doesn't exist anymore — which is exactly the
> kind of mistake that matters most in a legal product.
>
> So what we built is a scoring system. It has five sets of test questions
> with known-correct answers, a script that runs those questions through the
> real pipeline and scores the results automatically, and a small web app
> where a human can review and rate answers that need judgment a computer
> can't make alone — like whether a legal claim is actually supported by the
> evidence.
>
> We tested all of it. Nearly 400 automated checks pass across the whole
> project, the evaluation script runs end-to-end with zero failures, and the
> annotation app starts up and works correctly.
>
> But — and this is important — the results we're getting right now don't
> mean SmartLawAI is good yet. Every AI component in the pipeline is still a
> placeholder, because the real retrieval, answer-generation, and legal
> database pieces haven't been built. What today's results prove is that the
> *scoring machine* works. Also, our test data is small seed data right now —
> ten to forty examples per set, built by one person — not the three-hundred-
> plus, multi-rater dataset the research plan actually calls for.
>
> So next steps: we need real human raters to sit down and use the annotation
> app, starting with a small pilot, and we need to wait for the retrieval and
> legal-database pieces to land so the metrics that depend on them can
> actually produce numbers. Once both of those happen, M6 is what tells us,
> honestly, whether SmartLawAI is working."

---

## 14. FAQ for Teammates

**What exactly is M6?**
The scoring and testing system for the whole project — not a feature users
see, but the tool that tells us if the AI is working correctly.

**Why do we need evaluation at all?**
Because "it looks right to me" isn't evidence, and a legal AI that sounds
confident while being wrong can cause real harm. We need a repeatable,
honest way to measure quality.

**What are gold test sets?**
Files of test questions where we already know the correct answer (checked by
a human), used to check the AI's answers against.

**Why five separate gold sets instead of one big one?**
Because they test different things: can it find the right passage
(retrieval), does it know when to refuse (OOS), is its claim actually
supported by evidence (groundedness), does it know when a law changed
(repealed authority), and — separately — is our automatic checker itself
trustworthy (entailment).

**Why do we need human annotators if we have automatic metrics?**
Because some questions (is this legal claim genuinely true and well-
supported?) need real legal/language understanding a formula can't provide
reliably. Automatic metrics and human judgment check different things.

**What is the Streamlit app?**
A simple web page, opened with one command, that lets a human rater read a
question and answer, then click a rating. It's a research tool, not part of
the product.

**Who will use the annotation app?**
Whoever the team designates as raters — likely a mix of team members now,
and possibly practising lawyers later for a stronger study, per the research
plan.

**Are the current evaluation results real AI performance?**
No. Every AI component right now is a stand-in placeholder, so today's
numbers measure that the *scoring system* works, not that SmartLawAI itself
performs well. That will change once the real components (M2–M5) are built.

**What happens when new SmartLawAI modules are added?**
The same evaluation script runs unchanged — currently-unavailable metrics
(retrieval, citation validity, calibration) will simply start producing real
numbers as their required components (M2, M4, M5) land.

**What work still remains in M6?**
Scaling the gold sets to full size with real multi-rater human annotation,
double-checking the legal-authority mapping against official sources, and
running a real annotation pilot session.
