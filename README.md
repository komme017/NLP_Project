# Redline — Contract Review Prototype

Course prototype (MSBA) accompanying a business plan. Takes an inbound
third-party contract, compares each clause against a hand-written
"market standard" baseline, and flags deviations with a plain-language
explanation and (when confidence is high enough) a suggested redline.
See `prototype_build_spec.md` for the full build spec; this README covers
setup, running it, and what's been validated so far.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your Azure OpenAI endpoint/key
streamlit run app_2.py
```

Required environment variables (see `.env.example`) — only needed for
`analyze.py`'s explanation/redline calls, since classification now runs
locally:

```
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_KEY
AZURE_OPENAI_DEPLOYMENT   # gpt-4.1-mini
AZURE_OPENAI_API_VERSION  # 2024-10-21
```

`classify.py` additionally needs to reach a CUAD-fine-tuned RoBERTa
checkpoint (Hugging Face Hub by default, or a local directory) — see
**Classification model** below.

## Two UI variants

- **`app.py`** — flat, sortable list of results (one expander per clause).
- **`app_2.py`** (`streamlit run app_2.py`) — shows the full contract text
  with flagged clauses highlighted inline (severity color-coded), plus a
  flag picker in a side panel. Selecting a flag scrolls the document pane
  to that clause and outlines it. Both variants share the same pipeline
  (`ingest`/`segment`/`classify`/`analyze`/`costs`) and only differ in
  `render_*`/rendering code.

## Pipeline

Each stage is a separate module so it can be run/tested in isolation:

| Module | Responsibility |
|---|---|
| `ingest.py` | Load `.txt` or `.pdf` contract text (pypdf is imported lazily, inside `load_pdf`, so a broken/missing PDF crypto backend can't break plain `.txt` uploads — hit this for real in a sandboxed env, see commit history) |
| `segment.py` | Heuristic clause segmentation (heading regex, paragraph fallback); each returned clause also carries `start`/`end` character offsets into the original text, which `app_2.py` uses to splice highlights into the full document |
| `classify.py` | Local classification via a CUAD-fine-tuned RoBERTa QA model (no API calls, no cost) — see **Classification model** below |
| `baselines.py` | Hand-written market-standard clause per category |
| `analyze.py` | LLM deviation analysis: severity, explanation, suggested redline |
| `costs.py` | Per-call token logging, cost estimate, `runs.jsonl` |
| `app.py` | Streamlit UI tying it together |

## Classification model

`classify.py` uses the RoBERTa-base checkpoint fine-tuned on CUAD by
[The-Atticus-Project/cuad](https://github.com/The-Atticus-Project/cuad)
instead of prompting gpt-4.1-mini — `analyze.py` still uses gpt-4.1-mini for
the explanation/severity/redline generation, since that part genuinely
needs a generative model, not a classifier.

**This is not a drop-in classifier swap.** The CUAD checkpoint is trained
for extractive QA, not single-label classification: given one of 41 fixed
"Highlight the parts of this contract related to X" questions and a
contract, it extracts the answering span or says there's no answer
(SQuAD2.0-style). There's no "give me the category" mode. `classify.py`
gets a label per clause by running our candidate categories' CUAD questions
against the clause text and taking whichever one the model answers most
confidently (`CONFIDENCE_THRESHOLD = 0.5`); below that, or with no
confident answer, it falls back to `"Other"`.

**The bigger issue: CUAD's 41 categories don't cover this product's 8.**
Only 5 have a usable CUAD analog, and some of those are narrower than our
category (see `CUAD_QUESTIONS` in `classify.py`):

| Our category | CUAD question(s) used | Fit |
|---|---|---|
| Governing Law | Governing Law | exact |
| Termination | Termination For Convenience | narrower — only fires for for-convenience termination language |
| Limitation of Liability | Cap On Liability OR Uncapped Liability | reasonable |
| Intellectual Property Assignment | Ip Ownership Assignment | close |
| Warranty | Warranty Duration | narrower — only fires on duration language, not warranty disclaimers generally |
| Indemnification | *(none)* | CUAD has no question for this at all |
| Confidentiality | *(none)* | CUAD has no question for this at all |
| Payment Terms | *(none)* | CUAD has no question for this at all |

Indemnification, Confidentiality, and Payment Terms clauses **cannot ever
be classified as anything but "Other"** with this model — there's no CUAD
question to route them through, not a threshold or prompt-tuning problem.
`classify.CUAD_UNSUPPORTED_CATEGORIES` lists these explicitly, and
`classify_clauses` logs a warning if over 90% of a contract's clauses land
on "Other" (expected on a contract heavy with those three categories;
worth double-checking the model actually loaded if it looks too high on a
more varied contract).

**Getting an all-"Other"/N/A result for every clause has two known
causes**, both fixed in this version but worth knowing about directly:
1. An earlier version built its questions from our own category names
   directly (e.g. literally asking about `"Indemnification"`) instead of
   CUAD's actual training-question phrasing. QA models are very sensitive
   to exact question wording, so those mismatched questions scored near
   zero across the board and *everything* fell through to `"Other"`. Fixed
   by using CUAD's verbatim question text (see `CUAD_QUESTIONS`).
2. The model silently failing to load, with every clause falling back to
   `"Other"` and no visible signal that anything was wrong. Fixed by
   raising `RuntimeError` loudly instead — `app.py`/`app_2.py` catch it and
   show the actual error in the UI rather than a silent all-Other result.

`CUAD_MODEL_PATH` (env var) controls which checkpoint loads — defaults to
the Hugging Face Hub mirror `akdeniz27/roberta-base-cuad`
(`Rakib/roberta-base-on-cuad` is a documented alternative), or point it at
a local directory containing a Zenodo download instead.

**This sandbox cannot download the actual checkpoint to verify any of the
above against real weights.** The official checkpoint is distributed via
Zenodo (linked from the CUAD repo's README), and this environment's
network policy blocks both `zenodo.org` and `huggingface.co` outright —
so neither the default Hub mirror nor a manual Zenodo download can be
fetched here. What's verified here instead:
- The adapter logic (confidence thresholding, multi-question OR-ing per
  category, the three unsupported categories being structurally
  unreachable, the loud-failure-on-load-error path) against a mocked QA
  pipeline — see `tests/test_pipeline.py`.
- The real (non-mocked) failure path: pointing `classify.py` at a
  non-existent `CUAD_MODEL_PATH` in this sandbox and confirming it raises
  a clear, actionable `RuntimeError` rather than hanging or silently
  returning garbage.
- That `transformers.pipeline("question-answering", ...)` is still a valid
  registered pipeline type in the installed `transformers` version (5.15) —
  worth checking explicitly since that pipeline class isn't re-exported at
  the top level anymore in this version, which looked at first like it
  might have been removed.

What's genuinely unverified: real classification accuracy against real
weights. If you have the checkpoint locally, running the 5 test contracts
through and comparing against the old gpt-4.1-mini-prompted classifications
would be the natural next step and a good "Notes on failure" data point.

## Test data

`data/test_contracts/` holds 5 real contracts pulled from CUAD (Contract
Understanding Atticus Dataset — 510 SEC EDGAR commercial contracts), sourced
from the full contract text embedded in `CUADv1.json` in the
[TheAtticusProject/cuad](https://github.com/TheAtticusProject/cuad) repo.
Picked for varied length so the segmenter gets exercised differently by each:

| File | Chars | Segmentation method used |
|---|---|---|
| PrecheckHealthServicesInc ... Distributor Agreement | 6.3K | paragraph fallback |
| LEGACYTECHNOLOGYHOLDINGS ... Distributor Agreement | 15.1K | paragraph fallback |
| MTITECHNOLOGYCORP ... Reseller Agreement | 32.7K | heading split |
| ReynoldsConsumerProductsInc ... Supply Agreement | 61.7K | paragraph fallback |
| VERICELCORP ... Supply Agreement | 102K | heading split |

`baselines.py` was written from general knowledge of standard commercial
terms, not derived from any of these five contracts — so all five are
honest test cases for the baseline comparisons, not contracts the baselines
were fit to.

## What's been validated in this build session

This sandbox has no Azure OpenAI credentials, so `analyze.py` has **not**
been run against a live model here. It also can't reach the CUAD
checkpoint's actual weights (Zenodo/Hugging Face both blocked), so
`classify.py`'s real classification accuracy is unverified too — see
**Classification model** above for exactly what was and wasn't checked
there. Both modules are covered by mocked-response tests
(`tests/test_pipeline.py`) that verify the control-flow logic (retry/
fallback/confidence-handling for `analyze.py`, confidence-thresholding/
category-mapping/loud-failure for `classify.py`) behaves as specified.
Ingestion, segmentation, cost math, and both Streamlit UIs (including the
document-highlight/scroll-to-flag interaction in `app_2.py`) were exercised
directly — segmentation against all 5 real test contracts, the UIs in a
real browser via Playwright.

Run the offline tests:

```bash
python3 -m unittest tests/test_pipeline.py -v
```

**Before the live demo**, run the full pipeline end-to-end against all 5
test contracts with real Azure credentials and a real local CUAD
checkpoint, and skim `runs.jsonl` — that's the step this session couldn't
do, and where classifier confusion or fluent-but-wrong redlines (the kind
of thing the spec's "Notes on failure" section wants written up) would
actually surface. Worth specifically checking whether real CUAD weights
classify the Termination/Warranty/IP-Assignment clauses in the test
contracts sensibly, given those categories only have a narrower CUAD
analog to work with (see table above).

## Known segmentation issues (observed, see `segment.py` for detail)

- **Inline-numbered clauses**: when a contract's numbered clauses run
  together inside one long paragraph instead of starting on their own line
  (e.g. the PreCheck distributor agreement), the paragraph fallback keeps
  the whole run as one oversized clause instead of splitting it — 18
  numbered clauses collapse into 3 blobs on that contract.
- **Table-of-contents contamination**: a front-matter ToC that repeats
  article titles with page numbers (VERICELCORP) matches the same heading
  regex as the real section headings, producing near-duplicate low-content
  "clauses" ahead of the real ones.

## Explicitly out of scope

Tracked-changes `.docx` export, multi-tenancy, an adaptive playbook,
rule-based deterministic checks, user accounts/persistence beyond
`runs.jsonl`. See build spec for rationale.

Fine-tuned classification was originally on this list too ("prompting is
the documented prototype choice") — that decision was deliberately
overridden mid-build in favor of the CUAD RoBERTa model described above.
Worth weighing in the writeup: it cut classification's LLM API cost to
zero, but at the cost of 3 of 8 categories (Indemnification,
Confidentiality, Payment Terms) being permanently unclassifiable by that
model, and unverified real-world accuracy since this sandbox can't reach
the actual weights.
