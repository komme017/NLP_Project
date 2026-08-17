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
streamlit run app.py
```

Required environment variables (see `.env.example`):

```
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_KEY
AZURE_OPENAI_DEPLOYMENT   # gpt-4.1-mini
AZURE_OPENAI_API_VERSION  # 2024-10-21
```

## Pipeline

Each stage is a separate module so it can be run/tested in isolation:

| Module | Responsibility |
|---|---|
| `ingest.py` | Load `.txt` or `.pdf` contract text |
| `segment.py` | Heuristic clause segmentation (heading regex, paragraph fallback) |
| `classify.py` | Batched LLM classification into 8 categories + "Other" |
| `baselines.py` | Hand-written market-standard clause per category |
| `analyze.py` | LLM deviation analysis: severity, explanation, suggested redline |
| `costs.py` | Per-call token logging, cost estimate, `runs.jsonl` |
| `app.py` | Streamlit UI tying it together |

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

This sandbox has no Azure OpenAI credentials configured, so `classify.py`
and `analyze.py` have **not** been run against a live model here — only
against a mocked client (`tests/test_pipeline.py`) that verifies the
retry/fallback/confidence-handling logic behaves as specified when the model
returns garbage, off-taxonomy labels, or unparseable JSON. Ingestion,
segmentation, cost math, and the Streamlit UI boot were all exercised
directly against the real test contracts above.

Run the offline tests:

```bash
python3 -m unittest tests/test_pipeline.py -v
```

**Before the live demo**, run the full pipeline end-to-end against all 5
test contracts with real credentials and skim `runs.jsonl` — that's the
step this session couldn't do, and where classifier confusion or fluent-
but-wrong redlines (the kind of thing the spec's "Notes on failure" section
wants written up) would actually surface.

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
fine-tuned classification, rule-based deterministic checks, user accounts/
persistence beyond `runs.jsonl`. See build spec for rationale.
