# Build Spec: Contract Review Prototype ("Redline")

## Context

This is a course prototype for an MSBA class. It accompanies a business plan; the plan carries the grade, and the prototype exists to demonstrate that the core review loop works. **Optimize for a working demo in two days, not for production quality.** Prefer the simplest thing that visibly works over the correct-but-slower thing.

If you hit a decision point not covered here, pick the simpler option and leave a `# NOTE:` comment explaining the choice.

---

## What the product does

Takes an inbound third-party contract, compares each clause against a "market standard" baseline, and returns flagged deviations with plain-language explanations and suggested redline language.

The user is a fractional general counsel or a small in-house lawyer reviewing contracts sent to them by counterparties. They are not drafting; they are evaluating someone else's paper.

---

## Stack

- **Python 3.10+**
- **Streamlit** for the UI (single-file app is fine)
- **Azure OpenAI** for model calls — deployment `gpt-4.1-mini`, endpoint and key from environment variables
- No database. Session state and local JSON files only.
- No vector database. If similarity search is needed, use `sentence-transformers` embeddings with cosine similarity computed in numpy over a list of maybe 10 baseline clauses. This is not a scale problem.

### Credentials

Read from environment, never hardcode:

```
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_KEY
AZURE_OPENAI_DEPLOYMENT   # gpt-4.1-mini
AZURE_OPENAI_API_VERSION  # 2024-10-21
```

Create a `.env.example` listing these with placeholder values, and add `.env` to `.gitignore`. This matters — the repo gets submitted.

---

## Pipeline

Build these as separate functions in separate modules so each can be tested alone.

### 1. Ingestion (`ingest.py`)

Accept a `.txt` file first. Add `.pdf` support via `pypdf` only if time remains — plain text is enough for the demo, and CUAD ships text versions.

### 2. Clause segmentation (`segment.py`)

Split contract text into clauses.

**This is the messiest part of the build. Do not over-engineer it.** Contracts don't split cleanly. A heuristic splitter that works on the demo contracts is completely acceptable.

Suggested approach: split on numbered section headings (`1.`, `1.1`, `ARTICLE V`, `Section 3.2`) via regex, fall back to double-newline paragraph breaks, then discard fragments under ~200 characters as boilerplate.

Return a list of `{clause_id, heading, text}`.

Log what fraction of the document ends up in clauses vs. discarded — useful for the writeup.

### 3. Clause classification (`classify.py`)

For each clause, assign one label from a fixed taxonomy by prompting the model.

Start with these 8 categories plus `Other`:

```
Indemnification
Limitation of Liability
Termination
Confidentiality
Intellectual Property Assignment
Governing Law
Payment Terms
Warranty
Other
```

Prompt the model to return **only** the category name, nothing else. Validate the response against the allowed list; if it doesn't match, retry once, then assign `Other`.

Batch multiple clauses into one call where possible to cut cost and latency — ask for JSON mapping clause_id to category, and parse defensively (strip markdown fences before `json.loads`).

### 4. Baseline library (`baselines.py`)

A hand-written dict: for each of the 8 categories, one "market standard" version of that clause, plus a short note on what makes it balanced.

Write these yourself as plain strings in the file. Two or three sentences each is enough. This is the piece that makes the product more than a generic AI wrapper, so don't skip it, but don't spend an hour on prose either.

### 5. Deviation analysis (`analyze.py`)

For each classified clause, send the model **both** the contract's clause and the matching baseline, and ask for a structured comparison.

Request JSON with exactly these fields:

```json
{
  "severity": "high" | "medium" | "low" | "none",
  "explanation": "one or two sentences on how this differs from standard and why it matters",
  "suggested_redline": "proposed replacement language, or null if no change needed",
  "confidence": "high" | "low"
}
```

**Confidence handling:** if the clause was classified `Other`, or the model returns `confidence: low`, do not show a suggested redline. Display "Flagged for attorney review — no automated suggestion" instead. This is a deliberate product decision, not a limitation. Keep it.

Parse defensively. Wrap in try/except and degrade to a "could not analyze" row rather than crashing the whole run.

### 6. Cost instrumentation (`costs.py`)

**Do not skip this.** It fills a section of the business plan.

Log for every API call: prompt tokens, completion tokens, model, and the pipeline stage. Accumulate per contract. Display total tokens and estimated cost in the UI after a run, and append each run to `runs.jsonl`.

Use published commercial rates for the estimate, in a constant at the top of the file so it's easy to change. Azure access here is institutional and free, but the business plan needs real market cost.

### 7. UI (`app.py`)

Streamlit, single page:

- File uploader
- "Analyze" button
- Progress indicator during processing (this takes a while — don't leave the user staring at nothing)
- Results as a table or expandable rows: clause heading, category, severity (color-coded), explanation, suggested redline
- Sort by severity, high first
- A summary line at top: N clauses analyzed, N flagged high, N flagged for attorney review
- Token count and estimated cost at the bottom

Skip: authentication, saving, multi-user, tracked-changes Word export.

---

## Explicitly out of scope

Do not build these even if they seem natural. They're described in the business plan as product features but are not needed for the demo:

- Tracked-changes `.docx` export
- Per-client workspaces / multi-tenancy
- The adaptive playbook that learns from accepted/rejected redlines
- A fine-tuned classifier (prompting is the documented prototype choice)
- Rule-based deterministic checks (defined-term consistency, date reconciliation)
- User accounts, persistence beyond `runs.jsonl`

If everything else is done and time remains, the rule-based checks are the most interesting addition.

---

## Test data

CUAD (Contract Understanding Atticus Dataset) — 510 commercial contracts from SEC EDGAR with lawyer annotations across 41 clause categories. Download from the Atticus Project site or a Hugging Face mirror. The `full_contract_txt/` directory is what's needed.

Pick 3–5 contracts of varied length for testing. Keep at least one that was never used while developing the baselines, so there's an honest test case.

---

## Build order

Get a crude version of the whole pipeline running end-to-end before improving any single stage. A rough pipeline that works beats three polished stages and nothing to demo.

1. Hardcode one clause, one baseline → confirm an API call returns parseable JSON
2. Segmentation on one real contract → inspect the output by hand
3. Classification over those clauses
4. Full analysis loop
5. Streamlit UI
6. Cost logging
7. Run across the test contracts, note what breaks

---

## Notes on failure

Where the model produces something wrong or strange, **do not paper over it.** Leave a `# NOTE:` comment describing what happened. Those observations go into the writeup, and an honest account of what didn't work is worth more there than a clean demo that hides its rough edges.

Particular things worth noting if they occur: clauses the segmenter mangles, categories the classifier confuses, redline suggestions that are fluent but legally wrong.
