"""Clause classification: assign each clause one category from a fixed
taxonomy by prompting the model. Clauses are batched into a single call
where possible to cut cost and latency.
"""

import json
import logging

from baselines import CATEGORIES
from llm_client import get_client, get_deployment
from parsing import extract_json

logger = logging.getLogger(__name__)

BATCH_SIZE = 8
# Classification only needs to recognize the clause type, not reason over
# every word — truncating keeps batched prompts small. Full clause text is
# preserved elsewhere for the analysis stage.
CLASSIFY_CHARS = 600

SYSTEM_PROMPT = (
    "You are a contract clause classifier for a legal review tool. "
    "Assign each clause exactly one category from this fixed list, and "
    "nothing else:\n"
    + "\n".join(f"- {c}" for c in CATEGORIES)
    + "\nIf a clause does not clearly fit one of the specific categories, "
    "use \"Other\". Respond with ONLY the requested JSON, no commentary, "
    "no markdown fences."
)


def _chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _classify_batch(batch, cost_tracker=None) -> dict:
    payload = [
        {
            "clause_id": c["clause_id"],
            "heading": c["heading"],
            "text": c["text"][:CLASSIFY_CHARS],
        }
        for c in batch
    ]
    user_prompt = (
        "Classify each of these clauses. Return a JSON object mapping each "
        "clause_id to its category string, e.g. "
        '{"c1": "Termination", "c2": "Other"}.\n\n'
        f"Clauses:\n{json.dumps(payload, indent=2)}"
    )

    client = get_client()
    deployment = get_deployment()
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    if cost_tracker is not None:
        usage = response.usage
        cost_tracker.log_call(
            stage="classify",
            model=deployment,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )

    raw = response.choices[0].message.content
    try:
        parsed = extract_json(raw)
        if not isinstance(parsed, dict):
            raise ValueError("expected a JSON object")
        return parsed
    except (json.JSONDecodeError, ValueError):
        logger.warning("classify: could not parse batch response, will retry per-clause: %r", raw)
        return {}


def _classify_single(clause, cost_tracker=None):
    """Retry path for a clause that came back with an invalid/missing
    category. Asks for just the bare category name."""
    user_prompt = (
        "Classify this single contract clause. Reply with ONLY the category "
        "name from the list, nothing else.\n\n"
        f"Heading: {clause['heading']}\n"
        f"Text: {clause['text'][:CLASSIFY_CHARS]}"
    )
    client = get_client()
    deployment = get_deployment()
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    if cost_tracker is not None:
        usage = response.usage
        cost_tracker.log_call(
            stage="classify_retry",
            model=deployment,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )

    raw = response.choices[0].message.content.strip()
    for category in CATEGORIES:
        if raw.lower() == category.lower():
            return category
    return None


def classify_clauses(clauses, cost_tracker=None) -> dict:
    """Return {clause_id: category} for every clause. Unparseable or
    off-taxonomy responses fall back to a single-clause retry, then to
    "Other" — never left unclassified and never allowed to crash the run."""
    results: dict = {}
    for batch in _chunks(clauses, BATCH_SIZE):
        batch_result = _classify_batch(batch, cost_tracker)
        for clause_id, category in batch_result.items():
            if category in CATEGORIES:
                results[clause_id] = category
            else:
                logger.warning(
                    "classify: batch returned off-taxonomy category %r for %s", category, clause_id
                )

    for clause in clauses:
        cid = clause["clause_id"]
        if cid not in results:
            retried = _classify_single(clause, cost_tracker)
            results[cid] = retried if retried in CATEGORIES else "Other"
            if retried is None:
                logger.warning("classify: %s could not be classified, defaulting to Other", cid)

    return results
