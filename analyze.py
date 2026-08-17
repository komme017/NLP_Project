"""Deviation analysis: compare each classified clause against its market
baseline and request a structured severity/explanation/redline judgment.

Confidence handling (deliberate product decision, not a limitation): a
clause classified "Other" has no market baseline to compare against, so it
skips the model call entirely and is flagged for attorney review. A clause
that *does* have a baseline but comes back with confidence "low" still gets
no suggested redline surfaced — low-confidence redlines are worse than none,
because a fluent-but-wrong suggestion is more dangerous than an honest
"can't automate this."
"""

import json
import logging

from baselines import get_baseline
from llm_client import get_client, get_deployment
from parsing import extract_json

logger = logging.getLogger(__name__)

REVIEW_MESSAGE = "Flagged for attorney review — no automated suggestion"

SYSTEM_PROMPT = (
    "You are a contract review assistant helping a lawyer evaluate an "
    "inbound counterparty contract against market-standard baseline "
    "language. Compare the contract clause to the baseline and respond "
    "with ONLY a JSON object with exactly these fields:\n"
    '{"severity": "high" | "medium" | "low" | "none", '
    '"explanation": "one or two sentences on how this differs from '
    'standard and why it matters", '
    '"suggested_redline": "proposed replacement language, or null if no '
    'change needed", '
    '"confidence": "high" | "low"}\n'
    "Use \"none\" severity when the clause is materially equivalent to the "
    "baseline. Use confidence \"low\" when the clause is ambiguous, "
    "unusually structured, or you are not confident a clean redline is "
    "safe to propose. No markdown fences, no commentary outside the JSON."
)


def _no_baseline_result(clause, category):
    return {
        "clause_id": clause["clause_id"],
        "heading": clause["heading"],
        "category": category,
        "severity": None,
        "explanation": "No market-standard baseline exists for this category.",
        "suggested_redline": None,
        "confidence": "low",
        "show_redline": False,
        "review_message": REVIEW_MESSAGE,
        "status": "no_baseline",
    }


def _error_result(clause, category, reason):
    return {
        "clause_id": clause["clause_id"],
        "heading": clause["heading"],
        "category": category,
        "severity": None,
        "explanation": f"Could not analyze this clause ({reason}).",
        "suggested_redline": None,
        "confidence": "low",
        "show_redline": False,
        "review_message": REVIEW_MESSAGE,
        "status": "error",
    }


def analyze_clause(clause: dict, category: str, cost_tracker=None) -> dict:
    baseline = get_baseline(category)
    if baseline is None:
        return _no_baseline_result(clause, category)

    user_prompt = (
        f"Category: {category}\n\n"
        f"Contract clause under review:\n{clause['text']}\n\n"
        f"Market-standard baseline clause:\n{baseline['clause']}\n\n"
        f"Why the baseline is considered balanced:\n{baseline['note']}"
    )

    try:
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
                stage="analyze",
                model=deployment,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
            )

        raw = response.choices[0].message.content
        parsed = extract_json(raw)

        severity = parsed.get("severity")
        explanation = parsed.get("explanation", "")
        suggested_redline = parsed.get("suggested_redline")
        confidence = parsed.get("confidence", "low")

        if severity not in ("high", "medium", "low", "none"):
            logger.warning("analyze: %s got off-schema severity %r", clause["clause_id"], severity)
            severity = None

        low_confidence = confidence != "high"
        show_redline = bool(suggested_redline) and not low_confidence

        result = {
            "clause_id": clause["clause_id"],
            "heading": clause["heading"],
            "category": category,
            "severity": severity,
            "explanation": explanation,
            "suggested_redline": suggested_redline,
            "confidence": confidence,
            "show_redline": show_redline,
            "review_message": None if show_redline else REVIEW_MESSAGE,
            "status": "ok",
        }
        return result

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning("analyze: %s failed to parse model output: %s", clause["clause_id"], e)
        return _error_result(clause, category, "unparseable model response")
    except Exception as e:  # Azure/network errors etc. — never crash the whole run
        logger.warning("analyze: %s API call failed: %s", clause["clause_id"], e)
        return _error_result(clause, category, "API call failed")


def analyze_clauses(clauses: list, categories: dict, cost_tracker=None) -> list:
    """clauses: list of {clause_id, heading, text}. categories: {clause_id: category}."""
    results = []
    for clause in clauses:
        category = categories.get(clause["clause_id"], "Other")
        results.append(analyze_clause(clause, category, cost_tracker))
    return results
