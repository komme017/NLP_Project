"""Clause classification using a RoBERTa model fine-tuned on CUAD, run as
extractive QA: for each of the 41 CUAD categories, ask whether the contract
contains that clause type, and if so, where. Explanations/redlines are
still generated separately by GPT-4.1-mini in analyze.py — this module only
replaces *detection*, not the natural-language reasoning layer.
"""

import logging

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

from baselines import CATEGORIES

logger = logging.getLogger(__name__)

MODEL_NAME = "akdeniz27/roberta-base-cuad"  # swap for "Rakib/roberta-base-on-cuad" if preferred
CONFIDENCE_THRESHOLD = 0.5  # tune based on false-positive/negative tradeoff you observe

_qa_pipeline = None  # lazy-loaded singleton so the model is only loaded once per process


def _get_pipeline():
    global _qa_pipeline
    if _qa_pipeline is None:
        logger.info("Loading CUAD QA model %s...", MODEL_NAME)
        model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _qa_pipeline = pipeline(
            "question-answering",
            model=model,
            tokenizer=tokenizer,
            handle_impossible_answer=True,  # lets the model say "no answer" instead of forcing a span
        )
    return _qa_pipeline


def _cuad_question(category: str) -> str:
    """CUAD's own question template — matches how the model was trained,
    which matters for QA models since prompt phrasing affects accuracy."""
    return (
        f'Highlight the parts (if any) of this contract related to '
        f'"{category}" that should be reviewed by a lawyer.'
    )


def classify_clauses(clauses, cost_tracker=None) -> dict:
    """Return {clause_id: category}, using the CUAD QA model instead of an
    LLM prompt. For each clause, run all 41 category-questions against its
    own text and keep the highest-confidence match above threshold; if none
    clears the threshold, fall back to "Other". No API cost tracking here
    since this runs locally — cost_tracker is accepted for interface
    compatibility with the old classify_clauses but not used.
    """
    qa = _get_pipeline()
    results: dict = {}

    for clause in clauses:
        cid = clause["clause_id"]
        context = clause["text"]

        best_category = None
        best_score = 0.0

        for category in CATEGORIES:
            question = _cuad_question(category)
            try:
                answer = qa(question=question, context=context)
            except Exception as e:
                logger.warning("classify: QA call failed for %s / %s: %s", cid, category, e)
                continue

            # handle_impossible_answer=True returns empty string + low score
            # when the model thinks there's no match for this category
            if answer["answer"] and answer["score"] > best_score:
                best_score = answer["score"]
                best_category = category

        if best_category and best_score >= CONFIDENCE_THRESHOLD:
            results[cid] = best_category
        else:
            results[cid] = "Other"
            logger.info(
                "classify: %s — no category cleared threshold (best=%s @ %.2f), defaulting to Other",
                cid, best_category, best_score,
            )

    return results