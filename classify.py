"""Clause classification using a CUAD-fine-tuned RoBERTa QA model, run
locally (no API calls) — see https://github.com/The-Atticus-Project/cuad.

## Why this looks different from a normal classifier

CUAD's released checkpoint isn't a sequence classifier over a label set.
It's trained for extractive QA: given a fixed question like `Highlight the
parts (if any) of this contract related to "Cap On Liability" that should
be reviewed by a lawyer. Details: ...` and a contract, it either extracts
the answering span or (SQuAD2.0-style) says there's no answer. CUAD has 41
of these fixed category-questions (see CUADv1.json in the repo above); a
label per clause has to be reverse-engineered by running our candidate
categories' questions against the clause and taking whichever one the model
answers most confidently. Explanations/redlines are still generated
separately by gpt-4.1-mini in analyze.py — this module only replaces
*detection*, not the natural-language reasoning layer.

# NOTE: CUAD's 41 categories do not cover this product's taxonomy 1:1.
# Only 5 of our 8 categories have a reasonable CUAD analog (see
# CUAD_QUESTIONS below — Termination maps to the narrower "Termination For
# Convenience", Warranty to the narrower "Warranty Duration", Limitation of
# Liability to the OR of "Cap On Liability"/"Uncapped Liability"). Indemnification,
# Confidentiality, and Payment Terms have no CUAD question at all — CUAD
# just never asked about them — so clauses of those types can *never* be
# predicted by this model and always fall through to "Other". That's a
# structural ceiling on this approach, not a bug: if you see every
# Indemnification/Confidentiality/Payment Terms clause landing in "Other",
# this is why. (This is very likely what was actually happening if
# *everything* showed up unclassified before — worth checking the 90%+
# Other-rate warning logged below to rule out a genuine load failure too.)
#
# An earlier version of this module built its questions from our own
# category names directly (e.g. literally asking about "Indemnification"
# or "Payment Terms") instead of CUAD's actual training-question phrasing.
# QA models are very sensitive to exact question wording, so those
# mismatched questions scored near zero across the board and *everything*
# fell through to "Other" -- this is almost certainly why classification
# looked completely broken. CUAD_QUESTIONS below uses the verbatim question
# text from CUADv1.json for the 5 categories that do have a real CUAD
# analog, which is the actual fix.

## Why classification used to fail with everything unclassified

The other common cause of every clause landing on "N/A"/"Other" is the
model failing to load (wrong model id/path, missing weights, missing
torch) and that failure getting silently absorbed somewhere upstream. This
module raises loudly instead — if the model can't load, classify_clauses
raises RuntimeError with an actionable message rather than quietly
returning "Other" for every clause and looking like a classification
result.

## Why this doesn't use transformers.pipeline("question-answering", ...)

That's the normal way to run extractive QA, but "question-answering" isn't
a registered pipeline task in every transformers version -- it's absent
from some installs' supported-task list entirely (confirmed against a real
install: KeyError listing table-question-answering and
document-question-answering as the only QA-flavored tasks available, no
plain extractive one). Rather than depend on that pipeline convenience
wrapper, this module drives AutoModelForQuestionAnswering/AutoTokenizer
directly and does its own start/end-span decoding -- those model/tokenizer
classes are far more fundamental and stable across versions than the
pipeline task registry sitting on top of them.
"""

import functools
import logging
import math
import os

import torch
from dotenv import load_dotenv
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from baselines import CATEGORIES

# Loaded here too (not just in llm_client.py) so CUAD_MODEL_PATH below picks
# up .env regardless of which module happens to be imported first -- this
# used to only work because analyze.py (which loads it via llm_client) was
# imported before classify.py in app.py/app_2.py, which is a fragile thing
# to depend on. load_dotenv() is a no-op if called more than once.
load_dotenv()

logger = logging.getLogger(__name__)

# Accepts either a Hugging Face Hub id or a local directory path. Defaults
# to a community-hosted mirror of the CUAD checkpoint (the official weights
# are only distributed via Zenodo, linked from the repo's README, with no
# Hub mirror from the Atticus Project itself) -- "Rakib/roberta-base-on-cuad"
# is a documented alternative if this one is ever unavailable. Override via
# env var to point at a local directory (e.g. an extracted Zenodo download)
# instead.
CUAD_MODEL_PATH = os.environ.get("CUAD_MODEL_PATH", "akdeniz27/roberta-base-cuad")

# clause text this long already gives the QA model enough context to work
# with; kept well under MAX_SEQ_LEN in *characters* so truncation to
# MAX_SEQ_LEN *tokens* below rarely has to cut anything real off.
MAX_CONTEXT_CHARS = 4000

# CUAD's own training truncates to this many tokens (question + context
# combined). Clauses are already segmented (far shorter than a full
# contract, which is what CUAD's sliding-window/doc_stride handling is
# for), so a single truncated pass is enough here -- no need to reproduce
# the multi-window stitching CUAD's own eval script does for full
# documents.
MAX_SEQ_LEN = 384

# a span longer than this is almost certainly the model latching onto
# something wrong rather than a real answer -- caps the O(n * k) search
# below and matches typical SQuAD-style postprocessing.
MAX_ANSWER_TOKENS = 60

# category -> one or more CUAD question strings (verbatim from CUADv1.json,
# since exact phrasing is what the model was trained on). Multiple
# questions per category are OR'd together — the clause is assigned that
# category if *either* question gets a confident answer.
CUAD_QUESTIONS = {
    "Governing Law": [
        'Highlight the parts (if any) of this contract related to "Governing Law" that '
        "should be reviewed by a lawyer. Details: Which state/country's law governs the "
        "interpretation of the contract?"
    ],
    "Termination": [
        'Highlight the parts (if any) of this contract related to "Termination For '
        'Convenience" that should be reviewed by a lawyer. Details: Can a party terminate '
        "this  contract without cause (solely by giving a notice and allowing a waiting  "
        "period to expire)?"
    ],
    "Limitation of Liability": [
        'Highlight the parts (if any) of this contract related to "Cap On Liability" that '
        "should be reviewed by a lawyer. Details: Does the contract include a cap on "
        "liability upon the breach of a party’s obligation? This includes time "
        "limitation for the counterparty to bring claims or maximum amount for recovery.",
        'Highlight the parts (if any) of this contract related to "Uncapped Liability" that '
        "should be reviewed by a lawyer. Details: Is a party’s liability uncapped upon "
        "the breach of its obligation in the contract? This also includes uncap liability "
        "for a particular type of breach such as IP infringement or breach of "
        "confidentiality obligation.",
    ],
    "Intellectual Property Assignment": [
        'Highlight the parts (if any) of this contract related to "Ip Ownership Assignment" '
        "that should be reviewed by a lawyer. Details: Does intellectual property created  "
        "by one party become the property of the counterparty, either per the terms of the "
        "contract or upon the occurrence of certain events?"
    ],
    "Warranty": [
        'Highlight the parts (if any) of this contract related to "Warranty Duration" that '
        "should be reviewed by a lawyer. Details: What is the duration of any  warranty "
        "against defects or errors in technology, products, or services  provided under "
        "the contract?"
    ],
}

# Categories in our taxonomy CUAD has no question for at all — see module
# docstring. Listed explicitly (rather than just being absent from
# CUAD_QUESTIONS) so the "structurally unsupported" set is easy to find and
# assert against in tests.
CUAD_UNSUPPORTED_CATEGORIES = sorted(set(CATEGORIES) - {"Other"} - set(CUAD_QUESTIONS))

CONFIDENCE_THRESHOLD = 0.5
OTHER_RATE_WARNING_THRESHOLD = 0.9


@functools.lru_cache(maxsize=1)
def _load_model():
    # If CUAD_MODEL_PATH looks like a local path, a missing directory
    # should fail immediately -- letting transformers/huggingface_hub treat
    # it as a Hub id instead means a network resolution attempt with its
    # own retry/backoff, which can hang for minutes in an environment with
    # no route to huggingface.co before finally producing the same "not
    # found" conclusion. Only genuine Hub ids (no local-path prefix) fall
    # through to the real network attempt below.
    looks_like_local_path = CUAD_MODEL_PATH.startswith((".", "/", "~"))
    if looks_like_local_path and not os.path.isdir(CUAD_MODEL_PATH):
        raise RuntimeError(
            f"CUAD_MODEL_PATH={CUAD_MODEL_PATH!r} looks like a local path but that "
            "directory doesn't exist. Download the checkpoint from the Zenodo link in "
            "https://github.com/The-Atticus-Project/cuad and point CUAD_MODEL_PATH at "
            "the extracted directory (or set it to a Hugging Face Hub id instead)."
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(CUAD_MODEL_PATH)
        model = AutoModelForQuestionAnswering.from_pretrained(CUAD_MODEL_PATH)
        model.eval()
        return tokenizer, model
    except Exception as e:
        raise RuntimeError(
            f"Could not load the CUAD QA model from CUAD_MODEL_PATH={CUAD_MODEL_PATH!r}: {e}. "
            "Download the checkpoint from the Zenodo link in "
            "https://github.com/The-Atticus-Project/cuad and set CUAD_MODEL_PATH to the "
            "local directory (or a reachable Hugging Face Hub id)."
        ) from e


def _answer_question(question: str, context: str, tokenizer, model) -> tuple:
    """Run one (question, context) pair directly through the model and
    decode the best answer span ourselves -- see module docstring for why
    this doesn't go through transformers.pipeline("question-answering").

    Returns (answer_text, confidence). answer_text is "" when the model
    prefers "no answer" (SQuAD2.0-style null-answer handling, matching how
    CUAD's checkpoint was trained): the <s>/[CLS] token at position 0 is
    the conventional null-answer slot, so a real span only wins if it
    outscores that position. confidence is sigmoid(best_span_score -
    null_score) -- roughly comparable to (but not numerically identical
    to) the confidence transformers.pipeline used to report; 0.5 means
    "exactly as confident as no answer at all", which is what
    CONFIDENCE_THRESHOLD is calibrated against.
    """
    inputs = tokenizer(
        question,
        context,
        max_length=MAX_SEQ_LEN,
        truncation="only_second",
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = inputs.pop("offset_mapping")[0].tolist()
    sequence_ids = inputs.sequence_ids(0)

    with torch.no_grad():
        outputs = model(**inputs)
    start_logits = outputs.start_logits[0].tolist()
    end_logits = outputs.end_logits[0].tolist()

    null_score = start_logits[0] + end_logits[0]

    context_positions = [i for i, sid in enumerate(sequence_ids) if sid == 1]
    if not context_positions:
        return "", 0.0
    context_position_set = set(context_positions)
    last_context_pos = context_positions[-1]

    best_score = null_score
    best_span = None
    for start in context_positions:
        s_logit = start_logits[start]
        for end in range(start, min(start + MAX_ANSWER_TOKENS, last_context_pos + 1)):
            if end not in context_position_set:
                continue
            score = s_logit + end_logits[end]
            if score > best_score:
                best_score = score
                best_span = (start, end)

    confidence = 1 / (1 + math.exp(-(best_score - null_score)))

    if best_span is None:
        return "", confidence

    char_start = offsets[best_span[0]][0]
    char_end = offsets[best_span[1]][1]
    return context[char_start:char_end], confidence


def _classify_one(clause: dict, tokenizer, model) -> str:
    context = clause["text"][:MAX_CONTEXT_CHARS]
    best_category = None
    best_score = 0.0

    for category, questions in CUAD_QUESTIONS.items():
        for question in questions:
            try:
                answer, score = _answer_question(question, context, tokenizer, model)
            except Exception as e:
                logger.warning(
                    "classify: QA call failed for %s / %r: %s", clause["clause_id"], category, e
                )
                continue

            if answer and score > best_score:
                best_score = score
                best_category = category

    if best_category is not None and best_score >= CONFIDENCE_THRESHOLD:
        return best_category
    return "Other"


def classify_clauses(clauses: list, cost_tracker=None) -> dict:
    """Return {clause_id: category}. cost_tracker is accepted for interface
    compatibility with the previous Azure-based classifier (app.py/app_2.py
    both pass it) but unused here — this model runs locally, no API cost.

    Raises RuntimeError if the model can't be loaded, rather than silently
    returning "Other" for every clause — a systemic setup failure should be
    loud, not indistinguishable from a real (if boring) classification
    result."""
    tokenizer, model = _load_model()

    results = {clause["clause_id"]: _classify_one(clause, tokenizer, model) for clause in clauses}

    if clauses:
        other_count = sum(1 for v in results.values() if v == "Other")
        other_rate = other_count / len(clauses)
        if other_rate > OTHER_RATE_WARNING_THRESHOLD:
            logger.warning(
                "classify: %d/%d clauses (%.0f%%) classified as Other. Expected for "
                "Indemnification/Confidentiality/Payment Terms clauses (CUAD has no "
                "question for those categories — see module docstring), but if this "
                "contract shouldn't be almost all Other, check CUAD_MODEL_PATH is "
                "actually loading real weights.",
                other_count,
                len(clauses),
                other_rate * 100,
            )

    return results
