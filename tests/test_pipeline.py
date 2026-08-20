"""Offline tests for the parts of the pipeline that don't need a live Azure
OpenAI call: segmentation, defensive JSON parsing, and the classify/analyze
control flow (mocking the model response). Run with:

    python3 -m unittest tests/test_pipeline.py -v

These don't replace a real run against Azure OpenAI (see README) — they
verify the retry/fallback/confidence-handling logic behaves as specified
even when the model returns garbage.
"""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import analyze
import classify
import segment
from parsing import extract_json


def fake_response(content: str, prompt_tokens=100, completion_tokens=20):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


class FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        return self._responses.pop(0)


class TestParsing(unittest.TestCase):
    def test_strips_markdown_fence(self):
        raw = '```json\n{"a": 1}\n```'
        self.assertEqual(extract_json(raw), {"a": 1})

    def test_plain_json(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})


class TestSegment(unittest.TestCase):
    def test_discards_short_fragments_and_reports_stats(self):
        text = "ARTICLE 1 DEFINITIONS\n" + ("x" * 300) + "\n\nARTICLE 2 TERM\n" + ("y" * 300) + "\n\nARTICLE 3 SHORT\nfoo\n\nARTICLE 4 MISC\n" + ("z" * 300)
        clauses, stats = segment.segment_contract(text)
        self.assertEqual(stats["method"], "heading")
        # the "foo" clause under ARTICLE 3 should be discarded as boilerplate
        self.assertTrue(all(len(c["text"]) >= segment.MIN_CLAUSE_CHARS for c in clauses))
        self.assertLess(stats["num_clauses_kept"], stats["num_segments_found"])

    def test_falls_back_to_paragraphs_when_no_headings(self):
        text = ("a" * 250) + "\n\n" + ("b" * 250) + "\n\n" + ("c" * 250)
        clauses, stats = segment.segment_contract(text)
        self.assertEqual(stats["method"], "paragraph")
        self.assertEqual(len(clauses), 3)

    def test_spans_round_trip_to_original_text_and_dont_overlap(self):
        text = (
            "ARTICLE 1 DEFINITIONS\n" + ("x" * 300) + "\n\n"
            "ARTICLE 2 TERM\n" + ("y" * 300) + "\n\n"
            "ARTICLE 3 MISC\n" + ("z" * 300)
        )
        clauses, _ = segment.segment_contract(text)
        for c in clauses:
            self.assertGreaterEqual(c["start"], 0)
            self.assertLessEqual(c["end"], len(text))
            self.assertLess(c["start"], c["end"])
            # the clause's extracted body must appear verbatim within its
            # own span of the original text (span may also include the
            # heading line, so this is a substring check, not equality)
            self.assertIn(c["text"], text[c["start"] : c["end"]])
        for a, b in zip(clauses, clauses[1:]):
            self.assertLessEqual(a["end"], b["start"])


def fake_answer_question(rules):
    """Stand-in for classify._answer_question. Matched by substring against
    the (verbatim CUAD) question text, so tests don't have to reproduce the
    full question strings — just enough to identify which CUAD category
    question is being asked. rules: list of (question_substring, answer,
    score)."""

    def _fake(question, context, tokenizer, model):
        for substr, answer, score in rules:
            if substr in question:
                return answer, score
        return "", 0.0

    return _fake


class TestClassify(unittest.TestCase):
    def test_confident_match_assigns_mapped_category(self):
        clauses = [{"clause_id": "c1", "heading": "Gov Law", "text": "x" * 250}]
        fake = fake_answer_question([("Governing Law", "Delaware law applies", 0.92)])
        with patch("classify._load_model", return_value=(None, None)), patch(
            "classify._answer_question", side_effect=fake
        ):
            result = classify.classify_clauses(clauses)
        self.assertEqual(result, {"c1": "Governing Law"})

    def test_below_confidence_threshold_falls_to_other(self):
        clauses = [{"clause_id": "c1", "heading": "Ambiguous", "text": "x" * 250}]
        fake = fake_answer_question([("Governing Law", "maybe Delaware", 0.2)])
        with patch("classify._load_model", return_value=(None, None)), patch(
            "classify._answer_question", side_effect=fake
        ):
            result = classify.classify_clauses(clauses)
        self.assertEqual(result, {"c1": "Other"})

    def test_second_question_in_a_multi_question_category_can_win(self):
        # "Limitation of Liability" maps to two CUAD questions (Cap On
        # Liability, Uncapped Liability) OR'd together
        clauses = [{"clause_id": "c1", "heading": "Liability", "text": "x" * 250}]
        fake = fake_answer_question([("Uncapped Liability", "no cap stated", 0.75)])
        with patch("classify._load_model", return_value=(None, None)), patch(
            "classify._answer_question", side_effect=fake
        ):
            result = classify.classify_clauses(clauses)
        self.assertEqual(result, {"c1": "Limitation of Liability"})

    def test_categories_with_no_cuad_question_are_structurally_unreachable(self):
        # CUAD has no question at all for these three categories, so no
        # matter how the QA model scores, _classify_one can only ever
        # return "Other" or a key from CUAD_QUESTIONS for a given clause —
        # these three simply aren't in that dict.
        self.assertEqual(
            set(classify.CUAD_UNSUPPORTED_CATEGORIES),
            {"Indemnification", "Confidentiality", "Payment Terms"},
        )
        for unsupported in classify.CUAD_UNSUPPORTED_CATEGORIES:
            self.assertNotIn(unsupported, classify.CUAD_QUESTIONS)

    def test_model_load_failure_raises_instead_of_silently_defaulting_everyone_to_other(self):
        clauses = [{"clause_id": "c1", "heading": "Anything", "text": "x" * 250}]
        with patch("classify._load_model", side_effect=RuntimeError("could not load model")):
            with self.assertRaises(RuntimeError):
                classify.classify_clauses(clauses)


class TestAnalyze(unittest.TestCase):
    def test_other_category_skips_api_call_entirely(self):
        clause = {"clause_id": "c1", "heading": "Misc", "text": "x" * 250}
        with patch("analyze.get_client") as mock_get_client:
            result = analyze.analyze_clause(clause, "Other")
        mock_get_client.assert_not_called()
        self.assertEqual(result["status"], "no_baseline")
        self.assertFalse(result["show_redline"])
        self.assertEqual(result["review_message"], analyze.REVIEW_MESSAGE)

    def test_low_confidence_suppresses_redline(self):
        clause = {"clause_id": "c1", "heading": "Term", "text": "x" * 250}
        payload = {
            "severity": "high",
            "explanation": "deviates a lot",
            "suggested_redline": "use this instead",
            "confidence": "low",
        }
        client = FakeClient([fake_response(json.dumps(payload))])
        with patch("analyze.get_client", return_value=client), patch(
            "analyze.get_deployment", return_value="gpt-4.1-mini"
        ):
            result = analyze.analyze_clause(clause, "Termination")
        self.assertFalse(result["show_redline"])
        self.assertEqual(result["review_message"], analyze.REVIEW_MESSAGE)
        self.assertEqual(result["severity"], "high")  # severity still shown even without redline

    def test_high_confidence_shows_redline(self):
        clause = {"clause_id": "c1", "heading": "Term", "text": "x" * 250}
        payload = {
            "severity": "medium",
            "explanation": "notice period is shorter than market",
            "suggested_redline": "use 60 days notice",
            "confidence": "high",
        }
        client = FakeClient([fake_response(json.dumps(payload))])
        with patch("analyze.get_client", return_value=client), patch(
            "analyze.get_deployment", return_value="gpt-4.1-mini"
        ):
            result = analyze.analyze_clause(clause, "Termination")
        self.assertTrue(result["show_redline"])
        self.assertEqual(result["suggested_redline"], "use 60 days notice")

    def test_malformed_response_degrades_without_crashing(self):
        clause = {"clause_id": "c1", "heading": "Term", "text": "x" * 250}
        client = FakeClient([fake_response("this is not json")])
        with patch("analyze.get_client", return_value=client), patch(
            "analyze.get_deployment", return_value="gpt-4.1-mini"
        ):
            result = analyze.analyze_clause(clause, "Termination")
        self.assertEqual(result["status"], "error")
        self.assertFalse(result["show_redline"])


if __name__ == "__main__":
    unittest.main()
