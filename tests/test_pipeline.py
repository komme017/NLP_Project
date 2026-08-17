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


class TestClassify(unittest.TestCase):
    def test_valid_batch_response_used_directly(self):
        clauses = [
            {"clause_id": "c1", "heading": "Term", "text": "x" * 250},
            {"clause_id": "c2", "heading": "Governing Law", "text": "y" * 250},
        ]
        client = FakeClient([fake_response(json.dumps({"c1": "Termination", "c2": "Governing Law"}))])
        with patch("classify.get_client", return_value=client), patch(
            "classify.get_deployment", return_value="gpt-4.1-mini"
        ):
            result = classify.classify_clauses(clauses)
        self.assertEqual(result, {"c1": "Termination", "c2": "Governing Law"})

    def test_garbage_batch_falls_back_to_per_clause_then_other(self):
        clauses = [{"clause_id": "c1", "heading": "???", "text": "x" * 250}]
        # batch call returns unparseable garbage, single-clause retry also garbage
        client = FakeClient([fake_response("not json at all"), fake_response("Not A Real Category")])
        with patch("classify.get_client", return_value=client), patch(
            "classify.get_deployment", return_value="gpt-4.1-mini"
        ):
            result = classify.classify_clauses(clauses)
        self.assertEqual(result, {"c1": "Other"})

    def test_off_taxonomy_category_falls_back_to_retry(self):
        clauses = [{"clause_id": "c1", "heading": "Odd", "text": "x" * 250}]
        client = FakeClient(
            [
                fake_response(json.dumps({"c1": "Made Up Category"})),
                fake_response("Confidentiality"),
            ]
        )
        with patch("classify.get_client", return_value=client), patch(
            "classify.get_deployment", return_value="gpt-4.1-mini"
        ):
            result = classify.classify_clauses(clauses)
        self.assertEqual(result, {"c1": "Confidentiality"})


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
