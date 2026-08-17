"""Defensive JSON parsing for model output.

Models frequently wrap JSON in ```json ... ``` fences even when told not to.
Shared by classify.py and analyze.py so both degrade the same way.
"""

import json
import re

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def extract_json(text: str):
    """Strip markdown code fences and parse JSON. Raises json.JSONDecodeError
    on failure — callers are expected to catch it and degrade gracefully."""
    cleaned = _FENCE_RE.sub("", text.strip()).strip()
    return json.loads(cleaned)
