"""Heuristic clause segmentation.

Contracts don't split cleanly. This is a heuristic that works well enough on
the demo contracts, not a robust legal-document parser. See build spec for
the deliberate scope decision.

Strategy: try to split on numbered/ARTICLE-style heading lines. If too few
headings are found (the document doesn't use a clean numbered-line style),
fall back to splitting on blank-line paragraph breaks. Either way, fragments
under MIN_CLAUSE_CHARS are discarded as boilerplate (signature blocks,
recital fragments, page-footer junk).
"""

import logging
import re

logger = logging.getLogger(__name__)

# Two tiers, tried coarse-to-fine. A document with clean ARTICLE/SECTION-level
# headings should split there — those roughly line up with the clause
# taxonomy this product classifies against. Falling straight to numbered
# subsections (1.1, 1.2, ...) fragments things like a Definitions article
# into dozens of single-term "clauses", which is worse for classification,
# not better, and finer patterns are also more likely to false-positive on
# table-of-contents lines ("5.5 Late Payments 20").
ARTICLE_SECTION_RE = re.compile(
    r"^[ \t]*("
    r"ARTICLE\s+(?:[IVXLCDM]+|\d+)\.?[^\n]*"
    r"|SECTION\s+\d+(?:\.\d+)*\.?:?[^\n]*"
    r")[ \t]*$",
    re.MULTILINE,
)

NUMBERED_RE = re.compile(
    r"^[ \t]*("
    r"\d+(?:\.\d+){1,3}\.?[ \t]+[^\n]{0,100}"
    r"|\d+\.[ \t]+[A-Z][^\n]{0,100}"
    r")[ \t]*$",
    re.MULTILINE,
)

MIN_CLAUSE_CHARS = 200
MIN_HEADINGS_FOR_SPLIT = 3


def _split_on_pattern(text: str, pattern: re.Pattern):
    matches = list(pattern.finditer(text))
    if len(matches) < MIN_HEADINGS_FOR_SPLIT:
        return None
    segments = []
    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        segments.append((heading, body))
    return segments


def _split_on_headings(text: str):
    return _split_on_pattern(text, ARTICLE_SECTION_RE) or _split_on_pattern(text, NUMBERED_RE)


def _split_on_paragraphs(text: str):
    paras = re.split(r"\n\s*\n", text)
    segments = []
    for p in paras:
        p = p.strip()
        if not p:
            continue
        heading = p.split("\n")[0][:80].strip()
        segments.append((heading, p))
    return segments


def segment_contract(text: str):
    """Return (clauses, stats).

    clauses is a list of {clause_id, heading, text}.
    stats reports the split method used and what fraction of the source
    document survived into a clause vs. was discarded as boilerplate.
    """
    heading_segments = _split_on_headings(text)
    method = "heading"
    segments = heading_segments
    if segments is None:
        segments = _split_on_paragraphs(text)
        method = "paragraph"

    total_chars = len(text)
    kept = []
    for heading, body in segments:
        if len(body) < MIN_CLAUSE_CHARS:
            continue
        kept.append(
            {
                "clause_id": f"c{len(kept) + 1}",
                "heading": heading or f"Clause {len(kept) + 1}",
                "text": body,
            }
        )

    kept_chars = sum(len(c["text"]) for c in kept)
    retained_fraction = kept_chars / total_chars if total_chars else 0.0

    stats = {
        "method": method,
        "total_chars": total_chars,
        "num_segments_found": len(segments),
        "num_clauses_kept": len(kept),
        "retained_fraction": round(retained_fraction, 3),
        "discarded_fraction": round(1 - retained_fraction, 3),
    }

    logger.info(
        "segmented via %s split: %d/%d segments kept, %.1f%% of document retained",
        method,
        len(kept),
        len(segments),
        retained_fraction * 100,
    )

    # NOTE (observed on CUAD test contracts):
    # 1. When a contract's numbered clauses run inline within one long
    #    paragraph (common in text extracted from scanned/flattened PDFs,
    #    e.g. PrecheckHealthServicesInc's distributor agreement) rather than
    #    starting on their own line, the heading regex finds too few matches
    #    and the paragraph fallback keeps each run as one oversized
    #    "clause" instead — on that contract, 18 numbered clauses collapse
    #    into just 3 paragraph blobs. Classification/analysis downstream
    #    then sees one blob covering several unrelated clause types.
    #    Inline numbering detection would fix this; out of scope for this
    #    prototype.
    # 2. A front-matter table of contents that repeats article titles with
    #    page numbers (e.g. VERICELCORP's supply agreement) matches the same
    #    ARTICLE/SECTION heading regex as the real section headings, so the
    #    ToC becomes its own set of near-duplicate low-content "clauses"
    #    ahead of the real ones. Harmless for classification (they mostly
    #    end up "Other" or get discarded as short), but worth knowing about
    #    if headings look duplicated in the output.

    return kept, stats
