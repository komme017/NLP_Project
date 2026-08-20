"""Redline: contract clause review prototype — document view variant.

Same pipeline as app.py, but instead of a flat list of expandable results,
this shows the full contract text with flagged clauses highlighted inline.
Flags are picked from a dropdown in the sidebar; picking one scrolls the
document pane to that clause and outlines it.
"""

import html
import logging

import streamlit as st
import streamlit.components.v1 as components

from analyze import analyze_clauses
from baselines import CATEGORIES
from classify import classify_clauses
from costs import CostTracker
from ingest import load_contract
from segment import segment_contract

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Redline — Document View", layout="wide")

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "none": 3, None: 4}
SEVERITY_LABEL = {"high": "HIGH", "medium": "MEDIUM", "low": "LOW", "none": "NONE", None: "N/A"}
SEVERITY_COLOR = {
    "high": "red",
    "medium": "orange",
    "low": "blue",
    "none": "green",
    None: "gray",
}
# Highlight background + accent border per severity, used in the inline
# document view. Kept light/translucent so contract text stays readable
# underneath in both light and dark Streamlit themes.
SEVERITY_HIGHLIGHT_BG = {
    "high": "rgba(220, 38, 38, 0.22)",
    "medium": "rgba(234, 88, 12, 0.20)",
    "low": "rgba(37, 99, 235, 0.18)",
    "none": "rgba(22, 163, 74, 0.14)",
    None: "rgba(107, 114, 128, 0.20)",
}
SEVERITY_HIGHLIGHT_BORDER = {
    "high": "#dc2626",
    "medium": "#ea580c",
    "low": "#2563eb",
    "none": "#16a34a",
    None: "#6b7280",
}
DOC_VIEW_HEIGHT_PX = 700


def run_pipeline(file_bytes: bytes, filename: str):
    cost_tracker = CostTracker()

    status = st.status("Analyzing contract...", expanded=True)

    status.write("Reading document...")
    text = load_contract(file_bytes, filename)

    status.write("Segmenting into clauses...")
    clauses, seg_stats = segment_contract(text)
    status.write(
        f"Found {seg_stats['num_clauses_kept']} clauses "
        f"({seg_stats['method']} split, {seg_stats['retained_fraction']*100:.0f}% of document retained)."
    )

    if not clauses:
        status.update(label="No clauses found.", state="error")
        return None

    status.write(f"Classifying {len(clauses)} clauses...")
    categories = classify_clauses(clauses, cost_tracker)

    status.write("Comparing each clause against market baselines...")
    results = analyze_clauses(clauses, categories, cost_tracker)

    run_record = cost_tracker.log_run(contract_name=filename, extra={"segmentation": seg_stats})

    status.update(label="Analysis complete.", state="complete")

    return {
        "text": text,
        "clauses": clauses,
        "results": results,
        "seg_stats": seg_stats,
        "cost_totals": cost_tracker.totals(),
        "run_record": run_record,
    }


def is_flag(result: dict) -> bool:
    """A clause is worth surfacing in the flag picker unless it's a clean,
    fully-analyzed match against the baseline (severity "none", status ok)."""
    return not (result["status"] == "ok" and result["severity"] == "none")


def render_summary(results):
    n_total = len(results)
    n_high = sum(1 for r in results if r["severity"] == "high")
    n_review = sum(1 for r in results if not r["show_redline"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Clauses analyzed", n_total)
    c2.metric("Flagged high severity", n_high)
    c3.metric("Flagged for attorney review", n_review)


def build_highlighted_document_html(text: str, clauses: list, results_by_id: dict, selected_id: str) -> str:
    """Splice highlight spans for each clause into the full document text,
    in original document order, escaping everything that isn't our own
    markup. Text between/around clauses (boilerplate the segmenter
    discarded, or gaps between spans) renders as plain text so the whole
    document is visible, not just the flagged parts."""
    ordered = sorted(clauses, key=lambda c: c["start"])
    pieces = []
    cursor = 0
    for clause in ordered:
        start, end = clause["start"], clause["end"]
        if start > cursor:
            pieces.append(html.escape(text[cursor:start]))

        result = results_by_id.get(clause["clause_id"])
        severity = result["severity"] if result else None
        bg = SEVERITY_HIGHLIGHT_BG.get(severity, SEVERITY_HIGHLIGHT_BG[None])
        border = SEVERITY_HIGHLIGHT_BORDER.get(severity, SEVERITY_HIGHLIGHT_BORDER[None])
        is_selected = clause["clause_id"] == selected_id
        selected_style = "outline: 2px solid #eab308; outline-offset: 1px;" if is_selected else ""

        span_text = html.escape(text[start:end])
        pieces.append(
            f'<span id="clause-{clause["clause_id"]}" '
            f'style="background-color: {bg}; border-left: 3px solid {border}; '
            f'padding: 1px 3px; border-radius: 2px; {selected_style}" '
            f'title="{html.escape(clause["heading"][:100])}">{span_text}</span>'
        )
        cursor = end

    if cursor < len(text):
        pieces.append(html.escape(text[cursor:]))

    body = "".join(pieces)
    return (
        f'<div style="height: {DOC_VIEW_HEIGHT_PX}px; overflow-y: auto; '
        f'padding: 16px; border: 1px solid rgba(128,128,128,0.3); border-radius: 8px; '
        f'white-space: pre-wrap; font-family: Georgia, \'Times New Roman\', serif; '
        f'font-size: 14px; line-height: 1.6;">{body}</div>'
    )


def scroll_to_clause(clause_id: str):
    """Scroll the document pane to the given clause's highlight span.
    st.components.v1.html renders in a same-origin iframe, so it can reach
    into window.parent.document to scroll the main page — the element may
    not exist yet on the first script tick, hence the short retry loop."""
    components.html(
        f"""
        <script>
        (function scrollWhenReady(tries) {{
            var el = window.parent.document.getElementById("clause-{clause_id}");
            if (el) {{
                el.scrollIntoView({{behavior: "smooth", block: "center"}});
            }} else if (tries > 0) {{
                setTimeout(function() {{ scrollWhenReady(tries - 1); }}, 100);
            }}
        }})(20);
        </script>
        """,
        height=0,
    )


def render_flags_sidebar(run_output) -> str:
    """Flag picker + detail panel, in the sidebar. Returns the selected
    clause_id (or None if there's nothing to flag), which the document view
    uses to decide what to highlight/scroll to."""
    results = run_output["results"]
    results_by_id = {r["clause_id"]: r for r in results}

    flags = [r for r in results if is_flag(r)]
    flags = sorted(flags, key=lambda r: SEVERITY_ORDER.get(r["severity"], 4))

    st.sidebar.subheader("Flags")
    if not flags:
        st.sidebar.success("No deviations flagged — every clause matched its baseline.")
        return None

    options = [r["clause_id"] for r in flags]
    labels = {
        r["clause_id"]: f"[{SEVERITY_LABEL[r['severity']]}] {r['heading'][:45]} · {r['category']}"
        for r in flags
    }
    selected_id = st.sidebar.selectbox(
        f"{len(flags)} flagged clause(s) — pick one to jump to it",
        options=options,
        format_func=lambda cid: labels[cid],
    )

    selected = results_by_id[selected_id]
    st.sidebar.markdown(
        f":{SEVERITY_COLOR.get(selected['severity'])}[**{SEVERITY_LABEL[selected['severity']]}**] "
        f"· {selected['category']}"
    )
    st.sidebar.markdown(f"**Explanation:** {selected['explanation']}")
    if selected["show_redline"]:
        st.sidebar.markdown("**Suggested redline:**")
        st.sidebar.code(selected["suggested_redline"], language=None)
    else:
        st.sidebar.warning(selected["review_message"])
    st.sidebar.caption(f"Confidence: {selected['confidence']} · Status: {selected['status']}")

    return selected_id


def render_document(run_output, selected_id: str):
    results_by_id = {r["clause_id"]: r for r in run_output["results"]}
    st.markdown("**Document** (scroll, or pick a flag in the sidebar to jump)")

    doc_html = build_highlighted_document_html(
        run_output["text"], run_output["clauses"], results_by_id, selected_id
    )
    scroll_script = ""
    if selected_id is not None:
        scroll_script = f"""
        <script>
        (function scrollWhenReady(tries) {{
            var el = document.getElementById("clause-{selected_id}");
            if (el) {{
                el.scrollIntoView({{behavior: "smooth", block: "center"}});
            }} else if (tries > 0) {{
                setTimeout(function() {{ scrollWhenReady(tries - 1); }}, 100);
            }}
        }})(20);
        </script>
        """

    components.html(doc_html + scroll_script, height=DOC_VIEW_HEIGHT_PX + 20, scrolling=True)


def render_costs(cost_totals):
    st.divider()
    st.subheader("Cost & usage")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("API calls", cost_totals["num_calls"])
    c2.metric("Prompt tokens", f"{cost_totals['prompt_tokens']:,}")
    c3.metric("Completion tokens", f"{cost_totals['completion_tokens']:,}")
    c4.metric("Estimated cost", f"${cost_totals['estimated_cost_usd']:.4f}")
    st.caption(
        "Estimated using published commercial per-token rates (see costs.py). "
        "Azure access for this project is institutional/free."
    )


def main():
    st.title("Redline - Contract Review Assistant")
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            background-color: #001f3f;
        }
        [data-testid="stSidebar"] * {
            color: #ffffff;
        }
        /* Selectbox closed control (the white dropdown box) */
        [data-testid="stSidebar"] div[data-baseweb="select"] * {
            color: #000000 !important;
        }
        /* Selectbox open dropdown list (renders outside the sidebar div) */
        div[data-baseweb="popover"] li {
            color: #000000 !important;
        }
        /* Suggested redline code block */
        [data-testid="stSidebar"] code,
        [data-testid="stSidebar"] pre,
        [data-testid="stSidebar"] pre *,
        [data-testid="stSidebar"] code * {
            color: #000000 !important;
        }
        [data-testid="stSidebar"] [data-testid="stCodeBlock"],
        [data-testid="stSidebar"] [data-testid="stCodeBlock"] * {
            color: #000000 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Upload an inbound third-party contract to see the full text with "
        "flagged clauses highlighted inline. Pick a flag from the sidebar "
        "dropdown to scroll the document to it."
    )

    with st.expander("Taxonomy Covered"):
        st.write(", ".join(CATEGORIES))

    uploaded = st.file_uploader("Upload Contract", type=["txt", "pdf"])

    if uploaded is not None and st.button("Analyze", type="primary"):
        file_bytes = uploaded.read()
        run_output = run_pipeline(file_bytes, uploaded.name)
        if run_output is not None:
            st.session_state["run_output"] = run_output
            st.session_state["contract_name"] = uploaded.name

    run_output = st.session_state.get("run_output")
    if run_output is not None:
        st.divider()
        st.subheader(f"Results — {st.session_state.get('contract_name', '')}")
        render_summary(run_output["results"])
        selected_id = render_flags_sidebar(run_output)
        render_document(run_output, selected_id)
        render_costs(run_output["cost_totals"])


if __name__ == "__main__":
    main()
