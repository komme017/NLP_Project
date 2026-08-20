"""Redline: contract clause review prototype. Single-page Streamlit app.

Upload a contract -> segment into clauses -> classify each clause -> compare
against a market-standard baseline -> show flagged deviations.
"""

import logging

import streamlit as st

from analyze import analyze_clauses
from baselines import CATEGORIES
from classify import classify_clauses
from costs import CostTracker
from ingest import load_contract
from segment import segment_contract

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Redline — Contract Review Prototype", layout="wide")

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "none": 3, None: 4}
SEVERITY_COLOR = {
    "high": "red",
    "medium": "orange",
    "low": "blue",
    "none": "green",
    None: "gray",
}
SEVERITY_LABEL = {
    "high": "HIGH",
    "medium": "MEDIUM",
    "low": "LOW",
    "none": "NONE",
    None: "N/A",
}


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
    try:
        categories = classify_clauses(clauses, cost_tracker)
    except RuntimeError as e:
        # classify.py raises loudly (rather than quietly returning "Other"
        # for everything) when the local CUAD model can't be loaded -- show
        # that clearly instead of a raw traceback.
        status.update(label="Classification failed.", state="error")
        st.error(str(e))
        return None

    status.write("Comparing each clause against market baselines...")
    results = analyze_clauses(clauses, categories, cost_tracker)

    run_record = cost_tracker.log_run(contract_name=filename, extra={"segmentation": seg_stats})

    status.update(label="Analysis complete.", state="complete")

    return {
        "results": results,
        "seg_stats": seg_stats,
        "cost_totals": cost_tracker.totals(),
        "run_record": run_record,
    }


def render_summary(results):
    n_total = len(results)
    n_high = sum(1 for r in results if r["severity"] == "high")
    n_review = sum(1 for r in results if not r["show_redline"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Clauses analyzed", n_total)
    c2.metric("Flagged high severity", n_high)
    c3.metric("Flagged for attorney review", n_review)


def render_results(results):
    sort_order = st.selectbox(
        "Sort by",
        ["Severity (high first)", "Document order"],
        index=0,
    )
    ordered = results
    if sort_order == "Severity (high first)":
        ordered = sorted(results, key=lambda r: SEVERITY_ORDER.get(r["severity"], 4))

    for r in ordered:
        color = SEVERITY_COLOR.get(r["severity"])
        label = SEVERITY_LABEL.get(r["severity"])
        title = f":{color}[**{label}**] — {r['heading'][:80]}  ·  _{r['category']}_"
        with st.expander(title):
            st.markdown(f"**Category:** {r['category']}")
            st.markdown(f"**Explanation:** {r['explanation']}")
            if r["show_redline"]:
                st.markdown("**Suggested redline:**")
                st.code(r["suggested_redline"], language=None)
            else:
                st.warning(r["review_message"])
            st.caption(f"Confidence: {r['confidence']} · Status: {r['status']}")


def render_costs(cost_totals):
    st.divider()
    st.subheader("Cost & usage")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("API calls", cost_totals["num_calls"])
    c2.metric("Prompt tokens", f"{cost_totals['prompt_tokens']:,}")
    c3.metric("Completion tokens", f"{cost_totals['completion_tokens']:,}")
    c4.metric("Estimated cost", f"${cost_totals['estimated_cost_usd']:.4f}")
    st.caption(
        "Covers analysis (gpt-4.1-mini) calls only — classification runs locally "
        "against a CUAD-fine-tuned model, no API cost. Estimated using published "
        "commercial per-token rates (see costs.py). Azure access for this project "
        "is institutional/free."
    )


def main():
    st.title("Redline — Contract Review Prototype")
    st.caption(
        "Upload an inbound third-party contract to flag clauses that deviate "
        "from market-standard terms, with plain-language explanations and "
        "suggested redlines."
    )

    with st.expander("Taxonomy covered"):
        st.write(", ".join(CATEGORIES))

    uploaded = st.file_uploader("Upload a contract", type=["txt", "pdf"])

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
        render_results(run_output["results"])
        render_costs(run_output["cost_totals"])


if __name__ == "__main__":
    main()
