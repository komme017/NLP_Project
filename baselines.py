"""Market-standard baseline clause library.

Hand-written, not derived from any specific contract in data/test_contracts —
these represent generic, balanced market terms a fractional GC would expect
to see, independent of any one counterparty's paper. That keeps the test
contracts (including the held-out one) honest evaluation cases rather than
clauses the baselines were reverse-engineered from.

# NOTE: The spec allows semantic-similarity clause matching (sentence-
# transformers + cosine similarity) as a fallback for cases where clause-to-
# baseline matching isn't a clean 1:1 lookup. It's not used here: classify.py
# already assigns each clause one of these exact category labels, so matching
# a clause to its baseline is a direct dict lookup by category, and adding
# an embedding model would be complexity with no job left for it to do.
"""

CATEGORIES = [
    "Indemnification",
    "Limitation of Liability",
    "Termination",
    "Confidentiality",
    "Intellectual Property Assignment",
    "Governing Law",
    "Payment Terms",
    "Warranty",
    "Other",
]

# Categories with an actual baseline to compare against. "Other" has none by
# definition — it's the catch-all for clauses that don't fit the taxonomy.
BASELINES = {
    "Indemnification": {
        "clause": (
            "Each party shall indemnify, defend, and hold harmless the other party "
            "from third-party claims arising out of the indemnifying party's breach "
            "of this Agreement, gross negligence, or willful misconduct. Neither "
            "party's indemnification obligations extend to claims caused by the "
            "other party's own negligence or breach."
        ),
        "note": (
            "Balanced because it is mutual, tied to fault, and capped by carve-outs "
            "for the other side's own negligence — rather than a one-sided "
            "obligation running only from one party to the other."
        ),
    },
    "Limitation of Liability": {
        "clause": (
            "Except for breaches of confidentiality, indemnification obligations, or "
            "gross negligence/willful misconduct, neither party's aggregate liability "
            "under this Agreement shall exceed the fees paid or payable in the twelve "
            "(12) months preceding the claim, and neither party shall be liable for "
            "indirect, incidental, or consequential damages."
        ),
        "note": (
            "Balanced because the cap and the consequential-damages waiver apply "
            "equally to both parties, with standard carve-outs for the claims that "
            "should not be capped."
        ),
    },
    "Termination": {
        "clause": (
            "Either party may terminate this Agreement for convenience upon sixty "
            "(60) days' written notice, or immediately for the other party's "
            "material breach that remains uncured thirty (30) days after written "
            "notice of the breach."
        ),
        "note": (
            "Balanced because both parties have symmetric for-convenience and "
            "for-cause rights, and the cure period gives the breaching party a "
            "real chance to fix the problem before losing the contract."
        ),
    },
    "Confidentiality": {
        "clause": (
            "Each party shall protect the other's confidential information using "
            "the same degree of care it uses for its own confidential information "
            "(and no less than reasonable care), and shall not disclose it except "
            "to employees or contractors with a need to know, for a period of three "
            "(3) years following disclosure."
        ),
        "note": (
            "Balanced because obligations run both directions and the duration is "
            "bounded rather than perpetual, which is what makes it enforceable and "
            "not just aspirational."
        ),
    },
    "Intellectual Property Assignment": {
        "clause": (
            "Each party retains all right, title, and interest in its pre-existing "
            "intellectual property. Any work product created jointly in the course "
            "of this Agreement shall be jointly owned, and neither party assigns "
            "background IP to the other absent a separate written agreement."
        ),
        "note": (
            "Balanced because it protects each side's pre-existing IP by default "
            "and only allocates newly created work product, rather than assigning "
            "everything — including background IP — to one party."
        ),
    },
    "Governing Law": {
        "clause": (
            "This Agreement shall be governed by the laws of the State of Delaware, "
            "without regard to conflict-of-laws principles, and the parties consent "
            "to the exclusive jurisdiction of the state and federal courts located "
            "in Delaware."
        ),
        "note": (
            "Flagged as a baseline of convenience, not fairness — governing law is "
            "rarely 'unbalanced' in the way liability or IP terms are. What matters "
            "is that it names a neutral, well-litigated jurisdiction rather than one "
            "party's home turf with unusual local rules."
        ),
    },
    "Payment Terms": {
        "clause": (
            "Invoices are due net thirty (30) days from the invoice date. Amounts "
            "not paid when due accrue interest at 1.5% per month or the maximum "
            "rate permitted by law, whichever is lower, and either party may "
            "dispute an invoice in good faith within fifteen (15) days without "
            "being deemed in default."
        ),
        "note": (
            "Balanced because net-30 is standard commercial practice, the interest "
            "rate is capped at the legal maximum, and there is a defined, non-"
            "punitive path to dispute a bad invoice."
        ),
    },
    "Warranty": {
        "clause": (
            "Each party warrants that it has the authority to enter into this "
            "Agreement and that its performance will not violate any other "
            "agreement or applicable law. Except as expressly stated, the "
            "products/services are provided 'as is,' and all other warranties, "
            "express or implied, including merchantability and fitness for a "
            "particular purpose, are disclaimed."
        ),
        "note": (
            "Balanced because it keeps the baseline authority/no-conflict "
            "warranties every commercial contract needs while disclaiming broad "
            "implied warranties instead of one party making sweeping guarantees "
            "about the other's use case."
        ),
    },
}


def get_baseline(category: str):
    """Return {clause, note} for a category, or None if there is no baseline
    (e.g. category == "Other")."""
    return BASELINES.get(category)
