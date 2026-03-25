"""Data augmentation: programmatically damage clean headings.

Generates realistic OCR artifacts at scale instead of hand-crafting.
Also generates adversarial negative examples (short non-heading lines).
"""

from __future__ import annotations
import random
import re

random.seed(42)

# ---------- Clean headings to damage ----------

CLEAN_HEADINGS = [
    "DEFINITIONS",
    "SCOPE OF SERVICES",
    "COMPENSATION AND BENEFITS",
    "TERM AND TERMINATION",
    "CONFIDENTIALITY",
    "REPRESENTATIONS AND WARRANTIES",
    "INDEMNIFICATION",
    "LIMITATION OF LIABILITY",
    "GOVERNING LAW",
    "DISPUTE RESOLUTION",
    "FORCE MAJEURE",
    "ASSIGNMENT",
    "NOTICES",
    "ENTIRE AGREEMENT",
    "AMENDMENTS",
    "SEVERABILITY",
    "WAIVER",
    "COUNTERPARTS",
    "INTELLECTUAL PROPERTY",
    "DATA PROTECTION",
    "INSURANCE REQUIREMENTS",
    "COMPLIANCE WITH LAWS",
    "SUBCONTRACTING",
    "AUDIT RIGHTS",
    "Article 1 Definitions",
    "Article 2 Services",
    "Article 3 Payment Terms",
    "Article 4 Warranties",
    "Article 5 Liability",
    "Article 6 Termination",
    "Section 1 General Provisions",
    "Section 2 Employment Policies",
    "Section 3 Compensation",
    "Section 4 Conduct and Ethics",
    "Section 5 Leave Policies",
    "PART I",
    "PART II",
    "PART III",
    "CHAPTER I GENERAL PROVISIONS",
    "CHAPTER II ASSESSMENT PROCEDURES",
    "CHAPTER III PUBLIC PARTICIPATION",
    "Item 1. Business",
    "Item 1A. Risk Factors",
    "Item 7. Management Discussion",
    "Item 8. Financial Statements",
    "SCHEDULE A",
    "SCHEDULE B",
    "EXHIBIT A",
    "EXHIBIT B",
    "APPENDIX 1",
    "FORM 10-K",
    "REGULATION NO 2023-45",
    "RECITALS",
    "MASTER SERVICES AGREEMENT",
    "PURCHASE ORDER TERMS",
    "NON-COMPETE AGREEMENT",
    "SETTLEMENT TERMS",
    "MUTUAL RELEASES",
    "NON-DISPARAGEMENT",
    "QUARTERLY REPORT",
    "SEGMENT ANALYSIS",
    "OUTLOOK",
    "BACKGROUND",
    "MILESTONES",
    "RECOMMENDATIONS",
]


# ---------- Body text templates (short lines that are NOT headings) ----------

SHORT_BODY_LINES = [
    "Revenue: $45.2M",
    "Net income: $4.8M",
    "Total: $200,000,000",
    "Duration: 6 months",
    "Fee: $150,000",
    "Start date: April 1, 2024",
    "Base salary: $275,000",
    "Questions? Contact us.",
    "See attached chart.",
    "Source: Internal records",
    "Prepared by: Finance",
    "Date: October 15, 2023",
    "Page 1 of 3",
    "Page 2 of 3",
    "Page 14 of 47",
    "Dear Board Members,",
    "Dear Mr. Johnson,",
    "Best regards,",
    "Sincerely,",
    "Respectfully submitted,",
    "Sarah Chen",
    "John Smith, CEO",
    "Program Director",
    "Chief Financial Officer",
    "EMPLOYER:",
    "EMPLOYEE:",
    "WITNESS:",
    "NOTARY PUBLIC:",
    "By: ___________________________",
    "Name: ________________________",
    "Title: _______________________",
    "Date: ________________________",
    "Signature: ___________________",
    "- Valid business license",
    "- Certificate of insurance",
    "- W-9 tax form",
    "- Packing slip",
    "- Background check required",
    "1. Vendor API stability",
    "2. Resource availability",
    "3. Regulatory timeline",
    "(a) first condition",
    "(b) second condition",
    "(c) third condition",
    "(i) sub-item one",
    "(ii) sub-item two",
    "Cc: Legal, Finance",
    "Encl: 3 attachments",
    "Ref: Contract No. 2024-001",
    "Re: Project Update",
    "Figure 1: Revenue",
    "Table 2: Expenses",
    "Chart 3: Growth",
    "* Subject to change",
    "** Non-GAAP measure",
    "*** Preliminary data",
    "N/A",
    "TBD",
    "See Section 4.2 above",
    "As defined in Article 1",
    "Per the terms herein",
    "Including but not limited to:",
    "The foregoing notwithstanding:",
    "Required documentation:",
    "Key risks:",
    "All deliveries must include:",
    "Effective immediately.",
    "Confidential — Do Not Distribute",
    "[INTENTIONALLY LEFT BLANK]",
    "[SIGNATURE PAGE FOLLOWS]",
    "[REMAINDER OF PAGE LEFT BLANK]",
    "AGREED AND ACCEPTED:",
    "ACKNOWLEDGED:",
    "End of Section",
    "---",
    "* * *",
    "###",
]

LONG_BODY_LINES = [
    "The Company shall pay the Contractor for Services rendered at the rates specified in Schedule B attached hereto and incorporated by reference.",
    "Each party agrees to maintain the confidentiality of Confidential Information received from the other party during the term of this Agreement.",
    "This Agreement shall commence on the Effective Date and shall continue for a period of twelve months unless earlier terminated in accordance with the provisions hereof.",
    "All employees are expected to familiarize themselves with the contents of this handbook and comply with all policies and procedures set forth herein.",
    "The organization is committed to equal employment opportunity and does not discriminate on any basis prohibited by applicable federal, state, or local law.",
    "Compensation is reviewed annually and adjustments are made based on performance evaluations, market data, and the overall financial condition of the organization.",
    "For a period of twelve months following termination, Employee shall not directly or indirectly engage in any business that competes with the Company.",
    "If any provision of this Agreement is held invalid or unenforceable by a court of competent jurisdiction, the remaining provisions shall continue in full force and effect.",
    "Revenue for fiscal year 2023 was $2.3 billion, an increase of 15% from the prior year, driven primarily by strong demand for our cloud platform.",
    "Management expects full-year revenue to be in the range of $180M to $185M, representing growth of 18% to 21% over the prior fiscal year.",
    "The screening shall evaluate potential impacts on air quality, water resources, endangered species habitat, and cultural heritage sites within the project area.",
    "A minimum comment period of 45 days shall be provided for draft Environmental Impact Statements to ensure adequate public participation in the review process.",
    "Each invited party shall contribute capital in proportion to their ownership interest as specified in the operating agreement attached as Exhibit A.",
    "The development will include approximately 500,000 square feet of Class A office space, 200 residential units, and 80,000 square feet of retail space.",
    "Late payments shall bear interest at the rate of 1.5% per month or the maximum rate permitted by applicable law, whichever is less.",
]


def letterspace_full(text: str) -> str:
    """'HELLO' → 'H E L L O'"""
    return " ".join(text.replace(" ", "  "))


def letterspace_partial(text: str, break_prob: float = 0.4) -> str:
    """Insert spaces at random positions within words."""
    result = []
    for word in text.split():
        chars = list(word)
        new_word = [chars[0]]
        for c in chars[1:]:
            if random.random() < break_prob:
                new_word.append(" ")
            new_word.append(c)
        result.append("".join(new_word))
    return " ".join(result)


def fragment_words(text: str, frag_prob: float = 0.5) -> str:
    """Break words into fragments: 'Section' → 'Sect ion'"""
    result = []
    for word in text.split():
        if len(word) > 4 and random.random() < frag_prob:
            pos = random.randint(2, len(word) - 2)
            result.append(word[:pos] + " " + word[pos:])
        else:
            result.append(word)
    return " ".join(result)


def mixed_damage(text: str) -> str:
    """Apply inconsistent damage: some chars spaced, some not."""
    result = []
    for word in text.split():
        if random.random() < 0.3:
            result.append(" ".join(word))
        elif random.random() < 0.4 and len(word) > 4:
            pos = random.randint(2, len(word) - 2)
            result.append(word[:pos] + " " + word[pos:])
        else:
            result.append(word)
    return " ".join(result)


DAMAGE_FNS = [
    ("full_letterspace", letterspace_full),
    ("partial_letterspace", lambda t: letterspace_partial(t, 0.4)),
    ("fragment", lambda t: fragment_words(t, 0.5)),
    ("mixed", mixed_damage),
    ("light_fragment", lambda t: fragment_words(t, 0.25)),
]


def generate_augmented_docs(n_docs: int = 20) -> list[list[tuple[str, bool]]]:
    """Generate augmented documents with OCR-damaged headings + adversarial body."""
    docs = []
    heading_pool = list(CLEAN_HEADINGS)
    body_short_pool = list(SHORT_BODY_LINES)
    body_long_pool = list(LONG_BODY_LINES)

    for doc_i in range(n_docs):
        doc: list[tuple[str, bool]] = []
        random.shuffle(heading_pool)
        random.shuffle(body_short_pool)

        # Pick damage strategy for this doc
        if doc_i < 4:
            # Clean docs (no damage)
            damage_fn = None
        elif doc_i < 8:
            # Mixed: some damaged, some clean
            damage_fn = "mixed_doc"
        else:
            # Pick a specific damage type
            _, damage_fn_actual = DAMAGE_FNS[doc_i % len(DAMAGE_FNS)]
            damage_fn = damage_fn_actual

        n_sections = random.randint(3, 6)
        for sec_i in range(n_sections):
            heading = heading_pool[sec_i % len(heading_pool)]

            if damage_fn is None:
                damaged_heading = heading
            elif damage_fn == "mixed_doc":
                if random.random() < 0.5:
                    _, fn = random.choice(DAMAGE_FNS)
                    damaged_heading = fn(heading)
                else:
                    damaged_heading = heading
            else:
                damaged_heading = damage_fn(heading)

            if sec_i > 0:
                doc.append(("", False))
            doc.append((damaged_heading, True))
            doc.append(("", False))

            # Add body: mix of long and short non-heading lines
            n_body = random.randint(2, 5)
            for _ in range(n_body):
                if random.random() < 0.3:
                    line = body_short_pool[random.randint(0, len(body_short_pool) - 1)]
                else:
                    line = body_long_pool[random.randint(0, len(body_long_pool) - 1)]
                doc.append((line, False))

            # Occasionally insert tricky short non-heading lines
            if random.random() < 0.4:
                doc.append(("", False))
                tricky = body_short_pool[random.randint(0, len(body_short_pool) - 1)]
                doc.append((tricky, False))

        docs.append(doc)

    return docs
