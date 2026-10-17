"""Date and deadline inconsistency detection engine."""
import re
from dataclasses import dataclass
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple
from dateutil import parser as date_parser


@dataclass
class ExtractedMilestoneDate:
    """Represents an extracted calendar date and its associated tender milestone."""
    raw_str: str
    parsed_date: date
    milestone_type: str        # e.g., 'pre_bid', 'submission', 'opening', 'delivery'
    clause_id: str


@dataclass
class DateInconsistency:
    """Represents a chronological or deadline conflict between tender clauses."""
    rule_id: str
    milestone_a: str
    milestone_b: str
    date_a: date
    date_b: date
    explanation: str
    priority: str              # "High" or "Medium"


# Date patterns
DATE_REGEXES = [
    # ISO format: 2026-11-30
    re.compile(r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b"),
    # Standard format: 30/11/2026 or 30-11-2026
    re.compile(r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b"),
    # Month name: November 30, 2026 or 30 November 2026 or Nov 30th 2026
    re.compile(
        r"\b(?:(\d{1,2}(?:st|nd|rd|th)?\s+)?(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{1,2}(?:st|nd|rd|th)?,?\s+)?(\d{4}))\b",
        re.IGNORECASE
    ),
]

# Milestone keywords
MILESTONE_PATTERNS = {
    "pre_bid": re.compile(r"\b(?:pre-bid|pre\s+bid|clarification\s+meeting|clarification\s+request|queries\s+due)\b", re.IGNORECASE),
    "submission": re.compile(r"\b(?:submission|closing\s+date|last\s+date|due\s+date|bid\s+due|tender\s+receipt|bids?\s+must\s+be\s+delivered|tenders?\s+must\s+be\s+submitted|delivered\s+by|submitted\s+no\s+later)\b", re.IGNORECASE),
    "opening": re.compile(r"\b(?:bid\s+opening|opening\s+date|tender\s+opening|technical\s+bid\s+opening)\b", re.IGNORECASE),
    "delivery": re.compile(r"\b(?:delivery\s+date|completion\s+date|go-live|commissioning\s+date)\b", re.IGNORECASE),
}


class DateDeadlineAnalyzer:
    """Analyzes chronological order and consistency across tender dates and milestone deadlines."""

    def extract_dates(self, text: str, clause_id: str = "ADHOC") -> List[ExtractedMilestoneDate]:
        """Extract explicit dates and identify associated tender milestones."""
        milestone = self._identify_milestone(text)
        if not milestone:
            return []

        extracted_dates: List[ExtractedMilestoneDate] = []
        found_spans = set()

        for regex in DATE_REGEXES:
            for match in regex.finditer(text):
                span = match.span()
                if any(s[0] <= span[0] and span[1] <= s[1] for s in found_spans):
                    continue
                found_spans.add(span)

                raw_match = match.group(0).strip()
                # Clean ordinal indicators (1st, 2nd, etc.)
                clean_str = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", raw_match, flags=re.IGNORECASE)

                try:
                    # Parse date safely
                    parsed_dt = date_parser.parse(clean_str, fuzzy=True, default=datetime(2026, 1, 1))
                    d = parsed_dt.date()
                    extracted_dates.append(
                        ExtractedMilestoneDate(
                            raw_str=raw_match,
                            parsed_date=d,
                            milestone_type=milestone,
                            clause_id=clause_id
                        )
                    )
                except (ValueError, OverflowError):
                    continue

        return extracted_dates

    def _identify_milestone(self, text: str) -> Optional[str]:
        """Classify clause text into a procurement milestone stage."""
        for m_name, pat in MILESTONE_PATTERNS.items():
            if pat.search(text):
                return m_name
        return None

    def compare_dates(
        self,
        text_a: str,
        text_b: str,
        clause_a_id: str = "A",
        clause_b_id: str = "B"
    ) -> Optional[DateInconsistency]:
        """Evaluate chronological ordering and consistency between milestone dates."""
        dates_a = self.extract_dates(text_a, clause_id=clause_a_id)
        dates_b = self.extract_dates(text_b, clause_id=clause_b_id)

        if not dates_a or not dates_b:
            return None

        for item_a in dates_a:
            for item_b in dates_b:
                m_a, d_a = item_a.milestone_type, item_a.parsed_date
                m_b, d_b = item_b.milestone_type, item_b.parsed_date

                # Check 1: Same milestone with conflicting dates (e.g. two conflicting submission deadlines)
                if m_a == m_b and d_a != d_b:
                    clean_m = m_a.replace("_", " ").title()
                    explanation = (
                        f"Potential deadline inconsistency: Clause {clause_a_id} specifies {clean_m} date as "
                        f"'{item_a.raw_str}' ({d_a.isoformat()}), whereas Clause {clause_b_id} specifies "
                        f"'{item_b.raw_str}' ({d_b.isoformat()}). Requires human review."
                    )
                    return DateInconsistency(
                        rule_id=f"RULE_DATE_{m_a.upper()}_CONFLICT",
                        milestone_a=m_a,
                        milestone_b=m_b,
                        date_a=d_a,
                        date_b=d_b,
                        explanation=explanation,
                        priority="High"
                    )

                # Check 2: Pre-bid meeting scheduled on or after submission deadline
                if (m_a == "pre_bid" and m_b == "submission" and d_a >= d_b) or \
                   (m_b == "pre_bid" and m_a == "submission" and d_b >= d_a):
                    pre_item = item_a if m_a == "pre_bid" else item_b
                    sub_item = item_b if m_a == "pre_bid" else item_a
                    explanation = (
                        f"Chronological impossibility detected: Pre-bid meeting ({pre_item.parsed_date.isoformat()} in Clause {pre_item.clause_id}) "
                        f"is scheduled on or after the bid submission deadline ({sub_item.parsed_date.isoformat()} in Clause {sub_item.clause_id}). "
                        f"Bidders cannot submit queries after the tender closes. Requires human review."
                    )
                    return DateInconsistency(
                        rule_id="RULE_DATE_PRE_BID_AFTER_SUBMISSION",
                        milestone_a="pre_bid",
                        milestone_b="submission",
                        date_a=pre_item.parsed_date,
                        date_b=sub_item.parsed_date,
                        explanation=explanation,
                        priority="High"
                    )

                # Check 3: Bid opening scheduled before submission deadline
                if (m_a == "opening" and m_b == "submission" and d_a < d_b) or \
                   (m_b == "opening" and m_a == "submission" and d_b < d_a):
                    open_item = item_a if m_a == "opening" else item_b
                    sub_item = item_b if m_a == "opening" else item_a
                    explanation = (
                        f"Chronological impossibility detected: Bid opening ({open_item.parsed_date.isoformat()} in Clause {open_item.clause_id}) "
                        f"is scheduled before the bid submission deadline ({sub_item.parsed_date.isoformat()} in Clause {sub_item.clause_id}). "
                        f"Tenders cannot be opened before submission closes. Requires human review."
                    )
                    return DateInconsistency(
                        rule_id="RULE_DATE_OPENING_BEFORE_SUBMISSION",
                        milestone_a="opening",
                        milestone_b="submission",
                        date_a=open_item.parsed_date,
                        date_b=sub_item.parsed_date,
                        explanation=explanation,
                        priority="High"
                    )

        return None
