"""Numerical inconsistency and financial discrepancy detection engine."""
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ExtractedNumerical:
    """Represents an extracted numerical or monetary value with its context and unit."""
    raw_str: str
    numeric_value: float
    unit: str                         # e.g., 'currency_usd', 'percent', 'months', 'days', 'units'
    operator: Optional[str] = None    # e.g., 'min', 'max', 'exact', 'range'
    context_topic: Optional[str] = None


@dataclass
class NumericalDiscrepancy:
    """Represents a detected numerical conflict between two clauses."""
    topic: str
    rule_id: str
    val_a: ExtractedNumerical
    val_b: ExtractedNumerical
    explanation: str
    priority: str                     # "High" or "Medium"


# Currency symbols and codes
CURRENCY_PATTERN = re.compile(
    r"""
    (?P<symbol>[\$€£₹]|USD|EUR|GBP|INR|CAD|AUD)\s*
    (?P<amount>\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s*
    (?P<scale>million|billion|lakh|crore|k)?
    """,
    re.IGNORECASE | re.VERBOSE
)

PERCENT_PATTERN = re.compile(
    r"""
    (?P<amount>\d+(?:\.\d+)?)\s*
    (?P<unit>%\s*per\s*(?:week|month|day|annum)|%|percent|percentage)
    """,
    re.IGNORECASE | re.VERBOSE
)

DURATION_PATTERN = re.compile(
    r"""
    (?P<amount>\d+)\s*
    (?P<unit>calendar\s+days|working\s+days|days|months|weeks|years)
    """,
    re.IGNORECASE | re.VERBOSE
)

OPERATOR_PATTERNS = {
    "min": re.compile(r"\b(?:at\s+least|minimum|not\s+less\s+than|no\s+less\s+than|min\.?)\b", re.IGNORECASE),
    "max": re.compile(r"\b(?:at\s+most|maximum|not\s+exceeding|up\s+to|capped\s+at|max\.?)\b", re.IGNORECASE),
    "exact": re.compile(r"\b(?:exactly|strictly|fixed\s+at)\b", re.IGNORECASE),
}

# Domain requirement topics to match for comparable requirements
REQUIREMENT_TOPICS = {
    "bid_security": re.compile(r"\b(?:earnest\s+money|emd|bid\s+security|tender\s+security|bid\s+bond)\b", re.IGNORECASE),
    "turnover": re.compile(r"\b(?:annual\s+turnover|average\s+turnover|financial\s+turnover|minimum\s+turnover)\b", re.IGNORECASE),
    "warranty": re.compile(r"\b(?:warranty|warranties|warrants?|guarantee\s+period|defects\s+liability)\b", re.IGNORECASE),
    "liquidated_damages": re.compile(r"\b(?:liquidated\s+damages|delay\s+penalty|penalty|delay\s+deduction|fines?)\b", re.IGNORECASE),
    "delivery_period": re.compile(r"\b(?:delivery\s+period|delivery\s+timeline|lead\s+time|completion\s+time)\b", re.IGNORECASE),
    "experience": re.compile(r"\b(?:years?\s+of\s+experience|relevant\s+experience|track\s+record)\b", re.IGNORECASE),
    "performance_security": re.compile(r"\b(?:performance\s+guarantee|performance\s+security|performance\s+bond)\b", re.IGNORECASE),
}

# Exception indicators (conditional carve-outs that justify differing numbers)
EXCEPTION_INDICATORS = re.compile(
    r"\b(?:for\s+msme|for\s+small\s+enterprises|lot\s+[0-9a-z]|phase\s+[0-9a-z]|tier\s+[0-9a-z]|in\s+case\s+of\s+joint\s+venture|unless\s+otherwise|option\s+[0-9a-z])\b",
    re.IGNORECASE
)


class NumericalAnalyzer:
    """Extracts numerical quantities and identifies conflicts between tender requirements."""

    def extract_numerical_entities(self, text: str) -> List[ExtractedNumerical]:
        """Extract currency, percentage, and duration quantities with surrounding context."""
        entities: List[ExtractedNumerical] = []

        # 1. Currency amounts
        for match in CURRENCY_PATTERN.finditer(text):
            amount_str = match.group("amount").replace(",", "")
            try:
                base_val = float(amount_str)
            except ValueError:
                continue

            scale = (match.group("scale") or "").lower()
            if scale in ("million", "m"):
                base_val *= 1_000_000
            elif scale in ("billion", "b"):
                base_val *= 1_000_000_000
            elif scale == "lakh":
                base_val *= 100_000
            elif scale == "crore":
                base_val *= 10_000_000
            elif scale == "k":
                base_val *= 1_000

            symbol = match.group("symbol").upper()
            operator = self._detect_operator(text, match.start())
            entities.append(
                ExtractedNumerical(
                    raw_str=match.group(0).strip(),
                    numeric_value=base_val,
                    unit=f"currency_{symbol}",
                    operator=operator
                )
            )

        # 2. Percentages
        for match in PERCENT_PATTERN.finditer(text):
            try:
                val = float(match.group("amount"))
            except ValueError:
                continue
            unit = match.group("unit").strip().lower()
            operator = self._detect_operator(text, match.start())
            entities.append(
                ExtractedNumerical(
                    raw_str=match.group(0).strip(),
                    numeric_value=val,
                    unit=f"percent_{unit}",
                    operator=operator
                )
            )

        # 3. Durations
        for match in DURATION_PATTERN.finditer(text):
            try:
                val = float(match.group("amount"))
            except ValueError:
                continue
            unit = match.group("unit").strip().lower()
            operator = self._detect_operator(text, match.start())
            entities.append(
                ExtractedNumerical(
                    raw_str=match.group(0).strip(),
                    numeric_value=val,
                    unit=f"duration_{unit}",
                    operator=operator
                )
            )

        return entities

    def _detect_operator(self, text: str, pos: int) -> str:
        """Inspect text preceding a number to detect comparison operators."""
        window_start = max(0, pos - 40)
        window = text[window_start:pos]
        for op_name, pat in OPERATOR_PATTERNS.items():
            if pat.search(window):
                return op_name
        return "exact"

    def identify_topic(self, text: str) -> Optional[str]:
        """Determine which requirement topic a clause addresses."""
        for topic, pattern in REQUIREMENT_TOPICS.items():
            if pattern.search(text):
                return topic
        return None

    def compare_clauses(
        self,
        text_a: str,
        text_b: str,
        clause_a_id: str = "A",
        clause_b_id: str = "B"
    ) -> Optional[NumericalDiscrepancy]:
        """Evaluate if two clauses describe the same requirement with conflicting numerical terms."""
        topic_a = self.identify_topic(text_a)
        topic_b = self.identify_topic(text_b)

        # Only compare if they plausibly address the same specific requirement
        if not topic_a or not topic_b or topic_a != topic_b:
            return None

        # Check for explicit conditionality / exceptions (e.g. Lot 1 vs Lot 2, MSME vs General)
        if EXCEPTION_INDICATORS.search(text_a) or EXCEPTION_INDICATORS.search(text_b):
            # Check if both have identical lot/exception tags or different ones
            match_a = EXCEPTION_INDICATORS.findall(text_a)
            match_b = EXCEPTION_INDICATORS.findall(text_b)
            if set(match_a) != set(match_b):
                # Discrepancy is likely an intentional carve-out/lot distinction
                return None

        nums_a = self.extract_numerical_entities(text_a)
        nums_b = self.extract_numerical_entities(text_b)

        if not nums_a or not nums_b:
            return None

        # Compare matching unit families
        for ent_a in nums_a:
            for ent_b in nums_b:
                if self._are_units_compatible(ent_a.unit, ent_b.unit):
                    norm_val_a = self._normalize_value(ent_a)
                    norm_val_b = self._normalize_value(ent_b)

                    if abs(norm_val_a - norm_val_b) > 1e-4:
                        # Values differ for the same requirement
                        priority = "High" if topic_a in {"bid_security", "warranty", "liquidated_damages_cap", "turnover"} else "Medium"
                        rule_id = f"RULE_NUM_{topic_a.upper()}_MISMATCH"

                        clean_topic = topic_a.replace("_", " ").title()
                        explanation = (
                            f"Potential numerical inconsistency detected in {clean_topic}. "
                            f"Clause {clause_a_id} specifies '{ent_a.raw_str}' ({ent_a.operator}) whereas "
                            f"Clause {clause_b_id} specifies '{ent_b.raw_str}' ({ent_b.operator}). "
                            f"Requires human review to verify whether this represents a drafting conflict or intended tier differentiation."
                        )

                        return NumericalDiscrepancy(
                            topic=topic_a,
                            rule_id=rule_id,
                            val_a=ent_a,
                            val_b=ent_b,
                            explanation=explanation,
                            priority=priority
                        )

        return None

    def _are_units_compatible(self, unit_a: str, unit_b: str) -> bool:
        """Check whether two units can be compared (e.g. both currencies, or both time durations)."""
        if unit_a.startswith("currency_") and unit_b.startswith("currency_"):
            return True
        if unit_a.startswith("percent_") and unit_b.startswith("percent_"):
            return True
        if unit_a.startswith("duration_") and unit_b.startswith("duration_"):
            return True
        return False

    def _normalize_value(self, ent: ExtractedNumerical) -> float:
        """Convert durations to standard baseline (days) if necessary."""
        val = ent.numeric_value
        unit = ent.unit
        if "months" in unit:
            return val * 30.0
        if "years" in unit:
            return val * 365.0
        if "weeks" in unit:
            return val * 7.0
        return val
