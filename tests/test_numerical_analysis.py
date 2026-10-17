"""Unit tests for numerical inconsistency engine."""
import pytest
from tenderguard.analysis.numerical import NumericalAnalyzer


@pytest.fixture
def analyzer():
    return NumericalAnalyzer()


def test_bid_security_currency_mismatch(analyzer):
    """Detect conflicting earnest money deposit amounts for the same requirement."""
    text_a = "The bidder must furnish an Earnest Money Deposit of $50,000 with the proposal."
    text_b = "Clause 14.2 Bid Security: The EMD required is $75,000 payable by bank draft."
    
    conflict = analyzer.compare_clauses(text_a, text_b, "C1", "C2")
    assert conflict is not None
    assert conflict.topic == "bid_security"
    assert conflict.rule_id == "RULE_NUM_BID_SECURITY_MISMATCH"
    assert conflict.priority == "High"
    assert "50,000" in conflict.explanation and "75,000" in conflict.explanation


def test_warranty_duration_mismatch(analyzer):
    """Detect conflicting warranty durations."""
    text_a = "The contractor warrants all equipment for a period of 12 months from final acceptance."
    text_b = "All hardware must include a comprehensive on-site warranty of 24 months."
    
    conflict = analyzer.compare_clauses(text_a, text_b, "C3", "C4")
    assert conflict is not None
    assert conflict.topic == "warranty"
    assert "12" in conflict.val_a.raw_str and "24" in conflict.val_b.raw_str


def test_penalty_cap_mismatch(analyzer):
    """Detect conflicting liquidated damages cap."""
    text_a = "Liquidated damages shall be capped at a maximum of 5% of total contract value."
    text_b = "Delay penalty is capped at 10% of total order value."
    
    conflict = analyzer.compare_clauses(text_a, text_b, "C5", "C6")
    assert conflict is not None
    assert conflict.topic == "liquidated_damages"
    assert conflict.val_a.numeric_value == 5.0
    assert conflict.val_b.numeric_value == 10.0


def test_exception_carveout_suppression(analyzer):
    """Ensure genuine tier or lot distinctions (e.g. MSME vs general) are not flagged as drafting conflicts."""
    text_a = "The earnest money deposit shall be $50,000 for standard commercial bidders."
    text_b = "For MSME registered enterprises, the earnest money deposit shall be $10,000."
    
    conflict = analyzer.compare_clauses(text_a, text_b, "C7", "C8")
    assert conflict is None, "Legitimate MSME exception should not trigger a numerical inconsistency finding."


def test_unrelated_numerical_clauses(analyzer):
    """Ensure clauses with numbers describing completely different topics are not compared."""
    text_a = "The bidder must have completed at least 3 similar projects in the last 5 years."
    text_b = "The liquidated damages shall be 0.5% per week of delay."
    
    conflict = analyzer.compare_clauses(text_a, text_b, "C9", "C10")
    assert conflict is None
