"""Unit tests for date and deadline inconsistency engine."""
from datetime import date
import pytest
from tenderguard.analysis.date_deadline import DateDeadlineAnalyzer


@pytest.fixture
def analyzer():
    return DateDeadlineAnalyzer()


def test_conflicting_submission_deadlines(analyzer):
    """Detect conflicting submission deadlines in different clauses."""
    text_a = "Bids must be delivered by 17:00 on November 15, 2026."
    text_b = "The tender submission deadline is November 30, 2026."
    
    inconsistency = analyzer.compare_dates(text_a, text_b, "C1", "C2")
    assert inconsistency is not None
    assert inconsistency.rule_id == "RULE_DATE_SUBMISSION_CONFLICT"
    assert inconsistency.priority == "High"
    assert "2026-11-15" in inconsistency.explanation
    assert "2026-11-30" in inconsistency.explanation


def test_pre_bid_after_submission_impossibility(analyzer):
    """Detect impossible sequence where pre-bid meeting occurs after tender closing."""
    text_a = "The pre-bid meeting will be held on December 05, 2026."
    text_b = "All tenders must be submitted no later than November 30, 2026."
    
    inconsistency = analyzer.compare_dates(text_a, text_b, "C3", "C4")
    assert inconsistency is not None
    assert inconsistency.rule_id == "RULE_DATE_PRE_BID_AFTER_SUBMISSION"
    assert inconsistency.priority == "High"
    assert "Pre-bid meeting" in inconsistency.explanation


def test_opening_before_submission_impossibility(analyzer):
    """Detect impossible sequence where bid opening occurs before submission deadline."""
    text_a = "Technical bid opening is scheduled for November 10, 2026."
    text_b = "Bid submission deadline is November 20, 2026."
    
    inconsistency = analyzer.compare_dates(text_a, text_b, "C5", "C6")
    assert inconsistency is not None
    assert inconsistency.rule_id == "RULE_DATE_OPENING_BEFORE_SUBMISSION"
    assert inconsistency.priority == "High"


def test_valid_chronological_sequence(analyzer):
    """Ensure properly ordered milestones produce no inconsistency findings."""
    text_a = "Pre-bid clarification meeting will be conducted on October 15, 2026."
    text_b = "Bid submission deadline is October 30, 2026."
    
    inconsistency = analyzer.compare_dates(text_a, text_b, "C7", "C8")
    assert inconsistency is None
