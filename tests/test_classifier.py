"""Unit tests for Transformer zero-shot clause classifier."""
import pytest
from tenderguard.models.classifier import ClauseClassifier


@pytest.fixture(scope="module")
def classifier():
    return ClauseClassifier()


def test_financial_clause_classification(classifier):
    """Verify that a clause specifying bid security and turnover classifies to Financial terms."""
    text = "The bidder must furnish an Earnest Money Deposit (EMD) of $75,000 via irrevocable bank guarantee."
    res = classifier.classify_text(text, clause_id="C_FIN")
    assert res.predicted_category == "Financial & Commercial Terms"
    assert res.confidence > 0.25
    assert "Financial & Commercial Terms" in res.all_scores


def test_deadline_clause_classification(classifier):
    """Verify that submission schedule clause classifies to Deadlines category."""
    text = "Bids and all supporting commercial proposals must be submitted no later than November 30, 2026."
    res = classifier.classify_text(text, clause_id="C_DEAD")
    assert res.predicted_category == "Deadlines & Submission Schedules"
    assert res.confidence > 0.20


def test_penalties_clause_classification(classifier):
    """Verify that liquidated damages clause classifies to Penalties category."""
    text = "Delay in completion of milestones shall attract liquidated damages at 0.5% per week, capped at 10% of contract value."
    res = classifier.classify_text(text, clause_id="C_PEN")
    assert res.predicted_category == "Penalties & Liquidated Damages"
    assert res.confidence > 0.25
