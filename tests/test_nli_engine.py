"""Unit tests for Natural Language Inference engine."""
import pytest
from tenderguard.models.nli_engine import NLIEngine


@pytest.fixture(scope="module")
def nli_engine():
    return NLIEngine()


def test_nli_contradiction_detection(nli_engine):
    """Ensure contradictory statements yield high contradiction probability."""
    premise = "The warranty period shall be exactly 12 months from final acceptance."
    hypothesis = "The warranty period shall be 36 months from final acceptance."
    
    pred = nli_engine.predict_pair(premise, hypothesis, "C1", "C2")
    assert pred.is_contradiction is True
    assert pred.contradiction_prob > 0.50
    assert pred.predicted_label == "contradiction"


def test_nli_entailment_detection(nli_engine):
    """Ensure complementary/entailed clauses yield low contradiction probability."""
    premise = "The contractor must deliver the equipment within 30 days of contract signing."
    hypothesis = "Delivery of equipment is required within one month of contract execution."
    
    pred = nli_engine.predict_pair(premise, hypothesis, "C3", "C4")
    assert pred.is_contradiction is False
    assert pred.contradiction_prob < 0.30


def test_nli_neutral_detection(nli_engine):
    """Ensure independent clauses yield neutral prediction without false contradiction."""
    premise = "The supplier must provide 100 enterprise laptop units."
    hypothesis = "The contractor shall provide on-site technical support during business hours."
    
    pred = nli_engine.predict_pair(premise, hypothesis, "C5", "C6")
    assert pred.is_contradiction is False
    assert pred.predicted_label == "neutral" or pred.neutral_prob > 0.40
