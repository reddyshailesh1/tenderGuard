"""Unit tests for HybridEngine combining NLI, embeddings, and rules."""
import pytest
from tenderguard.analysis.hybrid_engine import HybridEngine
from tenderguard.models.embeddings import CandidatePair
from tenderguard.models.nli_engine import NLIPrediction


@pytest.fixture
def hybrid_engine():
    return HybridEngine()


def test_hybrid_contradiction_and_numerical_synthesis(hybrid_engine):
    """Ensure high NLI contradiction combined with numerical conflict yields hybrid high-priority finding."""
    pair = CandidatePair(
        doc_id="DOC_HYB_01",
        clause_a_id="C10",
        clause_b_id="C20",
        clause_a_text="The warranty period shall be strictly 12 months from delivery.",
        clause_b_text="The warranty shall be 24 months from acceptance.",
        similarity_score=0.88,
        embedding_model="all-MiniLM-L6-v2",
        retrieval_method="SentenceTransformer-Cosine",
        page_a=2,
        page_b=7
    )
    
    nli_pred = NLIPrediction(
        clause_a_id="C10",
        clause_b_id="C20",
        predicted_label="contradiction",
        contradiction_prob=0.92,
        entailment_prob=0.02,
        neutral_prob=0.06,
        is_contradiction=True,
        model_identifier="typeform/distilbert-base-uncased-mnli"
    )
    
    finding = hybrid_engine.analyze_pair(pair, nli_pred, "FND_HYB_001")
    assert finding is not None
    assert finding.finding_type == "Hybrid Contradiction & Numerical Conflict"
    assert finding.priority == "High"
    assert "Page 2, Page 7" in finding.page_numbers
    assert finding.semantic_similarity == 0.88
    assert finding.nli_contradiction_score == 0.92
    assert "RULE_NUM_WARRANTY_MISMATCH + NLI_CONTRADICTION" in finding.rule_triggered
    assert "Potential critical contradiction" in finding.explanation


def test_pure_nli_semantic_contradiction(hybrid_engine):
    """Ensure non-numerical logical contradictions are captured via NLI."""
    pair = CandidatePair(
        doc_id="DOC_HYB_02",
        clause_a_id="C30",
        clause_b_id="C40",
        clause_a_text="Subcontracting of any part of the service is strictly prohibited.",
        clause_b_text="The contractor may freely engage approved subcontractors for service delivery.",
        similarity_score=0.75,
        embedding_model="all-MiniLM-L6-v2",
        retrieval_method="SentenceTransformer-Cosine",
        page_a=3,
        page_b=4
    )
    
    nli_pred = NLIPrediction(
        clause_a_id="C30",
        clause_b_id="C40",
        predicted_label="contradiction",
        contradiction_prob=0.85,
        entailment_prob=0.05,
        neutral_prob=0.10,
        is_contradiction=True,
        model_identifier="typeform/distilbert-base-uncased-mnli"
    )
    
    finding = hybrid_engine.analyze_pair(pair, nli_pred, "FND_HYB_002")
    assert finding is not None
    assert finding.finding_type == "NLI Semantic Contradiction"
    assert finding.priority == "High"
    assert finding.rule_triggered == "RULE_NLI_LOGICAL_CONTRADICTION"
    assert "subcontracting" in finding.clause_a_text.lower()


def test_complementary_entailed_clauses_produce_no_finding(hybrid_engine):
    """Ensure consistent and complementary clauses produce no spurious risk findings."""
    pair = CandidatePair(
        doc_id="DOC_HYB_03",
        clause_a_id="C50",
        clause_b_id="C60",
        clause_a_text="The contractor must deliver the equipment within 30 days of contract signing.",
        clause_b_text="Delivery must be completed within 30 calendar days.",
        similarity_score=0.92,
        embedding_model="all-MiniLM-L6-v2",
        retrieval_method="SentenceTransformer-Cosine",
        page_a=1,
        page_b=2
    )
    
    nli_pred = NLIPrediction(
        clause_a_id="C50",
        clause_b_id="C60",
        predicted_label="entailment",
        contradiction_prob=0.03,
        entailment_prob=0.94,
        neutral_prob=0.03,
        is_contradiction=False,
        model_identifier="typeform/distilbert-base-uncased-mnli"
    )
    
    finding = hybrid_engine.analyze_pair(pair, nli_pred, "FND_HYB_003")
    assert finding is None, "Entailed or consistent clauses must not produce contradiction findings."
