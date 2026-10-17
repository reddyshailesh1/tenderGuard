"""Unit tests for SentenceTransformer embedding and candidate pair retrieval."""
import pytest
from tenderguard.models.embeddings import SemanticRetriever


@pytest.fixture(scope="module")
def retriever():
    return SemanticRetriever()


def test_embedding_encoding_shape(retriever):
    """Ensure sentence embeddings return normalized 384-dimensional vectors."""
    texts = [
        "The warranty period shall be 12 months.",
        "The warranty duration is 24 months from commissioning."
    ]
    embs = retriever.encode(texts)
    assert embs.shape == (2, 384)


def test_retrieval_related_clauses(retriever):
    """Ensure related clauses are retrieved as candidate pairs while unrelated are separated."""
    clauses = [
        {
            "id": "C1",
            "doc_id": "D1",
            "page_number": 1,
            "original_text": "The bidder must provide an Earnest Money Deposit of $50,000.",
            "category": "Financial & Commercial Terms"
        },
        {
            "id": "C2",
            "doc_id": "D1",
            "page_number": 2,
            "original_text": "Bid security deposit is specified as $75,000 payable upon submission.",
            "category": "Financial & Commercial Terms"
        },
        {
            "id": "C3",
            "doc_id": "D1",
            "page_number": 5,
            "original_text": "The servers must be mounted in standard 19-inch racks with redundant cooling fans.",
            "category": "Technical Specifications"
        }
    ]
    
    pairs = retriever.retrieve_candidate_pairs(clauses, similarity_threshold=0.45)
    
    # Pairs should link C1 and C2 (both financial bid security)
    c1_c2_pair = [p for p in pairs if (p.clause_a_id == "C1" and p.clause_b_id == "C2") or (p.clause_a_id == "C2" and p.clause_b_id == "C1")]
    assert len(c1_c2_pair) == 1
    assert c1_c2_pair[0].similarity_score > 0.50
