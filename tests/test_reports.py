"""Unit tests for report generation (PDF, CSV, JSON)."""
import json
import pytest
from tenderguard.db.session import DatabaseManager
from tenderguard.export.report_generator import ReportGenerator


@pytest.fixture
def populated_db(temp_db: DatabaseManager):
    """Seed database with realistic document, clauses, and findings."""
    doc_id = "DOC_REP_001"
    temp_db.save_document({
        "id": doc_id,
        "filename": "municipal_water_tender.pdf",
        "file_hash": "rep_hash_12345",
        "total_pages": 4,
        "file_size_bytes": 102400,
        "scanned_pages_count": 0,
        "status": "analyzed"
    })
    
    clauses = [
        {
            "id": f"{doc_id}_C001",
            "doc_id": doc_id,
            "page_number": 1,
            "clause_number": "1.1",
            "original_text": "The Earnest Money Deposit is $50,000.",
            "char_start": 0,
            "char_end": 37,
            "category": "Financial & Commercial Terms",
            "category_confidence": 0.92,
            "classification_model": "test-model"
        },
        {
            "id": f"{doc_id}_C002",
            "doc_id": doc_id,
            "page_number": 3,
            "clause_number": "4.2",
            "original_text": "Bid security deposit required is $80,000.",
            "char_start": 0,
            "char_end": 41,
            "category": "Financial & Commercial Terms",
            "category_confidence": 0.94,
            "classification_model": "test-model"
        }
    ]
    temp_db.save_clauses(clauses)
    
    findings = [
        {
            "id": "FND_REP_001",
            "doc_id": doc_id,
            "finding_type": "Numerical Inconsistency",
            "clause_a_id": f"{doc_id}_C001",
            "clause_b_id": f"{doc_id}_C002",
            "clause_a_text": "The Earnest Money Deposit is $50,000.",
            "clause_b_text": "Bid security deposit required is $80,000.",
            "page_numbers": "Page 1, Page 3",
            "semantic_similarity": 0.89,
            "nli_label": "contradiction",
            "nli_contradiction_score": 0.88,
            "nli_entailment_score": 0.02,
            "nli_neutral_score": 0.10,
            "rule_triggered": "RULE_NUM_BID_SECURITY_MISMATCH",
            "explanation": "Potential inconsistency in bid security ($50,000 vs $80,000). Requires human review.",
            "priority": "High",
            "model_identifier": "Hybrid-ST-NLI-Rule"
        }
    ]
    temp_db.save_findings(findings)
    temp_db.upsert_review(
        finding_id="FND_REP_001",
        status="Confirmed Risk",
        reviewer_comment="Verified conflicting EMD clauses in Section 1 and Section 4.",
        reviewer_name="Lead Auditor"
    )
    
    return temp_db, doc_id


def test_csv_report_generation(populated_db):
    """Ensure CSV export produces well-formed tabular data with headers."""
    db, doc_id = populated_db
    generator = ReportGenerator(db)
    csv_text = generator.export_csv(doc_id)
    
    assert "finding_id,doc_id,finding_type" in csv_text
    assert "FND_REP_001" in csv_text
    assert "Confirmed Risk" in csv_text
    assert "Lead Auditor" in csv_text


def test_json_report_generation(populated_db):
    """Ensure JSON export produces valid JSON structure with full metadata."""
    db, doc_id = populated_db
    generator = ReportGenerator(db)
    json_text = generator.export_json(doc_id)
    
    data = json.loads(json_text)
    assert data["document"]["id"] == doc_id
    assert data["clauses_count"] == 2
    assert data["findings_count"] == 1
    assert data["findings"][0]["review_status"] == "Confirmed Risk"


def test_pdf_report_generation(populated_db):
    """Ensure ReportLab generates a valid PDF document with %PDF- header."""
    db, doc_id = populated_db
    generator = ReportGenerator(db)
    pdf_bytes = generator.export_pdf(doc_id)
    
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF-")
