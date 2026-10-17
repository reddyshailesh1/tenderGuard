"""Unit tests for SQLite database layer."""
import pytest
from tenderguard.db.session import DatabaseManager


def test_database_initialization(temp_db: DatabaseManager):
    """Ensure all required tables and indexes are created successfully."""
    with temp_db.get_connection() as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {t["name"] for t in tables}
        
    required_tables = {"documents", "pages", "clauses", "candidate_pairs", "findings", "reviews", "evaluations"}
    assert required_tables.issubset(table_names), f"Missing tables: {required_tables - table_names}"


def test_document_lifecycle(temp_db: DatabaseManager):
    """Test saving, retrieving, updating status, and deleting documents."""
    doc_data = {
        "id": "DOC_TEST_001",
        "filename": "tender_spec_2026.pdf",
        "file_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "total_pages": 15,
        "file_size_bytes": 102400,
        "scanned_pages_count": 0,
        "status": "uploaded",
        "error_message": None,
    }
    
    doc_id = temp_db.save_document(doc_data)
    assert doc_id == "DOC_TEST_001"
    
    retrieved = temp_db.get_document("DOC_TEST_001")
    assert retrieved is not None
    assert retrieved["filename"] == "tender_spec_2026.pdf"
    assert retrieved["total_pages"] == 15
    assert retrieved["status"] == "uploaded"
    
    temp_db.update_document_status("DOC_TEST_001", "analyzed", scanned_pages=1)
    updated = temp_db.get_document("DOC_TEST_001")
    assert updated["status"] == "analyzed"
    assert updated["scanned_pages_count"] == 1
    
    docs = temp_db.list_documents()
    assert len(docs) == 1
    
    temp_db.delete_document("DOC_TEST_001")
    assert temp_db.get_document("DOC_TEST_001") is None


def test_clauses_and_pages_persistence(temp_db: DatabaseManager):
    """Test persisting pages and clauses with proper foreign key relationships."""
    temp_db.save_document({
        "id": "DOC_002",
        "filename": "rfp_supply.pdf",
        "file_hash": "abc123hash",
        "total_pages": 2,
        "file_size_bytes": 50000,
        "status": "processing"
    })
    
    pages = [
        {"doc_id": "DOC_002", "page_number": 1, "char_count": 500, "is_scanned_or_empty": False, "raw_text": "Page 1 content..."},
        {"doc_id": "DOC_002", "page_number": 2, "char_count": 30, "is_scanned_or_empty": True, "raw_text": "Scanned page image"}
    ]
    temp_db.save_pages(pages)
    
    stored_pages = temp_db.get_pages("DOC_002")
    assert len(stored_pages) == 2
    assert stored_pages[1]["is_scanned_or_empty"] == 1
    
    clauses = [
        {
            "id": "DOC_002_C001",
            "doc_id": "DOC_002",
            "page_number": 1,
            "clause_number": "1.1",
            "original_text": "The bidder must furnish a bid security of $50,000.",
            "char_start": 0,
            "char_end": 50,
            "category": "Financial & Commercial Terms",
            "category_confidence": 0.95,
            "classification_model": "typeform/distilbert-base-uncased-mnli"
        },
        {
            "id": "DOC_002_C002",
            "doc_id": "DOC_002",
            "page_number": 1,
            "clause_number": "1.2",
            "original_text": "Bids must be submitted no later than October 30, 2026.",
            "char_start": 51,
            "char_end": 105,
            "category": "Deadlines & Submission Schedules",
            "category_confidence": 0.92,
            "classification_model": "typeform/distilbert-base-uncased-mnli"
        }
    ]
    temp_db.save_clauses(clauses)
    
    all_clauses = temp_db.get_clauses("DOC_002")
    assert len(all_clauses) == 2
    
    fin_clauses = temp_db.get_clauses("DOC_002", category="Financial & Commercial Terms")
    assert len(fin_clauses) == 1
    assert fin_clauses[0]["id"] == "DOC_002_C001"


def test_findings_and_human_review_separation(temp_db: DatabaseManager):
    """Test that AI findings are inserted with default Pending Review and can be reviewed independently."""
    temp_db.save_document({
        "id": "DOC_003",
        "filename": "procurement.pdf",
        "file_hash": "hash003",
        "total_pages": 5,
        "file_size_bytes": 80000,
        "status": "analyzed"
    })
    
    temp_db.save_clauses([{
        "id": "DOC_003_C001",
        "doc_id": "DOC_003",
        "page_number": 1,
        "clause_number": "3.1",
        "original_text": "Warranty shall be 12 months.",
        "char_start": 0,
        "char_end": 28,
        "category": "Penalties & Liquidated Damages",
        "category_confidence": 0.88,
        "classification_model": "test"
    }, {
        "id": "DOC_003_C002",
        "doc_id": "DOC_003",
        "page_number": 4,
        "clause_number": "8.4",
        "original_text": "Warranty shall be 24 months.",
        "char_start": 0,
        "char_end": 28,
        "category": "Penalties & Liquidated Damages",
        "category_confidence": 0.89,
        "classification_model": "test"
    }])
    
    finding = {
        "id": "FND_TEST_001",
        "doc_id": "DOC_003",
        "finding_type": "Numerical Inconsistency",
        "clause_a_id": "DOC_003_C001",
        "clause_b_id": "DOC_003_C002",
        "clause_a_text": "Warranty shall be 12 months.",
        "clause_b_text": "Warranty shall be 24 months.",
        "page_numbers": "Page 1, Page 4",
        "semantic_similarity": 0.91,
        "nli_label": "contradiction",
        "nli_contradiction_score": 0.82,
        "nli_entailment_score": 0.05,
        "nli_neutral_score": 0.13,
        "rule_triggered": "NUMERICAL_DURATION_CONFLICT",
        "explanation": "Potential inconsistency in warranty duration (12 months vs 24 months). Requires human review.",
        "priority": "High",
        "model_identifier": "Hybrid-Engine-v1"
    }
    
    temp_db.save_findings([finding])
    
    # Retrieve findings with review
    records = temp_db.get_findings_with_reviews(doc_id="DOC_003")
    assert len(records) == 1
    assert records[0]["review_status"] == "Pending Review"
    assert records[0]["priority"] == "High"
    
    # Human performs review
    temp_db.upsert_review(
        finding_id="FND_TEST_001",
        status="Confirmed Risk",
        reviewer_comment="Clause 8.4 appears to be an unupdated copy-paste error from an earlier template.",
        reviewer_name="Senior Procurement Officer"
    )
    
    updated_records = temp_db.get_findings_with_reviews(doc_id="DOC_003")
    assert len(updated_records) == 1
    assert updated_records[0]["review_status"] == "Confirmed Risk"
    assert updated_records[0]["reviewer_name"] == "Senior Procurement Officer"
    assert "copy-paste error" in updated_records[0]["reviewer_comment"]
    # Model predictions remained intact
    assert updated_records[0]["nli_contradiction_score"] == 0.82


def test_user_registration_and_authentication(temp_db: DatabaseManager):
    """Ensure user registration hashes passwords with salt and validates credentials."""
    # Seed default user was executed during init_db
    default_user = temp_db.authenticate_user("auditor", "tenderguard2026")
    assert default_user is not None
    assert default_user["username"] == "auditor"
    assert "password_hash" not in default_user

    # Register new user
    uid = temp_db.create_user(
        username="lead_auditor",
        email="lead@agency.gov",
        full_name="Dr. Alex Rivera",
        password="SecurePass2026!",
        organization="Federal Procurement Council",
        role="Lead Auditor"
    )
    assert uid > 0

    # Successful authentication by username
    auth_by_user = temp_db.authenticate_user("lead_auditor", "SecurePass2026!")
    assert auth_by_user is not None
    assert auth_by_user["full_name"] == "Dr. Alex Rivera"

    # Successful authentication by email
    auth_by_email = temp_db.authenticate_user("lead@agency.gov", "SecurePass2026!")
    assert auth_by_email is not None
    assert auth_by_email["id"] == uid

    # Failed authentication with incorrect password
    failed_auth = temp_db.authenticate_user("lead_auditor", "WrongPassword")
    assert failed_auth is None

