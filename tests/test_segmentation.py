"""Unit tests for clause segmentation engine."""
import pytest
from tenderguard.ingestion.pdf_extractor import ExtractedPage
from tenderguard.ingestion.segmenter import ClauseSegmenter


def test_clause_segmentation_with_headers():
    """Ensure clauses with section/clause headers are correctly segmented with numbers."""
    page_text = (
        "SECTION 1.1 Scope of Work\n"
        "The contractor shall deliver 500 laptops within 60 calendar days of contract signing.\n\n"
        "Clause 2.4 Warranty Period\n"
        "The warranty period for all delivered equipment shall be 24 months from acceptance.\n\n"
        "3.1 Liquidated Damages\n"
        "Failure to deliver within schedule will incur a penalty of 0.5% per week up to a maximum of 10%."
    )
    
    page = ExtractedPage(
        page_number=1,
        char_count=len(page_text),
        is_scanned_or_empty=False,
        raw_text=page_text
    )
    
    segmenter = ClauseSegmenter()
    clauses = segmenter.segment_page("DOC_001", page, start_clause_idx=1)
    
    assert len(clauses) == 3
    
    assert clauses[0].id == "DOC_001_C001"
    assert clauses[0].page_number == 1
    assert "1.1" in clauses[0].clause_number or "SECTION" in clauses[0].original_text
    assert "contractor shall deliver 500 laptops" in clauses[0].original_text
    
    assert clauses[1].id == "DOC_001_C002"
    assert "24 months" in clauses[1].original_text
    
    assert clauses[2].id == "DOC_001_C003"
    assert "0.5% per week" in clauses[2].original_text


def test_clause_noise_filtering():
    """Ensure running headers, footers, and standalone page numbers are filtered out."""
    page_text = (
        "Page 4 of 25\n\n"
        "Clause 5.1 Eligibility Criteria\n"
        "All bidders must be registered legal entities with at least 3 years in operation.\n\n"
        "CONFIDENTIAL\n\n"
        "------------------\n\n"
        "Clause 5.2 Turnover Requirements\n"
        "Bidders must demonstrate an average annual turnover of at least $1,000,000."
    )
    
    page = ExtractedPage(
        page_number=4,
        char_count=len(page_text),
        is_scanned_or_empty=False,
        raw_text=page_text
    )
    
    segmenter = ClauseSegmenter()
    clauses = segmenter.segment_page("DOC_002", page, start_clause_idx=1)
    
    # Noise like "Page 4 of 25", "CONFIDENTIAL", and dashes should not produce standalone clauses
    assert len(clauses) == 2
    for c in clauses:
        assert not c.original_text.lower().startswith("page 4")
        assert not c.original_text.lower() == "confidential"


def test_multi_page_document_segmentation():
    """Ensure multi-page document flows maintain continuous clause numbering and page numbers."""
    pages = [
        ExtractedPage(
            page_number=1,
            char_count=100,
            is_scanned_or_empty=False,
            raw_text="Clause 1.0 General Terms.\nThis agreement governs all supplies.\n\nClause 2.0 Pricing.\nPrices are fixed."
        ),
        ExtractedPage(
            page_number=2,
            char_count=0,
            is_scanned_or_empty=True,
            raw_text=""
        ),
        ExtractedPage(
            page_number=3,
            char_count=90,
            is_scanned_or_empty=False,
            raw_text="Clause 3.0 Delivery.\nDelivery must take place within 30 days."
        ),
    ]
    
    segmenter = ClauseSegmenter()
    all_clauses = segmenter.segment_document("DOC_MULTI", pages)
    
    assert len(all_clauses) == 3
    assert all_clauses[0].id == "DOC_MULTI_C001"
    assert all_clauses[0].page_number == 1
    assert all_clauses[1].id == "DOC_MULTI_C002"
    assert all_clauses[1].page_number == 1
    assert all_clauses[2].id == "DOC_MULTI_C003"
    assert all_clauses[2].page_number == 3
