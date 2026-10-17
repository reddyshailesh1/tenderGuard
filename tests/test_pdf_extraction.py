"""Unit tests for PDF extraction and validation."""
import io
import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from tenderguard.config import IngestionConfig
from tenderguard.ingestion.pdf_extractor import PDFExtractor


def create_sample_pdf(page_texts: list) -> bytes:
    """Helper to create a multi-page PDF in-memory using ReportLab."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    for text in page_texts:
        if text:  # If non-empty, write text
            c.drawString(100, 700, text)
        c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()


def test_pdf_validation_magic_bytes():
    """Ensure non-PDF files are rejected."""
    extractor = PDFExtractor()
    with pytest.raises(ValueError, match="not a valid PDF document"):
        extractor.validate_file(b"This is not a PDF file content", "fake.pdf")


def test_pdf_size_limit():
    """Ensure files exceeding max size are rejected."""
    config = IngestionConfig(max_file_size_bytes=100)
    extractor = PDFExtractor(config=config)
    large_content = b"%PDF-1.4 " + b"X" * 200
    with pytest.raises(ValueError, match="exceeds maximum allowed limit"):
        extractor.validate_file(large_content, "large.pdf")


def test_valid_pdf_extraction_page_preservation():
    """Ensure pages and text are extracted accurately with preserved 1-indexed page numbers."""
    page_texts = [
        "Clause 1.0: Eligibility. The bidder must have 5 years experience.",
        "Clause 2.0: Financials. The minimum annual turnover is $2,000,000.",
        "Clause 3.0: Submission. Submissions are due on November 15, 2026."
    ]
    pdf_bytes = create_sample_pdf(page_texts)
    
    extractor = PDFExtractor()
    extracted = extractor.extract(pdf_bytes, filename="tender_test.pdf", doc_id="DOC_UNIT_01")
    
    assert extracted.doc_id == "DOC_UNIT_01"
    assert extracted.total_pages == 3
    assert extracted.scanned_pages_count == 0
    assert len(extracted.pages) == 3
    
    for i, p in enumerate(extracted.pages):
        assert p.page_number == i + 1
        assert not p.is_scanned_or_empty
        assert "Clause" in p.raw_text


def test_scanned_or_empty_page_detection():
    """Ensure blank or near-blank pages are correctly flagged as scanned/empty."""
    page_texts = [
        "Normal text page with sufficient tender content and detailed requirements for all bidders.",
        "",  # Completely empty page (simulating scanned image page without digital text)
        "Another normal page describing delivery timelines and liquidated damages."
    ]
    pdf_bytes = create_sample_pdf(page_texts)
    
    config = IngestionConfig(min_page_chars_for_text=30)
    extractor = PDFExtractor(config=config)
    extracted = extractor.extract(pdf_bytes, filename="partially_scanned.pdf")
    
    assert extracted.total_pages == 3
    assert extracted.scanned_pages_count == 1
    assert extracted.pages[1].is_scanned_or_empty is True
    assert extracted.pages[0].is_scanned_or_empty is False
    assert extracted.pages[2].is_scanned_or_empty is False
