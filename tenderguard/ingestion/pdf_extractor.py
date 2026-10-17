"""PDF text extraction and validation module using PyMuPDF."""
import hashlib
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, List, Optional, Tuple, Union

import pymupdf  # PyMuPDF

from tenderguard.config import IngestionConfig

logger = logging.getLogger(__name__)


@dataclass
class ExtractedPage:
    """Represents text and metadata extracted from an individual document page."""
    page_number: int          # 1-indexed
    char_count: int
    is_scanned_or_empty: bool
    raw_text: str
    image_count: int = 0
    ocr_applied: bool = False


@dataclass
class ExtractedDocument:
    """Summary and page collection extracted from a PDF document."""
    doc_id: str
    filename: str
    file_hash: str
    file_size_bytes: int
    total_pages: int
    scanned_pages_count: int
    pages: List[ExtractedPage]


class PDFExtractor:
    """Validates PDF files and extracts text page-by-page preserving document geometry."""

    def __init__(self, config: Optional[IngestionConfig] = None):
        self.config = config or IngestionConfig()

    @staticmethod
    def compute_sha256(file_bytes: bytes) -> str:
        """Compute SHA256 digest of raw file content."""
        return hashlib.sha256(file_bytes).hexdigest()

    def validate_file(self, file_source: Union[str, Path, bytes, BinaryIO], filename: str) -> bytes:
        """Validate file size and magic bytes to ensure safe PDF processing."""
        if isinstance(file_source, (str, Path)):
            path = Path(file_source)
            if not path.is_file():
                raise FileNotFoundError(f"Tender file not found: {path}")
            size = path.stat().st_size
            if size > self.config.max_file_size_bytes:
                raise ValueError(
                    f"File size ({size / (1024 * 1024):.2f} MB) exceeds maximum allowed limit "
                    f"({self.config.max_file_size_bytes / (1024 * 1024):.2f} MB)."
                )
            file_bytes = path.read_bytes()
        elif isinstance(file_source, bytes):
            file_bytes = file_source
        elif hasattr(file_source, "read"):
            file_bytes = file_source.read()
        else:
            raise TypeError("Unsupported file source type.")

        if len(file_bytes) > self.config.max_file_size_bytes:
            raise ValueError(
                f"File size ({len(file_bytes) / (1024 * 1024):.2f} MB) exceeds maximum allowed limit."
            )

        # Validate PDF magic header '%PDF-'
        if not file_bytes.startswith(b"%PDF-"):
            raise ValueError(f"Uploaded file '{filename}' is not a valid PDF document (missing %PDF- header).")

        return file_bytes

    def extract(
        self,
        file_source: Union[str, Path, bytes, BinaryIO],
        filename: str,
        doc_id: Optional[str] = None
    ) -> ExtractedDocument:
        """Extract text from all pages in the PDF document."""
        file_bytes = self.validate_file(file_source, filename)
        file_hash = self.compute_sha256(file_bytes)
        
        if not doc_id:
            # Deterministic, unique document identifier based on hash prefix
            doc_id = f"DOC_{file_hash[:10].upper()}"

        try:
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        except Exception as e:
            logger.error("PyMuPDF failed to parse PDF document: %s", e)
            raise ValueError(f"Failed to parse PDF document '{filename}': {e}") from e

        total_pages = len(doc)
        if total_pages == 0:
            doc.close()
            raise ValueError(f"PDF document '{filename}' contains 0 pages.")

        extracted_pages: List[ExtractedPage] = []
        scanned_count = 0

        for page_idx in range(total_pages):
            page_num = page_idx + 1
            page = doc[page_idx]
            
            # Extract standard digital text
            text = page.get_text("text") or ""
            clean_text = text.strip()
            char_count = len(clean_text)
            
            images = page.get_images()
            image_count = len(images)
            ocr_applied = False
            is_scanned_or_empty = False

            # Detect scanned / low-text pages
            if char_count < self.config.min_page_chars_for_text:
                is_scanned_or_empty = True
                if self.config.enable_ocr_fallback and image_count > 0:
                    ocr_text = self._try_ocr_page(page)
                    if ocr_text and len(ocr_text.strip()) >= self.config.min_page_chars_for_text:
                        clean_text = ocr_text.strip()
                        char_count = len(clean_text)
                        is_scanned_or_empty = False
                        ocr_applied = True
                
                if is_scanned_or_empty:
                    scanned_count += 1
                    logger.warning(
                        "Page %d in '%s' detected as scanned/empty (characters: %d, images: %d).",
                        page_num, filename, char_count, image_count
                    )

            extracted_pages.append(
                ExtractedPage(
                    page_number=page_num,
                    char_count=char_count,
                    is_scanned_or_empty=is_scanned_or_empty,
                    raw_text=clean_text,
                    image_count=image_count,
                    ocr_applied=ocr_applied,
                )
            )

        doc.close()

        return ExtractedDocument(
            doc_id=doc_id,
            filename=filename,
            file_hash=file_hash,
            file_size_bytes=len(file_bytes),
            total_pages=total_pages,
            scanned_pages_count=scanned_count,
            pages=extracted_pages,
        )

    def _try_ocr_page(self, page: pymupdf.Page) -> Optional[str]:
        """Optional OCR fallback using pytesseract if installed."""
        try:
            import pytesseract
            from PIL import Image
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img)
            return text
        except Exception:
            # pytesseract or tesseract binary not installed; return None gracefully
            return None
