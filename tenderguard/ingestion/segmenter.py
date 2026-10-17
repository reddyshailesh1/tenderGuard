"""Clause segmentation engine for tender documents."""
import re
from dataclasses import dataclass
from typing import List, Optional

from tenderguard.config import IngestionConfig
from tenderguard.ingestion.pdf_extractor import ExtractedPage


@dataclass
class SegmentedClause:
    """Represents an atomic, identifiable clause from a tender document."""
    id: str
    doc_id: str
    page_number: int
    clause_number: Optional[str]
    original_text: str
    char_start: int
    char_end: int


# Regex pattern to match legal and tender clause headers
CLAUSE_HEADER_PATTERN = re.compile(
    r"""
    ^[\t ]*
    (?:
        (?:CLAUSE|SECTION|ARTICLE|SCHEDULE|ANNEXURE|APPENDIX|ITEM|PART)\s+([0-9IVXLCDM]+(?:\.[0-9]+)*(?:\([a-z0-9]+\))?)
        |
        ([0-9]{1,3}(?:\.[0-9]{1,3}){1,4}(?:\([a-z0-9]+\))?)
        |
        ([0-9]{1,2}\.)(?=\s+[A-Z])
        |
        (\([a-z0-9]{1,3}\))(?=\s+[A-Z0-9])
    )
    """,
    re.IGNORECASE | re.VERBOSE | re.MULTILINE
)

# Patterns for page artifacts to filter out (e.g. "Page 1 of 12", "Confidential")
NOISE_LINE_PATTERNS = [
    re.compile(r"^page\s+\d+\s*(of\s*\d+)?$", re.IGNORECASE),
    re.compile(r"^\d+\s*/\s*\d+$"),
    re.compile(r"^[-—_]{3,}$"),
    re.compile(r"^confidential\s*$", re.IGNORECASE),
]


class ClauseSegmenter:
    """Segments tender text page-by-page into identifiable, evidence-traceable clauses."""

    def __init__(self, config: Optional[IngestionConfig] = None):
        self.config = config or IngestionConfig()

    def segment_document(self, doc_id: str, pages: List[ExtractedPage]) -> List[SegmentedClause]:
        """Process all extracted pages of a document and return an ordered list of clauses."""
        all_clauses: List[SegmentedClause] = []
        clause_counter = 1

        for page in pages:
            if page.is_scanned_or_empty or not page.raw_text:
                continue

            page_clauses = self.segment_page(
                doc_id=doc_id,
                page=page,
                start_clause_idx=clause_counter
            )
            all_clauses.extend(page_clauses)
            clause_counter += len(page_clauses)

        return all_clauses

    def segment_page(self, doc_id: str, page: ExtractedPage, start_clause_idx: int = 1) -> List[SegmentedClause]:
        """Segment a single page into clauses using header recognition and paragraph boundaries."""
        text = page.raw_text
        if not text or len(text.strip()) < self.config.min_clause_chars:
            return []

        # Split text into candidate blocks based on double newlines or clause header matches
        raw_blocks = self._split_into_raw_blocks(text)
        
        clauses: List[SegmentedClause] = []
        current_idx = start_clause_idx

        for block, start_char, end_char in raw_blocks:
            clean_block = self._clean_block(block)
            
            # Skip noise or overly short fragments
            if len(clean_block) < self.config.min_clause_chars:
                continue
            if self._is_noise(clean_block):
                continue

            # Extract clause numbering identifier if present
            clause_num = self._extract_clause_number(clean_block)

            clause_id = f"{doc_id}_C{current_idx:03d}"
            clauses.append(
                SegmentedClause(
                    id=clause_id,
                    doc_id=doc_id,
                    page_number=page.page_number,
                    clause_number=clause_num,
                    original_text=clean_block,
                    char_start=start_char,
                    char_end=end_char
                )
            )
            current_idx += 1

        return clauses

    def _split_into_raw_blocks(self, text: str) -> List[tuple]:
        """Identify boundaries by double newlines or explicit clause header patterns."""
        # Find all clause header match positions
        header_positions = []
        for match in CLAUSE_HEADER_PATTERN.finditer(text):
            header_positions.append(match.start())

        # If no explicit clause headers found, fall back to paragraph split
        if not header_positions:
            blocks = []
            curr_pos = 0
            for part in re.split(r"\n\s*\n+", text):
                clean_part = part.strip()
                if clean_part:
                    start = text.find(part, curr_pos)
                    end = start + len(part)
                    blocks.append((clean_part, start, end))
                    curr_pos = end
            return blocks

        # Ensure first block is captured even before first header
        boundaries = sorted(list(set([0] + header_positions + [len(text)])))
        blocks = []

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            segment = text[start:end].strip()
            
            if not segment:
                continue

            # If a segment is very long and has clear internal paragraph breaks, subdivide it
            if len(segment) > self.config.max_clause_chars:
                sub_blocks = self._subdivide_large_segment(segment, start)
                blocks.extend(sub_blocks)
            else:
                blocks.append((segment, start, end))

        return blocks

    def _subdivide_large_segment(self, segment: str, offset: int) -> List[tuple]:
        """Subdivide unusually large sections on paragraph boundaries."""
        sub_blocks = []
        paragraphs = re.split(r"\n\s*\n+", segment)
        curr_offset = offset
        for p in paragraphs:
            clean_p = p.strip()
            if clean_p:
                p_start = segment.find(p, curr_offset - offset) + offset
                p_end = p_start + len(p)
                sub_blocks.append((clean_p, p_start, p_end))
                curr_offset = p_end
        return sub_blocks

    def _extract_clause_number(self, text: str) -> Optional[str]:
        """Extract clause numbering prefix (e.g. '1.2.3', 'Clause 4', 'Section 5(a)')."""
        match = CLAUSE_HEADER_PATTERN.match(text)
        if match:
            # Capture the non-empty group
            for group in match.groups():
                if group:
                    return group.strip()
        return None

    def _clean_block(self, text: str) -> str:
        """Normalize excessive internal whitespace while preserving original phrasing."""
        # Replace multiple spaces with a single space
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
        # Reconnect lines with a single space or newline if appropriate
        cleaned = " ".join([l for l in lines if l])
        return cleaned.strip()

    def _is_noise(self, text: str) -> bool:
        """Check if block is document noise (e.g. running headers, solitary page numbers)."""
        stripped = text.strip()
        for pat in NOISE_LINE_PATTERNS:
            if pat.match(stripped):
                return True
        return False
