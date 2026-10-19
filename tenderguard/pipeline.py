"""End-to-end processing pipeline orchestrating all ingestion, modeling, and analysis stages."""
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from tenderguard.analysis.hybrid_engine import Finding, HybridEngine
from tenderguard.config import DEFAULT_CONFIG, TenderGuardConfig
from tenderguard.db.session import DatabaseManager
from tenderguard.ingestion.pdf_extractor import ExtractedDocument, PDFExtractor
from tenderguard.ingestion.segmenter import ClauseSegmenter, SegmentedClause
from tenderguard.models.classifier import ClauseClassifier
from tenderguard.models.embeddings import CandidatePair, SemanticRetriever
from tenderguard.models.nli_engine import NLIEngine, NLIPrediction

logger = logging.getLogger(__name__)


class TenderPipeline:
    """Coordinates the end-to-end document processing, Transformer inference, and risk detection."""

    def __init__(
        self,
        config: Optional[TenderGuardConfig] = None,
        db_manager: Optional[DatabaseManager] = None
    ):
        self.config = config or DEFAULT_CONFIG
        self.db = db_manager or DatabaseManager(self.config.db_path)

        # Lazy component holders
        self._extractor: Optional[PDFExtractor] = None
        self._segmenter: Optional[ClauseSegmenter] = None
        self._classifier: Optional[ClauseClassifier] = None
        self._retriever: Optional[SemanticRetriever] = None
        self._nli_engine: Optional[NLIEngine] = None
        self._hybrid_engine: Optional[HybridEngine] = None

    @property
    def extractor(self) -> PDFExtractor:
        if self._extractor is None:
            self._extractor = PDFExtractor(self.config.ingestion)
        return self._extractor

    @property
    def segmenter(self) -> ClauseSegmenter:
        if self._segmenter is None:
            self._segmenter = ClauseSegmenter(self.config.ingestion)
        return self._segmenter

    @property
    def classifier(self) -> ClauseClassifier:
        if self._classifier is None:
            self._classifier = ClauseClassifier(self.config.models)
        return self._classifier

    @property
    def retriever(self) -> SemanticRetriever:
        if self._retriever is None:
            self._retriever = SemanticRetriever(self.config.models)
        return self._retriever

    @property
    def nli_engine(self) -> NLIEngine:
        if self._nli_engine is None:
            self._nli_engine = NLIEngine(self.config.models)
        return self._nli_engine

    @property
    def hybrid_engine(self) -> HybridEngine:
        if self._hybrid_engine is None:
            self._hybrid_engine = HybridEngine()
        return self._hybrid_engine

    def process_pdf(
        self,
        file_source: Union[str, Path, bytes],
        filename: str,
        doc_id: Optional[str] = None,
        user_id: Optional[int] = None,
        progress_callback: Optional[Callable[[str, float, str], None]] = None
    ) -> Dict[str, Any]:
        """Execute the complete end-to-end tender analysis workflow."""
        start_time = time.time()

        def _notify(stage: str, pct: float, msg: str):
            logger.info("[%s - %.0f%%] %s", stage, pct * 100, msg)
            if progress_callback:
                progress_callback(stage, pct, msg)

        # Stage 1: Extraction & Validation
        _notify("Extraction", 0.10, f"Validating and extracting text from '{filename}'...")
        target_doc_id = doc_id
        if not target_doc_id and user_id is not None:
            # Deterministic prefix with user isolation and short timestamp suffix
            file_bytes_temp = self.extractor.validate_file(file_source, filename)
            f_hash = self.extractor.compute_sha256(file_bytes_temp)
            ts = str(int(time.time()))[-5:]
            target_doc_id = f"DOC_U{user_id}_{f_hash[:6].upper()}_{ts}"

        extracted_doc: ExtractedDocument = self.extractor.extract(file_source, filename, doc_id=target_doc_id)
        active_doc_id = extracted_doc.doc_id

        # Persist document metadata with user ownership
        self.db.save_document({
            "id": active_doc_id,
            "user_id": user_id,
            "filename": extracted_doc.filename,
            "file_hash": extracted_doc.file_hash,
            "total_pages": extracted_doc.total_pages,
            "file_size_bytes": extracted_doc.file_size_bytes,
            "scanned_pages_count": extracted_doc.scanned_pages_count,
            "status": "processing"
        })

        # Persist extracted pages
        pages_to_save = [
            {
                "doc_id": active_doc_id,
                "page_number": p.page_number,
                "char_count": p.char_count,
                "is_scanned_or_empty": p.is_scanned_or_empty,
                "raw_text": p.raw_text
            }
            for p in extracted_doc.pages
        ]
        self.db.save_pages(pages_to_save)

        # Stage 2: Clause Segmentation
        _notify("Segmentation", 0.25, "Segmenting text into clauses and preserving page offsets...")
        clauses: List[SegmentedClause] = self.segmenter.segment_document(active_doc_id, extracted_doc.pages)
        if not clauses:
            self.db.update_document_status(active_doc_id, "failed", error_message="No extractable clauses found in document.")
            raise ValueError(f"No valid clauses could be extracted from '{filename}'.")

        # Save initial clauses
        clauses_to_save = [
            {
                "id": c.id,
                "doc_id": c.doc_id,
                "page_number": c.page_number,
                "clause_number": c.clause_number,
                "original_text": c.original_text,
                "char_start": c.char_start,
                "char_end": c.char_end,
                "category": None,
                "category_confidence": None,
                "classification_model": None
            }
            for c in clauses
        ]
        self.db.save_clauses(clauses_to_save)

        # Stage 3: Transformer Clause Classification
        _notify("Classification", 0.40, f"Classifying {len(clauses)} clauses into tender categories...")
        classification_results = self.classifier.classify_clauses(clauses)
        
        # Update clauses with classification results in DB and in-memory list
        classified_clauses_for_retrieval = []
        for c, res in zip(clauses, classification_results):
            self.db.update_clause_category(
                clause_id=c.id,
                category=res.predicted_category,
                confidence=res.confidence,
                model_name=res.model_identifier
            )
            classified_clauses_for_retrieval.append({
                "id": c.id,
                "doc_id": c.doc_id,
                "page_number": c.page_number,
                "clause_number": c.clause_number,
                "original_text": c.original_text,
                "category": res.predicted_category
            })

        # Stage 4: Semantic Embedding & Candidate Retrieval
        _notify("Retrieval", 0.60, "Computing dense sentence embeddings and retrieving candidate pairs...")
        candidate_pairs: List[CandidatePair] = self.retriever.retrieve_candidate_pairs(
            classified_clauses_for_retrieval,
            similarity_threshold=self.config.models.similarity_threshold
        )

        pairs_to_save = [
            {
                "doc_id": p.doc_id,
                "clause_a_id": p.clause_a_id,
                "clause_b_id": p.clause_b_id,
                "similarity_score": p.similarity_score,
                "embedding_model": p.embedding_model,
                "retrieval_method": p.retrieval_method
            }
            for p in candidate_pairs
        ]
        if pairs_to_save:
            self.db.save_candidate_pairs(pairs_to_save)

        # Stage 5: NLI Inference on Candidate Pairs
        _notify("NLI Inference", 0.75, f"Running pretrained NLI contradiction inference on {len(candidate_pairs)} candidate pairs...")
        nli_predictions: List[NLIPrediction] = self.nli_engine.predict_candidate_pairs(candidate_pairs)

        # Stage 6: Hybrid Synthesis (NLI + Numerical + Date Rules)
        _notify("Synthesis", 0.90, "Synthesizing model predictions, numerical checks, and deadline rules...")
        findings: List[Finding] = self.hybrid_engine.synthesize_all(candidate_pairs, nli_predictions, active_doc_id)

        findings_to_save = [
            {
                "id": f.id,
                "doc_id": f.doc_id,
                "finding_type": f.finding_type,
                "clause_a_id": f.clause_a_id,
                "clause_b_id": f.clause_b_id,
                "clause_a_text": f.clause_a_text,
                "clause_b_text": f.clause_b_text,
                "page_numbers": f.page_numbers,
                "semantic_similarity": f.semantic_similarity,
                "nli_label": f.nli_label,
                "nli_contradiction_score": f.nli_contradiction_score,
                "nli_entailment_score": f.nli_entailment_score,
                "nli_neutral_score": f.nli_neutral_score,
                "rule_triggered": f.rule_triggered,
                "explanation": f.explanation,
                "priority": f.priority,
                "model_identifier": f.model_identifier
            }
            for f in findings
        ]
        if findings_to_save:
            self.db.save_findings(findings_to_save)

        # Finalize document status
        self.db.update_document_status(active_doc_id, "analyzed", scanned_pages=extracted_doc.scanned_pages_count)
        elapsed = time.time() - start_time
        _notify("Complete", 1.0, f"Analysis complete in {elapsed:.2f}s. {len(findings)} potential risks identified.")

        return {
            "doc_id": active_doc_id,
            "filename": filename,
            "total_pages": extracted_doc.total_pages,
            "scanned_pages": extracted_doc.scanned_pages_count,
            "clauses_count": len(clauses),
            "candidate_pairs_count": len(candidate_pairs),
            "findings_count": len(findings),
            "elapsed_seconds": round(elapsed, 2)
        }
