"""Semantic embedding generation and candidate pair retrieval using Sentence Transformers."""
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from tenderguard.config import DEFAULT_CONFIG, ModelConfig
from tenderguard.ingestion.segmenter import SegmentedClause

logger = logging.getLogger(__name__)


@dataclass
class CandidatePair:
    """Represents two semantically relevant clauses selected for contradiction and conflict analysis."""
    doc_id: str
    clause_a_id: str
    clause_b_id: str
    clause_a_text: str
    clause_b_text: str
    similarity_score: float
    embedding_model: str
    retrieval_method: str
    category_a: Optional[str] = None
    category_b: Optional[str] = None
    page_a: Optional[int] = None
    page_b: Optional[int] = None


class SemanticRetriever:
    """Generates dense vector embeddings and retrieves semantically related candidate clause pairs."""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or DEFAULT_CONFIG.models
        self.model_name = self.config.embedding_model_name
        self._model: Optional[SentenceTransformer] = None

    def _load_model(self) -> SentenceTransformer:
        """Lazy load the SentenceTransformer model."""
        if self._model is None:
            logger.info("Loading SentenceTransformer model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode list of texts into L2-normalized dense embeddings."""
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        model = self._load_model()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return np.asarray(embeddings, dtype=np.float32)

    def retrieve_candidate_pairs(
        self,
        clauses: List[Union[SegmentedClause, Dict[str, Any]]],
        similarity_threshold: Optional[float] = None,
        top_k_per_clause: Optional[int] = None
    ) -> List[CandidatePair]:
        """Compute pairwise cosine similarities and retrieve candidate pairs above the threshold."""
        if len(clauses) < 2:
            return []

        threshold = similarity_threshold if similarity_threshold is not None else self.config.similarity_threshold
        top_k = top_k_per_clause if top_k_per_clause is not None else self.config.top_k_candidates

        # Standardize clause objects into dicts
        clause_items = []
        texts = []
        for c in clauses:
            if isinstance(c, SegmentedClause):
                item = {
                    "id": c.id,
                    "doc_id": c.doc_id,
                    "page_number": c.page_number,
                    "original_text": c.original_text,
                    "category": getattr(c, "category", None)
                }
            else:
                item = dict(c)
            clause_items.append(item)
            texts.append(item["original_text"])

        # Compute embeddings and similarity matrix
        embeddings = self.encode(texts)
        sim_matrix = cosine_similarity(embeddings, embeddings)

        candidate_pairs: List[CandidatePair] = []
        n = len(clause_items)

        # For each clause, track candidates to avoid combinatorial explosion
        clause_candidate_counts: Dict[str, int] = {c["id"]: 0 for c in clause_items}

        # Gather valid pairs
        pair_candidates = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(sim_matrix[i, j])
                item_a = clause_items[i]
                item_b = clause_items[j]

                # Exact duplicate detection or semantic similarity threshold
                is_exact_dup = (item_a["original_text"].strip().lower() == item_b["original_text"].strip().lower())
                
                # Boost candidate consideration if they share specific tender categories (Financial, Deadlines, Penalties)
                same_critical_category = (
                    item_a.get("category") and 
                    item_a.get("category") == item_b.get("category") and
                    item_a.get("category") in {
                        "Financial & Commercial Terms",
                        "Deadlines & Submission Schedules",
                        "Penalties & Liquidated Damages",
                        "Eligibility & Qualifications"
                    }
                )

                effective_threshold = threshold - 0.08 if same_critical_category else threshold

                if sim >= effective_threshold or is_exact_dup:
                    pair_candidates.append((sim, i, j, is_exact_dup))

        # Sort pairs by similarity descending
        pair_candidates.sort(key=lambda x: x[0], reverse=True)

        seen_pairs = set()
        for sim, i, j, is_exact_dup in pair_candidates:
            item_a = clause_items[i]
            item_b = clause_items[j]
            pair_key = (min(item_a["id"], item_b["id"]), max(item_a["id"], item_b["id"]))
            if pair_key in seen_pairs:
                continue

            if clause_candidate_counts[item_a["id"]] >= top_k and clause_candidate_counts[item_b["id"]] >= top_k:
                continue

            seen_pairs.add(pair_key)
            clause_candidate_counts[item_a["id"]] += 1
            clause_candidate_counts[item_b["id"]] += 1

            retrieval_method = "ExactDuplicate" if is_exact_dup else "SentenceTransformer-Cosine"

            candidate_pairs.append(
                CandidatePair(
                    doc_id=item_a["doc_id"],
                    clause_a_id=item_a["id"],
                    clause_b_id=item_b["id"],
                    clause_a_text=item_a["original_text"],
                    clause_b_text=item_b["original_text"],
                    similarity_score=sim,
                    embedding_model=self.model_name,
                    retrieval_method=retrieval_method,
                    category_a=item_a.get("category"),
                    category_b=item_b.get("category"),
                    page_a=item_a.get("page_number"),
                    page_b=item_b.get("page_number"),
                )
            )

        logger.info(
            "Retrieved %d candidate pairs across %d clauses (threshold=%.2f).",
            len(candidate_pairs), n, threshold
        )
        return candidate_pairs
