"""Clause classification using Pretrained Transformer models."""
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from tenderguard.config import DEFAULT_CONFIG, ModelConfig, TENDER_CATEGORIES
from tenderguard.ingestion.segmenter import SegmentedClause

logger = logging.getLogger(__name__)

# Canonical tender categories and semantic anchor descriptions
CATEGORY_ANCHORS: Dict[str, str] = {
    "Financial & Commercial Terms": "earnest money deposit, bank guarantee, bid security, pricing, payment terms, and annual turnover",
    "Deadlines & Submission Schedules": "submission deadline, closing date, submission schedule, delivery timeline, and time limits",
    "Penalties & Liquidated Damages": "liquidated damages, delay penalties, fines, deductions, defects liability, and warranty period",
    "Eligibility & Qualifications": "bidder eligibility, minimum experience, technical capability, certifications, and qualification criteria",
    "Technical Specifications": "technical specifications, hardware specs, software architecture, technical parameters, and standards",
    "Legal, Compliance & Contractual Terms": "governing law, dispute resolution, arbitration, jurisdiction, and legal contractual compliance",
    "Scope of Work & Deliverables": "scope of work, bill of quantities, required services, tasks to perform, and project deliverables",
    "General Administrative Provisions": "general instructions to bidders, notices, communications, and administrative rules",
}


@dataclass
class ClassificationResult:
    """Detailed result of Transformer-based clause classification."""
    clause_id: str
    predicted_category: str
    confidence: float
    all_scores: Dict[str, float]
    model_identifier: str
    method: str = "SentenceTransformer-Prototype-Classifier"


class ClauseClassifier:
    """Pretrained Transformer classifier using dense contextual representations and category prototypes."""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or DEFAULT_CONFIG.models
        self.model_name = self.config.embedding_model_name
        self._model: Optional[SentenceTransformer] = None
        self._anchor_embeddings: Optional[np.ndarray] = None
        self._category_names = list(CATEGORY_ANCHORS.keys())

    def _load_model(self) -> SentenceTransformer:
        """Lazy load the SentenceTransformer model and encode semantic category anchors."""
        if self._model is None:
            logger.info("Loading Transformer classifier model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            anchor_texts = [f"{cat}: {CATEGORY_ANCHORS[cat]}" for cat in self._category_names]
            self._anchor_embeddings = self._model.encode(anchor_texts, normalize_embeddings=True)
        return self._model

    def classify_text(
        self,
        text: str,
        candidate_labels: Optional[List[str]] = None,
        clause_id: str = "ADHOC"
    ) -> ClassificationResult:
        """Classify a single text snippet using genuine Transformer contextual representations."""
        model = self._load_model()
        clean_text = text.strip() if text else "General contractual terms"
        if len(clean_text) < 10:
            clean_text = f"General contractual provision: {clean_text}"

        text_emb = model.encode([clean_text], normalize_embeddings=True)
        sims = cosine_similarity(text_emb, self._anchor_embeddings)[0]

        # Temperature-scaled softmax over similarities
        temperature = 5.0
        exp_sims = np.exp(sims * temperature)
        probs = exp_sims / np.sum(exp_sims)

        scores_dict: Dict[str, float] = {}
        for cat, prob in zip(self._category_names, probs):
            scores_dict[cat] = round(float(prob), 4)

        top_idx = int(np.argmax(probs))
        top_cat = self._category_names[top_idx]
        top_confidence = round(float(probs[top_idx]), 4)

        return ClassificationResult(
            clause_id=clause_id,
            predicted_category=top_cat,
            confidence=top_confidence,
            all_scores=scores_dict,
            model_identifier=self.model_name,
            method="SentenceTransformer-Dense-Contextual-Prototype"
        )

    def classify_clauses(
        self,
        clauses: List[Union[SegmentedClause, Dict[str, str]]],
        candidate_labels: Optional[List[str]] = None,
        batch_size: int = 32
    ) -> List[ClassificationResult]:
        """Batch classify multiple clauses using vector matrix operations."""
        if not clauses:
            return []

        model = self._load_model()
        texts = []
        clause_ids = []

        for c in clauses:
            if isinstance(c, SegmentedClause):
                clause_ids.append(c.id)
                t = c.original_text.strip()
            else:
                clause_ids.append(c.get("id", "UNKNOWN"))
                t = c.get("original_text", "").strip()
            
            if len(t) < 10:
                t = f"General contractual provision: {t}"
            texts.append(t)

        embeddings = model.encode(texts, batch_size=batch_size, normalize_embeddings=True)
        sim_matrix = cosine_similarity(embeddings, self._anchor_embeddings)

        temperature = 5.0
        exp_sims = np.exp(sim_matrix * temperature)
        probs_matrix = exp_sims / np.sum(exp_sims, axis=1, keepdims=True)

        results: List[ClassificationResult] = []
        for i, cid in enumerate(clause_ids):
            row_probs = probs_matrix[i]
            scores_dict = {cat: round(float(row_probs[j]), 4) for j, cat in enumerate(self._category_names)}
            top_idx = int(np.argmax(row_probs))
            top_cat = self._category_names[top_idx]
            top_conf = round(float(row_probs[top_idx]), 4)

            results.append(
                ClassificationResult(
                    clause_id=cid,
                    predicted_category=top_cat,
                    confidence=top_conf,
                    all_scores=scores_dict,
                    model_identifier=self.model_name,
                    method="SentenceTransformer-Dense-Contextual-Prototype"
                )
            )

        return results
