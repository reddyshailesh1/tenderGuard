"""Hybrid analysis engine synthesizing Transformer NLI, semantic retrieval, and rule-based detectors."""
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from tenderguard.analysis.date_deadline import DateDeadlineAnalyzer, DateInconsistency
from tenderguard.analysis.numerical import NumericalAnalyzer, NumericalDiscrepancy
from tenderguard.models.embeddings import CandidatePair
from tenderguard.models.nli_engine import NLIPrediction

logger = logging.getLogger(__name__)


@dataclass
class Finding:
    """Represents a verified, evidence-backed risk or contradiction finding."""
    id: str
    doc_id: str
    finding_type: str
    clause_a_id: str
    clause_b_id: Optional[str]
    clause_a_text: str
    clause_b_text: Optional[str]
    page_numbers: str
    semantic_similarity: Optional[float]
    nli_label: Optional[str]
    nli_contradiction_score: Optional[float]
    nli_entailment_score: Optional[float]
    nli_neutral_score: Optional[float]
    rule_triggered: str
    explanation: str
    priority: str                       # "High", "Medium", "Low"
    model_identifier: str
    review_status: str = "Pending Review"
    reviewer_comment: str = ""


class HybridEngine:
    """Synthesizes Transformer NLI probabilities and dedicated rule engines into evidence-backed findings."""

    def __init__(self):
        self.numerical_analyzer = NumericalAnalyzer()
        self.date_analyzer = DateDeadlineAnalyzer()

    def analyze_pair(
        self,
        candidate_pair: CandidatePair,
        nli_pred: NLIPrediction,
        finding_id: str
    ) -> Optional[Finding]:
        """Combine NLI prediction, numerical checks, and date checks for a candidate pair."""
        text_a = candidate_pair.clause_a_text
        text_b = candidate_pair.clause_b_text
        cid_a = candidate_pair.clause_a_id
        cid_b = candidate_pair.clause_b_id
        pages_str = f"Page {candidate_pair.page_a or '?'}, Page {candidate_pair.page_b or '?'}"

        # 1. Run numerical analysis
        num_conflict = self.numerical_analyzer.compare_clauses(text_a, text_b, cid_a, cid_b)

        # 2. Run date & deadline analysis
        date_conflict = self.date_analyzer.compare_dates(text_a, text_b, cid_a, cid_b)

        # 3. Evaluate NLI contradiction
        is_nli_contra = nli_pred.is_contradiction

        # Combined Synthesis Rules

        # Case A: Both NLI Contradiction and Numerical Conflict detected (Strongest Evidence)
        if is_nli_contra and num_conflict:
            explanation = (
                f"Potential critical contradiction and numerical discrepancy. "
                f"Pretrained NLI model identified high logical contradiction risk ({nli_pred.contradiction_prob:.1%}). "
                f"{num_conflict.explanation}"
            )
            return Finding(
                id=finding_id,
                doc_id=candidate_pair.doc_id,
                finding_type="Hybrid Contradiction & Numerical Conflict",
                clause_a_id=cid_a,
                clause_b_id=cid_b,
                clause_a_text=text_a,
                clause_b_text=text_b,
                page_numbers=pages_str,
                semantic_similarity=candidate_pair.similarity_score,
                nli_label=nli_pred.predicted_label,
                nli_contradiction_score=nli_pred.contradiction_prob,
                nli_entailment_score=nli_pred.entailment_prob,
                nli_neutral_score=nli_pred.neutral_prob,
                rule_triggered=f"{num_conflict.rule_id} + NLI_CONTRADICTION",
                explanation=explanation,
                priority="High",
                model_identifier=f"{candidate_pair.embedding_model} + {nli_pred.model_identifier}"
            )

        # Case B: Both NLI Contradiction and Date Conflict detected
        if is_nli_contra and date_conflict:
            explanation = (
                f"Potential critical deadline contradiction. "
                f"NLI model identified contradiction risk ({nli_pred.contradiction_prob:.1%}). "
                f"{date_conflict.explanation}"
            )
            return Finding(
                id=finding_id,
                doc_id=candidate_pair.doc_id,
                finding_type="Hybrid Contradiction & Deadline Conflict",
                clause_a_id=cid_a,
                clause_b_id=cid_b,
                clause_a_text=text_a,
                clause_b_text=text_b,
                page_numbers=pages_str,
                semantic_similarity=candidate_pair.similarity_score,
                nli_label=nli_pred.predicted_label,
                nli_contradiction_score=nli_pred.contradiction_prob,
                nli_entailment_score=nli_pred.entailment_prob,
                nli_neutral_score=nli_pred.neutral_prob,
                rule_triggered=f"{date_conflict.rule_id} + NLI_CONTRADICTION",
                explanation=explanation,
                priority="High",
                model_identifier=f"{candidate_pair.embedding_model} + {nli_pred.model_identifier}"
            )

        # Case C: Numerical rule triggered independently (even if NLI was neutral/sub-threshold)
        if num_conflict:
            return Finding(
                id=finding_id,
                doc_id=candidate_pair.doc_id,
                finding_type="Numerical Inconsistency",
                clause_a_id=cid_a,
                clause_b_id=cid_b,
                clause_a_text=text_a,
                clause_b_text=text_b,
                page_numbers=pages_str,
                semantic_similarity=candidate_pair.similarity_score,
                nli_label=nli_pred.predicted_label,
                nli_contradiction_score=nli_pred.contradiction_prob,
                nli_entailment_score=nli_pred.entailment_prob,
                nli_neutral_score=nli_pred.neutral_prob,
                rule_triggered=num_conflict.rule_id,
                explanation=num_conflict.explanation,
                priority=num_conflict.priority,
                model_identifier=f"{candidate_pair.embedding_model} + {nli_pred.model_identifier} + RuleEngine"
            )

        # Case D: Date rule triggered independently
        if date_conflict:
            return Finding(
                id=finding_id,
                doc_id=candidate_pair.doc_id,
                finding_type="Date/Deadline Inconsistency",
                clause_a_id=cid_a,
                clause_b_id=cid_b,
                clause_a_text=text_a,
                clause_b_text=text_b,
                page_numbers=pages_str,
                semantic_similarity=candidate_pair.similarity_score,
                nli_label=nli_pred.predicted_label,
                nli_contradiction_score=nli_pred.contradiction_prob,
                nli_entailment_score=nli_pred.entailment_prob,
                nli_neutral_score=nli_pred.neutral_prob,
                rule_triggered=date_conflict.rule_id,
                explanation=date_conflict.explanation,
                priority=date_conflict.priority,
                model_identifier=f"{candidate_pair.embedding_model} + {nli_pred.model_identifier} + DateEngine"
            )

        # Case E: High NLI Contradiction with High Semantic Similarity (Pure Semantic Contradiction)
        if is_nli_contra and candidate_pair.similarity_score >= 0.50:
            priority = "High" if nli_pred.contradiction_prob >= 0.70 else "Medium"
            explanation = (
                f"Potential logical contradiction identified by pretrained NLI model "
                f"(contradiction score: {nli_pred.contradiction_prob:.1%}, semantic similarity: {candidate_pair.similarity_score:.2f}). "
                f"The requirements in Clause {cid_a} and Clause {cid_b} appear logically incompatible. "
                f"Requires human review to assess legal and operational impact."
            )
            return Finding(
                id=finding_id,
                doc_id=candidate_pair.doc_id,
                finding_type="NLI Semantic Contradiction",
                clause_a_id=cid_a,
                clause_b_id=cid_b,
                clause_a_text=text_a,
                clause_b_text=text_b,
                page_numbers=pages_str,
                semantic_similarity=candidate_pair.similarity_score,
                nli_label=nli_pred.predicted_label,
                nli_contradiction_score=nli_pred.contradiction_prob,
                nli_entailment_score=nli_pred.entailment_prob,
                nli_neutral_score=nli_pred.neutral_prob,
                rule_triggered="RULE_NLI_LOGICAL_CONTRADICTION",
                explanation=explanation,
                priority=priority,
                model_identifier=f"{candidate_pair.embedding_model} + {nli_pred.model_identifier}"
            )

        return None

    def synthesize_all(
        self,
        candidate_pairs: List[CandidatePair],
        nli_predictions: List[NLIPrediction],
        doc_id: str
    ) -> List[Finding]:
        """Synthesize findings across all candidate pairs with deduplication and unique finding IDs."""
        findings: List[Finding] = []
        finding_idx = 1

        for pair, nli_pred in zip(candidate_pairs, nli_predictions):
            f_id = f"FND_{doc_id}_{finding_idx:03d}"
            finding = self.analyze_pair(pair, nli_pred, f_id)
            if finding:
                findings.append(finding)
                finding_idx += 1

        logger.info("Hybrid analysis generated %d evidence-backed findings for doc %s.", len(findings), doc_id)
        return findings
