"""Benchmarking and evaluation framework comparing baseline and hybrid detection pipelines."""
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from tenderguard.analysis.date_deadline import DateDeadlineAnalyzer
from tenderguard.analysis.hybrid_engine import HybridEngine
from tenderguard.analysis.numerical import NumericalAnalyzer
from tenderguard.evaluation.dataset import BenchmarkCase
from tenderguard.models.embeddings import CandidatePair, SemanticRetriever
from tenderguard.models.nli_engine import NLIEngine

logger = logging.getLogger(__name__)


@dataclass
class PipelineMetrics:
    """Evaluation metrics for a specific pipeline approach."""
    pipeline_name: str
    status: str                         # "Evaluated" or "Not evaluated"
    total_cases: int
    precision: float
    recall: float
    f1_score: float
    fpr: float                          # False Positive Rate
    confusion_matrix: Dict[str, int]    # {"TN": int, "FP": int, "FN": int, "TP": int}
    avg_processing_time_ms: float
    details: List[Dict[str, Any]]


class BenchmarkEvaluator:
    """Executes comparative evaluation across Rule-based, Semantic, and Hybrid pipelines."""

    def __init__(self):
        self.numerical_analyzer = NumericalAnalyzer()
        self.date_analyzer = DateDeadlineAnalyzer()
        self._retriever: Optional[SemanticRetriever] = None
        self._nli_engine: Optional[NLIEngine] = None
        self._hybrid_engine: Optional[HybridEngine] = None

    @property
    def retriever(self) -> SemanticRetriever:
        if self._retriever is None:
            self._retriever = SemanticRetriever()
        return self._retriever

    @property
    def nli_engine(self) -> NLIEngine:
        if self._nli_engine is None:
            self._nli_engine = NLIEngine()
        return self._nli_engine

    @property
    def hybrid_engine(self) -> HybridEngine:
        if self._hybrid_engine is None:
            self._hybrid_engine = HybridEngine()
        return self._hybrid_engine

    def evaluate_rule_based(self, cases: List[BenchmarkCase]) -> PipelineMetrics:
        """Evaluate baseline 1: Rule-based Numerical and Date checks only."""
        if not cases:
            return self._not_evaluated("Rule-based baseline")

        y_true = []
        y_pred = []
        details = []
        start_time = time.time()

        for c in cases:
            y_true.append(c.ground_truth_label)
            num_res = self.numerical_analyzer.compare_clauses(c.clause_a, c.clause_b, "A", "B")
            date_res = self.date_analyzer.compare_dates(c.clause_a, c.clause_b, "A", "B")

            pred = 1 if (num_res or date_res) else 0
            y_pred.append(pred)
            details.append({
                "case_id": c.id,
                "case_type": c.case_type,
                "true_label": c.ground_truth_label,
                "pred_label": pred,
                "is_correct": bool(c.ground_truth_label == pred),
                "reason": (num_res.explanation if num_res else (date_res.explanation if date_res else "No rule triggered"))
            })

        total_time_ms = (time.time() - start_time) * 1000.0
        return self._compute_metrics("Rule-based baseline", y_true, y_pred, total_time_ms, details)

    def evaluate_semantic_similarity(self, cases: List[BenchmarkCase], threshold: float = 0.70) -> PipelineMetrics:
        """Evaluate baseline 2: SentenceTransformer cosine similarity alone."""
        if not cases:
            return self._not_evaluated("Semantic similarity baseline")

        y_true = []
        y_pred = []
        details = []
        start_time = time.time()

        # Batch encode for speed
        texts_a = [c.clause_a for c in cases]
        texts_b = [c.clause_b for c in cases]
        embs_a = self.retriever.encode(texts_a)
        embs_b = self.retriever.encode(texts_b)

        # Dot product of normalized embeddings = cosine similarity
        sims = np.sum(embs_a * embs_b, axis=1)

        for c, sim in zip(cases, sims):
            y_true.append(c.ground_truth_label)
            # High semantic similarity used naively as an indicator of conflicting topic
            pred = 1 if sim >= threshold else 0
            y_pred.append(pred)
            details.append({
                "case_id": c.id,
                "case_type": c.case_type,
                "true_label": c.ground_truth_label,
                "pred_label": pred,
                "is_correct": bool(c.ground_truth_label == pred),
                "reason": f"Cosine similarity: {sim:.3f} (Threshold: {threshold:.2f})"
            })

        total_time_ms = (time.time() - start_time) * 1000.0
        return self._compute_metrics("Semantic similarity baseline", y_true, y_pred, total_time_ms, details)

    def evaluate_hybrid_pipeline(self, cases: List[BenchmarkCase]) -> PipelineMetrics:
        """Evaluate full system: SentenceTransformer + NLI Transformer + Dedicated Rules."""
        if not cases:
            return self._not_evaluated("Hybrid Pipeline (ST + NLI + Rules)")

        y_true = []
        y_pred = []
        details = []
        start_time = time.time()

        # Prepare candidate pairs
        texts_a = [c.clause_a for c in cases]
        texts_b = [c.clause_b for c in cases]
        embs_a = self.retriever.encode(texts_a)
        embs_b = self.retriever.encode(texts_b)
        sims = np.sum(embs_a * embs_b, axis=1)

        for idx, (c, sim) in enumerate(zip(cases, sims)):
            y_true.append(c.ground_truth_label)

            cand_pair = CandidatePair(
                doc_id="BENCH",
                clause_a_id="A",
                clause_b_id="B",
                clause_a_text=c.clause_a,
                clause_b_text=c.clause_b,
                similarity_score=float(sim),
                embedding_model="all-MiniLM-L6-v2",
                retrieval_method="BenchmarkEvaluation"
            )

            # Pretrained NLI inference
            nli_pred = self.nli_engine.predict_pair(c.clause_a, c.clause_b, "A", "B", bidirectional=True)

            # Hybrid synthesis
            finding = self.hybrid_engine.analyze_pair(cand_pair, nli_pred, f"FND_B_{idx:03d}")
            pred = 1 if finding is not None else 0
            y_pred.append(pred)

            reason = finding.explanation if finding else f"No inconsistency detected (NLI: {nli_pred.predicted_label}, sim: {sim:.2f})"
            details.append({
                "case_id": c.id,
                "case_type": c.case_type,
                "true_label": c.ground_truth_label,
                "pred_label": pred,
                "is_correct": bool(c.ground_truth_label == pred),
                "reason": reason
            })

        total_time_ms = (time.time() - start_time) * 1000.0
        return self._compute_metrics("Hybrid Pipeline (ST + NLI + Rules)", y_true, y_pred, total_time_ms, details)

    def run_all_benchmarks(self, cases: Optional[List[BenchmarkCase]] = None) -> List[PipelineMetrics]:
        """Execute and return comparative metrics for all three approaches."""
        if not cases:
            return [
                self._not_evaluated("Rule-based baseline"),
                self._not_evaluated("Semantic similarity baseline"),
                self._not_evaluated("Hybrid Pipeline (ST + NLI + Rules)")
            ]

        m_rule = self.evaluate_rule_based(cases)
        m_sem = self.evaluate_semantic_similarity(cases)
        m_hyb = self.evaluate_hybrid_pipeline(cases)

        return [m_rule, m_sem, m_hyb]

    def _compute_metrics(
        self,
        name: str,
        y_true: List[int],
        y_pred: List[int],
        total_time_ms: float,
        details: List[Dict[str, Any]]
    ) -> PipelineMetrics:
        """Compute precision, recall, F1, FPR, and confusion matrix using scikit-learn."""
        p = float(precision_score(y_true, y_pred, zero_division=0))
        r = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        n = len(y_true)
        avg_time = total_time_ms / n if n > 0 else 0.0

        return PipelineMetrics(
            pipeline_name=name,
            status="Evaluated",
            total_cases=n,
            precision=round(p, 4),
            recall=round(r, 4),
            f1_score=round(f1, 4),
            fpr=round(fpr, 4),
            confusion_matrix={"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
            avg_processing_time_ms=round(avg_time, 2),
            details=details
        )

    def _not_evaluated(self, name: str) -> PipelineMetrics:
        """Return explicit 'Not evaluated' record when no evaluation dataset is available."""
        return PipelineMetrics(
            pipeline_name=name,
            status="Not evaluated",
            total_cases=0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            fpr=0.0,
            confusion_matrix={"TN": 0, "FP": 0, "FN": 0, "TP": 0},
            avg_processing_time_ms=0.0,
            details=[]
        )
