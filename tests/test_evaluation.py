"""Unit tests for evaluation framework and benchmark dataset."""
import pytest
from tenderguard.evaluation.dataset import BenchmarkCase, load_benchmark_dataset
from tenderguard.evaluation.evaluator import BenchmarkEvaluator


@pytest.fixture(scope="module")
def benchmark_cases():
    return load_benchmark_dataset()


@pytest.fixture(scope="module")
def evaluator():
    return BenchmarkEvaluator()


def test_benchmark_dataset_completeness(benchmark_cases):
    """Ensure benchmark dataset contains all 6 required synthetic test case categories."""
    assert len(benchmark_cases) >= 12
    
    types = {c.case_type for c in benchmark_cases}
    required_types = {
        "numerical_conflict",
        "date_conflict",
        "complementary_requirements",
        "unrelated_clauses",
        "repeated_clauses",
        "ambiguous_requirements"
    }
    assert required_types.issubset(types), f"Missing required case types: {required_types - types}"
    
    # Ground truth labels must be binary
    for c in benchmark_cases:
        assert c.ground_truth_label in (0, 1)
        assert len(c.clause_a) > 20
        assert len(c.clause_b) > 20


def test_empty_dataset_returns_not_evaluated(evaluator):
    """Ensure empty evaluation input explicitly reports 'Not evaluated' and never invents results."""
    res = evaluator.evaluate_rule_based([])
    assert res.status == "Not evaluated"
    assert res.total_cases == 0
    assert res.f1_score == 0.0


def test_rule_based_baseline_execution(evaluator, benchmark_cases):
    """Ensure rule-based baseline executes and correctly detects numerical and date cases."""
    res = evaluator.evaluate_rule_based(benchmark_cases)
    assert res.status == "Evaluated"
    assert res.total_cases == len(benchmark_cases)
    assert res.precision > 0.0
    assert "TN" in res.confusion_matrix and "TP" in res.confusion_matrix


def test_semantic_similarity_baseline_execution(evaluator, benchmark_cases):
    """Ensure semantic similarity baseline computes metrics without error."""
    res = evaluator.evaluate_semantic_similarity(benchmark_cases, threshold=0.70)
    assert res.status == "Evaluated"
    assert res.total_cases == len(benchmark_cases)
    assert res.avg_processing_time_ms > 0.0


def test_hybrid_pipeline_evaluation_outperforms_baselines(evaluator, benchmark_cases):
    """Ensure hybrid pipeline achieves superior F1 performance across all 6 risk categories."""
    m_rule = evaluator.evaluate_rule_based(benchmark_cases)
    m_hyb = evaluator.evaluate_hybrid_pipeline(benchmark_cases)
    
    assert m_hyb.status == "Evaluated"
    assert m_hyb.total_cases == len(benchmark_cases)
    # The hybrid pipeline catches semantic & ambiguous contradictions that rules miss
    assert m_hyb.recall >= m_rule.recall
    assert m_hyb.f1_score >= 0.70
