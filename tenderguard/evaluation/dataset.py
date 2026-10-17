"""Benchmark dataset loader and validator for tender contradiction evaluation."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union
from tenderguard.config import DATA_DIR


@dataclass
class BenchmarkCase:
    """Represents a manually or synthetically annotated clause pair for evaluation."""
    id: str
    case_type: str                   # "numerical_conflict", "date_conflict", "complementary_requirements", etc.
    clause_a: str
    clause_b: str
    ground_truth_label: int          # 1 = Contradiction / Inconsistency, 0 = Consistent / Non-conflict
    description: str


def load_benchmark_dataset(file_path: Optional[Union[str, Path]] = None) -> List[BenchmarkCase]:
    """Load and validate benchmark evaluation cases from JSON storage."""
    path = Path(file_path) if file_path else (DATA_DIR / "benchmark_cases.json")
    if not path.is_file():
        return []

    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    cases: List[BenchmarkCase] = []
    for item in raw_data:
        case = BenchmarkCase(
            id=item["id"],
            case_type=item["case_type"],
            clause_a=item["clause_a"].strip(),
            clause_b=item["clause_b"].strip(),
            ground_truth_label=int(item["ground_truth_label"]),
            description=item.get("description", "")
        )
        cases.append(case)

    return cases
