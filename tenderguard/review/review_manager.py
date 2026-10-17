"""Review manager separating human decisions from immutable AI model predictions."""
import logging
from typing import Any, Dict, List, Optional
from tenderguard.db.session import DatabaseManager

logger = logging.getLogger(__name__)

REVIEW_STATUS_CHOICES = [
    "Pending Review",
    "Confirmed Risk",
    "False Positive",
    "Clarification Required"
]


class ReviewManager:
    """Manages procurement specialist review actions on AI-generated findings."""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def submit_review(
        self,
        finding_id: str,
        status: str,
        comment: str,
        reviewer_name: str = "Procurement Specialist"
    ) -> None:
        """Record or update a human review decision."""
        if status not in REVIEW_STATUS_CHOICES:
            raise ValueError(f"Invalid review status '{status}'. Must be one of {REVIEW_STATUS_CHOICES}.")

        self.db.upsert_review(
            finding_id=finding_id,
            status=status,
            reviewer_comment=comment.strip(),
            reviewer_name=reviewer_name.strip()
        )
        logger.info("Updated review for finding %s: %s by %s", finding_id, status, reviewer_name)

    def get_review_summary(self, doc_id: Optional[str] = None) -> Dict[str, int]:
        """Compute aggregate counts of review decisions."""
        findings = self.db.get_findings_with_reviews(doc_id=doc_id)
        summary = {status: 0 for status in REVIEW_STATUS_CHOICES}
        for f in findings:
            st = f.get("review_status", "Pending Review")
            if st in summary:
                summary[st] += 1
            else:
                summary["Pending Review"] += 1
        return summary
