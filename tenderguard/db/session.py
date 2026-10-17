"""Database session manager and CRUD operations for TenderGuard."""
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from tenderguard.config import DEFAULT_DB_PATH
from tenderguard.db.schema import SCHEMA_SQL


class DatabaseManager:
    """Encapsulates SQLite connection management and typed repository operations."""

    def __init__(self, db_path: Union[str, Path] = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        """Create and return a configured SQLite connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_db(self) -> None:
        """Execute DDL statements to create tables and indexes."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.get_connection() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()
        self.seed_default_user()

    # ------------------ Document Operations ------------------

    def save_document(self, doc_data: Dict[str, Any]) -> str:
        """Insert or replace document record."""
        sql = """
        INSERT OR REPLACE INTO documents 
        (id, filename, file_hash, total_pages, file_size_bytes, scanned_pages_count, status, error_message)
        VALUES (:id, :filename, :file_hash, :total_pages, :file_size_bytes, :scanned_pages_count, :status, :error_message)
        """
        with self.get_connection() as conn:
            conn.execute(sql, {
                "id": doc_data["id"],
                "filename": doc_data["filename"],
                "file_hash": doc_data["file_hash"],
                "total_pages": doc_data.get("total_pages", 0),
                "file_size_bytes": doc_data.get("file_size_bytes", 0),
                "scanned_pages_count": doc_data.get("scanned_pages_count", 0),
                "status": doc_data.get("status", "uploaded"),
                "error_message": doc_data.get("error_message")
            })
            conn.commit()
        return doc_data["id"]

    def update_document_status(self, doc_id: str, status: str, scanned_pages: Optional[int] = None, error_message: Optional[str] = None) -> None:
        """Update processing status of a document."""
        with self.get_connection() as conn:
            if scanned_pages is not None:
                conn.execute(
                    "UPDATE documents SET status = ?, scanned_pages_count = ?, error_message = ? WHERE id = ?",
                    (status, scanned_pages, error_message, doc_id)
                )
            else:
                conn.execute(
                    "UPDATE documents SET status = ?, error_message = ? WHERE id = ?",
                    (status, error_message, doc_id)
                )
            conn.commit()

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve document metadata by ID."""
        with self.get_connection() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
            return dict(row) if row else None

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents sorted by upload time descending."""
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY uploaded_at DESC").fetchall()
            return [dict(r) for r in rows]

    def delete_document(self, doc_id: str) -> None:
        """Cascade delete a document and its dependent data."""
        with self.get_connection() as conn:
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            conn.commit()

    # ------------------ Page Operations ------------------

    def save_pages(self, pages: List[Dict[str, Any]]) -> None:
        """Batch insert extracted pages."""
        sql = """
        INSERT INTO pages (doc_id, page_number, char_count, is_scanned_or_empty, raw_text)
        VALUES (:doc_id, :page_number, :char_count, :is_scanned_or_empty, :raw_text)
        """
        with self.get_connection() as conn:
            conn.executemany(sql, pages)
            conn.commit()

    def get_pages(self, doc_id: str) -> List[Dict[str, Any]]:
        """Retrieve all pages for a document ordered by page_number."""
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM pages WHERE doc_id = ? ORDER BY page_number ASC", (doc_id,)).fetchall()
            return [dict(r) for r in rows]

    # ------------------ Clause Operations ------------------

    def save_clauses(self, clauses: List[Dict[str, Any]]) -> None:
        """Batch insert segmented clauses."""
        sql = """
        INSERT OR REPLACE INTO clauses 
        (id, doc_id, page_number, clause_number, original_text, char_start, char_end, category, category_confidence, classification_model)
        VALUES (:id, :doc_id, :page_number, :clause_number, :original_text, :char_start, :char_end, :category, :category_confidence, :classification_model)
        """
        with self.get_connection() as conn:
            conn.executemany(sql, clauses)
            conn.commit()

    def get_clauses(self, doc_id: str, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve clauses for a document with optional category filter."""
        with self.get_connection() as conn:
            if category:
                rows = conn.execute(
                    "SELECT * FROM clauses WHERE doc_id = ? AND category = ? ORDER BY page_number ASC, id ASC",
                    (doc_id, category)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM clauses WHERE doc_id = ? ORDER BY page_number ASC, id ASC",
                    (doc_id,)
                ).fetchall()
            return [dict(r) for r in rows]

    def update_clause_category(self, clause_id: str, category: str, confidence: float, model_name: str) -> None:
        """Update classification results for a single clause."""
        with self.get_connection() as conn:
            conn.execute(
                "UPDATE clauses SET category = ?, category_confidence = ?, classification_model = ? WHERE id = ?",
                (category, confidence, model_name, clause_id)
            )
            conn.commit()

    # ------------------ Candidate Pairs ------------------

    def save_candidate_pairs(self, pairs: List[Dict[str, Any]]) -> None:
        """Batch insert candidate clause pairs."""
        sql = """
        INSERT INTO candidate_pairs (doc_id, clause_a_id, clause_b_id, similarity_score, embedding_model, retrieval_method)
        VALUES (:doc_id, :clause_a_id, :clause_b_id, :similarity_score, :embedding_model, :retrieval_method)
        """
        with self.get_connection() as conn:
            conn.executemany(sql, pairs)
            conn.commit()

    def get_candidate_pairs(self, doc_id: str) -> List[Dict[str, Any]]:
        """Retrieve candidate pairs for a document."""
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM candidate_pairs WHERE doc_id = ? ORDER BY similarity_score DESC",
                (doc_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    # ------------------ Findings Operations ------------------

    def save_findings(self, findings: List[Dict[str, Any]]) -> None:
        """Insert findings and initialize default pending review records."""
        f_sql = """
        INSERT OR REPLACE INTO findings 
        (id, doc_id, finding_type, clause_a_id, clause_b_id, clause_a_text, clause_b_text, 
         page_numbers, semantic_similarity, nli_label, nli_contradiction_score, 
         nli_entailment_score, nli_neutral_score, rule_triggered, explanation, priority, model_identifier)
        VALUES 
        (:id, :doc_id, :finding_type, :clause_a_id, :clause_b_id, :clause_a_text, :clause_b_text,
         :page_numbers, :semantic_similarity, :nli_label, :nli_contradiction_score,
         :nli_entailment_score, :nli_neutral_score, :rule_triggered, :explanation, :priority, :model_identifier)
        """
        rev_sql = """
        INSERT OR IGNORE INTO reviews (finding_id, status, reviewer_name, reviewer_comment)
        VALUES (?, 'Pending Review', 'System Default', '')
        """
        with self.get_connection() as conn:
            for f in findings:
                conn.execute(f_sql, f)
                conn.execute(rev_sql, (f["id"],))
            conn.commit()

    def get_findings_with_reviews(
        self,
        doc_id: Optional[str] = None,
        priority: Optional[str] = None,
        review_status: Optional[str] = None,
        finding_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve findings joined with their corresponding human review decisions."""
        query = """
        SELECT f.*, 
               r.status AS review_status, 
               r.reviewer_name, 
               r.reviewer_comment, 
               r.reviewed_at
        FROM findings f
        LEFT JOIN reviews r ON f.id = r.finding_id
        WHERE 1=1
        """
        params: List[Any] = []
        if doc_id:
            query += " AND f.doc_id = ?"
            params.append(doc_id)
        if priority:
            query += " AND f.priority = ?"
            params.append(priority)
        if review_status:
            query += " AND r.status = ?"
            params.append(review_status)
        if finding_type:
            query += " AND f.finding_type = ?"
            params.append(finding_type)

        query += " ORDER BY CASE f.priority WHEN 'High' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END, f.id ASC"

        with self.get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    # ------------------ Human Review Operations ------------------

    def upsert_review(self, finding_id: str, status: str, reviewer_comment: str, reviewer_name: str = "Human Reviewer") -> None:
        """Upsert human review decision, cleanly separated from AI model prediction."""
        sql = """
        INSERT INTO reviews (finding_id, status, reviewer_comment, reviewer_name, reviewed_at)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(finding_id) DO UPDATE SET
            status = excluded.status,
            reviewer_comment = excluded.reviewer_comment,
            reviewer_name = excluded.reviewer_name,
            reviewed_at = CURRENT_TIMESTAMP
        """
        with self.get_connection() as conn:
            conn.execute(sql, (finding_id, status, reviewer_comment, reviewer_name))
            conn.commit()

    # ------------------ Evaluation Records ------------------

    def save_evaluation(self, eval_data: Dict[str, Any]) -> int:
        """Persist evaluation benchmark results."""
        sql = """
        INSERT INTO evaluations 
        (dataset_name, pipeline_name, precision, recall, f1_score, fpr, confusion_matrix_json, avg_processing_time_ms, total_cases, notes)
        VALUES (:dataset_name, :pipeline_name, :precision, :recall, :f1_score, :fpr, :confusion_matrix_json, :avg_processing_time_ms, :total_cases, :notes)
        """
        with self.get_connection() as conn:
            cursor = conn.execute(sql, {
                "dataset_name": eval_data["dataset_name"],
                "pipeline_name": eval_data["pipeline_name"],
                "precision": eval_data["precision"],
                "recall": eval_data["recall"],
                "f1_score": eval_data["f1_score"],
                "fpr": eval_data["fpr"],
                "confusion_matrix_json": json.dumps(eval_data.get("confusion_matrix", {})),
                "avg_processing_time_ms": eval_data.get("avg_processing_time_ms", 0.0),
                "total_cases": eval_data.get("total_cases", 0),
                "notes": eval_data.get("notes", "")
            })
            conn.commit()
            return cursor.lastrowid

    def list_evaluations(self) -> List[Dict[str, Any]]:
        """Retrieve historical evaluation benchmark runs."""
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM evaluations ORDER BY evaluated_at DESC").fetchall()
            results = []
            for r in rows:
                d = dict(r)
                try:
                    d["confusion_matrix"] = json.loads(d["confusion_matrix_json"])
                except Exception:
                    d["confusion_matrix"] = {}
                results.append(d)
            return results

    # ------------------ User Authentication Operations ------------------

    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """Hash password using SHA-256 with a unique salt."""
        import hashlib
        import secrets
        if salt is None:
            salt = secrets.token_hex(16)
        salted = f"{salt}{password}".encode("utf-8")
        pwd_hash = hashlib.sha256(salted).hexdigest()
        return pwd_hash, salt

    def create_user(
        self,
        username: str,
        email: str,
        full_name: str,
        password: str,
        organization: str = "",
        role: str = "Procurement Specialist"
    ) -> int:
        """Register a new user with hashed credentials."""
        clean_user = username.strip().lower()
        clean_email = email.strip().lower()
        pwd_hash, salt = self.hash_password(password)

        sql = """
        INSERT INTO users (username, email, full_name, organization, password_hash, salt, role)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                sql,
                (clean_user, clean_email, full_name.strip(), organization.strip(), pwd_hash, salt, role)
            )
            conn.commit()
            return cursor.lastrowid

    def authenticate_user(self, username_or_email: str, password: str) -> Optional[Dict[str, Any]]:
        """Verify user credentials and return user record if valid."""
        clean_ident = username_or_email.strip().lower()
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ? OR email = ?",
                (clean_ident, clean_ident)
            ).fetchone()
            if not row:
                return None
            
            user = dict(row)
            expected_hash, _ = self.hash_password(password, salt=user["salt"])
            if expected_hash == user["password_hash"]:
                # Exclude sensitive salt and hash from returned dict
                user.pop("password_hash", None)
                user.pop("salt", None)
                return user
            return None

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Retrieve user record by username."""
        with self.get_connection() as conn:
            row = conn.execute("SELECT * FROM users WHERE username = ?", (username.strip().lower(),)).fetchone()
            if row:
                d = dict(row)
                d.pop("password_hash", None)
                d.pop("salt", None)
                return d
            return None

    def seed_default_user(self) -> None:
        """Seed a default procurement specialist account if users table is empty."""
        with self.get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) AS cnt FROM users").fetchone()["cnt"]
            if count == 0:
                self.create_user(
                    username="auditor",
                    email="auditor@tenderguard.org",
                    full_name="Senior Procurement Officer",
                    password="tenderguard2026",
                    organization="Public Procurement Integrity Bureau",
                    role="Chief Procurement Auditor"
                )

