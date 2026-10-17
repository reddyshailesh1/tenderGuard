"""Database schema definitions for TenderGuard using SQLite."""

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    total_pages INTEGER NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    scanned_pages_count INTEGER DEFAULT 0,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT NOT NULL,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS pages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    char_count INTEGER NOT NULL,
    is_scanned_or_empty BOOLEAN NOT NULL DEFAULT 0,
    raw_text TEXT NOT NULL,
    FOREIGN KEY(doc_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS clauses (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    clause_number TEXT,
    original_text TEXT NOT NULL,
    char_start INTEGER,
    char_end INTEGER,
    category TEXT,
    category_confidence REAL,
    classification_model TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(doc_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS candidate_pairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL,
    clause_a_id TEXT NOT NULL,
    clause_b_id TEXT NOT NULL,
    similarity_score REAL NOT NULL,
    embedding_model TEXT NOT NULL,
    retrieval_method TEXT NOT NULL,
    FOREIGN KEY(doc_id) REFERENCES documents(id) ON DELETE CASCADE,
    FOREIGN KEY(clause_a_id) REFERENCES clauses(id) ON DELETE CASCADE,
    FOREIGN KEY(clause_b_id) REFERENCES clauses(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS findings (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    finding_type TEXT NOT NULL,
    clause_a_id TEXT NOT NULL,
    clause_b_id TEXT,
    clause_a_text TEXT NOT NULL,
    clause_b_text TEXT,
    page_numbers TEXT NOT NULL,
    semantic_similarity REAL,
    nli_label TEXT,
    nli_contradiction_score REAL,
    nli_entailment_score REAL,
    nli_neutral_score REAL,
    rule_triggered TEXT NOT NULL,
    explanation TEXT NOT NULL,
    priority TEXT NOT NULL,
    model_identifier TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(doc_id) REFERENCES documents(id) ON DELETE CASCADE,
    FOREIGN KEY(clause_a_id) REFERENCES clauses(id) ON DELETE CASCADE,
    FOREIGN KEY(clause_b_id) REFERENCES clauses(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    finding_id TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'Pending Review',
    reviewer_name TEXT DEFAULT 'Human Reviewer',
    reviewer_comment TEXT DEFAULT '',
    reviewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(finding_id) REFERENCES findings(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_name TEXT NOT NULL,
    pipeline_name TEXT NOT NULL,
    precision REAL NOT NULL,
    recall REAL NOT NULL,
    f1_score REAL NOT NULL,
    fpr REAL NOT NULL,
    confusion_matrix_json TEXT NOT NULL,
    avg_processing_time_ms REAL NOT NULL,
    total_cases INTEGER NOT NULL,
    evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    full_name TEXT NOT NULL,
    organization TEXT,
    password_hash TEXT NOT NULL,
    salt TEXT NOT NULL,
    role TEXT DEFAULT 'Procurement Specialist',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_clauses_doc_id ON clauses(doc_id);
CREATE INDEX IF NOT EXISTS idx_findings_doc_id ON findings(doc_id);
CREATE INDEX IF NOT EXISTS idx_candidate_pairs_doc_id ON candidate_pairs(doc_id);
CREATE INDEX IF NOT EXISTS idx_reviews_finding_id ON reviews(finding_id);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
"""
