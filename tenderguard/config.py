"""Global configuration for TenderGuard."""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

# Fix OpenMP runtime duplicate error on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
EXPORTS_DIR = BASE_DIR / "exports"
UPLOADS_DIR = BASE_DIR / "uploads"
DEFAULT_DB_PATH = BASE_DIR / "tenderguard.db"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
EXPORTS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)


TENDER_CATEGORIES: List[str] = [
    "Eligibility & Qualifications",
    "Financial & Commercial Terms",
    "Technical Specifications",
    "Deadlines & Submission Schedules",
    "Legal, Compliance & Contractual Terms",
    "Penalties & Liquidated Damages",
    "Scope of Work & Deliverables",
    "General Administrative Provisions",
]

# Zero-shot hypothesis template for tender classification
CLASSIFICATION_HYPOTHESIS_TEMPLATE = "This tender clause specifies {}."


@dataclass
class ModelConfig:
    """Configuration for pretrained Transformer and embedding models."""
    # Sentence Transformer embedding model
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Pretrained NLI Transformer model
    # typeform/distilbert-base-uncased-mnli is fast, lightweight (~260MB), and reliable for CPU inference
    nli_model_name: str = "typeform/distilbert-base-uncased-mnli"
    
    # Pretrained Zero-Shot Clause Classifier
    classifier_model_name: str = "typeform/distilbert-base-uncased-mnli"
    
    # Device: 'cpu', 'cuda', or 'auto'
    device: str = "auto"
    
    # Semantic similarity threshold for candidate pair retrieval (cosine similarity)
    similarity_threshold: float = 0.50
    
    # Maximum candidate pairs per clause to retrieve
    top_k_candidates: int = 5
    
    # Contradiction probability threshold to flag as potential NLI contradiction
    contradiction_threshold: float = 0.45
    
    # Local processing only - privacy guardrail
    local_processing_only: bool = True


@dataclass
class IngestionConfig:
    """PDF Ingestion and Clause Segmentation thresholds."""
    max_file_size_bytes: int = 50 * 1024 * 1024  # 50 MB
    min_page_chars_for_text: int = 40            # Under this, page is flagged as scanned/empty
    min_clause_chars: int = 25                   # Minimum length of meaningful clause
    max_clause_chars: int = 4000                 # Maximum clause length before sub-segmentation
    enable_ocr_fallback: bool = True             # Enable OCR if tesseract is available


@dataclass
class TenderGuardConfig:
    """Root configuration combining model, ingestion, and storage settings."""
    db_path: Path = DEFAULT_DB_PATH
    models: ModelConfig = field(default_factory=ModelConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    categories: List[str] = field(default_factory=lambda: list(TENDER_CATEGORIES))


# Global default configuration instance
DEFAULT_CONFIG = TenderGuardConfig()
