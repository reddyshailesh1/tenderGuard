"""Pytest configuration and shared fixtures for TenderGuard."""
import os
import tempfile
from pathlib import Path
import pytest

# Ensure OpenMP setting
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from tenderguard.db.session import DatabaseManager


@pytest.fixture
def temp_db(tmp_path):
    """Fixture providing an isolated temporary database for testing."""
    db_file = tmp_path / "test_tenderguard.db"
    manager = DatabaseManager(db_path=db_file)
    return manager
