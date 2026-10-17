"""TenderGuard - Intelligent Tender Risk Detection and Contradiction Analyzer."""
import os

# Prevent OpenMP library clash on Windows Anaconda
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

__version__ = "1.0.0"
