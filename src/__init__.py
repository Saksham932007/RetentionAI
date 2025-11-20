"""
RetentionAI: Production-Grade Churn Prediction Application.

A comprehensive ML-powered platform for predicting customer churn and
optimizing retention strategies using advanced machine learning, explainable
AI, and business impact simulation.
"""

__version__ = "0.1.0"
__author__ = "Saksham Kapoor"
__email__ = "saksham@example.com"

from .config import (
    PROJECT_ROOT, DATABASE_URL, MLFLOW_TRACKING_URI,
    MODEL_CONFIG, DATA_CONFIG, STREAMLIT_CONFIG, BUSINESS_CONFIG
)
from .database import DatabaseManager, get_database_manager
from .etl_pipeline import ETLPipeline, run_etl_pipeline
from .data_generator import TelcoChurnDataGenerator, generate_or_validate_data

__all__ = [
    # Configuration
    'PROJECT_ROOT', 'DATABASE_URL', 'MLFLOW_TRACKING_URI',
    'MODEL_CONFIG', 'DATA_CONFIG', 'STREAMLIT_CONFIG', 'BUSINESS_CONFIG',
    
    # Database
    'DatabaseManager', 'get_database_manager',
    
    # ETL Pipeline
    'ETLPipeline', 'run_etl_pipeline',
    
    # Data Generation
    'TelcoChurnDataGenerator', 'generate_or_validate_data'
]