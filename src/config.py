"""
Configuration management module for RetentionAI.

This module contains all configuration constants, file paths, and environment
settings used throughout the application. It provides a centralized location
for managing application settings.
"""

import os
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass


# Project Root Directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Data Directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model Directories
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Logging Directory
LOGS_DIR = PROJECT_ROOT / "logs"

# Notebooks Directory
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Database Configuration
DATABASE_DIR = PROJECT_ROOT
DATABASE_NAME = "retentionai.db"
DATABASE_URL = f"sqlite:///{DATABASE_DIR / DATABASE_NAME}"

# MLflow Configuration
MLFLOW_TRACKING_URI = f"file://{PROJECT_ROOT}/mlruns"
MLFLOW_EXPERIMENT_NAME = "churn_prediction_experiments"

# Random Seed for Reproducibility
RANDOM_SEED = 42

# Model Configuration
@dataclass
class ModelConfig:
    """Configuration for machine learning models."""
    
    # XGBoost Default Parameters
    xgboost_params: Dict[str, Any] = None
    
    # Optuna Optimization
    n_trials: int = 100
    optuna_timeout: int = 3600  # 1 hour
    
    # Cross Validation
    cv_folds: int = 5
    
    # Train/Validation/Test Split
    train_size: float = 0.7
    val_size: float = 0.15
    test_size: float = 0.15
    
    # Class Imbalance
    use_smote: bool = True
    smote_sampling_strategy: str = "auto"
    
    def __post_init__(self):
        """Initialize default XGBoost parameters."""
        if self.xgboost_params is None:
            self.xgboost_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': RANDOM_SEED,
                'n_jobs': -1
            }


@dataclass
class DataConfig:
    """Configuration for data processing."""
    
    # Target column
    target_column: str = "churn"
    
    # Categorical columns to encode
    categorical_columns: list = None
    
    # Numerical columns to scale
    numerical_columns: list = None
    
    # Columns to drop
    drop_columns: list = None
    
    # Feature engineering
    create_tenure_cohorts: bool = True
    use_target_encoding: bool = True
    
    def __post_init__(self):
        """Initialize default column configurations."""
        if self.categorical_columns is None:
            self.categorical_columns = [
                'gender', 'partner', 'dependents', 'phone_service',
                'multiple_lines', 'internet_service', 'online_security',
                'online_backup', 'device_protection', 'tech_support',
                'streaming_tv', 'streaming_movies', 'contract',
                'paperless_billing', 'payment_method'
            ]
        
        if self.numerical_columns is None:
            self.numerical_columns = [
                'senior_citizen', 'tenure', 'monthly_charges', 'total_charges'
            ]
        
        if self.drop_columns is None:
            self.drop_columns = ['customer_id']


@dataclass
class StreamlitConfig:
    """Configuration for Streamlit application."""
    
    # Page configuration
    page_title: str = "RetentionAI - Churn Prediction Dashboard"
    page_icon: str = "🧠"
    layout: str = "wide"
    
    # Dashboard settings
    max_customers_display: int = 1000
    default_customer_id: str = ""
    
    # Charts configuration
    chart_height: int = 400
    gauge_chart_size: int = 300


@dataclass
class BusinessConfig:
    """Configuration for business logic calculations."""
    
    # Default business metrics
    default_cac: float = 150.0  # Customer Acquisition Cost
    default_ltv: float = 1200.0  # Customer Lifetime Value
    default_retention_cost: float = 50.0  # Cost of retention intervention
    
    # ROI calculation parameters
    churn_reduction_rate: float = 0.25  # Expected reduction in churn from intervention
    intervention_success_rate: float = 0.7  # Success rate of retention efforts


# Global Configuration Instances
MODEL_CONFIG = ModelConfig()
DATA_CONFIG = DataConfig()
STREAMLIT_CONFIG = StreamlitConfig()
BUSINESS_CONFIG = BusinessConfig()

# Environment Variables
def get_env_var(key: str, default: str = "") -> str:
    """Get environment variable with default fallback."""
    return os.getenv(key, default)

# OpenAI Configuration (optional)
OPENAI_API_KEY = get_env_var("OPENAI_API_KEY", "")
OPENAI_MODEL = get_env_var("OPENAI_MODEL", "gpt-3.5-turbo")

# Logging Configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'default',
            'class': 'logging.StreamHandler',
        },
    },
    'root': {
        'level': 'INFO',
        'handlers': ['default'],
    },
}

# Create directories if they don't exist
def ensure_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR,
        MODELS_DIR, ARTIFACTS_DIR, LOGS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # Ensure all directories exist when module is run directly
    ensure_directories()
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Database URL: {DATABASE_URL}")
    print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")