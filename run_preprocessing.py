"""
Preprocessing execution script for RetentionAI.

This script loads data from the database, applies the preprocessing pipeline,
and saves processed datasets and artifacts for machine learning training.
"""

import logging
import argparse
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from database import get_database_manager
from preprocessing import DataPreprocessor
from config import LOGGING_CONFIG, PROCESSED_DATA_DIR, DATA_CONFIG
from utils import save_pickle, save_json, setup_logger

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def run_preprocessing_pipeline(
    table_name: str = "customer_data",
    output_prefix: str = "processed",
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    save_artifacts: bool = True
) -> Dict[str, Any]:
    """
    Run the complete preprocessing pipeline from database to processed datasets.
    
    Args:
        table_name: Database table to load data from
        output_prefix: Prefix for output files
        train_size: Training set proportion
        val_size: Validation set proportion  
        test_size: Test set proportion
        save_artifacts: Whether to save preprocessing artifacts
        
    Returns:
        dict: Processing report with paths and statistics
    """
    logger.info("Starting preprocessing pipeline execution")
    
    # Ensure output directory exists
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    report = {
        'status': 'started',
        'input_table': table_name,
        'output_files': {},
        'data_shapes': {},
        'target_distributions': {},
        'preprocessing_artifacts': None
    }
    
    try:
        # Step 1: Load data from database
        logger.info(f"Step 1: Loading data from table '{table_name}'")
        db_manager = get_database_manager()
        
        if not db_manager.table_exists(table_name):
            raise ValueError(f"Table '{table_name}' does not exist in database")
        
        df = db_manager.execute_query(f"SELECT * FROM {table_name}")
        logger.info(f"Loaded {len(df)} records from database")
        
        # Step 2: Initialize preprocessor and clean data
        logger.info("Step 2: Initializing preprocessor and cleaning data")
        preprocessor = DataPreprocessor()
        
        # Clean column names
        df = preprocessor.clean_column_names(df)
        
        # Handle missing values
        df = preprocessor.handle_missing_values(df)
        
        # Engineer features
        df = preprocessor.engineer_features(df)
        
        # Step 3: Prepare features and target
        logger.info("Step 3: Preparing features and target")
        X, y = preprocessor.prepare_target(df)
        
        report['data_shapes']['original'] = df.shape
        report['data_shapes']['features'] = X.shape
        report['data_shapes']['target'] = y.shape
        
        # Step 4: Create train/val/test splits
        logger.info("Step 4: Creating train/validation/test splits")
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.create_train_val_test_split(
            X, y, train_size=train_size, val_size=val_size, test_size=test_size
        )
        
        # Store shapes and distributions
        report['data_shapes']['train'] = (X_train.shape, y_train.shape)
        report['data_shapes']['val'] = (X_val.shape, y_val.shape)
        report['data_shapes']['test'] = (X_test.shape, y_test.shape)
        
        report['target_distributions']['train'] = y_train.value_counts().to_dict()
        report['target_distributions']['val'] = y_val.value_counts().to_dict()
        report['target_distributions']['test'] = y_test.value_counts().to_dict()
        
        # Step 5: Fit preprocessing pipeline on training data
        logger.info("Step 5: Fitting preprocessing transformers on training data")
        
        # Fit scaler on numerical columns
        feature_types = preprocessor.get_feature_types(X_train)
        preprocessor.fit_scaler(X_train, feature_types['numerical'])
        
        # Apply transformations to all sets
        X_train_processed = preprocessor.transform_numerical(X_train)
        X_val_processed = preprocessor.transform_numerical(X_val)
        X_test_processed = preprocessor.transform_numerical(X_test)
        
        # Step 6: Save processed datasets
        logger.info("Step 6: Saving processed datasets")
        
        # Save train set
        train_path = PROCESSED_DATA_DIR / f"{output_prefix}_train.pkl"
        save_pickle({
            'X': X_train_processed,
            'y': y_train,
            'feature_names': list(X_train_processed.columns),
            'feature_types': feature_types
        }, train_path)
        report['output_files']['train'] = str(train_path)
        
        # Save validation set
        val_path = PROCESSED_DATA_DIR / f"{output_prefix}_val.pkl"
        save_pickle({
            'X': X_val_processed,
            'y': y_val,
            'feature_names': list(X_val_processed.columns),
            'feature_types': feature_types
        }, val_path)
        report['output_files']['val'] = str(val_path)
        
        # Save test set
        test_path = PROCESSED_DATA_DIR / f"{output_prefix}_test.pkl"
        save_pickle({
            'X': X_test_processed,
            'y': y_test,
            'feature_names': list(X_test_processed.columns),
            'feature_types': feature_types
        }, test_path)
        report['output_files']['test'] = str(test_path)
        
        # Step 7: Save preprocessing artifacts
        if save_artifacts:
            logger.info("Step 7: Saving preprocessing artifacts")
            artifacts_path = PROCESSED_DATA_DIR / f"{output_prefix}_artifacts.pkl"
            preprocessor.save_preprocessing_artifacts(artifacts_path)
            report['preprocessing_artifacts'] = str(artifacts_path)
        
        # Step 8: Save processing report
        report_path = PROCESSED_DATA_DIR / f"{output_prefix}_report.json"
        save_json(report, report_path)
        report['output_files']['report'] = str(report_path)
        
        report['status'] = 'completed'
        logger.info("Preprocessing pipeline completed successfully")
        
        # Print summary
        print("\n" + "="*60)
        print("PREPROCESSING PIPELINE SUMMARY")
        print("="*60)
        print(f"Input table: {report['input_table']}")
        print(f"Original data shape: {report['data_shapes']['original']}")
        print(f"Feature shape: {report['data_shapes']['features']}")
        print(f"Train shape: {report['data_shapes']['train']}")
        print(f"Val shape: {report['data_shapes']['val']}")
        print(f"Test shape: {report['data_shapes']['test']}")
        print(f"Output files: {len(report['output_files'])}")
        for name, path in report['output_files'].items():
            print(f"  {name}: {Path(path).name}")
        print("="*60)
        
    except Exception as e:
        report['status'] = 'failed'
        report['error'] = str(e)
        logger.error(f"Preprocessing pipeline failed: {e}")
        raise
    
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline")
    parser.add_argument(
        "--table-name", 
        default="customer_data",
        help="Database table name to process"
    )
    parser.add_argument(
        "--output-prefix", 
        default="processed",
        help="Prefix for output files"
    )
    parser.add_argument(
        "--train-size", 
        type=float,
        default=0.7,
        help="Training set proportion"
    )
    parser.add_argument(
        "--val-size", 
        type=float,
        default=0.15,
        help="Validation set proportion"
    )
    parser.add_argument(
        "--test-size", 
        type=float,
        default=0.15,
        help="Test set proportion"
    )
    parser.add_argument(
        "--no-artifacts", 
        action="store_true",
        help="Don't save preprocessing artifacts"
    )
    
    args = parser.parse_args()
    
    try:
        report = run_preprocessing_pipeline(
            table_name=args.table_name,
            output_prefix=args.output_prefix,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            save_artifacts=not args.no_artifacts
        )
        
        print(f"\nPreprocessing completed: {report['status']}")
        
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        exit(1)