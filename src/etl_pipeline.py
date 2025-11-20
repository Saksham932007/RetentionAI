"""
ETL (Extract, Transform, Load) pipeline module for RetentionAI.

This module provides the core ETL functionality for loading raw data,
transforming it, and persisting it to the database for machine learning
pipeline consumption.
"""

import logging
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import numpy as np

from .config import (
    RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR,
    DATA_CONFIG, RANDOM_SEED
)
from .database import DatabaseManager, get_database_manager

# Configure logging
logger = logging.getLogger(__name__)


class ETLPipeline:
    """
    ETL Pipeline for processing customer churn data.
    
    Handles the complete data pipeline from raw CSV files through
    transformation and database persistence.
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize ETL pipeline.
        
        Args:
            db_manager: Database manager instance. If None, creates default.
        """
        self.db_manager = db_manager or get_database_manager()
        self.raw_data_path = RAW_DATA_DIR
        self.interim_data_path = INTERIM_DATA_DIR
        self.processed_data_path = PROCESSED_DATA_DIR
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        for directory in [self.raw_data_path, self.interim_data_path, self.processed_data_path]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def extract_raw_data(self, file_name: str = "WA_Fn-UseC_-Telco-Customer-Churn.csv") -> pd.DataFrame:
        """
        Extract raw data from CSV file.
        
        Args:
            file_name: Name of the CSV file in raw data directory
            
        Returns:
            pd.DataFrame: Raw customer data
            
        Raises:
            FileNotFoundError: If the specified file doesn't exist
        """
        file_path = self.raw_data_path / file_name
        
        if not file_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Successfully extracted {len(df)} records from {file_path}")
            logger.info(f"Data shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to extract data from {file_path}: {e}")
            raise
    
    def validate_raw_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate raw data quality and structure.
        
        Args:
            df: Raw DataFrame to validate
            
        Returns:
            dict: Validation report with statistics and issues
        """
        validation_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'issues': []
        }
        
        # Check for critical issues
        if validation_report['duplicate_rows'] > 0:
            validation_report['issues'].append(
                f"Found {validation_report['duplicate_rows']} duplicate rows"
            )
        
        # Check for high missing value percentage
        high_missing_cols = [
            col for col, missing_count in validation_report['missing_values'].items()
            if missing_count / len(df) > 0.5
        ]
        
        if high_missing_cols:
            validation_report['issues'].append(
                f"Columns with >50% missing values: {high_missing_cols}"
            )
        
        logger.info(f"Data validation completed: {len(validation_report['issues'])} issues found")
        return validation_report
    
    def basic_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply basic transformations to raw data.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            pd.DataFrame: Transformed DataFrame
        """
        df_transformed = df.copy()
        
        # Convert column names to snake_case
        df_transformed.columns = df_transformed.columns.str.lower().str.replace(' ', '_')
        
        # Handle TotalCharges column (known issue in telco dataset)
        if 'totalcharges' in df_transformed.columns:
            # Convert TotalCharges to numeric, handling spaces as NaN
            df_transformed['totalcharges'] = pd.to_numeric(
                df_transformed['totalcharges'].str.strip(), 
                errors='coerce'
            )
        
        # Standardize Yes/No columns to boolean
        yes_no_columns = [
            col for col in df_transformed.columns
            if df_transformed[col].dtype == 'object' and
            set(df_transformed[col].dropna().str.lower().unique()).issubset({'yes', 'no'})
        ]
        
        for col in yes_no_columns:
            df_transformed[col] = df_transformed[col].str.lower().map({'yes': 1, 'no': 0})
        
        # Handle churn target variable
        if 'churn' in df_transformed.columns:
            df_transformed['churn'] = df_transformed['churn'].str.lower().map({'yes': 1, 'no': 0})
        
        logger.info(f"Basic transformation completed: {len(yes_no_columns)} boolean columns converted")
        return df_transformed
    
    def load_data_to_db(self, df: pd.DataFrame, table_name: str = "customer_data") -> None:
        """
        Load processed DataFrame to database.
        
        Args:
            df: DataFrame to load
            table_name: Target table name in database
        """
        try:
            # Insert DataFrame into database
            self.db_manager.insert_dataframe(
                df=df,
                table_name=table_name,
                if_exists='replace',  # Replace existing table
                index=False
            )
            
            # Get table info for verification
            table_info = self.db_manager.get_table_info(table_name)
            logger.info(f"Data loaded to database table '{table_name}':")
            logger.info(f"  Rows: {table_info['row_count']}")
            logger.info(f"  Columns: {len(table_info['columns'])}")
            
        except Exception as e:
            logger.error(f"Failed to load data to database: {e}")
            raise
    
    def save_interim_data(self, df: pd.DataFrame, filename: str = "interim_data.csv") -> Path:
        """
        Save interim data to CSV for debugging/inspection.
        
        Args:
            df: DataFrame to save
            filename: Output filename
            
        Returns:
            Path: Path to saved file
        """
        output_path = self.interim_data_path / filename
        
        try:
            df.to_csv(output_path, index=False)
            logger.info(f"Interim data saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to save interim data: {e}")
            raise
    
    def run_etl_pipeline(
        self, 
        input_file: str = "WA_Fn-UseC_-Telco-Customer-Churn.csv",
        output_table: str = "customer_data",
        save_interim: bool = True
    ) -> Dict[str, Any]:
        """
        Run the complete ETL pipeline.
        
        Args:
            input_file: Input CSV filename
            output_table: Output database table name
            save_interim: Whether to save interim data to CSV
            
        Returns:
            dict: Pipeline execution report
        """
        logger.info("Starting ETL pipeline execution")
        
        pipeline_report = {
            'status': 'started',
            'input_file': input_file,
            'output_table': output_table,
            'steps_completed': [],
            'validation_report': None,
            'final_shape': None,
            'execution_time': None
        }
        
        try:
            # Step 1: Extract
            logger.info("Step 1: Extracting raw data")
            raw_df = self.extract_raw_data(input_file)
            pipeline_report['steps_completed'].append('extract')
            
            # Step 2: Validate
            logger.info("Step 2: Validating raw data")
            validation_report = self.validate_raw_data(raw_df)
            pipeline_report['validation_report'] = validation_report
            pipeline_report['steps_completed'].append('validate')
            
            # Step 3: Transform
            logger.info("Step 3: Applying basic transformations")
            transformed_df = self.basic_transform(raw_df)
            pipeline_report['steps_completed'].append('transform')
            
            # Step 4: Save interim (optional)
            if save_interim:
                logger.info("Step 4: Saving interim data")
                self.save_interim_data(transformed_df)
                pipeline_report['steps_completed'].append('save_interim')
            
            # Step 5: Load to database
            logger.info("Step 5: Loading data to database")
            self.load_data_to_db(transformed_df, output_table)
            pipeline_report['steps_completed'].append('load')
            
            # Final report
            pipeline_report['status'] = 'completed'
            pipeline_report['final_shape'] = transformed_df.shape
            logger.info("ETL pipeline completed successfully")
            
        except Exception as e:
            pipeline_report['status'] = 'failed'
            pipeline_report['error'] = str(e)
            logger.error(f"ETL pipeline failed: {e}")
            raise
        
        return pipeline_report


def run_etl_pipeline() -> Dict[str, Any]:
    """
    Convenience function to run ETL pipeline with default settings.
    
    Returns:
        dict: Pipeline execution report
    """
    etl = ETLPipeline()
    return etl.run_etl_pipeline()


if __name__ == "__main__":
    # Test ETL pipeline
    import logging.config
    from .config import LOGGING_CONFIG
    
    logging.config.dictConfig(LOGGING_CONFIG)
    
    try:
        report = run_etl_pipeline()
        print("\nETL Pipeline Report:")
        for key, value in report.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"ETL Pipeline failed: {e}")