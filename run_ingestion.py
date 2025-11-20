"""
Data ingestion script for RetentionAI.

This script orchestrates the complete data ingestion process from raw CSV
files through transformation and database persistence.
"""

import logging
import argparse
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from etl_pipeline import ETLPipeline
from data_generator import generate_or_validate_data
from config import LOGGING_CONFIG

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def run_data_ingestion(
    input_file: str = "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    output_table: str = "customer_data",
    generate_if_missing: bool = True,
    force_generate: bool = False
) -> Dict[str, Any]:
    """
    Run complete data ingestion pipeline.
    
    Args:
        input_file: Input CSV filename
        output_table: Database table name
        generate_if_missing: Generate synthetic data if file missing
        force_generate: Force data generation even if file exists
        
    Returns:
        dict: Ingestion report with all steps
    """
    logger.info("Starting data ingestion process")
    
    ingestion_report = {
        'status': 'started',
        'data_generation': None,
        'etl_pipeline': None,
        'final_status': 'unknown'
    }
    
    try:
        # Step 1: Ensure data exists (generate if needed)
        if generate_if_missing or force_generate:
            logger.info("Step 1: Checking/generating data file")
            data_report = generate_or_validate_data(
                output_file=input_file,
                force_generate=force_generate
            )
            ingestion_report['data_generation'] = data_report
            
            if data_report['issues'] and not data_report['generated']:
                raise Exception(f"Data validation failed: {data_report['issues']}")
        
        # Step 2: Run ETL pipeline
        logger.info("Step 2: Running ETL pipeline")
        etl = ETLPipeline()
        etl_report = etl.run_etl_pipeline(
            input_file=input_file,
            output_table=output_table,
            save_interim=True
        )
        ingestion_report['etl_pipeline'] = etl_report
        
        if etl_report['status'] != 'completed':
            raise Exception(f"ETL pipeline failed: {etl_report.get('error', 'Unknown error')}")
        
        # Success
        ingestion_report['final_status'] = 'success'
        logger.info("Data ingestion completed successfully")
        
        # Print summary
        print("\n" + "="*50)
        print("DATA INGESTION SUMMARY")
        print("="*50)
        
        if ingestion_report['data_generation']:
            dg = ingestion_report['data_generation']
            print(f"Data File: {dg['action']} - {dg['data_shape']}")
            
        if ingestion_report['etl_pipeline']:
            etl = ingestion_report['etl_pipeline']
            print(f"ETL Status: {etl['status']}")
            print(f"Final Shape: {etl['final_shape']}")
            print(f"Steps Completed: {', '.join(etl['steps_completed'])}")
            
            if etl['validation_report']:
                vr = etl['validation_report']
                print(f"Database Records: {vr['total_rows']}")
                print(f"Columns: {vr['total_columns']}")
                print(f"Issues Found: {len(vr['issues'])}")
        
        print("="*50)
        
    except Exception as e:
        ingestion_report['final_status'] = 'failed'
        ingestion_report['error'] = str(e)
        logger.error(f"Data ingestion failed: {e}")
        raise
    
    return ingestion_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run data ingestion pipeline")
    parser.add_argument(
        "--input-file", 
        default="WA_Fn-UseC_-Telco-Customer-Churn.csv",
        help="Input CSV filename"
    )
    parser.add_argument(
        "--output-table", 
        default="customer_data",
        help="Database table name"
    )
    parser.add_argument(
        "--force-generate", 
        action="store_true",
        help="Force data generation even if file exists"
    )
    parser.add_argument(
        "--no-generate", 
        action="store_true",
        help="Don't generate data if file missing"
    )
    
    args = parser.parse_args()
    
    try:
        report = run_data_ingestion(
            input_file=args.input_file,
            output_table=args.output_table,
            generate_if_missing=not args.no_generate,
            force_generate=args.force_generate
        )
        
        print(f"\nIngestion completed: {report['final_status']}")
        
    except Exception as e:
        print(f"Ingestion failed: {e}")
        exit(1)