"""
Batch inference module for RetentionAI.

This module provides functionality for loading trained models and making
batch predictions on new customer data for churn prediction.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import pickle
import json
from datetime import datetime

# ML Libraries
import xgboost as xgb
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
except ImportError:
    mlflow = None

try:
    from .config import MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_SEED
    from .preprocessing import DataPreprocessor
    from .train import ModelTrainer
    from .utils import save_json, save_pickle, load_pickle
    from .database import get_database_manager
except ImportError:
    from config import MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_SEED
    from preprocessing import DataPreprocessor
    from train import ModelTrainer
    from utils import save_json, save_pickle, load_pickle
    from database import get_database_manager

# Configure logging
logger = logging.getLogger(__name__)


class ChurnPredictor:
    """
    Churn prediction inference engine.
    
    Handles loading trained models, preprocessing new data, and making
    batch predictions for customer churn probability.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize churn predictor.
        
        Args:
            model_path: Path to trained model file
        """
        self.model = None
        self.preprocessor = None
        self.model_metadata = {}
        self.is_loaded = False
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: Union[str, Path]) -> None:
        """
        Load trained model from file.
        
        Args:
            model_path: Path to model file (.json, .pkl, or MLflow URI)
        """
        model_path = Path(model_path) if not isinstance(model_path, Path) else model_path
        
        try:
            if str(model_path).startswith('runs:/') or str(model_path).startswith('models:/'):
                # MLflow model URI
                self._load_from_mlflow(str(model_path))
            elif model_path.suffix == '.json':
                # XGBoost native format
                self._load_xgboost_model(model_path)
            elif model_path.suffix in ['.pkl', '.pickle']:
                # Pickle format
                self._load_pickle_model(model_path)
            else:
                raise ValueError(f"Unsupported model format: {model_path.suffix}")
            
            self.is_loaded = True
            logger.info(f"Model loaded successfully from: {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise
    
    def _load_from_mlflow(self, model_uri: str) -> None:
        """Load model from MLflow registry."""
        if mlflow is None:
            raise ImportError("MLflow not available for model loading")
        
        self.model = mlflow.sklearn.load_model(model_uri)
        
        # Try to load model metadata
        try:
            run_id = model_uri.split('/')[-3] if 'runs:/' in model_uri else None
            if run_id:
                client = mlflow.tracking.MlflowClient()
                run = client.get_run(run_id)
                self.model_metadata = {
                    'mlflow_run_id': run_id,
                    'metrics': run.data.metrics,
                    'params': run.data.params,
                    'tags': run.data.tags
                }
        except Exception as e:
            logger.warning(f"Could not load MLflow metadata: {e}")
    
    def _load_xgboost_model(self, model_path: Path) -> None:
        """Load XGBoost model from JSON file."""
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        
        # Try to load accompanying metadata
        metadata_path = model_path.with_suffix('.metadata.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.model_metadata = json.load(f)
    
    def _load_pickle_model(self, model_path: Path) -> None:
        """Load model from pickle file."""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        if isinstance(model_data, dict):
            self.model = model_data.get('model')
            self.model_metadata = model_data.get('metadata', {})
        else:
            self.model = model_data
    
    def load_preprocessor(self, preprocessor_path: Union[str, Path]) -> None:
        """
        Load preprocessing pipeline.
        
        Args:
            preprocessor_path: Path to preprocessing artifacts
        """
        try:
            self.preprocessor = DataPreprocessor()
            self.preprocessor.load_preprocessing_artifacts(preprocessor_path)
            logger.info(f"Preprocessor loaded from: {preprocessor_path}")
        except Exception as e:
            logger.error(f"Failed to load preprocessor: {e}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess input data for prediction.
        
        Args:
            df: Raw input DataFrame
            
        Returns:
            pd.DataFrame: Preprocessed features
        """
        if self.preprocessor is None:
            logger.warning("No preprocessor loaded. Using minimal preprocessing.")
            return self._minimal_preprocessing(df)
        
        # Apply preprocessing pipeline
        df_processed = df.copy()
        
        # Clean column names
        df_processed = self.preprocessor.clean_column_names(df_processed)
        
        # Handle missing values
        df_processed = self.preprocessor.handle_missing_values(df_processed)
        
        # Engineer features
        df_processed = self.preprocessor.engineer_features(df_processed)
        
        # Remove target and ID columns if present
        target_col = self.preprocessor.config.get('target_column', 'churn')
        drop_cols = self.preprocessor.config.get('drop_columns', [])
        
        df_processed = df_processed.drop(columns=[target_col] + drop_cols, errors='ignore')
        
        # Apply numerical scaling if available
        df_processed = self.preprocessor.transform_numerical(df_processed)
        
        logger.info(f"Data preprocessed: {df_processed.shape}")
        return df_processed
    
    def _minimal_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply minimal preprocessing when no preprocessor is available.
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: Minimally processed DataFrame
        """
        df_processed = df.copy()
        
        # Clean column names
        df_processed.columns = df_processed.columns.str.lower().str.replace(' ', '_')
        
        # Handle basic missing values
        for col in df_processed.select_dtypes(include=['object']).columns:
            df_processed[col].fillna('Unknown', inplace=True)
        
        for col in df_processed.select_dtypes(include=[np.number]).columns:
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
        
        # Remove obvious non-feature columns
        drop_cols = [col for col in df_processed.columns 
                    if any(keyword in col.lower() for keyword in ['id', 'churn', 'target'])]
        df_processed = df_processed.drop(columns=drop_cols, errors='ignore')
        
        logger.warning("Applied minimal preprocessing - consider loading proper preprocessor")
        return df_processed
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Make predictions on input data.
        
        Args:
            df: Input DataFrame with customer data
            
        Returns:
            dict: Prediction results
        """
        if not self.is_loaded or self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info(f"Making predictions for {len(df)} samples")
        
        # Preprocess data
        X = self.preprocess_data(df)
        
        # Make predictions
        try:
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)[:, 1]  # Probability of churn (class 1)
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
        
        # Calculate prediction statistics
        prediction_stats = {
            'total_predictions': len(predictions),
            'predicted_churners': int(np.sum(predictions)),
            'predicted_non_churners': int(len(predictions) - np.sum(predictions)),
            'churn_rate': float(np.mean(predictions)),
            'avg_churn_probability': float(np.mean(probabilities)),
            'high_risk_customers': int(np.sum(probabilities > 0.7)),
            'medium_risk_customers': int(np.sum((probabilities > 0.3) & (probabilities <= 0.7))),
            'low_risk_customers': int(np.sum(probabilities <= 0.3))
        }
        
        # Create detailed results DataFrame
        results_df = df.copy()
        results_df['churn_prediction'] = predictions
        results_df['churn_probability'] = probabilities
        results_df['risk_category'] = pd.cut(
            probabilities, 
            bins=[0, 0.3, 0.7, 1.0], 
            labels=['Low', 'Medium', 'High']
        )
        
        prediction_results = {
            'predictions': results_df,
            'statistics': prediction_stats,
            'model_metadata': self.model_metadata,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Predictions completed. Churn rate: {prediction_stats['churn_rate']:.3f}")
        
        return prediction_results
    
    def predict_batch_from_database(
        self, 
        table_name: str = "customer_data",
        where_clause: Optional[str] = None,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Make batch predictions on data from database.
        
        Args:
            table_name: Database table name
            where_clause: Optional SQL WHERE clause
            limit: Optional limit on number of records
            
        Returns:
            dict: Prediction results
        """
        # Load data from database
        db_manager = get_database_manager()
        
        # Build query
        query = f"SELECT * FROM {table_name}"
        if where_clause:
            query += f" WHERE {where_clause}"
        if limit:
            query += f" LIMIT {limit}"
        
        df = db_manager.execute_query(query)
        logger.info(f"Loaded {len(df)} records from database")
        
        # Make predictions
        return self.predict(df)
    
    def save_predictions(
        self, 
        prediction_results: Dict[str, Any], 
        output_path: Union[str, Path],
        format: str = 'csv'
    ) -> None:
        """
        Save prediction results to file.
        
        Args:
            prediction_results: Results from predict() method
            output_path: Output file path
            format: Output format ('csv', 'json', 'pickle')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        predictions_df = prediction_results['predictions']
        
        if format == 'csv':
            predictions_df.to_csv(output_path, index=False)
        elif format == 'json':
            # Convert DataFrame to dict for JSON serialization
            results_for_json = {
                'predictions': predictions_df.to_dict('records'),
                'statistics': prediction_results['statistics'],
                'model_metadata': prediction_results['model_metadata'],
                'timestamp': prediction_results['timestamp']
            }
            save_json(results_for_json, output_path)
        elif format == 'pickle':
            save_pickle(prediction_results, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Predictions saved to: {output_path}")


def load_best_model_from_mlflow(
    experiment_name: str,
    metric_name: str = 'val_auc',
    stage: str = 'Production'
) -> ChurnPredictor:
    """
    Load best model from MLflow experiment or registry.
    
    Args:
        experiment_name: MLflow experiment name
        metric_name: Metric to use for selecting best model
        stage: Model registry stage ('Production', 'Staging', etc.)
        
    Returns:
        ChurnPredictor: Loaded predictor instance
    """
    if mlflow is None:
        raise ImportError("MLflow not available")
    
    try:
        # Try to load from model registry first
        model_uri = f"models:/churn_prediction_model/{stage}"
        predictor = ChurnPredictor()
        predictor._load_from_mlflow(model_uri)
        logger.info(f"Loaded model from registry: {stage}")
        return predictor
        
    except Exception:
        # Fallback to finding best run from experiment
        logger.info("Registry model not found, searching experiment for best model")
        
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        
        # Get all runs and find best by metric
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="",
            order_by=[f"metrics.{metric_name} DESC"]
        )
        
        if not runs:
            raise ValueError(f"No runs found in experiment '{experiment_name}'")
        
        best_run = runs[0]
        model_uri = f"runs:/{best_run.info.run_id}/model"
        
        predictor = ChurnPredictor()
        predictor._load_from_mlflow(model_uri)
        
        logger.info(f"Loaded best model from run: {best_run.info.run_id}")
        logger.info(f"Best {metric_name}: {best_run.data.metrics.get(metric_name, 'N/A')}")
        
        return predictor


if __name__ == "__main__":
    # Test prediction functionality
    import argparse
    
    parser = argparse.ArgumentParser(description="Test churn prediction")
    parser.add_argument("--model-path", help="Path to model file")
    parser.add_argument("--data-path", help="Path to CSV data file")
    parser.add_argument("--output-path", help="Output path for predictions")
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        if args.model_path:
            predictor = ChurnPredictor(args.model_path)
        else:
            # Try to load best model from MLflow
            predictor = load_best_model_from_mlflow("churn_prediction_experiments")
        
        # Load data
        if args.data_path:
            df = pd.read_csv(args.data_path)
        else:
            # Load from database
            results = predictor.predict_batch_from_database()
            print("Prediction Statistics:")
            for key, value in results['statistics'].items():
                print(f"  {key}: {value}")
            
            if args.output_path:
                predictor.save_predictions(results, args.output_path)
                print(f"Results saved to: {args.output_path}")
            
            exit(0)
        
        # Make predictions
        results = predictor.predict(df)
        
        print("Prediction Statistics:")
        for key, value in results['statistics'].items():
            print(f"  {key}: {value}")
        
        # Save results if output path provided
        if args.output_path:
            predictor.save_predictions(results, args.output_path)
            print(f"Results saved to: {args.output_path}")
        
    except Exception as e:
        print(f"Prediction failed: {e}")
        exit(1)