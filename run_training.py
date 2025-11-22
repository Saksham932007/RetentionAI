"""
Training execution script for RetentionAI.

This script orchestrates the complete training pipeline including:
- Data loading and preprocessing
- Model training with hyperparameter optimization
- Model evaluation and validation
- MLflow experiment tracking
- Model artifact saving and registration
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import sys
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Ensure src is in path for imports
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from src.config import (
        PROCESSED_DATA_DIR, MODELS_DIR, MODEL_CONFIG, 
        RANDOM_SEED, EXPERIMENT_NAME
    )
    from src.database import get_database_manager
    from src.preprocessing import DataPreprocessor
    from src.train import ModelTrainer
    from src.utils import ensure_dir, save_json
    from src.etl_pipeline import ETLPipeline
except ImportError as e:
    logger.error(f"Failed to import modules: {e}")
    logger.error("Make sure all required packages are installed and src/ is in Python path")
    sys.exit(1)

try:
    import mlflow
    import mlflow.sklearn
except ImportError:
    mlflow = None
    logger.warning("MLflow not available - experiment tracking disabled")


class TrainingOrchestrator:
    """
    Orchestrates the complete model training pipeline.
    
    Coordinates data loading, preprocessing, model training, evaluation,
    and artifact management for the churn prediction model.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize training orchestrator.
        
        Args:
            config: Optional configuration overrides
        """
        self.config = config or {}
        self.db_manager = get_database_manager()
        self.preprocessor = DataPreprocessor()
        self.trainer = ModelTrainer()
        self.training_results = {}
        
        # Setup directories
        self.output_dir = Path(MODELS_DIR) / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ensure_dir(self.output_dir)
        
        logger.info(f"Training orchestrator initialized. Output: {self.output_dir}")
    
    def load_training_data(
        self, 
        table_name: str = "processed_data",
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load and split training data.
        
        Args:
            table_name: Database table containing processed data
            test_size: Test set proportion
            val_size: Validation set proportion
            
        Returns:
            tuple: (train_df, val_df, test_df)
        """
        logger.info(f"Loading training data from table: {table_name}")
        
        try:
            # Load data from database
            df = self.db_manager.execute_query(f"SELECT * FROM {table_name}")
            logger.info(f"Loaded {len(df)} records from database")
            
            # Check for required columns
            if 'churn' not in df.columns:
                raise ValueError("Target column 'churn' not found in data")
            
            # Split data
            from sklearn.model_selection import train_test_split
            
            # First split: train+val vs test
            train_val_df, test_df = train_test_split(
                df, 
                test_size=test_size, 
                random_state=RANDOM_SEED,
                stratify=df['churn']
            )
            
            # Second split: train vs val
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=val_size / (1 - test_size),  # Adjust val_size for remaining data
                random_state=RANDOM_SEED,
                stratify=train_val_df['churn']
            )
            
            logger.info(f"Data splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
            
            # Save data splits
            train_df.to_csv(self.output_dir / "train_data.csv", index=False)
            val_df.to_csv(self.output_dir / "val_data.csv", index=False)
            test_df.to_csv(self.output_dir / "test_data.csv", index=False)
            
            return train_df, val_df, test_df
            
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            raise
    
    def prepare_features(
        self, 
        train_df: pd.DataFrame, 
        val_df: pd.DataFrame, 
        test_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare features and targets for training.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame  
            test_df: Test DataFrame
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("Preparing features and targets")
        
        # Fit preprocessing on training data
        self.preprocessor.fit_preprocessing_pipeline(train_df)
        
        # Transform all datasets
        X_train = self.preprocessor.transform_features(train_df)
        X_val = self.preprocessor.transform_features(val_df)
        X_test = self.preprocessor.transform_features(test_df)
        
        # Extract targets
        y_train = train_df['churn'].values
        y_val = val_df['churn'].values
        y_test = test_df['churn'].values
        
        # Save preprocessing artifacts
        preprocessor_path = self.output_dir / "preprocessor.pkl"
        self.preprocessor.save_preprocessing_artifacts(preprocessor_path)
        
        logger.info(f"Feature shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_model(
        self, 
        X_train: np.ndarray, 
        X_val: np.ndarray, 
        y_train: np.ndarray, 
        y_val: np.ndarray,
        optimize_hyperparameters: bool = True,
        n_trials: int = 50
    ) -> Dict[str, Any]:
        """
        Train the churn prediction model.
        
        Args:
            X_train: Training features
            X_val: Validation features
            y_train: Training targets
            y_val: Validation targets
            optimize_hyperparameters: Whether to optimize hyperparameters
            n_trials: Number of optimization trials
            
        Returns:
            dict: Training results
        """
        logger.info("Starting model training")
        
        # Initialize MLflow run
        if mlflow:
            experiment_name = self.config.get('experiment_name', EXPERIMENT_NAME)
            mlflow.set_experiment(experiment_name)
        
        training_config = {
            'optimize_hyperparameters': optimize_hyperparameters,
            'n_trials': n_trials,
            'random_seed': RANDOM_SEED,
            'model_type': MODEL_CONFIG.algorithm,
            **self.config
        }
        
        # Train model
        results = self.trainer.train_model(
            X_train=X_train,
            X_val=X_val, 
            y_train=y_train,
            y_val=y_val,
            **training_config
        )
        
        self.training_results = results
        
        # Save model artifacts
        model_path = self.output_dir / "best_model.json"
        self.trainer.save_model(model_path)
        
        # Save training results
        results_path = self.output_dir / "training_results.json"
        save_json(results, results_path)
        
        logger.info("Model training completed")
        logger.info(f"Best validation AUC: {results.get('best_val_auc', 'N/A'):.4f}")
        
        return results
    
    def evaluate_model(
        self, 
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate trained model on test set.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            dict: Evaluation results
        """
        logger.info("Evaluating model on test set")
        
        # Test set evaluation
        test_results = self.trainer.evaluate_model(X_test, y_test)
        
        logger.info("Test Set Results:")
        for metric, value in test_results.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {metric}: {value:.4f}")
        
        # Save test results
        test_results_path = self.output_dir / "test_results.json"
        save_json(test_results, test_results_path)
        
        # Log to MLflow if available
        if mlflow and mlflow.active_run():
            for metric, value in test_results.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"test_{metric}", value)
        
        return test_results
    
    def generate_model_interpretability(
        self, 
        X_train: np.ndarray,
        X_test: np.ndarray, 
        feature_names: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Generate model interpretability analysis.
        
        Args:
            X_train: Training features for SHAP baseline
            X_test: Test features for SHAP analysis
            feature_names: Feature names for plots
            
        Returns:
            dict: Interpretability results
        """
        logger.info("Generating model interpretability analysis")
        
        try:
            # Get feature importance from model
            feature_importance = self.trainer.get_feature_importance()
            
            # Generate SHAP analysis if available
            shap_results = None
            if hasattr(self.trainer, 'generate_shap_analysis'):
                shap_results = self.trainer.generate_shap_analysis(
                    X_train=X_train,
                    X_test=X_test,
                    feature_names=feature_names,
                    save_plots=True,
                    plots_dir=self.output_dir / "interpretability"
                )
            
            interpretability_results = {
                'feature_importance': feature_importance,
                'shap_analysis': shap_results,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save interpretability results
            interp_path = self.output_dir / "interpretability_results.json"
            save_json(interpretability_results, interp_path)
            
            logger.info("Interpretability analysis completed")
            
            return interpretability_results
            
        except Exception as e:
            logger.error(f"Interpretability analysis failed: {e}")
            return {'error': str(e)}
    
    def run_full_pipeline(
        self,
        table_name: str = "processed_data",
        optimize_hyperparameters: bool = True,
        n_trials: int = 50,
        generate_interpretability: bool = True
    ) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Args:
            table_name: Database table with processed data
            optimize_hyperparameters: Whether to optimize hyperparameters
            n_trials: Number of optimization trials
            generate_interpretability: Whether to generate interpretability analysis
            
        Returns:
            dict: Complete pipeline results
        """
        logger.info("Starting full training pipeline")
        
        pipeline_results = {}
        
        try:
            # Start MLflow run if available
            if mlflow:
                with mlflow.start_run():
                    # Log configuration
                    mlflow.log_params({
                        'table_name': table_name,
                        'optimize_hyperparameters': optimize_hyperparameters,
                        'n_trials': n_trials,
                        'random_seed': RANDOM_SEED
                    })
                    
                    # Execute pipeline steps
                    return self._execute_pipeline_steps(
                        table_name, optimize_hyperparameters, n_trials, generate_interpretability
                    )
            else:
                # Execute without MLflow
                return self._execute_pipeline_steps(
                    table_name, optimize_hyperparameters, n_trials, generate_interpretability
                )
                
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            pipeline_results['error'] = str(e)
            pipeline_results['status'] = 'failed'
            
            # Save error results
            error_path = self.output_dir / "pipeline_error.json"
            save_json(pipeline_results, error_path)
            
            raise
    
    def _execute_pipeline_steps(
        self,
        table_name: str,
        optimize_hyperparameters: bool,
        n_trials: int,
        generate_interpretability: bool
    ) -> Dict[str, Any]:
        """Execute individual pipeline steps."""
        
        # Step 1: Load and split data
        train_df, val_df, test_df = self.load_training_data(table_name)
        
        # Step 2: Prepare features
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_features(
            train_df, val_df, test_df
        )
        
        # Step 3: Train model
        training_results = self.train_model(
            X_train, X_val, y_train, y_val, 
            optimize_hyperparameters, n_trials
        )
        
        # Step 4: Evaluate on test set
        test_results = self.evaluate_model(X_test, y_test)
        
        # Step 5: Generate interpretability (optional)
        interpretability_results = {}
        if generate_interpretability:
            feature_names = getattr(self.preprocessor, 'feature_names', None)
            interpretability_results = self.generate_model_interpretability(
                X_train, X_test, feature_names
            )
        
        # Compile final results
        pipeline_results = {
            'status': 'completed',
            'timestamp': datetime.now().isoformat(),
            'output_directory': str(self.output_dir),
            'data_info': {
                'train_size': len(train_df),
                'val_size': len(val_df),
                'test_size': len(test_df),
                'feature_count': X_train.shape[1]
            },
            'training_results': training_results,
            'test_results': test_results,
            'interpretability_results': interpretability_results
        }
        
        # Save complete results
        final_results_path = self.output_dir / "complete_pipeline_results.json"
        save_json(pipeline_results, final_results_path)
        
        logger.info("Training pipeline completed successfully")
        logger.info(f"Results saved to: {self.output_dir}")
        
        return pipeline_results


def main():
    """Main training execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run RetentionAI training pipeline")
    parser.add_argument("--table", default="processed_data", help="Database table name")
    parser.add_argument("--no-optimize", action="store_true", help="Skip hyperparameter optimization")
    parser.add_argument("--trials", type=int, default=50, help="Number of optimization trials")
    parser.add_argument("--no-interpretability", action="store_true", help="Skip interpretability analysis")
    parser.add_argument("--config", help="Path to configuration JSON file")
    
    args = parser.parse_args()
    
    # Load configuration if provided
    config = {}
    if args.config:
        import json
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Initialize orchestrator
    orchestrator = TrainingOrchestrator(config)
    
    try:
        # Run full pipeline
        results = orchestrator.run_full_pipeline(
            table_name=args.table,
            optimize_hyperparameters=not args.no_optimize,
            n_trials=args.trials,
            generate_interpretability=not args.no_interpretability
        )
        
        print("\n" + "="*50)
        print("TRAINING PIPELINE COMPLETED")
        print("="*50)
        print(f"Status: {results['status']}")
        print(f"Output Directory: {results['output_directory']}")
        print(f"Test AUC: {results['test_results'].get('auc', 'N/A'):.4f}")
        print(f"Test Accuracy: {results['test_results'].get('accuracy', 'N/A'):.4f}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"\nTraining failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()