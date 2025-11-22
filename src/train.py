"""
Model training module for RetentionAI.

This module provides the core training functionality using MLflow for experiment
tracking and model registry. Supports XGBoost and other gradient boosting models
with hyperparameter optimization.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from pathlib import Path
import pickle
import joblib
from datetime import datetime

# ML Libraries
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

try:
    from .config import (
        MODEL_CONFIG, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME,
        MODELS_DIR, RANDOM_SEED
    )
    from .utils import save_pickle, save_json, setup_logger
    from .preprocessing import DataPreprocessor
except ImportError:
    from config import (
        MODEL_CONFIG, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME,
        MODELS_DIR, RANDOM_SEED
    )
    from utils import save_pickle, save_json, setup_logger
    from preprocessing import DataPreprocessor

# Configure logging
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Model trainer for customer churn prediction.
    
    Handles training, evaluation, and model persistence with MLflow integration
    for experiment tracking and model registry.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize model trainer.
        
        Args:
            config: Configuration dictionary, uses MODEL_CONFIG if None
        """
        self.config = config or MODEL_CONFIG.__dict__
        self.model = None
        self.is_fitted = False
        
        # Training artifacts
        self.training_artifacts = {
            'model': None,
            'metrics': {},
            'feature_importance': {},
            'training_params': {},
            'preprocessing_pipeline': None
        }
        
        # Setup MLflow
        self._setup_mlflow()
    
    def _setup_mlflow(self) -> None:
        """Setup MLflow tracking and experiment."""
        try:
            # Set tracking URI
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            
            # Create or get experiment
            try:
                experiment_id = mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
            except mlflow.exceptions.MlflowException:
                # Experiment already exists
                experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
                experiment_id = experiment.experiment_id
            
            mlflow.set_experiment(experiment_id=experiment_id)
            
            logger.info(f"MLflow setup completed: experiment '{MLFLOW_EXPERIMENT_NAME}'")
            logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
            
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}. Training will continue without tracking.")
    
    def initialize_model(self, model_type: str = 'xgboost', **kwargs) -> None:
        """
        Initialize machine learning model.
        
        Args:
            model_type: Type of model ('xgboost', 'catboost', 'lightgbm')
            **kwargs: Additional model parameters
        """
        if model_type.lower() == 'xgboost':
            # Merge default params with custom kwargs
            params = self.config.get('xgboost_params', {}).copy()
            params.update(kwargs)
            
            self.model = xgb.XGBClassifier(**params)
            logger.info(f"Initialized XGBoost classifier with params: {params}")
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.training_artifacts['training_params'] = params
    
    def train_model(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        early_stopping_rounds: Optional[int] = 10
    ) -> Dict[str, Any]:
        """
        Train the machine learning model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            early_stopping_rounds: Early stopping patience
            
        Returns:
            dict: Training metrics and information
        """
        if self.model is None:
            self.initialize_model()
        
        logger.info(f"Starting model training: {X_train.shape[0]} training samples")
        
        training_start = datetime.now()
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"churn_model_{training_start.strftime('%Y%m%d_%H%M%S')}"):
            
            # Log parameters
            mlflow.log_params(self.training_artifacts['training_params'])
            mlflow.log_param("training_samples", len(X_train))
            mlflow.log_param("feature_count", X_train.shape[1])
            
            # Prepare training data
            if isinstance(self.model, xgb.XGBClassifier) and X_val is not None and y_val is not None:
                # Use validation set for early stopping
                eval_set = [(X_train, y_train), (X_val, y_val)]
                self.model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=False
                )
            else:
                # Train without validation monitoring
                self.model.fit(X_train, y_train)
            
            training_time = (datetime.now() - training_start).total_seconds()
            
            # Generate predictions for evaluation
            y_train_pred = self.model.predict(X_train)
            y_train_proba = self.model.predict_proba(X_train)[:, 1]
            
            # Calculate training metrics
            train_metrics = self._calculate_metrics(y_train, y_train_pred, y_train_proba, 'train')
            
            # Validation metrics if validation set provided
            val_metrics = {}
            if X_val is not None and y_val is not None:
                y_val_pred = self.model.predict(X_val)
                y_val_proba = self.model.predict_proba(X_val)[:, 1]
                val_metrics = self._calculate_metrics(y_val, y_val_pred, y_val_proba, 'val')
            
            # Feature importance
            feature_importance = self._extract_feature_importance(X_train.columns)
            
            # Store artifacts
            self.training_artifacts.update({
                'model': self.model,
                'metrics': {**train_metrics, **val_metrics},
                'feature_importance': feature_importance,
                'training_time': training_time,
                'model_type': type(self.model).__name__
            })
            
            # Log metrics to MLflow
            mlflow.log_metrics(train_metrics)
            if val_metrics:
                mlflow.log_metrics(val_metrics)
            
            mlflow.log_metric("training_time_seconds", training_time)
            
            # Log feature importance
            if feature_importance:
                for feature, importance in list(feature_importance.items())[:10]:  # Top 10
                    mlflow.log_metric(f"importance_{feature}", importance)
            
            # Log model
            mlflow.sklearn.log_model(
                self.model, 
                "model",
                registered_model_name="churn_prediction_model"
            )
            
            self.is_fitted = True
            logger.info(f"Model training completed in {training_time:.2f} seconds")
            
            return self.training_artifacts
    
    def _calculate_metrics(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray, 
        y_proba: np.ndarray,
        prefix: str
    ) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            prefix: Metric name prefix (e.g., 'train', 'val', 'test')
            
        Returns:
            dict: Dictionary of metrics
        """
        metrics = {
            f'{prefix}_accuracy': accuracy_score(y_true, y_pred),
            f'{prefix}_precision': precision_score(y_true, y_pred, zero_division=0),
            f'{prefix}_recall': recall_score(y_true, y_pred, zero_division=0),
            f'{prefix}_f1': f1_score(y_true, y_pred, zero_division=0),
            f'{prefix}_auc': roc_auc_score(y_true, y_proba),
        }
        
        # Class distribution
        unique, counts = np.unique(y_true, return_counts=True)
        for class_val, count in zip(unique, counts):
            metrics[f'{prefix}_class_{class_val}_count'] = count
            metrics[f'{prefix}_class_{class_val}_proportion'] = count / len(y_true)
        
        return metrics
    
    def _extract_feature_importance(self, feature_names: pd.Index) -> Dict[str, float]:
        """
        Extract feature importance from trained model.
        
        Args:
            feature_names: Names of features
            
        Returns:
            dict: Feature importance mapping
        """
        if not hasattr(self.model, 'feature_importances_'):
            return {}
        
        importances = self.model.feature_importances_
        feature_importance = dict(zip(feature_names, importances))
        
        # Sort by importance
        feature_importance = dict(sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        logger.info(f"Extracted feature importance for {len(feature_importance)} features")
        return feature_importance
    
    def evaluate_model(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            dict: Evaluation results
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        logger.info(f"Evaluating model on {len(X_test)} test samples")
        
        # Generate predictions
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        test_metrics = self._calculate_metrics(y_test, y_pred, y_proba, 'test')
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        evaluation_results = {
            'metrics': test_metrics,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'predictions': {
                'y_true': y_test.tolist(),
                'y_pred': y_pred.tolist(),
                'y_proba': y_proba.tolist()
            }
        }
        
        # Log test metrics to MLflow if in active run
        if mlflow.active_run():
            mlflow.log_metrics(test_metrics)
        
        logger.info(f"Test evaluation completed. AUC: {test_metrics['test_auc']:.4f}")
        
        return evaluation_results
    
    def cross_validate(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        cv_folds: int = 5,
        scoring: str = 'roc_auc'
    ) -> Dict[str, Any]:
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Target
            cv_folds: Number of CV folds
            scoring: Scoring metric
            
        Returns:
            dict: CV results
        """
        if self.model is None:
            self.initialize_model()
        
        logger.info(f"Running {cv_folds}-fold cross-validation")
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
        
        cv_scores = cross_val_score(
            self.model, X, y, 
            cv=cv, scoring=scoring, n_jobs=-1
        )
        
        cv_results = {
            'scores': cv_scores.tolist(),
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'scoring_metric': scoring,
            'n_folds': cv_folds
        }
        
        logger.info(f"CV {scoring}: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']:.4f})")
        
        return cv_results
    
    def save_model(self, filepath: Union[str, Path], include_artifacts: bool = True) -> None:
        """
        Save trained model and artifacts.
        
        Args:
            filepath: Path to save model
            include_artifacts: Whether to save training artifacts
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be trained before saving")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if hasattr(self.model, 'save_model'):
            # XGBoost native format
            model_path = filepath.with_suffix('.json')
            self.model.save_model(model_path)
        else:
            # Pickle fallback
            model_path = filepath.with_suffix('.pkl')
            save_pickle(self.model, model_path)
        
        # Save artifacts
        if include_artifacts:
            artifacts_path = filepath.with_suffix('.artifacts.pkl')
            save_pickle(self.training_artifacts, artifacts_path)
        
        logger.info(f"Model saved to: {model_path}")
        
    def load_model(self, filepath: Union[str, Path]) -> None:
        """
        Load saved model and artifacts.
        
        Args:
            filepath: Path to load model from
        """
        filepath = Path(filepath)
        
        # Load model
        if filepath.suffix == '.json':
            # XGBoost native format
            self.model = xgb.XGBClassifier()
            self.model.load_model(filepath)
        else:
            # Pickle format
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)
        
        # Load artifacts if available
        artifacts_path = filepath.with_suffix('.artifacts.pkl')
        if artifacts_path.exists():
            with open(artifacts_path, 'rb') as f:
                self.training_artifacts = pickle.load(f)
        
        self.is_fitted = True
        logger.info(f"Model loaded from: {filepath}")


if __name__ == "__main__":
    # Test model trainer
    import logging.config
    
    try:
        from config import LOGGING_CONFIG
        logging.config.dictConfig(LOGGING_CONFIG)
        
        # Test initialization
        trainer = ModelTrainer()
        trainer.initialize_model()
        print("ModelTrainer initialized successfully!")
        
    except Exception as e:
        print(f"ModelTrainer test failed: {e}")