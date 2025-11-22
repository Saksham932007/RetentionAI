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
import optuna
from optuna.integration import XGBoostPruningCallback
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns

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
            
            # Log artifacts (plots, feature importance, etc.)
            self._log_artifacts_to_mlflow(
                feature_importance, X_train, y_train, y_train_pred, y_train_proba
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
        # Basic classification metrics
        metrics = {
            f'{prefix}_accuracy': accuracy_score(y_true, y_pred),
            f'{prefix}_precision': precision_score(y_true, y_pred, zero_division=0),
            f'{prefix}_recall': recall_score(y_true, y_pred, zero_division=0),
            f'{prefix}_f1': f1_score(y_true, y_pred, zero_division=0),
            f'{prefix}_auc': roc_auc_score(y_true, y_proba),
        }
        
        # Additional metrics for imbalanced datasets
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        # Precision-Recall AUC (better for imbalanced datasets)
        metrics[f'{prefix}_pr_auc'] = average_precision_score(y_true, y_proba)
        
        # Confusion matrix components
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):  # Binary classification
            tn, fp, fn, tp = cm.ravel()
            
            # Specificity (True Negative Rate)
            metrics[f'{prefix}_specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            # False Positive Rate
            metrics[f'{prefix}_fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            
            # False Negative Rate
            metrics[f'{prefix}_fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            
            # Balanced Accuracy
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            metrics[f'{prefix}_balanced_accuracy'] = (sensitivity + specificity) / 2
            
            # Matthews Correlation Coefficient
            from sklearn.metrics import matthews_corrcoef
            metrics[f'{prefix}_mcc'] = matthews_corrcoef(y_true, y_pred)
            
            # Log Loss
            from sklearn.metrics import log_loss
            metrics[f'{prefix}_log_loss'] = log_loss(y_true, y_proba)
        
        # Class distribution
        unique, counts = np.unique(y_true, return_counts=True)
        for class_val, count in zip(unique, counts):
            metrics[f'{prefix}_class_{class_val}_count'] = count
            metrics[f'{prefix}_class_{class_val}_proportion'] = count / len(y_true)
        
        # Prediction distribution
        pred_unique, pred_counts = np.unique(y_pred, return_counts=True)
        for class_val, count in zip(pred_unique, pred_counts):
            metrics[f'{prefix}_pred_class_{class_val}_count'] = count
            metrics[f'{prefix}_pred_class_{class_val}_proportion'] = count / len(y_pred)
        
        # Probability statistics
        metrics[f'{prefix}_prob_mean'] = np.mean(y_proba)
        metrics[f'{prefix}_prob_std'] = np.std(y_proba)
        metrics[f'{prefix}_prob_min'] = np.min(y_proba)
        metrics[f'{prefix}_prob_max'] = np.max(y_proba)
        
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
    
    def _log_artifacts_to_mlflow(
        self,
        feature_importance: Dict[str, float],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        y_train_pred: np.ndarray,
        y_train_proba: np.ndarray
    ) -> None:
        """
        Log model artifacts to MLflow including plots and data.
        
        Args:
            feature_importance: Feature importance dictionary
            X_train: Training features
            y_train: Training target
            y_train_pred: Training predictions
            y_train_proba: Training prediction probabilities
        """
        try:
            # Create temporary directory for plots
            import tempfile
            temp_dir = Path(tempfile.mkdtemp())
            
            # 1. Feature Importance Plot
            if feature_importance:
                self._create_feature_importance_plot(feature_importance, temp_dir)
            
            # 2. Confusion Matrix Plot
            self._create_confusion_matrix_plot(y_train, y_train_pred, temp_dir)
            
            # 3. ROC Curve Plot
            self._create_roc_curve_plot(y_train, y_train_proba, temp_dir)
            
            # 4. Precision-Recall Curve Plot
            self._create_pr_curve_plot(y_train, y_train_proba, temp_dir)
            
            # 5. Prediction Distribution Plot
            self._create_prediction_distribution_plot(y_train_proba, temp_dir)
            
            # Log all plots
            mlflow.log_artifacts(str(temp_dir), "plots")
            
            # 6. Log feature importance as JSON
            importance_path = temp_dir / "feature_importance.json"
            save_json(feature_importance, importance_path)
            mlflow.log_artifact(str(importance_path), "data")
            
            # Clean up
            import shutil
            shutil.rmtree(temp_dir)
            
            logger.info("Model artifacts logged to MLflow")
            
        except Exception as e:
            logger.warning(f"Failed to log artifacts to MLflow: {e}")
    
    def _create_feature_importance_plot(self, feature_importance: Dict[str, float], output_dir: Path) -> None:
        """Create feature importance bar plot."""
        # Get top 20 features
        top_features = dict(list(feature_importance.items())[:20])
        
        plt.figure(figsize=(10, 8))
        features = list(top_features.keys())
        importances = list(top_features.values())
        
        plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        plt.savefig(output_dir / "feature_importance.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_confusion_matrix_plot(self, y_true: pd.Series, y_pred: np.ndarray, output_dir: Path) -> None:
        """Create confusion matrix heatmap."""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Churn', 'Churn'], 
                   yticklabels=['Not Churn', 'Churn'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        plt.savefig(output_dir / "confusion_matrix.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_roc_curve_plot(self, y_true: pd.Series, y_proba: np.ndarray, output_dir: Path) -> None:
        """Create ROC curve plot."""
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(output_dir / "roc_curve.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_pr_curve_plot(self, y_true: pd.Series, y_proba: np.ndarray, output_dir: Path) -> None:
        """Create Precision-Recall curve plot."""
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap_score = average_precision_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AP = {ap_score:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(output_dir / "precision_recall_curve.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_prediction_distribution_plot(self, y_proba: np.ndarray, output_dir: Path) -> None:
        """Create prediction probability distribution plot."""
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(y_proba, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Prediction Probability')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Probabilities')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.boxplot(y_proba)
        plt.ylabel('Prediction Probability')
        plt.title('Probability Distribution Box Plot')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "prediction_distribution.png", dpi=150, bbox_inches='tight')
        plt.close()
    
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
    
    def optimize_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame, 
        y_val: pd.Series,
        n_trials: int = None,
        timeout: Optional[int] = None,
        direction: str = 'maximize',
        metric: str = 'roc_auc'
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            direction: Optimization direction ('maximize' or 'minimize')
            metric: Metric to optimize
            
        Returns:
            dict: Optimization results with best parameters
        """
        if n_trials is None:
            n_trials = self.config.get('n_trials', 100)
        
        if timeout is None:
            timeout = self.config.get('optuna_timeout', 3600)
        
        logger.info(f"Starting hyperparameter optimization: {n_trials} trials, {timeout}s timeout")
        
        def objective(trial):
            # Define search space
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'random_state': RANDOM_SEED,
                'n_jobs': -1,
                
                # Hyperparameters to optimize
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 4.0),
            }
            
            # Create and train model
            model = xgb.XGBClassifier(**params)
            
            # Add pruning callback
            pruning_callback = XGBoostPruningCallback(trial, f'validation_1-{params["eval_metric"]}')
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                callbacks=[pruning_callback],
                verbose=False
            )
            
            # Calculate validation metric
            y_val_pred = model.predict(X_val)
            y_val_proba = model.predict_proba(X_val)[:, 1]
            
            if metric == 'roc_auc':
                score = roc_auc_score(y_val, y_val_proba)
            elif metric == 'f1':
                score = f1_score(y_val, y_val_pred)
            elif metric == 'accuracy':
                score = accuracy_score(y_val, y_val_pred)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            
            return score
        
        # Create study
        study = optuna.create_study(
            direction=direction,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        # Start MLflow run for optimization
        with mlflow.start_run(run_name=f"optuna_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # Optimize
            study.optimize(objective, n_trials=n_trials, timeout=timeout)
            
            # Log optimization results
            best_params = study.best_params
            best_value = study.best_value
            
            mlflow.log_params(best_params)
            mlflow.log_metric(f"best_{metric}", best_value)
            mlflow.log_metric("n_trials_completed", len(study.trials))
            
            # Train final model with best parameters
            final_params = self.config.get('xgboost_params', {}).copy()
            final_params.update(best_params)
            
            self.model = xgb.XGBClassifier(**final_params)
            self.training_artifacts['training_params'] = final_params
            
            optimization_results = {
                'best_params': best_params,
                'best_value': best_value,
                'n_trials': len(study.trials),
                'study': study,
                'optimization_metric': metric
            }
            
            logger.info(f"Optimization completed: best {metric} = {best_value:.4f}")
            logger.info(f"Best parameters: {best_params}")
            
            return optimization_results
    
    def handle_class_imbalance(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        strategy: str = 'smote',
        **kwargs
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Handle class imbalance in training data.
        
        Args:
            X_train: Training features
            y_train: Training target
            strategy: Imbalance handling strategy ('smote', 'scale_pos_weight', 'none')
            **kwargs: Additional parameters for the strategy
            
        Returns:
            tuple: Balanced (X_train, y_train) or original if using scale_pos_weight
        """
        # Calculate class distribution
        class_counts = y_train.value_counts().sort_index()
        minority_class = class_counts.idxmin()
        majority_class = class_counts.idxmax()
        imbalance_ratio = class_counts[majority_class] / class_counts[minority_class]
        
        logger.info(f"Class distribution: {class_counts.to_dict()}")
        logger.info(f"Imbalance ratio: {imbalance_ratio:.2f}")
        
        if strategy == 'smote':
            # Use SMOTE for oversampling minority class
            smote_params = {
                'sampling_strategy': kwargs.get('sampling_strategy', 'auto'),
                'random_state': RANDOM_SEED,
                'k_neighbors': kwargs.get('k_neighbors', 5)
            }
            
            smote = SMOTE(**smote_params)
            X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
            
            # Convert back to DataFrame/Series with proper indices
            X_balanced = pd.DataFrame(X_balanced, columns=X_train.columns)
            y_balanced = pd.Series(y_balanced, name=y_train.name)
            
            balanced_counts = y_balanced.value_counts().sort_index()
            logger.info(f"After SMOTE: {balanced_counts.to_dict()}")
            
            return X_balanced, y_balanced
            
        elif strategy == 'scale_pos_weight':
            # Use XGBoost's built-in class weight balancing
            if self.model is None:
                self.initialize_model()
            
            scale_pos_weight = class_counts[majority_class] / class_counts[minority_class]
            
            # Update model parameters
            if hasattr(self.model, 'set_params'):
                self.model.set_params(scale_pos_weight=scale_pos_weight)
            
            # Store in training artifacts
            self.training_artifacts['training_params']['scale_pos_weight'] = scale_pos_weight
            
            logger.info(f"Set scale_pos_weight to {scale_pos_weight:.3f}")
            
            return X_train, y_train
            
        elif strategy == 'none':
            logger.info("No class imbalance handling applied")
            return X_train, y_train
            
        else:
            raise ValueError(f"Unknown imbalance strategy: {strategy}")
    
    def train_with_imbalance_handling(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        imbalance_strategy: str = 'smote',
        early_stopping_rounds: Optional[int] = 10,
        **imbalance_kwargs
    ) -> Dict[str, Any]:
        """
        Train model with automatic class imbalance handling.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            imbalance_strategy: Strategy for handling imbalance
            early_stopping_rounds: Early stopping patience
            **imbalance_kwargs: Additional arguments for imbalance handling
            
        Returns:
            dict: Training results with imbalance handling info
        """
        logger.info(f"Training with imbalance strategy: {imbalance_strategy}")
        
        # Handle class imbalance
        X_train_balanced, y_train_balanced = self.handle_class_imbalance(
            X_train, y_train, strategy=imbalance_strategy, **imbalance_kwargs
        )
        
        # Train model with balanced data
        training_results = self.train_model(
            X_train_balanced, y_train_balanced,
            X_val, y_val,
            early_stopping_rounds
        )
        
        # Add imbalance handling info to results
        training_results['imbalance_strategy'] = imbalance_strategy
        training_results['original_train_shape'] = X_train.shape
        training_results['balanced_train_shape'] = X_train_balanced.shape
        
        if imbalance_strategy == 'smote':
            training_results['smote_applied'] = True
            training_results['data_augmentation_ratio'] = len(X_train_balanced) / len(X_train)
        
        logger.info(f"Training completed with {imbalance_strategy} strategy")
        
        return training_results


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