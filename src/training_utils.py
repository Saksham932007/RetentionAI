"""
Training utilities module for RetentionAI.

This module provides utility functions and classes for enhanced training workflows,
including data sampling, cross-validation, ensemble methods, and training diagnostics.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from pathlib import Path
import json
from datetime import datetime
import warnings

# Scientific computing
from sklearn.model_selection import (
    KFold, StratifiedKFold, cross_val_score, 
    validation_curve, learning_curve
)
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, 
    average_precision_score
)

# Machine Learning
import xgboost as xgb
try:
    from sklearn.ensemble import VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
except ImportError:
    VotingClassifier = None
    LogisticRegression = None
    RandomForestClassifier = None

# Plotting
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.style.use('ggplot')
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# MLflow
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    from .config import RANDOM_SEED, MODEL_CONFIG
    from .utils import save_json, save_pickle, ensure_dir
    from .train import ModelTrainer
except ImportError:
    from config import RANDOM_SEED, MODEL_CONFIG
    from utils import save_json, save_pickle, ensure_dir
    from train import ModelTrainer

# Configure logging
logger = logging.getLogger(__name__)


class CrossValidationTrainer:
    """
    Enhanced training with cross-validation capabilities.
    
    Provides robust model evaluation using k-fold cross-validation,
    hyperparameter stability analysis, and comprehensive metric reporting.
    """
    
    def __init__(
        self, 
        n_folds: int = 5,
        random_seed: int = RANDOM_SEED,
        scoring_metrics: Optional[List[str]] = None
    ):
        """
        Initialize cross-validation trainer.
        
        Args:
            n_folds: Number of cross-validation folds
            random_seed: Random seed for reproducibility
            scoring_metrics: List of scoring metrics to evaluate
        """
        self.n_folds = n_folds
        self.random_seed = random_seed
        self.scoring_metrics = scoring_metrics or [
            'accuracy', 'precision', 'recall', 'f1', 'roc_auc'
        ]
        
        self.cv_results = {}
        self.fold_models = []
        
        logger.info(f"CrossValidationTrainer initialized with {n_folds} folds")
    
    def run_cross_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_params: Optional[Dict[str, Any]] = None,
        stratified: bool = True,
        return_estimators: bool = False
    ) -> Dict[str, Any]:
        """
        Run k-fold cross-validation.
        
        Args:
            X: Feature matrix
            y: Target vector
            model_params: Model hyperparameters
            stratified: Whether to use stratified folds
            return_estimators: Whether to return trained estimators
            
        Returns:
            dict: Cross-validation results
        """
        logger.info(f"Running {self.n_folds}-fold cross-validation")
        
        # Setup cross-validation
        if stratified:
            cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_seed)
        else:
            cv = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_seed)
        
        # Initialize model
        model_config = {**MODEL_CONFIG.__dict__, **(model_params or {})}
        model = xgb.XGBClassifier(
            random_state=self.random_seed,
            **{k: v for k, v in model_config.items() if not k.startswith('_')}
        )
        
        # Track results
        cv_results = {
            'fold_scores': {metric: [] for metric in self.scoring_metrics},
            'fold_predictions': [],
            'fold_feature_importance': [],
            'fold_models': [] if return_estimators else None
        }
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            logger.info(f"Training fold {fold_idx + 1}/{self.n_folds}")
            
            # Split data
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Train model
            fold_model = model.__class__(**model.get_params())
            fold_model.fit(X_train_fold, y_train_fold)
            
            # Make predictions
            y_pred = fold_model.predict(X_val_fold)
            y_proba = fold_model.predict_proba(X_val_fold)[:, 1]
            
            # Calculate metrics
            fold_metrics = self._calculate_fold_metrics(y_val_fold, y_pred, y_proba)
            
            # Store results
            for metric in self.scoring_metrics:
                if metric in fold_metrics:
                    cv_results['fold_scores'][metric].append(fold_metrics[metric])
            
            cv_results['fold_predictions'].append({
                'fold': fold_idx,
                'val_indices': val_idx,
                'y_true': y_val_fold,
                'y_pred': y_pred,
                'y_proba': y_proba
            })
            
            # Feature importance
            if hasattr(fold_model, 'feature_importances_'):
                cv_results['fold_feature_importance'].append(fold_model.feature_importances_)
            
            if return_estimators:
                cv_results['fold_models'].append(fold_model)
        
        # Aggregate results
        cv_summary = self._summarize_cv_results(cv_results)
        
        self.cv_results = {
            'summary': cv_summary,
            'detailed_results': cv_results,
            'metadata': {
                'n_folds': self.n_folds,
                'stratified': stratified,
                'random_seed': self.random_seed,
                'model_params': model_config,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        logger.info("Cross-validation completed")
        for metric in self.scoring_metrics:
            if metric in cv_summary['mean_scores']:
                mean_score = cv_summary['mean_scores'][metric]
                std_score = cv_summary['std_scores'][metric]
                logger.info(f"  {metric}: {mean_score:.4f} ± {std_score:.4f}")
        
        return self.cv_results
    
    def _calculate_fold_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate metrics for a single fold."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, roc_auc_score
        )
        
        metrics = {}
        
        try:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except Exception as e:
            logger.warning(f"Error calculating some metrics: {e}")
        
        return metrics
    
    def _summarize_cv_results(self, cv_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize cross-validation results."""
        summary = {
            'mean_scores': {},
            'std_scores': {},
            'confidence_intervals': {},
            'score_ranges': {}
        }
        
        for metric in self.scoring_metrics:
            scores = cv_results['fold_scores'].get(metric, [])
            if scores:
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                
                summary['mean_scores'][metric] = float(mean_score)
                summary['std_scores'][metric] = float(std_score)
                summary['score_ranges'][metric] = {
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores))
                }
                
                # 95% confidence interval
                ci_margin = 1.96 * std_score / np.sqrt(len(scores))
                summary['confidence_intervals'][metric] = {
                    'lower': float(mean_score - ci_margin),
                    'upper': float(mean_score + ci_margin)
                }
        
        # Feature importance summary
        if cv_results['fold_feature_importance']:
            importance_matrix = np.array(cv_results['fold_feature_importance'])
            summary['feature_importance'] = {
                'mean': importance_matrix.mean(axis=0).tolist(),
                'std': importance_matrix.std(axis=0).tolist()
            }
        
        return summary
    
    def plot_cv_results(self, save_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Plot cross-validation results.
        
        Args:
            save_path: Path to save plots
            
        Returns:
            dict: Plot information
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting not available - matplotlib/seaborn not installed")
            return {'error': 'Plotting dependencies not available'}
        
        if not self.cv_results:
            logger.warning("No cross-validation results to plot")
            return {'error': 'No CV results available'}
        
        logger.info("Generating cross-validation plots")
        
        plots_info = {}
        
        # Setup plotting
        plt.style.use('ggplot')
        
        # 1. Metric distributions across folds
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(self.scoring_metrics):
            if i >= len(axes):
                break
                
            scores = self.cv_results['detailed_results']['fold_scores'].get(metric, [])
            if scores:
                ax = axes[i]
                ax.bar(range(1, len(scores) + 1), scores, alpha=0.7)
                ax.axhline(y=np.mean(scores), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(scores):.4f}')
                ax.set_title(f'{metric.upper()} Across Folds')
                ax.set_xlabel('Fold')
                ax.set_ylabel(metric.upper())
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(len(self.scoring_metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            cv_plot_path = Path(save_path) / "cv_metrics.png"
            ensure_dir(cv_plot_path.parent)
            plt.savefig(cv_plot_path, dpi=300, bbox_inches='tight')
            plots_info['cv_metrics'] = str(cv_plot_path)
        
        plt.show()
        plt.close()
        
        # 2. Feature importance stability
        if 'feature_importance' in self.cv_results['summary']:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            importance_matrix = np.array(
                self.cv_results['detailed_results']['fold_feature_importance']
            )
            
            # Mean importance with error bars
            mean_importance = importance_matrix.mean(axis=0)
            std_importance = importance_matrix.std(axis=0)
            
            # Get top 20 features
            top_indices = np.argsort(mean_importance)[-20:]
            
            ax1.barh(range(len(top_indices)), mean_importance[top_indices], 
                    xerr=std_importance[top_indices], alpha=0.7)
            ax1.set_title('Top 20 Feature Importance (Mean ± Std)')
            ax1.set_xlabel('Importance')
            ax1.set_ylabel('Feature Index')
            ax1.grid(True, alpha=0.3)
            
            # Importance stability heatmap
            sns.heatmap(
                importance_matrix[:, top_indices].T, 
                ax=ax2, 
                cmap='viridis',
                cbar_kws={'label': 'Importance'}
            )
            ax2.set_title('Feature Importance Across Folds')
            ax2.set_xlabel('Fold')
            ax2.set_ylabel('Feature Index')
            
            plt.tight_layout()
            
            if save_path:
                importance_plot_path = Path(save_path) / "feature_importance_cv.png"
                plt.savefig(importance_plot_path, dpi=300, bbox_inches='tight')
                plots_info['feature_importance'] = str(importance_plot_path)
            
            plt.show()
            plt.close()
        
        return plots_info


class EnsembleTrainer:
    """
    Ensemble training with multiple algorithms.
    
    Combines different algorithms and training strategies to create
    robust ensemble models for improved prediction performance.
    """
    
    def __init__(self, random_seed: int = RANDOM_SEED):
        """
        Initialize ensemble trainer.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        self.base_models = {}
        self.ensemble_model = None
        self.ensemble_results = {}
        
        logger.info("EnsembleTrainer initialized")
    
    def setup_base_models(self, model_configs: Optional[Dict[str, Dict]] = None) -> None:
        """
        Setup base models for ensemble.
        
        Args:
            model_configs: Dictionary of model names to configurations
        """
        if model_configs is None:
            model_configs = self._get_default_model_configs()
        
        self.base_models = {}
        
        for name, config in model_configs.items():
            try:
                if name == 'xgboost':
                    model = xgb.XGBClassifier(random_state=self.random_seed, **config)
                elif name == 'random_forest' and RandomForestClassifier:
                    model = RandomForestClassifier(random_state=self.random_seed, **config)
                elif name == 'logistic_regression' and LogisticRegression:
                    model = LogisticRegression(random_state=self.random_seed, **config)
                else:
                    logger.warning(f"Skipping unavailable model: {name}")
                    continue
                
                self.base_models[name] = model
                logger.info(f"Added base model: {name}")
                
            except Exception as e:
                logger.error(f"Failed to create model {name}: {e}")
        
        logger.info(f"Setup {len(self.base_models)} base models")
    
    def _get_default_model_configs(self) -> Dict[str, Dict]:
        """Get default model configurations."""
        return {
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2
            },
            'logistic_regression': {
                'max_iter': 1000,
                'C': 1.0
            }
        }
    
    def train_ensemble(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        voting: str = 'soft'
    ) -> Dict[str, Any]:
        """
        Train ensemble model.
        
        Args:
            X_train: Training features
            X_val: Validation features
            y_train: Training targets
            y_val: Validation targets
            voting: Voting strategy ('hard' or 'soft')
            
        Returns:
            dict: Ensemble training results
        """
        if not self.base_models:
            self.setup_base_models()
        
        logger.info(f"Training ensemble with {len(self.base_models)} base models")
        
        # Train individual models
        individual_results = {}
        trained_models = []
        
        for name, model in self.base_models.items():
            logger.info(f"Training {name}")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Validate
                val_pred = model.predict(X_val)
                val_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
                
                metrics = {
                    'accuracy': float(accuracy_score(y_val, val_pred)),
                    'f1_score': float(f1_score(y_val, val_pred, zero_division=0))
                }
                
                if val_proba is not None:
                    metrics['auc'] = float(roc_auc_score(y_val, val_proba))
                
                individual_results[name] = {
                    'metrics': metrics,
                    'model': model
                }
                
                trained_models.append((name, model))
                
                logger.info(f"  {name} AUC: {metrics.get('auc', 'N/A'):.4f}")
                
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
        
        # Create ensemble
        if VotingClassifier and len(trained_models) > 1:
            logger.info(f"Creating voting ensemble with {voting} voting")
            
            self.ensemble_model = VotingClassifier(
                estimators=trained_models,
                voting=voting
            )
            
            # Train ensemble
            self.ensemble_model.fit(X_train, y_train)
            
            # Evaluate ensemble
            ensemble_pred = self.ensemble_model.predict(X_val)
            ensemble_proba = None
            
            if voting == 'soft':
                ensemble_proba = self.ensemble_model.predict_proba(X_val)[:, 1]
            
            # Calculate ensemble metrics
            ensemble_metrics = {
                'accuracy': float(accuracy_score(y_val, ensemble_pred)),
                'f1_score': float(f1_score(y_val, ensemble_pred, zero_division=0))
            }
            
            if ensemble_proba is not None:
                ensemble_metrics['auc'] = float(roc_auc_score(y_val, ensemble_proba))
            
            logger.info(f"Ensemble AUC: {ensemble_metrics.get('auc', 'N/A'):.4f}")
            
        else:
            logger.warning("Cannot create ensemble - insufficient models or VotingClassifier unavailable")
            ensemble_metrics = {}
        
        # Compile results
        self.ensemble_results = {
            'individual_models': individual_results,
            'ensemble_metrics': ensemble_metrics,
            'voting_strategy': voting,
            'n_base_models': len(trained_models),
            'timestamp': datetime.now().isoformat()
        }
        
        return self.ensemble_results
    
    def get_best_model(self, metric: str = 'auc') -> Tuple[str, Any]:
        """
        Get the best performing model.
        
        Args:
            metric: Metric to use for selection
            
        Returns:
            tuple: (model_name, model_object)
        """
        if not self.ensemble_results:
            raise ValueError("No ensemble results available. Train ensemble first.")
        
        best_score = -float('inf')
        best_model_name = None
        best_model = None
        
        # Check individual models
        for name, results in self.ensemble_results['individual_models'].items():
            score = results['metrics'].get(metric, -float('inf'))
            if score > best_score:
                best_score = score
                best_model_name = name
                best_model = results['model']
        
        # Check ensemble
        if self.ensemble_model and metric in self.ensemble_results['ensemble_metrics']:
            ensemble_score = self.ensemble_results['ensemble_metrics'][metric]
            if ensemble_score > best_score:
                best_model_name = 'ensemble'
                best_model = self.ensemble_model
        
        logger.info(f"Best model: {best_model_name} ({metric}: {best_score:.4f})")
        
        return best_model_name, best_model


class LearningCurveAnalyzer:
    """
    Analyze learning curves to understand model behavior.
    
    Provides insights into training efficiency, overfitting, and
    data requirements through learning curve analysis.
    """
    
    def __init__(self, random_seed: int = RANDOM_SEED):
        """
        Initialize learning curve analyzer.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        self.curve_results = {}
        
    def analyze_learning_curve(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_params: Optional[Dict[str, Any]] = None,
        train_sizes: Optional[np.ndarray] = None,
        cv: int = 5,
        scoring: str = 'roc_auc'
    ) -> Dict[str, Any]:
        """
        Analyze learning curve.
        
        Args:
            X: Feature matrix
            y: Target vector
            model_params: Model parameters
            train_sizes: Array of training set sizes
            cv: Number of CV folds
            scoring: Scoring metric
            
        Returns:
            dict: Learning curve results
        """
        logger.info("Analyzing learning curve")
        
        # Setup model
        model_config = {**MODEL_CONFIG.__dict__, **(model_params or {})}
        model = xgb.XGBClassifier(
            random_state=self.random_seed,
            **{k: v for k, v in model_config.items() if not k.startswith('_')}
        )
        
        # Setup train sizes
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        # Generate learning curve
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y,
            train_sizes=train_sizes,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            random_state=self.random_seed
        )
        
        # Calculate statistics
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        self.curve_results = {
            'train_sizes': train_sizes_abs.tolist(),
            'train_scores': {
                'mean': train_mean.tolist(),
                'std': train_std.tolist(),
                'individual': train_scores.tolist()
            },
            'validation_scores': {
                'mean': val_mean.tolist(),
                'std': val_std.tolist(),
                'individual': val_scores.tolist()
            },
            'analysis': self._analyze_curve_characteristics(
                train_sizes_abs, train_mean, val_mean
            ),
            'scoring_metric': scoring,
            'cv_folds': cv,
            'model_params': model_config
        }
        
        logger.info("Learning curve analysis completed")
        
        return self.curve_results
    
    def _analyze_curve_characteristics(
        self,
        train_sizes: np.ndarray,
        train_scores: np.ndarray,
        val_scores: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze learning curve characteristics."""
        analysis = {}
        
        # Gap between train and validation
        score_gap = train_scores - val_scores
        analysis['overfitting_severity'] = float(np.mean(score_gap))
        
        # Convergence analysis
        if len(val_scores) >= 3:
            recent_improvement = val_scores[-1] - val_scores[-3]
            analysis['convergence_trend'] = 'improving' if recent_improvement > 0.001 else 'plateaued'
            analysis['recent_improvement'] = float(recent_improvement)
        
        # Optimal training size recommendation
        best_val_idx = np.argmax(val_scores)
        analysis['optimal_train_size'] = int(train_sizes[best_val_idx])
        analysis['optimal_val_score'] = float(val_scores[best_val_idx])
        
        # Data efficiency
        halfway_idx = len(train_sizes) // 2
        if halfway_idx < len(val_scores):
            score_at_half = val_scores[halfway_idx]
            final_score = val_scores[-1]
            analysis['data_efficiency'] = float((final_score - score_at_half) / final_score)
        
        return analysis
    
    def plot_learning_curve(self, save_path: Optional[Path] = None) -> Optional[str]:
        """
        Plot learning curve.
        
        Args:
            save_path: Path to save plot
            
        Returns:
            str: Path to saved plot or None
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting not available")
            return None
        
        if not self.curve_results:
            logger.warning("No learning curve results to plot")
            return None
        
        plt.figure(figsize=(10, 6))
        
        train_sizes = self.curve_results['train_sizes']
        train_mean = self.curve_results['train_scores']['mean']
        train_std = self.curve_results['train_scores']['std']
        val_mean = self.curve_results['validation_scores']['mean']
        val_std = self.curve_results['validation_scores']['std']
        
        # Plot training scores
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
        plt.fill_between(
            train_sizes, 
            np.array(train_mean) - np.array(train_std),
            np.array(train_mean) + np.array(train_std),
            alpha=0.1, color='blue'
        )
        
        # Plot validation scores
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation score')
        plt.fill_between(
            train_sizes,
            np.array(val_mean) - np.array(val_std),
            np.array(val_mean) + np.array(val_std),
            alpha=0.1, color='red'
        )
        
        plt.xlabel('Training Set Size')
        plt.ylabel(f'{self.curve_results["scoring_metric"].upper()} Score')
        plt.title('Learning Curve')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        # Add analysis annotations
        analysis = self.curve_results.get('analysis', {})
        if analysis:
            plt.text(
                0.02, 0.98, 
                f"Overfitting: {analysis.get('overfitting_severity', 0):.3f}\n"
                f"Trend: {analysis.get('convergence_trend', 'N/A')}",
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )
        
        if save_path:
            plot_path = Path(save_path) / "learning_curve.png"
            ensure_dir(plot_path.parent)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(plot_path)
        
        plt.show()
        plt.close()
        return None


if __name__ == "__main__":
    # Test training utilities
    logger.info("Testing training utilities")
    
    # Generate sample data
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10,
        n_redundant=5, n_clusters_per_class=1, random_state=RANDOM_SEED
    )
    
    # Test cross-validation
    cv_trainer = CrossValidationTrainer(n_folds=3)
    cv_results = cv_trainer.run_cross_validation(X, y)
    
    print("Cross-Validation Results:")
    for metric, score in cv_results['summary']['mean_scores'].items():
        print(f"  {metric}: {score:.4f}")
    
    # Test ensemble
    ensemble_trainer = EnsembleTrainer()
    if len(ensemble_trainer.base_models) > 0:
        # Split for ensemble testing
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED
        )
        
        ensemble_results = ensemble_trainer.train_ensemble(
            X_train, X_val, y_train, y_val
        )
        
        print("\nEnsemble Results:")
        for name, results in ensemble_results['individual_models'].items():
            auc = results['metrics'].get('auc', 'N/A')
            print(f"  {name}: {auc:.4f}" if auc != 'N/A' else f"  {name}: {auc}")
    
    # Test learning curve
    curve_analyzer = LearningCurveAnalyzer()
    curve_results = curve_analyzer.analyze_learning_curve(X, y, cv=3)
    
    print("\nLearning Curve Analysis:")
    analysis = curve_results.get('analysis', {})
    for key, value in analysis.items():
        print(f"  {key}: {value}")