"""
Model promotion system for RetentionAI.

This module manages the promotion of best-performing models from MLflow experiments
to production directories, handling model versioning, validation, and deployment artifacts.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import shutil
import json
from datetime import datetime
import pickle

try:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    from mlflow.tracking import MlflowClient
except ImportError:
    mlflow = None
    MlflowClient = None

try:
    from .config import MODELS_DIR, EXPERIMENT_NAME, RANDOM_SEED
    from .predict import ChurnPredictor, load_best_model_from_mlflow
    from .train import ModelTrainer
    from .utils import save_json, load_json, ensure_dir
    from .database import get_database_manager
except ImportError:
    from config import MODELS_DIR, EXPERIMENT_NAME, RANDOM_SEED
    from predict import ChurnPredictor, load_best_model_from_mlflow
    from train import ModelTrainer
    from utils import save_json, load_json, ensure_dir
    from database import get_database_manager

# Configure logging
logger = logging.getLogger(__name__)


class ModelPromoter:
    """
    Manages model promotion from experiments to production.
    
    Handles model selection, validation, versioning, and deployment
    artifact creation for the churn prediction system.
    """
    
    def __init__(self, production_dir: Optional[Union[str, Path]] = None):
        """
        Initialize model promoter.
        
        Args:
            production_dir: Production models directory
        """
        self.production_dir = Path(production_dir) if production_dir else Path(MODELS_DIR) / "production"
        ensure_dir(self.production_dir)
        
        self.client = None
        if mlflow and MlflowClient:
            self.client = MlflowClient()
        
        self.model_registry = self._load_model_registry()
        
        logger.info(f"Model promoter initialized. Production dir: {self.production_dir}")
    
    def _load_model_registry(self) -> Dict[str, Any]:
        """Load existing model registry or create new one."""
        registry_path = self.production_dir / "model_registry.json"
        
        if registry_path.exists():
            return load_json(registry_path)
        else:
            return {
                'models': {},
                'current_production': None,
                'promotion_history': [],
                'created_at': datetime.now().isoformat()
            }
    
    def _save_model_registry(self) -> None:
        """Save model registry to file."""
        registry_path = self.production_dir / "model_registry.json"
        save_json(self.model_registry, registry_path)
    
    def find_best_model(
        self,
        experiment_name: str,
        metric_name: str = 'val_auc',
        min_metric_value: Optional[float] = None,
        max_age_days: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Find the best model from MLflow experiment.
        
        Args:
            experiment_name: MLflow experiment name
            metric_name: Metric to optimize for
            min_metric_value: Minimum metric value threshold
            max_age_days: Maximum model age in days
            
        Returns:
            dict: Best model info or None if no suitable model found
        """
        if not self.client:
            logger.error("MLflow client not available")
            return None
        
        try:
            experiment = self.client.get_experiment_by_name(experiment_name)
            if not experiment:
                logger.error(f"Experiment '{experiment_name}' not found")
                return None
            
            # Search for runs with metric
            filter_string = f"metrics.{metric_name} > 0"
            if min_metric_value:
                filter_string = f"metrics.{metric_name} >= {min_metric_value}"
            
            runs = self.client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=filter_string,
                order_by=[f"metrics.{metric_name} DESC"],
                max_results=10
            )
            
            if not runs:
                logger.warning(f"No runs found with metric {metric_name}")
                return None
            
            # Filter by age if specified
            if max_age_days:
                from datetime import timedelta
                cutoff_date = datetime.now() - timedelta(days=max_age_days)
                runs = [run for run in runs if 
                       datetime.fromtimestamp(run.info.start_time / 1000) > cutoff_date]
            
            if not runs:
                logger.warning("No recent runs found within age limit")
                return None
            
            best_run = runs[0]
            
            model_info = {
                'run_id': best_run.info.run_id,
                'experiment_id': best_run.info.experiment_id,
                'experiment_name': experiment_name,
                'metric_name': metric_name,
                'metric_value': best_run.data.metrics.get(metric_name),
                'run_name': best_run.data.tags.get('mlflow.runName', 'Unnamed'),
                'start_time': best_run.info.start_time,
                'params': best_run.data.params,
                'metrics': best_run.data.metrics,
                'tags': best_run.data.tags,
                'artifacts': self._list_run_artifacts(best_run.info.run_id)
            }
            
            logger.info(f"Best model found - Run: {best_run.info.run_id}, {metric_name}: {model_info['metric_value']:.4f}")
            
            return model_info
            
        except Exception as e:
            logger.error(f"Failed to find best model: {e}")
            return None
    
    def _list_run_artifacts(self, run_id: str) -> List[str]:
        """List artifacts for a specific run."""
        try:
            artifacts = self.client.list_artifacts(run_id)
            return [artifact.path for artifact in artifacts]
        except Exception as e:
            logger.warning(f"Failed to list artifacts for run {run_id}: {e}")
            return []
    
    def validate_model(
        self,
        model_info: Dict[str, Any],
        validation_data_table: str = "processed_data",
        min_samples: int = 100,
        min_auc: float = 0.7
    ) -> Dict[str, Any]:
        """
        Validate model before promotion.
        
        Args:
            model_info: Model information from find_best_model()
            validation_data_table: Database table for validation
            min_samples: Minimum samples required for validation
            min_auc: Minimum AUC threshold
            
        Returns:
            dict: Validation results
        """
        logger.info(f"Validating model {model_info['run_id']}")
        
        validation_results = {
            'model_id': model_info['run_id'],
            'validation_timestamp': datetime.now().isoformat(),
            'passed': False,
            'metrics': {},
            'checks': {},
            'errors': []
        }
        
        try:
            # Load model from MLflow
            model_uri = f"runs:/{model_info['run_id']}/model"
            predictor = ChurnPredictor()
            predictor._load_from_mlflow(model_uri)
            
            # Load validation data
            db_manager = get_database_manager()
            
            # Check if validation table exists and has enough data
            query = f"SELECT COUNT(*) as count FROM {validation_data_table}"
            count_result = db_manager.execute_query(query)
            sample_count = count_result.iloc[0]['count']
            
            validation_results['checks']['sample_count'] = sample_count
            validation_results['checks']['min_samples'] = min_samples
            validation_results['checks']['sufficient_samples'] = sample_count >= min_samples
            
            if sample_count < min_samples:
                validation_results['errors'].append(f"Insufficient validation data: {sample_count} < {min_samples}")
                return validation_results
            
            # Load validation dataset (use 20% of data)
            val_query = f"SELECT * FROM {validation_data_table} ORDER BY RANDOM() LIMIT {int(sample_count * 0.2)}"
            val_df = db_manager.execute_query(val_query)
            
            # Make predictions
            prediction_results = predictor.predict(val_df)
            
            # Extract actual vs predicted
            y_true = val_df['churn'].values
            y_pred = prediction_results['predictions']['churn_prediction'].values
            y_proba = prediction_results['predictions']['churn_probability'].values
            
            # Calculate validation metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            metrics = {
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'precision': float(precision_score(y_true, y_pred, zero_division=0)),
                'recall': float(recall_score(y_true, y_pred, zero_division=0)),
                'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
                'auc': float(roc_auc_score(y_true, y_proba))
            }
            
            validation_results['metrics'] = metrics
            
            # Check AUC threshold
            validation_results['checks']['auc_threshold'] = min_auc
            validation_results['checks']['auc_passed'] = metrics['auc'] >= min_auc
            
            if metrics['auc'] < min_auc:
                validation_results['errors'].append(f"AUC below threshold: {metrics['auc']:.4f} < {min_auc}")
            
            # Check for reasonable performance bounds
            performance_checks = {
                'accuracy_reasonable': metrics['accuracy'] > 0.5,
                'precision_reasonable': metrics['precision'] > 0.1,
                'recall_reasonable': metrics['recall'] > 0.1
            }
            validation_results['checks'].update(performance_checks)
            
            # Overall validation result
            validation_results['passed'] = (
                validation_results['checks']['sufficient_samples'] and
                validation_results['checks']['auc_passed'] and
                all(performance_checks.values())
            )
            
            if validation_results['passed']:
                logger.info(f"Model validation passed - AUC: {metrics['auc']:.4f}")
            else:
                logger.warning(f"Model validation failed - Errors: {validation_results['errors']}")
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            validation_results['errors'].append(f"Validation error: {str(e)}")
        
        return validation_results
    
    def promote_model(
        self,
        model_info: Dict[str, Any],
        validation_results: Dict[str, Any],
        version: Optional[str] = None,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Promote validated model to production.
        
        Args:
            model_info: Model information
            validation_results: Validation results
            version: Optional version string
            force: Force promotion even if validation failed
            
        Returns:
            dict: Promotion results
        """
        if not validation_results['passed'] and not force:
            logger.error("Cannot promote model: validation failed")
            return {
                'success': False,
                'reason': 'Validation failed',
                'validation_errors': validation_results['errors']
            }
        
        if not version:
            version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Promoting model {model_info['run_id']} as {version}")
        
        try:
            # Create version directory
            version_dir = self.production_dir / version
            ensure_dir(version_dir)
            
            # Download and save model artifacts
            model_uri = f"runs:/{model_info['run_id']}/model"
            
            # Save model in multiple formats
            predictor = ChurnPredictor()
            predictor._load_from_mlflow(model_uri)
            
            # XGBoost native format
            if hasattr(predictor.model, 'save_model'):
                model_path = version_dir / "model.json"
                predictor.model.save_model(model_path)
            
            # Pickle format for compatibility
            pickle_path = version_dir / "model.pkl"
            with open(pickle_path, 'wb') as f:
                pickle.dump({
                    'model': predictor.model,
                    'metadata': predictor.model_metadata
                }, f)
            
            # Save model metadata
            model_metadata = {
                'version': version,
                'run_id': model_info['run_id'],
                'experiment_name': model_info['experiment_name'],
                'promotion_timestamp': datetime.now().isoformat(),
                'metrics': model_info['metrics'],
                'params': model_info['params'],
                'validation_results': validation_results,
                'model_files': ['model.json', 'model.pkl']
            }
            
            metadata_path = version_dir / "metadata.json"
            save_json(model_metadata, metadata_path)
            
            # Download additional artifacts from MLflow
            if self.client:
                artifacts_dir = version_dir / "artifacts"
                ensure_dir(artifacts_dir)
                
                try:
                    for artifact_path in model_info.get('artifacts', []):
                        if artifact_path != 'model':  # Skip model as we've already saved it
                            artifact_local_path = artifacts_dir / artifact_path
                            ensure_dir(artifact_local_path.parent)
                            self.client.download_artifacts(
                                model_info['run_id'], 
                                artifact_path, 
                                str(artifacts_dir)
                            )
                except Exception as e:
                    logger.warning(f"Failed to download some artifacts: {e}")
            
            # Update model registry
            self.model_registry['models'][version] = model_metadata
            
            # Update current production model
            previous_production = self.model_registry.get('current_production')
            self.model_registry['current_production'] = version
            
            # Add to promotion history
            promotion_record = {
                'version': version,
                'timestamp': datetime.now().isoformat(),
                'run_id': model_info['run_id'],
                'previous_production': previous_production,
                'promoted_by': 'model_promoter',
                'validation_passed': validation_results['passed'],
                'force_promotion': force
            }
            
            if 'promotion_history' not in self.model_registry:
                self.model_registry['promotion_history'] = []
            
            self.model_registry['promotion_history'].append(promotion_record)
            
            # Save updated registry
            self._save_model_registry()
            
            # Create production symlinks/shortcuts
            self._create_production_links(version)
            
            promotion_results = {
                'success': True,
                'version': version,
                'model_path': str(version_dir),
                'previous_production': previous_production,
                'promotion_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Model promoted successfully to version {version}")
            
            return promotion_results
            
        except Exception as e:
            logger.error(f"Model promotion failed: {e}")
            return {
                'success': False,
                'reason': f"Promotion error: {str(e)}"
            }
    
    def _create_production_links(self, version: str) -> None:
        """Create convenient links to current production model."""
        current_link = self.production_dir / "current"
        version_dir = self.production_dir / version
        
        # Remove existing link
        if current_link.exists() or current_link.is_symlink():
            current_link.unlink()
        
        try:
            # Create symlink to current version
            current_link.symlink_to(version, target_is_directory=True)
            logger.info(f"Created symlink: {current_link} -> {version}")
        except OSError:
            # Fallback for systems that don't support symlinks
            logger.warning("Symlink creation failed, creating directory copy")
            if current_link.exists():
                shutil.rmtree(current_link)
            shutil.copytree(version_dir, current_link)
    
    def list_production_models(self) -> List[Dict[str, Any]]:
        """List all production models."""
        models = []
        
        for version, metadata in self.model_registry.get('models', {}).items():
            model_info = {
                'version': version,
                'is_current': version == self.model_registry.get('current_production'),
                'promotion_timestamp': metadata.get('promotion_timestamp'),
                'validation_auc': metadata.get('validation_results', {}).get('metrics', {}).get('auc'),
                'run_id': metadata.get('run_id'),
                'model_path': str(self.production_dir / version)
            }
            models.append(model_info)
        
        # Sort by promotion timestamp (newest first)
        models.sort(key=lambda x: x['promotion_timestamp'] or '', reverse=True)
        
        return models
    
    def rollback_model(self, target_version: str) -> Dict[str, Any]:
        """
        Rollback to a previous model version.
        
        Args:
            target_version: Version to rollback to
            
        Returns:
            dict: Rollback results
        """
        if target_version not in self.model_registry.get('models', {}):
            return {
                'success': False,
                'reason': f"Version {target_version} not found in registry"
            }
        
        logger.info(f"Rolling back to version {target_version}")
        
        try:
            previous_production = self.model_registry.get('current_production')
            self.model_registry['current_production'] = target_version
            
            # Add rollback to promotion history
            rollback_record = {
                'version': target_version,
                'timestamp': datetime.now().isoformat(),
                'rollback_from': previous_production,
                'action': 'rollback',
                'promoted_by': 'model_promoter'
            }
            
            self.model_registry['promotion_history'].append(rollback_record)
            
            # Save registry and update links
            self._save_model_registry()
            self._create_production_links(target_version)
            
            logger.info(f"Rollback successful: {previous_production} -> {target_version}")
            
            return {
                'success': True,
                'current_version': target_version,
                'previous_version': previous_production,
                'rollback_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return {
                'success': False,
                'reason': f"Rollback error: {str(e)}"
            }
    
    def cleanup_old_models(self, keep_versions: int = 5) -> Dict[str, Any]:
        """
        Clean up old model versions.
        
        Args:
            keep_versions: Number of versions to keep
            
        Returns:
            dict: Cleanup results
        """
        models = self.list_production_models()
        current_version = self.model_registry.get('current_production')
        
        if len(models) <= keep_versions:
            return {
                'cleaned_versions': [],
                'kept_versions': [m['version'] for m in models],
                'message': f"Only {len(models)} versions exist, no cleanup needed"
            }
        
        # Keep current version plus most recent versions
        to_keep = [current_version] if current_version else []
        for model in models:
            if len(to_keep) < keep_versions and model['version'] not in to_keep:
                to_keep.append(model['version'])
        
        to_remove = [model['version'] for model in models if model['version'] not in to_keep]
        
        cleaned_versions = []
        for version in to_remove:
            try:
                version_dir = self.production_dir / version
                if version_dir.exists():
                    shutil.rmtree(version_dir)
                    cleaned_versions.append(version)
                
                # Remove from registry
                if version in self.model_registry.get('models', {}):
                    del self.model_registry['models'][version]
                
            except Exception as e:
                logger.error(f"Failed to remove version {version}: {e}")
        
        if cleaned_versions:
            self._save_model_registry()
            logger.info(f"Cleaned up {len(cleaned_versions)} old model versions")
        
        return {
            'cleaned_versions': cleaned_versions,
            'kept_versions': to_keep,
            'message': f"Cleaned up {len(cleaned_versions)} versions, kept {len(to_keep)}"
        }


def promote_best_model(
    experiment_name: str = EXPERIMENT_NAME,
    metric_name: str = 'val_auc',
    min_auc: float = 0.7,
    force: bool = False
) -> Dict[str, Any]:
    """
    Find and promote the best model from experiments.
    
    Args:
        experiment_name: MLflow experiment name
        metric_name: Metric to optimize for
        min_auc: Minimum AUC threshold for validation
        force: Force promotion even if validation fails
        
    Returns:
        dict: Promotion results
    """
    promoter = ModelPromoter()
    
    # Find best model
    logger.info("Searching for best model...")
    model_info = promoter.find_best_model(
        experiment_name=experiment_name,
        metric_name=metric_name
    )
    
    if not model_info:
        return {
            'success': False,
            'reason': 'No suitable model found in experiments'
        }
    
    # Validate model
    logger.info("Validating model...")
    validation_results = promoter.validate_model(
        model_info=model_info,
        min_auc=min_auc
    )
    
    # Promote if validation passed (or forced)
    if validation_results['passed'] or force:
        logger.info("Promoting model to production...")
        promotion_results = promoter.promote_model(
            model_info=model_info,
            validation_results=validation_results,
            force=force
        )
        
        return {
            'success': promotion_results['success'],
            'model_info': model_info,
            'validation_results': validation_results,
            'promotion_results': promotion_results
        }
    else:
        return {
            'success': False,
            'reason': 'Model validation failed',
            'model_info': model_info,
            'validation_results': validation_results
        }


if __name__ == "__main__":
    # CLI interface for model promotion
    import argparse
    
    parser = argparse.ArgumentParser(description="Promote models to production")
    parser.add_argument("--experiment", default=EXPERIMENT_NAME, help="MLflow experiment name")
    parser.add_argument("--metric", default="val_auc", help="Metric to optimize for")
    parser.add_argument("--min-auc", type=float, default=0.7, help="Minimum AUC threshold")
    parser.add_argument("--force", action="store_true", help="Force promotion even if validation fails")
    parser.add_argument("--list", action="store_true", help="List production models")
    parser.add_argument("--rollback", help="Rollback to specific version")
    parser.add_argument("--cleanup", type=int, help="Clean up old models, keep N versions")
    
    args = parser.parse_args()
    
    promoter = ModelPromoter()
    
    try:
        if args.list:
            models = promoter.list_production_models()
            print("\nProduction Models:")
            print("=" * 80)
            for model in models:
                current_marker = " (CURRENT)" if model['is_current'] else ""
                print(f"Version: {model['version']}{current_marker}")
                print(f"  AUC: {model['validation_auc']:.4f}" if model['validation_auc'] else "  AUC: N/A")
                print(f"  Promoted: {model['promotion_timestamp']}")
                print(f"  Path: {model['model_path']}")
                print()
        
        elif args.rollback:
            result = promoter.rollback_model(args.rollback)
            if result['success']:
                print(f"✓ Rolled back to version {args.rollback}")
            else:
                print(f"✗ Rollback failed: {result['reason']}")
        
        elif args.cleanup is not None:
            result = promoter.cleanup_old_models(args.cleanup)
            print(f"✓ {result['message']}")
            if result['cleaned_versions']:
                print(f"Removed: {', '.join(result['cleaned_versions'])}")
        
        else:
            # Promote best model
            result = promote_best_model(
                experiment_name=args.experiment,
                metric_name=args.metric,
                min_auc=args.min_auc,
                force=args.force
            )
            
            if result['success']:
                version = result['promotion_results']['version']
                auc = result['validation_results']['metrics']['auc']
                print(f"✓ Model promoted successfully as {version} (AUC: {auc:.4f})")
            else:
                print(f"✗ Promotion failed: {result['reason']}")
                if 'validation_results' in result:
                    errors = result['validation_results'].get('errors', [])
                    if errors:
                        print("Validation errors:")
                        for error in errors:
                            print(f"  - {error}")
    
    except Exception as e:
        print(f"Error: {e}")
        exit(1)