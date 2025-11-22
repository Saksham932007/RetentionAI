"""
Data preprocessing module for RetentionAI.

This module provides comprehensive data preprocessing capabilities including
feature engineering, encoding, scaling, and pipeline creation for machine
learning models.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pickle
from pathlib import Path

try:
    from .config import (
        DATA_CONFIG, PROCESSED_DATA_DIR, RANDOM_SEED
    )
    from .database import DatabaseManager, get_database_manager
except ImportError:
    from config import (
        DATA_CONFIG, PROCESSED_DATA_DIR, RANDOM_SEED
    )
    from database import DatabaseManager, get_database_manager

# Configure logging
logger = logging.getLogger(__name__)


class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Simple target encoder for high-cardinality categorical features.

    This estimator computes the mean target value per category during fit
    and replaces categories with the corresponding mean during transform.
    Unseen categories are replaced with the global mean.
    """

    def __init__(self, columns: Optional[List[str]] = None, target_column: str = 'churn'):
        self.columns = columns
        self.target_column = target_column
        self._mappings: Dict[str, Dict[Any, float]] = {}
        self._global_mean: float = 0.0

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[pd.Series] = None):
        if isinstance(X, np.ndarray):
            raise ValueError('TargetEncoder requires a pandas DataFrame as input')

        df = X.copy()
        if y is None:
            if self.target_column in df.columns:
                y = df[self.target_column]
            else:
                raise ValueError('Target series must be provided if target column not in X')

        self._global_mean = float(y.mean())

        cols = self.columns or [c for c in df.columns if df[c].dtype == 'object']
        for col in cols:
            mapping = df.groupby(col)[self.target_column].mean().to_dict() if self.target_column in df.columns else df.groupby(col)[y.name].mean().to_dict()
            # Ensure numeric floats
            self._mappings[col] = {k: float(v) for k, v in mapping.items()}

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if isinstance(X, np.ndarray):
            raise ValueError('TargetEncoder requires a pandas DataFrame as input')

        df = X.copy()
        for col, mapping in self._mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(self._global_mean).astype(float)

        return df

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


class DataPreprocessor:
    """
    Base data preprocessor for customer churn prediction.
    
    Handles data cleaning, feature engineering, encoding, and scaling
    for machine learning pipeline preparation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data preprocessor.
        
        Args:
            config: Configuration dictionary, uses DATA_CONFIG if None
        """
        self.config = config or DATA_CONFIG.__dict__
        self.is_fitted = False
        
        # Store preprocessing artifacts
        self.preprocessing_artifacts = {
            'column_mapping': {},
            'feature_names': [],
            'target_encoders': {},
            'scalers': {},
            'stats': {}
        }
    
    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize column names.
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with cleaned column names
        """
        df_cleaned = df.copy()
        
        # Original to cleaned mapping
        original_columns = df_cleaned.columns.tolist()
        cleaned_columns = []
        
        for col in original_columns:
            # Convert to snake_case
            cleaned_col = (
                col.lower()
                .replace(' ', '_')
                .replace('-', '_')
                .replace('__', '_')
                .strip('_')
            )
            cleaned_columns.append(cleaned_col)
        
        # Store mapping
        self.preprocessing_artifacts['column_mapping'] = dict(zip(original_columns, cleaned_columns))
        
        # Apply cleaned names
        df_cleaned.columns = cleaned_columns
        
        logger.info(f"Cleaned column names: {len(cleaned_columns)} columns standardized")
        return df_cleaned
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values with appropriate strategies.
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with missing values handled
        """
        df_filled = df.copy()
        missing_report = df_filled.isnull().sum()
        
        logger.info(f"Missing values before handling: {missing_report.sum()} total")
        
        # Handle TotalCharges column (common issue in telco dataset)
        if 'totalcharges' in df_filled.columns:
            # Convert string with spaces to NaN, then to numeric
            df_filled['totalcharges'] = pd.to_numeric(
                df_filled['totalcharges'].astype(str).str.strip(), 
                errors='coerce'
            )
            
            # Fill missing TotalCharges with MonthlyCharges * tenure
            if 'monthlycharges' in df_filled.columns and 'tenure' in df_filled.columns:
                mask = df_filled['totalcharges'].isnull()
                df_filled.loc[mask, 'totalcharges'] = (
                    df_filled.loc[mask, 'monthlycharges'] * df_filled.loc[mask, 'tenure']
                )
                logger.info(f"Filled {mask.sum()} missing TotalCharges values")
        
        # Handle categorical missing values (fill with mode)
        categorical_columns = df_filled.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if df_filled[col].isnull().sum() > 0:
                mode_value = df_filled[col].mode().iloc[0] if len(df_filled[col].mode()) > 0 else 'Unknown'
                df_filled[col].fillna(mode_value, inplace=True)
                logger.info(f"Filled missing values in {col} with mode: {mode_value}")
        
        # Handle numerical missing values (fill with median)
        numerical_columns = df_filled.select_dtypes(include=[np.number]).columns
        for col in numerical_columns:
            if df_filled[col].isnull().sum() > 0:
                median_value = df_filled[col].median()
                df_filled[col].fillna(median_value, inplace=True)
                logger.info(f"Filled missing values in {col} with median: {median_value}")
        
        missing_after = df_filled.isnull().sum()
        logger.info(f"Missing values after handling: {missing_after.sum()} total")
        
        return df_filled
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features from raw data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with engineered features
        """
        df_engineered = df.copy()
        
        # Tenure cohorts (grouping by years)
        if 'tenure' in df_engineered.columns and self.config.get('create_tenure_cohorts', True):
            df_engineered = self.create_tenure_cohorts(df_engineered)
        
        # Monthly charges per service ratio
        if all(col in df_engineered.columns for col in ['monthlycharges', 'totalcharges', 'tenure']):
            # Average monthly charge
            df_engineered['avg_monthly_charge'] = df_engineered['totalcharges'] / df_engineered['tenure']
            df_engineered['avg_monthly_charge'].fillna(df_engineered['monthlycharges'], inplace=True)
            
            # Charge increase rate
            df_engineered['charge_increase_rate'] = (
                df_engineered['monthlycharges'] - df_engineered['avg_monthly_charge']
            ) / df_engineered['avg_monthly_charge']
            df_engineered['charge_increase_rate'].fillna(0, inplace=True)
            
            logger.info("Created charge-related engineered features")
        
        # Service count (total number of services)
        service_columns = [
            col for col in df_engineered.columns 
            if any(service in col.lower() for service in ['service', 'security', 'backup', 'protection', 'support', 'streaming'])
            and df_engineered[col].dtype == 'object'
        ]
        
        if service_columns:
            # Count 'Yes' values across service columns
            df_engineered['total_services'] = 0
            for col in service_columns:
                if df_engineered[col].dtype == 'object':
                    df_engineered['total_services'] += (df_engineered[col].str.lower() == 'yes').astype(int)
            logger.info(f"Created total_services feature from {len(service_columns)} service columns")
        
        # Contract value mapping
        if 'contract' in df_engineered.columns:
            contract_mapping = {
                'month-to-month': 1,
                'one year': 12,
                'two year': 24
            }
            df_engineered['contract_months'] = df_engineered['contract'].str.lower().map(contract_mapping)
            df_engineered['contract_months'].fillna(1, inplace=True)  # Default to month-to-month
            logger.info("Created contract_months numerical feature")
        
        logger.info(f"Feature engineering completed: {df_engineered.shape[1]} total features")
        return df_engineered

    def create_tenure_cohorts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create tenure cohort buckets from the `tenure` column.

        Args:
            df: Input DataFrame containing a `tenure` column (in months)

        Returns:
            pd.DataFrame: DataFrame with a new `tenure_cohort` column
        """
        df_out = df.copy()

        df_out['tenure_cohort'] = pd.cut(
            df_out['tenure'],
            bins=[-1, 12, 24, 48, 72, float('inf')],
            labels=['0-1 year', '1-2 years', '2-4 years', '4-6 years', '6+ years']
        ).astype(str)

        logger.info("Created tenure cohort feature via create_tenure_cohorts()")
        return df_out
    
    def get_feature_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Categorize features by type for appropriate preprocessing.
        
        Args:
            df: Input DataFrame
            
        Returns:
            dict: Dictionary with feature type categories
        """
        # Exclude target and ID columns
        exclude_cols = [self.config.get('target_column', 'churn')] + self.config.get('drop_columns', [])
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        categorical_features = []
        numerical_features = []
        binary_features = []
        
        for col in feature_cols:
            if col in df.columns:
                unique_vals = df[col].nunique()
                dtype = df[col].dtype
                
                if dtype == 'object' or (dtype in ['int64', 'float64'] and unique_vals <= 10):
                    if unique_vals == 2:
                        binary_features.append(col)
                    else:
                        categorical_features.append(col)
                else:
                    numerical_features.append(col)
        
        feature_types = {
            'categorical': categorical_features,
            'numerical': numerical_features,
            'binary': binary_features,
            'all_features': feature_cols
        }
        
        logger.info(f"Feature types identified: {len(categorical_features)} categorical, "
                   f"{len(numerical_features)} numerical, {len(binary_features)} binary")
        
        return feature_types

    def fit_scaler(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> None:
        """
        Fit a StandardScaler on numerical columns and store the scaler artifact.

        Args:
            df: Input DataFrame
            columns: List of numerical columns to scale. If None, detect automatically.
        """
        cols = columns or self.get_feature_types(df)['numerical']
        if not cols:
            logger.info("No numerical columns found to fit scaler")
            return

        scaler = StandardScaler()
        scaler.fit(df[cols].astype(float))
        self.preprocessing_artifacts['scalers']['numerical'] = {
            'columns': cols,
            'scaler': scaler
        }
        logger.info(f"Fitted StandardScaler on {len(cols)} numerical columns")

    def transform_numerical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply stored StandardScaler to numerical columns.

        Args:
            df: Input DataFrame

        Returns:
            pd.DataFrame: DataFrame with scaled numerical columns
        """
        df_out = df.copy()
        scaler_info = self.preprocessing_artifacts.get('scalers', {}).get('numerical')
        if not scaler_info:
            logger.info("No fitted numerical scaler found; returning original DataFrame")
            return df_out

        cols = scaler_info['columns']
        scaler: StandardScaler = scaler_info['scaler']
        df_out[cols] = scaler.transform(df_out[cols].astype(float))
        logger.info(f"Applied StandardScaler to {len(cols)} numerical columns")
        return df_out

    def build_preprocessing_pipeline(
        self,
        numerical_columns: Optional[List[str]] = None,
        target_encode_columns: Optional[List[str]] = None,
        passthrough: str = 'passthrough'
    ) -> Pipeline:
        """
        Build a sklearn Pipeline that applies target encoding (supervised)
        followed by column transformations (scaling for numerical features).

        Args:
            numerical_columns: List of numerical columns to scale. If None,
                they will be inferred from the data during fit.
            target_encode_columns: List of categorical columns to target-encode.
            passthrough: How to treat remaining columns in ColumnTransformer.

        Returns:
            sklearn.pipeline.Pipeline: Composed preprocessing pipeline
        """
        # Initialize target encoder (it will be fitted within pipeline.fit)
        target_encoder = TargetEncoder(columns=target_encode_columns)

        # ColumnTransformer will scale numerical columns and passthrough the rest
        num_transformer = StandardScaler()

        column_transformer = ColumnTransformer(
            transformers=[
                ('num', num_transformer, numerical_columns or []),
            ],
            remainder=passthrough,
            sparse_threshold=0
        )

        pipeline = Pipeline(steps=[
            ('target_encoder', target_encoder),
            ('column_transformer', column_transformer)
        ])

        logger.info("Built preprocessing sklearn Pipeline")
        return pipeline
    
    def prepare_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare target variable and features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            tuple: (features_df, target_series)
        """
        target_col = self.config.get('target_column', 'churn')
        drop_cols = self.config.get('drop_columns', [])
        
        # Extract target
        if target_col in df.columns:
            y = df[target_col].copy()
            
            # Convert target to binary if needed
            if y.dtype == 'object':
                y = y.str.lower().map({'yes': 1, 'no': 0})
            
        else:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        # Prepare features
        X = df.drop(columns=[target_col] + drop_cols, errors='ignore')
        
        logger.info(f"Target prepared: {y.value_counts().to_dict()}")
        logger.info(f"Features prepared: {X.shape}")
        
        return X, y
    
    def create_train_test_split(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        test_size: float = 0.2,
        random_state: int = RANDOM_SEED
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Create stratified train-test split.
        
        Args:
            X: Features DataFrame
            y: Target Series
            test_size: Proportion of dataset for test set
            random_state: Random seed for reproducibility
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        logger.info(f"Train-test split created:")
        logger.info(f"  Training: {X_train.shape[0]} samples")
        logger.info(f"  Testing: {X_test.shape[0]} samples")
        logger.info(f"  Train target distribution: {y_train.value_counts().to_dict()}")
        logger.info(f"  Test target distribution: {y_test.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def save_preprocessing_artifacts(self, filepath: Union[str, Path]) -> None:
        """
        Save preprocessing artifacts for later use.
        
        Args:
            filepath: Path to save artifacts
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.preprocessing_artifacts, f)
        
        logger.info(f"Preprocessing artifacts saved to: {filepath}")
    
    def load_preprocessing_artifacts(self, filepath: Union[str, Path]) -> None:
        """
        Load preprocessing artifacts from file.
        
        Args:
            filepath: Path to load artifacts from
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Preprocessing artifacts not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            self.preprocessing_artifacts = pickle.load(f)
        
        self.is_fitted = True
        logger.info(f"Preprocessing artifacts loaded from: {filepath}")


if __name__ == "__main__":
    # Test preprocessing functionality
    import logging.config
    
    try:
        from config import LOGGING_CONFIG
        logging.config.dictConfig(LOGGING_CONFIG)
        
        # Test with sample data
        processor = DataPreprocessor()
        print("DataPreprocessor initialized successfully!")
        
    except Exception as e:
        print(f"Preprocessing test failed: {e}")