"""
Unit tests for preprocessing module.

This module contains comprehensive tests for the data preprocessing pipeline
to ensure correct functionality, shape consistency, and data validation.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import shutil

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from preprocessing import DataPreprocessor, TargetEncoder
from config import RANDOM_SEED


class TestDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = DataPreprocessor()
        
        # Create sample test data
        self.sample_data = pd.DataFrame({
            'customerID': ['C001', 'C002', 'C003', 'C004', 'C005'],
            'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
            'SeniorCitizen': [0, 1, 0, 1, 0],
            'Partner': ['Yes', 'No', 'Yes', 'No', 'Yes'],
            'Dependents': ['No', 'Yes', 'No', 'Yes', 'No'],
            'tenure': [12, 24, 36, 6, 48],
            'PhoneService': ['Yes', 'Yes', 'No', 'Yes', 'Yes'],
            'MonthlyCharges': [50.0, 75.5, 25.0, 89.9, 60.0],
            'TotalCharges': ['600.0', '1811.0', '900.0', '539.4', '2880.0'],
            'Churn': ['No', 'Yes', 'No', 'Yes', 'No']
        })
        
        # Create a temporary directory for test artifacts
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_clean_column_names(self):
        """Test column name cleaning functionality."""
        df_with_spaces = pd.DataFrame({
            'Customer ID': [1, 2, 3],
            'Phone Service': ['Yes', 'No', 'Yes'],
            'Monthly_Charges': [50, 60, 70]
        })
        
        cleaned_df = self.preprocessor.clean_column_names(df_with_spaces)
        
        expected_columns = ['customer_id', 'phone_service', 'monthly_charges']
        self.assertEqual(list(cleaned_df.columns), expected_columns)
        
        # Check mapping is stored
        self.assertIn('column_mapping', self.preprocessor.preprocessing_artifacts)
        mapping = self.preprocessor.preprocessing_artifacts['column_mapping']
        self.assertEqual(mapping['Customer ID'], 'customer_id')
    
    def test_handle_missing_values(self):
        """Test missing value handling."""
        df_with_missing = self.sample_data.copy()
        df_with_missing.loc[0, 'TotalCharges'] = ' '  # Space character
        df_with_missing.loc[1, 'gender'] = np.nan
        df_with_missing.loc[2, 'tenure'] = np.nan
        
        filled_df = self.preprocessor.handle_missing_values(df_with_missing)
        
        # Check no missing values remain
        self.assertEqual(filled_df.isnull().sum().sum(), 0)
        
        # Check TotalCharges is numeric
        self.assertTrue(pd.api.types.is_numeric_dtype(filled_df['TotalCharges']))
    
    def test_create_tenure_cohorts(self):
        """Test tenure cohort creation."""
        df_with_cohorts = self.preprocessor.create_tenure_cohorts(self.sample_data)
        
        self.assertIn('tenure_cohort', df_with_cohorts.columns)
        
        # Check specific mappings
        self.assertEqual(df_with_cohorts.loc[0, 'tenure_cohort'], '1-2 years')  # tenure=12
        self.assertEqual(df_with_cohorts.loc[1, 'tenure_cohort'], '1-2 years')  # tenure=24
        self.assertEqual(df_with_cohorts.loc[3, 'tenure_cohort'], '0-1 year')   # tenure=6
        self.assertEqual(df_with_cohorts.loc[4, 'tenure_cohort'], '4-6 years')  # tenure=48
    
    def test_engineer_features(self):
        """Test feature engineering functionality."""
        engineered_df = self.preprocessor.engineer_features(self.sample_data)
        
        # Check new features are created
        expected_new_features = ['tenure_cohort', 'avg_monthly_charge', 'charge_increase_rate', 'total_services']
        
        for feature in expected_new_features:
            self.assertIn(feature, engineered_df.columns)
        
        # Check feature types
        self.assertTrue(pd.api.types.is_numeric_dtype(engineered_df['avg_monthly_charge']))
        self.assertTrue(pd.api.types.is_numeric_dtype(engineered_df['charge_increase_rate']))
        self.assertTrue(pd.api.types.is_numeric_dtype(engineered_df['total_services']))
    
    def test_get_feature_types(self):
        """Test feature type detection."""
        feature_types = self.preprocessor.get_feature_types(self.sample_data)
        
        # Check structure
        required_keys = ['categorical', 'numerical', 'binary', 'all_features']
        for key in required_keys:
            self.assertIn(key, feature_types)
        
        # Check specific categorizations
        self.assertIn('tenure', feature_types['numerical'])
        self.assertIn('MonthlyCharges', feature_types['numerical'])
        self.assertIn('gender', feature_types['binary'])
        self.assertIn('Partner', feature_types['binary'])
        
        # Check customer ID is excluded
        self.assertNotIn('customerID', feature_types['all_features'])
    
    def test_prepare_target(self):
        """Test target preparation."""
        X, y = self.preprocessor.prepare_target(self.sample_data)
        
        # Check shapes
        self.assertEqual(X.shape[0], self.sample_data.shape[0])
        self.assertEqual(len(y), self.sample_data.shape[0])
        
        # Check target is binary
        self.assertTrue(set(y.unique()).issubset({0, 1}))
        
        # Check customerID is removed from features
        self.assertNotIn('customerID', X.columns)
        self.assertNotIn('Churn', X.columns)
    
    def test_create_train_val_test_split(self):
        """Test train/validation/test splitting."""
        X, y = self.preprocessor.prepare_target(self.sample_data)
        
        # Use larger dataset for meaningful splits
        X_large = pd.concat([X] * 20, ignore_index=True)
        y_large = pd.concat([y] * 20, ignore_index=True)
        
        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocessor.create_train_val_test_split(
            X_large, y_large, train_size=0.7, val_size=0.15, test_size=0.15
        )
        
        # Check shapes
        total_samples = len(X_large)
        self.assertAlmostEqual(len(X_train) / total_samples, 0.7, delta=0.05)
        self.assertAlmostEqual(len(X_val) / total_samples, 0.15, delta=0.05)
        self.assertAlmostEqual(len(X_test) / total_samples, 0.15, delta=0.05)
        
        # Check no data leakage
        train_indices = set(X_train.index)
        val_indices = set(X_val.index)
        test_indices = set(X_test.index)
        
        self.assertEqual(len(train_indices & val_indices), 0)
        self.assertEqual(len(train_indices & test_indices), 0)
        self.assertEqual(len(val_indices & test_indices), 0)
    
    def test_fit_and_transform_scaler(self):
        """Test scaler fitting and transformation."""
        X, y = self.preprocessor.prepare_target(self.sample_data)
        
        # Fit scaler
        self.preprocessor.fit_scaler(X)
        
        # Check scaler is stored
        self.assertIn('scalers', self.preprocessor.preprocessing_artifacts)
        self.assertIn('numerical', self.preprocessor.preprocessing_artifacts['scalers'])
        
        # Transform data
        X_scaled = self.preprocessor.transform_numerical(X)
        
        # Check shape is preserved
        self.assertEqual(X_scaled.shape, X.shape)
        
        # Check numerical columns are scaled (approximately zero mean, unit variance)
        numerical_cols = self.preprocessor.get_feature_types(X)['numerical']
        for col in numerical_cols:
            if col in X_scaled.columns:
                self.assertAlmostEqual(X_scaled[col].mean(), 0.0, delta=0.1)
                self.assertAlmostEqual(X_scaled[col].std(), 1.0, delta=0.1)
    
    def test_build_preprocessing_pipeline(self):
        """Test sklearn pipeline construction."""
        X, y = self.preprocessor.prepare_target(self.sample_data)
        feature_types = self.preprocessor.get_feature_types(X)
        
        pipeline = self.preprocessor.build_preprocessing_pipeline(
            numerical_columns=feature_types['numerical']
        )
        
        # Check pipeline structure
        self.assertEqual(len(pipeline.steps), 2)
        self.assertEqual(pipeline.steps[0][0], 'target_encoder')
        self.assertEqual(pipeline.steps[1][0], 'column_transformer')
        
        # Test fit
        pipeline.fit(X, y)
        
        # Test transform
        X_transformed = pipeline.transform(X)
        
        # Check output is DataFrame or array
        self.assertIsNotNone(X_transformed)
    
    def test_save_and_load_artifacts(self):
        """Test artifact persistence."""
        # Create some artifacts
        self.preprocessor.preprocessing_artifacts['test_key'] = 'test_value'
        
        # Save artifacts
        artifact_path = self.temp_dir / 'test_artifacts.pkl'
        self.preprocessor.save_preprocessing_artifacts(artifact_path)
        
        # Check file exists
        self.assertTrue(artifact_path.exists())
        
        # Load artifacts in new instance
        new_preprocessor = DataPreprocessor()
        new_preprocessor.load_preprocessing_artifacts(artifact_path)
        
        # Check artifacts are loaded
        self.assertEqual(new_preprocessor.preprocessing_artifacts['test_key'], 'test_value')
        self.assertTrue(new_preprocessor.is_fitted)


class TestTargetEncoder(unittest.TestCase):
    """Test cases for TargetEncoder transformer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.encoder = TargetEncoder(['category'])
        
        # Create sample data
        self.X = pd.DataFrame({
            'category': ['A', 'B', 'A', 'C', 'B', 'A'],
            'other_col': [1, 2, 3, 4, 5, 6]
        })
        self.y = pd.Series([1, 0, 1, 1, 0, 0])  # A: 2/3, B: 0/2, C: 1/1
    
    def test_fit_transform(self):
        """Test fitting and transformation."""
        X_transformed = self.encoder.fit_transform(self.X, self.y)
        
        # Check that category column is transformed
        self.assertTrue(pd.api.types.is_numeric_dtype(X_transformed['category']))
        
        # Check approximate target means
        # A should be close to 2/3 ≈ 0.67
        # B should be close to 0/2 = 0.0  
        # C should be close to 1/1 = 1.0
        a_values = X_transformed[X_transformed.index.isin([0, 2, 5])]['category']
        self.assertAlmostEqual(a_values.iloc[0], 2/3, delta=0.01)
    
    def test_unseen_categories(self):
        """Test handling of unseen categories."""
        # Fit on subset
        X_train = self.X.iloc[:4]
        y_train = self.y.iloc[:4]
        self.encoder.fit(X_train, y_train)
        
        # Transform with unseen category
        X_new = pd.DataFrame({
            'category': ['A', 'D'],  # D is unseen
            'other_col': [1, 2]
        })
        
        X_transformed = self.encoder.transform(X_new)
        
        # Unseen category D should get global mean
        global_mean = y_train.mean()
        self.assertAlmostEqual(X_transformed.loc[1, 'category'], global_mean, delta=0.01)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)