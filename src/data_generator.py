"""
Data generation script for RetentionAI.

This module provides functionality to generate synthetic customer churn data
or validate/enhance existing datasets. Useful for testing and when original
data is not available.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path

from .config import RAW_DATA_DIR, RANDOM_SEED

# Configure logging
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)


class TelcoChurnDataGenerator:
    """
    Generator for synthetic telco customer churn data.
    
    Creates realistic customer data with correlations similar to actual
    telecommunications customer behavior patterns.
    """
    
    def __init__(self, random_seed: int = RANDOM_SEED):
        """
        Initialize data generator.
        
        Args:
            random_seed: Random seed for reproducible generation
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Define categorical value distributions
        self.categorical_distributions = {
            'gender': ['Male', 'Female'],
            'partner': ['Yes', 'No'],
            'dependents': ['Yes', 'No'],
            'phone_service': ['Yes', 'No'],
            'multiple_lines': ['Yes', 'No', 'No phone service'],
            'internet_service': ['DSL', 'Fiber optic', 'No'],
            'online_security': ['Yes', 'No', 'No internet service'],
            'online_backup': ['Yes', 'No', 'No internet service'],
            'device_protection': ['Yes', 'No', 'No internet service'],
            'tech_support': ['Yes', 'No', 'No internet service'],
            'streaming_tv': ['Yes', 'No', 'No internet service'],
            'streaming_movies': ['Yes', 'No', 'No internet service'],
            'contract': ['Month-to-month', 'One year', 'Two year'],
            'paperless_billing': ['Yes', 'No'],
            'payment_method': [
                'Electronic check', 'Mailed check', 
                'Bank transfer (automatic)', 'Credit card (automatic)'
            ]
        }
    
    def generate_customer_data(self, n_customers: int = 7043) -> pd.DataFrame:
        """
        Generate synthetic customer data.
        
        Args:
            n_customers: Number of customers to generate
            
        Returns:
            pd.DataFrame: Generated customer data
        """
        logger.info(f"Generating {n_customers} synthetic customer records")
        
        # Initialize DataFrame
        data = {}
        
        # Customer ID
        data['customerID'] = [f"C{str(i).zfill(7)}" for i in range(1, n_customers + 1)]
        
        # Demographics
        data['gender'] = np.random.choice(self.categorical_distributions['gender'], n_customers)
        data['SeniorCitizen'] = np.random.choice([0, 1], n_customers, p=[0.84, 0.16])
        data['Partner'] = np.random.choice(self.categorical_distributions['partner'], n_customers)
        data['Dependents'] = np.random.choice(self.categorical_distributions['dependents'], n_customers)
        
        # Services
        data['PhoneService'] = np.random.choice(['Yes', 'No'], n_customers, p=[0.9, 0.1])
        
        # Multiple lines (dependent on phone service)
        multiple_lines = []
        for phone in data['PhoneService']:
            if phone == 'No':
                multiple_lines.append('No phone service')
            else:
                multiple_lines.append(np.random.choice(['Yes', 'No'], p=[0.4, 0.6]))
        data['MultipleLines'] = multiple_lines
        
        # Internet service
        data['InternetService'] = np.random.choice(
            ['DSL', 'Fiber optic', 'No'], n_customers, p=[0.25, 0.44, 0.31]
        )
        
        # Internet-dependent services
        internet_services = [
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        
        for service in internet_services:
            service_values = []
            for internet in data['InternetService']:
                if internet == 'No':
                    service_values.append('No internet service')
                else:
                    service_values.append(np.random.choice(['Yes', 'No'], p=[0.3, 0.7]))
            data[service] = service_values
        
        # Contract and billing
        data['Contract'] = np.random.choice(
            ['Month-to-month', 'One year', 'Two year'], 
            n_customers, p=[0.55, 0.21, 0.24]
        )
        data['PaperlessBilling'] = np.random.choice(['Yes', 'No'], n_customers, p=[0.6, 0.4])
        data['PaymentMethod'] = np.random.choice(
            self.categorical_distributions['payment_method'], n_customers
        )
        
        # Tenure (correlated with contract type)
        tenure_values = []
        for contract in data['Contract']:
            if contract == 'Month-to-month':
                tenure = np.random.exponential(12)  # Lower average tenure
            elif contract == 'One year':
                tenure = np.random.normal(24, 12)   # Medium tenure
            else:  # Two year
                tenure = np.random.normal(36, 18)   # Higher tenure
            
            tenure_values.append(max(1, min(72, int(tenure))))  # Bound between 1-72 months
        data['tenure'] = tenure_values
        
        # Monthly charges (correlated with services)
        monthly_charges = []
        for i in range(n_customers):
            base_charge = 20
            
            # Add charges based on services
            if data['PhoneService'][i] == 'Yes':
                base_charge += np.random.uniform(15, 25)
            
            if data['InternetService'][i] == 'DSL':
                base_charge += np.random.uniform(25, 35)
            elif data['InternetService'][i] == 'Fiber optic':
                base_charge += np.random.uniform(50, 80)
            
            # Add small charges for additional services
            services_count = sum([
                data['OnlineSecurity'][i] == 'Yes',
                data['OnlineBackup'][i] == 'Yes',
                data['DeviceProtection'][i] == 'Yes',
                data['TechSupport'][i] == 'Yes',
                data['StreamingTV'][i] == 'Yes',
                data['StreamingMovies'][i] == 'Yes'
            ])
            base_charge += services_count * np.random.uniform(2, 8)
            
            monthly_charges.append(round(base_charge, 2))
        
        data['MonthlyCharges'] = monthly_charges
        
        # Total charges (tenure * monthly charges with some variation)
        total_charges = []
        for i in range(n_customers):
            base_total = data['tenure'][i] * data['MonthlyCharges'][i]
            # Add some random variation (discounts, one-time charges, etc.)
            variation = np.random.uniform(0.9, 1.1)
            total_charges.append(round(base_total * variation, 2))
        
        data['TotalCharges'] = total_charges
        
        # Churn (target variable - correlated with other features)
        churn_probability = []
        for i in range(n_customers):
            prob = 0.15  # Base probability
            
            # Contract type impact
            if data['Contract'][i] == 'Month-to-month':
                prob += 0.25
            elif data['Contract'][i] == 'Two year':
                prob -= 0.1
            
            # Tenure impact (inverse correlation)
            if data['tenure'][i] < 12:
                prob += 0.2
            elif data['tenure'][i] > 48:
                prob -= 0.15
            
            # High charges increase churn probability
            if data['MonthlyCharges'][i] > 80:
                prob += 0.15
            
            # Internet service impact
            if data['InternetService'][i] == 'Fiber optic':
                prob += 0.1  # Often more expensive
            
            # Senior citizens more likely to churn
            if data['SeniorCitizen'][i] == 1:
                prob += 0.1
            
            # Electronic check payment (often associated with higher churn)
            if data['PaymentMethod'][i] == 'Electronic check':
                prob += 0.1
            
            churn_probability.append(max(0.05, min(0.8, prob)))
        
        # Generate actual churn based on probabilities
        data['Churn'] = [
            'Yes' if np.random.random() < prob else 'No'
            for prob in churn_probability
        ]
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        logger.info(f"Generated synthetic data: {df.shape}")
        logger.info(f"Churn rate: {(df['Churn'] == 'Yes').mean():.3f}")
        
        return df
    
    def add_realistic_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add realistic data quality issues to make synthetic data more realistic.
        
        Args:
            df: Clean DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with realistic noise
        """
        df_noisy = df.copy()
        
        # Add some missing values to TotalCharges (common issue)
        missing_indices = np.random.choice(
            df_noisy.index, 
            size=int(0.002 * len(df_noisy)),  # 0.2% missing
            replace=False
        )
        df_noisy.loc[missing_indices, 'TotalCharges'] = ' '  # Space character
        
        # Add some inconsistent formatting
        # Random capitalization
        for col in ['gender', 'Partner', 'Dependents']:
            if col in df_noisy.columns:
                mask = np.random.random(len(df_noisy)) < 0.02  # 2% affected
                df_noisy.loc[mask, col] = df_noisy.loc[mask, col].str.lower()
        
        logger.info("Added realistic data quality issues to synthetic data")
        return df_noisy


def generate_or_validate_data(
    output_file: str = "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    n_customers: int = 7043,
    force_generate: bool = False
) -> Dict[str, Any]:
    """
    Generate synthetic data or validate existing data file.
    
    Args:
        output_file: Output filename for generated data
        n_customers: Number of customers to generate if creating new data
        force_generate: Force generation even if file exists
        
    Returns:
        dict: Generation/validation report
    """
    output_path = RAW_DATA_DIR / output_file
    
    report = {
        'action': 'none',
        'file_path': str(output_path),
        'file_exists': output_path.exists(),
        'generated': False,
        'validated': False,
        'data_shape': None,
        'issues': []
    }
    
    # Check if file exists and decide action
    if output_path.exists() and not force_generate:
        logger.info(f"Data file exists: {output_path}")
        
        # Validate existing file
        try:
            df = pd.read_csv(output_path)
            report['action'] = 'validated'
            report['validated'] = True
            report['data_shape'] = df.shape
            
            # Basic validation checks
            expected_columns = [
                'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
                'tenure', 'PhoneService', 'MonthlyCharges', 'TotalCharges', 'Churn'
            ]
            
            missing_cols = [col for col in expected_columns if col not in df.columns]
            if missing_cols:
                report['issues'].append(f"Missing expected columns: {missing_cols}")
            
            if df.empty:
                report['issues'].append("Data file is empty")
            
            logger.info(f"Validated existing data: {df.shape}, {len(report['issues'])} issues")
            
        except Exception as e:
            report['issues'].append(f"Failed to read existing file: {e}")
            force_generate = True  # Fall back to generation
    
    # Generate new data if needed
    if not output_path.exists() or force_generate or report['issues']:
        logger.info(f"Generating synthetic data: {n_customers} customers")
        
        try:
            # Ensure raw data directory exists
            RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
            
            # Generate data
            generator = TelcoChurnDataGenerator()
            df = generator.generate_customer_data(n_customers)
            
            # Add realistic noise
            df = generator.add_realistic_noise(df)
            
            # Save to CSV
            df.to_csv(output_path, index=False)
            
            report['action'] = 'generated'
            report['generated'] = True
            report['data_shape'] = df.shape
            report['file_exists'] = True
            
            logger.info(f"Synthetic data generated and saved: {output_path}")
            
        except Exception as e:
            error_msg = f"Failed to generate synthetic data: {e}"
            report['issues'].append(error_msg)
            logger.error(error_msg)
            raise
    
    return report


if __name__ == "__main__":
    # Test data generation
    import logging.config
    from .config import LOGGING_CONFIG
    
    logging.config.dictConfig(LOGGING_CONFIG)
    
    try:
        # Generate or validate data
        report = generate_or_validate_data()
        
        print("\nData Generation/Validation Report:")
        for key, value in report.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"Data generation failed: {e}")