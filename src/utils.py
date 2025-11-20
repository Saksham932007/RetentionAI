"""
Utility functions for RetentionAI.

This module provides helper functions for logging, file operations,
data validation, and other common operations used throughout the application.
"""

import logging
import pickle
import json
import yaml
from typing import Any, Dict, Optional, Union, List
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

try:
    from .config import LOGS_DIR, PROCESSED_DATA_DIR
except ImportError:
    from config import LOGS_DIR, PROCESSED_DATA_DIR


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level
        format_string: Custom format string
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Ensure logs directory exists
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        file_path = LOGS_DIR / log_file
        
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def save_pickle(obj: Any, filepath: Union[str, Path]) -> None:
    """
    Save object to pickle file.
    
    Args:
        obj: Object to save
        filepath: Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    
    logging.getLogger(__name__).info(f"Saved pickle file: {filepath}")


def load_pickle(filepath: Union[str, Path]) -> Any:
    """
    Load object from pickle file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Any: Loaded object
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Pickle file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    
    logging.getLogger(__name__).info(f"Loaded pickle file: {filepath}")
    return obj


def save_json(obj: Dict[str, Any], filepath: Union[str, Path], indent: int = 2) -> None:
    """
    Save dictionary to JSON file.
    
    Args:
        obj: Dictionary to save
        filepath: Output file path
        indent: JSON indentation
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(obj, f, indent=indent, default=str)
    
    logging.getLogger(__name__).info(f"Saved JSON file: {filepath}")


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load dictionary from JSON file.
    
    Args:
        filepath: Input file path
        
    Returns:
        dict: Loaded dictionary
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        obj = json.load(f)
    
    logging.getLogger(__name__).info(f"Loaded JSON file: {filepath}")
    return obj


def save_yaml(obj: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """
    Save dictionary to YAML file.
    
    Args:
        obj: Dictionary to save
        filepath: Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        yaml.dump(obj, f, default_flow_style=False)
    
    logging.getLogger(__name__).info(f"Saved YAML file: {filepath}")


def load_yaml(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load dictionary from YAML file.
    
    Args:
        filepath: Input file path
        
    Returns:
        dict: Loaded dictionary
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"YAML file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        obj = yaml.safe_load(f)
    
    logging.getLogger(__name__).info(f"Loaded YAML file: {filepath}")
    return obj


def validate_dataframe(
    df: pd.DataFrame, 
    required_columns: Optional[List[str]] = None,
    min_rows: int = 1
) -> Dict[str, Any]:
    """
    Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        
    Returns:
        dict: Validation report
    """
    report = {
        'is_valid': True,
        'issues': [],
        'shape': df.shape,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        'null_counts': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.to_dict()
    }
    
    # Check minimum rows
    if len(df) < min_rows:
        report['issues'].append(f"Insufficient rows: {len(df)} < {min_rows}")
        report['is_valid'] = False
    
    # Check required columns
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            report['issues'].append(f"Missing required columns: {missing_cols}")
            report['is_valid'] = False
    
    # Check for empty DataFrame
    if df.empty:
        report['issues'].append("DataFrame is empty")
        report['is_valid'] = False
    
    # Check for all-null columns
    null_columns = [col for col, count in report['null_counts'].items() if count == len(df)]
    if null_columns:
        report['issues'].append(f"All-null columns: {null_columns}")
        report['is_valid'] = False
    
    return report


def get_memory_usage(obj: Any) -> float:
    """
    Get memory usage of an object in MB.
    
    Args:
        obj: Object to measure
        
    Returns:
        float: Memory usage in MB
    """
    if isinstance(obj, pd.DataFrame):
        return obj.memory_usage(deep=True).sum() / 1024 / 1024
    elif isinstance(obj, pd.Series):
        return obj.memory_usage(deep=True) / 1024 / 1024
    else:
        import sys
        return sys.getsizeof(obj) / 1024 / 1024


def create_timestamp_string() -> str:
    """
    Create timestamp string for file naming.
    
    Returns:
        str: Timestamp string (YYYYMMDD_HHMMSS)
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path: Ensured directory path
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, handling division by zero.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value for division by zero
        
    Returns:
        float: Division result or default
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values.
    
    Args:
        old_value: Original value
        new_value: New value
        
    Returns:
        float: Percentage change
    """
    return safe_divide((new_value - old_value), old_value, 0.0) * 100


def format_number(number: float, decimal_places: int = 2) -> str:
    """
    Format number with thousands separators and decimal places.
    
    Args:
        number: Number to format
        decimal_places: Number of decimal places
        
    Returns:
        str: Formatted number string
    """
    return f"{number:,.{decimal_places}f}"


def format_percentage(value: float, decimal_places: int = 1) -> str:
    """
    Format value as percentage.
    
    Args:
        value: Value to format (0-1 range)
        decimal_places: Number of decimal places
        
    Returns:
        str: Formatted percentage string
    """
    return f"{value * 100:.{decimal_places}f}%"


def truncate_string(text: str, max_length: int = 50, suffix: str = "...") -> str:
    """
    Truncate string to maximum length with suffix.
    
    Args:
        text: Text to truncate
        max_length: Maximum length including suffix
        suffix: Suffix to add when truncating
        
    Returns:
        str: Truncated string
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def get_class_distribution(y: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
    """
    Get class distribution statistics.
    
    Args:
        y: Target variable
        
    Returns:
        dict: Class distribution statistics
    """
    if isinstance(y, pd.Series):
        counts = y.value_counts()
    else:
        unique, counts = np.unique(y, return_counts=True)
        counts = dict(zip(unique, counts))
    
    total = sum(counts.values())
    
    return {
        'counts': dict(counts),
        'proportions': {k: v / total for k, v in counts.items()},
        'total': total,
        'n_classes': len(counts)
    }


if __name__ == "__main__":
    # Test utility functions
    logger = setup_logger("test_utils", "test_utils.log")
    logger.info("Testing utility functions")
    
    # Test data validation
    test_df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': ['a', 'b', 'c'],
        'C': [1.1, 2.2, 3.3]
    })
    
    validation_report = validate_dataframe(test_df, required_columns=['A', 'B'])
    print("Validation report:", validation_report)
    
    # Test other utilities
    print(f"Memory usage: {get_memory_usage(test_df):.2f} MB")
    print(f"Timestamp: {create_timestamp_string()}")
    print(f"Formatted number: {format_number(12345.678)}")
    print(f"Formatted percentage: {format_percentage(0.1234)}")
    
    print("Utility functions test completed!")