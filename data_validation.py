"""
Data validation and utility functions for the Retail Price Optimization Dashboard.
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidator:
    """Class to handle data validation and quality checks."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.validation_results = {}
    
    def validate_data_structure(self) -> Dict[str, bool]:
        """Validate the basic structure of the dataset."""
        results = {}
        
        # Check required columns
        required_columns = [
            'product_id', 'product_category_name', 'month_year', 
            'qty', 'total_price', 'unit_price', 'product_score'
        ]
        
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        results['has_required_columns'] = len(missing_columns) == 0
        
        if missing_columns:
            st.error(f"❌ Missing required columns: {missing_columns}")
            logger.error(f"Missing required columns: {missing_columns}")
        
        # Check for empty dataset
        results['is_not_empty'] = len(self.data) > 0
        if not results['is_not_empty']:
            st.error("❌ Dataset is empty!")
            logger.error("Dataset is empty")
        
        # Check for duplicate rows
        duplicate_count = self.data.duplicated().sum()
        results['has_no_duplicates'] = duplicate_count == 0
        if duplicate_count > 0:
            st.warning(f"⚠️ Found {duplicate_count} duplicate rows")
            logger.warning(f"Found {duplicate_count} duplicate rows")
        
        return results
    
    def validate_data_types(self) -> Dict[str, bool]:
        """Validate data types of key columns."""
        results = {}
        
        # Check numeric columns
        numeric_columns = ['qty', 'total_price', 'unit_price', 'product_score']
        for col in numeric_columns:
            if col in self.data.columns:
                try:
                    pd.to_numeric(self.data[col], errors='raise')
                    results[f'{col}_is_numeric'] = True
                except (ValueError, TypeError):
                    results[f'{col}_is_numeric'] = False
                    st.error(f"❌ Column '{col}' contains non-numeric values")
                    logger.error(f"Column '{col}' contains non-numeric values")
        
        return results
    
    def validate_data_ranges(self) -> Dict[str, bool]:
        """Validate that data values are within reasonable ranges."""
        results = {}
        
        # Check price ranges
        if 'total_price' in self.data.columns:
            price_data = pd.to_numeric(self.data['total_price'], errors='coerce')
            results['prices_positive'] = (price_data > 0).all()
            results['prices_reasonable'] = (price_data < 10000).all()  # Assuming max price < $10k
            
            if not results['prices_positive']:
                st.error("❌ Found negative or zero prices")
                logger.error("Found negative or zero prices")
            
            if not results['prices_reasonable']:
                st.warning("⚠️ Found unusually high prices (>$10,000)")
                logger.warning("Found unusually high prices")
        
        # Check quantity ranges
        if 'qty' in self.data.columns:
            qty_data = pd.to_numeric(self.data['qty'], errors='coerce')
            results['quantities_positive'] = (qty_data > 0).all()
            results['quantities_reasonable'] = (qty_data < 1000).all()  # Assuming max qty < 1000
            
            if not results['quantities_positive']:
                st.error("❌ Found negative or zero quantities")
                logger.error("Found negative or zero quantities")
        
        return results
    
    def validate_missing_data(self) -> Dict[str, any]:
        """Check for missing data and provide summary."""
        results = {}
        
        missing_data = self.data.isnull().sum()
        results['missing_data_summary'] = missing_data[missing_data > 0].to_dict()
        results['total_missing_percentage'] = (missing_data.sum() / (len(self.data) * len(self.data.columns))) * 100
        
        if results['missing_data_summary']:
            st.warning("⚠️ Missing data detected:")
            for col, count in results['missing_data_summary'].items():
                percentage = (count / len(self.data)) * 100
                st.write(f"  - {col}: {count} missing ({percentage:.1f}%)")
                logger.warning(f"Missing data in {col}: {count} ({percentage:.1f}%)")
        
        return results
    
    def get_data_quality_score(self) -> float:
        """Calculate overall data quality score."""
        all_results = {}
        all_results.update(self.validate_data_structure())
        all_results.update(self.validate_data_types())
        all_results.update(self.validate_data_ranges())
        all_results.update(self.validate_missing_data())
        
        # Calculate score based on critical validations
        critical_checks = [
            'has_required_columns', 'is_not_empty', 'has_no_duplicates',
            'prices_positive', 'quantities_positive'
        ]
        
        passed_checks = sum(1 for check in critical_checks if all_results.get(check, False))
        quality_score = (passed_checks / len(critical_checks)) * 100
        
        return quality_score
    
    def run_full_validation(self) -> Dict[str, any]:
        """Run complete data validation."""
        logger.info("Starting data validation...")
        
        validation_results = {
            'structure': self.validate_data_structure(),
            'data_types': self.validate_data_types(),
            'ranges': self.validate_data_ranges(),
            'missing_data': self.validate_missing_data(),
            'quality_score': self.get_data_quality_score()
        }
        
        self.validation_results = validation_results
        logger.info(f"Data validation completed. Quality score: {validation_results['quality_score']:.1f}%")
        
        return validation_results

def safe_data_operation(operation_name: str):
    """Decorator for safe data operations with error handling."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except FileNotFoundError as e:
                st.error(f"❌ File not found during {operation_name}: {str(e)}")
                logger.error(f"File not found during {operation_name}: {str(e)}")
                st.stop()
            except pd.errors.EmptyDataError as e:
                st.error(f"❌ Empty data file during {operation_name}: {str(e)}")
                logger.error(f"Empty data file during {operation_name}: {str(e)}")
                st.stop()
            except pd.errors.ParserError as e:
                st.error(f"❌ Data parsing error during {operation_name}: {str(e)}")
                logger.error(f"Data parsing error during {operation_name}: {str(e)}")
                st.stop()
            except Exception as e:
                st.error(f"❌ Unexpected error during {operation_name}: {str(e)}")
                logger.error(f"Unexpected error during {operation_name}: {str(e)}")
                st.stop()
        return wrapper
    return decorator

@safe_data_operation("data loading")
def load_and_validate_data(file_path: str) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """Load data with comprehensive validation."""
    logger.info(f"Loading data from {file_path}")
    
    # Load data
    data = pd.read_csv(file_path)
    
    # Validate data
    validator = DataValidator(data)
    validation_results = validator.run_full_validation()
    
    # Display quality score
    quality_score = validation_results['quality_score']
    if quality_score >= 90:
        st.success(f"✅ Data quality score: {quality_score:.1f}% (Excellent)")
    elif quality_score >= 70:
        st.warning(f"⚠️ Data quality score: {quality_score:.1f}% (Good)")
    else:
        st.error(f"❌ Data quality score: {quality_score:.1f}% (Needs improvement)")
    
    return data, validation_results

def validate_model_inputs(X: pd.DataFrame, y: pd.Series) -> bool:
    """Validate inputs for machine learning model."""
    try:
        # Check for empty data
        if len(X) == 0 or len(y) == 0:
            st.error("❌ Empty dataset provided for model training")
            return False
        
        # Check for missing values
        if X.isnull().any().any():
            st.error("❌ Features contain missing values")
            return False
        
        if y.isnull().any():
            st.error("❌ Target variable contains missing values")
            return False
        
        # Check data types
        if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
            st.error("❌ Features must be numeric")
            return False
        
        if not pd.api.types.is_numeric_dtype(y):
            st.error("❌ Target variable must be numeric")
            return False
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error validating model inputs: {str(e)}")
        logger.error(f"Error validating model inputs: {str(e)}")
        return False

def display_data_summary(data: pd.DataFrame):
    """Display comprehensive data summary."""
    st.subheader("📊 Dataset Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(data):,}")
    
    with col2:
        st.metric("Total Features", len(data.columns))
    
    with col3:
        numeric_cols = len(data.select_dtypes(include=[np.number]).columns)
        st.metric("Numeric Features", numeric_cols)
    
    with col4:
        categorical_cols = len(data.select_dtypes(include=['object']).columns)
        st.metric("Categorical Features", categorical_cols)
    
    # Display basic statistics
    st.subheader("📈 Basic Statistics")
    st.dataframe(data.describe(), use_container_width=True)
    
    # Display data types
    st.subheader("🔍 Data Types")
    dtype_df = pd.DataFrame({
        'Column': data.columns,
        'Data Type': data.dtypes,
        'Non-Null Count': data.count(),
        'Null Count': data.isnull().sum()
    })
    st.dataframe(dtype_df, use_container_width=True)