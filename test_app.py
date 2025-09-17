"""
Unit tests for the Retail Price Optimization Dashboard.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_validation import DataValidator, validate_model_inputs, load_and_validate_data
from config import Config

class TestDataValidator(unittest.TestCase):
    """Test cases for DataValidator class."""
    
    def setUp(self):
        """Set up test data."""
        self.sample_data = pd.DataFrame({
            'product_id': ['prod1', 'prod2', 'prod3'],
            'product_category_name': ['category1', 'category2', 'category1'],
            'month_year': ['01-01-2023', '02-01-2023', '03-01-2023'],
            'qty': [10, 20, 30],
            'total_price': [100.0, 200.0, 300.0],
            'unit_price': [10.0, 10.0, 10.0],
            'product_score': [4.5, 3.8, 4.2]
        })
        self.validator = DataValidator(self.sample_data)
    
    def test_validate_data_structure_success(self):
        """Test successful data structure validation."""
        results = self.validator.validate_data_structure()
        
        self.assertTrue(results['has_required_columns'])
        self.assertTrue(results['is_not_empty'])
        self.assertTrue(results['has_no_duplicates'])
    
    def test_validate_data_structure_missing_columns(self):
        """Test data structure validation with missing columns."""
        incomplete_data = self.sample_data.drop(columns=['product_id'])
        validator = DataValidator(incomplete_data)
        results = validator.validate_data_structure()
        
        self.assertFalse(results['has_required_columns'])
    
    def test_validate_data_structure_empty_data(self):
        """Test data structure validation with empty data."""
        empty_data = pd.DataFrame()
        validator = DataValidator(empty_data)
        results = validator.validate_data_structure()
        
        self.assertFalse(results['is_not_empty'])
    
    def test_validate_data_structure_duplicates(self):
        """Test data structure validation with duplicates."""
        duplicate_data = pd.concat([self.sample_data, self.sample_data])
        validator = DataValidator(duplicate_data)
        results = validator.validate_data_structure()
        
        self.assertFalse(results['has_no_duplicates'])
    
    def test_validate_data_types_success(self):
        """Test successful data type validation."""
        results = self.validator.validate_data_types()
        
        self.assertTrue(results['qty_is_numeric'])
        self.assertTrue(results['total_price_is_numeric'])
        self.assertTrue(results['unit_price_is_numeric'])
        self.assertTrue(results['product_score_is_numeric'])
    
    def test_validate_data_types_non_numeric(self):
        """Test data type validation with non-numeric data."""
        non_numeric_data = self.sample_data.copy()
        non_numeric_data['qty'] = ['ten', 'twenty', 'thirty']
        validator = DataValidator(non_numeric_data)
        results = validator.validate_data_types()
        
        self.assertFalse(results['qty_is_numeric'])
    
    def test_validate_data_ranges_success(self):
        """Test successful data range validation."""
        results = self.validator.validate_data_ranges()
        
        self.assertTrue(results['prices_positive'])
        self.assertTrue(results['prices_reasonable'])
        self.assertTrue(results['quantities_positive'])
        self.assertTrue(results['quantities_reasonable'])
    
    def test_validate_data_ranges_negative_prices(self):
        """Test data range validation with negative prices."""
        negative_price_data = self.sample_data.copy()
        negative_price_data['total_price'] = [-100.0, 200.0, 300.0]
        validator = DataValidator(negative_price_data)
        results = validator.validate_data_ranges()
        
        self.assertFalse(results['prices_positive'])
    
    def test_validate_data_ranges_high_prices(self):
        """Test data range validation with unusually high prices."""
        high_price_data = self.sample_data.copy()
        high_price_data['total_price'] = [100.0, 200.0, 15000.0]  # Above threshold
        validator = DataValidator(high_price_data)
        results = validator.validate_data_ranges()
        
        self.assertFalse(results['prices_reasonable'])
    
    def test_validate_missing_data_success(self):
        """Test missing data validation with no missing data."""
        results = self.validator.validate_missing_data()
        
        self.assertEqual(len(results['missing_data_summary']), 0)
        self.assertEqual(results['total_missing_percentage'], 0.0)
    
    def test_validate_missing_data_with_missing(self):
        """Test missing data validation with missing values."""
        missing_data = self.sample_data.copy()
        missing_data.loc[0, 'total_price'] = np.nan
        missing_data.loc[1, 'qty'] = np.nan
        validator = DataValidator(missing_data)
        results = validator.validate_missing_data()
        
        self.assertGreater(len(results['missing_data_summary']), 0)
        self.assertGreater(results['total_missing_percentage'], 0.0)
    
    def test_get_data_quality_score_perfect(self):
        """Test data quality score calculation with perfect data."""
        score = self.validator.get_data_quality_score()
        self.assertEqual(score, 100.0)
    
    def test_get_data_quality_score_with_issues(self):
        """Test data quality score calculation with data issues."""
        problematic_data = self.sample_data.copy()
        problematic_data.loc[0, 'total_price'] = -100.0  # Negative price
        validator = DataValidator(problematic_data)
        score = validator.get_data_quality_score()
        
        self.assertLess(score, 100.0)

class TestModelValidation(unittest.TestCase):
    """Test cases for model input validation."""
    
    def setUp(self):
        """Set up test data."""
        self.X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'feature3': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        self.y = pd.Series([100, 200, 300, 400, 500])
    
    def test_validate_model_inputs_success(self):
        """Test successful model input validation."""
        result = validate_model_inputs(self.X, self.y)
        self.assertTrue(result)
    
    def test_validate_model_inputs_empty_X(self):
        """Test model input validation with empty features."""
        empty_X = pd.DataFrame()
        result = validate_model_inputs(empty_X, self.y)
        self.assertFalse(result)
    
    def test_validate_model_inputs_empty_y(self):
        """Test model input validation with empty target."""
        empty_y = pd.Series()
        result = validate_model_inputs(self.X, empty_y)
        self.assertFalse(result)
    
    def test_validate_model_inputs_missing_values_X(self):
        """Test model input validation with missing values in features."""
        X_with_nan = self.X.copy()
        X_with_nan.loc[0, 'feature1'] = np.nan
        result = validate_model_inputs(X_with_nan, self.y)
        self.assertFalse(result)
    
    def test_validate_model_inputs_missing_values_y(self):
        """Test model input validation with missing values in target."""
        y_with_nan = self.y.copy()
        y_with_nan.loc[0] = np.nan
        result = validate_model_inputs(self.X, y_with_nan)
        self.assertFalse(result)
    
    def test_validate_model_inputs_non_numeric_X(self):
        """Test model input validation with non-numeric features."""
        X_non_numeric = pd.DataFrame({
            'feature1': ['a', 'b', 'c', 'd', 'e'],
            'feature2': [10, 20, 30, 40, 50]
        })
        result = validate_model_inputs(X_non_numeric, self.y)
        self.assertFalse(result)
    
    def test_validate_model_inputs_non_numeric_y(self):
        """Test model input validation with non-numeric target."""
        y_non_numeric = pd.Series(['a', 'b', 'c', 'd', 'e'])
        result = validate_model_inputs(self.X, y_non_numeric)
        self.assertFalse(result)

class TestConfig(unittest.TestCase):
    """Test cases for configuration settings."""
    
    def test_config_attributes(self):
        """Test that all required config attributes exist."""
        self.assertIsNotNone(Config.APP_TITLE)
        self.assertIsNotNone(Config.APP_ICON)
        self.assertIsNotNone(Config.DATA_FILE)
        self.assertIsInstance(Config.REQUIRED_COLUMNS, list)
        self.assertIsInstance(Config.NUMERIC_COLUMNS, list)
        self.assertIsInstance(Config.FEATURE_COLUMNS, list)
        self.assertIsInstance(Config.MODEL_TYPES, list)
        self.assertIsInstance(Config.CHART_TYPES, list)
        self.assertIsInstance(Config.COLOR_THEMES, dict)
    
    def test_config_validation(self):
        """Test configuration validation."""
        # This test will fail if the data file doesn't exist
        # In a real scenario, you might want to mock this
        with patch('os.path.exists', return_value=True):
            result = Config.validate_config()
            self.assertTrue(result)
    
    def test_get_env_config(self):
        """Test environment configuration retrieval."""
        config = Config.get_env_config()
        self.assertIsInstance(config, dict)
        self.assertIn('STREAMLIT_SERVER_PORT', config)
        self.assertIn('STREAMLIT_SERVER_ADDRESS', config)
        self.assertIn('LOG_LEVEL', config)

class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    @patch('pandas.read_csv')
    def test_load_and_validate_data_success(self, mock_read_csv):
        """Test successful data loading and validation."""
        # Mock the CSV reading
        mock_data = pd.DataFrame({
            'product_id': ['prod1', 'prod2'],
            'product_category_name': ['cat1', 'cat2'],
            'month_year': ['01-01-2023', '02-01-2023'],
            'qty': [10, 20],
            'total_price': [100.0, 200.0],
            'unit_price': [10.0, 10.0],
            'product_score': [4.5, 3.8]
        })
        mock_read_csv.return_value = mock_data
        
        data, validation_results = load_and_validate_data('test.csv')
        
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIsInstance(validation_results, dict)
        self.assertIn('quality_score', validation_results)
        mock_read_csv.assert_called_once_with('test.csv')
    
    @patch('pandas.read_csv')
    def test_load_and_validate_data_file_not_found(self, mock_read_csv):
        """Test data loading with file not found error."""
        mock_read_csv.side_effect = FileNotFoundError("File not found")
        
        with patch('streamlit.error') as mock_error:
            with patch('streamlit.stop') as mock_stop:
                try:
                    load_and_validate_data('nonexistent.csv')
                except SystemExit:
                    pass  # streamlit.stop() raises SystemExit
                
                mock_error.assert_called()
                mock_stop.assert_called()

def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestDataValidator))
    test_suite.addTest(unittest.makeSuite(TestModelValidation))
    test_suite.addTest(unittest.makeSuite(TestConfig))
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)