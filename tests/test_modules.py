"""
Test Suite for Retail Price Optimization Dashboard.

This module contains comprehensive unit tests for all components
of the retail price optimization system.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPricePredictor(unittest.TestCase):
    """Tests for Price Prediction models."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_data = pd.DataFrame({
            'product_id': ['p1', 'p2', 'p3', 'p4', 'p5'],
            'product_category_name': ['cat1', 'cat1', 'cat2', 'cat2', 'cat1'],
            'qty': [10, 20, 15, 25, 30],
            'total_price': [100.0, 200.0, 150.0, 250.0, 300.0],
            'unit_price': [10.0, 10.0, 10.0, 10.0, 10.0],
            'comp_1': [9.0, 9.5, 10.5, 11.0, 10.0],
            'comp_2': [9.5, 9.0, 10.0, 10.5, 9.5],
            'comp_3': [10.0, 10.5, 11.0, 11.5, 10.5],
            'product_score': [4.0, 4.5, 3.5, 4.0, 4.2]
        })
        self.sample_data['comp_price_diff'] = self.sample_data['unit_price'] - self.sample_data['comp_1']

    def test_data_preparation(self):
        """Test data preparation for modeling."""
        from models.price_predictor import PricePredictor

        predictor = PricePredictor()
        X, y = predictor.prepare_data(self.sample_data)

        self.assertEqual(X.shape[0], 5)
        self.assertEqual(y.shape[0], 5)
        self.assertIn('qty', X.columns)
        self.assertIn('unit_price', X.columns)

    def test_model_training(self):
        """Test model training."""
        from models.price_predictor import PricePredictor

        predictor = PricePredictor()
        predictor.fit(self.sample_data)

        self.assertTrue(predictor.is_fitted)

    def test_prediction(self):
        """Test price prediction."""
        from models.price_predictor import PricePredictor

        predictor = PricePredictor()
        predictor.fit(self.sample_data)

        predictions = predictor.predict(self.sample_data)

        self.assertEqual(len(predictions), 5)
        self.assertTrue(all(isinstance(p, (float, np.floating)) for p in predictions))

    def test_single_prediction(self):
        """Test single price prediction."""
        from models.price_predictor import PricePredictor

        predictor = PricePredictor()
        predictor.fit(self.sample_data)

        price = predictor.predict_single(
            qty=20,
            unit_price=10.0,
            comp_price=9.5,
            product_score=4.0
        )

        self.assertIsInstance(price, (float, np.floating))


class TestModelEvaluator(unittest.TestCase):
    """Tests for Model Evaluation."""

    def setUp(self):
        """Set up test fixtures."""
        self.y_true = np.array([100, 200, 150, 250, 300])
        self.y_pred = np.array([110, 190, 155, 245, 290])

    def test_evaluation_metrics(self):
        """Test evaluation metrics calculation."""
        from models.model_evaluator import ModelEvaluator

        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_model(self.y_true, self.y_pred, 'TestModel')

        self.assertIn('r2', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertGreater(metrics['r2'], 0)

    def test_residual_analysis(self):
        """Test residual analysis."""
        from models.model_evaluator import ModelEvaluator

        evaluator = ModelEvaluator()
        analysis = evaluator.residual_analysis(self.y_true, self.y_pred)

        self.assertIn('mean_residual', analysis)
        self.assertIn('std_residual', analysis)


class TestPriceOptimizer(unittest.TestCase):
    """Tests for Price Optimization."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_data = pd.DataFrame({
            'product_id': ['p1', 'p2', 'p3'],
            'product_category_name': ['cat1', 'cat1', 'cat2'],
            'qty': [10, 20, 15],
            'total_price': [100.0, 200.0, 150.0],
            'unit_price': [10.0, 10.0, 10.0],
            'comp_1': [9.0, 9.5, 10.5],
            'comp_2': [9.5, 9.0, 10.0],
            'comp_3': [10.0, 10.5, 11.0],
            'product_score': [4.0, 4.5, 3.5]
        })

    def test_cost_plus_pricing(self):
        """Test cost-plus pricing strategy."""
        from utils.price_optimizer import PriceOptimizer, PricingStrategy

        optimizer = PriceOptimizer(self.sample_data)
        result = optimizer.optimize_price('p1', PricingStrategy.COST_PLUS, markup=0.3)

        self.assertGreater(result.optimal_price, 0)
        self.assertEqual(result.strategy_used, 'cost_plus')

    def test_competitor_based_pricing(self):
        """Test competitor-based pricing."""
        from utils.price_optimizer import PriceOptimizer, PricingStrategy

        optimizer = PriceOptimizer(self.sample_data)
        result = optimizer.optimize_price(
            'p1',
            PricingStrategy.COMPETITOR_BASED,
            position='competitive'
        )

        self.assertGreater(result.optimal_price, 0)

    def test_invalid_product(self):
        """Test handling of invalid product ID."""
        from utils.price_optimizer import PriceOptimizer, PricingStrategy

        optimizer = PriceOptimizer(self.sample_data)

        with self.assertRaises(ValueError):
            optimizer.optimize_price('invalid_id', PricingStrategy.COST_PLUS)


class TestDataPreprocessor(unittest.TestCase):
    """Tests for Data Preprocessing."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_data = pd.DataFrame({
            'product_id': ['p1', 'p2', 'p3', 'p4'],
            'qty': [10, 20, None, 30],
            'total_price': [100.0, None, 150.0, 250.0],
            'unit_price': [10.0, 10.0, 10.0, 15.0],
            'comp_1': [9.0, 9.5, 10.5, 14.0],
            'product_score': [4.0, 4.5, 3.5, 5.0]
        })

    def test_missing_value_handling(self):
        """Test missing value imputation."""
        from utils.data_preprocessing import DataPreprocessor

        preprocessor = DataPreprocessor(self.sample_data)
        preprocessor.handle_missing_values(strategy='mean')

        self.assertEqual(preprocessor.data['qty'].isnull().sum(), 0)
        self.assertEqual(preprocessor.data['total_price'].isnull().sum(), 0)

    def test_feature_creation(self):
        """Test feature engineering."""
        from utils.data_preprocessing import DataPreprocessor

        preprocessor = DataPreprocessor(self.sample_data)
        preprocessor.create_features()

        self.assertIn('comp_price_diff', preprocessor.data.columns)

    def test_outlier_removal(self):
        """Test outlier removal."""
        from utils.data_preprocessing import DataPreprocessor

        data_with_outlier = self.sample_data.copy()
        data_with_outlier.loc[4] = ['p5', 1000, 10000.0, 100.0, 90.0, 1.0]

        preprocessor = DataPreprocessor(data_with_outlier)
        preprocessor.remove_outliers(method='iqr')

        self.assertLess(len(preprocessor.data), len(data_with_outlier))


class TestAnalyticsEngine(unittest.TestCase):
    """Tests for Analytics Engine."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_data = pd.DataFrame({
            'product_id': ['p1', 'p2', 'p3', 'p4', 'p5'],
            'product_category_name': ['cat1', 'cat1', 'cat2', 'cat2', 'cat1'],
            'qty': [10, 20, 15, 25, 30],
            'total_price': [100.0, 200.0, 150.0, 250.0, 300.0],
            'unit_price': [10.0, 10.0, 10.0, 10.0, 10.0],
            'product_score': [4.0, 4.5, 3.5, 4.0, 4.2],
            'customers': [5, 10, 8, 12, 15]
        })

    def test_descriptive_statistics(self):
        """Test descriptive statistics calculation."""
        from utils.analytics import AnalyticsEngine

        engine = AnalyticsEngine(self.sample_data)
        stats = engine.calculate_descriptive_statistics()

        self.assertIn('qty', stats.index)
        self.assertIn('mean', stats.columns)

    def test_category_analysis(self):
        """Test category performance analysis."""
        from utils.analytics import AnalyticsEngine

        engine = AnalyticsEngine(self.sample_data)
        category_perf = engine.analyze_category_performance()

        self.assertEqual(len(category_perf), 2)  # 2 categories

    def test_executive_summary(self):
        """Test executive summary generation."""
        from utils.analytics import AnalyticsEngine

        engine = AnalyticsEngine(self.sample_data)
        summary = engine.generate_executive_summary()

        self.assertEqual(summary['total_records'], 5)
        self.assertEqual(summary['total_products'], 5)


class TestFeatureImportanceAnalyzer(unittest.TestCase):
    """Tests for Feature Importance Analysis."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100)
        })
        self.y = pd.Series(
            self.X['feature1'] * 2 + self.X['feature2'] + np.random.randn(100) * 0.1
        )

    def test_correlation_analysis(self):
        """Test correlation analysis."""
        from models.feature_importance import FeatureImportanceAnalyzer

        analyzer = FeatureImportanceAnalyzer()
        result = analyzer.analyze_correlation(self.X, self.y)

        self.assertEqual(len(result), 3)

    def test_mutual_information(self):
        """Test mutual information calculation."""
        from models.feature_importance import FeatureImportanceAnalyzer

        analyzer = FeatureImportanceAnalyzer()
        result = analyzer.analyze_mutual_information(self.X, self.y)

        self.assertIn('feature', result.columns)
        self.assertIn('mutual_info', result.columns)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPricePredictor))
    suite.addTests(loader.loadTestsFromTestCase(TestModelEvaluator))
    suite.addTests(loader.loadTestsFromTestCase(TestPriceOptimizer))
    suite.addTests(loader.loadTestsFromTestCase(TestDataPreprocessor))
    suite.addTests(loader.loadTestsFromTestCase(TestAnalyticsEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureImportanceAnalyzer))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
