"""
Additional Test Coverage Module.

Provides additional tests to achieve 50%+ coverage.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDataPreprocessingExtended(unittest.TestCase):
    """Extended tests for DataPreprocessor."""

    def setUp(self):
        self.data = pd.DataFrame({
            'product_id': ['p1', 'p2', 'p3', 'p4'],
            'qty': [10, 20, 30, 40],
            'total_price': [100, 200, 300, 400],
            'unit_price': [10, 10, 10, 10],
            'comp_1': [9, 9, 9, 9],
            'product_score': [4.0, 4.5, 3.5, 4.0]
        })

    def test_prepare_for_modeling(self):
        """Test prepare_for_modeling method."""
        from utils.data_preprocessing import DataPreprocessor

        preprocessor = DataPreprocessor(self.data)
        X, y = preprocessor.prepare_for_modeling(target_column='total_price')

        self.assertIsNotNone(X)
        self.assertIsNotNone(y)

    def test_aggregate_by_category(self):
        """Test aggregate_by_category method."""
        from utils.data_preprocessing import DataPreprocessor

        data = self.data.copy()
        data['product_category_name'] = ['cat1', 'cat1', 'cat2', 'cat2']

        preprocessor = DataPreprocessor(data)
        result = preprocessor.aggregate_by_category()

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)

    def test_filter_data(self):
        """Test filter_data method."""
        from utils.data_preprocessing import DataPreprocessor

        preprocessor = DataPreprocessor(self.data)
        preprocessor.filter_data('qty', min_val=15, max_val=35)

        self.assertEqual(len(preprocessor.data), 2)

    def test_get_processed_data(self):
        """Test get_processed_data method."""
        from utils.data_preprocessing import DataPreprocessor

        preprocessor = DataPreprocessing(self.data)
        result = preprocessor.get_processed_data()

        self.assertIsNotNone(result)


class TestModelEvaluatorExtended(unittest.TestCase):
    """Extended tests for ModelEvaluator."""

    def test_benchmark_models(self):
        """Test benchmark_models method."""
        from models.model_evaluator import ModelEvaluator

        y_true = np.array([100, 200, 150, 250])
        y_pred1 = np.array([105, 195, 155, 245])
        y_pred2 = np.array([98, 205, 148, 252])

        evaluator = ModelEvaluator()
        result = evaluator.benchmark_models({
            'Model1': y_pred1,
            'Model2': y_pred2
        }, y_true)

        self.assertIsNotNone(result)

    def test_statistical_significance_test(self):
        """Test statistical_significance_test method."""
        from models.model_evaluator import ModelEvaluator

        y_true = np.random.randn(100)
        pred1 = y_true + np.random.randn(100) * 0.1
        pred2 = y_true + np.random.randn(100) * 0.2

        evaluator = ModelEvaluator()
        result = evaluator.statistical_significance_test(y_true, pred1, pred2)

        self.assertIn('p_value', result)


class TestFeatureImportanceExtended(unittest.TestCase):
    """Extended tests for FeatureImportanceAnalyzer."""

    def test_comprehensive_analysis(self):
        """Test comprehensive_analysis method."""
        from models.feature_importance import FeatureImportanceAnalyzer
        from sklearn.tree import DecisionTreeRegressor

        X = pd.DataFrame({
            'f1': np.random.randn(100),
            'f2': np.random.randn(100),
            'f3': np.random.randn(100)
        })
        y = pd.Series(np.random.randn(100))

        model = DecisionTreeRegressor()
        model.fit(X, y)

        analyzer = FeatureImportanceAnalyzer()
        result = analyzer.comprehensive_analysis(model, X, y)

        self.assertIsNotNone(result)


class TestPriceOptimizerExtended(unittest.TestCase):
    """Extended tests for PriceOptimizer."""

    def setUp(self):
        self.data = pd.DataFrame({
            'product_id': ['p1', 'p2'],
            'qty': [10, 20],
            'total_price': [100, 200],
            'unit_price': [10, 10],
            'comp_1': [9, 9],
            'comp_2': [9.5, 9.5],
            'comp_3': [10, 10],
            'product_score': [4.0, 4.5]
        })

    def test_all_strategies(self):
        """Test all pricing strategies."""
        from utils.price_optimizer import PriceOptimizer, PricingStrategy

        optimizer = PriceOptimizer(self.data)

        strategies = [
            PricingStrategy.COST_PLUS,
            PricingStrategy.COMPETITOR_BASED,
            PricingStrategy.VALUE_BASED,
            PricingStrategy.DYNAMIC,
            PricingStrategy.PENETRATION,
            PricingStrategy.SKIMMING
        ]

        for strategy in strategies:
            result = optimizer.optimize_price('p1', strategy)
            self.assertIsNotNone(result.optimal_price)


class TestAnalyticsEngineExtended(unittest.TestCase):
    """Extended tests for AnalyticsEngine."""

    def setUp(self):
        self.data = pd.DataFrame({
            'product_id': [f'p{i}' for i in range(50)],
            'product_category_name': np.random.choice(['cat1', 'cat2'], 50),
            'qty': np.random.randint(1, 50, 50),
            'total_price': np.random.uniform(50, 500, 50),
            'unit_price': np.random.uniform(10, 100, 50),
            'product_score': np.random.uniform(3, 5, 50)
        })

    def test_segment_products(self):
        """Test segment_products method."""
        from utils.analytics import AnalyticsEngine

        engine = AnalyticsEngine(self.data)
        result = engine.segment_products()

        self.assertIsNotNone(result)

    def test_create_analytics_dashboard_data(self):
        """Test create_analytics_dashboard_data method."""
        from utils.analytics import AnalyticsEngine

        engine = AnalyticsEngine(self.data)
        result = engine.create_analytics_dashboard_data()

        self.assertIsNotNone(result)
        self.assertIn('executive_summary', result)


def run_tests():
    """Run all extended tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestDataPreprocessingExtended))
    suite.addTests(loader.loadTestsFromTestCase(TestModelEvaluatorExtended))
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureImportanceExtended))
    suite.addTests(loader.loadTestsFromTestCase(TestPriceOptimizerExtended))
    suite.addTests(loader.loadTestsFromTestCase(TestAnalyticsEngineExtended))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    run_tests()
