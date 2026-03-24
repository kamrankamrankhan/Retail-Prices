"""
Comprehensive Test Suite for Retail Price Optimization.

This module provides extensive test coverage for all components.
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
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'product_id': [f'p{i}' for i in range(100)],
            'product_category_name': np.random.choice(['cat1', 'cat2', 'cat3'], 100),
            'qty': np.random.randint(1, 50, 100),
            'total_price': np.random.uniform(50, 500, 100),
            'unit_price': np.random.uniform(10, 100, 100),
            'comp_1': np.random.uniform(8, 95, 100),
            'comp_2': np.random.uniform(8, 95, 100),
            'comp_3': np.random.uniform(8, 95, 100),
            'product_score': np.random.uniform(3.0, 5.0, 100)
        })
        self.sample_data['comp_price_diff'] = self.sample_data['unit_price'] - self.sample_data['comp_1']

    def test_data_preparation(self):
        """Test data preparation for modeling."""
        from models.price_predictor import PricePredictor

        predictor = PricePredictor()
        X, y = predictor.prepare_data(self.sample_data)

        self.assertEqual(X.shape[0], 100)
        self.assertEqual(y.shape[0], 100)
        self.assertIn('qty', X.columns)

    def test_model_training_random_forest(self):
        """Test Random Forest model training."""
        from models.price_predictor import PricePredictor

        predictor = PricePredictor(model_type='random_forest')
        predictor.fit(self.sample_data)

        self.assertTrue(predictor.is_fitted)

    def test_model_training_gradient_boosting(self):
        """Test Gradient Boosting model training."""
        from models.price_predictor import PricePredictor

        predictor = PricePredictor(model_type='gradient_boosting')
        predictor.fit(self.sample_data)

        self.assertTrue(predictor.is_fitted)

    def test_model_training_decision_tree(self):
        """Test Decision Tree model training."""
        from models.price_predictor import PricePredictor

        predictor = PricePredictor(model_type='decision_tree')
        predictor.fit(self.sample_data)

        self.assertTrue(predictor.is_fitted)

    def test_model_training_linear(self):
        """Test Linear Regression model training."""
        from models.price_predictor import PricePredictor

        predictor = PricePredictor(model_type='linear')
        predictor.fit(self.sample_data)

        self.assertTrue(predictor.is_fitted)

    def test_prediction(self):
        """Test price prediction."""
        from models.price_predictor import PricePredictor

        predictor = PricePredictor()
        predictor.fit(self.sample_data)

        predictions = predictor.predict(self.sample_data)

        self.assertEqual(len(predictions), 100)
        self.assertTrue(all(isinstance(p, (float, np.floating)) for p in predictions))

    def test_single_prediction(self):
        """Test single price prediction."""
        from models.price_predictor import PricePredictor

        predictor = PricePredictor()
        predictor.fit(self.sample_data)

        price = predictor.predict_single(
            qty=20,
            unit_price=50.0,
            comp_price=45.0,
            product_score=4.0
        )

        self.assertIsInstance(price, (float, np.floating))

    def test_feature_importance(self):
        """Test feature importance extraction."""
        from models.price_predictor import PricePredictor

        predictor = PricePredictor()
        predictor.fit(self.sample_data)

        importance = predictor.get_feature_importance()

        self.assertIsNotNone(importance)
        self.assertIn('feature', importance.columns)
        self.assertIn('importance', importance.columns)

    def test_invalid_model_type(self):
        """Test invalid model type handling."""
        from models.price_predictor import PricePredictor

        with self.assertRaises(ValueError):
            PricePredictor(model_type='invalid_model')

    def test_prediction_before_training(self):
        """Test prediction before training raises error."""
        from models.price_predictor import PricePredictor

        predictor = PricePredictor()
        
        with self.assertRaises(ValueError):
            predictor.predict(self.sample_data)


class TestModelEvaluator(unittest.TestCase):
    """Tests for Model Evaluation."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.y_true = np.random.uniform(50, 500, 100)
        self.y_pred = self.y_true + np.random.normal(0, 20, 100)

    def test_evaluation_metrics(self):
        """Test evaluation metrics calculation."""
        from models.model_evaluator import ModelEvaluator

        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_model(self.y_true, self.y_pred, 'TestModel')

        self.assertIn('r2', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('mse', metrics)
        self.assertGreater(metrics['r2'], 0)

    def test_residual_analysis(self):
        """Test residual analysis."""
        from models.model_evaluator import ModelEvaluator

        evaluator = ModelEvaluator()
        analysis = evaluator.residual_analysis(self.y_true, self.y_pred)

        self.assertIn('mean_residual', analysis)
        self.assertIn('std_residual', analysis)
        self.assertIn('min_residual', analysis)
        self.assertIn('max_residual', analysis)

    def test_error_distribution(self):
        """Test prediction error distribution."""
        from models.model_evaluator import ModelEvaluator

        evaluator = ModelEvaluator()
        distribution = evaluator.prediction_error_distribution(self.y_true, self.y_pred)

        self.assertIn('errors', distribution)
        self.assertIn('percentage_errors', distribution)
        self.assertIn('mean_error', distribution)

    def test_evaluation_plot(self):
        """Test evaluation plot generation."""
        from models.model_evaluator import ModelEvaluator

        evaluator = ModelEvaluator()
        fig = evaluator.create_evaluation_plot(self.y_true, self.y_pred, 'Test')

        self.assertIsNotNone(fig)

    def test_evaluation_report(self):
        """Test evaluation report generation."""
        from models.model_evaluator import ModelEvaluator

        evaluator = ModelEvaluator()
        report = evaluator.generate_evaluation_report(self.y_true, self.y_pred, 'Test')

        self.assertIn('model_name', report)
        self.assertIn('metrics', report)
        self.assertIn('summary', report)


class TestPriceOptimizer(unittest.TestCase):
    """Tests for Price Optimization."""

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

    def test_value_based_pricing(self):
        """Test value-based pricing."""
        from utils.price_optimizer import PriceOptimizer, PricingStrategy

        optimizer = PriceOptimizer(self.sample_data)
        result = optimizer.optimize_price(
            'p1',
            PricingStrategy.VALUE_BASED,
            max_wtp=100.0
        )

        self.assertGreater(result.optimal_price, 0)

    def test_dynamic_pricing(self):
        """Test dynamic pricing strategy."""
        from utils.price_optimizer import PriceOptimizer, PricingStrategy

        optimizer = PriceOptimizer(self.sample_data)
        result = optimizer.optimize_price(
            'p1',
            PricingStrategy.DYNAMIC,
            demand_factor=1.2,
            inventory_level=0.5
        )

        self.assertGreater(result.optimal_price, 0)

    def test_penetration_pricing(self):
        """Test penetration pricing strategy."""
        from utils.price_optimizer import PriceOptimizer, PricingStrategy

        optimizer = PriceOptimizer(self.sample_data)
        result = optimizer.optimize_price(
            'p1',
            PricingStrategy.PENETRATION,
            market_share_target=0.1
        )

        self.assertGreater(result.optimal_price, 0)

    def test_skimming_pricing(self):
        """Test skimming pricing strategy."""
        from utils.price_optimizer import PriceOptimizer, PricingStrategy

        optimizer = PriceOptimizer(self.sample_data)
        result = optimizer.optimize_price(
            'p1',
            PricingStrategy.SKIMMING,
            market_maturity=0.3
        )

        self.assertGreater(result.optimal_price, 0)

    def test_invalid_product(self):
        """Test handling of invalid product ID."""
        from utils.price_optimizer import PriceOptimizer, PricingStrategy

        optimizer = PriceOptimizer(self.sample_data)

        with self.assertRaises(ValueError):
            optimizer.optimize_price('invalid_id', PricingStrategy.COST_PLUS)

    def test_price_elasticity_analysis(self):
        """Test price elasticity analysis."""
        from utils.price_optimizer import PriceOptimizer

        optimizer = PriceOptimizer(self.sample_data)
        elasticity = optimizer.analyze_price_elasticity('p1')

        self.assertIn('elasticity', elasticity)
        self.assertIn('interpretation', elasticity)

    def test_batch_optimization(self):
        """Test batch optimization."""
        from utils.price_optimizer import PriceOptimizer, PricingStrategy

        optimizer = PriceOptimizer(self.sample_data)
        results = optimizer.batch_optimization(['p1', 'p2', 'p3'], PricingStrategy.COST_PLUS)

        self.assertEqual(len(results), 3)

    def test_optimal_markup_calculation(self):
        """Test optimal markup calculation."""
        from utils.price_optimizer import PriceOptimizer

        optimizer = PriceOptimizer(self.sample_data)
        result = optimizer.calculate_optimal_markup('p1', target_margin=0.3)

        self.assertIn('optimal_price', result)
        self.assertIn('expected_margin', result)


class TestDataPreprocessor(unittest.TestCase):
    """Tests for Data Preprocessing."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_data = pd.DataFrame({
            'product_id': ['p1', 'p2', 'p3', 'p4', 'p5', 'p6'],
            'qty': [10, 20, None, 30, 40, 50],
            'total_price': [100.0, None, 150.0, 250.0, 300.0, 350.0],
            'unit_price': [10.0, 10.0, 10.0, 10.0, 15.0, 100.0],
            'comp_1': [9.0, 9.5, 10.5, 14.0, 14.0, 90.0],
            'product_score': [4.0, 4.5, 3.5, 5.0, 4.2, 4.8]
        })

    def test_missing_value_handling_mean(self):
        """Test missing value imputation with mean."""
        from utils.data_preprocessing import DataPreprocessor

        preprocessor = DataPreprocessor(self.sample_data)
        preprocessor.handle_missing_values(strategy='mean')

        self.assertEqual(preprocessor.data['qty'].isnull().sum(), 0)
        self.assertEqual(preprocessor.data['total_price'].isnull().sum(), 0)

    def test_missing_value_handling_median(self):
        """Test missing value imputation with median."""
        from utils.data_preprocessing import DataPreprocessor

        preprocessor = DataPreprocessor(self.sample_data)
        preprocessor.handle_missing_values(strategy='median')

        self.assertEqual(preprocessor.data['qty'].isnull().sum(), 0)

    def test_feature_creation(self):
        """Test feature engineering."""
        from utils.data_preprocessing import DataPreprocessor

        preprocessor = DataPreprocessor(self.sample_data)
        preprocessor.create_features()

        self.assertIn('comp_price_diff', preprocessor.data.columns)

    def test_outlier_removal_iqr(self):
        """Test outlier removal with IQR method."""
        from utils.data_preprocessing import DataPreprocessor

        preprocessor = DataPreprocessor(self.sample_data)
        initial_count = len(preprocessor.data)
        preprocessor.remove_outliers(method='iqr')

        self.assertLessEqual(len(preprocessor.data), initial_count)

    def test_remove_duplicates(self):
        """Test duplicate removal."""
        from utils.data_preprocessing import DataPreprocessor

        data_with_duplicates = pd.concat([self.sample_data, self.sample_data])
        preprocessor = DataPreprocessor(data_with_duplicates)
        preprocessor.remove_duplicates()

        self.assertEqual(len(preprocessor.data), len(self.sample_data))

    def test_normalize_standard(self):
        """Test standard normalization."""
        from utils.data_preprocessing import DataPreprocessor

        preprocessor = DataPreprocessor(self.sample_data)
        preprocessor.normalize_data(method='standard')

        # Check that mean is approximately 0 for normalized columns
        numeric_cols = preprocessor.data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in preprocessor.scalers:
                self.assertAlmostEqual(preprocessor.data[col].mean(), 0, places=10)

    def test_preprocessing_report(self):
        """Test preprocessing report generation."""
        from utils.data_preprocessing import DataPreprocessor

        preprocessor = DataPreprocessor(self.sample_data)
        preprocessor.handle_missing_values().create_features()
        report = preprocessor.generate_preprocessing_report()

        self.assertIn('original_shape', report)
        self.assertIn('processed_shape', report)


class TestAnalyticsEngine(unittest.TestCase):
    """Tests for Analytics Engine."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_data = pd.DataFrame({
            'product_id': [f'p{i}' for i in range(100)],
            'product_category_name': np.random.choice(['cat1', 'cat2', 'cat3'], 100),
            'qty': np.random.randint(1, 50, 100),
            'total_price': np.random.uniform(50, 500, 100),
            'unit_price': np.random.uniform(10, 100, 100),
            'product_score': np.random.uniform(3.0, 5.0, 100),
            'customers': np.random.randint(5, 50, 100),
            'month_year': pd.date_range('2023-01-01', periods=100, freq='D').strftime('%d-%m-%Y')
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

        self.assertEqual(len(category_perf), 3)

    def test_executive_summary(self):
        """Test executive summary generation."""
        from utils.analytics import AnalyticsEngine

        engine = AnalyticsEngine(self.sample_data)
        summary = engine.generate_executive_summary()

        self.assertEqual(summary['total_records'], 100)
        self.assertEqual(summary['total_products'], 100)

    def test_correlation_analysis(self):
        """Test correlation analysis."""
        from utils.analytics import AnalyticsEngine

        engine = AnalyticsEngine(self.sample_data)
        correlations = engine.analyze_correlations()

        self.assertIn('correlation_matrix', correlations)
        self.assertIn('high_correlations', correlations)

    def test_price_distribution_analysis(self):
        """Test price distribution analysis."""
        from utils.analytics import AnalyticsEngine

        engine = AnalyticsEngine(self.sample_data)
        distribution = engine.analyze_price_distribution()

        self.assertIn('mean', distribution)
        self.assertIn('median', distribution)
        self.assertIn('skewness', distribution)

    def test_product_segmentation(self):
        """Test product segmentation."""
        from utils.analytics import AnalyticsEngine

        engine = AnalyticsEngine(self.sample_data)
        segments = engine.segment_products()

        self.assertIn('revenue_segment', segments.columns)
        self.assertIn('volume_segment', segments.columns)

    def test_customer_metrics(self):
        """Test customer metrics calculation."""
        from utils.analytics import AnalyticsEngine

        engine = AnalyticsEngine(self.sample_data)
        metrics = engine.calculate_customer_metrics()

        self.assertIn('total_customers', metrics)


class TestFeatureImportanceAnalyzer(unittest.TestCase):
    """Tests for Feature Importance Analysis."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
            'feature4': np.random.randn(100)
        })
        self.y = pd.Series(
            self.X['feature1'] * 2 + self.X['feature2'] + np.random.randn(100) * 0.1
        )

    def test_correlation_analysis(self):
        """Test correlation analysis."""
        from models.feature_importance import FeatureImportanceAnalyzer

        analyzer = FeatureImportanceAnalyzer()
        result = analyzer.analyze_correlation(self.X, self.y)

        self.assertEqual(len(result), 4)

    def test_mutual_information(self):
        """Test mutual information calculation."""
        from models.feature_importance import FeatureImportanceAnalyzer

        analyzer = FeatureImportanceAnalyzer()
        result = analyzer.analyze_mutual_information(self.X, self.y)

        self.assertIn('feature', result.columns)
        self.assertIn('mutual_info', result.columns)

    def test_top_features(self):
        """Test top features extraction."""
        from models.feature_importance import FeatureImportanceAnalyzer

        analyzer = FeatureImportanceAnalyzer()
        analyzer.analyze_correlation(self.X, self.y)
        top_features = analyzer.get_top_features(n_features=2)

        self.assertEqual(len(top_features), 2)

    def test_importance_plot(self):
        """Test importance plot generation."""
        from models.feature_importance import FeatureImportanceAnalyzer

        analyzer = FeatureImportanceAnalyzer()
        analyzer.analyze_correlation(self.X, self.y)
        fig = analyzer.create_importance_plot()

        self.assertIsNotNone(fig)


class TestChartBuilder(unittest.TestCase):
    """Tests for Chart Builder."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_data = pd.DataFrame({
            'category': ['A', 'B', 'C', 'A', 'B', 'C'] * 10,
            'value': np.random.randn(60),
            'price': np.random.uniform(10, 100, 60)
        })

    def test_price_distribution_chart(self):
        """Test price distribution chart."""
        from utils.visualizations import ChartBuilder

        builder = ChartBuilder()
        fig = builder.create_price_distribution_chart(self.sample_data, 'price')

        self.assertIsNotNone(fig)

    def test_category_comparison_chart(self):
        """Test category comparison chart."""
        from utils.visualizations import ChartBuilder

        builder = ChartBuilder()
        fig = builder.create_category_comparison_chart(self.sample_data, 'category', 'price')

        self.assertIsNotNone(fig)

    def test_scatter_plot(self):
        """Test scatter plot."""
        from utils.visualizations import ChartBuilder

        builder = ChartBuilder()
        fig = builder.create_scatter_plot(self.sample_data, 'value', 'price')

        self.assertIsNotNone(fig)

    def test_correlation_heatmap(self):
        """Test correlation heatmap."""
        from utils.visualizations import ChartBuilder

        builder = ChartBuilder()
        fig = builder.create_correlation_heatmap(self.sample_data)

        self.assertIsNotNone(fig)

    def test_pie_chart(self):
        """Test pie chart."""
        from utils.visualizations import ChartBuilder

        builder = ChartBuilder()
        agg_data = self.sample_data.groupby('category')['price'].sum().reset_index()
        fig = builder.create_pie_chart(agg_data, 'category', 'price')

        self.assertIsNotNone(fig)


class TestTimeSeriesAnalyzer(unittest.TestCase):
    """Tests for Time Series Analysis."""

    def setUp(self):
        """Set up test fixtures."""
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        self.sample_data = pd.DataFrame({
            'date': dates.strftime('%d-%m-%Y'),
            'value': np.random.uniform(50, 150, 50)
        })

    def test_moving_average(self):
        """Test moving average calculation."""
        from utils.time_series import TimeSeriesAnalyzer

        analyzer = TimeSeriesAnalyzer(self.sample_data, 'date', 'value')
        ma = analyzer.calculate_moving_average(window=3)

        self.assertEqual(len(ma), 50)

    def test_exponential_smoothing(self):
        """Test exponential smoothing."""
        from utils.time_series import TimeSeriesAnalyzer

        analyzer = TimeSeriesAnalyzer(self.sample_data, 'date', 'value')
        es = analyzer.calculate_exponential_smoothing(alpha=0.3)

        self.assertEqual(len(es), 50)

    def test_forecast(self):
        """Test forecast generation."""
        from utils.time_series import TimeSeriesAnalyzer

        analyzer = TimeSeriesAnalyzer(self.sample_data, 'date', 'value')
        forecast = analyzer.forecast_arima_simple(periods=6)

        self.assertIn('forecast', forecast)
        self.assertEqual(len(forecast['forecast']), 6)

    def test_growth_metrics(self):
        """Test growth metrics calculation."""
        from utils.time_series import TimeSeriesAnalyzer

        analyzer = TimeSeriesAnalyzer(self.sample_data, 'date', 'value')
        metrics = analyzer.calculate_growth_metrics()

        self.assertIn('avg_monthly_growth', metrics)

    def test_anomaly_detection(self):
        """Test anomaly detection."""
        from utils.time_series import TimeSeriesAnalyzer

        analyzer = TimeSeriesAnalyzer(self.sample_data, 'date', 'value')
        anomalies = analyzer.detect_anomalies()

        self.assertIn('anomalies', anomalies)


class TestRecommendationEngine(unittest.TestCase):
    """Tests for Recommendation Engine."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_data = pd.DataFrame({
            'product_id': [f'p{i}' for i in range(50)],
            'product_category_name': np.random.choice(['cat1', 'cat2', 'cat3'], 50),
            'qty': np.random.randint(1, 50, 50),
            'total_price': np.random.uniform(50, 500, 50),
            'unit_price': np.random.uniform(10, 100, 50),
            'product_score': np.random.uniform(2.5, 5.0, 50),
            'freight_price': np.random.uniform(5, 20, 50),
            'comp_1': np.random.uniform(8, 95, 50)
        })

    def test_analyze_and_recommend(self):
        """Test recommendation generation."""
        from utils.recommendations import RecommendationEngine

        engine = RecommendationEngine(self.sample_data)
        recommendations = engine.analyze_and_recommend()

        self.assertIsInstance(recommendations, list)

    def test_action_plan_generation(self):
        """Test action plan generation."""
        from utils.recommendations import RecommendationEngine

        engine = RecommendationEngine(self.sample_data)
        engine.analyze_and_recommend()
        plan = engine.generate_action_plan()

        self.assertIn('immediate_actions', plan)
        self.assertIn('short_term_actions', plan)
        self.assertIn('long_term_actions', plan)


class TestDatabaseConnector(unittest.TestCase):
    """Tests for Database Connector."""

    def test_database_initialization(self):
        """Test database initialization."""
        from utils.database import DatabaseConnector

        db = DatabaseConnector(':memory:')
        
        self.assertIsNotNone(db)

    def test_insert_product(self):
        """Test product insertion."""
        from utils.database import DatabaseConnector

        db = DatabaseConnector(':memory:')
        result = db.insert_product({
            'product_id': 'test1',
            'product_category_name': 'cat1',
            'unit_price': 50.0,
            'product_score': 4.5
        })

        self.assertTrue(result)

    def test_get_product(self):
        """Test product retrieval."""
        from utils.database import DatabaseConnector

        db = DatabaseConnector(':memory:')
        db.insert_product({
            'product_id': 'test1',
            'product_category_name': 'cat1',
            'unit_price': 50.0,
            'product_score': 4.5
        })

        product = db.get_product('test1')

        self.assertIsNotNone(product)
        self.assertEqual(product['product_id'], 'test1')

    def test_save_prediction(self):
        """Test prediction saving."""
        from utils.database import DatabaseConnector

        db = DatabaseConnector(':memory:')
        db.insert_product({
            'product_id': 'test1',
            'product_category_name': 'cat1',
            'unit_price': 50.0,
            'product_score': 4.5
        })
        
        result = db.save_prediction('test1', 100.0, 'rf', 0.85)

        self.assertTrue(result)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestPricePredictor))
    suite.addTests(loader.loadTestsFromTestCase(TestModelEvaluator))
    suite.addTests(loader.loadTestsFromTestCase(TestPriceOptimizer))
    suite.addTests(loader.loadTestsFromTestCase(TestDataPreprocessor))
    suite.addTests(loader.loadTestsFromTestCase(TestAnalyticsEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureImportanceAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestChartBuilder))
    suite.addTests(loader.loadTestsFromTestCase(TestTimeSeriesAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestRecommendationEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestDatabaseConnector))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
