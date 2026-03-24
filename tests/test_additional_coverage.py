"""
Additional Tests to Improve Coverage.

This module provides additional tests for modules with low coverage.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestLogger(unittest.TestCase):
    """Tests for Logger module."""

    def test_setup_logger(self):
        """Test logger setup."""
        from utils.logger import setup_logger

        logger = setup_logger('test_logger')
        self.assertIsNotNone(logger)

    def test_logger_with_file(self):
        """Test logger with file output."""
        from utils.logger import setup_logger

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, 'test.log')
            logger = setup_logger('file_test', log_file=log_file)
            logger.info("Test message")

            self.assertTrue(os.path.exists(log_file))

    def test_performance_logger(self):
        """Test performance logger."""
        from utils.logger import PerformanceLogger

        perf = PerformanceLogger('test_perf')
        import time

        # Use the measure method correctly as context manager
        with perf.measure('operation'):
            time.sleep(0.01)

        metrics = perf.get_metrics()
        self.assertIn('operation', metrics)
        self.assertGreater(metrics['operation'], 0)

    def test_performance_logger_metrics(self):
        """Test performance logger metrics tracking."""
        from utils.logger import PerformanceLogger

        perf = PerformanceLogger('test_metrics')
        perf.metrics['test_op'] = 0.5

        metrics = perf.get_metrics()
        self.assertIn('test_op', metrics)

        perf.reset_metrics()
        self.assertEqual(len(perf.get_metrics()), 0)

    def test_get_logger(self):
        """Test get_logger function."""
        from utils.logger import get_logger

        logger = get_logger('module_test')
        self.assertIsNotNone(logger)


class TestDashboardUtils(unittest.TestCase):
    """Tests for Dashboard Utilities."""

    def test_create_metric_card(self):
        """Test metric card creation."""
        from utils.dashboard_utils import DashboardComponents

        card = DashboardComponents.create_metric_card('Revenue', 1000.50)
        self.assertEqual(card['title'], 'Revenue')
        self.assertIn('1000', card['value'])

    def test_create_kpi_row(self):
        """Test KPI row creation."""
        from utils.dashboard_utils import DashboardComponents

        kpis = DashboardComponents.create_kpi_row({'Sales': 100, 'Profit': 50})
        self.assertEqual(len(kpis), 2)

    def test_data_quality_completeness(self):
        """Test data quality completeness check."""
        from utils.dashboard_utils import DataQualityChecker

        data = pd.DataFrame({
            'a': [1, 2, None, 4],
            'b': [1, None, None, 4]
        })
        checker = DataQualityChecker(data)
        result = checker.check_completeness()

        self.assertLess(result['completeness_rate'], 1.0)
        self.assertIn('a', result['columns_with_missing'])

    def test_data_quality_uniqueness(self):
        """Test data quality uniqueness check."""
        from utils.dashboard_utils import DataQualityChecker

        data = pd.DataFrame({
            'a': [1, 1, 2, 2, 3]
        })
        checker = DataQualityChecker(data)
        result = checker.check_uniqueness()

        self.assertLess(result['uniqueness_rate'], 1.0)

    def test_data_quality_report(self):
        """Test full quality report."""
        from utils.dashboard_utils import DataQualityChecker

        data = pd.DataFrame({
            'unit_price': [10, 20, 30],
            'qty': [1, 2, 3]
        })
        checker = DataQualityChecker(data)
        report = checker.generate_quality_report()

        self.assertIn('completeness', report)
        self.assertIn('overall_score', report)

    def test_state_manager(self):
        """Test state manager."""
        from utils.dashboard_utils import StateManager

        sm = StateManager()
        sm.set('key1', 'value1')
        result = sm.get('key1')

        self.assertEqual(result, 'value1')
        self.assertIsNone(sm.get('nonexistent'))

    def test_filter_manager(self):
        """Test filter manager."""
        from utils.dashboard_utils import FilterManager

        data = pd.DataFrame({
            'price': [10, 20, 30, 40],
            'category': ['A', 'B', 'A', 'B']
        })

        fm = FilterManager(data)
        fm.add_filter('price', 'gt', 20)
        result = fm.apply_filters()

        self.assertEqual(len(result), 2)

    def test_export_manager_csv(self):
        """Test export to CSV."""
        from utils.dashboard_utils import ExportManager

        data = pd.DataFrame({'a': [1, 2, 3]})
        csv = ExportManager.to_csv(data)

        self.assertIn('a', csv)

    def test_export_manager_json(self):
        """Test export to JSON."""
        from utils.dashboard_utils import ExportManager

        data = pd.DataFrame({'a': [1, 2, 3]})
        json_str = ExportManager.to_json(data)

        self.assertIn('a', json_str)


class TestAdvancedReporting(unittest.TestCase):
    """Tests for Advanced Reporting."""

    def setUp(self):
        self.data = pd.DataFrame({
            'unit_price': [10, 20, 30, 40],
            'qty': [1, 2, 3, 4],
            'total_price': [10, 40, 90, 160]
        })

    def test_generate_full_report(self):
        """Test full report generation."""
        from utils.advanced_reporting import AdvancedReporter

        reporter = AdvancedReporter(self.data)
        report = reporter.generate_full_report()

        self.assertIn('metadata', report)
        self.assertIn('summary', report)

    def test_export_report_json(self):
        """Test report export to JSON."""
        from utils.advanced_reporting import AdvancedReporter

        reporter = AdvancedReporter(self.data)
        json_output = reporter.export_report(format='json')

        self.assertIn('metadata', json_output)


class TestDatabaseAdditional(unittest.TestCase):
    """Additional tests for Database Connector."""

    def test_get_all_products(self):
        """Test getting all products."""
        from utils.database import DatabaseConnector

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            db = DatabaseConnector(db_path)

            db.insert_product({'product_id': 'p1', 'unit_price': 10.0})
            db.insert_product({'product_id': 'p2', 'unit_price': 20.0})

            products = db.get_all_products()
            self.assertEqual(len(products), 2)

    def test_record_price_change(self):
        """Test recording price change."""
        from utils.database import DatabaseConnector

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            db = DatabaseConnector(db_path)

            result = db.record_price_change('p1', 10.0, 12.0, 'test')
            self.assertTrue(result)

    def test_get_database_stats(self):
        """Test database stats."""
        from utils.database import DatabaseConnector

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            db = DatabaseConnector(db_path)

            stats = db.get_database_stats()
            self.assertIn('total_products', stats)


if __name__ == '__main__':
    unittest.main()
