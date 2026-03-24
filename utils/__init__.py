"""
Utility modules for the Retail Price Optimization Dashboard.

This package provides comprehensive utilities for:
- Price optimization strategies
- Data preprocessing and cleaning
- Analytics and statistical analysis
- Visualization and charting
- Time series forecasting
- Recommendation engine
- Advanced reporting
- Database connectivity
- Dashboard components
"""

from .price_optimizer import PriceOptimizer, PricingStrategy
from .data_preprocessing import DataPreprocessor
from .analytics import AnalyticsEngine
from .visualizations import ChartBuilder
from .logger import setup_logger
from .time_series import TimeSeriesAnalyzer
from .recommendations import RecommendationEngine, Recommendation
from .advanced_reporting import AdvancedReporter
from .database import DatabaseConnector
from .dashboard_utils import (
    DashboardComponents,
    DataQualityChecker,
    StateManager,
    FilterManager,
    ExportManager
)

__all__ = [
    # Core utilities
    'PriceOptimizer',
    'PricingStrategy',
    'DataPreprocessor',
    'AnalyticsEngine',
    'ChartBuilder',
    'setup_logger',

    # Advanced utilities
    'TimeSeriesAnalyzer',
    'RecommendationEngine',
    'Recommendation',
    'AdvancedReporter',
    'DatabaseConnector',

    # Dashboard utilities
    'DashboardComponents',
    'DataQualityChecker',
    'StateManager',
    'FilterManager',
    'ExportManager'
]
