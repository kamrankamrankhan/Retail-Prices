"""
Analytics Engine Module.

This module provides comprehensive analytics and statistical analysis
functionality for retail price data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class StatisticalResult:
    """Container for statistical test results."""
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    interpretation: str


class AnalyticsEngine:
    """
    Comprehensive analytics engine for retail price analysis.

    Provides statistical analysis, trend detection, market analysis,
    and business intelligence capabilities.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the AnalyticsEngine.

        Args:
            data: DataFrame with retail price data
        """
        self.data = data
        self.analysis_results = {}

    def calculate_descriptive_statistics(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate comprehensive descriptive statistics.

        Args:
            columns: Columns to analyze, None for all numeric

        Returns:
            DataFrame with descriptive statistics
        """
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()

        stats_df = self.data[columns].describe().T

        # Add additional statistics
        stats_df['variance'] = self.data[columns].var()
        stats_df['skewness'] = self.data[columns].skew()
        stats_df['kurtosis'] = self.data[columns].kurtosis()
        stats_df['median'] = self.data[columns].median()
        stats_df['mad'] = self.data[columns].mad()  # Median Absolute Deviation
        stats_df['cv'] = stats_df['std'] / stats_df['mean'] * 100  # Coefficient of Variation

        return stats_df

    def analyze_correlations(self, method: str = 'pearson',
                             threshold: float = 0.5) -> Dict:
        """
        Analyze correlations between variables.

        Args:
            method: Correlation method ('pearson', 'spearman', 'kendall')
            threshold: Minimum correlation to include in results

        Returns:
            Dictionary with correlation analysis
        """
        numeric_data = self.data.select_dtypes(include=[np.number])
        corr_matrix = numeric_data.corr(method=method)

        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) >= threshold:
                    high_corr_pairs.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })

        return {
            'correlation_matrix': corr_matrix,
            'high_correlations': high_corr_pairs,
            'method': method
        }

    def perform_hypothesis_test(self, group1: pd.Series, group2: pd.Series,
                                 test_type: str = 'ttest',
                                 alpha: float = 0.05) -> StatisticalResult:
        """
        Perform statistical hypothesis testing.

        Args:
            group1: First group of data
            group2: Second group of data
            test_type: Type of test ('ttest', 'mannwhitney', 'ks')
            alpha: Significance level

        Returns:
            StatisticalResult with test results
        """
        if test_type == 'ttest':
            statistic, p_value = stats.ttest_ind(group1.dropna(), group2.dropna())
            test_name = "Independent Samples t-test"
        elif test_type == 'mannwhitney':
            statistic, p_value = stats.mannwhitneyu(group1.dropna(), group2.dropna())
            test_name = "Mann-Whitney U test"
        elif test_type == 'ks':
            statistic, p_value = stats.ks_2samp(group1.dropna(), group2.dropna())
            test_name = "Kolmogorov-Smirnov test"
        else:
            raise ValueError(f"Unknown test type: {test_type}")

        is_significant = p_value < alpha

        if is_significant:
            interpretation = f"The difference is statistically significant (p={p_value:.4f} < {alpha})"
        else:
            interpretation = f"The difference is not statistically significant (p={p_value:.4f} >= {alpha})"

        return StatisticalResult(
            test_name=test_name,
            statistic=statistic,
            p_value=p_value,
            is_significant=is_significant,
            interpretation=interpretation
        )

    def analyze_price_distribution(self, price_column: str = 'total_price') -> Dict:
        """
        Analyze the distribution of prices.

        Args:
            price_column: Name of the price column

        Returns:
            Dictionary with distribution analysis
        """
        prices = self.data[price_column].dropna()

        # Basic statistics
        analysis = {
            'mean': prices.mean(),
            'median': prices.median(),
            'std': prices.std(),
            'min': prices.min(),
            'max': prices.max(),
            'range': prices.max() - prices.min(),
            'iqr': prices.quantile(0.75) - prices.quantile(0.25),
            'skewness': prices.skew(),
            'kurtosis': prices.kurtosis()
        }

        # Normality tests
        if len(prices) >= 20:
            shapiro_stat, shapiro_p = stats.shapiro(prices.sample(min(5000, len(prices))))
            analysis['shapiro_wilk'] = {
                'statistic': shapiro_stat,
                'p_value': shapiro_p,
                'is_normal': shapiro_p > 0.05
            }

        # Percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        analysis['percentiles'] = {
            f'p{p}': prices.quantile(p / 100) for p in percentiles
        }

        return analysis

    def analyze_sales_trends(self, date_column: str = 'month_year',
                             value_column: str = 'total_price') -> Dict:
        """
        Analyze sales trends over time.

        Args:
            date_column: Name of date column
            value_column: Name of value column

        Returns:
            Dictionary with trend analysis
        """
        # Prepare time series data
        df = self.data.copy()

        if date_column in df.columns:
            try:
                df['date'] = pd.to_datetime(df[date_column], format='%d-%m-%Y')
                df = df.sort_values('date')

                # Monthly aggregation
                monthly = df.groupby(df['date'].dt.to_period('M'))[value_column].sum()

                # Calculate trend
                x = np.arange(len(monthly))
                y = monthly.values

                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

                trend_analysis = {
                    'trend_direction': 'increasing' if slope > 0 else 'decreasing',
                    'trend_strength': abs(r_value),
                    'slope': slope,
                    'r_squared': r_value ** 2,
                    'p_value': p_value,
                    'is_significant': p_value < 0.05,
                    'monthly_data': monthly.to_dict()
                }

                # Calculate growth rate
                if len(monthly) >= 2:
                    first_half = monthly[:len(monthly) // 2].mean()
                    second_half = monthly[len(monthly) // 2:].mean()
                    growth_rate = (second_half - first_half) / first_half * 100
                    trend_analysis['growth_rate_percent'] = growth_rate

                return trend_analysis

            except Exception as e:
                logger.error(f"Error analyzing trends: {str(e)}")
                return {'error': str(e)}

        return {'error': 'Date column not found'}

    def analyze_category_performance(self) -> pd.DataFrame:
        """
        Analyze performance by product category.

        Returns:
            DataFrame with category performance metrics
        """
        category_analysis = self.data.groupby('product_category_name').agg({
            'total_price': ['sum', 'mean', 'count', 'std'],
            'unit_price': ['mean', 'min', 'max'],
            'qty': ['sum', 'mean'],
            'product_score': 'mean'
        }).round(2)

        category_analysis.columns = ['_'.join(col).strip() for col in category_analysis.columns.values]
        category_analysis = category_analysis.reset_index()

        # Calculate market share
        total_revenue = category_analysis['total_price_sum'].sum()
        category_analysis['market_share'] = (
            category_analysis['total_price_sum'] / total_revenue * 100
        ).round(2)

        # Calculate revenue concentration
        category_analysis = category_analysis.sort_values('total_price_sum', ascending=False)
        category_analysis['cumulative_share'] = category_analysis['market_share'].cumsum()

        return category_analysis

    def analyze_competitive_position(self) -> pd.DataFrame:
        """
        Analyze competitive position against competitors.

        Returns:
            DataFrame with competitive analysis
        """
        df = self.data.copy()

        # Calculate price differences
        df['price_vs_comp1'] = df['unit_price'] - df['comp_1']
        df['price_vs_comp2'] = df['unit_price'] - df['comp_2']
        df['price_vs_comp3'] = df['unit_price'] - df['comp_3']

        competitive_analysis = df.groupby('product_category_name').agg({
            'price_vs_comp1': 'mean',
            'price_vs_comp2': 'mean',
            'price_vs_comp3': 'mean',
            'unit_price': 'mean',
            'comp_1': 'mean',
            'comp_2': 'mean',
            'comp_3': 'mean'
        }).round(2)

        # Determine position
        def get_position(row):
            avg_diff = (row['price_vs_comp1'] + row['price_vs_comp2'] + row['price_vs_comp3']) / 3
            if avg_diff > 5:
                return 'Premium'
            elif avg_diff < -5:
                return 'Budget'
            else:
                return 'Competitive'

        competitive_analysis['position'] = competitive_analysis.apply(get_position, axis=1)

        return competitive_analysis.reset_index()

    def calculate_customer_metrics(self) -> Dict:
        """
        Calculate customer-related metrics.

        Returns:
            Dictionary with customer metrics
        """
        metrics = {}

        if 'customers' in self.data.columns:
            metrics['total_customers'] = self.data['customers'].sum()
            metrics['avg_customers_per_transaction'] = self.data['customers'].mean()
            metrics['unique_customer_count'] = self.data['customers'].nunique()

        if 'qty' in self.data.columns and 'customers' in self.data.columns:
            metrics['avg_qty_per_customer'] = (
                self.data['qty'].sum() / self.data['customers'].sum()
            )

        if 'total_price' in self.data.columns and 'customers' in self.data.columns:
            metrics['avg_revenue_per_customer'] = (
                self.data['total_price'].sum() / self.data['customers'].sum()
            )

        return metrics

    def segment_products(self, n_segments: int = 4) -> pd.DataFrame:
        """
        Segment products based on performance metrics.

        Args:
            n_segments: Number of segments

        Returns:
            DataFrame with product segments
        """
        df = self.data.copy()

        # Calculate metrics per product
        product_metrics = df.groupby('product_id').agg({
            'total_price': 'sum',
            'qty': 'sum',
            'unit_price': 'mean',
            'product_score': 'mean'
        }).reset_index()

        # Create segments using quantiles
        product_metrics['revenue_segment'] = pd.qcut(
            product_metrics['total_price'],
            q=n_segments,
            labels=['Low', 'Medium', 'High', 'Top'][:n_segments]
        )

        product_metrics['volume_segment'] = pd.qcut(
            product_metrics['qty'],
            q=n_segments,
            labels=['Low', 'Medium', 'High', 'Top'][:n_segments]
        )

        return product_metrics

    def generate_executive_summary(self) -> Dict:
        """
        Generate executive summary of the data.

        Returns:
            Dictionary with executive summary
        """
        summary = {
            'total_records': len(self.data),
            'total_products': self.data['product_id'].nunique() if 'product_id' in self.data.columns else 0,
            'total_categories': self.data['product_category_name'].nunique() if 'product_category_name' in self.data.columns else 0,
            'date_range': {
                'start': self.data['month_year'].min() if 'month_year' in self.data.columns else None,
                'end': self.data['month_year'].max() if 'month_year' in self.data.columns else None
            }
        }

        if 'total_price' in self.data.columns:
            summary['total_revenue'] = self.data['total_price'].sum()
            summary['avg_transaction_value'] = self.data['total_price'].mean()

        if 'qty' in self.data.columns:
            summary['total_quantity_sold'] = self.data['qty'].sum()

        if 'product_score' in self.data.columns:
            summary['avg_product_score'] = self.data['product_score'].mean()

        # Top performer
        if 'product_category_name' in self.data.columns and 'total_price' in self.data.columns:
            top_category = self.data.groupby('product_category_name')['total_price'].sum().idxmax()
            summary['top_category'] = top_category

        return summary

    def create_analytics_dashboard_data(self) -> Dict:
        """
        Create all data needed for analytics dashboard.

        Returns:
            Dictionary with dashboard data
        """
        return {
            'executive_summary': self.generate_executive_summary(),
            'descriptive_stats': self.calculate_descriptive_statistics(),
            'category_performance': self.analyze_category_performance(),
            'competitive_position': self.analyze_competitive_position(),
            'price_distribution': self.analyze_price_distribution(),
            'customer_metrics': self.calculate_customer_metrics(),
            'correlations': self.analyze_correlations()
        }
