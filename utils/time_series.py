"""
Time Series Analysis Module.

This module provides time series analysis and forecasting capabilities
for retail price data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class TimeSeriesAnalyzer:
    """
    Time series analysis and forecasting class.

    Provides methods for trend analysis, seasonality detection,
    and time series forecasting.
    """

    def __init__(self, data: pd.DataFrame, date_column: str, value_column: str):
        """
        Initialize TimeSeriesAnalyzer.

        Args:
            data: DataFrame with time series data
            date_column: Name of date column
            value_column: Name of value column
        """
        self.data = data.copy()
        self.date_column = date_column
        self.value_column = value_column
        self.time_series = None

        self._prepare_time_series()

    def _prepare_time_series(self):
        """Prepare time series data."""
        try:
            self.data['date'] = pd.to_datetime(self.data[self.date_column], format='%d-%m-%Y')
            self.data = self.data.sort_values('date')
            self.time_series = self.data.groupby(self.data['date'].dt.to_period('M'))[
                self.value_column
            ].sum()
            self.time_series.index = self.time_series.index.to_timestamp()
        except Exception as e:
            logger.error(f"Error preparing time series: {str(e)}")

    def calculate_moving_average(self, window: int = 3) -> pd.Series:
        """
        Calculate moving average.

        Args:
            window: Window size for moving average

        Returns:
            Series with moving average values
        """
        if self.time_series is None:
            return pd.Series()

        return self.time_series.rolling(window=window).mean()

    def calculate_exponential_smoothing(self, alpha: float = 0.3) -> pd.Series:
        """
        Calculate exponential smoothing.

        Args:
            alpha: Smoothing factor (0-1)

        Returns:
            Series with smoothed values
        """
        if self.time_series is None:
            return pd.Series()

        return self.time_series.ewm(alpha=alpha).mean()

    def decompose_trend(self) -> Dict:
        """
        Decompose time series into trend, seasonal, and residual components.

        Returns:
            Dictionary with decomposition results
        """
        if self.time_series is None or len(self.time_series) < 6:
            return {'error': 'Insufficient data for decomposition'}

        values = self.time_series.values
        n = len(values)

        # Simple trend using moving average
        trend = pd.Series(values).rolling(window=min(3, n // 2), center=True).mean().values

        # Detrended series
        detrended = values - trend

        # Seasonal component (simplified)
        seasonal = np.zeros(n)
        for i in range(n):
            seasonal[i] = np.nanmean(detrended[i::12]) if n > 12 else detrended[i]

        # Residual
        residual = values - trend - seasonal

        return {
            'original': values,
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual
        }

    def detect_seasonality(self) -> Dict:
        """
        Detect seasonality patterns.

        Returns:
            Dictionary with seasonality analysis
        """
        if self.time_series is None or len(self.time_series) < 12:
            return {'has_seasonality': False, 'reason': 'Insufficient data'}

        values = self.time_series.values

        # Autocorrelation for lag 12 (yearly seasonality)
        if len(values) > 12:
            autocorr = np.corrcoef(values[:-12], values[12:])[0, 1]
        else:
            autocorr = 0

        has_seasonality = abs(autocorr) > 0.3

        return {
            'has_seasonality': has_seasonality,
            'autocorrelation': autocorr,
            'season_type': 'yearly' if has_seasonality else 'none'
        }

    def forecast_arima_simple(self, periods: int = 6) -> Dict:
        """
        Simple ARIMA-like forecast.

        Args:
            periods: Number of periods to forecast

        Returns:
            Dictionary with forecast results
        """
        if self.time_series is None:
            return {'error': 'No time series data'}

        values = self.time_series.values
        n = len(values)

        # Fit linear trend
        x = np.arange(n)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)

        # Generate forecast
        forecast_x = np.arange(n, n + periods)
        forecast_values = slope * forecast_x + intercept

        # Add seasonal adjustment
        seasonal_result = self.detect_seasonality()
        if seasonal_result['has_seasonality']:
            # Add simple seasonal pattern
            seasonal_factor = np.sin(np.arange(periods) * 2 * np.pi / 12) * 0.1
            forecast_values = forecast_values * (1 + seasonal_factor)

        # Calculate confidence intervals
        residuals = values - (slope * x + intercept)
        std_residual = np.std(residuals)
        z_score = 1.96  # 95% confidence

        lower_bound = forecast_values - z_score * std_residual
        upper_bound = forecast_values + z_score * std_residual

        return {
            'forecast': forecast_values,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'trend_slope': slope,
            'r_squared': r_value ** 2
        }

    def calculate_growth_metrics(self) -> Dict:
        """
        Calculate growth metrics.

        Returns:
            Dictionary with growth metrics
        """
        if self.time_series is None or len(self.time_series) < 2:
            return {}

        values = self.time_series.values

        # Period over period growth
        pop_growth = (values[1:] - values[:-1]) / values[:-1] * 100

        # Average growth rate
        avg_growth = np.mean(pop_growth)

        # Compound monthly growth rate
        cmgr = ((values[-1] / values[0]) ** (1 / (len(values) - 1)) - 1) * 100

        # Volatility (standard deviation of growth rates)
        volatility = np.std(pop_growth)

        return {
            'avg_monthly_growth': avg_growth,
            'cmgr': cmgr,
            'volatility': volatility,
            'growth_rates': pop_growth
        }

    def detect_anomalies(self, threshold: float = 2.0) -> Dict:
        """
        Detect anomalies in time series.

        Args:
            threshold: Z-score threshold for anomaly detection

        Returns:
            Dictionary with anomaly detection results
        """
        if self.time_series is None:
            return {'anomalies': []}

        values = self.time_series.values
        mean = np.mean(values)
        std = np.std(values)

        z_scores = np.abs((values - mean) / std)
        anomaly_indices = np.where(z_scores > threshold)[0]

        anomalies = []
        for idx in anomaly_indices:
            anomalies.append({
                'index': int(idx),
                'date': str(self.time_series.index[idx]),
                'value': float(values[idx]),
                'z_score': float(z_scores[idx])
            })

        return {
            'anomalies': anomalies,
            'threshold': threshold,
            'mean': mean,
            'std': std
        }

    def generate_forecast_report(self, periods: int = 6) -> Dict:
        """
        Generate comprehensive forecast report.

        Args:
            periods: Number of periods to forecast

        Returns:
            Dictionary with forecast report
        """
        forecast = self.forecast_arima_simple(periods)
        growth = self.calculate_growth_metrics()
        seasonality = self.detect_seasonality()
        anomalies = self.detect_anomalies()

        return {
            'forecast': forecast,
            'growth_metrics': growth,
            'seasonality': seasonality,
            'anomalies': anomalies,
            'summary': {
                'historical_periods': len(self.time_series) if self.time_series is not None else 0,
                'forecast_periods': periods,
                'trend_direction': 'increasing' if forecast.get('trend_slope', 0) > 0 else 'decreasing',
                'has_seasonality': seasonality.get('has_seasonality', False),
                'anomaly_count': len(anomalies.get('anomalies', []))
            }
        }
