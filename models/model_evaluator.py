"""
Model Evaluation Module.

This module provides comprehensive evaluation tools for assessing
machine learning model performance in retail price prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, median_absolute_error,
    mean_squared_log_error
)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation class.

    Provides various evaluation metrics, visualizations, and diagnostic tools
    for assessing model performance.
    """

    def __init__(self):
        self.evaluation_results = {}
        self.comparison_data = None

    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray,
                       model_name: str = 'Model') -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.

        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of the model being evaluated

        Returns:
            Dictionary of evaluation metrics
        """
        # Basic metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Additional metrics
        explained_var = explained_variance_score(y_true, y_pred)
        median_ae = median_absolute_error(y_true, y_pred)

        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-10))) * 100

        # Symmetric Mean Absolute Percentage Error
        smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100

        # Root Mean Squared Log Error (for positive values)
        if np.all(y_true > 0) and np.all(y_pred > 0):
            rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))
        else:
            rmsle = np.nan

        metrics = {
            'model_name': model_name,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'explained_variance': explained_var,
            'median_ae': median_ae,
            'mape': mape,
            'smape': smape,
            'rmsle': rmsle
        }

        self.evaluation_results[model_name] = metrics
        logger.info(f"Model '{model_name}' evaluation: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

        return metrics

    def compare_models(self, metrics_dict: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare multiple models based on their metrics.

        Args:
            metrics_dict: Dictionary of model metrics

        Returns:
            DataFrame with model comparison
        """
        comparison_df = pd.DataFrame(metrics_dict).T
        comparison_df = comparison_df.sort_values('r2', ascending=False)
        self.comparison_data = comparison_df

        return comparison_df

    def residual_analysis(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Perform residual analysis.

        Args:
            y_true: Actual values
            y_pred: Predicted values

        Returns:
            Dictionary with residual statistics
        """
        residuals = y_true - y_pred

        analysis = {
            'mean_residual': np.mean(residuals),
            'std_residual': np.std(residuals),
            'min_residual': np.min(residuals),
            'max_residual': np.max(residuals),
            'residual_skewness': pd.Series(residuals).skew(),
            'residual_kurtosis': pd.Series(residuals).kurtosis(),
            'residuals': residuals
        }

        return analysis

    def prediction_error_distribution(self, y_true: np.ndarray,
                                       y_pred: np.ndarray,
                                       bins: int = 20) -> Dict:
        """
        Analyze the distribution of prediction errors.

        Args:
            y_true: Actual values
            y_pred: Predicted values
            bins: Number of bins for histogram

        Returns:
            Dictionary with error distribution data
        """
        errors = y_pred - y_true
        percentage_errors = (errors / y_true) * 100

        distribution = {
            'errors': errors,
            'percentage_errors': percentage_errors,
            'mean_error': np.mean(errors),
            'mean_percentage_error': np.mean(percentage_errors),
            'error_std': np.std(errors),
            'within_5_percent': np.mean(np.abs(percentage_errors) <= 5) * 100,
            'within_10_percent': np.mean(np.abs(percentage_errors) <= 10) * 100,
            'within_20_percent': np.mean(np.abs(percentage_errors) <= 20) * 100
        }

        return distribution

    def create_evaluation_plot(self, y_true: np.ndarray, y_pred: np.ndarray,
                                model_name: str = 'Model') -> go.Figure:
        """
        Create a comprehensive evaluation visualization.

        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of the model

        Returns:
            Plotly Figure object
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'Predicted vs Actual',
                'Residual Distribution',
                'Error vs Predicted Value',
                'Cumulative Error Distribution'
            )
        )

        # Predicted vs Actual
        fig.add_trace(
            go.Scatter(
                x=y_true, y=y_pred,
                mode='markers',
                marker=dict(color='#667eea', size=6, opacity=0.6),
                name='Predictions'
            ),
            row=1, col=1
        )

        # Perfect prediction line
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines',
                line=dict(color='#f5576c', dash='dash'),
                name='Perfect Prediction'
            ),
            row=1, col=1
        )

        # Residual distribution
        residuals = y_true - y_pred
        fig.add_trace(
            go.Histogram(
                x=residuals,
                nbinsx=30,
                marker_color='#764ba2',
                name='Residuals'
            ),
            row=1, col=2
        )

        # Error vs Predicted Value
        fig.add_trace(
            go.Scatter(
                x=y_pred, y=residuals,
                mode='markers',
                marker=dict(color='#f093fb', size=6, opacity=0.6),
                name='Residuals vs Predicted'
            ),
            row=2, col=1
        )

        # Zero line
        fig.add_trace(
            go.Scatter(
                x=[y_pred.min(), y_pred.max()], y=[0, 0],
                mode='lines',
                line=dict(color='#667eea', dash='dash'),
                showlegend=False
            ),
            row=2, col=1
        )

        # Cumulative error distribution
        sorted_errors = np.sort(np.abs(residuals))
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100

        fig.add_trace(
            go.Scatter(
                x=sorted_errors, y=cumulative,
                mode='lines',
                line=dict(color='#4CAF50'),
                name='Cumulative Distribution'
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_xaxes(title_text='Actual Value', row=1, col=1)
        fig.update_yaxes(title_text='Predicted Value', row=1, col=1)
        fig.update_xaxes(title_text='Residual', row=1, col=2)
        fig.update_yaxes(title_text='Count', row=1, col=2)
        fig.update_xaxes(title_text='Predicted Value', row=2, col=1)
        fig.update_yaxes(title_text='Residual', row=2, col=1)
        fig.update_xaxes(title_text='Absolute Error', row=2, col=2)
        fig.update_yaxes(title_text='Cumulative %', row=2, col=2)

        fig.update_layout(
            title=f'Model Evaluation: {model_name}',
            height=700,
            showlegend=True,
            template='plotly_white'
        )

        return fig

    def generate_evaluation_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    model_name: str = 'Model') -> Dict:
        """
        Generate a comprehensive evaluation report.

        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of the model

        Returns:
            Dictionary with complete evaluation report
        """
        metrics = self.evaluate_model(y_true, y_pred, model_name)
        residuals = self.residual_analysis(y_true, y_pred)
        error_dist = self.prediction_error_distribution(y_true, y_pred)

        report = {
            'model_name': model_name,
            'metrics': metrics,
            'residual_analysis': {k: float(v) if isinstance(v, (np.floating, float)) else v
                                  for k, v in residuals.items() if k != 'residuals'},
            'error_distribution': {k: float(v) if isinstance(v, (np.floating, float)) else v
                                  for k, v in error_dist.items() if k not in ['errors', 'percentage_errors']},
            'summary': {
                'total_samples': len(y_true),
                'prediction_accuracy': f"{metrics['r2']*100:.2f}%",
                'average_error': f"${abs(error_dist['mean_error']):.2f}",
                'error_range': f"${error_dist['errors'].min():.2f} to ${error_dist['errors'].max():.2f}"
            }
        }

        return report

    def benchmark_models(self, predictions_dict: Dict[str, np.ndarray],
                         y_true: np.ndarray) -> pd.DataFrame:
        """
        Benchmark multiple models against each other.

        Args:
            predictions_dict: Dictionary mapping model names to their predictions
            y_true: Actual values

        Returns:
            DataFrame with benchmark results for all models
        """
        results = {}
        for model_name, y_pred in predictions_dict.items():
            metrics = self.evaluate_model(y_true, y_pred, model_name)
            results[model_name] = metrics

        benchmark_df = pd.DataFrame(results).T
        benchmark_df = benchmark_df.sort_values('r2', ascending=False)

        self.comparison_data = benchmark_df
        return benchmark_df

    def statistical_significance_test(self, y_true: np.ndarray,
                                      y_pred1: np.ndarray,
                                      y_pred2: np.ndarray,
                                      alpha: float = 0.05) -> Dict:
        """
        Perform statistical significance test between two models.

        Tests whether the difference in prediction errors between two models
        is statistically significant using a paired t-test.

        Args:
            y_true: Actual values
            y_pred1: Predictions from model 1
            y_pred2: Predictions from model 2
            alpha: Significance level

        Returns:
            Dictionary with test results
        """
        from scipy import stats

        # Calculate absolute errors for each model
        errors1 = np.abs(y_true - y_pred1)
        errors2 = np.abs(y_true - y_pred2)

        # Paired t-test on absolute errors
        t_statistic, p_value = stats.ttest_rel(errors1, errors2)

        # Effect size (Cohen's d)
        diff = errors1 - errors2
        cohens_d = np.mean(diff) / np.std(diff, ddof=1)

        # Determine significance
        is_significant = p_value < alpha

        # Which model is better?
        mean_error1 = np.mean(errors1)
        mean_error2 = np.mean(errors2)
        better_model = 'Model 1' if mean_error1 < mean_error2 else 'Model 2'

        result = {
            't_statistic': t_statistic,
            'p_value': p_value,
            'is_significant': is_significant,
            'alpha': alpha,
            'cohens_d': cohens_d,
            'better_model': better_model,
            'mean_error_model1': mean_error1,
            'mean_error_model2': mean_error2,
            'error_improvement': abs(mean_error1 - mean_error2),
            'interpretation': (
                f"The difference is statistically significant (p={p_value:.4f})"
                if is_significant else
                f"The difference is not statistically significant (p={p_value:.4f})"
            )
        }

        return result
