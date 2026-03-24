"""
Feature Importance Analysis Module.

This module provides tools for analyzing and visualizing feature importance
in machine learning models for retail price prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.feature_selection import mutual_info_regression
import plotly.graph_objects as go
import plotly.express as px
import logging

logger = logging.getLogger(__name__)


class FeatureImportanceAnalyzer:
    """
    Comprehensive feature importance analysis class.

    Provides multiple methods for analyzing feature importance including:
    - Tree-based feature importance
    - Permutation importance
    - Mutual information
    - Partial dependence analysis
    """

    def __init__(self):
        self.importance_results = {}
        self.feature_names = []

    def analyze_tree_importance(self, model, feature_names: List[str]) -> pd.DataFrame:
        """
        Extract feature importance from tree-based models.

        Args:
            model: Trained tree-based model
            feature_names: List of feature names

        Returns:
            DataFrame with feature importance scores
        """
        if not hasattr(model, 'feature_importances_'):
            raise ValueError("Model does not have feature_importances_ attribute")

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        self.importance_results['tree_importance'] = importance_df
        self.feature_names = feature_names

        logger.info(f"Tree-based importance analyzed for {len(feature_names)} features")

        return importance_df

    def analyze_permutation_importance(self, model, X: pd.DataFrame, y: pd.Series,
                                        n_repeats: int = 10,
                                        random_state: int = 42) -> pd.DataFrame:
        """
        Calculate permutation importance.

        Args:
            model: Trained model
            X: Feature DataFrame
            y: Target Series
            n_repeats: Number of permutation repeats
            random_state: Random state for reproducibility

        Returns:
            DataFrame with permutation importance scores
        """
        result = permutation_importance(
            model, X, y,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1
        )

        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std
        }).sort_values('importance_mean', ascending=False)

        self.importance_results['permutation_importance'] = importance_df

        logger.info(f"Permutation importance calculated with {n_repeats} repeats")

        return importance_df

    def analyze_mutual_information(self, X: pd.DataFrame, y: pd.Series,
                                    random_state: int = 42) -> pd.DataFrame:
        """
        Calculate mutual information between features and target.

        Args:
            X: Feature DataFrame
            y: Target Series
            random_state: Random state for reproducibility

        Returns:
            DataFrame with mutual information scores
        """
        mi_scores = mutual_info_regression(X, y, random_state=random_state)

        importance_df = pd.DataFrame({
            'feature': X.columns,
            'mutual_info': mi_scores
        }).sort_values('mutual_info', ascending=False)

        self.importance_results['mutual_information'] = importance_df

        logger.info(f"Mutual information calculated for {len(X.columns)} features")

        return importance_df

    def analyze_correlation(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Calculate correlation between features and target.

        Args:
            X: Feature DataFrame
            y: Target Series

        Returns:
            DataFrame with correlation scores
        """
        correlations = X.corrwith(y)

        importance_df = pd.DataFrame({
            'feature': correlations.index,
            'correlation': correlations.values,
            'abs_correlation': np.abs(correlations.values)
        }).sort_values('abs_correlation', ascending=False)

        self.importance_results['correlation'] = importance_df

        logger.info(f"Correlation analysis completed for {len(X.columns)} features")

        return importance_df

    def comprehensive_analysis(self, model, X: pd.DataFrame, y: pd.Series,
                               model_has_feature_importance: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Perform comprehensive feature importance analysis.

        Args:
            model: Trained model
            X: Feature DataFrame
            y: Target Series
            model_has_feature_importance: Whether model has feature_importances_

        Returns:
            Dictionary of all importance analysis results
        """
        results = {}

        # Tree-based importance
        if model_has_feature_importance and hasattr(model, 'feature_importances_'):
            results['tree_importance'] = self.analyze_tree_importance(model, X.columns.tolist())

        # Permutation importance
        try:
            results['permutation_importance'] = self.analyze_permutation_importance(model, X, y)
        except Exception as e:
            logger.warning(f"Permutation importance failed: {str(e)}")

        # Mutual information
        try:
            results['mutual_information'] = self.analyze_mutual_information(X, y)
        except Exception as e:
            logger.warning(f"Mutual information failed: {str(e)}")

        # Correlation
        try:
            results['correlation'] = self.analyze_correlation(X, y)
        except Exception as e:
            logger.warning(f"Correlation analysis failed: {str(e)}")

        return results

    def get_top_features(self, method: str = 'tree_importance',
                         n_features: int = 10) -> List[str]:
        """
        Get top N most important features.

        Args:
            method: Importance method to use
            n_features: Number of top features to return

        Returns:
            List of top feature names
        """
        if method not in self.importance_results:
            raise ValueError(f"Method '{method}' not found in results")

        importance_df = self.importance_results[method]

        # Determine the importance column name
        if 'importance' in importance_df.columns:
            col = 'importance'
        elif 'importance_mean' in importance_df.columns:
            col = 'importance_mean'
        elif 'mutual_info' in importance_df.columns:
            col = 'mutual_info'
        elif 'abs_correlation' in importance_df.columns:
            col = 'abs_correlation'
        else:
            col = importance_df.columns[1]

        top_features = importance_df.nlargest(n_features, col)['feature'].tolist()

        return top_features

    def create_importance_plot(self, method: str = 'tree_importance',
                                title: str = 'Feature Importance',
                                n_features: int = 15) -> go.Figure:
        """
        Create a feature importance bar plot.

        Args:
            method: Importance method to use
            title: Plot title
            n_features: Number of features to display

        Returns:
            Plotly Figure object
        """
        if method not in self.importance_results:
            raise ValueError(f"Method '{method}' not found in results")

        importance_df = self.importance_results[method].head(n_features)

        # Determine the importance column
        if 'importance' in importance_df.columns:
            col = 'importance'
        elif 'importance_mean' in importance_df.columns:
            col = 'importance_mean'
        elif 'mutual_info' in importance_df.columns:
            col = 'mutual_info'
        elif 'abs_correlation' in importance_df.columns:
            col = 'abs_correlation'
        else:
            col = importance_df.columns[1]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=importance_df[col],
            y=importance_df['feature'],
            orientation='h',
            marker_color='#667eea',
            text=importance_df[col].apply(lambda x: f'{x:.4f}'),
            textposition='outside'
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Importance Score',
            yaxis_title='Feature',
            height=max(400, n_features * 25),
            template='plotly_white',
            yaxis={'categoryorder': 'total ascending'}
        )

        return fig

    def generate_importance_report(self) -> Dict:
        """
        Generate a comprehensive importance report.

        Returns:
            Dictionary with importance analysis summary
        """
        if not self.importance_results:
            return {'error': 'No importance results available'}

        report = {
            'methods_used': list(self.importance_results.keys()),
            'top_features_by_method': {},
            'feature_rankings': {},
            'summary': {}
        }

        # Get top features for each method
        for method, df in self.importance_results.items():
            top_features = self.get_top_features(method, 5)
            report['top_features_by_method'][method] = top_features

        # Aggregate rankings across methods
        all_features = set()
        for df in self.importance_results.values():
            all_features.update(df['feature'].tolist())

        feature_scores = {f: 0 for f in all_features}
        for method, df in self.importance_results.items():
            for i, feature in enumerate(df['feature'].tolist()):
                feature_scores[feature] += len(df) - i  # Higher rank = higher score

        report['feature_rankings'] = dict(sorted(feature_scores.items(),
                                                 key=lambda x: x[1], reverse=True))

        # Summary statistics
        if 'tree_importance' in self.importance_results:
            df = self.importance_results['tree_importance']
            report['summary']['top_predictor'] = df.iloc[0]['feature']
            report['summary']['importance_concentration'] = df['importance'].head(3).sum() / df['importance'].sum()

        return report
