"""
Visualization Utilities Module.

This module provides comprehensive chart building utilities for
retail price analysis dashboards.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)


class ChartBuilder:
    """
    Comprehensive chart building class for retail price visualizations.

    Provides methods for creating various types of charts including
    interactive plots, dashboards, and statistical visualizations.
    """

    def __init__(self, theme: str = 'plotly_white'):
        """
        Initialize ChartBuilder.

        Args:
            theme: Plotly theme to use
        """
        self.theme = theme
        self.color_palette = [
            '#667eea', '#764ba2', '#f093fb', '#f5576c',
            '#4CAF50', '#2196F3', '#FF9800', '#9C27B0'
        ]

    def create_price_distribution_chart(self, data: pd.DataFrame,
                                         price_column: str = 'total_price',
                                         title: str = 'Price Distribution') -> go.Figure:
        """
        Create a comprehensive price distribution visualization.

        Args:
            data: DataFrame with price data
            price_column: Name of price column
            title: Chart title

        Returns:
            Plotly Figure object
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Histogram', 'Box Plot'),
            column_widths=[0.7, 0.3]
        )

        # Histogram
        fig.add_trace(
            go.Histogram(
                x=data[price_column],
                nbinsx=30,
                marker_color=self.color_palette[0],
                name='Distribution',
                showlegend=False
            ),
            row=1, col=1
        )

        # Box plot
        fig.add_trace(
            go.Box(
                y=data[price_column],
                marker_color=self.color_palette[1],
                name='Price',
                showlegend=False
            ),
            row=1, col=2
        )

        fig.update_layout(
            title=title,
            template=self.theme,
            height=400
        )

        return fig

    def create_category_comparison_chart(self, data: pd.DataFrame,
                                          category_column: str = 'product_category_name',
                                          value_column: str = 'total_price',
                                          title: str = 'Category Comparison') -> go.Figure:
        """
        Create a category comparison bar chart.

        Args:
            data: DataFrame with category data
            category_column: Name of category column
            value_column: Name of value column
            title: Chart title

        Returns:
            Plotly Figure object
        """
        category_data = data.groupby(category_column)[value_column].mean().reset_index()
        category_data = category_data.sort_values(value_column, ascending=True)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=category_data[value_column],
            y=category_data[category_column],
            orientation='h',
            marker_color=self.color_palette[0],
            text=category_data[value_column].apply(lambda x: f'${x:.2f}'),
            textposition='outside'
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Average Value',
            yaxis_title='Category',
            template=self.theme,
            height=max(400, len(category_data) * 30)
        )

        return fig

    def create_scatter_plot(self, data: pd.DataFrame, x_column: str, y_column: str,
                            color_column: Optional[str] = None,
                            size_column: Optional[str] = None,
                            title: str = 'Scatter Plot',
                            add_trendline: bool = True) -> go.Figure:
        """
        Create an interactive scatter plot.

        Args:
            data: DataFrame with data
            x_column: X-axis column
            y_column: Y-axis column
            color_column: Column for color coding
            size_column: Column for size variation
            title: Chart title
            add_trendline: Whether to add trendline

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        if color_column:
            categories = data[color_column].unique()
            for i, cat in enumerate(categories):
                cat_data = data[data[color_column] == cat]
                fig.add_trace(go.Scatter(
                    x=cat_data[x_column],
                    y=cat_data[y_column],
                    mode='markers',
                    marker=dict(
                        color=self.color_palette[i % len(self.color_palette)],
                        size=cat_data[size_column] / cat_data[size_column].max() * 20 + 5 if size_column else 10
                    ),
                    name=str(cat)
                ))
        else:
            fig.add_trace(go.Scatter(
                x=data[x_column],
                y=data[y_column],
                mode='markers',
                marker=dict(
                    color=self.color_palette[0],
                    size=data[size_column] / data[size_column].max() * 20 + 5 if size_column else 10,
                    opacity=0.6
                ),
                showlegend=False
            ))

        # Add trendline
        if add_trendline and len(data) > 2:
            z = np.polyfit(data[x_column], data[y_column], 1)
            p = np.poly1d(z)
            x_line = np.linspace(data[x_column].min(), data[x_column].max(), 100)
            fig.add_trace(go.Scatter(
                x=x_line,
                y=p(x_line),
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Trendline'
            ))

        fig.update_layout(
            title=title,
            xaxis_title=x_column,
            yaxis_title=y_column,
            template=self.theme
        )

        return fig

    def create_correlation_heatmap(self, data: pd.DataFrame,
                                    title: str = 'Correlation Heatmap') -> go.Figure:
        """
        Create a correlation heatmap.

        Args:
            data: DataFrame with numeric data
            title: Chart title

        Returns:
            Plotly Figure object
        """
        numeric_data = data.select_dtypes(include=[np.number])
        corr_matrix = numeric_data.corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))

        fig.update_layout(
            title=title,
            template=self.theme,
            height=600,
            width=700
        )

        return fig

    def create_pie_chart(self, data: pd.DataFrame, names_column: str,
                         values_column: str, title: str = 'Distribution') -> go.Figure:
        """
        Create a pie chart.

        Args:
            data: DataFrame with data
            names_column: Column for category names
            values_column: Column for values
            title: Chart title

        Returns:
            Plotly Figure object
        """
        fig = go.Figure(data=go.Pie(
            labels=data[names_column],
            values=data[values_column],
            hole=0.4,
            marker_colors=self.color_palette[:len(data)]
        ))

        fig.update_layout(
            title=title,
            template=self.theme
        )

        return fig

    def create_time_series_chart(self, data: pd.DataFrame, date_column: str,
                                  value_column: str, title: str = 'Time Series') -> go.Figure:
        """
        Create a time series line chart.

        Args:
            data: DataFrame with time series data
            date_column: Name of date column
            value_column: Name of value column
            title: Chart title

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=data[date_column],
            y=data[value_column],
            mode='lines+markers',
            line=dict(color=self.color_palette[0], width=2),
            marker=dict(size=6),
            name=value_column
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Value',
            template=self.theme,
            hovermode='x unified'
        )

        return fig

    def create_multi_metric_dashboard(self, data: pd.DataFrame,
                                       metrics: List[str]) -> go.Figure:
        """
        Create a multi-metric dashboard with gauges.

        Args:
            data: DataFrame with metric data
            metrics: List of metric column names

        Returns:
            Plotly Figure object
        """
        n_metrics = len(metrics)
        fig = make_subplots(
            rows=(n_metrics + 1) // 2, cols=2,
            subplot_titles=metrics,
            specs=[[{'type': 'indicator'}] * 2] * ((n_metrics + 1) // 2)
        )

        for i, metric in enumerate(metrics):
            row = i // 2 + 1
            col = i % 2 + 1

            fig.add_trace(go.Indicator(
                mode='number+delta',
                value=data[metric].mean(),
                delta={'reference': data[metric].median()},
                title={'text': metric},
                number={'prefix': '$' if 'price' in metric.lower() else ''}
            ), row=row, col=col)

        fig.update_layout(
            title='Key Metrics Dashboard',
            template=self.theme,
            height=200 * ((n_metrics + 1) // 2)
        )

        return fig

    def create_comparison_bar_chart(self, data: pd.DataFrame, x_column: str,
                                     y_columns: List[str], title: str = 'Comparison') -> go.Figure:
        """
        Create a grouped bar chart for comparison.

        Args:
            data: DataFrame with data
            x_column: X-axis column
            y_columns: List of Y-axis columns to compare
            title: Chart title

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        for i, col in enumerate(y_columns):
            fig.add_trace(go.Bar(
                x=data[x_column],
                y=data[col],
                name=col,
                marker_color=self.color_palette[i % len(self.color_palette)]
            ))

        fig.update_layout(
            title=title,
            xaxis_title=x_column,
            yaxis_title='Value',
            barmode='group',
            template=self.theme
        )

        return fig

    def create_waterfall_chart(self, data: pd.DataFrame, label_column: str,
                                value_column: str, title: str = 'Waterfall Chart') -> go.Figure:
        """
        Create a waterfall chart.

        Args:
            data: DataFrame with data
            label_column: Column for labels
            value_column: Column for values
            title: Chart title

        Returns:
            Plotly Figure object
        """
        fig = go.Figure(go.Waterfall(
            name='Waterfall',
            orientation='v',
            measure=['relative'] * len(data),
            x=data[label_column],
            y=data[value_column],
            connector={'line': {'color': 'rgb(63, 63, 63)'}},
            textposition='outside',
            text=data[value_column].apply(lambda x: f'${x:,.0f}')
        ))

        fig.update_layout(
            title=title,
            template=self.theme,
            showlegend=False
        )

        return fig

    def create_treemap(self, data: pd.DataFrame, path_columns: List[str],
                       value_column: str, title: str = 'Treemap') -> go.Figure:
        """
        Create a treemap visualization.

        Args:
            data: DataFrame with hierarchical data
            path_columns: List of columns forming the hierarchy
            value_column: Column for sizing
            title: Chart title

        Returns:
            Plotly Figure object
        """
        fig = px.treemap(
            data,
            path=path_columns,
            values=value_column,
            color=value_column,
            color_continuous_scale='Viridis'
        )

        fig.update_layout(
            title=title,
            template=self.theme
        )

        return fig

    def create_funnel_chart(self, stages: List[str], values: List[float],
                            title: str = 'Funnel Chart') -> go.Figure:
        """
        Create a funnel chart.

        Args:
            stages: List of stage names
            values: List of values for each stage
            title: Chart title

        Returns:
            Plotly Figure object
        """
        fig = go.Figure(go.Funnel(
            y=stages,
            x=values,
            textposition='inside',
            textinfo='value+percent initial',
            marker=dict(color=self.color_palette[:len(stages)])
        ))

        fig.update_layout(
            title=title,
            template=self.theme
        )

        return fig

    def create_violin_plot(self, data: pd.DataFrame, x_column: str,
                           y_column: str, title: str = 'Distribution Comparison') -> go.Figure:
        """
        Create a violin plot for distribution comparison.

        Args:
            data: DataFrame with data
            x_column: Column for grouping
            y_column: Column for distribution
            title: Chart title

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        categories = data[x_column].unique()
        for i, cat in enumerate(categories):
            fig.add_trace(go.Violin(
                x=data[data[x_column] == cat][y_column],
                name=str(cat),
                box_visible=True,
                meanline_visible=True,
                marker_color=self.color_palette[i % len(self.color_palette)]
            ))

        fig.update_layout(
            title=title,
            xaxis_title=y_column,
            yaxis_title=x_column,
            template=self.theme
        )

        return fig

    def export_chart(self, fig: go.Figure, filename: str,
                     format: str = 'html', width: int = 1200, height: int = 800):
        """
        Export chart to file.

        Args:
            fig: Plotly Figure object
            filename: Output filename
            format: Output format ('html', 'png', 'jpeg', 'svg', 'pdf')
            width: Width in pixels
            height: Height in pixels
        """
        if format == 'html':
            fig.write_html(filename)
        else:
            fig.write_image(filename, width=width, height=height)

        logger.info(f"Chart exported to {filename}")
