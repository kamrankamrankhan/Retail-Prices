"""
Visualization Utilities Module.

Provides comprehensive chart building utilities for retail price analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)


class ChartBuilder:
    """
    Comprehensive chart building class for retail price visualizations.
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
        """Create price distribution visualization."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Histogram', 'Box Plot'),
            column_widths=[0.7, 0.3]
        )

        fig.add_trace(
            go.Histogram(
                x=data[price_column],
                nbinsx=30,
                marker_color=self.color_palette[0],
                name='Distribution'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Box(
                y=data[price_column],
                marker_color=self.color_palette[1],
                name='Price'
            ),
            row=1, col=2
        )

        fig.update_layout(
            title=title,
            template=self.theme,
            height=400,
            showlegend=False
        )

        return fig

    def create_category_comparison_chart(self, data: pd.DataFrame,
                                          category_column: str,
                                          value_column: str,
                                          title: str = 'Category Comparison') -> go.Figure:
        """Create category comparison bar chart."""
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
                            title: str = 'Scatter Plot') -> go.Figure:
        """Create scatter plot."""
        fig = go.Figure()

        if color_column:
            categories = data[color_column].unique()
            for i, cat in enumerate(categories):
                cat_data = data[data[color_column] == cat]
                fig.add_trace(go.Scatter(
                    x=cat_data[x_column],
                    y=cat_data[y_column],
                    mode='markers',
                    marker=dict(color=self.color_palette[i % len(self.color_palette)]),
                    name=str(cat)
                ))
        else:
            fig.add_trace(go.Scatter(
                x=data[x_column],
                y=data[y_column],
                mode='markers',
                marker=dict(color=self.color_palette[0], opacity=0.6)
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
        """Create correlation heatmap."""
        numeric_data = data.select_dtypes(include=[np.number])
        corr_matrix = numeric_data.corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))

        fig.update_layout(
            title=title,
            template=self.theme,
            height=600
        )

        return fig

    def create_pie_chart(self, data: pd.DataFrame, names_column: str,
                         values_column: str, title: str = 'Distribution') -> go.Figure:
        """Create pie chart."""
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

    def create_line_chart(self, data: pd.DataFrame, x_column: str,
                          y_column: str, title: str = 'Line Chart') -> go.Figure:
        """Create line chart."""
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=data[x_column],
            y=data[y_column],
            mode='lines+markers',
            line=dict(color=self.color_palette[0], width=2),
            marker=dict(size=6)
        ))

        fig.update_layout(
            title=title,
            xaxis_title=x_column,
            yaxis_title=y_column,
            template=self.theme
        )

        return fig

    def create_multi_line_chart(self, data: pd.DataFrame, x_column: str,
                                 y_columns: List[str], title: str = 'Multi-Line Chart') -> go.Figure:
        """Create multi-line chart."""
        fig = go.Figure()

        for i, col in enumerate(y_columns):
            fig.add_trace(go.Scatter(
                x=data[x_column],
                y=data[col],
                mode='lines',
                line=dict(color=self.color_palette[i % len(self.color_palette)]),
                name=col
            ))

        fig.update_layout(
            title=title,
            xaxis_title=x_column,
            yaxis_title='Value',
            template=self.theme
        )

        return fig

    def create_bar_chart(self, data: pd.DataFrame, x_column: str,
                         y_column: str, title: str = 'Bar Chart') -> go.Figure:
        """Create bar chart."""
        fig = px.bar(data, x=x_column, y=y_column,
                     color_discrete_sequence=self.color_palette)

        fig.update_layout(
            title=title,
            template=self.theme
        )

        return fig

    def create_box_plot(self, data: pd.DataFrame, x_column: str,
                        y_column: str, title: str = 'Box Plot') -> go.Figure:
        """Create box plot."""
        fig = px.box(data, x=x_column, y=y_column,
                     color_discrete_sequence=self.color_palette)

        fig.update_layout(
            title=title,
            template=self.theme
        )

        return fig

    def create_violin_plot(self, data: pd.DataFrame, x_column: str,
                           y_column: str, title: str = 'Violin Plot') -> go.Figure:
        """Create violin plot."""
        fig = px.violin(data, x=x_column, y=y_column,
                        color_discrete_sequence=self.color_palette)

        fig.update_layout(
            title=title,
            template=self.theme
        )

        return fig

    def create_area_chart(self, data: pd.DataFrame, x_column: str,
                          y_column: str, title: str = 'Area Chart') -> go.Figure:
        """Create area chart."""
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=data[x_column],
            y=data[y_column],
            fill='tozeroy',
            mode='none',
            fillcolor=self.color_palette[0]
        ))

        fig.update_layout(
            title=title,
            xaxis_title=x_column,
            yaxis_title=y_column,
            template=self.theme
        )

        return fig

    def create_stacked_bar_chart(self, data: pd.DataFrame, x_column: str,
                                  y_columns: List[str], title: str = 'Stacked Bar') -> go.Figure:
        """Create stacked bar chart."""
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
            barmode='stack',
            template=self.theme
        )

        return fig

    def create_waterfall_chart(self, data: pd.DataFrame, label_column: str,
                                value_column: str, title: str = 'Waterfall') -> go.Figure:
        """Create waterfall chart."""
        fig = go.Figure(go.Waterfall(
            x=data[label_column],
            y=data[value_column],
            connector={'line': {'color': 'rgb(63, 63, 63)'}}
        ))

        fig.update_layout(
            title=title,
            template=self.theme
        )

        return fig
