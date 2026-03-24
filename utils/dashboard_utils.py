"""
Streamlit Dashboard Enhancement Module.

Additional components and utilities for enhanced dashboard functionality.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from functools import wraps
import time
import logging

logger = logging.getLogger(__name__)


def timing_decorator(func: Callable) -> Callable:
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} executed in {execution_time:.4f} seconds")
        return result
    return wrapper


def cache_dataframe(ttl: int = 3600):
    """
    Decorator to cache DataFrame results.

    Args:
        ttl: Time to live in seconds
    """
    def decorator(func: Callable) -> Callable:
        @st.cache_data(ttl=ttl)
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


class DashboardComponents:
    """Reusable dashboard components."""

    @staticmethod
    def metric_card(title: str, value: str, delta: Optional[str] = None,
                   color: str = '#667eea') -> str:
        """
        Create an HTML metric card.

        Args:
            title: Metric title
            value: Metric value
            delta: Optional delta value
            color: Card color

        Returns:
            HTML string for the card
        """
        delta_html = f"<div style='font-size: 0.9rem; opacity: 0.8;'>{delta}</div>" if delta else ""
        return f"""
        <div style="
            background: linear-gradient(135deg, {color} 0%, {color}99 100%);
            padding: 1.2rem;
            border-radius: 12px;
            color: white;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        ">
            <div style="font-size: 0.9rem; opacity: 0.9; margin-bottom: 0.5rem;">{title}</div>
            <div style="font-size: 1.8rem; font-weight: bold;">{value}</div>
            {delta_html}
        </div>
        """

    @staticmethod
    def section_header(title: str, icon: str = "📊") -> str:
        """
        Create a styled section header.

        Args:
            title: Section title
            icon: Emoji icon

        Returns:
            HTML string for the header
        """
        return f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            color: white;
            font-size: 1.3rem;
            font-weight: bold;
        ">
            {icon} {title}
        </div>
        """

    @staticmethod
    def info_box(message: str, box_type: str = 'info') -> str:
        """
        Create a styled info box.

        Args:
            message: Box message
            box_type: Type of box (info, success, warning, error)

        Returns:
            HTML string for the box
        """
        colors = {
            'info': '#2196F3',
            'success': '#4CAF50',
            'warning': '#FF9800',
            'error': '#f44336'
        }
        color = colors.get(box_type, colors['info'])

        return f"""
        <div style="
            background-color: {color}22;
            border-left: 4px solid {color};
            padding: 1rem;
            border-radius: 5px;
            margin: 0.5rem 0;
        ">
            {message}
        </div>
        """


class DataQualityChecker:
    """Data quality checking utilities."""

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.issues = []

    def check_missing_values(self, threshold: float = 0.1) -> bool:
        """Check for missing values above threshold."""
        missing_ratio = self.data.isnull().sum().sum() / (len(self.data) * len(self.data.columns))
        if missing_ratio > threshold:
            self.issues.append(f"High missing value ratio: {missing_ratio:.2%}")
            return False
        return True

    def check_duplicates(self) -> bool:
        """Check for duplicate rows."""
        duplicates = self.data.duplicated().sum()
        if duplicates > 0:
            self.issues.append(f"Found {duplicates} duplicate rows")
            return False
        return True

    def check_outliers(self, columns: List[str], method: str = 'iqr') -> Dict:
        """Check for outliers in specified columns."""
        outliers = {}
        for col in columns:
            if col not in self.data.columns:
                continue

            data = self.data[col].dropna()
            if method == 'iqr':
                Q1, Q3 = data.quantile([0.25, 0.75])
                IQR = Q3 - Q1
                outlier_count = ((data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)).sum()
            else:
                mean, std = data.mean(), data.std()
                outlier_count = ((data < mean - 3 * std) | (data > mean + 3 * std)).sum()

            outliers[col] = outlier_count

        return outliers

    def generate_report(self) -> Dict:
        """Generate data quality report."""
        report = {
            'total_rows': len(self.data),
            'total_columns': len(self.data.columns),
            'missing_values': self.data.isnull().sum().to_dict(),
            'duplicates': self.data.duplicated().sum(),
            'issues': self.issues,
            'memory_usage': f"{self.data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
        }
        return report


class StateManager:
    """Streamlit session state management."""

    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """Get value from session state."""
        if key not in st.session_state:
            st.session_state[key] = default
        return st.session_state[key]

    @staticmethod
    def set(key: str, value: Any):
        """Set value in session state."""
        st.session_state[key] = value

    @staticmethod
    def clear():
        """Clear session state."""
        st.session_state.clear()

    @staticmethod
    def get_or_compute(key: str, compute_func: Callable, *args, **kwargs) -> Any:
        """Get value or compute and store it."""
        if key not in st.session_state:
            st.session_state[key] = compute_func(*args, **kwargs)
        return st.session_state[key]


class FilterManager:
    """Data filtering utilities."""

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.filters = {}

    def add_filter(self, column: str, filter_type: str, value: Any):
        """Add a filter condition."""
        self.filters[column] = {
            'type': filter_type,
            'value': value
        }

    def apply_filters(self) -> pd.DataFrame:
        """Apply all filters and return filtered data."""
        filtered_data = self.data.copy()

        for column, filter_config in self.filters.items():
            if column not in filtered_data.columns:
                continue

            filter_type = filter_config['type']
            value = filter_config['value']

            if filter_type == 'range':
                filtered_data = filtered_data[
                    (filtered_data[column] >= value[0]) &
                    (filtered_data[column] <= value[1])
                ]
            elif filter_type == 'equals':
                filtered_data = filtered_data[filtered_data[column] == value]
            elif filter_type == 'contains':
                filtered_data = filtered_data[filtered_data[column].str.contains(value, na=False)]
            elif filter_type == 'in':
                filtered_data = filtered_data[filtered_data[column].isin(value)]

        return filtered_data

    def clear_filters(self):
        """Clear all filters."""
        self.filters = {}


class ExportManager:
    """Data export utilities."""

    @staticmethod
    def to_csv(data: pd.DataFrame) -> str:
        """Export DataFrame to CSV string."""
        return data.to_csv(index=False)

    @staticmethod
    def to_json(data: pd.DataFrame) -> str:
        """Export DataFrame to JSON string."""
        return data.to_json(orient='records', indent=2)

    @staticmethod
    def to_excel(data: pd.DataFrame) -> bytes:
        """Export DataFrame to Excel bytes."""
        import io
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            data.to_excel(writer, index=False, sheet_name='Data')
        return output.getvalue()

    @staticmethod
    def create_download_button(data: pd.DataFrame, filename: str,
                               format: str = 'csv', key: str = None):
        """Create a Streamlit download button."""
        if format == 'csv':
            content = ExportManager.to_csv(data)
            mime = 'text/csv'
            ext = 'csv'
        elif format == 'json':
            content = ExportManager.to_json(data)
            mime = 'application/json'
            ext = 'json'
        elif format == 'excel':
            content = ExportManager.to_excel(data)
            mime = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            ext = 'xlsx'
        else:
            raise ValueError(f"Unsupported format: {format}")

        st.download_button(
            label=f"📥 Download {format.upper()}",
            data=content,
            file_name=f"{filename}.{ext}",
            mime=mime,
            key=key
        )
