"""
Dashboard Utilities Module.

This module provides utility classes and functions for dashboard
components, state management, and data quality checking.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
import json
import logging

logger = logging.getLogger(__name__)


class DashboardComponents:
    """
    Dashboard component generators for Streamlit applications.
    """

    @staticmethod
    def create_metric_card(title: str, value: Any, delta: Optional[float] = None,
                           format_string: str = '{:.2f}') -> Dict:
        """
        Create a metric card configuration.

        Args:
            title: Metric title
            value: Metric value
            delta: Optional change value
            format_string: Format string for value

        Returns:
            Dictionary with metric configuration
        """
        return {
            'title': title,
            'value': format_string.format(value) if isinstance(value, (int, float)) else str(value),
            'delta': delta
        }

    @staticmethod
    def create_kpi_row(data: Dict[str, float]) -> List[Dict]:
        """
        Create a row of KPI metrics.

        Args:
            data: Dictionary of metric names and values

        Returns:
            List of metric card configurations
        """
        return [
            DashboardComponents.create_metric_card(name, value)
            for name, value in data.items()
        ]


class DataQualityChecker:
    """
    Data quality validation and checking utilities.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize DataQualityChecker.

        Args:
            data: DataFrame to check
        """
        self.data = data
        self.quality_report = {}

    def check_completeness(self) -> Dict:
        """
        Check data completeness.

        Returns:
            Dictionary with completeness metrics
        """
        total_cells = self.data.size
        missing_cells = self.data.isnull().sum().sum()
        completeness = 1 - (missing_cells / total_cells) if total_cells > 0 else 0

        return {
            'total_cells': total_cells,
            'missing_cells': int(missing_cells),
            'completeness_rate': float(completeness),
            'columns_with_missing': self.data.columns[self.data.isnull().any()].tolist()
        }

    def check_uniqueness(self, subset: Optional[List[str]] = None) -> Dict:
        """
        Check data uniqueness.

        Args:
            subset: Columns to check for uniqueness

        Returns:
            Dictionary with uniqueness metrics
        """
        if subset:
            duplicate_count = self.data[subset].duplicated().sum()
        else:
            duplicate_count = self.data.duplicated().sum()

        uniqueness = 1 - (duplicate_count / len(self.data)) if len(self.data) > 0 else 0

        return {
            'total_rows': len(self.data),
            'duplicate_rows': int(duplicate_count),
            'uniqueness_rate': float(uniqueness)
        }

    def check_validity(self, rules: Dict[str, Callable]) -> Dict:
        """
        Check data validity against rules.

        Args:
            rules: Dictionary of column names and validation functions

        Returns:
            Dictionary with validity metrics
        """
        results = {}
        for column, rule in rules.items():
            if column in self.data.columns:
                valid_count = self.data[column].apply(rule).sum()
                results[column] = {
                    'valid_count': int(valid_count),
                    'invalid_count': int(len(self.data) - valid_count),
                    'validity_rate': float(valid_count / len(self.data)) if len(self.data) > 0 else 0
                }

        return results

    def generate_quality_report(self) -> Dict:
        """
        Generate comprehensive quality report.

        Returns:
            Dictionary with all quality checks
        """
        self.quality_report = {
            'completeness': self.check_completeness(),
            'uniqueness': self.check_uniqueness(),
            'validity': self.check_validity({
                'unit_price': lambda x: x > 0,
                'qty': lambda x: x >= 0
            }),
            'overall_score': 0.0
        }

        # Calculate overall score
        scores = [
            self.quality_report['completeness']['completeness_rate'],
            self.quality_report['uniqueness']['uniqueness_rate']
        ]
        self.quality_report['overall_score'] = float(np.mean(scores))

        return self.quality_report


class StateManager:
    """
    Session state management for dashboard applications.
    """

    def __init__(self):
        """Initialize StateManager."""
        self._state = {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get state value.

        Args:
            key: State key
            default: Default value if key not found

        Returns:
            State value or default
        """
        return self._state.get(key, default)

    def set(self, key: str, value: Any):
        """
        Set state value.

        Args:
            key: State key
            value: Value to set
        """
        self._state[key] = value

    def clear(self):
        """Clear all state."""
        self._state.clear()

    def to_json(self) -> str:
        """
        Export state to JSON string.

        Returns:
            JSON string of state
        """
        return json.dumps(self._state, default=str)

    def from_json(self, json_string: str):
        """
        Load state from JSON string.

        Args:
            json_string: JSON string to load
        """
        self._state = json.loads(json_string)


class FilterManager:
    """
    Data filtering utilities for dashboards.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize FilterManager.

        Args:
            data: DataFrame to filter
        """
        self.data = data
        self.filters = {}

    def add_filter(self, column: str, filter_type: str, value: Any):
        """
        Add a filter.

        Args:
            column: Column to filter on
            filter_type: Type of filter ('eq', 'ne', 'gt', 'lt', 'gte', 'lte', 'in', 'contains')
            value: Filter value
        """
        self.filters[column] = {
            'type': filter_type,
            'value': value
        }

    def apply_filters(self) -> pd.DataFrame:
        """
        Apply all filters to data.

        Returns:
            Filtered DataFrame
        """
        result = self.data.copy()

        for column, filter_config in self.filters.items():
            if column not in result.columns:
                continue

            filter_type = filter_config['type']
            value = filter_config['value']

            if filter_type == 'eq':
                result = result[result[column] == value]
            elif filter_type == 'ne':
                result = result[result[column] != value]
            elif filter_type == 'gt':
                result = result[result[column] > value]
            elif filter_type == 'lt':
                result = result[result[column] < value]
            elif filter_type == 'gte':
                result = result[result[column] >= value]
            elif filter_type == 'lte':
                result = result[result[column] <= value]
            elif filter_type == 'in':
                result = result[result[column].isin(value)]
            elif filter_type == 'contains':
                result = result[result[column].astype(str).str.contains(value, case=False)]

        return result

    def clear_filters(self):
        """Clear all filters."""
        self.filters.clear()


class ExportManager:
    """
    Data export utilities for dashboards.
    """

    @staticmethod
    def to_csv(data: pd.DataFrame, filename: str = 'export.csv') -> str:
        """
        Export DataFrame to CSV.

        Args:
            data: DataFrame to export
            filename: Output filename

        Returns:
            CSV string
        """
        return data.to_csv(index=False)

    @staticmethod
    def to_json(data: pd.DataFrame, filename: str = 'export.json') -> str:
        """
        Export DataFrame to JSON.

        Args:
            data: DataFrame to export
            filename: Output filename

        Returns:
            JSON string
        """
        return data.to_json(orient='records', indent=2)

    @staticmethod
    def to_excel(data: pd.DataFrame, filename: str = 'export.xlsx') -> bytes:
        """
        Export DataFrame to Excel.

        Args:
            data: DataFrame to export
            filename: Output filename

        Returns:
            Excel file bytes
        """
        from io import BytesIO
        buffer = BytesIO()
        data.to_excel(buffer, index=False, engine='openpyxl')
        buffer.seek(0)
        return buffer.getvalue()
