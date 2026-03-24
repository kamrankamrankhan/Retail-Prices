"""
Data Preprocessing Module.

This module provides comprehensive data preprocessing utilities
for retail price data including cleaning, transformation, and feature engineering.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Comprehensive data preprocessing class for retail price data.

    Handles data cleaning, transformation, feature engineering,
    and preparation for machine learning models.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the DataPreprocessor.

        Args:
            data: Input DataFrame to preprocess
        """
        self.data = data.copy()
        self.original_data = data.copy()
        self.preprocessing_log = []
        self.scalers = {}
        self.encoders = {}

    def get_preprocessing_summary(self) -> Dict:
        """Get summary of preprocessing steps performed."""
        return {
            'steps_performed': self.preprocessing_log,
            'original_shape': self.original_data.shape,
            'current_shape': self.data.shape,
            'rows_removed': len(self.original_data) - len(self.data)
        }

    def remove_duplicates(self, subset: Optional[List[str]] = None) -> 'DataPreprocessor':
        """
        Remove duplicate rows from the dataset.

        Args:
            subset: Columns to consider for duplicates

        Returns:
            self for method chaining
        """
        initial_count = len(self.data)
        self.data = self.data.drop_duplicates(subset=subset)
        removed = initial_count - len(self.data)

        self.preprocessing_log.append(f"Removed {removed} duplicate rows")
        logger.info(f"Removed {removed} duplicate rows")

        return self

    def handle_missing_values(self, strategy: str = 'mean',
                              columns: Optional[List[str]] = None) -> 'DataPreprocessor':
        """
        Handle missing values in the dataset.

        Args:
            strategy: Strategy for imputation ('mean', 'median', 'mode', 'drop', 'knn')
            columns: Specific columns to handle, None for all numeric columns

        Returns:
            self for method chaining
        """
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()

        missing_before = self.data[columns].isnull().sum().sum()

        if strategy == 'drop':
            self.data = self.data.dropna(subset=columns)
        elif strategy == 'knn':
            imputer = KNNImputer(n_neighbors=5)
            self.data[columns] = imputer.fit_transform(self.data[columns])
        else:
            imputer = SimpleImputer(strategy=strategy)
            self.data[columns] = imputer.fit_transform(self.data[columns])

        missing_after = self.data[columns].isnull().sum().sum()

        self.preprocessing_log.append(
            f"Handled missing values using '{strategy}' strategy: {missing_before - missing_after} values filled"
        )
        logger.info(f"Handled {missing_before - missing_after} missing values using {strategy}")

        return self

    def remove_outliers(self, columns: Optional[List[str]] = None,
                        method: str = 'iqr', threshold: float = 1.5) -> 'DataPreprocessor':
        """
        Remove outliers from the dataset.

        Args:
            columns: Columns to check for outliers
            method: Method for outlier detection ('iqr', 'zscore')
            threshold: Threshold for outlier detection

        Returns:
            self for method chaining
        """
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()

        initial_count = len(self.data)
        outlier_mask = pd.Series([False] * len(self.data))

        for col in columns:
            if col not in self.data.columns:
                continue

            if method == 'iqr':
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                col_outliers = (self.data[col] < lower_bound) | (self.data[col] > upper_bound)
            else:  # zscore
                z_scores = np.abs((self.data[col] - self.data[col].mean()) / self.data[col].std())
                col_outliers = z_scores > threshold

            outlier_mask = outlier_mask | col_outliers

        self.data = self.data[~outlier_mask]
        removed = initial_count - len(self.data)

        self.preprocessing_log.append(f"Removed {removed} outliers using {method} method")
        logger.info(f"Removed {removed} outliers")

        return self

    def normalize_data(self, columns: Optional[List[str]] = None,
                       method: str = 'standard') -> 'DataPreprocessor':
        """
        Normalize/Scale numeric columns.

        Args:
            columns: Columns to normalize
            method: Scaling method ('standard', 'minmax')

        Returns:
            self for method chaining
        """
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()

        if method == 'standard':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()

        self.data[columns] = scaler.fit_transform(self.data[columns])

        for col in columns:
            self.scalers[col] = scaler

        self.preprocessing_log.append(f"Normalized {len(columns)} columns using {method} scaling")
        logger.info(f"Normalized columns: {columns}")

        return self

    def encode_categorical(self, columns: Optional[List[str]] = None,
                           method: str = 'label') -> 'DataPreprocessor':
        """
        Encode categorical variables.

        Args:
            columns: Categorical columns to encode
            method: Encoding method ('label', 'onehot')

        Returns:
            self for method chaining
        """
        if columns is None:
            columns = self.data.select_dtypes(include=['object']).columns.tolist()

        for col in columns:
            if col not in self.data.columns:
                continue

            if method == 'label':
                encoder = LabelEncoder()
                self.data[col] = encoder.fit_transform(self.data[col].astype(str))
                self.encoders[col] = encoder
            elif method == 'onehot':
                dummies = pd.get_dummies(self.data[col], prefix=col)
                self.data = pd.concat([self.data.drop(col, axis=1), dummies], axis=1)

        self.preprocessing_log.append(f"Encoded {len(columns)} categorical columns using {method} encoding")
        logger.info(f"Encoded categorical columns: {columns}")

        return self

    def create_features(self) -> 'DataPreprocessor':
        """
        Create derived features for price analysis.

        Returns:
            self for method chaining
        """
        # Price difference with competitors
        if 'unit_price' in self.data.columns and 'comp_1' in self.data.columns:
            self.data['comp_price_diff'] = self.data['unit_price'] - self.data['comp_1']
            self.data['comp_price_diff_pct'] = (
                self.data['unit_price'] - self.data['comp_1']
            ) / self.data['comp_1'] * 100

        # Total competitor average
        comp_cols = [col for col in ['comp_1', 'comp_2', 'comp_3'] if col in self.data.columns]
        if comp_cols:
            self.data['avg_competitor_price'] = self.data[comp_cols].mean(axis=1)

        # Price competitiveness indicator
        if 'avg_competitor_price' in self.data.columns:
            self.data['is_competitive'] = (
                self.data['unit_price'] <= self.data['avg_competitor_price'] * 1.05
            ).astype(int)

        # Revenue metrics
        if 'qty' in self.data.columns and 'unit_price' in self.data.columns:
            self.data['estimated_revenue'] = self.data['qty'] * self.data['unit_price']

        # Profit margin estimation
        if 'unit_price' in self.data.columns and 'freight_price' in self.data.columns:
            self.data['estimated_margin'] = (
                self.data['unit_price'] - self.data['freight_price']
            ) / self.data['unit_price'] * 100

        # Product score category
        if 'product_score' in self.data.columns:
            self.data['score_category'] = pd.cut(
                self.data['product_score'],
                bins=[0, 3, 4, 5],
                labels=['Low', 'Medium', 'High']
            )

        # Date features
        if 'month_year' in self.data.columns:
            try:
                self.data['date'] = pd.to_datetime(self.data['month_year'], format='%d-%m-%Y')
                self.data['year'] = self.data['date'].dt.year
                self.data['month'] = self.data['date'].dt.month
                self.data['quarter'] = self.data['date'].dt.quarter
            except Exception as e:
                logger.warning(f"Could not parse date column: {str(e)}")

        # Volume to weight ratio
        if 'volume' in self.data.columns and 'product_weight_g' in self.data.columns:
            self.data['volume_weight_ratio'] = (
                self.data['volume'] / self.data['product_weight_g']
            )

        self.preprocessing_log.append("Created derived features")
        logger.info("Created derived features")

        return self

    def filter_data(self, column: str, min_val: Optional[float] = None,
                    max_val: Optional[float] = None) -> 'DataPreprocessor':
        """
        Filter data based on column values.

        Args:
            column: Column to filter on
            min_val: Minimum value (inclusive)
            max_val: Maximum value (inclusive)

        Returns:
            self for method chaining
        """
        initial_count = len(self.data)

        if min_val is not None:
            self.data = self.data[self.data[column] >= min_val]
        if max_val is not None:
            self.data = self.data[self.data[column] <= max_val]

        removed = initial_count - len(self.data)
        self.preprocessing_log.append(f"Filtered {column}: removed {removed} rows")
        logger.info(f"Filtered data on {column}, removed {removed} rows")

        return self

    def aggregate_by_category(self, agg_dict: Optional[Dict] = None) -> pd.DataFrame:
        """
        Aggregate data by product category.

        Args:
            agg_dict: Aggregation dictionary

        Returns:
            Aggregated DataFrame
        """
        if agg_dict is None:
            agg_dict = {
                'total_price': ['sum', 'mean', 'count'],
                'unit_price': ['mean', 'min', 'max'],
                'qty': ['sum', 'mean'],
                'product_score': 'mean'
            }

        aggregated = self.data.groupby('product_category_name').agg(agg_dict)
        aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns.values]

        self.preprocessing_log.append("Aggregated data by category")
        logger.info("Aggregated data by category")

        return aggregated.reset_index()

    def prepare_for_modeling(self, target_column: str = 'total_price',
                             feature_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for machine learning modeling.

        Args:
            target_column: Name of target variable
            feature_columns: List of feature columns

        Returns:
            Tuple of features (X) and target (y)
        """
        if feature_columns is None:
            # Auto-select numeric features
            feature_columns = self.data.select_dtypes(
                include=[np.number]
            ).columns.tolist()
            feature_columns = [col for col in feature_columns if col != target_column]

        X = self.data[feature_columns].copy()
        y = self.data[target_column].copy()

        # Final check for missing values
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())

        self.preprocessing_log.append(f"Prepared data for modeling with {len(feature_columns)} features")
        logger.info(f"Prepared data for modeling: X shape {X.shape}, y shape {y.shape}")

        return X, y

    def get_processed_data(self) -> pd.DataFrame:
        """Get the processed DataFrame."""
        return self.data

    def reset_preprocessing(self) -> 'DataPreprocessor':
        """Reset to original data."""
        self.data = self.original_data.copy()
        self.preprocessing_log = []
        self.scalers = {}
        self.encoders = {}
        logger.info("Reset preprocessing to original data")
        return self

    def generate_preprocessing_report(self) -> Dict:
        """Generate comprehensive preprocessing report."""
        report = {
            'original_shape': self.original_data.shape,
            'processed_shape': self.data.shape,
            'rows_removed': len(self.original_data) - len(self.data),
            'columns_removed': [
                col for col in self.original_data.columns
                if col not in self.data.columns
            ],
            'new_columns': [
                col for col in self.data.columns
                if col not in self.original_data.columns
            ],
            'preprocessing_steps': self.preprocessing_log,
            'missing_values_before': self.original_data.isnull().sum().to_dict(),
            'missing_values_after': self.data.isnull().sum().to_dict(),
            'data_types': self.data.dtypes.to_dict()
        }

        return report
