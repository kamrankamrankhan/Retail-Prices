"""
Price Prediction Models Module.

This module contains various machine learning models for retail price prediction
including Decision Trees, Random Forests, Gradient Boosting, and ensemble methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Data class to store model performance metrics."""
    model_name: str
    mse: float
    rmse: float
    mae: float
    r2: float
    mape: float
    training_time: float

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        return {
            'model_name': self.model_name,
            'mse': self.mse,
            'rmse': self.rmse,
            'mae': self.mae,
            'r2': self.r2,
            'mape': self.mape,
            'training_time': self.training_time
        }


class BaseModel(ABC):
    """Abstract base class for all prediction models."""

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_fitted = False
        self.feature_importance_ = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseModel':
        """Fit the model to training data."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        pass

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance if available."""
        if self.feature_importance_ is not None:
            return self.feature_importance_
        return None


class DecisionTreeModel(BaseModel):
    """Decision Tree Regressor for price prediction."""

    def __init__(self, max_depth: int = 10, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, random_state: int = 42):
        super().__init__("Decision Tree")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'DecisionTreeModel':
        """Fit the Decision Tree model."""
        logger.info(f"Training Decision Tree model...")
        self.model.fit(X, y)
        self.is_fitted = True

        # Calculate feature importance
        self.feature_importance_ = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info(f"Decision Tree model trained successfully")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using Decision Tree."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)


class RandomForestModel(BaseModel):
    """Random Forest Regressor for price prediction."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 10,
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 random_state: int = 42, n_jobs: int = -1):
        super().__init__("Random Forest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'RandomForestModel':
        """Fit the Random Forest model."""
        logger.info(f"Training Random Forest model with {self.n_estimators} estimators...")
        self.model.fit(X, y)
        self.is_fitted = True

        # Calculate feature importance
        self.feature_importance_ = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info(f"Random Forest model trained successfully")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using Random Forest."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)


class GradientBoostingModel(BaseModel):
    """Gradient Boosting Regressor for price prediction."""

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 5, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, random_state: int = 42):
        super().__init__("Gradient Boosting")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'GradientBoostingModel':
        """Fit the Gradient Boosting model."""
        logger.info(f"Training Gradient Boosting model...")
        self.model.fit(X, y)
        self.is_fitted = True

        # Calculate feature importance
        self.feature_importance_ = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info(f"Gradient Boosting model trained successfully")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using Gradient Boosting."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)


class LinearRegressionModel(BaseModel):
    """Linear Regression model for price prediction."""

    def __init__(self, model_type: str = 'linear', alpha: float = 1.0):
        super().__init__(f"Linear Regression ({model_type})")
        self.model_type = model_type
        self.alpha = alpha
        self.scaler = StandardScaler()

        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'ridge':
            self.model = Ridge(alpha=alpha)
        elif model_type == 'lasso':
            self.model = Lasso(alpha=alpha)
        elif model_type == 'elasticnet':
            self.model = ElasticNet(alpha=alpha)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LinearRegressionModel':
        """Fit the Linear Regression model."""
        logger.info(f"Training {self.name} model...")
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        logger.info(f"{self.name} model trained successfully")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using Linear Regression."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class ModelTrainer:
    """
    Comprehensive model training and evaluation class.

    This class provides functionality to train multiple models,
    compare their performance, and select the best model.
    """

    def __init__(self, models: Optional[List[BaseModel]] = None):
        """
        Initialize the ModelTrainer.

        Args:
            models: List of model instances to train
        """
        self.models = models or self._get_default_models()
        self.model_metrics: Dict[str, ModelMetrics] = {}
        self.best_model = None
        self.best_model_name = None

    def _get_default_models(self) -> List[BaseModel]:
        """Get default set of models to train."""
        return [
            DecisionTreeModel(),
            RandomForestModel(),
            GradientBoostingModel(),
            LinearRegressionModel('linear'),
            LinearRegressionModel('ridge'),
        ]

    def prepare_features(self, data: pd.DataFrame,
                         feature_columns: List[str],
                         target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for model training.

        Args:
            data: Input DataFrame
            feature_columns: List of feature column names
            target_column: Target column name

        Returns:
            Tuple of features DataFrame and target Series
        """
        X = data[feature_columns].copy()
        y = data[target_column].copy()

        # Handle missing values
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())

        return X, y

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                         model_name: str, training_time: float) -> ModelMetrics:
        """Calculate all evaluation metrics."""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        return ModelMetrics(
            model_name=model_name,
            mse=mse,
            rmse=rmse,
            mae=mae,
            r2=r2,
            mape=mape,
            training_time=training_time
        )

    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, ModelMetrics]:
        """
        Train all models and evaluate their performance.

        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target

        Returns:
            Dictionary of model metrics
        """
        logger.info(f"Training {len(self.models)} models...")

        for model in self.models:
            start_time = time.time()

            try:
                # Train model
                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_test)

                # Calculate training time
                training_time = time.time() - start_time

                # Calculate metrics
                metrics = self.calculate_metrics(y_test.values, y_pred,
                                                model.name, training_time)
                self.model_metrics[model.name] = metrics

                logger.info(f"{model.name}: RMSE={metrics.rmse:.4f}, R2={metrics.r2:.4f}")

            except Exception as e:
                logger.error(f"Error training {model.name}: {str(e)}")
                continue

        # Select best model based on R2 score
        if self.model_metrics:
            self.best_model_name = max(self.model_metrics.keys(),
                                       key=lambda x: self.model_metrics[x].r2)
            self.best_model = next(m for m in self.models if m.name == self.best_model_name)
            logger.info(f"Best model: {self.best_model_name}")

        return self.model_metrics

    def get_metrics_dataframe(self) -> pd.DataFrame:
        """Get all metrics as a DataFrame."""
        if not self.model_metrics:
            return pd.DataFrame()

        metrics_list = [m.to_dict() for m in self.model_metrics.values()]
        return pd.DataFrame(metrics_list).sort_values('r2', ascending=False)

    def cross_validate(self, X: pd.DataFrame, y: pd.Series,
                      cv: int = 5) -> Dict[str, Dict]:
        """
        Perform cross-validation for all models.

        Args:
            X: Features
            y: Target
            cv: Number of folds

        Returns:
            Dictionary of cross-validation results
        """
        cv_results = {}

        for model in self.models:
            try:
                scores = cross_val_score(model.model, X, y, cv=cv, scoring='r2')
                cv_results[model.name] = {
                    'mean_r2': scores.mean(),
                    'std_r2': scores.std(),
                    'all_scores': scores
                }
                logger.info(f"{model.name} CV R2: {scores.mean():.4f} (+/- {scores.std():.4f})")
            except Exception as e:
                logger.error(f"Cross-validation error for {model.name}: {str(e)}")

        return cv_results

    def save_model(self, model: BaseModel, filepath: str):
        """Save a trained model to disk."""
        joblib.dump(model, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> BaseModel:
        """Load a trained model from disk."""
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model


class PricePredictor:
    """
    Main price prediction class that wraps all functionality.

    This class provides a high-level interface for price prediction
    including data preparation, model training, and prediction.
    """

    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize the PricePredictor.

        Args:
            model_type: Type of model to use ('decision_tree', 'random_forest',
                       'gradient_boosting', 'linear')
        """
        self.model_type = model_type
        self.model = self._create_model(model_type)
        self.is_fitted = False
        self.feature_columns = ['qty', 'unit_price', 'comp_1', 'product_score', 'comp_price_diff']
        self.target_column = 'total_price'

    def _create_model(self, model_type: str) -> BaseModel:
        """Create a model based on the specified type."""
        models = {
            'decision_tree': DecisionTreeModel(),
            'random_forest': RandomForestModel(),
            'gradient_boosting': GradientBoostingModel(),
            'linear': LinearRegressionModel('linear'),
            'ridge': LinearRegressionModel('ridge'),
            'lasso': LinearRegressionModel('lasso')
        }

        if model_type not in models:
            raise ValueError(f"Unknown model type: {model_type}")

        return models[model_type]

    def prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for training or prediction."""
        # Calculate derived features
        if 'comp_price_diff' not in data.columns:
            data['comp_price_diff'] = data['unit_price'] - data['comp_1']

        X = data[self.feature_columns].copy()
        y = data[self.target_column].copy() if self.target_column in data.columns else None

        # Handle missing values
        X = X.fillna(X.mean())
        if y is not None:
            y = y.fillna(y.mean())

        return X, y

    def fit(self, data: pd.DataFrame) -> 'PricePredictor':
        """Fit the model to the data."""
        X, y = self.prepare_data(data)

        if y is None:
            raise ValueError("Target column not found in data")

        self.model.fit(X, y)
        self.is_fitted = True

        return self

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make price predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X, _ = self.prepare_data(data)
        return self.model.predict(X)

    def predict_single(self, qty: float, unit_price: float, comp_price: float,
                       product_score: float) -> float:
        """Make a single price prediction."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        data = pd.DataFrame({
            'qty': [qty],
            'unit_price': [unit_price],
            'comp_1': [comp_price],
            'product_score': [product_score],
            'comp_price_diff': [unit_price - comp_price]
        })

        return self.predict(data)[0]

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance from the model."""
        return self.model.get_feature_importance()
