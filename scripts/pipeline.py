#!/usr/bin/env python
"""
Data Pipeline Script.

This script provides utilities for running the data pipeline,
including data validation, preprocessing, and model training.
"""

import argparse
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_data(data_path: str) -> bool:
    """Validate the input data file."""
    import pandas as pd

    logger.info(f"Validating data from: {data_path}")

    try:
        data = pd.read_csv(data_path)

        # Basic validation
        if data.empty:
            logger.error("Data file is empty")
            return False

        logger.info(f"Data loaded: {len(data)} records, {len(data.columns)} columns")

        # Check for required columns
        required = ['product_id', 'product_category_name', 'qty', 'total_price', 'unit_price']
        missing = [col for col in required if col not in data.columns]

        if missing:
            logger.error(f"Missing required columns: {missing}")
            return False

        # Check for null values
        null_counts = data[required].isnull().sum()
        if null_counts.sum() > 0:
            logger.warning(f"Found null values: {null_counts[null_counts > 0].to_dict()}")

        logger.info("Data validation passed")
        return True

    except Exception as e:
        logger.error(f"Data validation failed: {str(e)}")
        return False


def preprocess_data(data_path: str, output_path: str) -> bool:
    """Preprocess the data and save to output."""
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.data_preprocessing import DataPreprocessor

    logger.info(f"Preprocessing data from: {data_path}")

    try:
        data = pd.read_csv(data_path)

        preprocessor = DataPreprocessor(data)
        preprocessor.remove_duplicates()
        preprocessor.handle_missing_values(strategy='mean')
        preprocessor.create_features()

        processed_data = preprocessor.get_processed_data()
        processed_data.to_csv(output_path, index=False)

        logger.info(f"Preprocessed data saved to: {output_path}")
        logger.info(f"Preprocessing steps: {preprocessor.preprocessing_log}")

        return True

    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        return False


def train_model(data_path: str, model_type: str = 'random_forest') -> bool:
    """Train a price prediction model."""
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.price_predictor import PricePredictor

    logger.info(f"Training {model_type} model")

    try:
        data = pd.read_csv(data_path)

        if 'comp_price_diff' not in data.columns:
            data['comp_price_diff'] = data['unit_price'] - data['comp_1']

        predictor = PricePredictor(model_type=model_type)
        predictor.fit(data)

        # Get feature importance
        importance = predictor.get_feature_importance()
        if importance is not None:
            logger.info("Feature importance:")
            for _, row in importance.head().iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        logger.info("Model trained successfully")
        return True

    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        return False


def generate_report(data_path: str, output_path: str) -> bool:
    """Generate analytics report."""
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.advanced_reporting import AdvancedReporter

    logger.info("Generating report")

    try:
        data = pd.read_csv(data_path)
        reporter = AdvancedReporter(data)

        report = reporter.generate_executive_report()

        # Save report
        import json
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Report saved to: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Retail Price Optimization Data Pipeline'
    )

    parser.add_argument(
        'command',
        choices=['validate', 'preprocess', 'train', 'report'],
        help='Command to execute'
    )

    parser.add_argument(
        '--data', '-d',
        default='retail_price.csv',
        help='Path to data file'
    )

    parser.add_argument(
        '--output', '-o',
        help='Output file path'
    )

    parser.add_argument(
        '--model-type', '-m',
        default='random_forest',
        choices=['random_forest', 'gradient_boosting', 'decision_tree', 'linear'],
        help='Model type for training'
    )

    args = parser.parse_args()

    if args.command == 'validate':
        success = validate_data(args.data)
        sys.exit(0 if success else 1)

    elif args.command == 'preprocess':
        output = args.output or 'retail_price_processed.csv'
        success = preprocess_data(args.data, output)
        sys.exit(0 if success else 1)

    elif args.command == 'train':
        success = train_model(args.data, args.model_type)
        sys.exit(0 if success else 1)

    elif args.command == 'report':
        output = args.output or 'report.json'
        success = generate_report(args.data, output)
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
