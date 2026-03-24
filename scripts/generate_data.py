#!/usr/bin/env python
"""
Utility to generate synthetic retail data for testing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import argparse


def generate_retail_data(
    n_records: int = 1000,
    n_products: int = 50,
    n_categories: int = 10,
    output_path: str = 'retail_price_synthetic.csv'
):
    """
    Generate synthetic retail price data.

    Args:
        n_records: Number of records to generate
        n_products: Number of unique products
        n_categories: Number of product categories
        output_path: Output file path
    """
    np.random.seed(42)
    random.seed(42)

    # Generate categories
    categories = [f'category_{i}' for i in range(n_categories)]

    # Generate products
    products = [f'product_{i}' for i in range(n_products)]
    product_categories = {
        p: random.choice(categories) for p in products
    }

    # Generate dates
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_records)]

    # Generate data
    data = []

    for i in range(n_records):
        product = random.choice(products)
        category = product_categories[product]

        # Base price
        unit_price = round(np.random.uniform(10, 100), 2)

        # Competitor prices
        comp_1 = round(unit_price * np.random.uniform(0.85, 1.15), 2)
        comp_2 = round(unit_price * np.random.uniform(0.80, 1.20), 2)
        comp_3 = round(unit_price * np.random.uniform(0.75, 1.25), 2)

        # Quantity
        qty = np.random.randint(1, 50)

        # Total price
        total_price = round(unit_price * qty, 2)

        # Freight
        freight_price = round(np.random.uniform(5, 20), 2)

        # Product score
        product_score = round(np.random.uniform(3.0, 5.0), 1)

        record = {
            'product_id': product,
            'product_category_name': category,
            'month_year': dates[i].strftime('%d-%m-%Y'),
            'qty': qty,
            'total_price': total_price,
            'freight_price': freight_price,
            'unit_price': unit_price,
            'product_score': product_score,
            'customers': np.random.randint(5, 50),
            'comp_1': comp_1,
            'comp_2': comp_2,
            'comp_3': comp_3,
        }

        data.append(record)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(output_path, index=False)

    print(f"Generated {n_records} records")
    print(f"Products: {n_products}")
    print(f"Categories: {n_categories}")
    print(f"Saved to: {output_path}")

    return df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic retail data'
    )

    parser.add_argument(
        '--records', '-n',
        type=int,
        default=1000,
        help='Number of records to generate'
    )

    parser.add_argument(
        '--products', '-p',
        type=int,
        default=50,
        help='Number of unique products'
    )

    parser.add_argument(
        '--output', '-o',
        default='retail_price_synthetic.csv',
        help='Output file path'
    )

    args = parser.parse_args()

    generate_retail_data(
        n_records=args.records,
        n_products=args.products,
        output_path=args.output
    )


if __name__ == '__main__':
    main()
