"""
Database Connector Module.

This module provides database connection and operations for
data persistence in the Retail Price Optimization system.
"""

import sqlite3
import pandas as pd
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
import logging
import os

logger = logging.getLogger(__name__)


class DatabaseConnector:
    """
    Database connector for SQLite operations.

    Provides methods for database CRUD operations and data persistence.
    """

    def __init__(self, db_path: str = 'retail_data.db'):
        """
        Initialize DatabaseConnector.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_tables()

    @contextmanager
    def get_connection(self):
        """Get database connection context manager."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def _ensure_tables(self):
        """Create tables if they don't exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Products table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS products (
                    product_id TEXT PRIMARY KEY,
                    product_category_name TEXT,
                    unit_price REAL,
                    product_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Sales table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sales (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    product_id TEXT,
                    month_year TEXT,
                    qty INTEGER,
                    total_price REAL,
                    freight_price REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (product_id) REFERENCES products(product_id)
                )
            ''')

            # Price history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS price_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    product_id TEXT,
                    old_price REAL,
                    new_price REAL,
                    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    reason TEXT,
                    FOREIGN KEY (product_id) REFERENCES products(product_id)
                )
            ''')

            # Predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    product_id TEXT,
                    predicted_price REAL,
                    actual_price REAL,
                    model_name TEXT,
                    confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (product_id) REFERENCES products(product_id)
                )
            ''')

            conn.commit()
            logger.info("Database tables verified")

    def insert_product(self, product_data: Dict) -> bool:
        """
        Insert a single product record.

        Args:
            product_data: Dictionary with product data

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO products
                    (product_id, product_category_name, unit_price, product_score)
                    VALUES (?, ?, ?, ?)
                ''', (
                    product_data.get('product_id'),
                    product_data.get('product_category_name'),
                    product_data.get('unit_price'),
                    product_data.get('product_score')
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error inserting product: {str(e)}")
            return False

    def bulk_insert_products(self, df: pd.DataFrame) -> int:
        """
        Bulk insert products from DataFrame.

        Args:
            df: DataFrame with product data

        Returns:
            Number of records inserted
        """
        try:
            with self.get_connection() as conn:
                df.to_sql('products', conn, if_exists='append', index=False)
                return len(df)
        except Exception as e:
            logger.error(f"Error bulk inserting products: {str(e)}")
            return 0

    def get_product(self, product_id: str) -> Optional[Dict]:
        """
        Get a single product by ID.

        Args:
            product_id: Product identifier

        Returns:
            Dictionary with product data or None
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT * FROM products WHERE product_id = ?',
                    (product_id,)
                )
                row = cursor.fetchone()
                if row:
                    columns = [desc[0] for desc in cursor.description]
                    return dict(zip(columns, row))
                return None
        except Exception as e:
            logger.error(f"Error getting product: {str(e)}")
            return None

    def get_all_products(self, category: Optional[str] = None) -> pd.DataFrame:
        """
        Get all products, optionally filtered by category.

        Args:
            category: Optional category filter

        Returns:
            DataFrame with products
        """
        try:
            with self.get_connection() as conn:
                if category:
                    query = 'SELECT * FROM products WHERE product_category_name = ?'
                    df = pd.read_sql(query, conn, params=(category,))
                else:
                    df = pd.read_sql('SELECT * FROM products', conn)
                return df
        except Exception as e:
            logger.error(f"Error getting products: {str(e)}")
            return pd.DataFrame()

    def record_price_change(self, product_id: str, old_price: float,
                            new_price: float, reason: str = '') -> bool:
        """
        Record a price change in history.

        Args:
            product_id: Product identifier
            old_price: Previous price
            new_price: New price
            reason: Reason for change

        Returns:
            True if successful
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO price_history
                    (product_id, old_price, new_price, reason)
                    VALUES (?, ?, ?, ?)
                ''', (product_id, old_price, new_price, reason))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error recording price change: {str(e)}")
            return False

    def save_prediction(self, product_id: str, predicted_price: float,
                        model_name: str, confidence: float,
                        actual_price: Optional[float] = None) -> bool:
        """
        Save a prediction record.

        Args:
            product_id: Product identifier
            predicted_price: Predicted price
            model_name: Name of the model used
            confidence: Confidence score
            actual_price: Optional actual price for validation

        Returns:
            True if successful
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO predictions
                    (product_id, predicted_price, actual_price, model_name, confidence)
                    VALUES (?, ?, ?, ?, ?)
                ''', (product_id, predicted_price, actual_price, model_name, confidence))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving prediction: {str(e)}")
            return False

    def get_prediction_history(self, product_id: str,
                                limit: int = 10) -> pd.DataFrame:
        """
        Get prediction history for a product.

        Args:
            product_id: Product identifier
            limit: Maximum number of records

        Returns:
            DataFrame with prediction history
        """
        try:
            with self.get_connection() as conn:
                query = '''
                    SELECT * FROM predictions
                    WHERE product_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                '''
                df = pd.read_sql(query, conn, params=(product_id, limit))
                return df
        except Exception as e:
            logger.error(f"Error getting prediction history: {str(e)}")
            return pd.DataFrame()

    def execute_query(self, query: str, params: tuple = None) -> List[Dict]:
        """
        Execute a custom SQL query.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            List of dictionaries with results
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)

                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return []

    def get_database_stats(self) -> Dict:
        """
        Get database statistics.

        Returns:
            Dictionary with database stats
        """
        stats = {}

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Count products
                cursor.execute('SELECT COUNT(*) FROM products')
                stats['total_products'] = cursor.fetchone()[0]

                # Count sales
                cursor.execute('SELECT COUNT(*) FROM sales')
                stats['total_sales'] = cursor.fetchone()[0]

                # Count predictions
                cursor.execute('SELECT COUNT(*) FROM predictions')
                stats['total_predictions'] = cursor.fetchone()[0]

                # Database size
                stats['db_size_bytes'] = os.path.getsize(self.db_path)

        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")

        return stats

    def backup_database(self, backup_path: str) -> bool:
        """
        Create a backup of the database.

        Args:
            backup_path: Path for backup file

        Returns:
            True if successful
        """
        try:
            with self.get_connection() as conn:
                backup_conn = sqlite3.connect(backup_path)
                conn.backup(backup_conn)
                backup_conn.close()
                logger.info(f"Database backed up to {backup_path}")
                return True
        except Exception as e:
            logger.error(f"Error backing up database: {str(e)}")
            return False

    def close(self):
        """Close database connection."""
        # Connections are managed via context manager
        logger.info("Database connection closed")
