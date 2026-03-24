"""
Report Generation Module.

This module provides utilities for generating comprehensive reports
in various formats including PDF, Excel, and HTML.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging
import io

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Comprehensive report generation class.

    Generates various types of reports for retail price analysis.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize ReportGenerator.

        Args:
            data: DataFrame with retail price data
        """
        self.data = data
        self.report_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def generate_executive_summary(self) -> Dict:
        """
        Generate executive summary report.

        Returns:
            Dictionary with executive summary data
        """
        summary = {
            'report_title': 'Retail Price Analysis - Executive Summary',
            'report_date': self.report_date,
            'metrics': {
                'total_records': len(self.data),
                'total_products': self.data['product_id'].nunique() if 'product_id' in self.data.columns else 0,
                'total_categories': self.data['product_category_name'].nunique() if 'product_category_name' in self.data.columns else 0,
                'total_revenue': float(self.data['total_price'].sum()) if 'total_price' in self.data.columns else 0,
                'average_order_value': float(self.data['total_price'].mean()) if 'total_price' in self.data.columns else 0,
                'total_quantity': int(self.data['qty'].sum()) if 'qty' in self.data.columns else 0,
                'average_product_score': float(self.data['product_score'].mean()) if 'product_score' in self.data.columns else 0
            },
            'top_categories': [],
            'recommendations': []
        }

        # Top categories
        if 'product_category_name' in self.data.columns and 'total_price' in self.data.columns:
            top_cats = self.data.groupby('product_category_name')['total_price'].sum().nlargest(5)
            summary['top_categories'] = [
                {'name': cat, 'revenue': float(rev)}
                for cat, rev in top_cats.items()
            ]

        # Generate recommendations
        summary['recommendations'] = self._generate_recommendations()

        return summary

    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Price optimization
        if 'unit_price' in self.data.columns and 'comp_1' in self.data.columns:
            avg_diff = (self.data['unit_price'] - self.data['comp_1']).mean()
            if avg_diff > 5:
                recommendations.append(
                    "Consider price optimization - current prices are above market average"
                )
            elif avg_diff < -5:
                recommendations.append(
                    "Opportunity to increase prices while maintaining competitiveness"
                )

        # Product score
        if 'product_score' in self.data.columns:
            low_score_products = (self.data['product_score'] < 3.5).sum()
            if low_score_products > 0:
                recommendations.append(
                    f"{low_score_products} products have low scores - consider quality improvements"
                )

        # Revenue concentration
        if 'product_category_name' in self.data.columns and 'total_price' in self.data.columns:
            cat_revenue = self.data.groupby('product_category_name')['total_price'].sum()
            top_3_share = cat_revenue.nlargest(3).sum() / cat_revenue.sum()
            if top_3_share > 0.7:
                recommendations.append(
                    "High revenue concentration in top 3 categories - consider diversification"
                )

        return recommendations

    def generate_category_report(self) -> pd.DataFrame:
        """
        Generate detailed category performance report.

        Returns:
            DataFrame with category metrics
        """
        if 'product_category_name' not in self.data.columns:
            return pd.DataFrame()

        category_report = self.data.groupby('product_category_name').agg({
            'total_price': ['sum', 'mean', 'count', 'std'],
            'unit_price': ['mean', 'min', 'max'],
            'qty': ['sum', 'mean'],
            'product_score': 'mean'
        }).round(2)

        # Flatten column names
        category_report.columns = ['_'.join(col).strip() for col in category_report.columns.values]
        category_report = category_report.reset_index()

        # Calculate market share
        total_revenue = category_report['total_price_sum'].sum()
        category_report['market_share'] = (
            category_report['total_price_sum'] / total_revenue * 100
        ).round(2)

        return category_report

    def generate_product_report(self, top_n: int = 50) -> pd.DataFrame:
        """
        Generate product performance report.

        Args:
            top_n: Number of top products to include

        Returns:
            DataFrame with product metrics
        """
        if 'product_id' not in self.data.columns:
            return pd.DataFrame()

        product_report = self.data.groupby(['product_id', 'product_category_name']).agg({
            'total_price': ['sum', 'mean', 'count'],
            'unit_price': 'mean',
            'qty': 'sum',
            'product_score': 'mean'
        }).round(2)

        product_report.columns = ['_'.join(col).strip() for col in product_report.columns.values]
        product_report = product_report.reset_index()

        # Sort by revenue and get top N
        product_report = product_report.sort_values('total_price_sum', ascending=False).head(top_n)

        return product_report

    def generate_pricing_report(self) -> Dict:
        """
        Generate pricing analysis report.

        Returns:
            Dictionary with pricing analysis
        """
        pricing_report = {
            'report_date': self.report_date,
            'price_statistics': {},
            'competitor_comparison': {},
            'pricing_recommendations': []
        }

        if 'unit_price' in self.data.columns:
            pricing_report['price_statistics'] = {
                'mean': float(self.data['unit_price'].mean()),
                'median': float(self.data['unit_price'].median()),
                'std': float(self.data['unit_price'].std()),
                'min': float(self.data['unit_price'].min()),
                'max': float(self.data['unit_price'].max())
            }

        if 'comp_1' in self.data.columns:
            pricing_report['competitor_comparison'] = {
                'avg_price_difference': float(
                    (self.data['unit_price'] - self.data['comp_1']).mean()
                ),
                'products_above_competitor': int(
                    (self.data['unit_price'] > self.data['comp_1']).sum()
                ),
                'products_below_competitor': int(
                    (self.data['unit_price'] < self.data['comp_1']).sum()
                )
            }

        return pricing_report

    def export_to_excel(self, filename: str, include_sheets: List[str] = None):
        """
        Export reports to Excel file.

        Args:
            filename: Output filename
            include_sheets: List of sheets to include
        """
        if include_sheets is None:
            include_sheets = ['Executive Summary', 'Category Report', 'Product Report', 'Pricing Report']

        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            if 'Executive Summary' in include_sheets:
                summary = self.generate_executive_summary()
                summary_df = pd.DataFrame([summary['metrics']])
                summary_df.to_excel(writer, sheet_name='Executive Summary', index=False)

            if 'Category Report' in include_sheets:
                cat_report = self.generate_category_report()
                cat_report.to_excel(writer, sheet_name='Category Report', index=False)

            if 'Product Report' in include_sheets:
                prod_report = self.generate_product_report()
                prod_report.to_excel(writer, sheet_name='Product Report', index=False)

            if 'Pricing Report' in include_sheets:
                pricing_report = self.generate_pricing_report()
                pricing_df = pd.DataFrame([pricing_report['price_statistics']])
                pricing_df.to_excel(writer, sheet_name='Pricing Report', index=False)

        logger.info(f"Report exported to {filename}")

    def export_to_csv(self, filename: str, report_type: str = 'category'):
        """
        Export report to CSV file.

        Args:
            filename: Output filename
            report_type: Type of report ('category', 'product', 'pricing')
        """
        if report_type == 'category':
            report = self.generate_category_report()
        elif report_type == 'product':
            report = self.generate_product_report()
        else:
            report = self.generate_pricing_report()
            if isinstance(report, dict):
                report = pd.DataFrame([report])

        report.to_csv(filename, index=False)
        logger.info(f"Report exported to {filename}")

    def generate_html_report(self) -> str:
        """
        Generate HTML formatted report.

        Returns:
            HTML string of the report
        """
        summary = self.generate_executive_summary()
        category_report = self.generate_category_report()

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Retail Price Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #667eea; }}
                h2 {{ color: #764ba2; border-bottom: 2px solid #667eea; padding-bottom: 10px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #667eea; color: white; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .metric {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                          color: white; padding: 20px; border-radius: 10px; margin: 10px; }}
                .recommendations {{ background-color: #e8f5e9; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>📊 Retail Price Analysis Report</h1>
            <p><strong>Generated:</strong> {summary['report_date']}</p>

            <h2>📈 Executive Summary</h2>
            <div class="metric">
                <p><strong>Total Revenue:</strong> ${summary['metrics']['total_revenue']:,.2f}</p>
                <p><strong>Total Products:</strong> {summary['metrics']['total_products']}</p>
                <p><strong>Total Categories:</strong> {summary['metrics']['total_categories']}</p>
                <p><strong>Average Order Value:</strong> ${summary['metrics']['average_order_value']:.2f}</p>
            </div>

            <h2>📁 Top Categories</h2>
            <table>
                <tr><th>Category</th><th>Revenue</th></tr>
                {"".join([f"<tr><td>{c['name']}</td><td>${c['revenue']:,.2f}</td></tr>" for c in summary['top_categories']])}
            </table>

            <h2>💡 Recommendations</h2>
            <div class="recommendations">
                <ul>
                    {"".join([f"<li>{r}</li>" for r in summary['recommendations']])}
                </ul>
            </div>

            <h2>📊 Category Performance</h2>
            {category_report.to_html(index=False) if not category_report.empty else '<p>No data available</p>'}

            <footer style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd;">
                <p>Generated by Retail Price Optimization Dashboard</p>
            </footer>
        </body>
        </html>
        """

        return html
