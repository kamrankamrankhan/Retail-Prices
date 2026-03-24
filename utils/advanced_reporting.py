"""
Advanced Reporting Module.

Provides comprehensive reporting capabilities for executive summaries,
performance reports, and strategic analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import io
import logging

logger = logging.getLogger(__name__)


class AdvancedReporter:
    """
    Advanced reporting class for comprehensive business reporting.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize AdvancedReporter.

        Args:
            data: DataFrame with retail price data
        """
        self.data = data
        self.report_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def generate_executive_report(self) -> Dict:
        """
        Generate executive summary report.

        Returns:
            Dictionary with executive report
        """
        report = {
            'title': 'Retail Price Optimization - Executive Report',
            'generated_at': self.report_date,
            'period': self._get_data_period(),
            'key_metrics': self._calculate_key_metrics(),
            'highlights': self._generate_highlights(),
            'concerns': self._generate_concerns(),
            'opportunities': self._identify_opportunities(),
            'next_steps': self._suggest_next_steps()
        }

        return report

    def _get_data_period(self) -> Dict:
        """Get data period information."""
        if 'month_year' in self.data.columns:
            dates = self.data['month_year'].unique()
            return {
                'start': min(dates) if len(dates) > 0 else 'N/A',
                'end': max(dates) if len(dates) > 0 else 'N/A',
                'records': len(self.data)
            }
        return {'start': 'N/A', 'end': 'N/A', 'records': len(self.data)}

    def _calculate_key_metrics(self) -> Dict:
        """Calculate key performance metrics."""
        metrics = {}

        if 'total_price' in self.data.columns:
            metrics['total_revenue'] = float(self.data['total_price'].sum())
            metrics['avg_transaction'] = float(self.data['total_price'].mean())

        if 'qty' in self.data.columns:
            metrics['total_units'] = int(self.data['qty'].sum())

        if 'product_id' in self.data.columns:
            metrics['unique_products'] = int(self.data['product_id'].nunique())

        if 'product_category_name' in self.data.columns:
            metrics['categories'] = int(self.data['product_category_name'].nunique())

        if 'product_score' in self.data.columns:
            metrics['avg_product_score'] = float(self.data['product_score'].mean())

        if 'unit_price' in self.data.columns:
            metrics['avg_unit_price'] = float(self.data['unit_price'].mean())

        return metrics

    def _generate_highlights(self) -> List[str]:
        """Generate positive highlights."""
        highlights = []

        # Revenue growth
        if 'total_price' in self.data.columns and len(self.data) > 0:
            total = self.data['total_price'].sum()
            if total > 100000:
                highlights.append(f"Strong total revenue of ${total:,.2f}")

        # Product performance
        if 'product_score' in self.data.columns:
            avg_score = self.data['product_score'].mean()
            if avg_score >= 4.0:
                highlights.append(f"Excellent average product score of {avg_score:.2f}")

        # Category performance
        if 'product_category_name' in self.data.columns:
            n_categories = self.data['product_category_name'].nunique()
            if n_categories >= 5:
                highlights.append(f"Diverse product portfolio with {n_categories} categories")

        return highlights

    def _generate_concerns(self) -> List[str]:
        """Generate areas of concern."""
        concerns = []

        # Price competition
        if 'unit_price' in self.data.columns and 'comp_1' in self.data.columns:
            price_diff = (self.data['unit_price'] - self.data['comp_1']).mean()
            if price_diff > 10:
                concerns.append(f"Prices significantly above market average (${price_diff:.2f})")

        # Product quality
        if 'product_score' in self.data.columns:
            low_score_pct = (self.data['product_score'] < 3.0).sum() / len(self.data) * 100
            if low_score_pct > 10:
                concerns.append(f"{low_score_pct:.1f}% of products have low quality scores")

        # Revenue concentration
        if 'product_category_name' in self.data.columns and 'total_price' in self.data.columns:
            cat_revenue = self.data.groupby('product_category_name')['total_price'].sum()
            top_3_share = cat_revenue.nlargest(3).sum() / cat_revenue.sum()
            if top_3_share > 0.7:
                concerns.append(f"High revenue concentration ({top_3_share*100:.1f}% in top 3 categories)")

        return concerns

    def _identify_opportunities(self) -> List[str]:
        """Identify business opportunities."""
        opportunities = []

        # Pricing opportunity
        if 'unit_price' in self.data.columns and 'comp_1' in self.data.columns:
            price_diff = (self.data['unit_price'] - self.data['comp_1']).mean()
            if price_diff < -5:
                opportunities.append("Opportunity to increase prices while staying competitive")

        # Category expansion
        if 'product_category_name' in self.data.columns:
            underperforming = self._get_underperforming_categories()
            if underperforming:
                opportunities.append(f"Growth potential in underperforming categories: {', '.join(underperforming[:3])}")

        return opportunities

    def _get_underperforming_categories(self) -> List[str]:
        """Get underperforming categories."""
        if 'product_category_name' not in self.data.columns or 'total_price' not in self.data.columns:
            return []

        cat_revenue = self.data.groupby('product_category_name')['total_price'].sum()
        avg_revenue = cat_revenue.mean()
        underperforming = cat_revenue[cat_revenue < avg_revenue * 0.5].index.tolist()
        return underperforming

    def _suggest_next_steps(self) -> List[str]:
        """Suggest next steps."""
        return [
            "Review pricing strategy for products significantly above market price",
            "Analyze and improve low-scoring products",
            "Develop growth strategy for underperforming categories",
            "Implement inventory optimization based on sales velocity",
            "Consider promotional strategies for high-margin products"
        ]

    def generate_category_report(self) -> pd.DataFrame:
        """Generate detailed category performance report."""
        if 'product_category_name' not in self.data.columns:
            return pd.DataFrame()

        report = self.data.groupby('product_category_name').agg({
            'total_price': ['sum', 'mean', 'count', 'std'],
            'unit_price': ['mean', 'min', 'max'],
            'qty': ['sum', 'mean'],
            'product_score': 'mean'
        }).round(2)

        report.columns = ['_'.join(col) for col in report.columns]
        report = report.reset_index()

        # Add derived metrics
        total_revenue = report['total_price_sum'].sum()
        report['market_share'] = (report['total_price_sum'] / total_revenue * 100).round(2)

        return report.sort_values('total_price_sum', ascending=False)

    def generate_product_report(self, top_n: int = 50) -> pd.DataFrame:
        """Generate product performance report."""
        if 'product_id' not in self.data.columns:
            return pd.DataFrame()

        report = self.data.groupby(['product_id', 'product_category_name']).agg({
            'total_price': ['sum', 'mean', 'count'],
            'unit_price': 'mean',
            'qty': 'sum',
            'product_score': 'mean'
        }).round(2)

        report.columns = ['_'.join(col) for col in report.columns]
        report = report.reset_index()

        return report.sort_values('total_price_sum', ascending=False).head(top_n)

    def export_report_package(self) -> Dict[str, any]:
        """
        Export complete report package.

        Returns:
            Dictionary with all reports
        """
        return {
            'executive_summary': self.generate_executive_report(),
            'category_report': self.generate_category_report().to_dict('records'),
            'product_report': self.generate_product_report().to_dict('records'),
            'export_timestamp': self.report_date
        }
