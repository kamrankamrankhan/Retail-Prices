"""
Recommendation Engine Module.

This module provides intelligent recommendations for pricing strategies
and product optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    """Data class for a single recommendation."""
    category: str
    priority: str  # 'high', 'medium', 'low'
    title: str
    description: str
    impact: str
    action: str


class RecommendationEngine:
    """
    Intelligent recommendation engine for retail optimization.

    Analyzes data and generates actionable recommendations for
    pricing, inventory, and product strategies.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize RecommendationEngine.

        Args:
            data: DataFrame with retail price data
        """
        self.data = data
        self.recommendations: List[Recommendation] = []

    def analyze_and_recommend(self) -> List[Recommendation]:
        """
        Run all analyses and generate recommendations.

        Returns:
            List of Recommendation objects
        """
        self.recommendations = []

        self._analyze_pricing()
        self._analyze_competition()
        self._analyze_performance()
        self._analyze_inventory()
        self._analyze_quality()

        return self.recommendations

    def _add_recommendation(self, category: str, priority: str, title: str,
                           description: str, impact: str, action: str):
        """Add a recommendation to the list."""
        self.recommendations.append(Recommendation(
            category=category,
            priority=priority,
            title=title,
            description=description,
            impact=impact,
            action=action
        ))

    def _analyze_pricing(self):
        """Analyze pricing and generate recommendations."""
        # Price variance analysis
        if 'unit_price' in self.data.columns:
            price_std = self.data['unit_price'].std()
            price_mean = self.data['unit_price'].mean()
            cv = price_std / price_mean

            if cv > 0.5:
                self._add_recommendation(
                    category='Pricing',
                    priority='medium',
                    title='High Price Variance Detected',
                    description=f'Price coefficient of variation is {cv:.2f}, indicating inconsistent pricing.',
                    impact='May confuse customers and reduce trust',
                    action='Consider standardizing pricing within categories'
                )

        # Margin analysis
        if 'unit_price' in self.data.columns and 'freight_price' in self.data.columns:
            margin = (self.data['unit_price'] - self.data['freight_price']) / self.data['unit_price']
            low_margin_pct = (margin < 0.1).sum() / len(margin) * 100

            if low_margin_pct > 10:
                self._add_recommendation(
                    category='Pricing',
                    priority='high',
                    title='Low Profit Margin Products',
                    description=f'{low_margin_pct:.1f}% of products have margins below 10%.',
                    impact='Reduced profitability and sustainability',
                    action='Review pricing strategy for low-margin products or negotiate better supplier terms'
                )

    def _analyze_competition(self):
        """Analyze competitive position."""
        if 'unit_price' in self.data.columns and 'comp_1' in self.data.columns:
            price_diff = self.data['unit_price'] - self.data['comp_1']
            avg_diff = price_diff.mean()

            if avg_diff > 10:
                self._add_recommendation(
                    category='Competition',
                    priority='medium',
                    title='Premium Pricing Position',
                    description=f'Average price is ${avg_diff:.2f} above competitor.',
                    impact='May lose price-sensitive customers',
                    action='Ensure value proposition justifies premium or consider competitive pricing'
                )
            elif avg_diff < -10:
                self._add_recommendation(
                    category='Competition',
                    priority='medium',
                    title='Budget Pricing Position',
                    description=f'Average price is ${abs(avg_diff):.2f} below competitor.',
                    impact='Potential margin left on table',
                    action='Consider gradual price increase to capture more value'
                )

    def _analyze_performance(self):
        """Analyze product performance."""
        if 'product_category_name' in self.data.columns and 'total_price' in self.data.columns:
            cat_revenue = self.data.groupby('product_category_name')['total_price'].sum()
            total_revenue = cat_revenue.sum()

            # Pareto analysis
            top_3_share = cat_revenue.nlargest(3).sum() / total_revenue

            if top_3_share > 0.8:
                self._add_recommendation(
                    category='Performance',
                    priority='high',
                    title='High Revenue Concentration',
                    description=f'Top 3 categories account for {top_3_share*100:.1f}% of revenue.',
                    impact='Business risk if top categories decline',
                    action='Diversify product portfolio and invest in developing categories'
                )

            # Underperforming categories
            avg_revenue = cat_revenue.mean()
            underperforming = cat_revenue[cat_revenue < avg_revenue * 0.5]

            if len(underperforming) > 0:
                self._add_recommendation(
                    category='Performance',
                    priority='low',
                    title='Underperforming Categories',
                    description=f'{len(underperforming)} categories are significantly below average.',
                    impact='Inefficient resource allocation',
                    action='Evaluate discontinuation or revitalization strategies'
                )

    def _analyze_inventory(self):
        """Analyze inventory patterns."""
        if 'qty' in self.data.columns:
            avg_qty = self.data['qty'].mean()
            low_qty_products = (self.data['qty'] < avg_qty * 0.3).sum()

            if low_qty_products > 0:
                self._add_recommendation(
                    category='Inventory',
                    priority='medium',
                    title='Low Stock Products',
                    description=f'{low_qty_products} products have critically low stock levels.',
                    impact='Risk of stockouts and lost sales',
                    action='Review inventory management and reorder points'
                )

    def _analyze_quality(self):
        """Analyze product quality."""
        if 'product_score' in self.data.columns:
            low_score_threshold = 3.0
            low_score_products = (self.data['product_score'] < low_score_threshold).sum()
            low_score_pct = low_score_products / len(self.data) * 100

            if low_score_pct > 5:
                self._add_recommendation(
                    category='Quality',
                    priority='high',
                    title='Low Quality Score Products',
                    description=f'{low_score_pct:.1f}% of products have scores below {low_score_threshold}.',
                    impact='Customer satisfaction and reputation risk',
                    action='Investigate quality issues and consider product improvements or removal'
                )

    def get_recommendations_by_priority(self, priority: str) -> List[Recommendation]:
        """Get recommendations filtered by priority."""
        return [r for r in self.recommendations if r.priority == priority]

    def get_recommendations_by_category(self, category: str) -> List[Recommendation]:
        """Get recommendations filtered by category."""
        return [r for r in self.recommendations if r.category == category]

    def generate_action_plan(self) -> Dict:
        """
        Generate prioritized action plan.

        Returns:
            Dictionary with action plan
        """
        high_priority = self.get_recommendations_by_priority('high')
        medium_priority = self.get_recommendations_by_priority('medium')
        low_priority = self.get_recommendations_by_priority('low')

        return {
            'immediate_actions': [
                {
                    'title': r.title,
                    'action': r.action,
                    'impact': r.impact
                }
                for r in high_priority
            ],
            'short_term_actions': [
                {
                    'title': r.title,
                    'action': r.action,
                    'impact': r.impact
                }
                for r in medium_priority
            ],
            'long_term_actions': [
                {
                    'title': r.title,
                    'action': r.action,
                    'impact': r.impact
                }
                for r in low_priority
            ],
            'summary': {
                'total_recommendations': len(self.recommendations),
                'high_priority_count': len(high_priority),
                'medium_priority_count': len(medium_priority),
                'low_priority_count': len(low_priority)
            }
        }

    def export_recommendations(self, format: str = 'dataframe') -> any:
        """
        Export recommendations in specified format.

        Args:
            format: Export format ('dataframe', 'dict', 'list')

        Returns:
            Recommendations in specified format
        """
        if format == 'dataframe':
            return pd.DataFrame([
                {
                    'Category': r.category,
                    'Priority': r.priority,
                    'Title': r.title,
                    'Description': r.description,
                    'Impact': r.impact,
                    'Action': r.action
                }
                for r in self.recommendations
            ])
        elif format == 'dict':
            return [
                {
                    'category': r.category,
                    'priority': r.priority,
                    'title': r.title,
                    'description': r.description,
                    'impact': r.impact,
                    'action': r.action
                }
                for r in self.recommendations
            ]
        else:
            return self.recommendations
