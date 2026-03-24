"""
Price Optimization Engine Module.

This module provides sophisticated price optimization algorithms and strategies
for retail price management including dynamic pricing, competitor-based pricing,
and demand-based optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PricingStrategy(Enum):
    """Available pricing strategies."""
    COST_PLUS = "cost_plus"
    COMPETITOR_BASED = "competitor_based"
    VALUE_BASED = "value_based"
    DYNAMIC = "dynamic"
    PENETRATION = "penetration"
    SKIMMING = "skimming"
    BUNDLE = "bundle"
    PSYCHOLOGICAL = "psychological"


@dataclass
class OptimizationResult:
    """Result of price optimization."""
    optimal_price: float
    expected_revenue: float
    expected_profit: float
    confidence_score: float
    strategy_used: str
    price_range: Tuple[float, float]
    recommendations: List[str]


class PriceOptimizer:
    """
    Advanced price optimization engine.

    This class provides multiple pricing strategies and optimization algorithms
    to determine optimal prices for retail products.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the PriceOptimizer.

        Args:
            data: DataFrame containing product and pricing data
        """
        self.data = data
        self.strategies = {}
        self._register_default_strategies()

    def _register_default_strategies(self):
        """Register default pricing strategies."""
        self.strategies = {
            PricingStrategy.COST_PLUS: self._cost_plus_pricing,
            PricingStrategy.COMPETITOR_BASED: self._competitor_based_pricing,
            PricingStrategy.VALUE_BASED: self._value_based_pricing,
            PricingStrategy.DYNAMIC: self._dynamic_pricing,
            PricingStrategy.PENETRATION: self._penetration_pricing,
            PricingStrategy.SKIMMING: self._skimming_pricing,
        }

    def _cost_plus_pricing(self, unit_price: float, markup: float = 0.3) -> float:
        """
        Calculate price using cost-plus strategy.

        Args:
            unit_price: Base cost/price
            markup: Markup percentage

        Returns:
            Optimized price
        """
        return unit_price * (1 + markup)

    def _competitor_based_pricing(self, unit_price: float, comp_prices: List[float],
                                   position: str = 'competitive') -> float:
        """
        Calculate price based on competitor prices.

        Args:
            unit_price: Our unit price
            comp_prices: List of competitor prices
            position: Pricing position ('budget', 'competitive', 'premium')

        Returns:
            Optimized price
        """
        avg_comp_price = np.mean(comp_prices)

        if position == 'budget':
            return avg_comp_price * 0.9
        elif position == 'premium':
            return avg_comp_price * 1.15
        else:  # competitive
            return avg_comp_price

    def _value_based_pricing(self, unit_price: float, product_score: float,
                              max_willingness_to_pay: float) -> float:
        """
        Calculate price based on perceived value.

        Args:
            unit_price: Base unit price
            product_score: Product quality score (1-10)
            max_willingness_to_pay: Maximum price customers will pay

        Returns:
            Optimized price
        """
        # Value factor based on product score
        value_factor = product_score / 10.0

        # Price between unit price and max willingness to pay
        price_range = max_willingness_to_pay - unit_price
        value_premium = price_range * value_factor

        return unit_price + value_premium

    def _dynamic_pricing(self, base_price: float, demand_factor: float,
                         inventory_level: float, seasonality: float = 1.0) -> float:
        """
        Calculate dynamic price based on market conditions.

        Args:
            base_price: Base price
            demand_factor: Demand multiplier (>1 high demand, <1 low demand)
            inventory_level: Current inventory level (0-1)
            seasonality: Seasonal adjustment factor

        Returns:
            Dynamic price
        """
        # Adjust for demand
        demand_adjustment = 1 + (demand_factor - 1) * 0.2

        # Adjust for inventory (lower inventory = higher price)
        inventory_adjustment = 1 + (1 - inventory_level) * 0.1

        # Apply seasonality
        dynamic_price = base_price * demand_adjustment * inventory_adjustment * seasonality

        return dynamic_price

    def _penetration_pricing(self, unit_price: float, market_share_target: float = 0.1) -> float:
        """
        Calculate penetration pricing for market entry.

        Args:
            unit_price: Base unit price
            market_share_target: Target market share percentage

        Returns:
            Penetration price
        """
        # Lower price to gain market share
        discount_factor = 1 - (market_share_target * 0.5)
        return unit_price * discount_factor

    def _skimming_pricing(self, unit_price: float, product_score: float,
                          market_maturity: float = 0.5) -> float:
        """
        Calculate skimming price for premium products.

        Args:
            unit_price: Base unit price
            product_score: Product quality score
            market_maturity: Market maturity level (0=new, 1=mature)

        Returns:
            Skimming price
        """
        # Premium for high-quality, new products
        quality_premium = (product_score / 10) * 0.3
        maturity_discount = market_maturity * 0.2

        return unit_price * (1 + quality_premium - maturity_discount)

    def optimize_price(self, product_id: str, strategy: PricingStrategy,
                       **kwargs) -> OptimizationResult:
        """
        Optimize price for a specific product.

        Args:
            product_id: Product identifier
            strategy: Pricing strategy to use
            **kwargs: Additional parameters for the strategy

        Returns:
            OptimizationResult with optimal price and details
        """
        # Get product data
        product_data = self.data[self.data['product_id'] == product_id]

        if product_data.empty:
            raise ValueError(f"Product {product_id} not found")

        unit_price = product_data['unit_price'].iloc[0]
        comp_prices = [
            product_data['comp_1'].iloc[0],
            product_data['comp_2'].iloc[0],
            product_data['comp_3'].iloc[0]
        ]
        comp_prices = [p for p in comp_prices if not np.isnan(p)]

        product_score = product_data['product_score'].iloc[0]
        qty = product_data['qty'].iloc[0]

        # Calculate optimal price based on strategy
        if strategy == PricingStrategy.COST_PLUS:
            optimal_price = self._cost_plus_pricing(unit_price, kwargs.get('markup', 0.3))
        elif strategy == PricingStrategy.COMPETITOR_BASED:
            optimal_price = self._competitor_based_pricing(
                unit_price, comp_prices, kwargs.get('position', 'competitive')
            )
        elif strategy == PricingStrategy.VALUE_BASED:
            optimal_price = self._value_based_pricing(
                unit_price, product_score, kwargs.get('max_wtp', unit_price * 2)
            )
        elif strategy == PricingStrategy.DYNAMIC:
            optimal_price = self._dynamic_pricing(
                unit_price,
                kwargs.get('demand_factor', 1.0),
                kwargs.get('inventory_level', 0.5),
                kwargs.get('seasonality', 1.0)
            )
        elif strategy == PricingStrategy.PENETRATION:
            optimal_price = self._penetration_pricing(
                unit_price, kwargs.get('market_share_target', 0.1)
            )
        elif strategy == PricingStrategy.SKIMMING:
            optimal_price = self._skimming_pricing(
                unit_price, product_score, kwargs.get('market_maturity', 0.5)
            )
        else:
            optimal_price = unit_price

        # Calculate expected revenue and profit
        expected_revenue = optimal_price * qty
        expected_profit = (optimal_price - unit_price * 0.7) * qty  # Assuming 70% cost

        # Calculate confidence score
        confidence = self._calculate_confidence(optimal_price, comp_prices, product_score)

        # Generate price range
        price_range = (optimal_price * 0.9, optimal_price * 1.1)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            optimal_price, unit_price, comp_prices, strategy
        )

        return OptimizationResult(
            optimal_price=round(optimal_price, 2),
            expected_revenue=round(expected_revenue, 2),
            expected_profit=round(expected_profit, 2),
            confidence_score=round(confidence, 2),
            strategy_used=strategy.value,
            price_range=(round(price_range[0], 2), round(price_range[1], 2)),
            recommendations=recommendations
        )

    def _calculate_confidence(self, optimal_price: float, comp_prices: List[float],
                              product_score: float) -> float:
        """Calculate confidence score for the optimal price."""
        confidence = 70.0  # Base confidence

        # Adjust for competition data availability
        if len(comp_prices) >= 3:
            confidence += 10
        elif len(comp_prices) >= 1:
            confidence += 5

        # Adjust for price competitiveness
        if comp_prices:
            avg_comp = np.mean(comp_prices)
            if optimal_price < avg_comp * 0.95:
                confidence += 5  # Competitive price
            elif optimal_price > avg_comp * 1.1:
                confidence -= 5  # Premium price might be risky

        # Adjust for product score
        if product_score >= 4.0:
            confidence += 5

        return min(100, max(0, confidence))

    def _generate_recommendations(self, optimal_price: float, current_price: float,
                                   comp_prices: List[float], strategy: PricingStrategy) -> List[str]:
        """Generate pricing recommendations."""
        recommendations = []

        # Price change recommendation
        if optimal_price > current_price * 1.1:
            recommendations.append("Consider gradual price increase to avoid customer backlash")
        elif optimal_price < current_price * 0.9:
            recommendations.append("Lower price may help gain market share")

        # Competition-based recommendations
        if comp_prices:
            avg_comp = np.mean(comp_prices)
            if optimal_price > avg_comp:
                recommendations.append("Price is above market average - ensure value proposition is clear")
            else:
                recommendations.append("Competitive pricing position may attract price-sensitive customers")

        # Strategy-specific recommendations
        if strategy == PricingStrategy.DYNAMIC:
            recommendations.append("Monitor demand and inventory levels closely for adjustments")
        elif strategy == PricingStrategy.PENETRATION:
            recommendations.append("Plan for gradual price increase after gaining market share")

        return recommendations

    def batch_optimize(self, product_ids: List[str],
                       strategy: PricingStrategy = PricingStrategy.COMPETITOR_BASED,
                       **kwargs) -> pd.DataFrame:
        """
        Optimize prices for multiple products.

        Args:
            product_ids: List of product IDs
            strategy: Pricing strategy to use
            **kwargs: Additional parameters

        Returns:
            DataFrame with optimization results
        """
        results = []

        for product_id in product_ids:
            try:
                result = self.optimize_price(product_id, strategy, **kwargs)
                results.append({
                    'product_id': product_id,
                    'optimal_price': result.optimal_price,
                    'expected_revenue': result.expected_revenue,
                    'expected_profit': result.expected_profit,
                    'confidence': result.confidence_score,
                    'strategy': result.strategy_used
                })
            except Exception as e:
                logger.error(f"Error optimizing {product_id}: {str(e)}")

        return pd.DataFrame(results)

    def analyze_price_elasticity(self, product_id: str) -> Dict:
        """
        Analyze price elasticity for a product.

        Args:
            product_id: Product identifier

        Returns:
            Dictionary with elasticity analysis
        """
        product_data = self.data[self.data['product_id'] == product_id]

        if len(product_data) < 3:
            return {'error': 'Insufficient data for elasticity analysis'}

        prices = product_data['unit_price'].values
        quantities = product_data['qty'].values

        # Calculate price elasticity using regression
        price_changes = np.diff(prices) / prices[:-1]
        qty_changes = np.diff(quantities) / quantities[:-1]

        # Avoid division by zero
        valid_indices = (price_changes != 0) & (qty_changes != 0)
        if not valid_indices.any():
            return {'elasticity': 0, 'interpretation': 'inelastic'}

        elasticity = np.mean(qty_changes[valid_indices] / price_changes[valid_indices])

        # Interpret elasticity
        if elasticity < -1:
            interpretation = 'elastic'
        elif elasticity > -1:
            interpretation = 'inelastic'
        else:
            interpretation = 'unit_elastic'

        return {
            'elasticity': round(elasticity, 3),
            'interpretation': interpretation,
            'recommendation': self._elasticity_recommendation(elasticity)
        }

    def _elasticity_recommendation(self, elasticity: float) -> str:
        """Generate recommendation based on elasticity."""
        if elasticity < -2:
            return "Highly elastic - small price decreases can significantly increase sales"
        elif elasticity < -1:
            return "Elastic - consider competitive pricing strategies"
        elif elasticity < -0.5:
            return "Moderately inelastic - price changes have limited impact on demand"
        else:
            return "Inelastic - price increases may not significantly affect demand"

    def calculate_optimal_markup(self, product_id: str,
                                  target_margin: float = 0.3) -> Dict:
        """
        Calculate optimal markup for target profit margin.

        Args:
            product_id: Product identifier
            target_margin: Target profit margin (0-1)

        Returns:
            Dictionary with markup analysis
        """
        product_data = self.data[self.data['product_id'] == product_id]

        if product_data.empty:
            raise ValueError(f"Product {product_id} not found")

        unit_price = product_data['unit_price'].iloc[0]

        # Calculate required markup for target margin
        # Formula: markup = target_margin / (1 - target_margin)
        required_markup = target_margin / (1 - target_margin)

        # Calculate prices
        cost_price = unit_price * 0.7  # Assuming 70% of price is cost
        optimal_price = cost_price * (1 + required_markup)

        return {
            'current_price': unit_price,
            'estimated_cost': round(cost_price, 2),
            'required_markup': round(required_markup * 100, 2),
            'optimal_price': round(optimal_price, 2),
            'expected_margin': round(target_margin * 100, 2)
        }
