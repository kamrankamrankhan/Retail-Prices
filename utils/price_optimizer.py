"""
Price Optimizer Utility Module.

Provides advanced pricing strategies and optimization algorithms.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
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
    PSYCHOLOGICAL = "psychological"
    BUNDLE = "bundle"


@dataclass
class OptimizationResult:
    """Result of price optimization."""
    product_id: str
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
    
    Provides multiple pricing strategies and optimization algorithms
    for retail price management.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize PriceOptimizer.
        
        Args:
            data: DataFrame containing product and pricing data
        """
        self.data = data.copy()

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
        product_data = self.data[self.data['product_id'] == product_id]

        if product_data.empty:
            raise ValueError(f"Product {product_id} not found")

        unit_price = product_data['unit_price'].iloc[0]
        comp_prices = [
            product_data['comp_1'].iloc[0] if 'comp_1' in product_data.columns else unit_price,
            product_data['comp_2'].iloc[0] if 'comp_2' in product_data.columns else unit_price,
            product_data['comp_3'].iloc[0] if 'comp_3' in product_data.columns else unit_price
        ]
        comp_prices = [p for p in comp_prices if not np.isnan(p)]
        
        product_score = product_data['product_score'].iloc[0] if 'product_score' in product_data.columns else 4.0
        qty = product_data['qty'].iloc[0] if 'qty' in product_data.columns else 1

        # Calculate optimal price based on strategy
        if strategy == PricingStrategy.COST_PLUS:
            markup = kwargs.get('markup', 0.3)
            optimal_price = self._cost_plus_pricing(unit_price, markup)
        elif strategy == PricingStrategy.COMPETITOR_BASED:
            position = kwargs.get('position', 'competitive')
            optimal_price = self._competitor_based_pricing(unit_price, comp_prices, position)
        elif strategy == PricingStrategy.VALUE_BASED:
            max_wtp = kwargs.get('max_wtp', unit_price * 2)
            optimal_price = self._value_based_pricing(unit_price, product_score, max_wtp)
        elif strategy == PricingStrategy.DYNAMIC:
            demand_factor = kwargs.get('demand_factor', 1.0)
            inventory_level = kwargs.get('inventory_level', 0.5)
            optimal_price = self._dynamic_pricing(unit_price, demand_factor, inventory_level)
        elif strategy == PricingStrategy.PENETRATION:
            market_share_target = kwargs.get('market_share_target', 0.1)
            optimal_price = self._penetration_pricing(unit_price, market_share_target)
        elif strategy == PricingStrategy.SKIMMING:
            market_maturity = kwargs.get('market_maturity', 0.5)
            optimal_price = self._skimming_pricing(unit_price, product_score, market_maturity)
        else:
            optimal_price = unit_price

        # Calculate expected revenue and profit
        expected_revenue = optimal_price * qty
        cost = unit_price * 0.7  # Assuming 70% cost
        expected_profit = (optimal_price - cost) * qty

        # Calculate confidence score
        confidence = min(95, max(60, 70 + (product_score - 5) * 3))

        # Generate recommendations
        recommendations = self._generate_recommendations(
            optimal_price, unit_price, comp_prices, strategy
        )

        return OptimizationResult(
            product_id=product_id,
            optimal_price=round(optimal_price, 2),
            expected_revenue=round(expected_revenue, 2),
            expected_profit=round(expected_profit, 2),
            confidence_score=round(confidence, 2),
            strategy_used=strategy.value,
            price_range=(round(optimal_price * 0.9, 2), round(optimal_price * 1.1, 2)),
            recommendations=recommendations
        )

    def _cost_plus_pricing(self, unit_price: float, markup: float) -> float:
        """Calculate cost-plus price."""
        return unit_price * (1 + markup)

    def _competitor_based_pricing(self, unit_price: float, comp_prices: List[float],
                                   position: str) -> float:
        """Calculate competitor-based price."""
        if not comp_prices:
            return unit_price
        
        avg_comp_price = np.mean(comp_prices)
        
        if position == 'budget':
            return avg_comp_price * 0.9
        elif position == 'premium':
            return avg_comp_price * 1.15
        else:  # competitive
            return avg_comp_price

    def _value_based_pricing(self, unit_price: float, product_score: float,
                              max_wtp: float) -> float:
        """Calculate value-based price."""
        value_factor = product_score / 10.0
        price_range = max_wtp - unit_price
        value_premium = price_range * value_factor
        return unit_price + value_premium

    def _dynamic_pricing(self, base_price: float, demand_factor: float,
                         inventory_level: float) -> float:
        """Calculate dynamic price."""
        demand_adjustment = 1 + (demand_factor - 1) * 0.2
        inventory_adjustment = 1 + (1 - inventory_level) * 0.1
        return base_price * demand_adjustment * inventory_adjustment

    def _penetration_pricing(self, unit_price: float, market_share_target: float) -> float:
        """Calculate penetration price."""
        discount_factor = 1 - (market_share_target * 0.5)
        return unit_price * discount_factor

    def _skimming_pricing(self, unit_price: float, product_score: float,
                          market_maturity: float) -> float:
        """Calculate skimming price."""
        quality_premium = (product_score / 10) * 0.3
        maturity_discount = market_maturity * 0.2
        return unit_price * (1 + quality_premium - maturity_discount)

    def _generate_recommendations(self, optimal_price: float, current_price: float,
                                   comp_prices: List[float], strategy: PricingStrategy) -> List[str]:
        """Generate pricing recommendations."""
        recommendations = []
        
        if optimal_price > current_price * 1.1:
            recommendations.append("Consider gradual price increase")
        elif optimal_price < current_price * 0.9:
            recommendations.append("Lower price may help gain market share")
        
        if comp_prices:
            avg_comp = np.mean(comp_prices)
            if optimal_price > avg_comp:
                recommendations.append("Price is above market average")
            else:
                recommendations.append("Competitive pricing position")
        
        return recommendations

    def analyze_price_elasticity(self, product_id: str) -> Dict:
        """Analyze price elasticity for a product."""
        product_data = self.data[self.data['product_id'] == product_id]
        
        if len(product_data) < 3:
            return {'elasticity': 0, 'interpretation': 'insufficient_data'}
        
        prices = product_data['unit_price'].values
        quantities = product_data['qty'].values if 'qty' in product_data.columns else np.ones(len(prices))
        
        price_changes = np.diff(prices) / prices[:-1]
        qty_changes = np.diff(quantities) / quantities[:-1]
        
        valid_indices = (price_changes != 0) & (qty_changes != 0)
        if not valid_indices.any():
            return {'elasticity': 0, 'interpretation': 'no_variation'}
        
        elasticity = np.mean(qty_changes[valid_indices] / price_changes[valid_indices])
        
        if elasticity < -1:
            interpretation = 'elastic'
        elif elasticity > -1:
            interpretation = 'inelastic'
        else:
            interpretation = 'unit_elastic'
        
        return {
            'elasticity': round(elasticity, 3),
            'interpretation': interpretation
        }

    def batch_optimization(self, product_ids: List[str],
                           strategy: PricingStrategy, **kwargs) -> pd.DataFrame:
        """Optimize prices for multiple products."""
        results = []
        
        for product_id in product_ids:
            try:
                result = self.optimize_price(product_id, strategy, **kwargs)
                results.append({
                    'product_id': product_id,
                    'optimal_price': result.optimal_price,
                    'expected_revenue': result.expected_revenue,
                    'expected_profit': result.expected_profit,
                    'confidence': result.confidence_score
                })
            except Exception as e:
                logger.error(f"Error optimizing {product_id}: {str(e)}")
        
        return pd.DataFrame(results)

    def calculate_optimal_markup(self, product_id: str, target_margin: float = 0.3) -> Dict:
        """Calculate optimal markup for target margin."""
        product_data = self.data[self.data['product_id'] == product_id]
        
        if product_data.empty:
            raise ValueError(f"Product {product_id} not found")
        
        unit_price = product_data['unit_price'].iloc[0]
        cost = unit_price * 0.7  # Assuming 70% cost
        
        required_markup = target_margin / (1 - target_margin)
        optimal_price = cost * (1 + required_markup)
        
        return {
            'current_price': unit_price,
            'estimated_cost': round(cost, 2),
            'required_markup': round(required_markup * 100, 2),
            'optimal_price': round(optimal_price, 2),
            'expected_margin': round(target_margin * 100, 2)
        }
