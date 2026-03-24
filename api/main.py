"""
API Module for Retail Price Optimization.

This module provides RESTful API endpoints for price prediction
and optimization services.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Retail Price Optimization API",
    description="API for retail price prediction and optimization",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class PricePredictionRequest(BaseModel):
    """Request model for price prediction."""
    qty: float
    unit_price: float
    comp_1: float
    product_score: float
    comp_price_diff: Optional[float] = None


class PricePredictionResponse(BaseModel):
    """Response model for price prediction."""
    predicted_price: float
    confidence: float
    price_range: Dict[str, float]


class OptimizationRequest(BaseModel):
    """Request model for price optimization."""
    product_id: str
    strategy: str = "competitor_based"
    markup: Optional[float] = 0.3
    position: Optional[str] = "competitive"


class OptimizationResponse(BaseModel):
    """Response model for price optimization."""
    product_id: str
    optimal_price: float
    expected_revenue: float
    expected_profit: float
    confidence: float
    strategy: str
    recommendations: List[str]


class BatchOptimizationRequest(BaseModel):
    """Request model for batch optimization."""
    product_ids: List[str]
    strategy: str = "competitor_based"


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    version: str


# Global variables for model and data
_model = None
_data = None


def load_model():
    """Load the trained model."""
    global _model
    if _model is None:
        # Import and initialize model
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from models.price_predictor import PricePredictor
        _model = PricePredictor(model_type='random_forest')
    return _model


def load_data():
    """Load the retail price data."""
    global _data
    if _data is None:
        import os
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'retail_price.csv')
        _data = pd.read_csv(data_path)
        _data['comp_price_diff'] = _data['unit_price'] - _data['comp_1']
    return _data


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info("Starting Retail Price Optimization API...")
    load_data()
    logger.info("Data loaded successfully")


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with API info."""
    return HealthResponse(status="running", version="1.0.0")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="1.0.0")


@app.post("/predict", response_model=PricePredictionResponse)
async def predict_price(request: PricePredictionRequest):
    """
    Predict price based on product parameters.

    Args:
        request: PricePredictionRequest with product parameters

    Returns:
        PricePredictionResponse with predicted price
    """
    try:
        model = load_model()
        data = load_data()

        # Train model if not fitted
        if not model.is_fitted:
            model.fit(data)

        # Prepare input
        comp_diff = request.comp_price_diff or (request.unit_price - request.comp_1)
        predicted = model.predict_single(
            qty=request.qty,
            unit_price=request.unit_price,
            comp_price=request.comp_1,
            product_score=request.product_score
        )

        # Calculate confidence
        confidence = min(95, max(60, 70 + (request.product_score - 5) * 3))

        return PricePredictionResponse(
            predicted_price=round(predicted, 2),
            confidence=round(confidence, 1),
            price_range={
                "low": round(predicted * 0.85, 2),
                "high": round(predicted * 1.15, 2)
            }
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimize", response_model=OptimizationResponse)
async def optimize_price(request: OptimizationRequest):
    """
    Optimize price for a specific product.

    Args:
        request: OptimizationRequest with product ID and strategy

    Returns:
        OptimizationResponse with optimal price
    """
    try:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils.price_optimizer import PriceOptimizer, PricingStrategy

        data = load_data()
        optimizer = PriceOptimizer(data)

        # Map strategy string to enum
        strategy_map = {
            'cost_plus': PricingStrategy.COST_PLUS,
            'competitor_based': PricingStrategy.COMPETITOR_BASED,
            'value_based': PricingStrategy.VALUE_BASED,
            'dynamic': PricingStrategy.DYNAMIC,
            'penetration': PricingStrategy.PENETRATION,
            'skimming': PricingStrategy.SKIMMING
        }

        strategy = strategy_map.get(request.strategy, PricingStrategy.COMPETITOR_BASED)

        result = optimizer.optimize_price(
            product_id=request.product_id,
            strategy=strategy,
            markup=request.markup,
            position=request.position
        )

        return OptimizationResponse(
            product_id=request.product_id,
            optimal_price=result.optimal_price,
            expected_revenue=result.expected_revenue,
            expected_profit=result.expected_profit,
            confidence=result.confidence_score,
            strategy=result.strategy_used,
            recommendations=result.recommendations
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Optimization error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimize/batch")
async def batch_optimize(request: BatchOptimizationRequest):
    """
    Optimize prices for multiple products.

    Args:
        request: BatchOptimizationRequest with product IDs

    Returns:
        List of optimization results
    """
    try:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils.price_optimizer import PriceOptimizer, PricingStrategy

        data = load_data()
        optimizer = PriceOptimizer(data)

        strategy_map = {
            'cost_plus': PricingStrategy.COST_PLUS,
            'competitor_based': PricingStrategy.COMPETITOR_BASED,
            'value_based': PricingStrategy.VALUE_BASED,
            'dynamic': PricingStrategy.DYNAMIC
        }

        strategy = strategy_map.get(request.strategy, PricingStrategy.COMPETITOR_BASED)

        results = optimizer.batch_optimize(request.product_ids, strategy)

        return results.to_dict('records')

    except Exception as e:
        logger.error(f"Batch optimization error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/products")
async def list_products(
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: int = Query(50, ge=1, le=500, description="Number of products to return")
):
    """
    List products with optional filtering.

    Args:
        category: Optional category filter
        limit: Maximum number of products to return

    Returns:
        List of products
    """
    try:
        data = load_data()

        if category:
            data = data[data['product_category_name'] == category]

        products = data.head(limit).to_dict('records')

        return {
            "total": len(data),
            "returned": len(products),
            "products": products
        }

    except Exception as e:
        logger.error(f"Error listing products: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/categories")
async def list_categories():
    """List all product categories."""
    try:
        data = load_data()
        categories = data['product_category_name'].unique().tolist()

        return {
            "total": len(categories),
            "categories": categories
        }

    except Exception as e:
        logger.error(f"Error listing categories: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/summary")
async def get_analytics_summary():
    """Get analytics summary."""
    try:
        data = load_data()

        summary = {
            "total_records": len(data),
            "total_products": data['product_id'].nunique(),
            "total_categories": data['product_category_name'].nunique(),
            "total_revenue": float(data['total_price'].sum()),
            "avg_price": float(data['unit_price'].mean()),
            "avg_quantity": float(data['qty'].mean()),
            "avg_product_score": float(data['product_score'].mean())
        }

        return summary

    except Exception as e:
        logger.error(f"Error getting analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
