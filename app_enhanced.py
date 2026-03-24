"""
Enhanced Streamlit Application with All New Features.

This is the main entry point for the enhanced Retail Price Optimization Dashboard.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import io
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
from models.price_predictor import PricePredictor, ModelTrainer
from models.model_evaluator import ModelEvaluator
from models.feature_importance import FeatureImportanceAnalyzer
from utils.price_optimizer import PriceOptimizer, PricingStrategy
from utils.data_preprocessing import DataPreprocessor
from utils.analytics import AnalyticsEngine
from utils.visualizations import ChartBuilder
from utils.dashboard_utils import DashboardComponents, StateManager

# Set page config
st.set_page_config(
    page_title="Retail Price Optimization Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and cache the retail price data."""
    data = pd.read_csv('retail_price.csv')
    data['comp_price_diff'] = data['unit_price'] - data['comp_1']
    return data


def main():
    """Main application function."""
    # Load data
    data = load_data()

    # Header
    st.markdown('<div class="main-header">📊 Retail Price Optimization Dashboard</div>',
               unsafe_allow_html=True)

    # Navigation
    page = st.sidebar.radio(
        "Navigate",
        ["📈 Dashboard", "🤖 ML Models", "💰 Price Optimization", "📊 Analytics", "⚙️ Settings"]
    )

    if page == "📈 Dashboard":
        show_dashboard(data)
    elif page == "🤖 ML Models":
        show_ml_models(data)
    elif page == "💰 Price Optimization":
        show_optimization(data)
    elif page == "📊 Analytics":
        show_analytics(data)
    elif page == "⚙️ Settings":
        show_settings()


def show_dashboard(data):
    """Display main dashboard."""
    st.header("📊 Overview Dashboard")

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💰 Total Revenue", f"${data['total_price'].sum():,.2f}")
    with col2:
        st.metric("📦 Total Products", data['product_id'].nunique())
    with col3:
        st.metric("📁 Categories", data['product_category_name'].nunique())
    with col4:
        st.metric("⭐ Avg Score", f"{data['product_score'].mean():.2f}")

    st.markdown("---")

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Revenue by Category")
        revenue_by_cat = data.groupby('product_category_name')['total_price'].sum().reset_index()
        fig = px.bar(revenue_by_cat, x='product_category_name', y='total_price',
                    color='total_price', color_continuous_scale='Viridis')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🥧 Revenue Distribution")
        fig = px.pie(revenue_by_cat, values='total_price', names='product_category_name',
                    hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    # Price Prediction Calculator
    st.markdown("---")
    st.subheader("🎯 Quick Price Prediction")

    col1, col2 = st.columns(2)

    with col1:
        qty = st.slider("Quantity", 1, 100, 50)
        unit_price = st.number_input("Unit Price", 0.01, 1000.0, 50.0)
        comp_price = st.number_input("Competitor Price", 0.01, 1000.0, 45.0)

    with col2:
        product_score = st.slider("Product Score", 1.0, 5.0, 4.0)

        if st.button("🔮 Predict Price", use_container_width=True):
            # Train model and predict
            predictor = PricePredictor(model_type='random_forest')
            predictor.fit(data)

            predicted = predictor.predict_single(qty, unit_price, comp_price, product_score)

            st.success(f"Predicted Total Price: ${predicted:.2f}")
            st.info(f"Price Range: ${predicted * 0.85:.2f} - ${predicted * 1.15:.2f}")


def show_ml_models(data):
    """Display ML models page."""
    st.header("🤖 Machine Learning Models")

    # Model selection
    model_type = st.selectbox(
        "Select Model",
        ["Random Forest", "Gradient Boosting", "Decision Tree", "Linear Regression"]
    )

    # Model parameters
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test Size (%)", 10, 50, 20)
    with col2:
        random_state = st.slider("Random State", 1, 100, 42)

    if st.button("🚀 Train Model", use_container_width=True):
        with st.spinner("Training model..."):
            # Prepare data
            X = data[['qty', 'unit_price', 'comp_1', 'product_score', 'comp_price_diff']]
            y = data['total_price']

            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size/100, random_state=random_state
            )

            # Train model
            model_map = {
                "Random Forest": "random_forest",
                "Gradient Boosting": "gradient_boosting",
                "Decision Tree": "decision_tree",
                "Linear Regression": "linear"
            }

            predictor = PricePredictor(model_type=model_map[model_type])
            predictor.fit(data)

            # Evaluate
            y_pred = predictor.predict(data.loc[X_test.index])

            evaluator = ModelEvaluator()
            metrics = evaluator.evaluate_model(y_test.values, y_pred, model_type)

            # Display results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R² Score", f"{metrics['r2']:.4f}")
            with col2:
                st.metric("RMSE", f"${metrics['rmse']:.2f}")
            with col3:
                st.metric("MAE", f"${metrics['mae']:.2f}")

            # Show evaluation plot
            st.subheader("📊 Model Evaluation")
            fig = evaluator.create_evaluation_plot(y_test.values, y_pred, model_type)
            st.plotly_chart(fig, use_container_width=True)

            # Feature importance
            st.subheader("🔍 Feature Importance")
            importance = predictor.get_feature_importance()
            if importance is not None:
                fig = px.bar(importance, x='importance', y='feature', orientation='h')
                st.plotly_chart(fig, use_container_width=True)


def show_optimization(data):
    """Display price optimization page."""
    st.header("💰 Price Optimization")

    # Product selection
    products = data['product_id'].unique()
    selected_product = st.selectbox("Select Product", products)

    # Strategy selection
    strategy = st.selectbox(
        "Pricing Strategy",
        ["Competitor Based", "Cost Plus", "Value Based", "Dynamic", "Penetration", "Skimming"]
    )

    if st.button("🎯 Optimize Price", use_container_width=True):
        with st.spinner("Optimizing..."):
            optimizer = PriceOptimizer(data)

            strategy_map = {
                "Competitor Based": PricingStrategy.COMPETITOR_BASED,
                "Cost Plus": PricingStrategy.COST_PLUS,
                "Value Based": PricingStrategy.VALUE_BASED,
                "Dynamic": PricingStrategy.DYNAMIC,
                "Penetration": PricingStrategy.PENETRATION,
                "Skimming": PricingStrategy.SKIMMING
            }

            result = optimizer.optimize_price(
                selected_product,
                strategy_map[strategy]
            )

            # Display results
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Optimal Price", f"${result.optimal_price:.2f}")
                st.metric("Expected Revenue", f"${result.expected_revenue:.2f}")

            with col2:
                st.metric("Expected Profit", f"${result.expected_profit:.2f}")
                st.metric("Confidence", f"{result.confidence_score:.1f}%")

            # Recommendations
            st.subheader("💡 Recommendations")
            for rec in result.recommendations:
                st.markdown(f"- {rec}")

            # Price elasticity
            st.subheader("📈 Price Elasticity Analysis")
            elasticity = optimizer.analyze_price_elasticity(selected_product)
            st.write(f"**Elasticity:** {elasticity.get('elasticity', 'N/A')}")
            st.write(f"**Interpretation:** {elasticity.get('interpretation', 'N/A')}")
            st.write(f"**Recommendation:** {elasticity.get('recommendation', 'N/A')}")


def show_analytics(data):
    """Display analytics page."""
    st.header("📊 Advanced Analytics")

    # Initialize analytics engine
    engine = AnalyticsEngine(data)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Statistics", "📁 Categories", "🏆 Products", "📊 Correlations"
    ])

    with tab1:
        st.subheader("Descriptive Statistics")
        stats = engine.calculate_descriptive_statistics()
        st.dataframe(stats, use_container_width=True)

    with tab2:
        st.subheader("Category Performance")
        cat_perf = engine.analyze_category_performance()
        st.dataframe(cat_perf, use_container_width=True)

    with tab3:
        st.subheader("Product Segmentation")
        segments = engine.segment_products()
        st.dataframe(segments.head(20), use_container_width=True)

    with tab4:
        st.subheader("Correlation Analysis")
        corr_result = engine.analyze_correlations()
        corr_matrix = corr_result['correlation_matrix']

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        st.plotly_chart(fig, use_container_width=True)


def show_settings():
    """Display settings page."""
    st.header("⚙️ Settings")

    st.subheader("Application Info")
    st.info("Retail Price Optimization Dashboard v2.0.0")

    st.subheader("Data Info")
    data = load_data()
    st.write(f"**Records:** {len(data):,}")
    st.write(f"**Products:** {data['product_id'].nunique()}")
    st.write(f"**Categories:** {data['product_category_name'].nunique()}")

    st.subheader("Export Data")
    export_format = st.selectbox("Format", ["CSV", "Excel", "JSON"])

    if st.button("📥 Export Data"):
        if export_format == "CSV":
            csv = data.to_csv(index=False)
            st.download_button("Download CSV", csv, "retail_data.csv", "text/csv")
        elif export_format == "Excel":
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                data.to_excel(writer, index=False)
            st.download_button("Download Excel", output.getvalue(),
                             "retail_data.xlsx",
                             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            json_data = data.to_json(orient='records', indent=2)
            st.download_button("Download JSON", json_data, "retail_data.json", "application/json")


if __name__ == "__main__":
    main()
