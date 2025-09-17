"""
Retail Price Optimization Dashboard

A comprehensive Streamlit application for analyzing retail prices and training
machine learning models to predict optimal pricing strategies.

Author: Your Name
Date: 2024
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import plotly.io as pio
import warnings
warnings.filterwarnings('ignore')

# Configuration
pio.templates.default = "plotly_white"

@st.cache_data
def load_data():
    """Load and cache the retail price dataset."""
    try:
        data = pd.read_csv('retail_price.csv')
        return data
    except FileNotFoundError:
        st.error("❌ Error: retail_price.csv file not found!")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()

def setup_page_config():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Retail Price Analytics Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def apply_custom_css():
    """Apply custom CSS styling."""
    st.markdown("""
    <style>
        .main {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            min-height: 100vh;
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
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .chart-container {
            background: rgba(255,255,255,0.95);
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            margin: 1.5rem 0;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .metric-card {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 1.2rem;
            border-radius: 12px;
            color: white;
            text-align: center;
            margin: 1rem 0;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
    </style>
    """, unsafe_allow_html=True)

def create_sidebar(data):
    """Create the sidebar with controls and metrics."""
    with st.sidebar:
        st.markdown('<div class="sidebar-header">🎯 Analysis Control Panel</div>', unsafe_allow_html=True)
        
        # Data Overview
        st.markdown("**📊 Data Overview**")
        total_products = len(data)
        avg_price = data['total_price'].mean()
        max_price = data['total_price'].max()
        min_price = data['total_price'].min()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Products", f"{total_products:,}")
            st.metric("Max Price", f"${max_price:.2f}")
        with col2:
            st.metric("Avg Price", f"${avg_price:.2f}")
            st.metric("Min Price", f"${min_price:.2f}")
        
        # Filters
        st.markdown("**🔍 Data Filters**")
        price_range = st.slider(
            "Price Range ($)",
            min_value=float(data['total_price'].min()),
            max_value=float(data['total_price'].max()),
            value=(float(data['total_price'].min()), float(data['total_price'].max())),
            key='price_filter'
        )
        
        categories = ['All'] + list(data['product_category_name'].unique())
        selected_category = st.selectbox('Product Category', categories, key='category_filter')
        
        qty_range = st.slider(
            "Quantity Range",
            min_value=int(data['qty'].min()),
            max_value=int(data['qty'].max()),
            value=(int(data['qty'].min()), int(data['qty'].max())),
            key='qty_filter'
        )
        
        # Visualization Options
        st.markdown("**📈 Visualization Options**")
        chart_options = [
            'Histogram', 'Box Plot', 'Scatter Plot', 'Bar Chart', 
            'Correlation Heatmap', 'Bar Chart - Price Difference',
            'Time Series Analysis', 'Price Distribution by Category'
        ]
        selected_chart = st.selectbox('Select a visualization:', chart_options, key='chart_selector')
        
        # Model Options
        st.markdown("**🤖 Modeling Options**")
        model_type = st.selectbox('Model Type', ['Decision Tree', 'Random Forest', 'Gradient Boosting'], key='model_type')
        test_size = st.slider('Test Size (%)', 10, 50, 20, key='test_size')
        random_state = st.slider('Random State', 1, 100, 42, key='random_state')
        
        model_button = st.button('🚀 Train Model', key='model_button')
        
        return {
            'price_range': price_range,
            'selected_category': selected_category,
            'qty_range': qty_range,
            'selected_chart': selected_chart,
            'model_type': model_type,
            'test_size': test_size,
            'random_state': random_state,
            'model_button': model_button
        }

def filter_data(data, filters):
    """Apply filters to the dataset."""
    filtered_data = data.copy()
    filtered_data = filtered_data[
        (filtered_data['total_price'] >= filters['price_range'][0]) &
        (filtered_data['total_price'] <= filters['price_range'][1]) &
        (filtered_data['qty'] >= filters['qty_range'][0]) &
        (filtered_data['qty'] <= filters['qty_range'][1])
    ]
    
    if filters['selected_category'] != 'All':
        filtered_data = filtered_data[filtered_data['product_category_name'] == filters['selected_category']]
    
    return filtered_data

def create_visualizations(data, chart_type):
    """Create various visualizations based on selection."""
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    if chart_type == 'Histogram':
        st.header('📊 Distribution of Total Price')
        fig = px.histogram(data, x='total_price', nbins=20, color_discrete_sequence=['#667eea'])
        fig.update_layout(
            title_font_size=20,
            title_font_color='#333',
            plot_bgcolor='rgba(255,255,255,0.9)',
            paper_bgcolor='rgba(255,255,255,0.9)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    elif chart_type == 'Box Plot':
        st.header('📦 Box Plot of Unit Price')
        fig = px.box(data, y='unit_price', color_discrete_sequence=['#764ba2'])
        fig.update_layout(
            title_font_size=20,
            title_font_color='#333',
            plot_bgcolor='rgba(255,255,255,0.9)',
            paper_bgcolor='rgba(255,255,255,0.9)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    elif chart_type == 'Scatter Plot':
        st.header('📈 Quantity vs Total Price with Trendline')
        fig = px.scatter(data, x='qty', y='total_price', trendline='ols', color_discrete_sequence=['#f093fb'])
        fig.update_layout(
            title_font_size=20,
            title_font_color='#333',
            plot_bgcolor='rgba(255,255,255,0.9)',
            paper_bgcolor='rgba(255,255,255,0.9)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    elif chart_type == 'Correlation Heatmap':
        st.header('🔥 Correlation Heatmap of Numerical Features')
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
        correlation_matrix = data[numeric_columns].corr()
        
        fig = go.Figure(go.Heatmap(
            x=correlation_matrix.columns, 
            y=correlation_matrix.columns, 
            z=correlation_matrix.values,
            colorscale='Viridis'
        ))
        fig.update_layout(
            title_font_size=20,
            title_font_color='#333',
            plot_bgcolor='rgba(255,255,255,0.9)',
            paper_bgcolor='rgba(255,255,255,0.9)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    elif chart_type == 'Time Series Analysis':
        st.header('📅 Price Trends Over Time')
        data['month_year'] = pd.to_datetime(data['month_year'], format='%d-%m-%Y')
        monthly_avg = data.groupby('month_year')['total_price'].mean().reset_index()
        
        fig = px.line(monthly_avg, x='month_year', y='total_price', 
                     title='Average Price Over Time', color_discrete_sequence=['#4CAF50'])
        fig.update_layout(
            title_font_size=20,
            title_font_color='#333',
            plot_bgcolor='rgba(255,255,255,0.9)',
            paper_bgcolor='rgba(255,255,255,0.9)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def train_model(data, model_type, test_size, random_state):
    """Train machine learning model and display results."""
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.header('🤖 Model Performance Analysis')
    
    with st.spinner('Training model...'):
        # Prepare data
        data['comp_price_diff'] = data['unit_price'] - data['comp_1']
        
        # Feature selection
        feature_columns = ['qty', 'unit_price', 'comp_1', 'product_score', 'comp_price_diff']
        X = data[feature_columns]
        y = data['total_price']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=random_state
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        if model_type == 'Decision Tree':
            model = DecisionTreeRegressor(random_state=random_state)
        elif model_type == 'Random Forest':
            model = RandomForestRegressor(n_estimators=100, random_state=random_state)
        elif model_type == 'Gradient Boosting':
            model = GradientBoostingRegressor(random_state=random_state)
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Display predictions vs actual
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_test, y=y_pred, 
            mode='markers', 
            marker=dict(color='#667eea', size=8), 
            name='Predicted vs. Actual'
        ))
        fig.add_trace(go.Scatter(
            x=[min(y_test), max(y_test)], 
            y=[min(y_test), max(y_test)], 
            mode='lines', 
            marker=dict(color='#f5576c'), 
            name='Perfect Prediction'
        ))
        fig.update_layout(
            title=f'{model_type} - Predicted vs Actual Retail Price',
            xaxis_title='Actual Retail Price',
            yaxis_title='Predicted Retail Price',
            title_font_size=20,
            title_font_color='#333',
            plot_bgcolor='rgba(255,255,255,0.9)',
            paper_bgcolor='rgba(255,255,255,0.9)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R² Score", f"{r2:.3f}")
        with col2:
            st.metric("RMSE", f"{rmse:.2f}")
        with col3:
            st.metric("MAE", f"{mae:.2f}")
        with col4:
            st.metric("MSE", f"{mse:.2f}")
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            st.subheader('📊 Feature Importance')
            feature_importance = pd.DataFrame({
                'Feature': feature_columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(feature_importance, x='Feature', y='Importance', 
                        color='Importance', color_continuous_scale='Viridis')
            fig.update_layout(
                title_font_size=16,
                title_font_color='#333',
                plot_bgcolor='rgba(255,255,255,0.9)',
                paper_bgcolor='rgba(255,255,255,0.9)'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main application function."""
    setup_page_config()
    apply_custom_css()
    
    # Load data
    data = load_data()
    
    # Header
    st.markdown('<div class="main-header">📊 Retail Price Analytics Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar
    filters = create_sidebar(data)
    
    # Filter data
    filtered_data = filter_data(data, filters)
    
    # Create visualizations
    create_visualizations(filtered_data, filters['selected_chart'])
    
    # Train model if requested
    if filters['model_button']:
        train_model(filtered_data, filters['model_type'], filters['test_size'], filters['random_state'])

if __name__ == "__main__":
    main()