import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from datetime import datetime, timedelta
import io

# Set page config for better appearance
st.set_page_config(
    page_title="Retail Price Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Set Plotly template to white background
import plotly.io as pio
pio.templates.default = "plotly_white"

# Custom CSS styling with enhanced background colors
st.markdown("""
<style>
    /* Main background styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        min-height: 100vh;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Sidebar background */
    .css-1d391kg {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
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
    
    .top-nav {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0.8rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: white;
        font-weight: bold;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .nav-item {
        background: rgba(255,255,255,0.2);
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .nav-item:hover {
        background: rgba(255,255,255,0.3);
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
        color: white;
        font-weight: bold;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .sidebar-section {
        background: rgba(255,255,255,0.95);
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .filter-section {
        background: rgba(255,255,255,0.95);
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 5px solid #f093fb;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
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
    
    .chart-container {
        background: rgba(255,255,255,0.95);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 1.5rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .stSelectbox > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.8rem 1.5rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    .status-active {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
    }
    
    .status-inactive {
        background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
    }
    
    /* Enhanced metric styling */
    .stMetric {
        background: rgba(255,255,255,0.9);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #2c3e50;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    /* Text styling */
    .stMarkdown {
        color: #34495e;
    }
</style>
""", unsafe_allow_html=True)

# Load data
data = pd.read_csv('retail_price.csv')

# Top navigation bar
st.markdown("""
<div class="top-nav">
    <div style="display: flex; align-items: center;">
        <span style="font-size: 1.3rem;">📊 Retail Analytics</span>
    </div>
    <div style="display: flex; gap: 1.2rem;">
        <div class="nav-item">📈 Dashboard</div>
        <div class="nav-item">📊 Reports</div>
        <div class="nav-item">⚙️ Settings</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main header with enhanced styling
st.markdown('<div class="main-header">📊 Retail Price Analytics Dashboard</div>', unsafe_allow_html=True)

# Enhanced sidebar with better styling and organization
with st.sidebar:
    st.markdown('<div class="sidebar-header">🎯 Analysis Control Panel</div>', unsafe_allow_html=True)
    
    # Data Overview Section
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
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
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Filter Section
    st.markdown('<div class="filter-section">', unsafe_allow_html=True)
    st.markdown("**🔍 Data Filters**")
    
    # Price range filter
    price_range = st.slider(
        "Price Range ($)",
        min_value=float(data['total_price'].min()),
        max_value=float(data['total_price'].max()),
        value=(float(data['total_price'].min()), float(data['total_price'].max())),
        key='price_filter'
    )
    
    # Category filter
    categories = ['All'] + list(data['product_category_name'].unique())
    selected_category = st.selectbox('Product Category', categories, key='category_filter')
    
    # Quantity filter
    qty_range = st.slider(
        "Quantity Range",
        min_value=int(data['qty'].min()),
        max_value=int(data['qty'].max()),
        value=(int(data['qty'].min()), int(data['qty'].max())),
        key='qty_filter'
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Visualization section
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**📈 Visualization Options**")
    chart_options = ['Histogram', 'Box Plot', 'Scatter Plot', 'Bar Chart', 'Correlation Heatmap', 'Bar Chart - Price Difference']
    selected_chart = st.selectbox('Select a visualization:', chart_options, key='chart_selector')
    
    # Chart customization options
    if selected_chart in ['Histogram', 'Box Plot', 'Bar Chart']:
        color_theme = st.selectbox('Color Theme', ['Blue', 'Purple', 'Pink', 'Green'], key='color_theme')
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Modeling section
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**🤖 Modeling Options**")
    
    # Model parameters
    test_size = st.slider('Test Size (%)', 10, 50, 20, key='test_size')
    random_state = st.slider('Random State', 1, 100, 42, key='random_state')
    
    model_button = st.button('🚀 Train Decision Tree Model', key='model_button')
    st.markdown('</div>', unsafe_allow_html=True)
    
    # System Status
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**🔧 System Status**")
    st.markdown('<span class="status-indicator status-active"></span>Data Loaded', unsafe_allow_html=True)
    st.markdown('<span class="status-indicator status-active"></span>Model Ready', unsafe_allow_html=True)
    st.markdown('<span class="status-indicator status-active"></span>Charts Active', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Apply filters to data
filtered_data = data.copy()
filtered_data = filtered_data[
    (filtered_data['total_price'] >= price_range[0]) &
    (filtered_data['total_price'] <= price_range[1]) &
    (filtered_data['qty'] >= qty_range[0]) &
    (filtered_data['qty'] <= qty_range[1])
]

if selected_category != 'All':
    filtered_data = filtered_data[filtered_data['product_category_name'] == selected_category]

# Color mapping
color_map = {
    'Blue': '#667eea',
    'Purple': '#764ba2', 
    'Pink': '#f093fb',
    'Green': '#4CAF50'
}

selected_color = color_map.get(color_theme, '#667eea') if 'color_theme' in locals() else '#667eea'

# Main content area
if selected_chart == 'Histogram':
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.header('📊 Distribution of Total Price')
    fig = px.histogram(filtered_data, x='total_price', nbins=20, 
                      color_discrete_sequence=[selected_color])
    fig.update_layout(
        title_font_size=20,
        title_font_color='#333',
        plot_bgcolor='rgba(255,255,255,0.9)',
        paper_bgcolor='rgba(255,255,255,0.9)'
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif selected_chart == 'Box Plot':
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.header('📦 Box Plot of Unit Price')
    fig = px.box(filtered_data, y='unit_price', color_discrete_sequence=[selected_color])
    fig.update_layout(
        title_font_size=20,
        title_font_color='#333',
        plot_bgcolor='rgba(255,255,255,0.9)',
        paper_bgcolor='rgba(255,255,255,0.9)'
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif selected_chart == 'Scatter Plot':
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.header('📈 Quantity vs Total Price with Trendline')
    fig = px.scatter(filtered_data, x='qty', y='total_price', trendline='ols',
                    color_discrete_sequence=[selected_color])
    fig.update_layout(
        title_font_size=20,
        title_font_color='#333',
        plot_bgcolor='rgba(255,255,255,0.9)',
        paper_bgcolor='rgba(255,255,255,0.9)'
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif selected_chart == 'Bar Chart':
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.header('📊 Average Total Price by Product Category')
    avg_price_by_category = filtered_data.groupby('product_category_name')['total_price'].mean().reset_index()
    fig = px.bar(avg_price_by_category, x='product_category_name', y='total_price',
                color_discrete_sequence=[selected_color])
    fig.update_layout(
        title_font_size=20,
        title_font_color='#333',
        plot_bgcolor='rgba(255,255,255,0.9)',
        paper_bgcolor='rgba(255,255,255,0.9)',
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif selected_chart == 'Correlation Heatmap':
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.header('🔥 Correlation Heatmap of Numerical Features')
    numeric_columns = filtered_data.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = filtered_data[numeric_columns].corr()

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
    st.markdown('</div>', unsafe_allow_html=True)

elif selected_chart == 'Bar Chart - Price Difference':
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.header('💰 Average Competitor Price Difference by Product Category')
    filtered_data['comp_price_diff'] = filtered_data['unit_price'] - filtered_data['comp_1']

    avg_price_diff_by_category = filtered_data.groupby('product_category_name')['comp_price_diff'].mean().reset_index()
    fig = px.bar(avg_price_diff_by_category, x='product_category_name', y='comp_price_diff',
                color_discrete_sequence=['#f093fb'])
    fig.update_layout(
        title_font_size=20,
        title_font_color='#333',
        plot_bgcolor='rgba(255,255,255,0.9)',
        paper_bgcolor='rgba(255,255,255,0.9)',
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Train a Decision Tree Regressor
if model_button:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.header('🤖 Predicted vs Actual Retail Price')
    
    with st.spinner('Training model...'):
        filtered_data['comp_price_diff'] = filtered_data['unit_price'] - filtered_data['comp_1']
        
        X = filtered_data[['qty', 'unit_price', 'comp_1', 'product_score', 'comp_price_diff']]
        y = filtered_data['total_price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=random_state)
        
        model = DecisionTreeRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
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
            name='Ideal Prediction'
        ))
        fig.update_layout(
            title='Predicted vs Actual Retail Price',
            xaxis_title='Actual Retail Price',
            yaxis_title='Predicted Retail Price',
            title_font_size=20,
            title_font_color='#333',
            plot_bgcolor='rgba(255,255,255,0.9)',
            paper_bgcolor='rgba(255,255,255,0.9)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        mse = mean_squared_error(y_test, y_pred)
        st.markdown(f'<div class="metric-card">Mean Squared Error: {mse:.2f}</div>', unsafe_allow_html=True)
        
        # Model performance metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Samples", len(X_train))
        with col2:
            st.metric("Test Samples", len(X_test))
        with col3:
            st.metric("Features Used", len(X.columns))
    st.markdown('</div>', unsafe_allow_html=True)

# Price Prediction Calculator Section
st.markdown("---")
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; margin: 2rem 0;">
    <h2 style="color: white; text-align: center; margin: 0;">🎯 Price Prediction Calculator</h2>
    <p style="color: white; text-align: center; margin: 0.5rem 0 0 0; opacity: 0.9;">Input product parameters to get real-time price predictions</p>
</div>
""", unsafe_allow_html=True)

# Create two columns for better layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Product Parameters")
    
    # Get unique product categories for dropdown
    product_categories = data['product_category_name'].unique()
    
    # Input form
    with st.form("price_prediction_form"):
        selected_category = st.selectbox(
            "🏷️ Product Category",
            options=product_categories,
            help="Select the product category"
        )
        
        quantity = st.slider(
            "📦 Quantity",
            min_value=1,
            max_value=1000,
            value=100,
            help="Number of units"
        )
        
        unit_price = st.number_input(
            "💰 Unit Price",
            min_value=0.01,
            max_value=1000.0,
            value=50.0,
            step=0.01,
            format="%.2f",
            help="Price per unit"
        )
        
        competitor_price = st.number_input(
            "🏪 Competitor Price",
            min_value=0.01,
            max_value=1000.0,
            value=45.0,
            step=0.01,
            format="%.2f",
            help="Competitor's price for similar product"
        )
        
        product_score = st.slider(
            "⭐ Product Score",
            min_value=1.0,
            max_value=10.0,
            value=7.0,
            step=0.1,
            help="Product quality/rating score"
        )
        
        submitted = st.form_submit_button("🚀 Predict Price", use_container_width=True)

with col2:
    st.subheader("🎯 Prediction Results")
    
    if submitted:
        # Calculate price difference
        price_diff = unit_price - competitor_price
        
        # Prepare input for prediction
        input_data = [[quantity, unit_price, competitor_price, product_score, price_diff]]
        
        # Train model if not already trained
        if 'trained_model' not in st.session_state:
            data['comp_price_diff'] = data['unit_price'] - data['comp_1']
            X = data[['qty', 'unit_price', 'comp_1', 'product_score', 'comp_price_diff']]
            y = data['total_price']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = DecisionTreeRegressor()
            model.fit(X_train, y_train)
            st.session_state.trained_model = model
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
        
        # Make prediction
        model = st.session_state.trained_model
        predicted_price = model.predict(input_data)[0]
        
        # Calculate confidence (simplified - using model's feature importance)
        confidence = min(95, max(60, 70 + (product_score - 5) * 3))
        
        # Display results
        st.markdown(f"""
        <div style="background: #f0f2f6; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #667eea;">
            <h3 style="color: #667eea; margin-top: 0;">Predicted Total Price</h3>
            <h2 style="color: #2c3e50; margin: 0.5rem 0;">${predicted_price:.2f}</h2>
            <p style="color: #7f8c8d; margin: 0;">Confidence: {confidence:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Price comparison
        st.markdown("### 📈 Price Analysis")
        
        col2_1, col2_2, col2_3 = st.columns(3)
        
        with col2_1:
            st.metric("Predicted Price", f"${predicted_price:.2f}")
        
        with col2_2:
            st.metric("Unit Price", f"${unit_price:.2f}")
        
        with col2_3:
            profit_margin = ((predicted_price - (unit_price * quantity)) / (unit_price * quantity)) * 100
            st.metric("Profit Margin", f"{profit_margin:.1f}%")
        
        # Price range estimation
        price_range_min = predicted_price * 0.85
        price_range_max = predicted_price * 1.15
        
        st.markdown(f"""
        <div style="background: #e8f5e8; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
            <h4 style="color: #27ae60; margin-top: 0;">📊 Expected Price Range</h4>
            <p style="margin: 0;"><strong>Low:</strong> ${price_range_min:.2f} | <strong>High:</strong> ${price_range_max:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.info("👆 Fill out the form and click 'Predict Price' to see results!")

# Advanced Analytics Dashboard Section
st.markdown("---")
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; margin: 2rem 0;">
    <h2 style="color: white; text-align: center; margin: 0;">📊 Advanced Analytics Dashboard</h2>
    <p style="color: white; text-align: center; margin: 0.5rem 0 0 0; opacity: 0.9;">Deep dive into retail performance metrics and market insights</p>
</div>
""", unsafe_allow_html=True)

# Create tabs for different analytics sections
tab1, tab2, tab3, tab4 = st.tabs(["📈 KPI Dashboard", "🎯 Market Analysis", "📋 Export Tools", "🔍 Data Explorer"])

with tab1:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.header("📈 Key Performance Indicators")
    
    # Calculate KPIs
    total_revenue = filtered_data['total_price'].sum()
    avg_order_value = filtered_data['total_price'].mean()
    total_products = len(filtered_data)
    avg_quantity = filtered_data['qty'].mean()
    
    # Create KPI cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="💰 Total Revenue",
            value=f"${total_revenue:,.2f}",
            delta=f"{len(filtered_data)} orders"
        )
    
    with col2:
        st.metric(
            label="📊 Avg Order Value",
            value=f"${avg_order_value:.2f}",
            delta=f"{avg_quantity:.1f} avg qty"
        )
    
    with col3:
        st.metric(
            label="📦 Total Products",
            value=f"{total_products:,}",
            delta=f"{len(filtered_data['product_category_name'].unique())} categories"
        )
    
    with col4:
        price_variance = filtered_data['total_price'].std()
        st.metric(
            label="📈 Price Variance",
            value=f"${price_variance:.2f}",
            delta=f"±{price_variance/avg_order_value*100:.1f}%"
        )
    
    # Revenue by Category Chart
    st.subheader("💰 Revenue Distribution by Category")
    revenue_by_category = filtered_data.groupby('product_category_name')['total_price'].sum().reset_index()
    revenue_by_category = revenue_by_category.sort_values('total_price', ascending=False)
    
    fig_revenue = px.pie(
        revenue_by_category, 
        values='total_price', 
        names='product_category_name',
        title="Revenue Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_revenue.update_layout(
        title_font_size=16,
        title_font_color='#333',
        plot_bgcolor='rgba(255,255,255,0.9)',
        paper_bgcolor='rgba(255,255,255,0.9)'
    )
    st.plotly_chart(fig_revenue, use_container_width=True)
    
    # Top Performing Products
    st.subheader("🏆 Top 10 Products by Revenue")
    top_products = filtered_data.nlargest(10, 'total_price')[['product_category_name', 'qty', 'unit_price', 'total_price']]
    st.dataframe(top_products, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.header("🎯 Market Analysis & Segmentation")
    
    # Market Share Analysis
    st.subheader("📊 Market Share by Product Category")
    market_share = filtered_data.groupby('product_category_name').agg({
        'total_price': ['sum', 'count', 'mean'],
        'qty': 'sum'
    }).round(2)
    
    market_share.columns = ['Total Revenue', 'Order Count', 'Avg Price', 'Total Quantity']
    market_share['Market Share %'] = (market_share['Total Revenue'] / market_share['Total Revenue'].sum() * 100).round(2)
    market_share = market_share.sort_values('Market Share %', ascending=False)
    
    st.dataframe(market_share, use_container_width=True)
    
    # Price Elasticity Analysis
    st.subheader("📈 Price vs Quantity Analysis")
    fig_elasticity = px.scatter(
        filtered_data, 
        x='unit_price', 
        y='qty', 
        size='total_price',
        color='product_category_name',
        hover_data=['product_score'],
        title="Price Elasticity Analysis"
    )
    fig_elasticity.update_layout(
        title_font_size=16,
        title_font_color='#333',
        plot_bgcolor='rgba(255,255,255,0.9)',
        paper_bgcolor='rgba(255,255,255,0.9)'
    )
    st.plotly_chart(fig_elasticity, use_container_width=True)
    
    # Competitor Analysis
    st.subheader("🏪 Competitor Price Analysis")
    competitor_analysis = filtered_data.groupby('product_category_name').agg({
        'unit_price': 'mean',
        'comp_1': 'mean',
        'comp_2': 'mean',
        'comp_3': 'mean'
    }).round(2)
    
    competitor_analysis['Price Advantage'] = competitor_analysis['unit_price'] - competitor_analysis['comp_1']
    competitor_analysis['Competitive Position'] = competitor_analysis['Price Advantage'].apply(
        lambda x: 'Premium' if x > 0 else 'Budget' if x < -5 else 'Competitive'
    )
    
    st.dataframe(competitor_analysis, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.header("📋 Export & Reporting Tools")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Export Filtered Data")
        
        # Export options
        export_format = st.selectbox("Export Format", ["CSV", "Excel", "JSON"])
        
        if st.button("📥 Download Filtered Data", key="export_filtered"):
            if export_format == "CSV":
                csv = filtered_data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"retail_data_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            elif export_format == "Excel":
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    filtered_data.to_excel(writer, sheet_name='Retail Data', index=False)
                st.download_button(
                    label="Download Excel",
                    data=output.getvalue(),
                    file_name=f"retail_data_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            elif export_format == "JSON":
                json_data = filtered_data.to_json(orient='records', indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"retail_data_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    with col2:
        st.subheader("📈 Generate Summary Report")
        
        if st.button("📋 Generate Report", key="generate_report"):
            # Create summary report
            report_data = {
                'Analysis Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Total Records': len(filtered_data),
                'Total Revenue': f"${filtered_data['total_price'].sum():,.2f}",
                'Average Order Value': f"${filtered_data['total_price'].mean():.2f}",
                'Product Categories': len(filtered_data['product_category_name'].unique()),
                'Price Range': f"${filtered_data['total_price'].min():.2f} - ${filtered_data['total_price'].max():.2f}",
                'Top Category': filtered_data.groupby('product_category_name')['total_price'].sum().idxmax(),
                'Average Quantity': f"{filtered_data['qty'].mean():.1f}"
            }
            
            report_df = pd.DataFrame(list(report_data.items()), columns=['Metric', 'Value'])
            st.dataframe(report_df, use_container_width=True)
            
            # Download report
            csv_report = report_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Summary Report",
                data=csv_report,
                file_name=f"retail_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.header("🔍 Advanced Data Explorer")
    
    # Data quality metrics
    st.subheader("📊 Data Quality Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        missing_data = filtered_data.isnull().sum().sum()
        st.metric("Missing Values", missing_data)
    
    with col2:
        duplicates = filtered_data.duplicated().sum()
        st.metric("Duplicate Records", duplicates)
    
    with col3:
        data_types = len(filtered_data.dtypes.unique())
        st.metric("Data Types", data_types)
    
    # Statistical Summary
    st.subheader("📈 Statistical Summary")
    numeric_cols = filtered_data.select_dtypes(include=[np.number]).columns
    st.dataframe(filtered_data[numeric_cols].describe(), use_container_width=True)
    
    # Data Distribution
    st.subheader("📊 Data Distribution Analysis")
    selected_column = st.selectbox("Select Column for Distribution", numeric_cols)
    
    if selected_column:
        fig_dist = px.histogram(
            filtered_data, 
            x=selected_column, 
            nbins=30,
            title=f"Distribution of {selected_column}"
        )
        fig_dist.update_layout(
            title_font_size=16,
            title_font_color='#333',
            plot_bgcolor='rgba(255,255,255,0.9)',
            paper_bgcolor='rgba(255,255,255,0.9)'
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Outlier Detection
    st.subheader("🚨 Outlier Detection")
    if st.button("Detect Outliers", key="detect_outliers"):
        outlier_cols = ['total_price', 'unit_price', 'qty']
        outliers = {}
        
        for col in outlier_cols:
            Q1 = filtered_data[col].quantile(0.25)
            Q3 = filtered_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers[col] = filtered_data[(filtered_data[col] < lower_bound) | (filtered_data[col] > upper_bound)]
        
        for col, outlier_data in outliers.items():
            if len(outlier_data) > 0:
                st.write(f"**{col}**: {len(outlier_data)} outliers detected")
                st.dataframe(outlier_data[[col, 'product_category_name']].head(), use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

