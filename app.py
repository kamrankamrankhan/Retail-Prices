import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Set Plotly template to white background
import plotly.io as pio
pio.templates.default = "plotly_white"

# Custom CSS for better styling
st.set_page_config(
    page_title="Retail Price Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)




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

