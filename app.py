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

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        text-align: center;
        color: white;
        font-weight: bold;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .stSelectbox > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 8px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Load data
data = pd.read_csv('retail_price.csv')

# Main header with enhanced styling
st.markdown('<div class="main-header">📊 Retail Price Analytics Dashboard</div>', unsafe_allow_html=True)

# Enhanced sidebar with better styling
with st.sidebar:
    st.markdown('<div class="sidebar-header">🎯 Analysis Options</div>', unsafe_allow_html=True)
    
    # Visualization section
    st.markdown("**📈 Visualization Options**")
    chart_options = ['Histogram', 'Box Plot', 'Scatter Plot', 'Bar Chart', 'Correlation Heatmap', 'Bar Chart - Price Difference']
    selected_chart = st.selectbox('Select a visualization:', chart_options, key='chart_selector')
    
    st.markdown("---")
    
    # Modeling section
    st.markdown("**🤖 Modeling Options**")
    model_button = st.button('🚀 Train Decision Tree Model', key='model_button')
    
    st.markdown("---")
    
    # Data insights section
    st.markdown("**📊 Quick Insights**")
    total_products = len(data)
    avg_price = data['total_price'].mean()
    st.metric("Total Products", f"{total_products:,}")
    st.metric("Avg Price", f"${avg_price:.2f}")

# Main content area
if selected_chart == 'Histogram':
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.header('📊 Distribution of Total Price')
    fig = px.histogram(data, x='total_price', nbins=20, 
                      color_discrete_sequence=['#667eea'])
    fig.update_layout(
        title_font_size=20,
        title_font_color='#333',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif selected_chart == 'Box Plot':
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.header('📦 Box Plot of Unit Price')
    fig = px.box(data, y='unit_price', color_discrete_sequence=['#764ba2'])
    fig.update_layout(
        title_font_size=20,
        title_font_color='#333',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif selected_chart == 'Scatter Plot':
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.header('📈 Quantity vs Total Price with Trendline')
    fig = px.scatter(data, x='qty', y='total_price', trendline='ols',
                    color_discrete_sequence=['#667eea'])
    fig.update_layout(
        title_font_size=20,
        title_font_color='#333',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif selected_chart == 'Bar Chart':
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.header('📊 Average Total Price by Product Category')
    avg_price_by_category = data.groupby('product_category_name')['total_price'].mean().reset_index()
    fig = px.bar(avg_price_by_category, x='product_category_name', y='total_price',
                color_discrete_sequence=['#764ba2'])
    fig.update_layout(
        title_font_size=20,
        title_font_color='#333',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif selected_chart == 'Correlation Heatmap':
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
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
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif selected_chart == 'Bar Chart - Price Difference':
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.header('💰 Average Competitor Price Difference by Product Category')
    data['comp_price_diff'] = data['unit_price'] - data['comp_1']

    avg_price_diff_by_category = data.groupby('product_category_name')['comp_price_diff'].mean().reset_index()
    fig = px.bar(avg_price_diff_by_category, x='product_category_name', y='comp_price_diff',
                color_discrete_sequence=['#f093fb'])
    fig.update_layout(
        title_font_size=20,
        title_font_color='#333',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Train a Decision Tree Regressor
if model_button:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.header('🤖 Predicted vs Actual Retail Price')
    
    with st.spinner('Training model...'):
        data['comp_price_diff'] = data['unit_price'] - data['comp_1']
        
        X = data[['qty', 'unit_price', 'comp_1', 'product_score', 'comp_price_diff']]
        y = data['total_price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
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
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        mse = mean_squared_error(y_test, y_pred)
        st.markdown(f'<div class="metric-card">Mean Squared Error: {mse:.2f}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

