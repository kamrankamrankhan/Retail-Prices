import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

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

# Load data
data = pd.read_csv('retail_price.csv')

# Modern Header with CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        font-size: 3rem;
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    .main-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Modern Header
st.markdown("""
<div class="main-header">
    <h1>📊 Retail Price Analysis Dashboard</h1>
    <p>🔍 Explore retail pricing patterns and train predictive models</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with modern styling
st.sidebar.markdown("""
<div class="sidebar-header">
    <h3>🎯 Analysis Tools</h3>
</div>
""", unsafe_allow_html=True)

st.sidebar.header('📈 Visualization Options')
chart_options = ['Histogram', 'Box Plot', 'Scatter Plot', 'Bar Chart', 'Correlation Heatmap', 'Bar Chart - Price Difference']
selected_chart = st.sidebar.selectbox('Select a visualization:', chart_options)

if selected_chart == 'Histogram':
    st.header('Distribution of Total Price')
    fig = px.histogram(data, x='total_price', nbins=20)
    st.plotly_chart(fig)

elif selected_chart == 'Box Plot':
    st.header('Box Plot of Unit Price')
    fig = px.box(data, y='unit_price')
    st.plotly_chart(fig)

elif selected_chart == 'Scatter Plot':
    st.header('Quantity vs Total Price with Trendline')
    fig = px.scatter(data, x='qty', y='total_price', trendline='ols')
    st.plotly_chart(fig)

elif selected_chart == 'Bar Chart':
    st.header('Average Total Price by Product Category')
    avg_price_by_category = data.groupby('product_category_name')['total_price'].mean().reset_index()
    fig = px.bar(avg_price_by_category, x='product_category_name', y='total_price')
    st.plotly_chart(fig)

elif selected_chart == 'Correlation Heatmap':
    st.header('Correlation Heatmap of Numerical Features')
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = data[numeric_columns].corr()

    fig = go.Figure(go.Heatmap(x=correlation_matrix.columns, y=correlation_matrix.columns, z=correlation_matrix.values))
    st.plotly_chart(fig)

elif selected_chart == 'Bar Chart - Price Difference':
    st.header('Average Competitor Price Difference by Product Category')
    data['comp_price_diff'] = data['unit_price'] - data['comp_1']

    avg_price_diff_by_category = data.groupby('product_category_name')['comp_price_diff'].mean().reset_index()
    fig = px.bar(avg_price_diff_by_category, x='product_category_name', y='comp_price_diff')
    st.plotly_chart(fig)

# Train a Decision Tree Regressor
st.sidebar.header('Modeling Options')
model_button = st.sidebar.button('Train Decision Tree Regressor Model')

if model_button:
    st.header('Predicted vs Actual Retail Price')
    data['comp_price_diff'] = data['unit_price'] - data['comp_1']
    
    X = data[['qty', 'unit_price', 'comp_1', 'product_score', 'comp_price_diff']]
    y = data['total_price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', marker=dict(color='blue'), name='Predicted vs. Actual'))
    fig.add_trace(go.Scatter(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)], mode='lines', marker=dict(color='red'), name='Ideal Prediction'))
    fig.update_layout(title='Predicted vs Actual Retail Price', xaxis_title='Actual Retail Price', yaxis_title='Predicted Retail Price')
    st.plotly_chart(fig)
    
    mse = mean_squared_error(y_test, y_pred)
    st.write(f'Mean Squared Error: {mse:.2f}')

