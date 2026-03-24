# Streamlit Dashboard Page - Inventory Analysis

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(
    page_title="Inventory Analysis",
    page_icon="📦",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv('retail_price.csv')
    return data

data = load_data()

# Page header
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
    <h1 style="color: white; margin: 0;">📦 Inventory Analysis Dashboard</h1>
    <p style="color: white; margin: 0.5rem 0 0 0; opacity: 0.9;">Monitor inventory levels, product performance, and stock optimization</p>
</div>
""", unsafe_allow_html=True)

# Sidebar filters
with st.sidebar:
    st.header("🔍 Filters")

    categories = ['All'] + list(data['product_category_name'].unique())
    selected_category = st.selectbox("Category", categories)

    # Date range
    if 'month_year' in data.columns:
        dates = data['month_year'].unique()
        selected_dates = st.multiselect("Dates", dates, default=dates[:5])

    st.markdown("---")
    st.markdown("**📊 Quick Stats**")
    st.metric("Total Products", data['product_id'].nunique())
    st.metric("Total Categories", data['product_category_name'].nunique())

# Filter data
filtered_data = data.copy()
if selected_category != 'All':
    filtered_data = filtered_data[filtered_data['product_category_name'] == selected_category]

# KPI Section
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_qty = filtered_data['qty'].sum()
    st.metric("📦 Total Quantity", f"{total_qty:,}")

with col2:
    avg_qty = filtered_data['qty'].mean()
    st.metric("📊 Avg Quantity/Product", f"{avg_qty:.1f}")

with col3:
    unique_products = filtered_data['product_id'].nunique()
    st.metric("🏷️ Unique Products", unique_products)

with col4:
    total_revenue = filtered_data['total_price'].sum()
    st.metric("💰 Total Revenue", f"${total_revenue:,.2f}")

st.markdown("---")

# Inventory Distribution Chart
col1, col2 = st.columns(2)

with col1:
    st.subheader("📦 Quantity Distribution by Category")

    qty_by_category = filtered_data.groupby('product_category_name')['qty'].sum().reset_index()
    qty_by_category = qty_by_category.sort_values('qty', ascending=True)

    fig = px.bar(
        qty_by_category,
        x='qty',
        y='product_category_name',
        orientation='h',
        color='qty',
        color_continuous_scale='Viridis',
        title="Total Quantity by Category"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🥧 Revenue Share by Category")

    revenue_by_category = filtered_data.groupby('product_category_name')['total_price'].sum().reset_index()

    fig = px.pie(
        revenue_by_category,
        values='total_price',
        names='product_category_name',
        hole=0.4,
        title="Revenue Distribution"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# Product Performance Table
st.subheader("🏆 Top Performing Products")

product_performance = filtered_data.groupby(['product_id', 'product_category_name']).agg({
    'qty': 'sum',
    'total_price': 'sum',
    'unit_price': 'mean',
    'product_score': 'mean'
}).reset_index()

product_performance = product_performance.sort_values('total_price', ascending=False)
product_performance.columns = ['Product ID', 'Category', 'Total Qty', 'Total Revenue', 'Avg Price', 'Avg Score']

st.dataframe(
    product_performance.head(15).style.background_gradient(subset=['Total Revenue'], cmap='Blues'),
    use_container_width=True
)

# Stock Analysis
st.markdown("---")
st.subheader("📈 Stock Analysis")

col1, col2 = st.columns(2)

with col1:
    # Quantity vs Revenue Scatter
    fig = px.scatter(
        filtered_data,
        x='qty',
        y='total_price',
        color='product_category_name',
        size='product_score',
        hover_data=['product_id'],
        title="Quantity vs Revenue by Product"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Price Score Distribution
    fig = px.box(
        filtered_data,
        x='product_category_name',
        y='product_score',
        title="Product Score Distribution by Category",
        color='product_category_name'
    )
    fig.update_layout(height=400, showlegend=False)
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

# Inventory Recommendations
st.markdown("---")
st.subheader("💡 Inventory Recommendations")

# Calculate metrics for recommendations
category_metrics = filtered_data.groupby('product_category_name').agg({
    'qty': ['sum', 'mean'],
    'total_price': 'sum',
    'product_score': 'mean'
}).round(2)

category_metrics.columns = ['Total Qty', 'Avg Qty', 'Total Revenue', 'Avg Score']
category_metrics = category_metrics.reset_index()

# Generate recommendations
recommendations = []

for _, row in category_metrics.iterrows():
    if row['Avg Score'] >= 4.0 and row['Total Qty'] > category_metrics['Total Qty'].mean():
        recommendations.append(f"✅ **{row['product_category_name']}**: High-performing category. Consider increasing stock levels.")
    elif row['Avg Score'] < 3.5:
        recommendations.append(f"⚠️ **{row['product_category_name']}**: Low product score. Consider quality improvements or discontinuation.")
    elif row['Total Revenue'] > category_metrics['Total Revenue'].quantile(0.75):
        recommendations.append(f"💰 **{row['product_category_name']}**: Top revenue generator. Prioritize inventory allocation.")

for rec in recommendations:
    st.markdown(rec)

# Footer
st.markdown("---")
st.markdown("*Last updated: Data loaded from retail_price.csv*")
