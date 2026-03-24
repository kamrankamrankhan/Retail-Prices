# Streamlit Dashboard Page - Sales Forecast

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Sales Forecast",
    page_icon="📈",
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
    <h1 style="color: white; margin: 0;">📈 Sales Forecast Dashboard</h1>
    <p style="color: white; margin: 0.5rem 0 0 0; opacity: 0.9;">Predictive analytics and sales forecasting for retail optimization</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Forecast Settings")

    forecast_periods = st.slider("Forecast Periods (months)", 1, 12, 6)

    confidence_level = st.select_slider(
        "Confidence Level",
        options=[80, 90, 95],
        value=90
    )

    selected_category = st.selectbox(
        "Category Filter",
        ['All'] + list(data['product_category_name'].unique())
    )

    st.markdown("---")
    st.markdown("**📊 Model Info**")
    st.info("Using moving average and trend analysis for forecasting")

# Filter data
filtered_data = data.copy()
if selected_category != 'All':
    filtered_data = filtered_data[filtered_data['product_category_name'] == selected_category]

# Process dates
try:
    filtered_data['date'] = pd.to_datetime(filtered_data['month_year'], format='%d-%m-%Y')
    filtered_data = filtered_data.sort_values('date')
except:
    st.warning("Could not parse dates properly")

# Historical KPIs
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_historical_revenue = filtered_data['total_price'].sum()
    st.metric("💰 Historical Revenue", f"${total_historical_revenue:,.2f}")

with col2:
    avg_monthly_revenue = filtered_data.groupby(filtered_data['date'].dt.to_period('M'))['total_price'].sum().mean()
    st.metric("📊 Avg Monthly Revenue", f"${avg_monthly_revenue:,.2f}")

with col3:
    total_qty = filtered_data['qty'].sum()
    st.metric("📦 Total Units Sold", f"{total_qty:,}")

with col4:
    avg_price = filtered_data['unit_price'].mean()
    st.metric("💵 Avg Unit Price", f"${avg_price:.2f}")

st.markdown("---")

# Monthly aggregation
monthly_data = filtered_data.groupby(filtered_data['date'].dt.to_period('M')).agg({
    'total_price': 'sum',
    'qty': 'sum',
    'unit_price': 'mean'
}).reset_index()

monthly_data['date'] = monthly_data['date'].dt.to_timestamp()

# Simple forecast using moving average and trend
def generate_forecast(historical_data, periods, confidence=90):
    """Generate simple forecast based on historical trends."""
    values = historical_data['total_price'].values

    # Calculate trend
    if len(values) >= 3:
        trend = np.polyfit(range(len(values)), values, 1)
        slope = trend[0]
        intercept = trend[1]
    else:
        slope = 0
        intercept = values.mean()

    # Calculate seasonal factors (simplified)
    seasonal_factor = 1 + 0.1 * np.sin(np.arange(periods) * np.pi / 6)

    # Generate forecast
    last_idx = len(values)
    forecast_values = []
    for i in range(periods):
        base_value = intercept + slope * (last_idx + i)
        seasonal_value = base_value * seasonal_factor[i]
        forecast_values.append(seasonal_value)

    # Calculate confidence intervals
    std_dev = np.std(values) if len(values) > 1 else values.mean() * 0.1
    z_score = {80: 1.28, 90: 1.645, 95: 1.96}[confidence]

    lower_bound = [v - z_score * std_dev for v in forecast_values]
    upper_bound = [v + z_score * std_dev for v in forecast_values]

    return forecast_values, lower_bound, upper_bound

# Generate forecast
forecast_values, lower_bound, upper_bound = generate_forecast(
    monthly_data, forecast_periods, confidence_level
)

# Create forecast dates
last_date = monthly_data['date'].max()
forecast_dates = pd.date_range(
    start=last_date + pd.DateOffset(months=1),
    periods=forecast_periods,
    freq='MS'
)

# Create forecast visualization
st.subheader("📈 Revenue Forecast")

fig = go.Figure()

# Historical data
fig.add_trace(go.Scatter(
    x=monthly_data['date'],
    y=monthly_data['total_price'],
    mode='lines+markers',
    name='Historical',
    line=dict(color='#667eea', width=2),
    marker=dict(size=8)
))

# Forecast
fig.add_trace(go.Scatter(
    x=forecast_dates,
    y=forecast_values,
    mode='lines+markers',
    name='Forecast',
    line=dict(color='#f5576c', width=2, dash='dash'),
    marker=dict(size=8)
))

# Confidence interval
fig.add_trace(go.Scatter(
    x=list(forecast_dates) + list(forecast_dates[::-1]),
    y=upper_bound + lower_bound[::-1],
    fill='toself',
    fillcolor='rgba(245, 87, 108, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    name=f'{confidence_level}% Confidence'
))

fig.update_layout(
    title='Revenue Forecast with Confidence Interval',
    xaxis_title='Date',
    yaxis_title='Revenue ($)',
    hovermode='x unified',
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# Forecast table
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Forecast Details")

    forecast_df = pd.DataFrame({
        'Period': [d.strftime('%b %Y') for d in forecast_dates],
        'Forecast ($)': [f'${v:,.2f}' for v in forecast_values],
        f'Lower {confidence_level}%': [f'${v:,.2f}' for v in lower_bound],
        f'Upper {confidence_level}%': [f'${v:,.2f}' for v in upper_bound]
    })

    st.dataframe(forecast_df, use_container_width=True, hide_index=True)

with col2:
    st.subheader("📊 Forecast Summary")

    total_forecast = sum(forecast_values)
    avg_forecast = np.mean(forecast_values)
    growth_rate = ((forecast_values[-1] - forecast_values[0]) / forecast_values[0]) * 100 if forecast_values[0] != 0 else 0

    st.metric("Total Forecasted Revenue", f"${total_forecast:,.2f}")
    st.metric("Average Monthly Forecast", f"${avg_forecast:,.2f}")
    st.metric("Projected Growth Rate", f"{growth_rate:.1f}%")

# Category-wise Forecast
st.markdown("---")
st.subheader("📁 Category-wise Forecast")

if selected_category == 'All':
    category_forecasts = []
    categories = data['product_category_name'].unique()[:6]  # Top 6 categories

    for cat in categories:
        cat_data = data[data['product_category_name'] == cat].copy()
        try:
            cat_data['date'] = pd.to_datetime(cat_data['month_year'], format='%d-%m-%Y')
            cat_monthly = cat_data.groupby(cat_data['date'].dt.to_period('M'))['total_price'].sum().reset_index()
            cat_monthly['date'] = cat_monthly['date'].dt.to_timestamp()

            if len(cat_monthly) >= 2:
                forecast, _, _ = generate_forecast(cat_monthly, 3)
                category_forecasts.append({
                    'Category': cat,
                    'Month 1': f'${forecast[0]:,.2f}',
                    'Month 2': f'${forecast[1]:,.2f}',
                    'Month 3': f'${forecast[2]:,.2f}',
                    '3M Total': f'${sum(forecast):,.2f}'
                })
        except:
            continue

    if category_forecasts:
        forecast_table = pd.DataFrame(category_forecasts)
        st.dataframe(forecast_table, use_container_width=True, hide_index=True)

# Insights
st.markdown("---")
st.subheader("💡 Key Insights")

insights_col1, insights_col2 = st.columns(2)

with insights_col1:
    st.markdown("**📈 Positive Trends**")
    if growth_rate > 0:
        st.success(f"Revenue is projected to grow by {growth_rate:.1f}% over the forecast period")
    if avg_forecast > avg_monthly_revenue:
        st.success("Forecasted monthly revenue exceeds historical average")

with insights_col2:
    st.markdown("**⚠️ Areas to Watch**")
    if len(monthly_data) < 6:
        st.warning("Limited historical data may affect forecast accuracy")
    if np.std(forecast_values) > np.std(monthly_data['total_price']):
        st.warning("High forecast volatility detected")

# Footer
st.markdown("---")
st.caption("Forecast generated using trend analysis and seasonal adjustment. Results are estimates and actual results may vary.")
