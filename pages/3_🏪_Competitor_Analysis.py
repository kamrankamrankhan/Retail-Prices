# Streamlit Dashboard Page - Competitor Analysis

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(
    page_title="Competitor Analysis",
    page_icon="🏪",
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
    <h1 style="color: white; margin: 0;">🏪 Competitor Analysis Dashboard</h1>
    <p style="color: white; margin: 0.5rem 0 0 0; opacity: 0.9;">Compare your pricing strategy against competitors and identify market opportunities</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🔍 Analysis Settings")

    categories = ['All'] + list(data['product_category_name'].unique())
    selected_category = st.selectbox("Select Category", categories)

    competitor = st.selectbox(
        "Compare Against",
        ['All Competitors', 'Competitor 1', 'Competitor 2', 'Competitor 3']
    )

    st.markdown("---")
    st.markdown("**📊 Quick Stats**")

    avg_price_diff = data['unit_price'].mean() - data['comp_1'].mean()
    st.metric("Avg Price Difference", f"${avg_price_diff:.2f}")

# Filter data
filtered_data = data.copy()
if selected_category != 'All':
    filtered_data = filtered_data[filtered_data['product_category_name'] == selected_category]

# Calculate competitor metrics
filtered_data['comp_avg'] = filtered_data[['comp_1', 'comp_2', 'comp_3']].mean(axis=1)
filtered_data['price_diff_1'] = filtered_data['unit_price'] - filtered_data['comp_1']
filtered_data['price_diff_2'] = filtered_data['unit_price'] - filtered_data['comp_2']
filtered_data['price_diff_3'] = filtered_data['unit_price'] - filtered_data['comp_3']
filtered_data['price_position'] = filtered_data.apply(
    lambda x: 'Premium' if x['unit_price'] > x['comp_avg'] * 1.05
    else ('Budget' if x['unit_price'] < x['comp_avg'] * 0.95 else 'Competitive'),
    axis=1
)

# KPI Section
col1, col2, col3, col4 = st.columns(4)

with col1:
    premium_count = (filtered_data['price_position'] == 'Premium').sum()
    st.metric("🏆 Premium Products", premium_count)

with col2:
    competitive_count = (filtered_data['price_position'] == 'Competitive').sum()
    st.metric("⚖️ Competitive Products", competitive_count)

with col3:
    budget_count = (filtered_data['price_position'] == 'Budget').sum()
    st.metric("💰 Budget Products", budget_count)

with col4:
    avg_position = filtered_data['price_diff_1'].mean()
    position_label = "Above Market" if avg_position > 0 else "Below Market"
    st.metric("📊 Avg Position", position_label, f"${abs(avg_position):.2f}")

st.markdown("---")

# Price Position Distribution
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Price Position Distribution")

    position_counts = filtered_data['price_position'].value_counts().reset_index()
    position_counts.columns = ['Position', 'Count']

    colors = {'Premium': '#667eea', 'Competitive': '#4CAF50', 'Budget': '#f5576c'}

    fig = px.pie(
        position_counts,
        values='Count',
        names='Position',
        hole=0.4,
        color='Position',
        color_discrete_map=colors,
        title="Distribution of Price Positions"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("💵 Price Comparison by Category")

    comparison_data = filtered_data.groupby('product_category_name').agg({
        'unit_price': 'mean',
        'comp_1': 'mean',
        'comp_2': 'mean',
        'comp_3': 'mean'
    }).reset_index()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Our Price',
        x=comparison_data['product_category_name'],
        y=comparison_data['unit_price'],
        marker_color='#667eea'
    ))

    fig.add_trace(go.Bar(
        name='Competitor Avg',
        x=comparison_data['product_category_name'],
        y=comparison_data[['comp_1', 'comp_2', 'comp_3']].mean(axis=1),
        marker_color='#f5576c'
    ))

    fig.update_layout(
        barmode='group',
        title='Our Price vs Competitor Average',
        xaxis_title='Category',
        yaxis_title='Average Price ($)',
        xaxis_tickangle=-45,
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# Detailed Competitor Comparison
st.subheader("🔍 Detailed Competitor Analysis")

# Competitor price difference chart
st.markdown("**Price Difference from Each Competitor**")

diff_data = filtered_data.groupby('product_category_name').agg({
    'price_diff_1': 'mean',
    'price_diff_2': 'mean',
    'price_diff_3': 'mean'
}).reset_index()

fig = go.Figure()

fig.add_trace(go.Bar(
    name='vs Competitor 1',
    x=diff_data['product_category_name'],
    y=diff_data['price_diff_1'],
    marker_color='#667eea'
))

fig.add_trace(go.Bar(
    name='vs Competitor 2',
    x=diff_data['product_category_name'],
    y=diff_data['price_diff_2'],
    marker_color='#764ba2'
))

fig.add_trace(go.Bar(
    name='vs Competitor 3',
    x=diff_data['product_category_name'],
    y=diff_data['price_diff_3'],
    marker_color='#f093fb'
))

fig.update_layout(
    barmode='group',
    title='Price Difference by Competitor',
    xaxis_title='Category',
    yaxis_title='Price Difference ($)',
    xaxis_tickangle=-45,
    height=450
)

st.plotly_chart(fig, use_container_width=True)

# Competitive Position Table
st.markdown("---")
st.subheader("📋 Category-wise Competitive Position")

category_position = filtered_data.groupby('product_category_name').agg({
    'unit_price': 'mean',
    'comp_1': 'mean',
    'comp_2': 'mean',
    'comp_3': 'mean',
    'price_position': lambda x: x.mode().iloc[0] if len(x) > 0 else 'N/A'
}).round(2).reset_index()

category_position.columns = ['Category', 'Our Price', 'Comp 1', 'Comp 2', 'Comp 3', 'Position']

# Add styling
def color_position(val):
    if val == 'Premium':
        return 'background-color: #e3f2fd'
    elif val == 'Budget':
        return 'background-color: #ffebee'
    else:
        return 'background-color: #e8f5e9'

st.dataframe(
    category_position.style.applymap(color_position, subset=['Position']),
    use_container_width=True
)

# Market Insights
st.markdown("---")
st.subheader("💡 Market Insights")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**📈 Opportunities**")

    opportunities = []
    for _, row in category_position.iterrows():
        if row['Position'] == 'Budget':
            opportunities.append(f"✅ **{row['Category']}**: Opportunity to increase prices while staying competitive")

    if opportunities:
        for opp in opportunities:
            st.success(opp)
    else:
        st.info("No immediate pricing opportunities identified")

with col2:
    st.markdown("**⚠️ Risks**")

    risks = []
    for _, row in category_position.iterrows():
        if row['Position'] == 'Premium':
            risks.append(f"⚠️ **{row['Category']}**: Higher than market - may lose price-sensitive customers")

    if risks:
        for risk in risks:
            st.warning(risk)
    else:
        st.info("No significant competitive risks identified")

# Recommendations
st.markdown("---")
st.subheader("🎯 Strategic Recommendations")

# Calculate overall strategy
avg_diff = filtered_data['price_diff_1'].mean()

if avg_diff > 5:
    strategy = "Premium positioning - ensure value proposition is clear"
elif avg_diff < -5:
    strategy = "Budget positioning - consider gradual price increase to improve margins"
else:
    strategy = "Competitive positioning - maintain current strategy"

st.info(f"**Recommended Strategy:** {strategy}")

# Footer
st.markdown("---")
st.caption("*Competitor data sourced from retail_price.csv. Analysis based on average prices per category.*")
