
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Electricity Demand Forecasting",
    page_icon="⚡",
    layout="wide"
)

DATA_PATH = Path(
    "data/processed/electricity_demand_cleaned.csv"
)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

df = load_data()

st.title("⚡ Electricity Demand Analytics")
st.caption("Historical electricity demand analysis | 2020–2024")

# Sidebar
st.sidebar.header("Dashboard Filters")

years = sorted(df["year"].unique())

selected_years = st.sidebar.multiselect(
    "Select Year",
    years,
    default=years
)

filtered = df[df["year"].isin(selected_years)]

if filtered.empty:
    st.warning("Please select at least one year.")
    st.stop()

# KPI metrics
c1, c2, c3, c4 = st.columns(4)

c1.metric("Average Demand", f"{filtered['demand'].mean():.2f}")
c2.metric("Peak Demand", f"{filtered['demand'].max():.2f}")
c3.metric("Minimum Demand", f"{filtered['demand'].min():.2f}")
c4.metric("Total Records", f"{len(filtered):,}")

st.divider()

# Historical demand
st.subheader("Historical Electricity Demand")

fig = px.line(
    filtered,
    x="timestamp",
    y="demand",
    title="Hourly Electricity Demand"
)

st.plotly_chart(fig, use_container_width=True)

# Monthly analysis
st.subheader("Monthly Demand Analysis")

monthly = (
    filtered.groupby("month")["demand"]
    .mean()
    .reset_index()
)

fig2 = px.bar(
    monthly,
    x="month",
    y="demand",
    title="Average Demand by Month"
)

st.plotly_chart(fig2, use_container_width=True)

# Weather analysis
st.subheader("Weather and Demand Relationship")

fig3 = px.scatter(
    filtered.sample(min(3000, len(filtered)), random_state=42),
    x="temperature",
    y="demand",
    color="humidity",
    title="Temperature vs Electricity Demand"
)

st.plotly_chart(fig3, use_container_width=True)

# Data preview
st.subheader("Historical Dataset")

st.markdown(
    """
    <style>
    .dataframe-container {
        overflow-x: auto;
        max-height: 500px;
        overflow-y: auto;
    }
    table {
        width: 100%;
        border-collapse: collapse;
    }
    th, td {
        padding: 8px;
        border: 1px solid #444;
        text-align: left;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="dataframe-container">'
    + filtered.head(100).to_html(index=False)
    + '</div>',
    unsafe_allow_html=True
)

# Download
csv = filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Historical Data",
    csv,
    "electricity_demand_data.csv",
    "text/csv"
)
