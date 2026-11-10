
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Electricity Demand Forecasting",
    page_icon="⚡",
    layout="wide"
)

# --------------------------------------------------
# 2. DATASET LOCATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "electricity_demand_synthetic_2020_2024.csv"
)

# --------------------------------------------------
# 3. LOAD DATASET
# --------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


if not DATA_PATH.exists():
    st.error("Dataset not found. Please check the data folder on GitHub.")
    st.write("Expected dataset location:", str(DATA_PATH))
    st.stop()

df = load_data()

# --------------------------------------------------
# 4. DASHBOARD TITLE
# --------------------------------------------------

st.title("⚡ Electricity Demand Forecasting Dashboard")

st.markdown(
    "### Historical Electricity Demand Analysis | 2020–2024"
)

st.markdown(
    "An interactive machine learning project for analysing "
    "historical electricity demand and weather patterns."
)

st.divider()

# --------------------------------------------------
# 5. FIVE-YEAR FILTER
# --------------------------------------------------

st.sidebar.title("Dashboard Controls")

years = sorted(df["timestamp"].dt.year.unique())

selected_years = st.sidebar.multiselect(
    "Select Historical Years",
    options=years,
    default=years
)

if not selected_years:
    st.warning("Please select at least one year.")
    st.stop()

filtered = df[
    df["timestamp"].dt.year.isin(selected_years)
].copy()

# --------------------------------------------------
# 6. KEY PERFORMANCE INDICATORS
# --------------------------------------------------

st.subheader("Electricity Demand Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Records",
    f"{len(filtered):,}"
)

col2.metric(
    "Average Demand",
    f"{filtered['demand'].mean():.2f}"
)

col3.metric(
    "Maximum Demand",
    f"{filtered['demand'].max():.2f}"
)

col4.metric(
    "Minimum Demand",
    f"{filtered['demand'].min():.2f}"
)

st.divider()

# --------------------------------------------------
# 7. HISTORICAL ELECTRICITY DEMAND
# --------------------------------------------------

st.subheader("1. Historical Electricity Demand")

daily_demand = (
    filtered.set_index("timestamp")["demand"]
    .resample("D")
    .mean()
    .reset_index()
)

fig1 = px.line(
    daily_demand,
    x="timestamp",
    y="demand",
    title="Daily Average Electricity Demand",
    labels={
        "timestamp": "Date",
        "demand": "Average Demand"
    }
)

fig1.update_layout(
    hovermode="x unified",
    template="plotly_white"
)

st.plotly_chart(fig1, use_container_width=True)

st.caption(
    "This chart displays the daily average of the hourly "
    "demand observations in the selected historical years."
)

st.divider()

# --------------------------------------------------
# 8. MONTHLY DEMAND ANALYSIS
# --------------------------------------------------

st.subheader("2. Monthly Demand Analysis")

monthly_demand = (
    filtered.groupby("month")["demand"]
    .mean()
    .reset_index()
)

month_names = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December"
}

monthly_demand["Month"] = (
    monthly_demand["month"].map(month_names)
)

fig2 = px.bar(
    monthly_demand,
    x="Month",
    y="demand",
    title="Average Electricity Demand by Month",
    labels={
        "demand": "Average Demand",
        "Month": "Month"
    },
    color="demand",
    color_continuous_scale="Blues"
)

fig2.update_layout(template="plotly_white")

st.plotly_chart(fig2, use_container_width=True)

st.caption(
    "Monthly analysis helps identify seasonal variations "
    "in electricity demand."
)

st.divider()

# --------------------------------------------------
# 9. WEATHER AND DEMAND RELATIONSHIP
# --------------------------------------------------

st.subheader("3. Weather and Demand Relationship")

weather_col1, weather_col2 = st.columns(2)

with weather_col1:

    fig3 = px.scatter(
        filtered,
        x="temperature",
        y="demand",
        title="Temperature vs Electricity Demand",
        labels={
            "temperature": "Temperature",
            "demand": "Electricity Demand"
        },
        opacity=0.5
    )

    fig3.update_layout(template="plotly_white")

    st.plotly_chart(fig3, use_container_width=True)

with weather_col2:

    fig4 = px.scatter(
        filtered,
        x="humidity",
        y="demand",
        title="Humidity vs Electricity Demand",
        labels={
            "humidity": "Humidity (%)",
            "demand": "Electricity Demand"
        },
        opacity=0.5
    )

    fig4.update_layout(template="plotly_white")

    st.plotly_chart(fig4, use_container_width=True)

st.caption(
    "These plots examine the association between weather "
    "variables and electricity demand."
)

st.divider()

# --------------------------------------------------
# 10. HISTORICAL DATASET
# --------------------------------------------------

st.subheader("4. Historical Dataset")

st.write(
    "Explore the historical electricity demand records "
    "used in this project."
)

st.write("Dataset shape:", filtered.shape)

# Display first 100 rows as HTML to avoid dataframe rendering
# issues on computers where PyArrow is blocked.

preview = filtered.head(100)

st.markdown(
    preview.to_html(index=False, border=0),
    unsafe_allow_html=True
)

csv_data = filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Historical Dataset (CSV)",
    data=csv_data,
    file_name="electricity_demand_historical_data.csv",
    mime="text/csv"
)

st.divider()

# --------------------------------------------------
# 11. PROJECT INFORMATION
# --------------------------------------------------

st.subheader("About This Project")

st.write(
    """
    This project analyses a synthetic hourly electricity
    demand dataset covering 2020 to 2024.

    The dashboard explores historical demand trends,
    monthly demand patterns, weather relationships and
    dataset characteristics.

    The analysed data will support the development and
    evaluation of machine learning forecasting models.
    """
)

st.info(
    "Dataset note: The electricity demand dataset used "
    "in this project is synthetic and is not actual "
    "measured electricity grid data."
)

st.caption(
    "M.Tech Machine Learning Project | Electricity Demand Forecasting"
)
