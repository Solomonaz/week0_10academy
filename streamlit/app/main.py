import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data, calculate_summary_statistics, get_anomalies

# Load data
data = load_data()

# Ensure Timestamp is correctly parsed
data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')
data = data.dropna(subset=['Timestamp'])  # Drop rows with invalid timestamps

# Dashboard Title
st.title("Weather Data Dashboard")

# Summary Statistics
summary_stats = calculate_summary_statistics(data)
st.subheader("Summary Statistics")
st.write(summary_stats)

# Interactive Slider for date selection
start_date, end_date = st.slider(
    "Select date range:",
    min_value=data['Timestamp'].min().date(),
    max_value=data['Timestamp'].max().date(),
    value=(data['Timestamp'].min().date(), data['Timestamp'].max().date())
)

# Filter data by date range
filtered_data = data[(data['Timestamp'].dt.date >= start_date) & (data['Timestamp'].dt.date <= end_date)]

# Time Series Analysis
st.subheader("Time Series Analysis")
if not filtered_data.empty:
    st.line_chart(filtered_data.set_index('Timestamp')[['GHI', 'DNI', 'Tamb']])
else:
    st.warning("No data available for the selected date range.")

# Correlation Analysis
st.subheader("Correlation Matrix")
if not filtered_data.empty:
    correlation_matrix = filtered_data[['GHI', 'DNI', 'DHI', 'Tamb']].corr()
    fig, ax = plt.subplots()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
else:
    st.warning("No data available to display correlation matrix.")

# Anomaly Detection
st.subheader("Anomalies in Data")
if not filtered_data.empty:
    anomalies = get_anomalies(filtered_data)
    for column, anomaly_data in anomalies.items():
        st.write(f"Anomalies in {column}:")
        st.write(anomaly_data)
else:
    st.warning("No data available for anomaly detection.")

# Deploy the app
st.markdown("### Deploying to Streamlit Community Cloud")
st.markdown("Follow the [Streamlit deployment guide](https://docs.streamlit.io/streamlit-cloud/get-started) to deploy your app.")
