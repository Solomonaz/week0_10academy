import pandas as pd
import numpy as np
from scipy import stats


def load_data():
    # Load your data here
    benin_data = pd.read_csv('src/benin-malanville.csv')
    togo_data = pd.read_csv('src/togo-dapaong_qc.csv')
    sierraleone_data = pd.read_csv('src/sierraleone-bumbuna.csv')

    # Combine datasets for comprehensive analysis
    data = pd.concat([benin_data, togo_data, sierraleone_data], ignore_index=True)
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')  # Ensure correct conversion
    return data


def calculate_summary_statistics(df):
    summary = df.describe()
    summary.loc['median'] = df.median()
    return summary


def get_anomalies(df):
    anomalies = {}
    for column in ['GHI', 'DNI', 'DHI']:
        # Calculate Z-scores
        z_scores = np.abs(stats.zscore(df[column]))
        # Identify anomalies
        anomalies[column] = df[z_scores > 3]  # Filter directly without using iloc
    return anomalies