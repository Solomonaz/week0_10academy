import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the datasets
try:
    togo_data = pd.read_csv('src/togo-dapaong_qc.csv')
    sierraleone_data = pd.read_csv('src/sierraleone-bumbuna.csv')
    benin_data = pd.read_csv('src/benin-malanville.csv')
except FileNotFoundError as e:
    print(f"File not found: {e}")
    exit()

# Combine datasets for a comprehensive analysis
data = pd.concat([togo_data, sierraleone_data, benin_data], ignore_index=True)

# Check for required columns
required_columns = ['GHI', 'DNI', 'DHI', 'Tamb', 'Timestamp', 'WD', 'WS']
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

# Function to calculate summary statistics
def calculate_summary_statistics(df):
    summary = df.describe()  # Basic statistics
    summary.loc['median'] = df.median(numeric_only=True)  # Add median
    return summary

# Calculate summary statistics
summary_stats = calculate_summary_statistics(data)
print("Summary Statistics:\n", summary_stats)

# Data Quality Check
# Check for missing values
missing_values = data.isnull().sum()
print("Missing values in each column:\n", missing_values)

# Handling missing values (impute or drop)
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
data.dropna(subset=['Timestamp'], inplace=True)

# Outlier detection using Z-scores for GHI, DNI, DHI
for column in ['GHI', 'DNI', 'DHI']:
    if column in data.columns:
        z_scores = np.abs(stats.zscore(data[column].dropna()))
        data = data[z_scores < 3]

# Time Series Analysis
data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')
data.dropna(subset=['Timestamp'], inplace=True)

plt.figure(figsize=(12, 6))
plt.plot(data['Timestamp'], data['GHI'], label='GHI')
plt.plot(data['Timestamp'], data['DNI'], label='DNI')
plt.plot(data['Timestamp'], data['Tamb'], label='Tamb')
plt.xlabel('Time')
plt.ylabel('Values')
plt.title('Time Series Analysis of GHI, DNI, and Tamb')
plt.legend()
plt.show()

# Correlation Analysis
correlation_matrix = data[['GHI', 'DNI', 'DHI', 'Tamb']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Wind Analysis: Wind Rose
def wind_rose(data):
    if 'WD' in data.columns and data['WD'].dtype in [np.float64, np.int64]:
        plt.figure(figsize=(8, 8))
        plt.subplot(projection='polar')
        plt.hist(np.radians(data['WD']), bins=30, color='blue', alpha=0.6)
        plt.title('Wind Rose')
        plt.show()
    else:
        print("WD column is missing or not numeric.")

wind_rose(data)

# Histograms
plt.figure(figsize=(12, 6))
for column in ['GHI', 'DNI', 'DHI', 'WS']:
    if column in data.columns:
        plt.subplot(2, 2, ['GHI', 'DNI', 'DHI', 'WS'].index(column) + 1)
        plt.hist(data[column], bins=30, alpha=0.7, color='blue')
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Z-Score Analysis for anomalies
# Z-Score Analysis for anomalies
for column in ['GHI', 'DNI', 'DHI']:
    if column in data.columns:
        data['z_score_' + column] = np.abs(stats.zscore(data[column].dropna()))
        anomalies = data[data['z_score_' + column] > 3]
        print(f'Anomalies in {column}: {len(anomalies)}')

# Data Cleaning
# Remove erroneous data (example: GHI < 0)
data = data[data['GHI'] >= 0]

# Save cleaned data
data.to_csv('cleaned_data.csv', index=False)
print("Cleaned data saved to 'cleaned_data.csv'")