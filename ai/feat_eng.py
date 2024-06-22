import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the transactions data
transactions = pd.read_csv('data/transactions.csv')


# Convert value and gas_price to numeric types
transactions['value'] = pd.to_numeric(transactions['value'], errors='coerce')
transactions['gas_price'] = pd.to_numeric(transactions['gas_price'], errors='coerce')

# Initialize new proxy columns
transactions['dumb_flow_proxy'] = 0.0
transactions['urgency_proxy'] = 0.0
transactions['transaction_burstiness_proxy'] = 0.0
transactions['gas_price_volatility_proxy'] = 0.0
transactions['large_transaction_impact_proxy'] = 0.0

# Rolling window size
rolling_window_size = 50

# Calculate rolling statistics
transactions['rolling_avg_gas_price'] = transactions['gas_price'].rolling(window=rolling_window_size).mean()
transactions['rolling_std_gas_price'] = transactions['gas_price'].rolling(window=rolling_window_size).std()
transactions['rolling_avg_value'] = transactions['value'].rolling(window=rolling_window_size).mean()

# Function to calculate dumb flow proxy
def calculate_dumb_flow_proxy(row):
    if row['rolling_avg_gas_price'] > 0:
        gas_price_factor = row['gas_price'] / row['rolling_avg_gas_price']
    else:
        gas_price_factor = 1
    if row['rolling_avg_value'] > 0:
        value_factor = 1 if row['value'] == 0 else row['value'] / row['rolling_avg_value']
    else:
        value_factor = 1
    
    dumb_flow_score = gas_price_factor * value_factor
    return dumb_flow_score

# Function to calculate urgency proxy
def calculate_urgency_proxy(row):
    if row['rolling_avg_gas_price'] > 0:
        gas_price_factor = row['gas_price'] / row['rolling_avg_gas_price']
    else:
        gas_price_factor = 1
    urgency_score = gas_price_factor * row['value']
    return urgency_score

# Calculate transaction burstiness proxy
transactions['transaction_count'] = transactions.groupby('from').cumcount() + 1
transactions['transaction_burstiness_proxy'] = transactions['transaction_count'].rolling(window=rolling_window_size).mean()

# Calculate gas price volatility proxy
transactions['gas_price_volatility_proxy'] = transactions['rolling_std_gas_price']

# Function to calculate large transaction impact proxy
def calculate_large_transaction_impact_proxy(row):
    if row['rolling_avg_value'] > 0:
        large_tx_impact_score = row['value'] / row['rolling_avg_value']
    else:
        large_tx_impact_score = row['value']
    return large_tx_impact_score

# Apply functions to calculate proxy features
transactions['dumb_flow_proxy'] = transactions.apply(calculate_dumb_flow_proxy, axis=1)
transactions['urgency_proxy'] = transactions.apply(calculate_urgency_proxy, axis=1)
transactions['large_transaction_impact_proxy'] = transactions.apply(calculate_large_transaction_impact_proxy, axis=1)

# Normalize the proxy features
scaler = StandardScaler()
transactions[['dumb_flow_proxy', 'urgency_proxy', 'transaction_burstiness_proxy', 'gas_price_volatility_proxy', 'large_transaction_impact_proxy']] = scaler.fit_transform(
    transactions[['dumb_flow_proxy', 'urgency_proxy', 'transaction_burstiness_proxy', 'gas_price_volatility_proxy', 'large_transaction_impact_proxy']]
)

# Save the transactions with the new features to a CSV file
transactions.to_csv('data/transactions_with_proxies.csv', index=True)

# Display the first few rows with the new features
print(transactions.head())
