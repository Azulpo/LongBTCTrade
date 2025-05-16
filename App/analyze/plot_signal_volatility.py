# plot_signal_volatility.py – Visualize forecast volatility and price over time

import pandas as pd
import matplotlib.pyplot as plt
import os

"""
Script: plot_signal_volatility.py

Description:
Visualizes Bitcoin price alongside a custom forecasted volatility signal using 15-minute interval data.
Volatility is calculated as rolling standard deviation over multiple time windows (1d, 3d, 9d, 14d),
and weighted based on values provided in an external CSV file.

Key Features:
- Resamples raw BTC data to 15-minute intervals
- Computes and combines multiple volatility windows into a single forecast
- Plots BTC price (left axis) vs. forecast volatility (right axis)
- Marks a horizontal entry threshold (e.g., 0.0005) for strategy visualization

Intended Use:
This script is useful for visually analyzing low-volatility environments and fine-tuning volatility-based trading entry logic.
"""

# -------- File Paths --------
data_file = '/Data/btc_1min_6mo.csv'
weights_file = '/Data/volatility_weights_15m.csv'

# -------- Load Data --------
df = pd.read_csv(data_file, parse_dates=['Datetime'])
df = df.set_index('Datetime')
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df = df['Close'].dropna().to_frame()
df = df.resample('15min').last().dropna()

# -------- Volatility Forecast --------
weights_df = pd.read_csv(weights_file)
vol_windows = {'vol_1d': 96, 'vol_3d': 288, 'vol_9d': 864, 'vol_14d': 1344}
weights = dict(zip(weights_df['window'], weights_df['weight']))

for label, window in vol_windows.items():
    df[label] = df['Close'].pct_change().rolling(window).std()

df['forecast_vol'] = sum(df[label] * weights.get(label, 0) for label in vol_windows)

# -------- Plot --------
fig, ax1 = plt.subplots(figsize=(15, 8))

# BTC Price
color = 'tab:blue'
ax1.set_xlabel('Date')
ax1.set_ylabel('BTC Price ($)', color=color)
ax1.plot(df.index, df['Close'], color=color, label='BTC Price')
ax1.tick_params(axis='y', labelcolor=color)

# Forecast Volatility
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Forecast Volatility', color=color)
ax2.plot(df.index, df['forecast_vol'], color=color, label='Forecast Volatility', alpha=0.6)
ax2.axhline(y=0.0005, color='green', linestyle='--', label='Entry Threshold')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
plt.title('BTC Price vs Forecast Volatility (15-min intervals)')
plt.grid(True)
plt.show()
