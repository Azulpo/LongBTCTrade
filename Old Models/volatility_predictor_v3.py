# volatility_predictor_v3.py — Adaptive threshold version

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# -------- Settings --------
data_file = '/Users/aaron/PycharmProjects/PythonProject1/Data/btc_1min_6mo.csv'
resample_interval = '15min'
entry_percentile = 60  # Adaptive percentile threshold

# -------- Load BTC Price Data --------
df = pd.read_csv(data_file, parse_dates=['Datetime'])
df = df.set_index('Datetime')
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df = df['Close'].dropna().to_frame()
df = df.resample(resample_interval).last().dropna()

# -------- Compute Rolling Volatility Windows --------
vol_windows = {
    'vol_1d': 96,
    'vol_3d': 288,
    'vol_9d': 864,
    'vol_14d': 1344
}

for name, window in vol_windows.items():
    df[name] = df['Close'].pct_change().rolling(window).std()

# -------- Load Optimal Weights --------
weights_df = pd.read_csv('/Users/aaron/PycharmProjects/PythonProject1/Data/volatility_weights_15m.csv')
weights = dict(zip(weights_df['window'], weights_df['weight']))

# -------- Forecast Volatility (Weighted Sum) --------
df['forecast_vol'] = sum(df[name] * weights.get(name, 0) for name in vol_windows)

# -------- Adaptive Threshold Using Rolling Percentile --------
thresh_window = 500  # Roughly 5 days
rolling_threshold = df['forecast_vol'].rolling(window=thresh_window).quantile(entry_percentile / 100.0)
df['adaptive_thresh'] = rolling_threshold

# -------- Plot --------
fig, ax1 = plt.subplots(figsize=(15, 8))
ax1.set_xlabel('Date')
ax1.set_ylabel('BTC Price ($)', color='tab:blue')
ax1.plot(df.index, df['Close'], color='tab:blue', label='BTC Price')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Forecast Volatility', color='tab:red')
ax2.plot(df.index, df['forecast_vol'], color='tab:red', alpha=0.6, label='Forecast Volatility')
ax2.plot(df.index, df['adaptive_thresh'], color='green', linestyle='--', label='Adaptive Threshold')
ax2.tick_params(axis='y', labelcolor='tab:red')

fig.tight_layout()
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
plt.title('BTC Price vs Forecast Volatility (with Adaptive Threshold)')
plt.grid(True)
plt.show()

# Save for strategy usage
df[['forecast_vol', 'adaptive_thresh']].dropna().to_csv('/Users/aaron/PycharmProjects/PythonProject1/Data/vol_forecast_v3.csv')
print("\n✅ Saved forecast_vol and adaptive_thresh to vol_forecast_v3.csv")
