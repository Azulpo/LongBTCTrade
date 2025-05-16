# adaptive_trader.py

"""
adaptive_trader.py

This looks at 6 months of BTC data. Not useful! Backtest a volatility-driven BTC trading strategy using forecasted daily volatility
from multiple rolling windows. Enters long positions when volatility exceeds a
threshold, exits on profit greater than threshold. Tracks balance, Sharpe, and
drawdown. Includes fees and plots performance vs buy-and-hold BTC.

Dependencies:
- pandas
- numpy
- matplotlib
- Data file: btc_1min_6mo.csv (resampled to 15m)
- Weights file: volatility_weights_15m.csv (from volatility_predictor.py)

Author: [Zulpo]
Last updated: [Insert Date]
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# -------- File Paths --------
data_file = "/Data/btc_1min_6mo.csv"
weights_file = '/Data/volatility_weights_15m.csv'

# -------- Parameters --------
entry_threshold = 0.0005   # Now using daily vol forecast scale
starting_balance = 10000
entry_fee_pct = 0.0025
exit_fee_pct = 0.0040

# -------- Load Data & Resample to 15-min Bars --------
df = pd.read_csv(data_file, parse_dates=['Datetime'])
df = df.set_index('Datetime')
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df = df['Close'].dropna().to_frame()
df = df.resample('15min').last().dropna()

# -------- Load Volatility Weights --------
weights_df = pd.read_csv(weights_file)
vol_windows = {
    'vol_1d': 96,
    'vol_3d': 288,
    'vol_9d': 864,
    'vol_14d': 1344
}
weights = dict(zip(weights_df['window'], weights_df['weight']))

# -------- Compute Rolling Volatility Features --------
for label, window in vol_windows.items():
    df[label] = df['Close'].pct_change().rolling(window).std()

# -------- Forecast Daily Volatility --------
df['forecast_vol'] = sum(df[label] * weights.get(label, 0) for label in vol_windows)

# -------- Simulate Adaptive Strategy --------
account_balance = starting_balance
position_open = False
entry_price = 0

balances = []
trade_times = []
trade_pnls = []

for current_time, row in df.iterrows():
    if pd.isna(row['forecast_vol']):
        continue

    if not position_open:
        if row['forecast_vol'] > entry_threshold:
            entry_price = row['Close']
            entry_fee = account_balance * entry_fee_pct
            account_balance -= entry_fee
            position_open = True
    else:
        pnl_pct = (row['Close'] - entry_price) / entry_price
        if pnl_pct > entry_threshold:
            gain = account_balance * pnl_pct
            exit_fee = (account_balance + gain) * exit_fee_pct
            account_balance += gain - exit_fee
            balances.append(account_balance)
            trade_pnls.append(pnl_pct)
            trade_times.append(current_time)
            position_open = False

# -------- Final Output --------
final_balance = account_balance
returns = np.array(trade_pnls)
sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if len(returns) > 1 else np.nan
balance_series = pd.Series(balances, index=trade_times)
rolling_max = balance_series.cummax()
drawdown = (balance_series - rolling_max) / rolling_max
max_drawdown = drawdown.min() if not drawdown.empty else np.nan

print("\n--- Adaptive Strategy Results ---")
print(f"Final Balance: ${final_balance:.2f}")
print(f"Total Trades: {len(balances)}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")

# -------- Plot Performance --------
if not balance_series.empty:
    btc_price = df.loc[balance_series.index, 'Close']
    btc_normalized = btc_price / btc_price.iloc[0] * starting_balance

    plt.figure(figsize=(14,6))
    plt.plot(balance_series.index, balance_series.values, label='Strategy Balance')
    plt.plot(btc_normalized.index, btc_normalized.values, label='BTC Buy-and-Hold', linestyle='--')
    plt.title('Adaptive Trader vs BTC')
    plt.xlabel('Date')
    plt.ylabel('Value ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
