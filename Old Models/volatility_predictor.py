# adaptive_trader.py – restored to prior version with static sizing, simple threshold

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# -------- File Paths --------
data_file = '/Users/aaron/PycharmProjects/PythonProject1/Data/btc_1min_6mo.csv'
weights_file = '/Users/aaron/PycharmProjects/PythonProject1/Data/volatility_weights_15m.csv'

# -------- Parameters --------
entry_threshold = 0.0005
starting_balance = 10000
entry_fee_pct = 0.0025
exit_fee_pct = 0.0040

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

# -------- Trading Simulation --------
account_balance = starting_balance
position_open = False
entry_price = 0

balances, trade_times, trade_pnls = [], [], []
entry_fees, exit_fees = [], []

for current_time, row in df.iterrows():
    if pd.isna(row['forecast_vol']):
        continue

    if not position_open:
        if row['forecast_vol'] > entry_threshold:
            position_size = account_balance
            entry_fee = position_size * entry_fee_pct
            account_balance -= entry_fee

            entry_price = row['Close']
            entry_fees.append(entry_fee)
            position_open = True
    else:
        pnl_pct = (row['Close'] - entry_price) / entry_price
        if pnl_pct > entry_threshold:
            gain = account_balance * pnl_pct
            exit_fee = (account_balance + gain) * exit_fee_pct
            account_balance += gain - exit_fee
            balances.append(account_balance)
            trade_times.append(current_time)
            trade_pnls.append(pnl_pct)
            exit_fees.append(exit_fee)
            position_open = False

# -------- Metrics --------
final_balance = account_balance
returns = np.array(trade_pnls)
sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if len(returns) > 1 else np.nan
balance_series = pd.Series(balances, index=trade_times)
rolling_max = balance_series.cummax()
drawdown = (balance_series - rolling_max) / rolling_max
max_drawdown = drawdown.min() if not drawdown.empty else np.nan

btc_start_price = df['Close'].iloc[0]
btc_end_price = df['Close'].iloc[-1]
btc_return = (btc_end_price - btc_start_price) / btc_start_price * 100

total_entry_fees = sum(entry_fees)
total_exit_fees = sum(exit_fees)
total_fees = total_entry_fees + total_exit_fees
total_trade_pnl_excl_fees = (final_balance + total_fees - starting_balance)
fee_impact_pct = (total_fees / (abs(total_trade_pnl_excl_fees) + total_fees)) * 100 if (total_trade_pnl_excl_fees + total_fees) != 0 else 0

# -------- Output --------
print("\n--- Adaptive Strategy Results ---")
print(f"Final Balance: ${final_balance:.2f}")
print(f"Total Return: {(final_balance - starting_balance) / starting_balance * 100:.2f}%")
print(f"Total Trades: {len(balances)}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")

print("\n--- BTC Buy-and-Hold Benchmark ---")
print(f"BTC Start Price: ${btc_start_price:.2f}")
print(f"BTC End Price: ${btc_end_price:.2f}")
print(f"BTC Buy-and-Hold Return: {btc_return:.2f}%")

print("\n--- Fee Analysis ---")
print(f"Total Entry Fees Paid: ${total_entry_fees:.2f}")
print(f"Total Exit Fees Paid: ${total_exit_fees:.2f}")
print(f"Total Fees Paid: ${total_fees:.2f}")
print(f"Total Trade PnL (excluding fees): ${total_trade_pnl_excl_fees:.2f}")
print(f"Fees Consumed: {fee_impact_pct:.2f}% of Total Movement")

# -------- Plot --------
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


