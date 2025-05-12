# adaptive_trader_v3.py — Uses adaptive volatility threshold

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# -------- Settings --------
data_file = '/Users/aaron/PycharmProjects/PythonProject1/Data/btc_1min_6mo.csv'
vol_file = '/Users/aaron/PycharmProjects/PythonProject1/Data/vol_forecast_v3.csv'
entry_fee_pct = 0.0025
exit_fee_pct = 0.0040

# -------- Load Data --------
price_df = pd.read_csv(data_file, parse_dates=['Datetime']).set_index('Datetime')
vol_df = pd.read_csv(vol_file, parse_dates=['Datetime']).set_index('Datetime')

price_df['Close'] = pd.to_numeric(price_df['Close'], errors='coerce')
price_df = price_df['Close'].dropna().to_frame()

# Merge forecast and price
df = price_df.merge(vol_df, how='inner', left_index=True, right_index=True)

# -------- Simulate Strategy --------
account_balance = 10000
position_open = False
entry_price = 0
position_size = 0

balances = []
trade_times = []
entries = []
exits = []
returns = []
total_entry_fees = 0
total_exit_fees = 0
total_trade_pnl = 0

for time, row in df.iterrows():
    price = row['Close']
    forecast_vol = row['forecast_vol']
    threshold = row['adaptive_thresh']

    if not position_open:
        if forecast_vol > threshold:
            entry_price = price
            position_size = account_balance  # All-in for now
            entry_fee = position_size * entry_fee_pct
            account_balance -= entry_fee
            total_entry_fees += entry_fee
            entries.append((time, price))
            position_open = True
    else:
        # Trailing stop logic — sell if price drops 2% from entry
        if price < entry_price * 0.98:
            pnl = position_size * ((price - entry_price) / entry_price)
            account_balance += pnl
            exit_fee = account_balance * exit_fee_pct
            account_balance -= exit_fee

            balances.append(account_balance)
            returns.append(pnl / position_size)
            trade_times.append(time)
            exits.append((time, price))
            total_trade_pnl += pnl
            total_exit_fees += exit_fee
            position_open = False

# Final output
final_balance = balances[-1] if balances else account_balance

# -------- Metrics --------
total_return = (final_balance - 10000) / 10000 * 100
sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252*24*4) if len(returns) > 1 else float('nan')
balance_series = pd.Series(balances, index=trade_times)
max_drawdown = ((balance_series - balance_series.cummax()) / balance_series.cummax()).min()

# -------- Plot --------
btc_normalized = df['Close'] / df['Close'].iloc[0] * 10000

plt.figure(figsize=(14, 6))
plt.plot(balance_series.index, balance_series.values, label='Strategy Balance', linestyle='-')
plt.plot(btc_normalized.index, btc_normalized.values, label='BTC Buy-and-Hold', linestyle='--')

entry_times, entry_prices = zip(*entries) if entries else ([], [])
exit_times, exit_prices = zip(*exits) if exits else ([], [])
plt.scatter(entry_times, entry_prices, color='green', marker='^', label='Entries')
plt.scatter(exit_times, exit_prices, color='red', marker='v', label='Exits')

plt.title('Adaptive Trader V3 vs BTC')
plt.xlabel('Date')
plt.ylabel('Value ($)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------- Results --------
print("\n--- Adaptive Strategy Results ---")
print(f"Final Balance: ${final_balance:.2f}")
print(f"Total Return: {total_return:.2f}%")
print(f"Total Trades: {len(balances)}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")

btc_start = df['Close'].iloc[0]
btc_end = df['Close'].iloc[-1]
btc_return = (btc_end - btc_start) / btc_start * 100
print("\n--- BTC Buy-and-Hold Benchmark ---")
print(f"BTC Start Price: ${btc_start:.2f}")
print(f"BTC End Price: ${btc_end:.2f}")
print(f"BTC Buy-and-Hold Return: {btc_return:.2f}%")

print("\n--- Fee Analysis ---")
total_fees = total_entry_fees + total_exit_fees
print(f"Total Entry Fees Paid: ${total_entry_fees:.2f}")
print(f"Total Exit Fees Paid: ${total_exit_fees:.2f}")
print(f"Total Fees Paid: ${total_fees:.2f}")
print(f"Total Trade PnL (excluding fees): ${total_trade_pnl:.2f}")

if total_trade_pnl + total_fees != 0:
    fee_impact = total_fees / (abs(total_trade_pnl) + total_fees) * 100
    print(f"Fees Consumed: {fee_impact:.2f}% of Total Movement")
else:
    print("Fees Consumed: 0.00% of Total Movement")