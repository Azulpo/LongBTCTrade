import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# -------- Settings --------
base_path = '/Users/aaron/PycharmProjects/PythonProject1'
data_file = os.path.join(base_path, 'Data', 'btc_1min_6mo.csv')

starting_balance = 10000         # Start with $10,000
entry_fee_pct = 0.00025           # 0.25% maker fee (entry)
exit_fee_pct = 0.00040            # 0.40% taker fee (exit)
lookback_minutes = 15            # 15-minute momentum window

# -------- Load Data --------
btc = pd.read_csv(data_file, parse_dates=['Datetime'])
btc = btc.set_index('Datetime')

btc['Open'] = pd.to_numeric(btc['Open'], errors='coerce')
btc['Close'] = pd.to_numeric(btc['Close'], errors='coerce')
btc['Low'] = pd.to_numeric(btc['Low'], errors='coerce')
btc = btc.dropna()

# -------- Identify Signal --------
btc['Return'] = btc['Close'].pct_change()
btc['Momentum'] = btc['Return'].rolling(lookback_minutes).sum()

# -------- Simulate Trading --------
account_balance = starting_balance
balances = []
trade_times = []
returns = []

position_open = False
entry_price = None
entry_time = None
peak_price = None  # Track peak after entry

total_entry_fee_paid = 0
total_exit_fee_paid = 0
total_trade_pnl = 0

# Identify 15-minute timestamps
btc['is_15min_mark'] = btc.index.minute % 15 == 0

# Set 15-minute momentum signal (we want NEGATIVE now)
btc['Signal15m'] = btc['Momentum'] < 0

for current_time, row in btc.iterrows():
    if not position_open:
        if row['is_15min_mark'] and row['Signal15m']:
            # Enter long with maker order
            entry_price = row['Open']
            entry_time = current_time
            peak_price = entry_price
            entry_fee = account_balance * entry_fee_pct
            account_balance -= entry_fee
            total_entry_fee_paid += entry_fee
            position_open = True
    else:
        # Update peak price
        peak_price = max(peak_price, row['Close'])

        # Check trailing stop (3% from peak)
        current_price = row['Close']
        drawdown_from_peak = (current_price - peak_price) / peak_price

        # Calculate 5-minute fast drop
        five_minutes_ago = current_time - pd.Timedelta(minutes=5)
        if five_minutes_ago in btc.index:
            price_5min_ago = btc.loc[five_minutes_ago, 'Close']
            drop_5min = (current_price - price_5min_ago) / price_5min_ago
        else:
            drop_5min = 0  # No data yet, don't trigger

        if drawdown_from_peak <= -0.03 or drop_5min <= -0.03:
            # Sell condition triggered
            pnl_pct = (current_price - entry_price) / entry_price
            pnl = account_balance * pnl_pct
            account_balance += pnl
            exit_fee = account_balance * exit_fee_pct
            account_balance -= exit_fee

            total_trade_pnl += pnl
            total_exit_fee_paid += exit_fee

            balances.append(account_balance)
            returns.append(pnl / (account_balance - pnl))
            trade_times.append(current_time)
            position_open = False

# -------- Results --------
if balances:
    final_balance = balances[-1]
else:
    final_balance = starting_balance

total_return = (final_balance - starting_balance) / starting_balance * 100

# -------- Calculate Sharpe Ratio --------
returns = np.array(returns)
if len(returns) > 1 and np.std(returns) > 0:
    sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252*24*4)
else:
    sharpe_ratio = np.nan

# -------- Calculate Max Drawdown --------
balance_series = pd.Series(balances, index=trade_times)
rolling_max = balance_series.cummax()
drawdown = (balance_series - rolling_max) / rolling_max
max_drawdown = drawdown.min()

# -------- Plot Strategy vs BTC --------
# (Fixed) Always use full BTC close prices for stable buy-and-hold benchmark
btc_price_series = btc['Close']
btc_normalized = btc_price_series / btc_price_series.iloc[0] * starting_balance

plt.figure(figsize=(14,6))
plt.plot(balance_series.index, balance_series.values, label='Strategy Balance', linestyle='-')
plt.plot(btc_price_series.index, btc_normalized.values, label='BTC Buy-and-Hold', linestyle='--')
plt.title('Strategy vs BTC Buy-and-Hold Comparison')
plt.xlabel('Date')
plt.ylabel('Value ($)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------- Final Output --------
btc_start_price = btc_price_series.iloc[0]
btc_end_price = btc_price_series.iloc[-1]
btc_return = (btc_end_price - btc_start_price) / btc_start_price * 100

print("\n--- Strategy Results ---")
print(f"Final Balance: ${final_balance:.2f}")
print(f"Total Return: {total_return:.2f}%")
print(f"Total Trades: {len(balances)}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")

print("\n--- BTC Buy-and-Hold Benchmark ---")
print(f"BTC Start Price: ${btc_start_price:.2f}")
print(f"BTC End Price: ${btc_end_price:.2f}")
print(f"BTC Buy-and-Hold Return: {btc_return:.2f}%")

print("\n--- Fee Analysis ---")
total_fee_paid = total_entry_fee_paid + total_exit_fee_paid
print(f"Total Entry Fees Paid: ${total_entry_fee_paid:.2f}")
print(f"Total Exit Fees Paid: ${total_exit_fee_paid:.2f}")
print(f"Total Fees Paid: ${total_fee_paid:.2f}")
print(f"Total Trade PnL (excluding fees): ${total_trade_pnl:.2f}")

print("\n--- Dataset Time Range ---")
print(f"Start Date: {btc.index.min()}")
print(f"End Date: {btc.index.max()}")

if total_trade_pnl + total_fee_paid != 0:
    fee_impact_pct = (total_fee_paid / (abs(total_trade_pnl) + total_fee_paid)) * 100
    print(f"Fees Consumed: {fee_impact_pct:.2f}% of Total Movement")
else:
    print("No trades to calculate fee impact.")