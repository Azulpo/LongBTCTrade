import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

"""
Script: filtered_reversal_strategy_modular.py

Description:
A flexible BTC long-only reversal strategy backtester using 1-minute data.
The strategy enters trades when 15-minute momentum is negative, but only if a configurable number of filters pass.

Supported Filters:
1. Chop Filter – Enters only in low-volatility environments
2. Momentum Filter – Confirms that short-term momentum is improving
3. Trend Filter – Confirms that price is in a broader uptrend (4-hour SMA)

You can toggle each filter independently and set how many must pass for a trade to trigger (default: at least 2 of 3).
Exits are based on:
- 3.47% trailing stop from peak
- 3% drop over the past 5 minutes

Outputs:
- Performance metrics (return, Sharpe, drawdown, trades)
- Fee analysis
- Strategy vs BTC buy-and-hold comparison chart

This script is designed for rapid experimentation with filter combinations to improve reversal entry quality.
"""

# -------- Settings --------
base_path = '/'
data_file = os.path.join(base_path, 'Data', 'btc_1min_6mo.csv')

starting_balance = 10000
entry_fee_pct = 0.0025
exit_fee_pct = 0.0040

# Reversal Strategy Parameters
min_momentum_threshold = -0.0082
stop_loss_pct = 0.0347
lookback_minutes = 15

# Filters
use_chop_filter = True
use_momentum_filter = True
use_trend_filter = True
required_filters_pass = 2  # Flexible mode: at least 2 of 3 must pass

chop_volatility_threshold = 0.0015  # 0.15% stddev over 1h returns
momentum_slope_lookback = 5  # 5-minute slope check
trend_lookback_minutes = 240  # 4-hour trend

# -------- Load Data --------
btc = pd.read_csv(data_file, parse_dates=['Datetime'])
btc = btc.set_index('Datetime')

btc['Open'] = pd.to_numeric(btc['Open'], errors='coerce')
btc['Close'] = pd.to_numeric(btc['Close'], errors='coerce')
btc['Low'] = pd.to_numeric(btc['Low'], errors='coerce')
btc = btc.dropna()

# -------- Create Features --------
btc['Return'] = btc['Close'].pct_change()
btc['Momentum_15m'] = btc['Return'].rolling(lookback_minutes).sum()
btc['Volatility_1h'] = btc['Return'].rolling(60).std()
btc['Momentum_5m'] = btc['Return'].rolling(momentum_slope_lookback).sum()
btc['MA_4h'] = btc['Close'].rolling(trend_lookback_minutes).mean()

btc['is_15min_mark'] = btc.index.minute % 15 == 0

# -------- Simulate Trading --------
account_balance = starting_balance
balances = []
trade_times = []
returns = []

position_open = False
entry_price = None
peak_price = None
total_entry_fee_paid = 0
total_exit_fee_paid = 0
total_trade_pnl = 0

for current_time, row in btc.iterrows():
    if not position_open:
        if row['is_15min_mark'] and row['Momentum_15m'] < min_momentum_threshold:
            filters_passed = 0

            # Chop Filter
            if use_chop_filter:
                if row['Volatility_1h'] < chop_volatility_threshold:
                    filters_passed += 1

            # Momentum Continuation Filter
            if use_momentum_filter:
                if row['Momentum_5m'] > row['Momentum_15m']:  # momentum improving
                    filters_passed += 1

            # Macro Trend Filter
            if use_trend_filter:
                if row['MA_4h'] < row['Close']:  # close above 4H MA
                    filters_passed += 1

            if filters_passed >= required_filters_pass:
                entry_price = row['Open']
                peak_price = entry_price
                entry_fee = account_balance * entry_fee_pct
                account_balance -= entry_fee
                total_entry_fee_paid += entry_fee
                position_open = True
    else:
        current_price = row['Close']
        peak_price = max(peak_price, current_price)

        # Trailing Stop
        drawdown_from_peak = (current_price - peak_price) / peak_price

        # 5-minute crash
        five_minutes_ago = current_time - pd.Timedelta(minutes=5)
        if five_minutes_ago in btc.index:
            price_5min_ago = btc.loc[five_minutes_ago, 'Close']
            drop_5min = (current_price - price_5min_ago) / price_5min_ago
        else:
            drop_5min = 0

        if drawdown_from_peak <= -stop_loss_pct or drop_5min <= -0.03:
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

returns = np.array(returns)
if len(returns) > 1 and np.std(returns) > 0:
    sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252*24*4)
else:
    sharpe_ratio = np.nan

balance_series = pd.Series(balances, index=trade_times)
rolling_max = balance_series.cummax()
drawdown = (balance_series - rolling_max) / rolling_max
max_drawdown = drawdown.min()

btc_price_series = btc['Close']
btc_normalized = btc_price_series / btc_price_series.iloc[0] * starting_balance

plt.figure(figsize=(14,6))
plt.plot(balance_series.index, balance_series.values, label='Strategy Balance', linestyle='-')
plt.plot(btc_price_series.index, btc_normalized.values, label='BTC Buy-and-Hold', linestyle='--')
plt.title('Filtered Reversal Strategy vs BTC')
plt.xlabel('Date')
plt.ylabel('Value ($)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

btc_start_price = btc_price_series.iloc[0]
btc_end_price = btc_price_series.iloc[-1]
btc_return = (btc_end_price - btc_start_price) / btc_start_price * 100

print("\n--- Filtered Strategy Results ---")
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
