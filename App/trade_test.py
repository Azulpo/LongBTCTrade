import pandas as pd
import matplotlib.pyplot as plt

# -------- Load the 15-min BTC data --------
btc = pd.read_csv('../data/btc_15min_60d.csv', parse_dates=['Datetime'])

# Parse timestamps
btc['Datetime'] = pd.to_datetime(btc['Datetime'], utc=True)
btc = btc.set_index('Datetime')
btc = btc.tz_convert('America/New_York')

# Make sure Open, Close, Low are numeric
btc['Open'] = pd.to_numeric(btc['Open'], errors='coerce')
btc['Low'] = pd.to_numeric(btc['Low'], errors='coerce')
btc['Close'] = pd.to_numeric(btc['Close'], errors='coerce')

# -------- Build the Momentum Trading Strategy --------
starting_balance = 1000
account_balance = starting_balance
account_balances = []
trade_times = []

risk_per_trade = starting_balance
position_open = False
entry_price = None
stop_price = None
hold_candles = 4  # Hold for 4 candles (1 hour) if no stop hit

# -------- Simulate --------
btc['Signal'] = btc['Close'] > btc['Open']  # Simple green candle signal

i = 0
while i < len(btc) - 1:
    row = btc.iloc[i]
    next_row = btc.iloc[i + 1]

    # If no open position
    if not position_open:
        if row['Signal']:
            # Open a long at next candle's open
            entry_price = next_row['Open']
            stop_price = entry_price * (1 - 0.01)
            entry_index = i + 1  # Track entry index
            position_open = True
            i += 1
            continue

    # If in position
    if position_open:
        # Monitor up to hold_candles
        exit_index = min(entry_index + hold_candles, len(btc) - 1)
        stopped_out = False

        for j in range(entry_index, exit_index):
            low = btc.iloc[j]['Low']
            close = btc.iloc[j]['Close']
            if low <= stop_price:
                # Hit stop loss
                pnl = account_balance * (-0.01)
                account_balance += pnl
                trade_times.append(btc.index[j])
                account_balances.append(account_balance)
                position_open = False
                i = j  # continue from stopout candle
                stopped_out = True
                break

        if not stopped_out:
            # Exit after hold_candles at close price
            close_price = btc.iloc[exit_index]['Close']
            pnl_pct = (close_price - entry_price) / entry_price
            pnl = account_balance * pnl_pct
            account_balance += pnl
            trade_times.append(btc.index[exit_index])
            account_balances.append(account_balance)
            position_open = False
            i = exit_index  # jump forward

    i += 1

# -------- Plot PnL curve --------
plt.figure(figsize=(14, 6))
plt.plot(trade_times, account_balances, marker='', linestyle='-')
plt.title('Simple Momentum Strategy with 1% Stop Loss (BTC 15-min)')
plt.xlabel('Date')
plt.ylabel('Account Balance ($)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------- Final Stats --------
final_balance = account_balances[-1] if account_balances else starting_balance
total_return = (final_balance - starting_balance) / starting_balance * 100

print(f"\nFinal Balance: ${final_balance:.2f}")
print(f"Total Return: {total_return:.2f}% over {len(account_balances)} trades")