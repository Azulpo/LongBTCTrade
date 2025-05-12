import pandas as pd
import numpy as np
import os
import random
from datetime import timedelta

# -------- Settings --------
base_path = '/Users/aaron/PycharmProjects/PythonProject1'
data_file = os.path.join(base_path, 'Data', 'btc_1min_6mo.csv')
output_file = os.path.join(base_path, 'Data', 'parameter_search_results.csv')

starting_balance = 10000
entry_fee_pct = 0.0025
exit_fee_pct = 0.0040
lookback_minutes = 15  # We keep 15 min momentum window for now

num_trials = 500  # Number of random parameter combinations

# -------- Load Data --------
btc = pd.read_csv(data_file, parse_dates=['Datetime'])
btc = btc.set_index('Datetime')

btc['Open'] = pd.to_numeric(btc['Open'], errors='coerce')
btc['Close'] = pd.to_numeric(btc['Close'], errors='coerce')
btc['Low'] = pd.to_numeric(btc['Low'], errors='coerce')
btc = btc.dropna()

btc['Return'] = btc['Close'].pct_change()
btc['Momentum'] = btc['Return'].rolling(lookback_minutes).sum()

# -------- Random Search --------
results = []

for trial in range(num_trials):
    # Random parameters
    min_momentum_threshold = random.uniform(0.001, 0.01)  # +0.1% to +1.0%
    stop_loss_pct = random.uniform(0.01, 0.05)             # 1% to 5%
    max_hold_minutes = random.randint(30, 4320)            # 30 mins to 3 days

    account_balance = starting_balance
    balances = []
    trade_times = []
    returns = []

    position_open = False
    entry_price = None
    entry_time = None

    for current_time, row in btc.iterrows():
        if not position_open:
            if row['Momentum'] >= min_momentum_threshold:
                # Enter long
                entry_price = row['Open']
                stop_price = entry_price * (1 - stop_loss_pct)
                entry_time = current_time
                entry_fee = account_balance * entry_fee_pct
                account_balance -= entry_fee
                position_open = True
        else:
            # Manage open position
            low_price = row['Low']
            if low_price <= stop_price:
                # Stop hit
                pnl = account_balance * (-stop_loss_pct)
                account_balance += pnl
                exit_fee = account_balance * exit_fee_pct
                account_balance -= exit_fee

                balances.append(account_balance)
                returns.append(pnl / (account_balance + (-pnl)))
                trade_times.append(current_time)
                position_open = False
                continue

            elapsed_minutes = (current_time - entry_time).total_seconds() / 60
            if elapsed_minutes >= max_hold_minutes:
                # Exit at Close
                close_price = row['Close']
                pnl_pct = (close_price - entry_price) / entry_price
                pnl = account_balance * pnl_pct
                account_balance += pnl
                exit_fee = account_balance * exit_fee_pct
                account_balance -= exit_fee

                balances.append(account_balance)
                returns.append(pnl / (account_balance - pnl))
                trade_times.append(current_time)
                position_open = False

    if balances:
        final_balance = balances[-1]
    else:
        final_balance = starting_balance

    total_return = (final_balance - starting_balance) / starting_balance * 100

    # Sharpe Ratio
    returns = np.array(returns)
    if len(returns) > 1:
        sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252*24*4)
    else:
        sharpe_ratio = np.nan

    # Max Drawdown
    balance_series = pd.Series(balances, index=trade_times)
    if not balance_series.empty:
        rolling_max = balance_series.cummax()
        drawdown = (balance_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
    else:
        max_drawdown = np.nan

    results.append({
        'min_momentum_threshold': min_momentum_threshold,
        'stop_loss_pct': stop_loss_pct,
        'max_hold_minutes': max_hold_minutes,
        'final_balance': final_balance,
        'total_return_pct': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown_pct': max_drawdown,
        'total_trades': len(balances)
    })

# -------- Save Results --------
results_df = pd.DataFrame(results)
results_df.to_csv(output_file, index=False)

print("\n✅ Parameter Search Complete!")
print(f"Saved {len(results)} results to {output_file}")