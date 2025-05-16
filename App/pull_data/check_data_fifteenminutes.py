import pandas as pd

"""
Script: check_btc_data_range.py

Description:
Loads 15-minute BTC data, converts timestamps from UTC to America/New_York timezone,
and prints key dataset stats including the first timestamp, last timestamp,
and total number of days covered.

Useful for validating time zone alignment and confirming dataset coverage before backtesting or analysis.
"""

# Load your existing BTC 15-min data
btc = pd.read_csv('../data/btc_15min_60d.csv', parse_dates=['Datetime'])

# Convert timestamps
btc['Datetime'] = pd.to_datetime(btc['Datetime'], utc=True)
btc = btc.set_index('Datetime')
btc = btc.tz_convert('America/New_York')

# Check the first and last timestamps
print(f"First timestamp: {btc.index.min()}")
print(f"Last timestamp: {btc.index.max()}")

# Calculate how many days
days_covered = (btc.index.max() - btc.index.min()).days
print(f"\nTotal days covered: {days_covered} days")