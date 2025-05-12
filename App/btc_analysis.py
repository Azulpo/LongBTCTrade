import pandas as pd

btc = pd.read_csv('../data/btc_15min_60d.csv', parse_dates=['Datetime'])
btc['Datetime'] = pd.to_datetime(btc['Datetime'], utc=True)
btc = btc.set_index('Datetime')
btc = btc.tz_convert('America/New_York')

btc['Open'] = pd.to_numeric(btc['Open'], errors='coerce')
btc['Close'] = pd.to_numeric(btc['Close'], errors='coerce')

# -------- Improved Overnight Analysis --------
results = []
dates = btc.index.normalize().unique()

for date in dates[:-1]:
    try:
        start_target = date + pd.Timedelta(hours=16)  # 4:00 PM
        end_target = (date + pd.Timedelta(days=1)) + pd.Timedelta(hours=9, minutes=30)  # 9:30 AM next day

        # Find nearest time within a 15-minute window
        open_window = btc.loc[(btc.index >= start_target - pd.Timedelta(minutes=15)) &
                              (btc.index <= start_target + pd.Timedelta(minutes=15))]
        close_window = btc.loc[(btc.index >= end_target - pd.Timedelta(minutes=15)) &
                               (btc.index <= end_target + pd.Timedelta(minutes=15))]

        if not open_window.empty and not close_window.empty:
            open_price = open_window.iloc[0]['Open']
            close_price = close_window.iloc[0]['Close']
            pct_change = (close_price - open_price) / open_price * 100

            results.append({
                'Start Date': start_target.date(),
                'Open Price (~4PM)': open_price,
                'Close Price (~9:30AM)': close_price,
                'Pct Change (%)': pct_change
            })
    except Exception as e:
        print(f"Error processing {date}: {e}")
        continue

# -------- Save Results --------
overnight_moves = pd.DataFrame(results)

if not overnight_moves.empty:
    overnight_moves.to_csv('overnight_btc_moves.csv', index=False)

    print("\n--- Overnight Bitcoin Move Analysis ---\n")
    print(overnight_moves.head())

    print(f"\nAverage Overnight Move: {overnight_moves['Pct Change (%)'].mean():.4f}%")
    print(f"Standard Deviation (Volatility) of Overnight Move: {overnight_moves['Pct Change (%)'].std():.4f}%")
    print(f"Positive Nights (% Up): {(overnight_moves['Pct Change (%)'] > 0).mean() * 100:.2f}%")
    print(f"Negative Nights (% Down): {(overnight_moves['Pct Change (%)'] < 0).mean() * 100:.2f}%")
    print("\nResults saved to 'overnight_btc_moves.csv'")
else:
    print("\nNo overnight data available — check your dataset.")