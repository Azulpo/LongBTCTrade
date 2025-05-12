import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------- Load BTC Data --------
data_file = "/Users/aaron/PycharmProjects/PythonProject1/Data/btc_binance_1d.csv"
btc_df = pd.read_csv(data_file)
btc_df["Datetime"] = pd.to_datetime(btc_df["Datetime"])
btc_df = btc_df.set_index("Datetime")
btc_df = btc_df["2023-01-01":]  # Filter from Jan 2023 onward

# -------- Feature Engineering --------
btc_df["Return"] = btc_df["Close"].pct_change()
btc_df["MA_14"] = btc_df["Close"].rolling(14).mean()
btc_df["MA_30"] = btc_df["Close"].rolling(30).mean()
btc_df["ROC"] = btc_df["Close"].pct_change(periods=5)
btc_df["Slope_MA"] = btc_df["MA_14"] - btc_df["MA_30"]
btc_df["Volume_Surge"] = btc_df["Volume"] / btc_df["Volume"].rolling(14).mean()
btc_df = btc_df.dropna()

# -------- Signal Definition --------
btc_df["Future_5d_Return"] = btc_df["Close"].shift(-5) / btc_df["Close"] - 1
btc_df = btc_df.dropna()
threshold = btc_df["Future_5d_Return"].quantile(0.90)
btc_df["Signal"] = btc_df["Future_5d_Return"] >= threshold

# -------- Backtest Logic --------
initial_balance = 10000.0
usd_balance = initial_balance
entry_fee = 0.0025
exit_fee = 0.0040
position = None
trade_log = []

for date, row in btc_df.iterrows():
    if position is None and row["Signal"]:
        entry_price = row["Close"]
        btc_amount = usd_balance * (1 - entry_fee) / entry_price
        position = {
            "btc": btc_amount,
            "entry_price": entry_price,
            "entry_date": date,
            "entry_usd": usd_balance,
            "signal_strength": row["Future_5d_Return"]
        }
    elif position and date == position["entry_date"] + pd.Timedelta(days=5):
        exit_price = row["Close"]
        usd_balance = position["btc"] * exit_price * (1 - exit_fee)
        trade_log.append({
            "entry_date": position["entry_date"],
            "exit_date": date,
            "entry_price": position["entry_price"],
            "exit_price": exit_price,
            "btc_pct": 100.0,
            "entry_usd": position["entry_usd"],
            "exit_usd": usd_balance,
            "pnl": usd_balance - position["entry_usd"],
            "fees_paid": (entry_fee + exit_fee) * position["entry_usd"],
            "signal_strength": position["signal_strength"]
        })
        position = None

# -------- Final Metrics --------
trade_df = pd.DataFrame(trade_log)
final_balance = usd_balance
total_return = (final_balance - initial_balance) / initial_balance * 100
sharpe_ratio = (trade_df["pnl"].mean() / trade_df["pnl"].std()) * np.sqrt(252/5) if len(trade_df) > 1 else np.nan
total_fees = trade_df["fees_paid"].sum()
start_date = trade_df["entry_date"].min()
end_date = trade_df["exit_date"].max()

# -------- Output --------
print("\n--- Strategy Results ---")
print(f"Simulation Period: {start_date.date()} to {end_date.date()}")
print(f"Starting USD AUM: ${initial_balance:.2f}")
print(f"Final USD Balance: ${final_balance:.2f}")
print(f"Total Return: {total_return:.2f}%")
print(f"Total Trades: {len(trade_df)}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Total Fees Paid: ${total_fees:.2f}")

print("\n--- Trade Log ---")
print(trade_df[[
    "entry_date", "exit_date", "entry_price", "exit_price",
    "btc_pct", "entry_usd", "exit_usd", "pnl", "fees_paid", "signal_strength"
]].to_string(index=False))

# -------- Plotting --------
plt.figure(figsize=(14, 6))
plt.plot(btc_df["Close"], label="BTC Price", color='gray', linewidth=1)

# Plot trades
for _, trade in trade_df.iterrows():
    plt.scatter(trade["entry_date"], trade["entry_price"], color="green", marker="^", s=80, label="Entry" if _ == 0 else "")
    plt.scatter(trade["exit_date"], trade["exit_price"], color="red", marker="v", s=80, label="Exit" if _ == 0 else "")
    plt.plot([trade["entry_date"], trade["exit_date"]],
             [trade["entry_price"], trade["exit_price"]],
             linestyle='--', color='blue', alpha=0.5)

# Annotate signal strength
for _, trade in trade_df.iterrows():
    plt.text(trade["entry_date"], trade["entry_price"] * 0.97,
             f"{trade['signal_strength']*100:.1f}%",
             fontsize=8, color='blue', ha='center')

plt.title("BTC Trades with Entry/Exit & Signal Strength")
plt.xlabel("Date")
plt.ylabel("BTC Price (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()