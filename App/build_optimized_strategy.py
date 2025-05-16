import pandas as pd
import os

# -------- Settings --------
base_path = '/Users/aaron/PycharmProjects/LongBTCTrade
data_file = os.path.join(base_path, 'Data', 'parameter_search_results.csv')

# -------- Load Data --------
print("\n📥 Loading parameter search results...")
df = pd.read_csv(data_file)

# Drop NaNs and meaningless entries
df = df.dropna()
df = df[df['total_trades'] > 2]

print("\n✅ Data Loaded")
print(f"Total strategies available: {len(df)}")

# -------- Select Top Performers --------
top_percent = 0.10  # Top 10%
top_n = int(len(df) * top_percent)

df_sorted = df.sort_values(by='total_return_pct', ascending=False)
top_df = df_sorted.head(top_n)

print(f"\n✅ Selected Top {top_n} Strategies (Top 10%)")

# -------- Analyze Top Performers --------
print("\n--- Top Strategies Parameter Summary ---")
print(top_df[['min_momentum_threshold', 'stop_loss_pct', 'max_hold_minutes']].describe())

# -------- Suggested Parameters --------
suggested_momentum = top_df['min_momentum_threshold'].mean()
suggested_stop_loss = top_df['stop_loss_pct'].mean()
suggested_hold_minutes = top_df['max_hold_minutes'].mean()

print("\n--- Suggested Optimized Parameters ---")
print(f"Suggested Min Momentum Threshold: {suggested_momentum:.4f}")
print(f"Suggested Stop Loss %: {suggested_stop_loss:.4f}")
print(f"Suggested Max Hold Minutes: {suggested_hold_minutes:.0f} minutes")

# -------- Save Top Strategies --------
output_file = os.path.join(base_path, 'Data', 'top_strategies.csv')
top_df.to_csv(output_file, index=False)
print(f"\n✅ Top strategies saved to: {output_file}")