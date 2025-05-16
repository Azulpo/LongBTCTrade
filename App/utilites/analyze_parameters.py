import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import time
import os

"""
Utility Script: Parameter Search Analysis & Modeling

Description:
This script is used to analyze and model the results of a trading strategy parameter search.
It performs the following tasks:

1. Loads and preprocesses parameter search results from a CSV file.
2. Visualizes the relationship between key strategy parameters and total return percentage using scatterplots.
3. Trains an XGBoost regression model to predict total return based on input parameters:
   - min_momentum_threshold
   - stop_loss_pct
   - max_hold_minutes
4. Evaluates model performance using test set MSE.
5. Displays feature importances to identify which parameters have the most predictive value.

Use Case:
This utility is helpful for identifying which strategy parameters are most influential in producing profitable results and for guiding future parameter optimization efforts.
"""

# -------- Settings --------
base_path = base_path = '/'
data_file = os.path.join(base_path, 'Data', 'parameter_search_results.csv')

# -------- Load Data --------
print("\n📥 Loading parameter search results...")
df = pd.read_csv(data_file)

# Drop rows with NaNs
df = df.dropna()

# Focus on meaningful results only
df = df[df['total_trades'] > 2]  # Remove strategies that barely traded

print("\n✅ Data Loaded")
print(df.describe())

# -------- Quick Visualizations --------
print("\n📈 Plotting simple scatterplots...")

plt.figure(figsize=(12,6))
plt.scatter(df['min_momentum_threshold'], df['total_return_pct'], alpha=0.5)
plt.title('Momentum Threshold vs Total Return')
plt.xlabel('Min Momentum Threshold')
plt.ylabel('Total Return (%)')
plt.grid(True)
plt.show()

plt.figure(figsize=(12,6))
plt.scatter(df['stop_loss_pct'], df['total_return_pct'], alpha=0.5)
plt.title('Stop Loss % vs Total Return')
plt.xlabel('Stop Loss %')
plt.ylabel('Total Return (%)')
plt.grid(True)
plt.show()

plt.figure(figsize=(12,6))
plt.scatter(df['max_hold_minutes'], df['total_return_pct'], alpha=0.5)
plt.title('Max Hold Minutes vs Total Return')
plt.xlabel('Max Hold Minutes')
plt.ylabel('Total Return (%)')
plt.grid(True)
plt.show()

# -------- XGBoost Model Training --------
print("\n🚀 Starting XGBoost training with progress bar...")

features = ['min_momentum_threshold', 'stop_loss_pct', 'max_hold_minutes']
target = 'total_return_pct'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Progress tracking
start_time = time.time()

# Wrap training with tqdm for visual progress
model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)

for _ in tqdm(range(1), desc="Training XGBoost Model"):
    model.fit(X_train, y_train)

end_time = time.time()

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f"\n✅ XGBoost Model Trained")
print(f"Test MSE: {mse:.4f}")
print(f"Training Time: {end_time - start_time:.2f} seconds")

# -------- Feature Importance --------
print("\n🔍 Analyzing feature importance...")

importances = model.feature_importances_

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\n--- Feature Importance ---")
print(importance_df)

plt.figure(figsize=(8,6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.title('Feature Importance (XGBoost)')
plt.gca().invert_yaxis()
plt.grid(True)
plt.show()