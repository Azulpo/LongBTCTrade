import pandas as pd
from binance.client import Client
import time

"""
Script: fetch_binance_daily_data.py

Description:
Fetches historical daily OHLCV data for BTC/USDT from Binance using the Binance API.
Processes and saves a simplified version containing only:
- Datetime (open time)
- Close price
- Volume

Default settings:
- Symbol: BTCUSDT
- Interval: 1 day
- Start date: January 1, 2023

Output:
- Saves a cleaned CSV file: 'btc_binance_1d.csv'

Useful for research, analysis, or seeding backtesting scripts with recent BTC price data.
"""

api_key = 'your_api_key'      # optional for public data
api_secret = 'your_api_secret'
client = Client(api_key, api_secret)

def get_daily_klines(symbol="BTCUSDT", interval="1d", start_str="1 Jan, 2023"):
    klines = client.get_historical_klines(symbol, interval, start_str)
    df = pd.DataFrame(klines, columns=[
        "Open Time", "Open", "High", "Low", "Close", "Volume",
        "Close Time", "Quote Asset Volume", "Number of Trades",
        "Taker Buy Base", "Taker Buy Quote", "Ignore"
    ])
    df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms")
    df["Close"] = pd.to_numeric(df["Close"])
    df["Volume"] = pd.to_numeric(df["Volume"])
    df = df[["Open Time", "Close", "Volume"]].rename(columns={"Open Time": "Datetime"})
    return df

btc_df = get_daily_klines()
btc_df.to_csv("btc_binance_1d.csv", index=False)