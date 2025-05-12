import pandas as pd
import requests
import time
from datetime import datetime, timedelta
import os
from tqdm import tqdm  # <-- Put this at the top with the others!

# -------- Settings --------
symbol = 'BTCUSDT'
interval = '1m'
limit = 1000  # Max candles per Binance request
days_back = 180  # ~6 months
sleep_time = 0.4  # Sleep between requests

# Base URL for Binance API
base_url = 'https://api.binance.com/api/v3/klines'

# Output path (adjust to YOUR project structure)
base_path = '/Users/aaron/PycharmProjects/PythonProject1'
output_file = os.path.join(base_path, 'Data', 'btc_1min_6mo.csv')

# -------- Helper Functions --------
def date_to_milliseconds(date_obj):
    return int(date_obj.timestamp() * 1000)

def get_klines(symbol, interval, start_time, end_time, limit=1000):
    headers = {'User-Agent': 'Mozilla/5.0'}
    url = f'{base_url}?symbol={symbol}&interval={interval}&startTime={start_time}&endTime={end_time}&limit={limit}'
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching data: {response.status_code} | {response.text}")
        return []

# -------- Main Download Logic --------
def download_binance_data():
    start_date = datetime.utcnow() - timedelta(days=days_back)
    end_date = datetime.utcnow()
    current_start = start_date

    all_data = []

    print("Starting Binance 1-minute BTCUSDT download...")

    # Estimate how many loops needed
    total_minutes = (end_date - start_date).total_seconds() / 60
    total_loops = int(total_minutes / limit) + 1

    with tqdm(total=total_loops, desc="Downloading BTC candles") as pbar:
        while current_start < end_date:
            current_end = current_start + timedelta(minutes=limit)
            start_ms = date_to_milliseconds(current_start)
            end_ms = date_to_milliseconds(current_end)

            data = get_klines(symbol, interval, start_ms, end_ms, limit)
            if not data:
                break

            all_data.extend(data)

            # Update to next batch
            last_close_time = data[-1][6] / 1000
            current_start = datetime.utcfromtimestamp(last_close_time) + timedelta(milliseconds=1)

            time.sleep(sleep_time)

            pbar.update(1)

    print(f"Downloaded {len(all_data)} candles.")

    # Create DataFrame
    columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
               'Close time', 'Quote asset volume', 'Number of trades',
               'Taker buy base', 'Taker buy quote', 'Ignore']

    df = pd.DataFrame(all_data, columns=columns)

    # Clean and format
    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
    df['Open'] = df['Open'].astype(float)
    df['High'] = df['High'].astype(float)
    df['Low'] = df['Low'].astype(float)
    df['Close'] = df['Close'].astype(float)
    df['Volume'] = df['Volume'].astype(float)

    df = df[['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df = df.rename(columns={'Open time': 'Datetime'})

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Saved cleaned data to: {output_file}")

# -------- Run it --------
if __name__ == "__main__":
    download_binance_data()