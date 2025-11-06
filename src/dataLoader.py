import pandas as pd
import yfinance as yf
import os


def load_apple_data(force_refresh=False):
    """Load Apple stock data, either from CSV or download fresh"""
    csv_path = './data/apple_data.csv'

    if os.path.exists(csv_path) and not force_refresh:
        print("Apple data found, loading from cache...")
        appleData = pd.read_csv(csv_path, index_col=0)
    else:
        print("Downloading fresh data from Yahoo Finance...")
        appleData = yf.download(tickers="AAPL", period="2y")  # Get 2 years

        # Flatten before saving
        if isinstance(appleData.columns, pd.MultiIndex):
            appleData.columns = appleData.columns.get_level_values(0)
        appleData.to_csv(csv_path)

    # Flatten after loading
    if isinstance(appleData.columns, pd.MultiIndex):
        appleData.columns = appleData.columns.get_level_values(0)

    for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
        appleData[col] = pd.to_numeric(appleData[col])

    return appleData