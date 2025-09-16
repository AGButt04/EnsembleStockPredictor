import pandas as pd
import yfinance as yf
import os

def load_apple_data():
    """Load Apple stock data, either from CSV or download fresh"""
    if os.path.exists('./data/apple_data.csv'):
        print("Apple data found, loading data...")
        appleData = pd.read_csv("./data/apple_data.csv", index_col=0)
    else:
        print("Apple data not found, downloading...")
        appleData = yf.download(tickers="AAPL", period="1y")
        # Flatten before saving
        if isinstance(appleData.columns, pd.MultiIndex):
            appleData.columns = appleData.columns.get_level_values(0)
        appleData.to_csv('./data/apple_data.csv')

    # Also flatten after loading, just in case
    if isinstance(appleData.columns, pd.MultiIndex):
        appleData.columns = appleData.columns.get_level_values(0)

    for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
        appleData[col] = pd.to_numeric(appleData[col])

    return appleData