# Step 1: Import libraries
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import os

# Step 2: Try downloading Apple stock data
print("Starting stock data exploration...")

# Your code goes here - try downloading AAPL data for the last year
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

# Step 3: Explore the data
# Try these pandas operations once you have the data:
# - .head() to see first few rows
# - .info() to see data types
# - .describe() to see statistics
print(appleData.head())
print(appleData.info())
print(appleData.describe())

print(f"Data Shape: {appleData.shape}")
print(f"Columns: {appleData.columns}")
print("Data types:")
print(appleData.dtypes)
print("\nFirst few values of Close column:")
print(appleData['Close'].head())

for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
    appleData[col] = pd.to_numeric(appleData[col])

closingPrices = appleData['Close']
print(closingPrices.head())

print(appleData.isnull().sum())

print(appleData.columns.to_list())

closingPrices.plot(title="Closing Prices - Apple Last Year")
plt.ylabel("$Price")
plt.show()

if __name__ == "__main__":
    # Your main code execution
    pass