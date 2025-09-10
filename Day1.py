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


for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
    appleData[col] = pd.to_numeric(appleData[col])

# print(appleData.isnull().sum())
#
# print(appleData.columns.to_list())

closing = appleData['Close']
closingPrices = closing.tolist()
positives = 0
negatives = 0
negative_changes = []
positive_changes = []
for cP in range(1, len(closingPrices)):
    priceChange = closingPrices[cP] - closingPrices[cP - 1]
    percentageChange = (priceChange / closingPrices[cP]) * 100
    if priceChange > 0:
        positive_changes.append(priceChange)
        positives += 1
    else:
        negative_changes.append(priceChange)
        negatives += 1

    # print(f"Price: {closingPrices[cP]:.2f}, Change: {priceChange:.2f}")
    # print(f"Percentage: {percentageChange:.2f}%")

up_avg = sum(positive_changes)/len(positive_changes)
down_avg = sum(negative_changes)/len(negative_changes)
total_gains = positives * up_avg
total_losses = negatives * down_avg
net_change = total_gains - total_losses
print(f"\nTotal gains: {total_gains}")
print(f"Total losses: {total_losses}")
print(f"Estimated net change: {net_change :.2f}")
print(f"Average up day: ${up_avg:.2f}")
print(f"Average down day: ${down_avg:.2f}")
print(f"Positive changes: {positives}, Negative changes: {negatives}")

# Calculating moving averages:
MA_10 = closing.rolling(window=10).mean()
MA_50 = closing.rolling(window=50).mean()

appleData['MA_10'] = MA_10
appleData['MA_50'] = MA_50

print(appleData[['Close', 'MA_10', 'MA_50']].tail(10))

# Adding volatility
appleData['Daily_Return'] = appleData['Close'].pct_change()
appleData['Volatility'] = appleData['Daily_Return'].rolling(window=10).std()
print(appleData[['Close', 'Volatility']].tail(10))
print("Daily returns of the last 5 days:")
print(appleData['Daily_Return'].tail(5) * 100)

# Adding Price Yesterday and Volume Yesterday as features
appleData['Price_Yesterday'] = appleData['Close'].shift(1)
appleData['Volume_Yesterday'] = appleData['Volume'].shift(1)
print(appleData[['Close', 'Price_Yesterday']].tail(10))
print(appleData[['Close', 'Volume_Yesterday']].tail(10))

# Adding target columns which will have tomorrow's prices
# which we want to predict using our model
appleData['target'] = appleData['Close'].shift(-1)
print(appleData[['Close', 'target']].tail(10))

# Price features: Close, Price_Yesterday
# Trend features: MA_10, MA_50
# Volatility features: Daily_Return, Volatility_10
# Volume features: Volume, Volume_Yesterday
# Target: Target (tomorrow's closing price)

# Checking how many null values are there and dropping them for our model
print("Missing Values: ")
print(appleData.isnull().sum())

mlData = appleData.dropna()
print(f"Our Dataset's shape: {mlData.shape}")


if __name__ == "__main__":
    # Your main code execution
    pass