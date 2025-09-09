# Step 1: Import libraries
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import os

from numpy.ma.core import negative

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
# print(appleData.head())
# print(appleData.info())
# print(appleData.describe())

# print(f"Data Shape: {appleData.shape}")
# print(f"Columns: {appleData.columns}")
# print(f"Data range: {appleData.index[0]} to {appleData.index[-1]}")
# print("Data types:")
# print(appleData.dtypes)
# print("\nFirst few values of Close column:")
# print(appleData['Close'].head())

for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
    appleData[col] = pd.to_numeric(appleData[col])

print(appleData.isnull().sum())

print(appleData.columns.to_list())

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


# Pandas can do this automatically
# appleData['Daily_Change'] = appleData['Close'].diff()
# print(appleData[['Close', 'Daily_Change']].head(10))

# Plotting the closing prices to see the trends
closing.plot(title="Closing Prices - Apple Last Year")
plt.ylabel("$Price")
plt.show()

if __name__ == "__main__":
    # Your main code execution
    pass