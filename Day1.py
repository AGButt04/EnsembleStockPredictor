# Step 1: Import libraries
import pandas as pd
import tensorflow
import yfinance as yf
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import Sequential, Input

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

appleData[['Close', 'MA_10', 'MA_50']].tail(10)

# Adding volatility
appleData['Daily_Return'] = appleData['Close'].pct_change()
appleData['Volatility'] = appleData['Daily_Return'].rolling(window=10).std()
print(appleData[['Close', 'Volatility']].tail(10))
print("Daily returns of the last 5 days:")

# Adding Price Yesterday and Volume Yesterday as features
appleData['Price_Yesterday'] = appleData['Close'].shift(1)
appleData['Volume_Yesterday'] = appleData['Volume'].shift(1)
print(appleData[['Close', 'Price_Yesterday']].tail(10))
print(appleData[['Close', 'Volume_Yesterday']].tail(10))

# Adding target columns which will have tomorrow's prices
# which we want to predict using our model
appleData['Price_Tomorrow'] = appleData['Close'].shift(-1)
(appleData[['Close', 'Price_Tomorrow']].tail(10))

# Price features: Close, Price_Yesterday
# Trend features: MA_10, MA_50
# Volatility features: Daily_Return, Volatility_10
# Volume features: Volume, Volume_Yesterday
# Target: Target (tomorrow's closing price)

# Dropping the null values to clean the data for the model
mlData = appleData.dropna()
print(f"Our Dataset's shape: {mlData.shape}")
print(mlData.head())
print(mlData.describe())

features = ['Close', 'Daily_Return', 'Price_Yesterday',
            'MA_10', 'MA_50', 'Volatility']

X = mlData[features]
y = mlData['Price_Tomorrow']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Training the model and first predictions
model1 = LinearRegression()
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)
# Evaluate the model
mse_1 = mean_squared_error(y_test, y_pred1)
r2_1 = r2_score(y_test, y_pred1)

print(f"Linear Model Performance:")
print(f"Mean Squared Error: ${mse_1:.2f}")
print(f"R² Score: {r2_1:.4f}")
print(f"Root Mean Squared Error: ${np.sqrt(mse_1):.2f}")

# Checking important features:
feature_importance = pd.DataFrame({
    'features': features,
    'coefficient': model1.coef_
})
feature_importance['Abs_coefficient'] = abs(feature_importance['coefficient'])
feature_importance = feature_importance.sort_values('Abs_coefficient', ascending=False)
print("Feature importance:")
print(feature_importance)

# Dropping useless features after analysis - Volume and Volume Yesterday.

# Developing a second random forest model to see the results and the comparison
model2 = RandomForestRegressor()
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
# Evaluate the model
mse_2 = mean_squared_error(y_test, y_pred2)
r2_2 = r2_score(y_test, y_pred2)

print(f"Random forest Performance:")
print(f"Mean Squared Error: ${mse_2:.2f}")
print(f"R² Score: {r2_2:.4f}")
print(f"Root Mean Squared Error: ${np.sqrt(mse_2):.2f}")

# Third model would be the average predictions of both models
# Simple ensemble - average both predictions
ensemble_pred = (y_pred1 + y_pred2) / 2
# Evaluate the model
ensemble_mse = mean_squared_error(y_test, ensemble_pred)
ensemble_r2 = r2_score(y_test, ensemble_pred)

print(f"Ensemble Performance:")
print(f"Mean Squared Error: ${ensemble_mse:.2f}")
print(f"R² Score: {ensemble_r2:.4f}")
print(f"Root Mean Squared Error: ${np.sqrt(ensemble_mse):.2f}")

# # Visualizing all models
# plt.figure(figsize=(14, 8))
# # Linear Regression
# plt.scatter(y_test, y_pred1, alpha=0.7, label="Linear Regression", color='red')
# # Random Forest
# plt.scatter(y_test, y_pred2, alpha=0.7, label="Random Forest", color='blue')
# # Ensemble
# plt.scatter(y_test, ensemble_pred, alpha=0.7, label="Ensemble (Avg)", color='green')
# # Perfect prediction line
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label="Perfect Prediction")
#
# plt.xlabel('Actual Price')
# plt.ylabel('Predicted Price')
# plt.title('Actual vs Predicted Stock Prices (Linear, RF, Ensemble)')
# plt.legend()
# plt.show()

# Training the first Long Short-Term Memory Neural Network
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length : i])
        y.append(data[i])

    X = np.array(X)
    y = np.array(y)
    # Reshape for LSTM: (Samples, Seq_Length, features=1)
    X = X.reshape(X.shape[0], seq_length, 1)
    y = y.reshape(-1, 1)
    return X, y

# Pipeline with train/test split (chronlogically)
prices = mlData['Close'].values.reshape(-1, 1)
test_size = int(len(prices) * 0.8)
train_data = prices[:test_size]
test_data = prices[test_size:]

# Scaling the inputs for the smooth model
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

# Creating sequences using the function
seq_length = 10
X_train_N, y_train_N = create_sequences(train_scaled.flatten(), seq_length)
X_test_temp, y_test_temp = create_sequences(test_scaled.flatten(), seq_length)

print(f"X_train_N: {X_train_N.shape} Y_train_N: {y_train_N.shape}")
print(f"X_test_temp: {X_test_temp.shape} y_test_temp: {y_test_temp.shape}")

# Build the model
model3 = Sequential([
    Input(shape=(seq_length,1)),
    LSTM(50),
    Dense(1)
])

model3.compile(optimizer='adam', loss='mse')
print(model3.summary())

# Training the model
print("Training the LSTM model...")
history = model3.fit(
    X_train_N, y_train_N,
    epochs=50, # Number of training rounds
    batch_size=16, # Process 16 sequences in a round
    validation_split=0.2, # Use 20% of training for validation
    verbose=1 # Show progress
)

# Make predictions
y_pred3 = model3.predict(X_test_temp)

# Since data is scaled, convert it back
y_pred3_unscaled = scaler.inverse_transform(y_pred3)
y_pred3_scaled = scaler.inverse_transform(y_test_temp)

print("LSTM Model Training complete!")


if __name__ == "__main__":
    # Your main code execution
    pass