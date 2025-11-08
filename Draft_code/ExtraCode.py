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

# Pandas can do this automatically
# appleData['Daily_Change'] = appleData['Close'].diff()
# print(appleData[['Close', 'Daily_Change']].head(10))

# Plotting Closing Price
# plt.figure(figsize=(14, 8))
#
# plt.subplot(3, 1, 1)  # 3 rows, 1 col, first plot
# plt.plot(appleData['Close'], label="Close Price")
# plt.title("Apple Closing Price")
# plt.ylabel("Price ($)")
# plt.legend()
#
# # Plot volatility
# plt.subplot(3, 1, 2)  # second plot
# plt.plot(appleData['Volatility'], color="orange", label="Volatility")
# plt.title("Apple Volatility (Daily % Change Std)")
# plt.ylabel("Volatility")
# plt.xlabel("Date")
# plt.legend()
#
# # Plot Daily Returns
# plt.subplot(3, 1, 3)  # second plot
# plt.plot(appleData['Daily_Return'], label="Daily Returns", color="green")
# plt.axhline(0, linestyle="--", color="red")  # reference line at 0
# plt.title("Apple Daily Returns")
# plt.ylabel("% Return")
# plt.xlabel("Date")
# plt.legend()
#
# plt.tight_layout()
# plt.show()

# Checking how many null values are there and dropping them for our model
# print("Missing Values: ")
# print(appleData.isnull().sum())

# Dropping Volume and Volume Yesterday.
# features = ['Close', 'Volume', 'Daily_Return', 'Price_Yesterday',
#           'Volume_Yesterday', 'MA_10', 'MA_50', 'Volatility']

# Plotting feature importance
# plt.figure(figsize=(8, 5))
# plt.barh(
#     feature_importance['features'],
#     feature_importance['coefficient'],
#     color='skyblue'
# )
# plt.axvline(x=0, color='red', linestyle='--')  # reference line at 0
# plt.title("Feature Importance (Linear Regression Coefficients)")
# plt.xlabel("Coefficient Value")
# plt.ylabel("Feature")
# plt.gca().invert_yaxis()  # largest on top
# plt.show()
