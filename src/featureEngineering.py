from sklearn.model_selection import train_test_split

def create_features(data):
    """Create ML features from raw stock data"""
    # Create features
    data['MA_10'] = data['Close'].rolling(window=10).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    data['Daily_Return'] = data['Close'].pct_change()
    data['Volatility'] = data['Daily_Return'].rolling(window=10).std()
    data['Price_Yesterday'] = data['Close'].shift(1)
    data['Volume_Yesterday'] = data['Volume'].shift(1)
    data['Price_Tomorrow'] = data['Close'].shift(-1)  # Target

    # Clean data
    clean_data = data.dropna()

    return clean_data


def prepare_model_data(data):
    """Prepare features and target for ML models"""
    features = ['Close', 'Daily_Return', 'Price_Yesterday',
                'MA_10', 'MA_50', 'Volatility']

    X = data[features]
    y = data['Price_Tomorrow']

    return train_test_split(X, y, test_size=0.2, random_state=42)