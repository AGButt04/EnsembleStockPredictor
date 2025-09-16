import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential


def train_linear_regression(X_train, y_train):
    """Train linear regression model"""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    """Train random forest model"""
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model


def create_sequences_1d(data_1d, seq_length=10):
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(seq_length, len(data_1d)):
        X.append(data_1d[i - seq_length:i])
        y.append(data_1d[i])
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    y = y.reshape(-1, 1)
    return X, y


def train_lstm(data, seq_length=10):
    """Train LSTM model with chronological split"""
    prices = data['Close'].values.reshape(-1, 1)
    train_size = int(len(prices) * 0.8)

    train_prices = prices[:train_size]
    test_prices = prices[train_size:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_prices)
    test_scaled = scaler.transform(test_prices)

    X_train, y_train = create_sequences_1d(train_scaled.flatten(), seq_length)
    X_test, y_test = create_sequences_1d(test_scaled.flatten(), seq_length)

    model = Sequential([
        LSTM(50, input_shape=(seq_length, 1)),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=16,
              validation_split=0.2, verbose=0)

    return model, scaler, X_test, y_test


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)

    return {
        'mse': mse,
        'r2': r2,
        'rmse': rmse,
        'predictions': y_pred
    }


def save_model(model, filepath):
    """Save model to file"""
    joblib.dump(model, filepath)


def load_model(filepath):
    """Load model from file"""
    return joblib.load(filepath)