import streamlit as st
import numpy as np
import pandas as pd
from src.dataLoader import load_apple_data
from src.featureEngineering import create_features
import joblib

@st.cache_resource
def load_models():
    linear_model = joblib.load('models/linear_model.pkl')
    rf_model = joblib.load('models/random_forest_model.pkl')
    # lstm_model = joblib.load('models/lstm_model.h5')
    return linear_model, rf_model

def make_prediction(data):
    """Make predictions using loaded models"""
    processed = create_features(data.copy())

    features = ['Close', 'Daily_Return', 'Price_Yesterday',
                'MA_10', 'MA_50', 'Volatility']

    if len(processed) == 0:
        return None, None, None

    X_latest = processed[features].iloc[-1:].values

    # Make Predictions
    linear_pred = linear_model.predict(X_latest)[0]
    rf_pred = rf_model.predict(X_latest)[0]
    ensemble_pred = (linear_pred + rf_pred) / 2

    return linear_pred, rf_pred, ensemble_pred

def get_refresh_data(data):
    start_date = st.sidebar.date_input("Start Date",
                                       value=data.index[0],
                                       min_value=data.index[0],
                                       max_value=data.index[-1])

    end_date = st.sidebar.date_input("End Date",
                                     value=data.index[-1],
                                     min_value=data.index[0],
                                     max_value=data.index[-1])

    filtered = data[str(start_date):str(end_date)]
    return filtered

linear_model, rf_model = load_models()

st.set_page_config(page_title="Apple Stock Dashboard", page_icon="🍎", layout='wide')
st.markdown(
    """
    <div style="display: flex; align-items: center; gap: 10px;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/f/fa/Apple_logo_black.svg" width="40">
        <h1 style="margin: 0;">Apple Stock Prediction and Backtesting Dashboard</h1>
    </div>
    """,
    unsafe_allow_html=True
)
st.caption("Built with Machine Learning models and real stock data for quantitative analysis.")

tabs = st.tabs(["📊 Predictions Dashboard", "💹 Backtesting"])


st.sidebar.header("Dashboard Controls")
# In your dashboard, add to sidebar:
if st.sidebar.button("Refresh Data"):
    data = load_apple_data(force_refresh=True)
    st.rerun()
else:
    data = load_apple_data()

st.sidebar.subheader("Date Range")
filtered_data = get_refresh_data(data)
with tabs[0]:
    # Move your date selector to the sidebar:

    st.subheader("🤖 Model Selection")
    selected_model = st.sidebar.selectbox(
        "Choose Model for Analysis:",
        ["Linear Regression", "Random Forest", "LSTM", "Ensemble"]
    )

    current_price = filtered_data["Close"].iloc[-1]
    previous_price = filtered_data["Close"].iloc[-2]
    daily_change = current_price - previous_price

    # Add a section for model predictions
    st.subheader("Model Predictions")
    linear_pred, rf_pred, ensemble_pred = make_prediction(filtered_data)

    # Replace your old placeholder prediction section with this:

    if linear_pred is not None:
        if selected_model == "Linear Regression":
            st.metric("Tomorrow's Predicted Price", f"${linear_pred:.2f}")
            st.write("Model: Linear Regression")
            st.write("R² Score: 0.9162")

        elif selected_model == "Random Forest":
            st.metric("Tomorrow's Predicted Price", f"${rf_pred:.2f}")
            st.write("Model: Random Forest")
            st.write("R² Score: 0.9151")

        elif selected_model == "LSTM":
            st.write("LSTM predictions coming soon...")
            st.write("R² Score: 0.5049")

        else:  # Ensemble
            st.metric("Tomorrow's Predicted Price", f"${ensemble_pred:.2f}")
            st.write("Model: Ensemble (Linear + Random Forest)")
            st.write("R² Score: 0.9197 (Best Performance)")

        # Show comparison of all models
        st.write("---")
        st.write("**All Model Predictions:**")
        col_pred1, col_pred2, col_pred3 = st.columns(3)
        with col_pred1:
            st.write(f"Linear: ${linear_pred:.2f}")
        with col_pred2:
            st.write(f"Random Forest: ${rf_pred:.2f}")
        with col_pred3:
            st.write(f"Ensemble: ${ensemble_pred:.2f}")
    else:
        st.error("Not enough data to make predictions. Please select a longer date range.")

    st.metric("Current Stock Price", f"${current_price:.2f}")
    st.metric("Previous Stock Price", f"${previous_price:.2f}")
    st.metric("Price Change", f"${daily_change:.2f}", delta=daily_change)

    # Add the simple line chart and Volume bar chart for Analysis
    recent_data = filtered_data

    # Create columns to show charts side by side
    col1, col2 = st.columns(2)
    # Calculate moving averages for the recent data
    recent_data['MA_10'] = recent_data['Close'].rolling(10).mean()

    # Debug: Check what data you have
    # st.write("Debug info:")
    # st.write(f"Recent data shape: {recent_data.shape}")
    # st.write(f"Data after dropna: {recent_data[['Close', 'MA_10', 'MA_50']].dropna().shape}")

    with col1:
        st.write("**Price Trend with Moving Averages**")
        chart_data = recent_data[['Close', 'MA_10']].dropna()
        st.line_chart(chart_data)

        # Add daily trading range info
        st.write("**Daily Price Range (High - Low)**")
        recent_data['Daily_Range'] = recent_data['High'] - recent_data['Low']
        st.bar_chart(recent_data['Daily_Range'])

    with col2:
        st.write("**Volume**")
        st.bar_chart(recent_data["Volume"])

    # Add this after your columns, before the model performance section:
    st.subheader("Market Summary")

    col3, col4, col5 = st.columns(3)

    with col3:
        avg_volume = recent_data['Volume'].mean()
        st.metric("Average Volume (30d)", f"{avg_volume:,.0f}")

    with col4:
        avg_range = recent_data['Daily_Range'].mean()
        st.metric("Average Daily Range", f"${avg_range:.2f}")

    with col5:
        volatility = recent_data['Close'].pct_change().std() * 100
        st.metric("Price Volatility", f"{volatility:.1f}%")

with tabs[1]:
    st.header("💹 Model-Based Strategy Backtesting")

    # Recompute features for backtest and make predictions for the entire dataset
    processed = create_features(filtered_data.copy())
    features = ['Close', 'Daily_Return', 'Price_Yesterday', 'MA_10', 'MA_50', 'Volatility']
    X_all = processed[features].values

    # Predict for all rows
    linear_pred_all = linear_model.predict(X_all).flatten()
    rf_pred_all = rf_model.predict(X_all).flatten()
    ensemble_pred_all = (linear_pred_all + rf_pred_all) / 2

    # User controls
    st.sidebar.subheader("Backtesting Controls")
    model_choice = st.sidebar.selectbox("Choose Model for Backtest", ["Linear Regression", "Random Forest", "Ensemble"])
    threshold = st.slider("Signal Threshold ($USD)", 0.0, 10.0, 0.5, step=0.1)

    if model_choice == "Linear Regression":
        predictions = linear_pred_all
    elif model_choice == "Random Forest":
        predictions = rf_pred_all
    else:
        predictions = ensemble_pred_all

    close_prices = processed["Close"].values
    predictions = np.array(predictions).flatten()

    # Align arrays
    min_len = min(len(predictions), len(close_prices))
    predictions = predictions[:min_len]
    close_prices = close_prices[:min_len]

    # Actual Returns
    returns = (close_prices[1:] - close_prices[:-1]) / close_prices[:-1]

    # Signal logic: When predicted - actual exceeds threshold
    signal = np.where(np.abs(predictions[:-1] - close_prices[:-1]) > threshold,
                      np.sign(predictions[:-1] - close_prices[:-1]), 0)

    # Strategy returns and cumulative growth
    strategy_returns = signal * returns
    cumulative_growth = np.cumprod(1 + strategy_returns) - 1
    buy_hold = np.cumprod(1 + returns) - 1

    # Performance metric
    if np.std(strategy_returns) != 0:
        sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
    else:
        sharpe = 0

    total_return = cumulative_growth[-1] * 100
    num_trades = np.count_nonzero(signal)

    # Display metrics
    st.subheader("📊 Strategy Performance Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Sharpe Ratio", f"{sharpe:.2f}")
    col2.metric("Total Return", f"{total_return:.2f}%")
    col3.metric("Number of Trades", f"{num_trades}")

    # Plot cumulative returns
    st.subheader("📈 Strategy vs Buy & Hold")
    st.line_chart({
        "Strategy": cumulative_growth,
        "Buy & Hold": buy_hold
    })

    # Optional: show data table for recent trades
    with st.expander("View Signal Data"):
        # Build aligned DataFrame for display
        df_backtest = pd.DataFrame({
            "Date": processed.index[1:len(returns) + 1],
            "Close": close_prices[1:len(returns) + 1],
            "Predicted": predictions[:len(returns)],
            "Signal": signal[:len(returns)],
            "Daily Return": returns,
            "Strategy Return": strategy_returns
        })

        st.dataframe(df_backtest.tail(20))



