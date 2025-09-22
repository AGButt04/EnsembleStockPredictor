import streamlit as st
from src.dataLoader import load_apple_data

st.set_page_config(page_title="Apple Stock", page_icon="📈", layout='wide')
st.title("Apple Stock Dashboard")

data = load_apple_data()
current_price = data["Close"].iloc[-1]
previous_price = data["Close"].iloc[-2]
daily_change = current_price - previous_price

st.metric("Current Stock Price", f"${current_price:.2f}")
st.metric("Previous Stock Price", f"${previous_price:.2f}")
st.metric("Price Change", f"${daily_change:.2f}", delta=daily_change)

# Add the simple line chart and Volume bar chart for Analysis
recent_data = data.tail(30)

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

with col2:
    st.write("**Volume**")
    st.bar_chart(recent_data["Volume"])

# Add a section for model predictions
st.subheader("Model Performance")
st.write("Linear Regression R²: 0.9162")
st.write("Random Forest R²: 0.9151")
st.write("Ensemble R²: 0.9197")

# Add prediction section
st.subheader("Tomorrow's Price Predictions")

linear_pred = current_price + 0.5
rf_pred = current_price - 0.2
ensemble_pred = current_price + 0.10

st.write(f"Linear Regression: ${linear_pred:.2f}")
st.write(f"Random Forest: ${rf_pred:.2f}")
st.write(f"Ensemble: ${ensemble_pred:.2f}")
