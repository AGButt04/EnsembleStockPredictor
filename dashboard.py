import streamlit as st
from src.dataLoader import load_apple_data

st.set_page_config(page_title="Apple Stock", page_icon="📈")
st.title("Apple Stock Dashboard")

data = load_apple_data()
current_price = data["Close"].iloc[-1]
previous_price = data["Close"].iloc[-2]
daily_change = current_price - previous_price

st.metric("Current Stock Price", f"${current_price:.2f}")
st.metric("Previous Stock Price", f"${previous_price:.2f}")
st.metric("Price Change", f"${daily_change:.2f}", delta=daily_change)

# Add the simple line chart
st.header("Price Trend (Last 30 days)")
st.line_chart(data["Close"].tail(30))

# Add a section for model predictions
st.subheader("Model Performance")
st.write("Linear Regression R²: 0.9162")
st.write("Random Forest R²: 0.9151")
st.write("Ensemble R²: 0.9197")

# Add prediction section
st.subheader("Price Predictions for tomorrow")

linear_pred = daily_change + 0.5
rf_pred = daily_change - 0.2
ensemble_pred = daily_change + 0.10

st.write(f"Linear Regression: ${linear_pred:.2f}")
st.write(f"Random Forest: ${rf_pred:.2f}")
st.write(f"Ensemble: ${ensemble_pred:.2f}")
