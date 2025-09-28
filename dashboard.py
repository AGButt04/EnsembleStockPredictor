import streamlit as st
from src.dataLoader import load_apple_data

st.set_page_config(page_title="Apple Stock", page_icon="📈", layout='wide')
st.title("Apple Stock Dashboard")

data = load_apple_data()

# Add this near the top, after your imports but before the main content:
st.sidebar.header("Dashboard Controls")

# Move your date selector to the sidebar:
st.sidebar.subheader("Date Range")
start_date = st.sidebar.date_input("Start Date",
                                  value=data.index[0],
                                  min_value=data.index[0],
                                  max_value=data.index[-1])

end_date = st.sidebar.date_input("End Date",
                                value=data.index[-1],
                                min_value=data.index[0],
                                max_value=data.index[-1])

filtered_data = data[str(start_date):str(end_date)]

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
linear_pred = current_price + 0.5
rf_pred = current_price - 0.2
ensemble_pred = current_price + 0.10

if selected_model == "Linear Regression":
    st.write(f"**{selected_model} Prediction:** ${linear_pred:.2f}")
    st.write("R² Score: 0.9162")
elif selected_model == "Random Forest":
    st.write(f"**{selected_model} Prediction:** ${rf_pred:.2f}")
    st.write("R² Score: 0.9151")
elif selected_model == "LSTM":
    st.write(f"**{selected_model} Prediction:** Coming soon...")
    st.write("R² Score: 0.5049")
else:  # Ensemble
    st.write(f"**{selected_model} Prediction:** ${ensemble_pred:.2f}")
    st.write("R² Score: 0.9197 (Best Performance)")

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
