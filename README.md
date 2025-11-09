# 🍎 Apple Stock Prediction and Backtesting Dashboard

A machine learning–powered platform that predicts **Apple (AAPL)** stock prices, evaluates performance through **backtesting**, and visualizes results using an interactive **Streamlit dashboard**.

---

## 🎯 Project Overview

This project demonstrates a complete **ML + Quant Research pipeline** for financial forecasting and strategy evaluation.  
It combines multiple predictive models, feature engineering, and real-data backtesting to analyze how ML-based trading signals perform in the market.

### Core Highlights
- **End-to-end ML pipeline** for financial time-series prediction  
- **Multiple models:** Linear Regression, Random Forest, LSTM  
- **Ensemble approach** for higher accuracy (R² ≈ 0.92)  
- **Quant backtesting module** with Sharpe ratio, total return, and cumulative growth  
- **Interactive dashboard** for real-time exploration and model evaluation  

---

## 🛠️ Technical Stack

- **Languages:** Python 3.9  
- **ML/Data:** scikit-learn, TensorFlow, pandas, numpy  
- **Data Source:** Yahoo Finance API (via `yfinance`)  
- **Visualization:** Streamlit, matplotlib, plotly  
- **Deployment:** Streamlit Cloud  
---

## 📁 Project Structure

```bash
stock_predictor_ml/
├── data/                     # Stock price data  
├── models/                   # Saved trained models  
├── src/
│   ├── dataLoader.py          # Data acquisition  
│   ├── featureEngineering.py  # Feature creation  
│   ├── models.py              # ML model training  
│   └── ensemble.py            # Ensemble predictions  
├── dashboard.py               # Streamlit web interface (Predictions + Backtesting)  
├── main.py                    # Training pipeline  
└── requirements.txt           # Dependencies  
```

## 🔧 Features

### 🧩 Data Pipeline
- Automated data download from Yahoo Finance  
- Feature engineering (moving averages, lag returns, volatility)  
- Chronological time-series train/test split  

### 🤖 Machine Learning Models
| Model | Purpose |
|-------|----------|
| **Linear Regression** | Captures linear trends and short-term momentum |
| **Random Forest** | Handles non-linear dependencies |
| **LSTM Neural Network** | Learns sequential temporal behavior |
| **Ensemble** | Averages models for robust prediction accuracy |

### 💹 Backtesting & Quant Analysis
- Real ML-based signals converted into **long/short trades**  
- Compute **daily returns**, **Sharpe ratio**, and **total profit**  
- Compare **strategy vs buy-and-hold performance**  
- Visualize **cumulative growth** and trade behavior  

---

## 🚀 Getting Started

### Installation
```bash
git clone https://github.com/yourusername/apple-stock-dashboard
cd apple-stock-dashboard
pip install -r requirements.txt
```

### Training Models
```bash
python main.py
```
### Running Dashboard
```bash
streamlit run dashboard.py
```

### Model's Performance
| Model             | R² Score | RMSE  |
| ----------------- | -------- | ----- |
| Linear Regression | 0.9162   | $5.32 |
| Random Forest     | 0.9151   | $5.30 |
| LSTM              | 0.5049   | $7.44 |
| Ensemble          | 0.9197   | $5.18 |

## 🧠 Key Learnings
- Feature engineering (MAs, volatility) significantly improves accuracy
- Ensemble models outperform individual models by ~5%
- Sharpe ratio backtests validate model consistency
- Cumulative growth tracking helps quantify profitability

## ⚠️ Limitations & Disclaimer
- Educational project only - not for actual trading  
- Models trained on historical data may not predict future market conditions  
- Does not account for: news events, earnings, macroeconomic factors, transaction costs  
- Past performance does not guarantee future results  

## 🔮 Future Enhancements
- Integrate transaction cost simulation
- Add max drawdown and Sortino ratio metrics
- Extend to multi-stock portfolio backtesting
- Incorporate sentiment or macroeconomic data
- Experiment with Transformer-based price prediction 

## 👤 Author
**Abdul Ghani Butt**  
Computer Science Sophomore @ Widener University  
[LinkedIn](https://www.linkedin.com/in/abdul-ghani-butt-290056338/) | [GitHub](https://github.com/AGButt04)  

## 📝 License
This project is for educational purposes.  

Built as part of my journey to become an AI/ML or Quant Engineer.  

