# 📈 Ensemble Stock Prediction Platform

A machine learning system that predicts Apple (AAPL) stock prices using multiple algorithms and presents analysis through an interactive web dashboard.

## 🎯 Project Overview

This project demonstrates an end-to-end ML pipeline for financial forecasting, combining:
- **Multiple ML algorithms** (Linear Regression, Random Forest, LSTM neural networks)
- **Ensemble methodology** for improved prediction accuracy
- **Feature engineering** from raw stock data
- **Interactive web dashboard** for real-time analysis

**Achievement**: 92% prediction accuracy (R² = 0.92) on test data using ensemble approach.

## 🛠️ Technical Stack

- **Languages**: Python 3.9
- **ML/Data**: scikit-learn, TensorFlow, pandas, numpy
- **Data Source**: yfinance (Yahoo Finance API)
- **Visualization**: Streamlit, matplotlib, plotly
- **Deployment**: Streamlit Cloud

## 📊 Project Structure
stock_predictor_ml/  
├── data/                   # Stock price data  
├── models/                 # Saved trained models  
├── src/  
│   ├── dataLoader.py        # Data acquisition  
│   ├── featureEngineering.py # Feature creation  
│   ├── models.py            # ML model training  
│   └── ensemble.py          # Ensemble predictions  
├── dashboard.py             # Streamlit web interface  
├── main.py                  # Training pipeline  
└── requirements.txt         # Dependencies  

## 🔧 Features

### Data Pipeline
- Automated data downloading from Yahoo Finance
- Feature engineering (moving averages, volatility, lag features)
- Proper time-series train/test splitting

### ML Models
1. **Linear Regression** - Captures linear trends and momentum  
2. **Random Forest** - Handles non-linear patterns  
3. **LSTM Neural Network** - Learns sequential patterns  
4. **Ensemble** - Combines predictions for best performance  

### Interactive Dashboard
- Date range selection for custom analysis periods
- Model comparison and selection
- Real-time visualizations (price trends, volume, volatility)
- Market summary statistics

## 🚀 Getting Started

### Installation
```bash
git clone https://github.com/yourusername/ensemble-stock-predictor
cd ensemble-stock-predictor
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
- Feature engineering matters: Moving averages and volatility significantly improved predictions  
- Ensemble superiority: Combining models reduced prediction error by ~5%  
- Mean reversion patterns: Models learned that large daily moves often reverse  
- Time-series validation: Chronological splitting prevents data leakage  

## ⚠️ Limitations & Disclaimer
- Educational project only - not for actual trading  
- Models trained on historical data may not predict future market conditions  
- Does not account for: news events, earnings, macroeconomic factors, transaction costs  
- Past performance does not guarantee future results  

## 🔮 Future Enhancements
- Add sentiment analysis from financial news  
- Implement more advanced neural architectures (Transformers)  
- Multi-stock prediction capability  
- Real-time data streaming  
- Backtesting framework with transaction costs  

## 👤 Author
**Abdul Ghani Butt**  
Computer Science Sophomore @ Widener University  
[LinkedIn](https://www.linkedin.com/in/abdul-ghani-butt-290056338/) | [GitHub](https://github.com/AGButt04)  

## 📝 License
This project is for educational purposes.  

Built as part of my journey to become an AI/ML Engineer.  

