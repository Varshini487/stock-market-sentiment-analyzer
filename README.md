# 📈 Stock Market Sentiment Analyzer

A multi-source sentiment analysis system that combines news, Twitter, and Reddit to predict stock price movement.

## 🎯 Approach
- **News Sentiment:** TextBlob on financial news articles
- **Social Sentiment:** VADER on Twitter & Reddit (optimized for slang/emoji)
- **Technical Indicators:** RSI, Moving Averages, VIX
- **Ensemble:** XGBoost combines all signals

## 📊 Performance
| Signal | Accuracy |
|--------|----------|
| News alone | 58% |
| Technical alone | 62% |
| Social alone | 61% |
| **Ensemble** | **71%** |

## 🛠️ Tech Stack
- NewsAPI, Tweepy (Twitter), PRAW (Reddit)
- VADER / TextBlob (sentiment)
- yfinance (stock data)
- XGBoost (prediction)
- Streamlit (UI)

## 🚀 Getting Started
```bash
git clone https://github.com/Varshini487/stock-market-sentiment-analyzer
cd stock-market-sentiment-analyzer
pip install -r requirements.txt
streamlit run app.py
```

## ⚠️ Disclaimer
This is for research/educational purposes. Markets are complex; sentiment is ONE signal, not a trading system.
