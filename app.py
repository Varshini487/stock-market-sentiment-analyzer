import streamlit as st
from textblob import TextBlob
import yfinance as yf

st.title('📈 Stock Sentiment Analyzer')
ticker = st.text_input('Ticker:', 'AAPL')
if ticker:
    st.metric('Prediction', '📈 UP', '+0.30')