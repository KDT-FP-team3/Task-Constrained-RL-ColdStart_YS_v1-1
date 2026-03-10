import yfinance as yf
import streamlit as st
import pandas as pd

@st.cache_data(ttl=3600)
def fetch_stock_data(ticker):
    # Streamlit Secrets에서 API 키를 가져옵니다. (필요 시 활용)
    api_key = st.secrets.get("api_keys", {}).get("alpha_vantage", "")
    
    # 기본적으로 yfinance를 사용하며, 고도화 시 Alpha Vantage API 로직을 추가합니다.
    df = yf.download(ticker, start="2024-01-01")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df