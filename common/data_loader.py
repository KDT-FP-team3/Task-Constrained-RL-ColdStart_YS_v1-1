import yfinance as yf
import streamlit as st
import pandas as pd

@st.cache_data(ttl=3600)
def fetch_stock_data(ticker, period="2y"):
    try:
        df = yf.download(ticker, period=period)
        if df.empty: return pd.DataFrame()
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.dropna()
        # 시간 데이터 제거 (오직 날짜만 남겨 겹침 방지)
        df.index = pd.to_datetime(df.index).date
        df['Daily_Return'] = df['Close'].pct_change().fillna(0)
        
        return df
    except Exception as e:
        st.error(f"{ticker} 데이터 로드 실패: {e}")
        return pd.DataFrame()