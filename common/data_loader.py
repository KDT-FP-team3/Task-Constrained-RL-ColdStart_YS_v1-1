import yfinance as yf
import streamlit as st
import pandas as pd

@st.cache_data(ttl=3600)
def fetch_stock_data(ticker, period="2y"):
    """
    yfinance를 사용하여 최근 2년 치 주가 데이터를 로드합니다.
    STATIC 제약 조건에 사용할 EMA(지수이동평균)를 함께 계산합니다.
    """
    try:
        # 2년치 데이터 로드
        df = yf.download(ticker, period=period)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # 결측치 처리
        df = df.dropna()
        
        # STATIC 센서: 20일 지수이동평균(EMA) 계산
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        
        # 일일 수익률 계산 (보상용)
        df['Daily_Return'] = df['Close'].pct_change().fillna(0)
        
        return df
    except Exception as e:
        st.error(f"{ticker} 데이터 로드 중 에러 발생: {e}")
        return pd.DataFrame()