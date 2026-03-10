import yfinance as yf
import streamlit as st
import pandas as pd

@st.cache_data(ttl=3600)
def fetch_stock_data(ticker, period="2y"):
    """
    yfinance를 사용하여 최근 2년 치 주가 데이터를 로드합니다.
    인덱스 데이터 무결성을 강화하고 STATIC 제약 조건을 위한 지표를 계산합니다.
    """
    try:
        df = yf.download(ticker, period=period)
        if df.empty: return pd.DataFrame()
        
        # 멀티인덱스 컬럼 정리 (yfinance 최신 버전 대응)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.dropna()
        
        # 시간 데이터 제거 (오직 날짜만 남겨 그래프 x축 깔끔하게 정리)
        df.index = pd.to_datetime(df.index).date
        
        # 🌟 [복구된 핵심 로직] STATIC 에이전트의 제약 조건 센서 (20일 EMA)
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        
        # 일일 수익률 계산 (강화학습 보상 산정용)
        df['Daily_Return'] = df['Close'].pct_change().fillna(0)
        
        # 이동평균 및 수익률 계산 시 초기 결측치(NaN)가 발생하므로 한 번 더 정리
        df = df.dropna()
        
        return df
    except Exception as e:
        st.error(f"{ticker} 데이터 로드 실패: {e}")
        return pd.DataFrame()