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
        
        # STATIC 에이전트의 제약 조건 센서
        # [수정] EMA_20 → EMA_10: 10일 EMA는 가격 변화에 더 빠르게 반응하여
        # 단기 조정 후 회복 구간에서도 price >= EMA 조건이 충족될 수 있게 합니다.
        # EMA_20은 반응이 느려 상승 회복기에도 40~60% 날을 '매수 금지'로 만들어
        # STATIC 에이전트가 영구 현금보유(수평 직선)가 되는 핵심 원인이었습니다.
        df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
        
        # 일일 수익률 계산 (강화학습 보상 산정용)
        df['Daily_Return'] = df['Close'].pct_change().fillna(0)
        
        # 이동평균 및 수익률 계산 시 초기 결측치(NaN)가 발생하므로 한 번 더 정리
        df = df.dropna()
        
        return df
    except Exception as e:
        st.error(f"{ticker} 데이터 로드 실패: {e}")
        return pd.DataFrame()