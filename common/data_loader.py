import yfinance as yf
import streamlit as st
import pandas as pd

@st.cache_data(ttl=3600)
def load_historical_data(ticker, start_date="2023-01-01", end_date="2024-01-01"):
    """
    yfinance를 사용하여 주가 데이터를 캐싱하여 로드합니다.
    여러 팀원이 동일한 종목(예: SPY)을 호출해도 API는 한 번만 요청됩니다.
    """
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        # 멀티인덱스 칼럼 정리 (yfinance 최신 버전 대응)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        st.error(f"{ticker} 데이터 로드 중 에러 발생: {e}")
        return pd.DataFrame()