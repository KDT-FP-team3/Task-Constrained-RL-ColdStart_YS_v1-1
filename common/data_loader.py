import yfinance as yf
import streamlit as st
import pandas as pd

def _postprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """공통 후처리: 컬럼 정리 → EMA_10 → Daily_Return → dropna"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # 컬럼명 중복 제거 (yfinance 버전에 따라 발생)
    df = df.loc[:, ~df.columns.duplicated()]
    if 'Close' not in df.columns:
        return pd.DataFrame()
    df = df.dropna(subset=['Close'])
    df.index = pd.to_datetime(df.index).date
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['Daily_Return'] = df['Close'].pct_change().fillna(0)
    df = df.dropna()
    return df


@st.cache_data(ttl=3600)
def fetch_stock_data(ticker, period="2y"):
    """
    yfinance를 사용하여 최근 2년 치 주가 데이터를 로드합니다.
    yf.download → yf.Ticker().history() 순으로 폴백하여 안정성 강화.
    """
    # ── 1차 시도: yf.download (progress=False 로 콘솔 출력 억제) ──
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if not df.empty:
            result = _postprocess_df(df)
            if not result.empty:
                return result
    except Exception:
        pass

    # ── 2차 시도 (폴백): yf.Ticker().history() ──
    try:
        df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if not df.empty:
            result = _postprocess_df(df)
            if not result.empty:
                return result
    except Exception as e:
        st.error(f"{ticker} 데이터 로드 실패: {e}")

    return pd.DataFrame()