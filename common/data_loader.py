import yfinance as yf
import streamlit as st
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeout

# interval → 요청 기간 매핑 (yfinance API 제한 반영)
# 15m: 최대 60일 / 1h: 최대 730일(~2y) / 1d~: 무제한
_INTERVAL_PERIOD = {
    "15m": "60d",
    "1h":  "730d",
    "1d":  "2y",
    "1wk": "10y",
    "1mo": "10y",
}

_FETCH_TIMEOUT = 15  # yfinance 요청 최대 대기 시간 (초)


def _run_with_timeout(fn, timeout=_FETCH_TIMEOUT):
    """ThreadPoolExecutor로 fn()을 실행하고 timeout 초 내 결과 반환. 초과 시 None."""
    with ThreadPoolExecutor(max_workers=1) as _ex:
        _fut = _ex.submit(fn)
        try:
            return _fut.result(timeout=timeout)
        except _FuturesTimeout:
            return None
        except Exception:
            return None


def _postprocess_df(df: pd.DataFrame, interval: str = "1d") -> pd.DataFrame:
    """공통 후처리: 컬럼 정리 → EMA_10 → Daily_Return → dropna"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # 컬럼명 중복 제거 (yfinance 버전에 따라 발생)
    df = df.loc[:, ~df.columns.duplicated()]
    if 'Close' not in df.columns:
        return pd.DataFrame()
    df = df.dropna(subset=['Close'])
    # 인트라데이(15m/1h)는 datetime 유지, 그 외는 date로 변환
    if interval in ("15m", "1h"):
        df.index = pd.to_datetime(df.index)
    else:
        df.index = pd.to_datetime(df.index).date
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['Daily_Return'] = df['Close'].pct_change().fillna(0)
    # [P3] 변동성 신호: 10봉 Rolling 표준편차 (데이터 부족 구간은 expanding으로 보완)
    _std_rolling = df['Daily_Return'].rolling(10).std()
    _std_expand  = df['Daily_Return'].expanding().std()
    df['Rolling_Std'] = _std_rolling.where(_std_rolling.notna(), _std_expand).fillna(0.0)
    df = df.dropna()
    return df


@st.cache_data(ttl=3600)
def fetch_stock_data(ticker, period="2y", interval="1d"):
    """
    yfinance를 사용하여 주가 데이터를 로드합니다.
    interval: '15m' | '1h' | '1d' | '1wk' | '1mo'
    yf.download → yf.Ticker().history() 순으로 폴백하여 안정성 강화.
    각 요청은 _FETCH_TIMEOUT 초 내 응답이 없으면 포기하고 폴백으로 전환.
    """
    _period = _INTERVAL_PERIOD.get(interval, period)

    # ── 1차 시도: yf.download ──
    try:
        _raw = _run_with_timeout(
            lambda: yf.download(ticker, period=_period, interval=interval,
                                progress=False, auto_adjust=True)
        )
        if _raw is not None and not _raw.empty:
            result = _postprocess_df(_raw, interval)
            if not result.empty:
                return result
    except Exception:
        pass

    # ── 2차 시도 (폴백): yf.Ticker().history() ──
    try:
        _raw2 = _run_with_timeout(
            lambda: yf.Ticker(ticker).history(period=_period, interval=interval,
                                              auto_adjust=True)
        )
        if _raw2 is not None and not _raw2.empty:
            result = _postprocess_df(_raw2, interval)
            if not result.empty:
                return result
    except Exception as e:
        st.error(f"{ticker} 데이터 로드 실패: {e}")

    return pd.DataFrame()
