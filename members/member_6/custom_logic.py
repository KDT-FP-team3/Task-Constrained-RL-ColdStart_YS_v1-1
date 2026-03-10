import pandas as pd

def apply_volume_filter(df, volume_threshold_ratio=1.5):
    """
    팀원 6의 독자적 제약 조건 로직:
    최근 거래량이 20일 평균 거래량의 특정 비율(기본 1.5배)을 
    초과할 때만 매수를 허용하는 마스크를 생성합니다.
    """
    if "Volume" not in df.columns:
        return pd.Series([True] * len(df), index=df.index)
        
    avg_volume = df["Volume"].rolling(window=20).mean()
    valid_volume_mask = df["Volume"] > (avg_volume * volume_threshold_ratio)
    
    # 결측치 처리 (초기 20일)
    valid_volume_mask = valid_volume_mask.fillna(False)
    return valid_volume_mask