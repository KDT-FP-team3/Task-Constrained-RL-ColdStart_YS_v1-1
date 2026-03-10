# 중앙 종목 레지스트리 (인덱스 매핑)
STOCK_REGISTRY = {
    0: {"ticker": "SPY", "name": "S&P 500 ETF"},
    1: {"ticker": "QQQ", "name": "Nasdaq 100 ETF"},
    2: {"ticker": "^KS11", "name": "KOSPI 지수"},
    3: {"ticker": "^KQ11", "name": "KOSDAQ 지수"},
    4: {"ticker": "NVDA", "name": "엔비디아"},
    5: {"ticker": "TSLA", "name": "테슬라"},
    6: {"ticker": "005930.KS", "name": "삼성전자"},
    7: {"ticker": "000660.KS", "name": "SK하이닉스"}
}

def get_stock_info(indices):
    """팀원이 선택한 인덱스 리스트를 받아 종목 정보를 반환합니다."""
    return [STOCK_REGISTRY[idx] for idx in indices if idx in STOCK_REGISTRY]