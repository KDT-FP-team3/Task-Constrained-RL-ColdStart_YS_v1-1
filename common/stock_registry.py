# common/stock_registry.py
STOCK_REGISTRY = {
    0: {"ticker": "SPY", "name": "S&P 500 ETF"},
    1: {"ticker": "QQQ", "name": "Nasdaq 100 ETF"},
    2: {"ticker": "^KS11", "name": "KOSPI 지수"},
    3: {"ticker": "^KQ11", "name": "KOSDAQ 지수"},
    4: {"ticker": "NVDA", "name": "엔비디아"},
    5: {"ticker": "TSLA", "name": "테슬라"},
    6: {"ticker": "GOOGL", "name": "구글"},            # 신규 추가
    7: {"ticker": "MSFT", "name": "마이크로소프트"},    # 신규 추가
    8: {"ticker": "005930.KS", "name": "삼성전자"},     # 인덱스 변경
    9: {"ticker": "000660.KS", "name": "SK하이닉스"}    # 인덱스 변경
}

def get_stock_by_index(index):
    return STOCK_REGISTRY.get(index, None)

def get_ticker_by_name(name):
    """이름으로 티커를 역추적하는 편의 함수"""
    for info in STOCK_REGISTRY.values():
        if info["name"] == name:
            return info["ticker"]
    return None