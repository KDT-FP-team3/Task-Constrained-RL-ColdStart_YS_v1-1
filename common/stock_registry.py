# 중앙 종목 관리: 인덱스 번호로 종목을 호출합니다.
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

def get_stock_by_index(index):
    return STOCK_REGISTRY.get(index, None)