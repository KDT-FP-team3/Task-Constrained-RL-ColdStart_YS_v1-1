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
    9: {"ticker": "000660.KS", "name": "SK하이닉스"},   # 인덱스 변경
   10: {"ticker": "SCHD", "name": "미국배당다우존스 ETF"},  # 신규 추가
   11: {"ticker": "RGLD", "name": "로열 골드"}             # 신규 추가
}

# 종목별 거래 수수료 (매수·매도 각각 편도 기준)
# 국내주식: 위탁수수료 0.015% + 증권거래세 0.20% (매도 시)
# 미국주식·ETF: 위탁수수료 0.05% (매수·매도 동일, 거래세 없음)
FEE_REGISTRY = {
    "SPY":       {"buy": 0.0005, "sell": 0.0005,
                  "label": "미국 ETF — 매수 0.05% + 매도 0.05% = 왕복 0.10%"},
    "QQQ":       {"buy": 0.0005, "sell": 0.0005,
                  "label": "미국 ETF — 매수 0.05% + 매도 0.05% = 왕복 0.10%"},
    "^KS11":     {"buy": 0.00015, "sell": 0.00215,
                  "label": "국내 지수 — 매수 0.015% + 매도 0.215% (위탁+거래세) = 왕복 0.23%"},
    "^KQ11":     {"buy": 0.00015, "sell": 0.00215,
                  "label": "국내 지수 — 매수 0.015% + 매도 0.215% (위탁+거래세) = 왕복 0.23%"},
    "NVDA":      {"buy": 0.0005, "sell": 0.0005,
                  "label": "미국 주식 — 매수 0.05% + 매도 0.05% = 왕복 0.10%"},
    "TSLA":      {"buy": 0.0005, "sell": 0.0005,
                  "label": "미국 주식 — 매수 0.05% + 매도 0.05% = 왕복 0.10%"},
    "GOOGL":     {"buy": 0.0005, "sell": 0.0005,
                  "label": "미국 주식 — 매수 0.05% + 매도 0.05% = 왕복 0.10%"},
    "MSFT":      {"buy": 0.0005, "sell": 0.0005,
                  "label": "미국 주식 — 매수 0.05% + 매도 0.05% = 왕복 0.10%"},
    "005930.KS": {"buy": 0.00015, "sell": 0.00215,
                  "label": "국내 주식 — 매수 0.015% + 매도 0.215% (위탁+거래세) = 왕복 0.23%"},
    "000660.KS": {"buy": 0.00015, "sell": 0.00215,
                  "label": "국내 주식 — 매수 0.015% + 매도 0.215% (위탁+거래세) = 왕복 0.23%"},
    "SCHD":      {"buy": 0.0005, "sell": 0.0005,
                  "label": "미국 ETF — 매수 0.05% + 매도 0.05% = 왕복 0.10%"},
    "RGLD":      {"buy": 0.0005, "sell": 0.0005,
                  "label": "미국 주식 — 매수 0.05% + 매도 0.05% = 왕복 0.10%"},
}

def get_stock_by_index(index):
    return STOCK_REGISTRY.get(index, None)

def get_ticker_by_name(name):
    """이름으로 티커를 역추적하는 편의 함수"""
    for info in STOCK_REGISTRY.values():
        if info["name"] == name:
            return info["ticker"]
    return None

def get_fee_info(ticker):
    """티커에 해당하는 수수료 정보 반환. 미등록 종목은 미국주식 기준 적용."""
    return FEE_REGISTRY.get(ticker, {"buy": 0.0005, "sell": 0.0005,
                                     "label": "미국 주식 — 왕복 0.10%"})