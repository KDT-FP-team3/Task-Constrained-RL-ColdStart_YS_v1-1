# 1. 메타 정보 및 종목 인덱스 선택
MEMBER_NAME = "Member 4 (US Tech)"
TARGET_STOCK_INDICES = [0, 1]  # 0: SPY, 1: QQQ

# 2. 기본 RL 파라미터
RL_PARAMS = {
    "learning_rate": 0.01,
    "discount_factor": 0.98,
    "exploration_rate": 0.10
}

# 3. 추가 확장 파라미터 (팀원이 자유롭게 추가 가능, 웹에 자동 표시됨)
CUSTOM_PARAMS = {
    "ema_window": 20,
    "use_volume_filter": True,
    "profit_take_threshold": 0.05
}

# 4. 커스텀 함수 주입 (선택사항)
def custom_reward_function(current_price, ema_value):
    """팀원이 독자적으로 개발한 커스텀 보상 함수 예시"""
    if current_price > ema_value:
        return 1.5
    return -1.0