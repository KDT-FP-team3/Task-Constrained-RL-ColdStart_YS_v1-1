MEMBER_NAME = "Member 6 (US Market)"
TARGET_INDICES = [0, 1, 4] # SPY, QQQ, NVDA 자동 선택

RL_PARAMS = {"lr": 0.01, "gamma": 0.98}

# 팀원이 자유롭게 추가하는 파라미터 (app.py가 자동 감지)
CUSTOM_PARAMS = {
    "buy_probability_threshold": 0.85,
    "sell_signal_ema": 60,
    "user_notes": "미국 기술주 중심 공격적 투자 전략"
}