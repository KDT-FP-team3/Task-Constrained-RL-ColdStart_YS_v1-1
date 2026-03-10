MEMBER_NAME = "Member 6"
TARGET_INDICES = [3, 9] # SPY, QQQ, NVDA 자동 선택

# 종목별 완전히 독립적인 파라미터 셋업!
RL_PARAMS = {
    TARGET_INDICES[0]: {
        "lr": 0.01, "gamma": 0.98, "epsilon": 0.10, 
        "episodes": 100, "seed": 2026
    },
    TARGET_INDICES[1]: {
        "lr": 0.05, "gamma": 0.90, "epsilon": 0.20, 
        "episodes": 300, "seed": 777
    },
    "default": {
        "lr": 0.01, "gamma": 0.98, "epsilon": 0.10, "episodes": 100, "seed": 2026
    }
}