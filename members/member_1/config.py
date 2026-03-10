MEMBER_NAME = "Member 1"
TARGET_INDICES = [0, 4] # SPY, QQQ, NVDA 자동 선택

# 멤버가 종목별로 기본값을 직접 입력해서 설정합니다!
RL_PARAMS = {
    "S&P 500 ETF": {
        "lr": 0.01, "gamma": 0.98, "epsilon": 0.10, 
        "episodes": 100, "seed": 2026
    },
    "엔비디아": {
        "lr": 0.05, "gamma": 0.90, "epsilon": 0.20, 
        "episodes": 300, "seed": 777
    },
    # 만약 리스트에 추가했지만 아래에 설정이 없는 종목을 위한 기본값
    "default": {
        "lr": 0.01, "gamma": 0.98, "epsilon": 0.10, "episodes": 100, "seed": 1234
    }
}