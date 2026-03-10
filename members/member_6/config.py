MEMBER_NAME = "Member 6"
TARGET_INDICES = [3, 9] # SPY, QQQ, NVDA 자동 선택

# 종목별 완전히 독립적인 파라미터 셋업!
RL_PARAMS = {
    "default": {
        "lr": 0.01, "gamma": 0.98, "epsilon": 0.1, "episodes": 100, "seed": 2026
    },
    "엔비디아": {
        "lr": 0.05,            # 엔비디아는 변동성이 크므로 학습률을 높임
        "gamma": 0.90,         # 단기 추세에 민감하게 반응
        "epsilon": 0.2, 
        "episodes": 200,       # 학습을 더 많이 시킴
        "seed": 777            # 고유 시드 사용
    }
}