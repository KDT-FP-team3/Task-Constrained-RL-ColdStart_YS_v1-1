MEMBER_NAME = "Member 4"
TARGET_INDICES = [1, 7] # QQQ, MSFT

# [조정 근거] Alpha +5.4% 양호 → 안정성 강화로 우위 확대
# QQQ: 252일 윈도우 + epsilon↓로 STATIC 우위 공고화
# MSFT: 300일 유지하되 epsilon↓, lr↓로 Q값 안정화
RL_PARAMS = {
    TARGET_INDICES[0]: {  # QQQ: 1년 윈도우, 탐험 축소
        "lr": 0.005, "gamma": 0.99, "epsilon": 0.05,
        "episodes": 252, "seed": 2026
    },
    TARGET_INDICES[1]: {  # MSFT: 300일 유지, 안정 수렴
        "lr": 0.005, "gamma": 0.99, "epsilon": 0.05,
        "episodes": 300, "seed": 2026
    },
    "default": {
        "lr": 0.005, "gamma": 0.99, "epsilon": 0.05, "episodes": 252, "seed": 2026
    }
}
