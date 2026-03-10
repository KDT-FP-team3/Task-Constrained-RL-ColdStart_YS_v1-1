MEMBER_NAME = "Member 2"
TARGET_INDICES = [0, 5] # SPY, TSLA

# [조정 근거] Alpha +1.7% 양호 → 기존 방향 유지하되 안정성 소폭 강화
# SPY: 기존 유지 (이미 STATIC 우위)
# TSLA: epsilon↓, lr↓로 Q값 안정성 향상 → STATIC 우위 확대
RL_PARAMS = {
    TARGET_INDICES[0]: {  # SPY: 기존 파라미터 유지 (STATIC 이미 우세)
        "lr": 0.01, "gamma": 0.98, "epsilon": 0.10,
        "episodes": 100, "seed": 2026
    },
    TARGET_INDICES[1]: {  # TSLA: 변동성 높음, 안정 수렴 강화
        "lr": 0.005, "gamma": 0.99, "epsilon": 0.05,
        "episodes": 300, "seed": 2026
    },
    "default": {
        "lr": 0.01, "gamma": 0.98, "epsilon": 0.10, "episodes": 100, "seed": 2026
    }
}
