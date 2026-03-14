MEMBER_NAME = "Member 5"
TARGET_INDICES = [4] # NVDA

# [파라미터 설정 근거]
# ▶ episodes=500: 일봉 500일(약 2년) 기준, 워크포워드 70%(350일) 학습
# ▶ lr=0.03: Actor-Critic 정책 수렴에 적합한 학습률
# ▶ gamma=0.93: 일간 단기 할인율 (장기편향 방지)
# ▶ epsilon=0.15: 4상태 탐험에 충분한 탐험율
# ▶ seed=314: NVDA(반도체 고변동성) — 수학적 다양성, 고변동 환경 적합
RL_PARAMS = {
    TARGET_INDICES[0]: {
        "lr": 0.03, "gamma": 0.93, "epsilon": 0.15,
        "episodes": 500, "train_episodes": 100, "seed": 314
    },
    "default": {
        "lr": 0.03, "gamma": 0.93, "epsilon": 0.15, "episodes": 500, "train_episodes": 100, "seed": 42
    }
}
