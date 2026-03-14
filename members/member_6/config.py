MEMBER_NAME = "Member 6"
TARGET_INDICES = [5] # TSLA

# [파라미터 설정 근거]
# ▶ episodes=500: 일봉 500일(약 2년) 기준, 워크포워드 70%(350일) 학습
# ▶ lr=0.03: Actor-Critic 정책 수렴에 적합한 학습률
# ▶ gamma=0.93: 일간 단기 할인율 (장기편향 방지)
# ▶ epsilon=0.15: 4상태 탐험에 충분한 탐험율
# ▶ seed=99: TSLA(최고 변동성) — 단순하고 넓은 탐험 범위 확보
RL_PARAMS = {
    TARGET_INDICES[0]: {
        "lr": 0.03, "gamma": 0.93, "epsilon": 0.15,
        "episodes": 500, "train_episodes": 300, "seed": 99
    },
    "default": {
        "lr": 0.03, "gamma": 0.93, "epsilon": 0.15, "episodes": 500, "train_episodes": 300, "seed": 42
    }
}
