MEMBER_NAME = "Member 3"
TARGET_INDICES = [2] # KOSPI (^KS11)

# [파라미터 설정 근거]
# ▶ episodes=500: 일봉 500일(약 2년) 기준, 워크포워드 70%(350일) 학습
# ▶ lr=0.03: Actor-Critic 정책 수렴에 적합한 학습률
# ▶ gamma=0.93: 일간 단기 할인율 (장기편향 방지)
# ▶ epsilon=0.15: 4상태 탐험에 충분한 탐험율
# ▶ seed=2024: KOSPI(한국 시장) — 국내 시장 기준 연도 시드
RL_PARAMS = {
    TARGET_INDICES[0]: {
        "lr": 0.03, "gamma": 0.93, "epsilon": 0.15,
        "episodes": 500, "train_episodes": 300, "seed": 2024
    },
    "default": {
        "lr": 0.03, "gamma": 0.93, "epsilon": 0.15, "episodes": 500, "train_episodes": 300, "seed": 42
    }
}
