MEMBER_NAME = "Member 1"
TARGET_INDICES = [0] # SPY

# [파라미터 설정 근거 — improve 4-7 기준 최적값]
# ▶ episodes=500: 일봉 500일(약 2년) 기준, 워크포워드 70%(350일) 학습
# ▶ lr=0.0577: PG Actor-Critic Optimizer 탐색 최적값 (SPY 안정 지수 적합)
# ▶ gamma=0.8938: 단기 할인율 — 일봉 단기 거래 피드백 최적화
# ▶ epsilon=0.1624: STATIC ε — 4상태 탐험에 최적
# ▶ v_epsilon=0.1706: Vanilla ε — STATIC와 독립 최적화
# ▶ seed=42: SPY(안정 지수) — 재현성 클래식 시드
RL_PARAMS = {
    TARGET_INDICES[0]: {
        "lr": 0.0577, "gamma": 0.8938, "epsilon": 0.1624, "v_epsilon": 0.1706,
        "episodes": 500, "train_episodes": 300, "seed": 42
    },
    "default": {
        "lr": 0.0577, "gamma": 0.8938, "epsilon": 0.1624, "v_epsilon": 0.1706,
        "episodes": 500, "train_episodes": 300, "seed": 42
    }
}
