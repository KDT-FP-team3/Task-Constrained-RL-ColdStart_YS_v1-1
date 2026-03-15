MEMBER_NAME = "Member 1"
TARGET_INDICES = [0] # SPY

# [파라미터 설정 근거 — improve 4-9 기준 최적값]
# ▶ episodes=500: 일봉 500일(약 2년) 기준, 워크포워드 70%(350일) 학습
# ▶ lr=0.0496: PG Actor-Critic Optimizer 탐색 최적값 (improve 4-8 코드 기반 재탐색)
# ▶ gamma=0.8863: 단기 할인율 — 일봉 단기 거래 피드백 최적화
# ▶ epsilon=0.1190: STATIC ε — 4상태 탐험에 최적 (entropy_coeff=0.05 환경)
# ▶ v_epsilon=0.0993: Vanilla ε — STATIC와 독립 최적화 (Q-floor margin 0.005 환경)
# ▶ seed=42: SPY(안정 지수) — 재현성 클래식 시드
RL_PARAMS = {
    TARGET_INDICES[0]: {
        "lr": 0.0496, "gamma": 0.8863, "epsilon": 0.1190, "v_epsilon": 0.0993,
        "episodes": 500, "train_episodes": 300, "seed": 42
    },
    "default": {
        "lr": 0.0496, "gamma": 0.8863, "epsilon": 0.1190, "v_epsilon": 0.0993,
        "episodes": 500, "train_episodes": 300, "seed": 42
    }
}
