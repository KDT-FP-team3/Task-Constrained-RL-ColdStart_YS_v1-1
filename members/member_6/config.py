MEMBER_NAME = "Member 6"
TARGET_INDICES = [5] # TSLA

# [파라미터 설정 근거 — improve 4-9 기준 최적값]
# ▶ episodes=500: 일봉 500일(약 2년) 기준, 워크포워드 70%(350일) 학습
# ▶ lr=0.0364: PG Actor-Critic Optimizer 탐색 최적값 (improve 4-8 코드 기반 재탐색)
# ▶ gamma=0.8873: 단기 할인율 — TSLA 단기 변동성 필터링
# ▶ epsilon=0.1283: STATIC ε — 4상태 탐험율 (entropy_coeff=0.05 환경)
# ▶ v_epsilon=0.0842: Vanilla ε — 독립 최적화 (Q-floor margin 0.005 환경)
# ▶ seed=99: TSLA(최고 변동성) — 단순하고 넓은 탐험 범위 확보
# ▶ 결과: STATIC +138.94% vs Market +139.16% (Alpha +38.74%)
RL_PARAMS = {
    TARGET_INDICES[0]: {
        "lr": 0.0364, "gamma": 0.8873, "epsilon": 0.1283, "v_epsilon": 0.0842,
        "episodes": 500, "train_episodes": 300, "seed": 99
    },
    "default": {
        "lr": 0.0364, "gamma": 0.8873, "epsilon": 0.1283, "v_epsilon": 0.0842,
        "episodes": 500, "train_episodes": 300, "seed": 42
    }
}
