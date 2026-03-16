MEMBER_NAME = "Member 2"
TARGET_INDICES = [1] # QQQ

# [파라미터 설정 근거 — improve 4-9 기준 최적값]
# ▶ episodes=500: 일봉 500일(약 2년) 기준, 워크포워드 70%(350일) 학습
# ▶ lr=0.0650: PG Actor-Critic Optimizer 탐색 최적값 (improve 4-8 코드 기반 재탐색)
# ▶ gamma=0.9075: 중간 할인율 — 기술주 중기 추세 반영 (4-7의 0.8805에서 상향)
# ▶ epsilon=0.1005: STATIC ε — 4상태 탐험율 (entropy_coeff=0.05 환경)
# ▶ v_epsilon=0.1043: Vanilla ε — STATIC와 독립 최적화 (Q-floor margin 0.005 환경)
# ▶ seed=137: QQQ(기술주 지수) — 분산 환경 안정 시드
RL_PARAMS = {
    TARGET_INDICES[0]: {
        "lr": 0.0650, "gamma": 0.9075, "epsilon": 0.1005, "v_epsilon": 0.1043,
        "episodes": 500, "train_episodes": 300, "seed": 137,
        "use_vol": False, "roll_period": None   # [P3/P4] QQQ: 기술 지수 — 4상태 고정, 재학습 불필요
    },
    "default": {
        "lr": 0.0650, "gamma": 0.9075, "epsilon": 0.1005, "v_epsilon": 0.1043,
        "episodes": 500, "train_episodes": 300, "seed": 42,
        "use_vol": False, "roll_period": None
    }
}
