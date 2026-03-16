MEMBER_NAME = "Member 5"
TARGET_INDICES = [6] # GOOGL

# [파라미터 설정 근거 — 초기값, 미최적화]
# ▶ episodes=500: 일봉 500일(약 2년) 기준, 워크포워드 70%(350일) 학습
# ▶ lr=0.0500: 중간 학습률 — 대형 기술주 안정 수렴 초기값 (QQQ 기반)
# ▶ gamma=0.9100: 중간 할인율 — 기술주 중기 추세 반영
# ▶ epsilon=0.1000: STATIC ε — 4상태 탐험율 초기값
# ▶ v_epsilon=0.1000: Vanilla ε — 독립 탐험율 초기값
# ▶ seed=42: 기본값 (Simulation으로 최적화 필요)
# ※ GOOGL: 대형 기술주, QQQ 유사 특성 — Simulation 후 파라미터 갱신 권장
RL_PARAMS = {
    TARGET_INDICES[0]: {
        "lr": 0.0500, "gamma": 0.9100, "epsilon": 0.1000, "v_epsilon": 0.1000,
        "episodes": 500, "train_episodes": 300, "seed": 42,
        "use_vol": False, "roll_period": None   # [P3/P4] GOOGL: 초기값 — Simulation 최적화 후 갱신 권장
    },
    "default": {
        "lr": 0.0500, "gamma": 0.9100, "epsilon": 0.1000, "v_epsilon": 0.1000,
        "episodes": 500, "train_episodes": 300, "seed": 42,
        "use_vol": False, "roll_period": None
    }
}
