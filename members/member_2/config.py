MEMBER_NAME = "Member 2"
TARGET_INDICES = [1] # QQQ

# [파라미터 설정 근거 — improve 4-7 기준 최적값]
# ▶ episodes=500: 일봉 500일(약 2년) 기준, 워크포워드 70%(350일) 학습
# ▶ lr=0.0627: PG Actor-Critic Optimizer 탐색 최적값 (QQQ 기술주 지수 적합)
# ▶ gamma=0.8805: 단기 할인율 — 기술주 단기 변동성 최적화
# ▶ epsilon=0.1028: STATIC ε — 4상태 탐험율 (낮은 값으로 수렴성 강화)
# ▶ v_epsilon=0.1640: Vanilla ε — STATIC와 독립 최적화
# ▶ seed=137: QQQ(기술주 지수) — 분산 환경 안정 시드
RL_PARAMS = {
    TARGET_INDICES[0]: {
        "lr": 0.0627, "gamma": 0.8805, "epsilon": 0.1028, "v_epsilon": 0.1640,
        "episodes": 500, "train_episodes": 300, "seed": 137
    },
    "default": {
        "lr": 0.0627, "gamma": 0.8805, "epsilon": 0.1028, "v_epsilon": 0.1640,
        "episodes": 500, "train_episodes": 300, "seed": 42
    }
}
