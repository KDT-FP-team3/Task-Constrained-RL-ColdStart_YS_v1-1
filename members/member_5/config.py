MEMBER_NAME = "Member 5"
TARGET_INDICES = [4] # NVDA

# [파라미터 설정 근거 — improve 4-7 기준 최적값]
# ▶ episodes=500: 일봉 500일(약 2년) 기준, 워크포워드 70%(350일) 학습
# ▶ lr=0.0497: 중간 학습률 — NVDA 반도체 고변동성 환경에서 안정 수렴
# ▶ gamma=0.9183: 중간 할인율 — 반도체 사이클 단기~중기 반영
# ▶ epsilon=0.0443: 낮은 STATIC ε — 수렴된 정책에서 정밀 탐험
# ▶ v_epsilon=0.1055: Vanilla ε — 독립 최적화
# ▶ seed=314: NVDA(반도체 고변동성) — 수학적 다양성(π 근사), 고변동 환경 적합
# ▶ 결과: STATIC +195.85% vs Market +105.33% (Alpha +120.01%) — 최고 성과 종목
RL_PARAMS = {
    TARGET_INDICES[0]: {
        "lr": 0.0497, "gamma": 0.9183, "epsilon": 0.0443, "v_epsilon": 0.1055,
        "episodes": 500, "train_episodes": 300, "seed": 314
    },
    "default": {
        "lr": 0.0497, "gamma": 0.9183, "epsilon": 0.0443, "v_epsilon": 0.1055,
        "episodes": 500, "train_episodes": 300, "seed": 42
    }
}
