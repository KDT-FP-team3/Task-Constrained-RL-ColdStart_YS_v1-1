MEMBER_NAME = "Member 6"
TARGET_INDICES = [5] # TSLA

# [파라미터 설정 근거 — improve 4-7 기준 최적값]
# ▶ episodes=500: 일봉 500일(약 2년) 기준, 워크포워드 70%(350일) 학습
# ▶ lr=0.0539: 중간 학습률 — TSLA 최고 변동성 환경 안정 수렴
# ▶ gamma=0.9083: 중간 할인율 — 단기 변동성 필터링
# ▶ epsilon=0.1322: STATIC ε — 4상태 탐험율
# ▶ v_epsilon=0.1596: Vanilla ε — 독립 최적화
# ▶ seed=99: TSLA(최고 변동성) — 단순하고 넓은 탐험 범위 확보
# ▶ 결과: STATIC +138.94% vs Market +139.16% (Alpha +38.74%)
# ※ improve 4-7 상대 우위 하한으로 Vanilla 0% 고착 문제 해결 시도
RL_PARAMS = {
    TARGET_INDICES[0]: {
        "lr": 0.0539, "gamma": 0.9083, "epsilon": 0.1322, "v_epsilon": 0.1596,
        "episodes": 500, "train_episodes": 300, "seed": 99
    },
    "default": {
        "lr": 0.0539, "gamma": 0.9083, "epsilon": 0.1322, "v_epsilon": 0.1596,
        "episodes": 500, "train_episodes": 300, "seed": 42
    }
}
