MEMBER_NAME = "Member 4"
TARGET_INDICES = [3] # KOSDAQ (^KQ11)

# [파라미터 설정 근거 — improve 4-7 기준 최적값]
# ▶ episodes=500: 일봉 500일(약 2년) 기준, 워크포워드 70%(350일) 학습
# ▶ lr=0.0168: 낮은 학습률 — KOSDAQ 고변동성 과적합 방지 최적값
# ▶ gamma=0.9084: 중간 할인율 — 단기/중기 균형
# ▶ epsilon=0.0863: 낮은 STATIC ε — 제한적 탐험으로 수렴성 강화
# ▶ v_epsilon=0.1157: Vanilla ε — 독립 최적화
# ▶ seed=777: KOSDAQ(소형주 고변동성) — 고분산 시장 탐험 시드
# ※ 구조적 OOS 주의: KOSPI와 동일한 워크포워드 한계 존재 (학습=하락, OOS=급등)
#   → sigma=0.233으로 수렴 미달; 현재 기준 최선값 사용
RL_PARAMS = {
    TARGET_INDICES[0]: {
        "lr": 0.0168, "gamma": 0.9084, "epsilon": 0.0863, "v_epsilon": 0.1157,
        "episodes": 500, "train_episodes": 300, "seed": 777,
        "use_vol": False, "roll_period": None   # [P3/P4] KOSDAQ: OOS 구조 한계 — 설정 변경 효과 미미
    },
    "default": {
        "lr": 0.0168, "gamma": 0.9084, "epsilon": 0.0863, "v_epsilon": 0.1157,
        "episodes": 500, "train_episodes": 300, "seed": 42,
        "use_vol": False, "roll_period": None
    }
}
