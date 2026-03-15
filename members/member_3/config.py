MEMBER_NAME = "Member 3"
TARGET_INDICES = [2] # KOSPI (^KS11)

# [파라미터 설정 근거 — improve 4-7 기준 최적값]
# ▶ episodes=500: 일봉 500일(약 2년) 기준, 워크포워드 70%(350일) 학습
# ▶ lr=0.0227: 낮은 학습률 — KOSPI 훈련 구간(횡보/하락) 과적합 방지
# ▶ gamma=0.9569: 높은 할인율 — 국내 지수 장기 추세 반영
# ▶ epsilon=0.1386: STATIC ε — 4상태 탐험율
# ▶ v_epsilon=0.1762: Vanilla ε — 높은 탐험으로 다양한 정책 탐색
# ▶ seed=2024: KOSPI(한국 시장) — 국내 시장 기준 연도 시드
# ※ 구조적 OOS 주의: 학습 구간(2024~2025 상반기)은 횡보, OOS(2025 하반기~)는 급등
#   → 시장 대비 alpha는 음수일 수 있으나 최선값 기준 파라미터 유지
RL_PARAMS = {
    TARGET_INDICES[0]: {
        "lr": 0.0227, "gamma": 0.9569, "epsilon": 0.1386, "v_epsilon": 0.1762,
        "episodes": 500, "train_episodes": 300, "seed": 2024
    },
    "default": {
        "lr": 0.0227, "gamma": 0.9569, "epsilon": 0.1386, "v_epsilon": 0.1762,
        "episodes": 500, "train_episodes": 300, "seed": 42
    }
}
