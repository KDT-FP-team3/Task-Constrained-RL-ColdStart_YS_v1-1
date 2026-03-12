MEMBER_NAME = "Member 5"
TARGET_INDICES = [4] # NVDA

# [파라미터 최적화 근거 - improve 2-9]
# ▶ episodes=80: train_episodes=max(240,500)=500 확보, 평가 윈도우 80일 → 빠른 피드백
# ▶ lr=0.03: lr_actor=lr×1.0(구 0.6x) 기준 실질 5배 상향 → Actor-Critic 빠른 수렴
# ▶ gamma=0.93: 일간 단기거래 최적 할인율 (구 0.98은 장기편향 → TD오차 노이즈)
# ▶ epsilon=0.15: 4상태 공간 충분한 탐험 (구 0.10 대비 확장)
# ▶ seed=314: NVDA(반도체 고변동성) — 수학적 다양성, 고변동 환경 적합
# ▶ Auto Run Count 권장: 6 (n_iters=48, _n_eval=2 → 총 96회 평가, 균형적 속도/품질)
RL_PARAMS = {
    TARGET_INDICES[0]: {
        "lr": 0.03, "gamma": 0.93, "epsilon": 0.15,
        "episodes": 80, "seed": 314
    },
    "default": {
        "lr": 0.03, "gamma": 0.93, "epsilon": 0.15, "episodes": 80, "seed": 42
    }
}
