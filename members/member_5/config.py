MEMBER_NAME = "Member 5"
TARGET_INDICES = [10] # SCHD 미국배당다우존스 ETF

# [improve 7-3] STATIC 유지, use_vol=True (8-state) — 저변동성 ETF 신호 품질 개선
#               roll_period=60 — SCHD 느린 레짐 변화에 장기 롤링 재학습 적용
# [파라미터 — Simulation 저장: gap=0.5700, s_final=18.21%, v_final=17.58%]
RL_PARAMS = {
    TARGET_INDICES[0]: {
        "lr": 0.0624, "gamma": 0.9449, "epsilon": 0.1708, "v_epsilon": 0.0879,
        "episodes": 300, "train_episodes": 200, "seed": 314,
        "use_vol": True, "roll_period": 60,
        "algorithm": "STATIC_H"
    },
    "default": {
        "lr": 0.0624, "gamma": 0.9449, "epsilon": 0.1708, "v_epsilon": 0.0879,
        "episodes": 300, "train_episodes": 200, "seed": 314,
        "use_vol": True, "roll_period": 60,
        "algorithm": "STATIC_H"
    },
}
