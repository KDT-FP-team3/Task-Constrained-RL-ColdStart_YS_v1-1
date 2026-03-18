MEMBER_NAME = "Member 6"
TARGET_INDICES = [11] # RGLD 로열 골드

# [파라미터 — Simulation 저장: gap=17.8100, s_final=106.18%, v_final=88.25%]
# [improve 7-3] STATIC 유지, use_vol=True (8-state) — 원자재 사이클 변동성 신호로 시드 다양성 유도
RL_PARAMS = {
    TARGET_INDICES[0]: {
        "lr": 0.0269, "gamma": 0.9380, "epsilon": 0.1509, "v_epsilon": 0.1268,
        "episodes": 300, "train_episodes": 150, "seed": 99,
        "use_vol": True, "roll_period": None,
        "algorithm": "STATIC_H"
    },
    "default": {
        "lr": 0.0269, "gamma": 0.9380, "epsilon": 0.1509, "v_epsilon": 0.1268,
        "episodes": 300, "train_episodes": 150, "seed": 99,
        "use_vol": True, "roll_period": None,
        "algorithm": "STATIC_H"
    },
}
