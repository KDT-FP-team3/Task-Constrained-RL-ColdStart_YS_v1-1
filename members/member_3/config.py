MEMBER_NAME = "Member 3"
TARGET_INDICES = [2] # KOSPI 지수

# [파라미터 — Simulation 저장: gap=17.5303, s_final=157.06%, v_final=137.93%]
RL_PARAMS = {
    TARGET_INDICES[0]: {
        "lr": 0.049807, "gamma": 0.880582, "epsilon": 0.023155, "v_epsilon": 0.160969,
        "episodes": 500, "train_episodes": 300, "seed": 2024,
        "use_vol": True, "roll_period": None,
        "algorithm": "STATIC_H"
    },
    "default": {
        "lr": 0.049807, "gamma": 0.880582, "epsilon": 0.023155, "v_epsilon": 0.160969,
        "episodes": 500, "train_episodes": 300, "seed": 42,
        "use_vol": True, "roll_period": None,
        "algorithm": "STATIC_H"
    },
}
