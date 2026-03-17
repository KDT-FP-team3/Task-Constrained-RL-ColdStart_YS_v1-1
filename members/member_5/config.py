MEMBER_NAME = "Member 5"
TARGET_INDICES = [10] # SCHD 미국배당다우존스 ETF

# [파라미터 — Simulation 저장: gap=0.5700, s_final=18.21%, v_final=17.58%]
RL_PARAMS = {
    TARGET_INDICES[0]: {
        "lr": 0.0624, "gamma": 0.9449, "epsilon": 0.1708, "v_epsilon": 0.0879,
        "episodes": 300, "train_episodes": 150, "seed": 314,
        "use_vol": False, "roll_period": None,
        "algorithm": "STATIC"
    },
    "default": {
        "lr": 0.0624, "gamma": 0.9449, "epsilon": 0.1708, "v_epsilon": 0.0879,
        "episodes": 300, "train_episodes": 150, "seed": 314,
        "use_vol": False, "roll_period": None,
        "algorithm": "STATIC"
    },
}
