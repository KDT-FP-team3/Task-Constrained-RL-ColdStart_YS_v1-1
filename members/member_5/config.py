MEMBER_NAME = "Member 5"
TARGET_INDICES = [6] # 구글

# [파라미터 — Simulation 저장: gap=-3.3183, s_final=112.48%, v_final=115.67%]
RL_PARAMS = {
    TARGET_INDICES[0]: {
        "lr": 0.005, "gamma": 0.905935, "epsilon": 0.141718, "v_epsilon": 0.172208,
        "episodes": 500, "train_episodes": 300, "seed": 42,
        "use_vol": False, "roll_period": None
    },
    "default": {
        "lr": 0.005, "gamma": 0.905935, "epsilon": 0.141718, "v_epsilon": 0.172208,
        "episodes": 500, "train_episodes": 300, "seed": 42,
        "use_vol": False, "roll_period": None
    },
}
