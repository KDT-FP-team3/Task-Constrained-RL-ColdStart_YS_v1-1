MEMBER_NAME = "Member 3"
TARGET_INDICES = [2] # KOSPI 지수

# [파라미터 — Simulation 저장: gap=13.5467, s_final=119.99%, v_final=106.15%]
RL_PARAMS = {
    TARGET_INDICES[0]: {
        "lr": 0.005, "gamma": 0.908675, "epsilon": 0.146537, "v_epsilon": 0.19471,
        "episodes": 500, "train_episodes": 300, "seed": 2024,
        "use_vol": False, "roll_period": None
    },
    "default": {
        "lr": 0.005, "gamma": 0.908675, "epsilon": 0.146537, "v_epsilon": 0.19471,
        "episodes": 500, "train_episodes": 300, "seed": 42,
        "use_vol": False, "roll_period": None
    },
}
