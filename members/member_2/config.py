MEMBER_NAME = "Member 2"
TARGET_INDICES = [1] # Nasdaq 100 ETF

# [파라미터 — Simulation 저장: gap=35.6212, s_final=73.90%, v_final=38.20%]
RL_PARAMS = {
    TARGET_INDICES[0]: {
        "lr": 0.072192, "gamma": 0.966283, "epsilon": 0.21158, "v_epsilon": 0.062005,
        "episodes": 300, "train_episodes": 150, "seed": 137,
        "use_vol": False, "roll_period": None
    },
    "default": {
        "lr": 0.072192, "gamma": 0.966283, "epsilon": 0.21158, "v_epsilon": 0.062005,
        "episodes": 300, "train_episodes": 150, "seed": 42,
        "use_vol": False, "roll_period": None
    },
}
