MEMBER_NAME = "Member 1"
TARGET_INDICES = [0] # S&P 500 ETF

# [파라미터 — Ghost best 반영: gap=7.8300, s_final=24.56%, v_final=15.66%]
RL_PARAMS = {
    TARGET_INDICES[0]: {
        "lr": 0.0786, "gamma": 0.8689, "epsilon": 0.1619, "v_epsilon": 0.1380,
        "episodes": 300, "train_episodes": 150, "seed": 42,
        "use_vol": False, "roll_period": None
    },
    "default": {
        "lr": 0.0786, "gamma": 0.8689, "epsilon": 0.1619, "v_epsilon": 0.1380,
        "episodes": 300, "train_episodes": 150, "seed": 42,
        "use_vol": False, "roll_period": None
    },
}
