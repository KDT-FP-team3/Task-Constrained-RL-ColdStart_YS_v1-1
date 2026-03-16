MEMBER_NAME = "Member 1"
TARGET_INDICES = [0] # S&P 500 ETF

# [파라미터 — Simulation 저장: gap=1.2669, s_final=33.92%, v_final=32.57%]
RL_PARAMS = {
    TARGET_INDICES[0]: {
        "lr": 0.007952, "gamma": 0.928212, "epsilon": 0.132516, "v_epsilon": 0.174669,
        "episodes": 300, "train_episodes": 150, "seed": 42,
        "use_vol": False, "roll_period": None
    },
    "default": {
        "lr": 0.007952, "gamma": 0.928212, "epsilon": 0.132516, "v_epsilon": 0.174669,
        "episodes": 300, "train_episodes": 150, "seed": 42,
        "use_vol": False, "roll_period": None
    },
}
