MEMBER_NAME = "Member 1"
TARGET_INDICES = [0] # S&P 500 ETF

# [파라미터 — Simulation 저장: gap=8.9587, s_final=23.68%, v_final=14.54%]
RL_PARAMS = {
    TARGET_INDICES[0]: {
        "lr": 0.07268, "gamma": 0.938065, "epsilon": 0.133828, "v_epsilon": 0.102277,
        "episodes": 500, "train_episodes": 300, "seed": 42,
        "use_vol": True, "roll_period": None,
        "algorithm": "STATIC_H"
    },
    "default": {
        "lr": 0.07268, "gamma": 0.938065, "epsilon": 0.133828, "v_epsilon": 0.102277,
        "episodes": 500, "train_episodes": 300, "seed": 42,
        "use_vol": True, "roll_period": None,
        "algorithm": "STATIC_H"
    },
}
