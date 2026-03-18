MEMBER_NAME = "Member 2"
TARGET_INDICES = [1] # Nasdaq 100 ETF

# [파라미터 — Simulation 저장: gap=30.8096, s_final=48.40%, v_final=16.81%]
RL_PARAMS = {
    TARGET_INDICES[0]: {
        "lr": 0.080412, "gamma": 0.907645, "epsilon": 0.119118, "v_epsilon": 0.177335,
        "episodes": 500, "train_episodes": 300, "seed": 137,
        "use_vol": False, "roll_period": None,
        "algorithm": "STATIC_H"
    },
    "default": {
        "lr": 0.080412, "gamma": 0.907645, "epsilon": 0.119118, "v_epsilon": 0.177335,
        "episodes": 500, "train_episodes": 300, "seed": 42,
        "use_vol": False, "roll_period": None,
        "algorithm": "STATIC_H"
    },
}
