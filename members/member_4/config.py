MEMBER_NAME = "Member 4"
TARGET_INDICES = [3] # KOSDAQ 지수

# [파라미터 — Simulation 저장: gap=22.4345, s_final=53.38%, v_final=29.87%]
RL_PARAMS = {
    TARGET_INDICES[0]: {
        "lr": 0.038007, "gamma": 0.921727, "epsilon": 0.07596, "v_epsilon": 0.061587,
        "episodes": 500, "train_episodes": 300, "seed": 777,
        "use_vol": True, "roll_period": 30,
        "algorithm": "STATIC_H"
    },
    "default": {
        "lr": 0.038007, "gamma": 0.921727, "epsilon": 0.07596, "v_epsilon": 0.061587,
        "episodes": 500, "train_episodes": 300, "seed": 42,
        "use_vol": True, "roll_period": 30,
        "algorithm": "STATIC_H"
    },
}
