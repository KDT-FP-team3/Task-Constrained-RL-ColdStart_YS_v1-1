MEMBER_NAME = "Member 1"
TARGET_INDICES = [0] # S&P 500 ETF

# [파라미터 — Simulation 저장: gap=8.6600, s_final=24.04%, v_final=14.78%]
# [파라미터 — improve 7-1-5: gamma 0.8732→0.95 (장기 수익 중시), epsilon 0.1667→0.08 (과다 탐험 억제)]
RL_PARAMS = {
    TARGET_INDICES[0]: {
        "lr": 0.0802, "gamma": 0.95, "epsilon": 0.08, "v_epsilon": 0.1347,
        "episodes": 300, "train_episodes": 150, "seed": 42,
        "use_vol": False, "roll_period": None,
        "algorithm": "STATIC"
    },
    "default": {
        "lr": 0.0802, "gamma": 0.95, "epsilon": 0.08, "v_epsilon": 0.1347,
        "episodes": 300, "train_episodes": 150, "seed": 42,
        "use_vol": False, "roll_period": None,
        "algorithm": "STATIC"
    },
}
