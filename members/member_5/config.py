MEMBER_NAME = "Member 5"
TARGET_INDICES = [10] # SCHD 미국배당다우존스 ETF

# [파라미터 — 초기값 (미최적화): gap=0.0000, s_final=0.00%, v_final=0.00%]
RL_PARAMS = {
    TARGET_INDICES[0]: {
        "lr": 0.05, "gamma": 0.92, "epsilon": 0.12, "v_epsilon": 0.12,
        "episodes": 300, "train_episodes": 150, "seed": 42,
        "use_vol": False, "roll_period": None
    },
    "default": {
        "lr": 0.05, "gamma": 0.92, "epsilon": 0.12, "v_epsilon": 0.12,
        "episodes": 300, "train_episodes": 150, "seed": 42,
        "use_vol": False, "roll_period": None
    },
}
