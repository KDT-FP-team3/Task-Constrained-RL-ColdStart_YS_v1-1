MEMBER_NAME = "Member 6"
TARGET_INDICES = [8] # 삼성전자

# [파라미터 — Simulation 저장: gap=-999.0000, s_final=0.00%, v_final=0.00%]
RL_PARAMS = {
    TARGET_INDICES[0]: {
        "lr": 0.017, "gamma": 0.92, "epsilon": 0.169364, "v_epsilon": 0.144686,
        "episodes": 300, "train_episodes": 150, "seed": 42,
        "use_vol": True, "roll_period": 20
    },
    "default": {
        "lr": 0.017, "gamma": 0.92, "epsilon": 0.169364, "v_epsilon": 0.144686,
        "episodes": 300, "train_episodes": 150, "seed": 42,
        "use_vol": True, "roll_period": 20
    },
}
