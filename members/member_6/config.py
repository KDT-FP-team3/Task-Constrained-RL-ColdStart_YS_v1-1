MEMBER_NAME = "Member 6"
TARGET_INDICES = [11] # 로열 골드

# [파라미터 — Simulation 저장: gap=12.1685, s_final=105.17%, v_final=91.93%]
RL_PARAMS = {
    TARGET_INDICES[0]: {
        "lr": 0.0222, "gamma": 0.950344, "epsilon": 0.164422, "v_epsilon": 0.139944,
        "episodes": 300, "train_episodes": 150, "seed": 100,
        "use_vol": True, "roll_period": None,
        "algorithm": "STATIC_H"
    },
    "default": {
        "lr": 0.0222, "gamma": 0.950344, "epsilon": 0.164422, "v_epsilon": 0.139944,
        "episodes": 300, "train_episodes": 150, "seed": 100,
        "use_vol": True, "roll_period": None,
        "algorithm": "STATIC_H"
    },
}
