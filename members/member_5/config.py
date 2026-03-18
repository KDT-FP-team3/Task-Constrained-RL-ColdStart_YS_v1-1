MEMBER_NAME = "Member 5"
TARGET_INDICES = [10] # 미국배당다우존스 ETF

# [파라미터 — Simulation 저장: gap=1.9999, s_final=28.61%, v_final=26.44%]
RL_PARAMS = {
    TARGET_INDICES[0]: {
        "lr": 0.025708, "gamma": 0.917577, "epsilon": 0.037793, "v_epsilon": 0.106362,
        "episodes": 500, "train_episodes": 400, "seed": 314,
        "use_vol": True, "roll_period": 60,
        "algorithm": "STATIC_H"
    },
    "default": {
        "lr": 0.025708, "gamma": 0.917577, "epsilon": 0.037793, "v_epsilon": 0.106362,
        "episodes": 500, "train_episodes": 400, "seed": 314,
        "use_vol": True, "roll_period": 60,
        "algorithm": "STATIC_H"
    },
}
