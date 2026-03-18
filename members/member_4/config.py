MEMBER_NAME = "Member 4"
TARGET_INDICES = [3] # KOSDAQ 지수

# [improve 7-2] use_vol=True(8상태), roll_period=30 — KOSDAQ 레짐 변화 적응
# [파라미터 — Simulation 저장: gap=21.3568, s_final=48.50%, v_final=26.96%]
RL_PARAMS = {
    TARGET_INDICES[0]: {
        "lr": 0.038007, "gamma": 0.921727, "epsilon": 0.07596, "v_epsilon": 0.061587,
        "episodes": 300, "train_episodes": 150, "seed": 777,
        "use_vol": True, "roll_period": 30,
        "algorithm": "STATIC_H"
    },
    "default": {
        "lr": 0.038007, "gamma": 0.921727, "epsilon": 0.07596, "v_epsilon": 0.061587,
        "episodes": 300, "train_episodes": 150, "seed": 42,
        "use_vol": True, "roll_period": 30,
        "algorithm": "STATIC_H"
    },
}
