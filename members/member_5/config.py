MEMBER_NAME = "Member 5"
TARGET_INDICES = [10] # SCHD 미국배당다우존스 ETF

# [improve 7-3] algorithm: STATIC→A2C (신경망 5D 특징 — EMA/Bull 4-state 한계 극복)
#               SCHD 배당ETF: 일변동 낮아 tabular 4-state 신호 품질 부족
#               A2C 온라인 TD: 느린 레짐 변화 종목에 적합
#               epsilon=0.1708 → A2C ε-greedy 탐색률로 그대로 활용
RL_PARAMS = {
    TARGET_INDICES[0]: {
        "lr": 0.0624, "gamma": 0.9449, "epsilon": 0.1708, "v_epsilon": 0.0879,
        "episodes": 300, "train_episodes": 200, "seed": 314,
        "use_vol": True, "roll_period": None,
        "algorithm": "A2C"
    },
    "default": {
        "lr": 0.0624, "gamma": 0.9449, "epsilon": 0.1708, "v_epsilon": 0.0879,
        "episodes": 300, "train_episodes": 200, "seed": 314,
        "use_vol": True, "roll_period": None,
        "algorithm": "A2C"
    },
}
