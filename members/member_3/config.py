MEMBER_NAME = "Member 3"
TARGET_INDICES = [2] # KOSPI 지수

# [파라미터 — Simulation 저장: gap=13.5467, s_final=119.99%, v_final=106.15%]
# [improve 7-3] use_vol=True (8-state) — 변동성 신호 추가로 MDD -19.24% 개선 목표
#               BUY율 95% (285/299) → 고변동성 구간 CASH 전환 기회 확보
RL_PARAMS = {
    TARGET_INDICES[0]: {
        "lr": 0.005, "gamma": 0.908675, "epsilon": 0.146537, "v_epsilon": 0.19471,
        "episodes": 300, "train_episodes": 150, "seed": 2024,
        "use_vol": True, "roll_period": None,
        "algorithm": "STATIC"
    },
    "default": {
        "lr": 0.005, "gamma": 0.908675, "epsilon": 0.146537, "v_epsilon": 0.19471,
        "episodes": 300, "train_episodes": 150, "seed": 42,
        "use_vol": True, "roll_period": None,
        "algorithm": "STATIC"
    },
}
