MEMBER_NAME = "Member 6"
TARGET_INDICES = [11] # RGLD 로열 골드

# [파라미터 — Simulation 저장: gap=17.8100, s_final=106.18%, v_final=88.25%]
# [improve 7-3] algorithm: STATIC→DDPG (연속 포지션 [0,1] — 원자재 사이클 특성에 부분 포지션 적합)
#               STATIC 5시드 완전 동일 결과(seed 무효) → DDPG OU noise로 탐색 다양성 복원
#               epsilon=0.20 (OU noise 스케일 강화 — RGLD 높은 변동성 탐색)
RL_PARAMS = {
    TARGET_INDICES[0]: {
        "lr": 0.0269, "gamma": 0.9380, "epsilon": 0.20, "v_epsilon": 0.1268,
        "episodes": 300, "train_episodes": 80, "seed": 99,
        "use_vol": False, "roll_period": None,
        "algorithm": "DDPG"
    },
    "default": {
        "lr": 0.0269, "gamma": 0.9380, "epsilon": 0.20, "v_epsilon": 0.1268,
        "episodes": 300, "train_episodes": 80, "seed": 99,
        "use_vol": False, "roll_period": None,
        "algorithm": "DDPG"
    },
}
