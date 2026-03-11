MEMBER_NAME = "Member 6"
TARGET_INDICES = [5] # SPY, NVDA

# [조정 근거] episodes = Trading Days(df_full.tail(N)) + 학습 반복 횟수 동시 제어
# 기존 100일 → 최근 5개월만 학습: 단기 상승장에서 Vanilla 압도적 유리
# 조정: episodes↑(윈도우 확대→조정+회복 사이클 포함), epsilon↓(EMA 신호 순도↑),
#       lr↓(안정 수렴), gamma↑(장기 보상 중시)
RL_PARAMS = {
    TARGET_INDICES[0]: {
        "lr": 0.01, "gamma": 0.98, "epsilon": 0.10,
        "episodes": 180, "seed": 2026
    },
    "default": {
        "lr": 0.01, "gamma": 0.98, "epsilon": 0.10, "episodes": 100, "seed": 2026
    }
}
