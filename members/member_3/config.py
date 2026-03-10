MEMBER_NAME = "Member 3"
TARGET_INDICES = [1, 6] # QQQ, GOOGL

# [조정 근거] Alpha -5.2% → STATIC이 소폭 열위
# QQQ: 나스닥 기술주 ETF, 252일 윈도우로 기술주 조정 사이클 포함
# GOOGL: 장기 회복세, 300일로 충분한 학습 데이터 확보
# epsilon↓(0.05): 탐험 노이즈 감소 → EMA 위/아래 신호 선명화
# lr↓(0.005): Q값 안정 수렴
RL_PARAMS = {
    TARGET_INDICES[0]: {  # QQQ: 나스닥 ETF, 1년 윈도우
        "lr": 0.005, "gamma": 0.99, "epsilon": 0.05,
        "episodes": 252, "seed": 2026
    },
    TARGET_INDICES[1]: {  # GOOGL: 300일 윈도우, 안정 수렴
        "lr": 0.005, "gamma": 0.99, "epsilon": 0.05,
        "episodes": 300, "seed": 2026
    },
    "default": {
        "lr": 0.005, "gamma": 0.99, "epsilon": 0.05, "episodes": 252, "seed": 2026
    }
}
