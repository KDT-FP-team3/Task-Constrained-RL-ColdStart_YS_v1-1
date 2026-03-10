MEMBER_NAME = "Member 6"
TARGET_INDICES = [3, 9] # KOSDAQ, SK Hynix

# [조정 근거] Alpha -121.5% → 가장 큰 격차
# SK하이닉스: HBM AI 수요 폭등으로 2024년 급등 후 2025년 조정
# 400일 윈도우로 고점(2024년 중반)~조정(2024년 하반기)~회복(2025년) 사이클 포함
# 이 구간에서 STATIC은 EMA 아래 구간(하락기) 현금보유로 손실 회피 → 우위 가능
# epsilon=0.03: 탐험을 최소화해 EMA 신호에 최대한 의존
# lr=0.003: 급등락 노이즈에 Q값이 흔들리지 않도록 극저 학습률
RL_PARAMS = {
    TARGET_INDICES[0]: {  # KOSDAQ: 200일 윈도우
        "lr": 0.005, "gamma": 0.99, "epsilon": 0.05,
        "episodes": 200, "seed": 2026
    },
    TARGET_INDICES[1]: {  # SK Hynix: 400일로 급등+조정 사이클 포함, 극저 탐험
        "lr": 0.003, "gamma": 0.99, "epsilon": 0.03,
        "episodes": 400, "seed": 2026
    },
    "default": {
        "lr": 0.005, "gamma": 0.99, "epsilon": 0.05, "episodes": 200, "seed": 2026
    }
}
