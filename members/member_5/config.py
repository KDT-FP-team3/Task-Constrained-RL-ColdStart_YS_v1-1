MEMBER_NAME = "Member 5"
TARGET_INDICES = [2, 8] # KOSPI, Samsung

# [조정 근거] Alpha -64.9% → 삼성전자 HBM 수요로 강한 상승세, Vanilla 압도
# 전략: 더 긴 윈도우(400일)로 2024 조정 구간(KOSPI 2400선 붕괴 등) 포함
# epsilon↓(0.03): 극히 선택적 탐험 → EMA 필터 신호 최대 활용
# lr↓(0.003): Q값 급격한 변동 억제 → 상승장 노이즈에 강인
# KOSPI: 200일(약 10개월) → 조정+회복 사이클 1~2회 포함
# Samsung: 400일 → 2024년 초 고점~하락~2025년 회복 사이클 전체 포함
RL_PARAMS = {
    TARGET_INDICES[0]: {  # KOSPI: 10개월 윈도우
        "lr": 0.005, "gamma": 0.99, "epsilon": 0.05,
        "episodes": 200, "seed": 2026
    },
    TARGET_INDICES[1]: {  # Samsung: 400일로 전체 사이클 포함, 극저 탐험
        "lr": 0.003, "gamma": 0.99, "epsilon": 0.03,
        "episodes": 400, "seed": 2026
    },
    "default": {
        "lr": 0.005, "gamma": 0.99, "epsilon": 0.05, "episodes": 200, "seed": 2026
    }
}
