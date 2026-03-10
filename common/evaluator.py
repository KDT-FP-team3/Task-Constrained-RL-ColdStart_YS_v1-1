# common/evaluator.py
import numpy as np

# --- 1. 통계 지표 계산 함수 (기존 내용 유지) ---
def calculate_metrics(returns):
    if len(returns) == 0: return 0.0, 0.0
    total_return = returns[-1]
    volatility = np.std(returns)
    return round(total_return, 2), round(volatility, 2)

# --- 2. [신규 추가] Chainers Trading Personality Type (CTPT) 산출 및 색상 매핑 ---
# MBTI의 4자리 대신 RL 파라미터 3개를 기반으로 3자리 코드를 만듭니다.
# 부족한 설명은 제미나이의 논리로 채웠습니다.

def calculate_ctpt_and_color(lr, gamma, epsilon):
    """
    강화학습 파라미터(lr, gamma, epsilon)를 기반으로 3자리 투자 성향 코드와
    고유한 색상을 산출합니다.
    """
    
    # 임계값(Thresholds) 설정 - 연구 성향에 따라 조정 가능
    threshold_lr = 0.01
    threshold_gamma = 0.95
    threshold_epsilon = 0.10
    
    # 3자리 성향 코드 생성
    type_code = ""
    # 1. Past vs Adaptive
    type_code += "P" if lr < threshold_lr else "A"
    # 2. Short-term vs Long-term
    type_code += "S" if gamma < threshold_gamma else "L"
    # 3. Routine vs Adventurous
    type_code += "R" if epsilon < threshold_epsilon else "V"
    
    # 성향 유형별 고유 색상 매핑 (박사님 요청 사항 반영)
    # 성향의 분위기에 따라 색상을 다르게 배정했습니다.
    color_map = {
        "PSR": "#607d8b", "PSV": "#ff9800", "PLR": "#3f51b5", "PLV": "#e91e63",
        "ASR": "#f44336", "ASV": "#ffc107", "ALR": "#4caf50", "ALV": "#2196f3" # ALV: 적응-장기-공격 (스타일리시한 파랑)
    }
    
    # 유형에 대한 설명 (도넛 그래프 hover 시 표시용)
    type_desc_map = {
        "ALV": "Adaptive-Long-term-Adventurous (시장 적응형 모험가)",
        "PLR": "Patient-Long-term-Routine (신중한 장기 전략가)",
        # 필요에 따라 다른 유형의 설명도 추가할 수 있습니다.
    }
    
    color = color_map.get(type_code, "#9e9e9e") # 매핑 안 되면 회색
    desc = type_desc_map.get(type_code, "Unknown Type")
    
    return type_code, color, desc