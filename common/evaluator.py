import numpy as np


def calculate_softmax_weights(scores, temperature=1.0):
    """
    Softmax 기반 동적 포트폴리오 비중.
    높은 점수를 받은 멤버에게 더 많은 자본 비중을 배분합니다.

    Parameters
    ----------
    scores : array-like
        각 멤버의 성과 점수 (예: avg_return / (1 + abs(avg_mdd))).
    temperature : float
        낮을수록 최고 점수 집중, 높을수록 균등 배분 (기본 1.0).

    Returns
    -------
    np.ndarray — 합계가 1인 비중 배열
    """
    scores = np.array(scores, dtype=float)
    if len(scores) == 0:
        return np.array([])
    z = scores / max(temperature, 1e-9)
    z -= z.max()          # 수치 안정성: overflow 방지
    exp_z = np.exp(z)
    return exp_z / exp_z.sum()


def calculate_metrics(returns_percent_array):
    if len(returns_percent_array) == 0: return 0.0, 0.0
    total_return = returns_percent_array[-1]
    volatility = np.std(returns_percent_array)
    return round(total_return, 2), round(volatility, 2)

def calculate_mdd(returns_percent_array):
    if len(returns_percent_array) == 0: return 0.0
    wealth_index = 1.0 + (returns_percent_array / 100.0)
    peak = np.maximum.accumulate(wealth_index)
    drawdown = (wealth_index - peak) / peak
    max_drawdown = np.min(drawdown) * 100 
    return round(max_drawdown, 2)

def calculate_ctpt_and_color(lr, gamma, epsilon):
    # 파라미터 기반 3자리 성향 코드 생성
    type_code = "A" if lr >= 0.01 else "P"
    type_code += "L" if gamma >= 0.95 else "S"
    type_code += "V" if epsilon >= 0.10 else "R"
    
    color_map = {
        "PSR": "#607d8b", "PSV": "#ff9800", "PLR": "#3f51b5", "PLV": "#e91e63", 
        "ASR": "#f44336", "ASV": "#ffc107", "ALR": "#4caf50", "ALV": "#2196f3"
    }
    desc_map = {
        "PSR": "보수형", "PSV": "탐색형", "PLR": "신중한 장기형", "PLV": "유연한 장기형",
        "ASR": "단기 민첩형", "ASV": "단기 모험형", "ALR": "안정적 성장형", "ALV": "적응형 모험가"
    }
    return type_code, desc_map.get(type_code, "Unknown"), color_map.get(type_code, "#9e9e9e")