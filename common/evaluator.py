import numpy as np

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