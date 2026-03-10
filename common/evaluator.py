import numpy as np

def calculate_expected_alpha(strategy_returns, benchmark_returns):
    """
    시장 평균(Benchmark) 대비 전략의 초과 수익률(Alpha) 기대치를 계산합니다.
    """
    if len(strategy_returns) == 0 or len(benchmark_returns) == 0:
        return 0.0
    return np.mean(strategy_returns) - np.mean(benchmark_returns)

def calculate_median_return(strategy_returns):
    """
    극단적인 이상치(Outlier)에 왜곡되지 않은 중앙값(Median) 수익률을 계산합니다.
    """
    if len(strategy_returns) == 0:
        return 0.0
    return np.median(strategy_returns)

def calculate_volatility(strategy_returns):
    """전략의 위험도(표준편차)를 계산합니다."""
    return np.std(strategy_returns)