import numpy as np


def run_rl_simulation(df, lr=0.01, gamma=0.98, epsilon=0.10, episodes=100, use_static=False, seed=2026):
    """
    Q-Learning 알고리즘을 지정된 에피소드(episodes) 횟수만큼 반복 학습한 후,
    최종 누적 수익률 배열을 반환합니다.
    """
    np.random.seed(seed)
    n_days = len(df)

    returns = df['Daily_Return'].values
    prices = df['Close'].values
    emas = df['EMA_20'].values

    # ==========================================
    # 상태 공간 정의
    # ==========================================
    # Vanilla: 2 상태 (0: 하락추세, 1: 상승추세)
    # STATIC: 4 상태 → EMA 위치 정보를 상태에 통합하여 학습 신호 분리
    #   state = is_bull + 2 * is_above_ema
    #   0: 하락추세 + EMA 아래 → 매수 금지
    #   1: 상승추세 + EMA 아래 → 매수 금지
    #   2: 하락추세 + EMA 위   → 매수 가능 (하락 주의)
    #   3: 상승추세 + EMA 위   → 매수 가능 (핵심 매수 신호)
    #
    # [2-state 문제점] STATIC에서 2-state를 사용하면 'EMA 위'와 'EMA 아래' 상황이
    # 같은 state로 뭉쳐져 Q-값이 서로 다른 시장 조건의 보상 신호를 혼합합니다.
    # 이로 인해 Q[state, 매수]가 수렴에 실패하거나 음수로 편향되어
    # 평가 단계에서 영구 현금보유(수평 직선)가 됩니다.

    if use_static:
        n_states = 4
        q_table = np.zeros((n_states, 2))
        # [핵심] 비대칭 낙관적 초기화 (Asymmetric Optimistic Initialization):
        # EMA 위 상태(2, 3)에서만 매수 Q-값을 현금보유 Q-값보다 높게 초기화.
        # → 초기 탐욕 정책이 EMA 위에서 매수를 선택 → 즉시 실제 보상 수집 시작
        # → 수익성이 있으면 Q[3,1]이 증가, 없으면 감소 → 빠른 수렴
        # np.full((2,2), 0.01) 방식은 argmax([0.01,0.01])=0 이므로 효과 없음
        q_table[2, 1] = 0.005   # 하락+EMA위: 약한 매수 초기화 (하락 중이라 불확실)
        q_table[3, 1] = 0.01    # 상승+EMA위: 강한 매수 초기화 (상승 + EMA 돌파 = 핵심 신호)
    else:
        n_states = 2
        q_table = np.zeros((n_states, 2))

    def make_state(ret, price, ema):
        """시장 관측치로부터 상태 인덱스를 계산"""
        is_bull = 1 if ret > 0 else 0
        if use_static:
            is_above_ema = 1 if price >= ema else 0
            return is_bull + 2 * is_above_ema  # 0, 1, 2, 3
        return is_bull

    # ==========================================
    # 1. 훈련 (Training) 단계
    # ==========================================
    for _ in range(episodes):
        state = make_state(returns[0], prices[0], emas[0])

        for t in range(1, n_days):
            # STATIC: 상태 자체에 EMA 위치가 인코딩됨 (state >= 2 → EMA 위 → 매수 가능)
            # Vanilla: 항상 매수 가능
            can_buy = (state >= 2) if use_static else True

            # Epsilon-Greedy 행동 선택 (탐험 시에도 STATIC 제약 적용)
            if np.random.rand() < epsilon:
                action = np.random.choice([0, 1]) if can_buy else 0
            else:
                q_values = q_table[state].copy()
                if not can_buy:
                    q_values[1] = -np.inf
                action = np.argmax(q_values)

            reward = returns[t] if action == 1 else 0
            next_state = make_state(returns[t], prices[t], emas[t])
            next_can_buy = (next_state >= 2) if use_static else True

            # Q-Table 업데이트 (벨만 방정식)
            next_q_values = q_table[next_state].copy()
            if not next_can_buy:
                next_q_values[1] = -np.inf

            best_next_action = np.argmax(next_q_values)
            td_target = reward + gamma * q_table[next_state, best_next_action]
            q_table[state, action] += lr * (td_target - q_table[state, action])

            state = next_state

    # ==========================================
    # 2. 평가 (Evaluation) 단계
    # ==========================================
    # 학습이 끝난 최적의 Q-Table을 바탕으로 최종 시뮬레이션을 수행합니다.
    cumulative_return = np.zeros(n_days)
    current_capital = 1.0
    state = make_state(returns[0], prices[0], emas[0])

    for t in range(1, n_days):
        can_buy = (state >= 2) if use_static else True
        q_values = q_table[state].copy()
        if not can_buy:
            q_values[1] = -np.inf
        action = np.argmax(q_values)

        reward = returns[t] if action == 1 else 0
        current_capital *= (1 + reward)
        cumulative_return[t] = (current_capital - 1) * 100

        state = make_state(returns[t], prices[t], emas[t])

    return cumulative_return
