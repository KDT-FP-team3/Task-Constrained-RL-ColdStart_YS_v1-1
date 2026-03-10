import numpy as np

def run_rl_simulation(df, lr=0.01, gamma=0.98, epsilon=0.10, episodes=100, use_static=False, seed=2026):
    """
    Q-Learning 알고리즘을 지정된 에피소드(episodes) 횟수만큼 반복 학습한 후,
    최종 누적 수익률 배열을 반환합니다.
    """
    np.random.seed(seed)
    n_days = len(df)

    # Q-Table: 상태(0:하락추세, 1:상승추세) x 행동(0:현금보유, 1:주식보유)
    # [수정] 낙관적 초기화(Optimistic Initialization): 초기값을 소폭 양수로 설정하여
    # 탐험 초기에 argmax([0,0])=0 으로 인한 영구 현금보유 고착을 방지합니다.
    q_table = np.full((2, 2), 0.01)

    returns = df['Daily_Return'].values
    prices = df['Close'].values
    emas = df['EMA_20'].values

    # ==========================================
    # 1. 훈련 (Training) 단계
    # ==========================================
    # 사이드바에서 설정한 Episodes 횟수만큼 과거 데이터를 반복 학습하여 Q-Table을 완성합니다.
    for _ in range(episodes):
        state = 1 if returns[0] > 0 else 0

        for t in range(1, n_days):
            # [수정] STATIC 제약 조건: 탐험(epsilon) 단계에서도 동일하게 적용
            # 기존 코드는 랜덤 탐험 시 제약을 무시하여, EMA 아래에서 불법 매수가 발생하고
            # 그로 인한 음수 보상이 Q[state, buy]를 오염시켜 STATIC 에이전트가
            # 평가 단계에서 영구 현금보유(수평 직선)가 되는 버그가 있었습니다.
            can_buy = not (use_static and prices[t-1] < emas[t-1])

            # Epsilon-Greedy 행동 선택 (제약 조건 반영)
            if np.random.rand() < epsilon:
                action = np.random.choice([0, 1]) if can_buy else 0
            else:
                q_values = q_table[state].copy()
                if not can_buy:
                    q_values[1] = -np.inf
                action = np.argmax(q_values)

            reward = returns[t] if action == 1 else 0
            next_state = 1 if returns[t] > 0 else 0

            # Q-Table 업데이트 (벨만 방정식)
            next_q_values = q_table[next_state].copy()
            if use_static and prices[t] < emas[t]:
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
    state = 1 if returns[0] > 0 else 0
    
    for t in range(1, n_days):
        # 평가 시에는 탐험(epsilon)을 하지 않고 오직 학습된 최선의 선택만 합니다.
        q_values = q_table[state].copy()
        if use_static and prices[t-1] < emas[t-1]:
            q_values[1] = -np.inf 
        action = np.argmax(q_values)
        
        # 보상 및 자본 업데이트
        reward = returns[t] if action == 1 else 0
        current_capital *= (1 + reward)
        cumulative_return[t] = (current_capital - 1) * 100
        
        state = 1 if returns[t] > 0 else 0

    return cumulative_return