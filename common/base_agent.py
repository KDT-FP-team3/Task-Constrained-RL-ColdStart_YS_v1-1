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
    emas = df['EMA_10'].values  # [수정] EMA_20 → EMA_10: data_loader 변경에 맞춰 업데이트

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

    if use_static:
        n_states = 4
        q_table = np.zeros((n_states, 2))
        # [수정] 비대칭 낙관적 초기화 값 대폭 상향:
        # 기존 Q[2,1]=0.005, Q[3,1]=0.01 → Q[2,1]=0.05, Q[3,1]=0.05
        # 이유: 초기값이 작으면 훈련 중 몇 번의 손실로도 Q[state,1]이 음수로 전락하여
        # 평가 시 영구 현금보유(수평 직선)가 됩니다. 높은 초기값은 더 많은 실제 경험
        # 데이터가 쌓인 후에야 매수 선호를 포기하도록 하여 빠른 수렴을 유도합니다.
        q_table[2, 1] = 0.05   # 하락+EMA위: 매수 가능, 불확실 구간
        q_table[3, 1] = 0.05   # 상승+EMA위: 핵심 매수 신호, 강한 초기 매수 선호

        # [수정] STATIC 추가 학습 반복: max(episodes * 2, 200)
        # EMA 위 상태(state 2,3) 방문 횟수가 적을 수 있으므로 더 많은 반복으로
        # Q[3,1]이 실제 보상을 충분히 반영하도록 합니다.
        train_episodes = max(episodes * 2, 200)
    else:
        n_states = 2
        q_table = np.zeros((n_states, 2))
        # [수정] Vanilla 낙관적 초기화: 상승 상태(state=1)에서 매수 Q값을 소폭 높게 설정
        # np.zeros 초기화 시 argmax([0,0])=0 → 항상 현금보유로 수렴하는 편향 제거
        # state=0 (하락추세): 매수 Q값 미설정 (현금보유 선호 유지)
        # state=1 (상승추세): 매수 Q값 0.01 설정 → 상승장에서 초기 매수 시도 유도
        q_table[1, 1] = 0.01   # 상승추세: 초기 매수 선호
        train_episodes = episodes

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
    for _ in range(train_episodes):
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
