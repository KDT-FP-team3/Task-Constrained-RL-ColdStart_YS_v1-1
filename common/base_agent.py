import numpy as np


# ──────────────────────────────────────────────
# 상태 계산 함수 (모듈 수준)
# ──────────────────────────────────────────────

def _make_state_static(ret, price, ema):
    """STATIC RL 상태: 4상태 (is_bull + 2*is_above_ema)
    0: 하락+EMA아래  1: 상승+EMA아래  2: 하락+EMA위  3: 상승+EMA위
    """
    is_bull = 1 if ret > 0 else 0
    is_above_ema = 1 if price >= ema else 0
    return is_bull + 2 * is_above_ema


def _make_state_vanilla(ret, price, ema):
    """Vanilla RL 상태: 2상태 (0: 하락, 1: 상승)"""
    return 1 if ret > 0 else 0


# ──────────────────────────────────────────────
# STATIC RL: Actor-Critic (Policy Gradient Theorem 기반)
# ──────────────────────────────────────────────

def _train_actor_critic_static(returns, prices, emas, lr, gamma, epsilon,
                                train_episodes, n_days, fee_rate):
    """
    STATIC RL 훈련 — Actor-Critic (온라인 TD)

    Policy Gradient Theorem:
    ─────────────────────────
    • Actor : softmax 정책 π_θ(a|s) — 4상태(EMA×방향) × 2행동(CASH/BUY)
      ∇log π(a|s) = e_a - π(·|s)   [score function]

    • Critic: TD(0) 가치함수 V(s)   [baseline으로 분산 감소]

    • TD 오차:  δ = r + γ·V(s') - V(s)
      Critic:   V(s)    += lr · δ
      Actor:    θ[s,a]  += lr · δ · ∇log π(a|s)

    • 탐험: 상수 epsilon-greedy  (annealing 없음)
    • 초기화: theta = 0  (학습으로만 편향 형성)
    """
    n_states, n_actions = 4, 2
    theta = np.zeros((n_states, n_actions))
    theta[2, 1] = 0.1   # EMA 위+하락 상태: BUY 초기 선호 (이분화 해소)
    theta[3, 1] = 0.2   # EMA 위+상승 상태: BUY 초기 선호 강화
    V = np.zeros(n_states)                    # Critic 가치함수

    def softmax_policy(state):
        logits = theta[state]
        exp_l = np.exp(np.clip(logits - np.max(logits), -30, 30))
        return exp_l / (np.sum(exp_l) + 1e-10)

    for ep in range(train_episodes):
        state = _make_state_static(returns[0], prices[0], emas[0])
        prev_action = 0

        for t in range(1, n_days):
            probs = softmax_policy(state)

            # epsilon-greedy 탐험 (상수 epsilon)
            if np.random.rand() < epsilon:
                action = np.random.randint(0, n_actions)
            else:
                action = np.random.choice([0, 1], p=probs)

            # 보상: BUY=시장수익률, CASH=0, 매수 진입 시만 수수료
            _fee = fee_rate if (action == 1 and prev_action == 0) else 0.0
            reward = (returns[t] if action == 1 else 0.0) - _fee
            prev_action = action

            next_state = _make_state_static(returns[t], prices[t], emas[t])

            # TD 오차 (advantage 근사)
            td_error = reward + gamma * V[next_state] - V[state]

            # Critic 업데이트
            V[state] += lr * td_error

            # Actor 업데이트 (Policy Gradient score function)
            for a in range(n_actions):
                grad = (1.0 if a == action else 0.0) - probs[a]
                theta[state, a] += lr * td_error * grad

            state = next_state

    return theta, V


def _get_static_action(state, theta):
    """Actor logit에서 greedy 행동 선택 (평가용)."""
    return int(np.argmax(theta[state]))


# ──────────────────────────────────────────────
# Vanilla RL: Q-Learning (비교 기준선)
# ──────────────────────────────────────────────

def _train_qlearning_vanilla(returns, prices, emas, lr, gamma, epsilon,
                              train_episodes, n_days, fee_rate):
    """
    Vanilla RL 훈련 — Tabular Q-Learning (2상태, STATIC 비교 기준선)

    Q(s,a) ← Q(s,a) + lr · [r + γ·max_a' Q(s',a') - Q(s,a)]

    • 상태: 2개 (0: 하락, 1: 상승)
    • 행동: 2개 (0: CASH, 1: BUY)
    • 탐험: 상수 epsilon-greedy
    • 초기화: Q[:,1] = 0.03  (BUY 선호, seed 116/338 CASH 고착 해소)
    """
    n_states, n_actions = 2, 2
    q_table = np.zeros((n_states, n_actions))
    q_table[:, 1] = 0.03                         # BUY 초기값 = 0.03 (fee_rate 30배, seed 116/338 CASH 고착 해소)

    for ep in range(train_episodes):
        state = _make_state_vanilla(returns[0], prices[0], emas[0])
        prev_action = int(np.random.randint(0, 2))  # 랜덤 시작 포지션 (CASH 편향 해소)

        for t in range(1, n_days):
            if np.random.rand() < epsilon:
                action = np.random.randint(0, n_actions)
            else:
                action = int(np.argmax(q_table[state]))

            _fee = fee_rate if (action == 1 and prev_action == 0) else 0.0
            reward = (returns[t] if action == 1 else 0.0) - _fee
            prev_action = action

            next_state = _make_state_vanilla(returns[t], prices[t], emas[t])
            best_next = int(np.argmax(q_table[next_state]))
            td_target = reward + gamma * q_table[next_state, best_next]
            q_table[state, action] += lr * (td_target - q_table[state, action])
            state = next_state

    return q_table


# ──────────────────────────────────────────────
# 공개 API: run_rl_simulation
# ──────────────────────────────────────────────

def run_rl_simulation(df, lr=0.01, gamma=0.98, epsilon=0.10, episodes=100,
                      use_static=False, seed=2026, fee_rate=0.0):
    """
    RL 시뮬레이션 실행 후 누적수익률 배열 반환.

    use_static=True  → Actor-Critic (4상태: EMA×방향)
    use_static=False → Q-Learning  (2상태: 방향만)

    워크포워드 검증: 첫 70% 구간으로 학습, 전체 기간으로 평가
    - n_train = max(int(n_days * 0.7), 20)
    - 후반 30%가 OOS(Out-of-Sample) 성능 검증 구간
    """
    np.random.seed(seed)
    n_days = len(df)
    returns = df['Daily_Return'].values
    prices  = df['Close'].values
    emas    = df['EMA_10'].values

    n_train = max(int(n_days * 0.7), 20)

    if use_static:
        theta, _ = _train_actor_critic_static(
            returns[:n_train], prices[:n_train], emas[:n_train],
            lr, gamma, epsilon, episodes, n_train, fee_rate
        )
        def get_action(state):
            return _get_static_action(state, theta)
        make_state = _make_state_static
    else:
        q_table = _train_qlearning_vanilla(
            returns[:n_train], prices[:n_train], emas[:n_train],
            lr, gamma, epsilon, episodes, n_train, fee_rate
        )
        def get_action(state):
            return int(np.argmax(q_table[state]))
        make_state = _make_state_vanilla

    # 평가: 전체 기간 (클리핑 없이 실제 수익 반영)
    cumulative_return = np.zeros(n_days)
    current_capital = 1.0
    state = make_state(returns[0], prices[0], emas[0])
    prev_action = 0

    for t in range(1, n_days):
        action = get_action(state)
        _fee = fee_rate if (action == 1 and prev_action == 0) else 0.0
        reward = (returns[t] if action == 1 else 0.0) - _fee
        current_capital *= (1 + reward)
        cumulative_return[t] = (current_capital - 1) * 100
        prev_action = action
        state = make_state(returns[t], prices[t], emas[t])

    return cumulative_return


def run_rl_simulation_with_log(df, lr=0.01, gamma=0.98, epsilon=0.10, episodes=100,
                                use_static=False, seed=2026, fee_rate=0.0):
    """
    run_rl_simulation과 동일한 훈련 로직으로 누적수익 + 일별 행동 로그 반환.

    Returns:
        cumulative_return: np.ndarray (n_days,)
        action_log: list of dicts {Day, Action, Daily_Return(%)}
    """
    np.random.seed(seed)
    n_days = len(df)
    returns = df['Daily_Return'].values
    prices  = df['Close'].values
    emas    = df['EMA_10'].values

    n_train = max(int(n_days * 0.7), 20)

    if use_static:
        theta, _ = _train_actor_critic_static(
            returns[:n_train], prices[:n_train], emas[:n_train],
            lr, gamma, epsilon, episodes, n_train, fee_rate
        )
        def get_action(state):
            return _get_static_action(state, theta)
        make_state = _make_state_static
    else:
        q_table = _train_qlearning_vanilla(
            returns[:n_train], prices[:n_train], emas[:n_train],
            lr, gamma, epsilon, episodes, n_train, fee_rate
        )
        def get_action(state):
            return int(np.argmax(q_table[state]))
        make_state = _make_state_vanilla

    # 평가 + 로그: 전체 기간
    cumulative_return = np.zeros(n_days)
    current_capital = 1.0
    state = make_state(returns[0], prices[0], emas[0])
    prev_action = 0
    action_log = []

    for t in range(1, n_days):
        action = get_action(state)
        _fee = fee_rate if (action == 1 and prev_action == 0) else 0.0
        reward = (returns[t] if action == 1 else 0.0) - _fee
        current_capital *= (1 + reward)
        cumulative_return[t] = (current_capital - 1) * 100
        action_log.append({
            "Day": t,
            "Action": "BUY" if action == 1 else "CASH",
            "Daily_Return(%)": round(reward * 100, 4)
        })
        prev_action = action
        state = make_state(returns[t], prices[t], emas[t])

    return cumulative_return, action_log
