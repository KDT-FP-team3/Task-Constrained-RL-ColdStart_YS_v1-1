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

    Policy Gradient Theorem 적용:
    ─────────────────────────────
    • Actor: softmax 정책 π_θ(a|s) = softmax(θ[s,:])
      ∇log π(a|s) = e_a - π(·|s)   [score function]

    • Critic: TD(0) 가치함수 V(s)   [baseline]

    • REINFORCE with baseline:
      δ = r + γ·V(s') - V(s)       [TD 오차 = advantage 근사]
      Critic: V(s) += lr_c · δ
      Actor : θ[s,a] += lr_a · δ · ∇log π(a|s)
                      = lr_a · δ · (1[a==action] - π(a|s))

    EMA 위(state>=2)에서만 매수 허용 → 아래에서는 action=0 강제
    """
    n_states, n_actions = 4, 2

    # Actor logit 초기화 (낙관적: 상승+EMA위에서 매수 선호)
    theta = np.zeros((n_states, n_actions))
    theta[2, 1] = 0.5   # 하락+EMA위: 매수 가능
    theta[3, 1] = 1.2   # 상승+EMA위: 핵심 매수 신호

    # Critic 가치함수
    V = np.zeros(n_states)

    # Actor / Critic 학습률 분리
    lr_actor  = lr * 0.6   # 정책 업데이트 (신중)
    lr_critic = lr * 2.0   # 가치 추정 (빠른 수렴)

    # 엡실론 스케줄
    eps_start = min(epsilon * 3.5, 0.75)

    def softmax_policy(state, can_buy):
        logits = theta[state].copy()
        if not can_buy:
            logits[1] = -50.0          # 매수 불가 → 확률 0
        max_l = np.max(logits)
        exp_l = np.exp(np.clip(logits - max_l, -30, 30))
        return exp_l / (np.sum(exp_l) + 1e-10)

    for ep in range(train_episodes):
        eps_t = eps_start - (eps_start - epsilon) * ep / max(train_episodes - 1, 1)
        state = _make_state_static(returns[0], prices[0], emas[0])
        prev_action = 0

        for t in range(1, n_days):
            can_buy = (state >= 2)
            probs = softmax_policy(state, can_buy)

            # 엡실론-그리디 탐험
            if np.random.rand() < eps_t:
                action = np.random.choice([0, 1]) if can_buy else 0
            else:
                action = np.random.choice([0, 1], p=probs)

            # 보상 (수수료 + 클리핑)
            _fee = fee_rate if (action == 1 and prev_action == 0) else 0.0
            raw_reward = (returns[t] if action == 1 else 0.0) - _fee
            reward = max(min(raw_reward, 0.03), -0.025)
            prev_action = action

            next_state = _make_state_static(returns[t], prices[t], emas[t])

            # TD 오차 (advantage 근사)
            td_error = reward + gamma * V[next_state] - V[state]

            # Critic 업데이트: V(s) += lr_c · δ
            V[state] += lr_critic * td_error

            # Actor 업데이트: Policy Gradient Theorem
            # θ[s,a] += lr_a · δ · (1[a==action] - π(a|s))
            for a in range(n_actions):
                grad = (1.0 if a == action else 0.0) - probs[a]
                theta[state, a] += lr_actor * td_error * grad

            state = next_state

    return theta, V


def _get_static_action(state, theta):
    """Actor logit에서 greedy 행동 선택 (평가용)."""
    can_buy = (state >= 2)
    logits = theta[state].copy()
    if not can_buy:
        logits[1] = -50.0
    return int(np.argmax(logits))


# ──────────────────────────────────────────────
# Vanilla RL: Q-Learning (비교 기준선)
# ──────────────────────────────────────────────

def _train_qlearning_vanilla(returns, prices, emas, lr, gamma, epsilon,
                              train_episodes, n_days, fee_rate):
    """
    Vanilla RL 훈련 — Tabular Q-Learning (2상태)
    STATIC RL과의 비교를 위해 간단한 알고리즘 유지.
    """
    n_states, n_actions = 2, 2
    q_table = np.zeros((n_states, n_actions))
    q_table[1, 1] = 0.05   # 상승 상태 초기 매수 선호

    eps_start = min(epsilon * 3.0, 0.6)

    for ep in range(train_episodes):
        eps_t = eps_start - (eps_start - epsilon) * ep / max(train_episodes - 1, 1)
        state = _make_state_vanilla(returns[0], prices[0], emas[0])
        prev_action = 0

        for t in range(1, n_days):
            if np.random.rand() < eps_t:
                action = np.random.randint(0, n_actions)
            else:
                action = int(np.argmax(q_table[state]))

            _fee = fee_rate if (action == 1 and prev_action == 0) else 0.0
            raw_reward = (returns[t] if action == 1 else 0.0) - _fee
            reward = max(min(raw_reward, 0.03), -0.025)
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

    use_static=True  → Actor-Critic (Policy Gradient Theorem + REINFORCE with baseline)
    use_static=False → Q-Learning (Vanilla RL 기준선)
    """
    np.random.seed(seed)
    n_days = len(df)
    returns = df['Daily_Return'].values
    prices  = df['Close'].values
    emas    = df['EMA_10'].values

    if use_static:
        train_episodes = max(episodes * 3, 500)
        theta, _ = _train_actor_critic_static(
            returns, prices, emas, lr, gamma, epsilon,
            train_episodes, n_days, fee_rate
        )
        def get_action(state):
            return _get_static_action(state, theta)
        make_state = _make_state_static
    else:
        q_table = _train_qlearning_vanilla(
            returns, prices, emas, lr, gamma, epsilon,
            episodes, n_days, fee_rate
        )
        def get_action(state):
            return int(np.argmax(q_table[state]))
        make_state = _make_state_vanilla

    # 평가 단계
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

    if use_static:
        train_episodes = max(episodes * 3, 500)
        theta, _ = _train_actor_critic_static(
            returns, prices, emas, lr, gamma, epsilon,
            train_episodes, n_days, fee_rate
        )
        def get_action(state):
            return _get_static_action(state, theta)
        make_state = _make_state_static
    else:
        q_table = _train_qlearning_vanilla(
            returns, prices, emas, lr, gamma, epsilon,
            episodes, n_days, fee_rate
        )
        def get_action(state):
            return int(np.argmax(q_table[state]))
        make_state = _make_state_vanilla

    # 평가 + 로그
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
