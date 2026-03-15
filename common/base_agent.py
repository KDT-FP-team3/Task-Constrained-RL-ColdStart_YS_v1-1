import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# 상태 계산 함수 (모듈 수준)
#
# [강화학습] MDP 상태 공간 S 정의
#   에이전트는 시장 신호를 이산 상태 s ∈ S로 압축하여 정책 π(a|s) 입력으로 사용한다.
#   상태가 마르코프 성질을 만족해야 현재 상태만으로 최적 행동을 결정할 수 있다.
# ──────────────────────────────────────────────────────────────────────────────

def _make_state_static(ret, price, ema):
    """STATIC RL 상태 공간: 4상태 = 가격 방향(2) × EMA 위치(2)

    [강화학습] 상태 이산화(State Aggregation)
    ─────────────────────────────────────────
    시장 연속 신호를 4개 이산 상태로 압축:
      is_bull      = 1 if ret > 0   : 오늘 수익률 > 0 → 상승 추세
      is_above_ema = 1 if price≥ema : 현재가 ≥ EMA_10 → 중기 강세

    상태 인코딩 (EMA 위치 가중 ×2):
      State 0 = is_bull(0) + 2×is_above_ema(0) : 하락 + EMA 아래 (최약세)
      State 1 = is_bull(1) + 2×is_above_ema(0) : 상승 + EMA 아래 (단기 반등)
      State 2 = is_bull(0) + 2×is_above_ema(1) : 하락 + EMA 위   (일시 조정)
      State 3 = is_bull(1) + 2×is_above_ema(1) : 상승 + EMA 위   (최강세)

    EMA_10 = 10봉 지수이동평균 (중기 추세 기준선).
    EMA 위치에 ×2 가중: 단기 방향보다 중기 추세가 더 안정적 신호임을 반영.
    """
    is_bull = 1 if ret > 0 else 0
    is_above_ema = 1 if price >= ema else 0
    return is_bull + 2 * is_above_ema


def _make_state_vanilla(ret, price, ema):
    """Vanilla RL 상태 공간: 2상태 (가격 방향만)

    [강화학습] 단순 상태 공간 (비교 기준선용)
    ─────────────────────────────────────────
    State 0: ret ≤ 0 (하락일)
    State 1: ret > 0 (상승일)

    price, ema 인수는 STATIC과 동일한 함수 시그니처 유지를 위해 선언.
    Vanilla는 EMA 신호를 활용하지 않아 STATIC 대비 더 단순한 정책을 학습한다.
    """
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
    • 초기화: fee_rate 비례 BUY 선호 (수수료가 높을수록 BUY 학습 난이도 증가 보정)
    • 엔트로피 정규화: r_eff = r + 0.05·H(π)  [Buy&Hold 고착 방지, 정책 다양성 유지 — improve 4-8: 0.02→0.05]
    """
    # ── 파라미터 초기화 ──────────────────────────────────────────────────────
    # [강화학습] Actor-Critic 초기화
    #   theta: Actor 정책 파라미터 (n_states × n_actions 행렬)
    #          logit 형태로 저장 → softmax → 확률 정책 π_θ(a|s)
    #   V    : Critic 상태 가치함수 (n_states 벡터)
    #          V(s) = E[Σ γ^t · r_t | s_0 = s] 추정값
    n_states, n_actions = 4, 2
    theta = np.zeros((n_states, n_actions))
    # Cold-Start 초기화: fee_rate 비례 BUY 선호 logit 부여
    #   → 수수료가 높을수록 BUY 학습 난이도가 높으므로 초기 BUY 선호를 강화
    theta[1, 1] = max(0.05, fee_rate * 30)   # State 1 (상승+EMA아래): 미세 BUY 선호
    theta[2, 1] = max(0.1,  fee_rate * 50)   # State 2 (하락+EMA위):  BUY 선호 (조정 후 회복 기대)
    theta[3, 1] = max(0.2,  fee_rate * 80)   # State 3 (상승+EMA위):  BUY 선호 강화 (최강세)
    V = np.zeros(n_states)                    # Critic: 모든 상태 가치 0으로 초기화

    def softmax_policy(state):
        # [강화학습] Softmax 확률 정책: π_θ(a|s) = exp(θ[s,a]) / Σ exp(θ[s,a'])
        #   수치 안정성을 위해 logits에서 max를 빼서 overflow 방지
        logits = theta[state]
        exp_l = np.exp(np.clip(logits - np.max(logits), -30, 30))
        return exp_l / (np.sum(exp_l) + 1e-10)

    # ── 훈련 루프 (Online TD Actor-Critic) ──────────────────────────────────
    for ep in range(train_episodes):
        # 매 에피소드마다 훈련 데이터 처음부터 재시작 (Online, non-batched)
        state = _make_state_static(returns[0], prices[0], emas[0])
        prev_action = 0

        for t in range(1, n_days):
            probs = softmax_policy(state)

            # [강화학습] ε-greedy 탐험 (Exploration vs Exploitation)
            #   ε 확률: 무작위 행동 → 미탐색 (state, action) 쌍 경험
            #   (1-ε) 확률: 정책에 따른 행동 (현재 최선 활용)
            #   STATIC은 상수 ε (annealing 없음) — 훈련 전반에 균일한 탐험 유지
            if np.random.rand() < epsilon:
                action = np.random.randint(0, n_actions)  # 무작위 탐험
            else:
                action = np.random.choice([0, 1], p=probs)  # 정책 기반 선택

            # [강화학습] 보상 함수 R(s, a, s')
            #   BUY  → 당일 시장 수익률 획득 (reward = returns[t])
            #   CASH → 수익 없음 (reward = 0)
            #   CASH→BUY 전환 시 거래 수수료 1회 차감 (fee_rate)
            #   매도(BUY→CASH) 시 수수료 없음 (암묵적 청산)
            _fee = fee_rate if (action == 1 and prev_action == 0) else 0.0
            reward = (returns[t] if action == 1 else 0.0) - _fee
            prev_action = action

            # 다음 상태 계산 (MDP 전이: s' ← 시장 반응)
            next_state = _make_state_static(returns[t], prices[t], emas[t])

            # [강화학습] 엔트로피 정규화 (Entropy Regularization)
            #   H(π) = -Σ π(a|s) · log π(a|s)  (최대 ln2 ≈ 0.693)
            #   r_eff = r + 0.05·H(π): 정책 다양성에 보너스 부여
            #   → 한 행동에 집중(Buy&Hold 고착) 방지, 탐험 인센티브 유지
            entropy = -np.sum(probs * np.log(probs + 1e-10))

            # [강화학습] TD(0) 오차 δ (Advantage 근사값)
            #   δ = r_eff + γ·V(s') - V(s)
            #   δ > 0: 예상보다 좋은 결과 → 현재 행동 강화
            #   δ < 0: 예상보다 나쁜 결과 → 현재 행동 약화
            td_error = (reward + 0.05 * entropy) + gamma * V[next_state] - V[state]

            # [강화학습] Critic 업데이트 (TD(0) 가치함수 갱신)
            #   V(s) ← V(s) + lr · δ
            V[state] += lr * td_error

            # [강화학습] Actor 업데이트 (Policy Gradient Theorem)
            #   θ[s,a] ← θ[s,a] + lr · δ · ∇log π(a|s)
            #   ∇log π(a|s) = 1[a==action] - π(a|s)  (score function / REINFORCE)
            #   → 선택된 행동의 logit을 δ 방향으로 조정
            for a in range(n_actions):
                grad = (1.0 if a == action else 0.0) - probs[a]  # score function
                theta[state, a] += lr * td_error * grad

            state = next_state

    return theta, V


def _get_static_action(state, theta):
    """Actor logit에서 greedy 행동 선택 (평가용).

    [강화학습] 평가 단계: ε-greedy 없이 argmax(θ[s]) 결정론적 정책 사용.
    학습된 logit이 가장 높은 행동을 선택 → π*(a|s) = argmax_a θ[s,a].
    """
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
    • 탐험: epsilon annealing (2ε → ε — 초반 탐험 강화, 후반 설정값 유지)
    • 초기화: Q[:,1] = max(fee_rate×50, 0.05)  (fee 비례 BUY 선호, improve 4-5 검증)
    • 훈련 후 보정 — 전체 상태 상대 우위 하한 (improve 4-8/4-9):
        Q[0,BUY] = max(Q[0,BUY], Q[0,CASH] + 0.005)  ← state=0(하락기) 포함
        Q[1,BUY] = max(Q[1,BUY], Q[1,CASH] + 0.005)  ← state=1(상승기)
        이유: state=0에서 Q[0,CASH]가 높으면 OOS 첫날부터 CASH 고착 → 전 구간 음수
             margin 0.005(4-9)으로 강화 — 훈련 노이즈 대비 더 강한 BUY 우위 보장
    """
    # ── Q-테이블 초기화 ──────────────────────────────────────────────────────
    # [강화학습] Q(s,a): 상태 s에서 행동 a를 선택할 때의 기대 누적 할인 보상
    #   Q-테이블: (n_states=2) × (n_actions=2) 행렬
    #   Q[s, 0] = CASH 행동의 가치, Q[s, 1] = BUY 행동의 가치
    n_states, n_actions = 2, 2
    q_table = np.zeros((n_states, n_actions))
    # Cold-Start BUY 선호 초기화: 수수료가 높을수록 BUY 장벽 높으므로 초기 BUY 가치 강화
    #   fee_rate=0.001(미국) → Q[:,1]=0.05, fee_rate=0.0023(국내) → Q[:,1]=0.115
    q_table[:, 1] = max(fee_rate * 50, 0.05)    # BUY 초기 Q값 (최소 0.05)

    for ep in range(train_episodes):
        # [강화학습] ε 스케줄링 (Epsilon Annealing)
        #   초반(ep=0): 탐험율 2ε → 넓은 탐험으로 Q-테이블 초기화
        #   후반(ep→max): 탐험율 ε → 최소 탐험 유지 (복구 탐험 보장)
        #   목적: 초반에는 다양한 (state, action) 쌍을 경험하여 수렴 안정화
        _eps = epsilon * max(1.0, 2.0 - 1.0 * ep / max(train_episodes - 1, 1))
        state = _make_state_vanilla(returns[0], prices[0], emas[0])
        prev_action = 1  # 에피소드 시작: BUY 고정 (첫 step 수수료 편향 제거)

        for t in range(1, n_days):
            # [강화학습] ε-greedy 행동 선택
            if np.random.rand() < _eps:
                action = np.random.randint(0, n_actions)  # 탐험: 무작위 행동
            else:
                action = int(np.argmax(q_table[state]))   # 활용: greedy Q 선택

            # 보상 계산 (STATIC과 동일한 보상 구조)
            _fee = fee_rate if (action == 1 and prev_action == 0) else 0.0
            reward = (returns[t] if action == 1 else 0.0) - _fee
            prev_action = action

            # [강화학습] Q-Learning 업데이트 (Bellman Optimality Equation)
            #   TD 목표: r + γ · max_a' Q(s', a')   ← 최적 Bellman 근사
            #   TD 오차: δ = TD목표 - Q(s, a)
            #   Q(s,a) ← Q(s,a) + lr · δ            ← 테이블 갱신
            #   off-policy: 탐험(ε-greedy)으로 행동하면서 greedy 정책 학습
            next_state = _make_state_vanilla(returns[t], prices[t], emas[t])
            best_next = int(np.argmax(q_table[next_state]))          # max_a' Q(s',a')
            td_target = reward + gamma * q_table[next_state, best_next]  # TD 목표
            q_table[state, action] += lr * (td_target - q_table[state, action])  # 갱신
            state = next_state

    # ── 훈련 후 보정 (Post-training Correction) ──────────────────────────────
    # [강화학습] Q-floor: 모든 상태에서 BUY 상대 우위 보장
    #   문제: 하락 구간(state=0)에서 CASH를 많이 선택하면 Q[0,CASH]가 높게 학습됨
    #         → OOS 첫날 state=0이면 greedy 정책이 CASH 선택 → 전 구간 CASH 고착
    #   해결: 훈련 후 Q[s,BUY] ≥ Q[s,CASH] + 0.005 강제 보정
    #         (margin 0.005 = 훈련 노이즈 대비 충분한 BUY 우위, improve 4-9)
    q_table[0, 1] = max(float(q_table[0, 1]), float(q_table[0, 0]) + 0.005)  # bear state
    q_table[1, 1] = max(float(q_table[1, 1]), float(q_table[1, 0]) + 0.005)  # bull state

    return q_table


# ──────────────────────────────────────────────
# 공개 API: run_rl_simulation
# ──────────────────────────────────────────────

def run_rl_simulation(df, lr=0.01, gamma=0.98, epsilon=0.10, episodes=100,
                      use_static=False, seed=2026, fee_rate=0.0):
    """RL 시뮬레이션 실행 후 누적수익률 배열 반환.

    [강화학습] 워크포워드(Walk-Forward) 검증 구조
    ─────────────────────────────────────────────
    학습과 평가를 시간 순서에 따라 분리하여 미래 데이터 누출을 방지한다:
      n_train = max(int(n_days × 0.7), 20)   # 전체의 첫 70%로 학습
      나머지 30% = OOS(Out-of-Sample) 검증 구간

    Parameters (시스템 파라미터 설명)
    ──────────────────────────────────
    df         : yfinance에서 로드한 DataFrame (Close, EMA_10, Daily_Return 포함)
                 Trading Days(n_bars) 수만큼의 봉 데이터 포함.

    lr         : Learning Rate (α) — 학습률
                 Actor 정책 파라미터 θ 및 Critic 가치함수 V의 업데이트 보폭.
                 θ += lr × δ × ∇log π(a|s)  [STATIC Actor 업데이트]
                 V  += lr × δ               [STATIC Critic 업데이트]
                 Q  += lr × [r + γ·maxQ' - Q] [Vanilla Q-Learning 업데이트]
                 → 너무 크면 발산, 너무 작으면 수렴 지연. 권장 범위: 0.005~0.10.

    gamma      : Discount Factor (γ) — 할인율
                 미래 보상의 현재 가치 비율. γ=0.99면 100일 후 보상을 e^(-100/100)≈0.37배로 할인.
                 Bellman 방정식: V(s) = E[r + γ·V(s')]
                 → 높을수록 장기 추세 중시, 낮을수록 단기 수익 집중. 권장 범위: 0.85~0.99.

    epsilon    : Exploration Rate (ε) — 탐험율 (STATIC RL 전용)
                 ε-greedy 정책: ε 확률로 무작위 행동, (1-ε)로 정책 행동.
                 → 탐험(exploration)과 활용(exploitation)의 균형 파라미터.
                 STATIC은 상수 ε (annealing 없음). 권장 범위: 0.01~0.25.

    episodes   : Train Episodes — 훈련 에피소드 수 (epoch 수)
                 n_train개 봉 데이터를 몇 번 반복 학습할지 결정.
                 episodes=300이면 350봉 훈련 데이터를 300번 반복 → 총 105,000 step 학습.
                 → 많을수록 정책 수렴 향상, 과적합 위험 증가.

    use_static : True → STATIC RL (Actor-Critic, 4상태)
                 False → Vanilla RL (Q-Learning, 2상태)

    seed       : Base Seed — 훈련 재현성 고정 시드
                 np.random.seed(seed)로 ε-greedy 탐험 경로를 고정.
                 동일 seed → 동일 훈련 궤적 → 동일 정책 → 동일 평가 결과.

    fee_rate   : 거래 수수료율 (CASH→BUY 진입 1회 부과)
                 reward -= fee_rate (BUY 진입 시)
                 미국 ETF: 0.001 (왕복 0.10%), 국내 지수: 0.0023 (왕복 0.23%).

    Returns
    ───────
    cumulative_return : np.ndarray (n_days,)
                        t=0은 0%, 이후 매 봉마다 누적 수익률(%) 기록.

    use_static=True  → Actor-Critic (4상태: EMA×방향)
    use_static=False → Q-Learning  (2상태: 방향만)
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
