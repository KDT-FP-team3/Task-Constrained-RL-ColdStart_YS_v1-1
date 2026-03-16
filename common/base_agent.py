import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# [P0] RL 알고리즘 제어 상수 — 변수 분리 원칙
# ──────────────────────────────────────────────────────────────────────────────
# 모든 매직넘버를 이름 있는 상수로 분리.
# 파라미터 하나를 바꿀 때 코드 내부를 탐색하지 않고 이 블록만 수정하면 됨.
# ══════════════════════════════════════════════════════════════════════════════

TRAIN_RATIO       = 0.7    # 워크포워드: 앞 70% → 학습 / 나머지 30% → OOS 평가
ENTROPY_COEFF     = 0.05   # r_eff = r + ENTROPY_COEFF × H(π)  (Buy&Hold 고착 방지, 4-8 확정)
Q_FLOOR_MARGIN    = 0.005  # Q[s,BUY] ≥ Q[s,CASH] + Q_FLOOR_MARGIN  (4-9 확정)
EMA_SIGNAL_WEIGHT = 2      # state = is_bull×1 + is_above_ema×EMA_SIGNAL_WEIGHT (비트 위치 가중)


# ══════════════════════════════════════════════════════════════════════════════
# [P3] 상태 인코딩 — 선형 조합 (Linear Combination) 원리 명시화
# ──────────────────────────────────────────────────────────────────────────────
# 상태 공간 S = 독립적인 이진 시장 신호들의 선형 조합 (비트 인코딩).
#
#   state = Σ(signal_i × 2^i)
#
# 각 signal_i 는 독립 시장 지표 (0 또는 1):
#   bit 0 (×1) : is_bull       = 단기 방향  (오늘 수익률 > 0)
#   bit 1 (×2) : is_above_ema  = 중기 추세  (현재가 ≥ EMA_10)
#   bit 2 (×4) : is_high_vol   = 변동성 레짐 (rolling_std > threshold) [P3 선택적]
#
# → signals 리스트 길이에 따라 상태 수 자동 결정:
#      len=2 → 4상태 (기본, 하위 호환)
#      len=3 → 8상태 (변동성 신호 추가 시)
# ══════════════════════════════════════════════════════════════════════════════

def _encode_state(signals: list) -> int:
    """이진 시장 신호 → 이산 상태 인덱스 (선형 조합 비트 인코딩).

    [강화학습] 상태 이산화 (State Aggregation) — 선형 조합 원리
    ─────────────────────────────────────────────────────────────
    state = Σ(signal_i × 2^i)

    신호 분리 원칙:
      • 각 signal_i 는 독립적 — 하나를 추가/제거해도 다른 신호 불변
      • 비트 위치가 곧 신호의 '중요도 가중치' (상위 비트 = 더 많은 상태 분기)
      • n개 신호 → 2^n 상태 (지수적 표현력, 선형 공간만 사용)

    Examples:
      [0, 0]    → 0  (하락 + EMA아래)
      [1, 0]    → 1  (상승 + EMA아래)
      [0, 1]    → 2  (하락 + EMA위)
      [1, 1]    → 3  (상승 + EMA위)
      [0, 0, 1] → 4  (하락 + EMA아래 + 고변동성)
      [1, 1, 1] → 7  (상승 + EMA위  + 고변동성)
    """
    return sum(s * (2 ** i) for i, s in enumerate(signals))


def _make_state_static(ret, price, ema, vol=None, vol_threshold=None):
    """STATIC RL 상태 공간 (P3: 변동성 신호 선택적 추가)

    [강화학습] 상태 이산화 — 선형 조합 비트 인코딩
    ─────────────────────────────────────────────
    기본 (4상태, vol=None):
      bit 0: is_bull      = 단기 방향    (ret > 0)
      bit 1: is_above_ema = 중기 추세    (price ≥ EMA_10)

    확장 (8상태, vol 제공 시):
      bit 0: is_bull      = 단기 방향
      bit 1: is_above_ema = 중기 추세
      bit 2: is_high_vol  = 변동성 레짐  (rolling_std > threshold)
        → 변동성이 높을 때 에이전트가 더 기민하게 반응하도록 상태 분기

    EMA_SIGNAL_WEIGHT(=2): is_above_ema 가 2번째 비트 위치에 배치됨.
    """
    signals = [int(ret > 0), int(price >= ema)]
    if vol is not None and vol_threshold is not None:
        signals.append(int(vol > vol_threshold))
    return _encode_state(signals)


def _make_state_vanilla(ret, price=None, ema=None):
    """Vanilla RL 상태 공간: 2상태 (가격 방향만)

    [강화학습] 단순 상태 공간 (비교 기준선용)
    ─────────────────────────────────────────
    bit 0: is_bull = ret > 0
      State 0: 하락일  State 1: 상승일

    price, ema 인수는 STATIC과 동일한 호출 패턴 호환을 위해 선언 (미사용).
    """
    return _encode_state([int(ret > 0)])


# ══════════════════════════════════════════════════════════════════════════════
# STATIC RL: Actor-Critic (Policy Gradient Theorem 기반)
# ══════════════════════════════════════════════════════════════════════════════

def _train_actor_critic_static(returns, prices, emas, lr, gamma, epsilon,
                                train_episodes, n_days, fee_rate,
                                vols=None, vol_threshold=None):
    """STATIC RL 훈련 — Actor-Critic (온라인 TD)

    [P3] vols, vol_threshold 추가 → n_states 자동 결정 (4 또는 8)

    Policy Gradient Theorem:
    ─────────────────────────
    • Actor : softmax 정책 π_θ(a|s) — n_states × 2행동(CASH/BUY)
      ∇log π(a|s) = e_a - π(·|s)   [score function]

    • Critic: TD(0) 가치함수 V(s)   [baseline으로 분산 감소]

    • TD 오차:  δ = r_eff + γ·V(s') - V(s)
      Critic:   V(s)    += lr · δ
      Actor:    θ[s,a]  += lr · δ · ∇log π(a|s)

    • 엔트로피 정규화 [P0]:
        r_eff = r + ENTROPY_COEFF · H(π)   (= r + 0.05·H(π))
    """
    # [P3] 변동성 신호 유무에 따라 상태 수 자동 결정
    use_vol  = (vols is not None and vol_threshold is not None)
    n_states = 8 if use_vol else 4
    n_actions = 2

    # ── Actor-Critic 초기화 ─────────────────────────────────────────────────
    theta = np.zeros((n_states, n_actions))
    # Cold-Start 초기화: fee_rate 비례 BUY 선호 logit 부여
    theta[1, 1] = max(0.05, fee_rate * 30)   # 상승+EMA아래: 미세 BUY 선호
    theta[2, 1] = max(0.1,  fee_rate * 50)   # 하락+EMA위:  BUY 선호 (조정 후 회복 기대)
    theta[3, 1] = max(0.2,  fee_rate * 80)   # 상승+EMA위:  BUY 선호 강화
    if use_vol:
        # 8상태: 고변동성 상태에도 동일 선호 패턴 적용 (bit2=1 → 상태 인덱스 +4)
        theta[5, 1] = max(0.05, fee_rate * 30)  # 상승+EMA아래+고변동성
        theta[6, 1] = max(0.1,  fee_rate * 50)  # 하락+EMA위+고변동성
        theta[7, 1] = max(0.2,  fee_rate * 80)  # 상승+EMA위+고변동성
    V = np.zeros(n_states)

    def softmax_policy(state):
        logits = theta[state]
        exp_l = np.exp(np.clip(logits - np.max(logits), -30, 30))
        return exp_l / (np.sum(exp_l) + 1e-10)

    # ── 훈련 루프 (Online TD Actor-Critic) ──────────────────────────────────
    for _ in range(train_episodes):
        _v0 = float(vols[0]) if use_vol else None
        state = _make_state_static(returns[0], prices[0], emas[0], _v0, vol_threshold)
        prev_action = 0

        for t in range(1, n_days):
            probs = softmax_policy(state)

            if np.random.rand() < epsilon:
                action = np.random.randint(0, n_actions)
            else:
                action = np.random.choice([0, 1], p=probs)

            _fee   = fee_rate if (action == 1 and prev_action == 0) else 0.0
            reward = (returns[t] if action == 1 else 0.0) - _fee
            prev_action = action

            _vt = float(vols[t]) if use_vol else None
            next_state = _make_state_static(returns[t], prices[t], emas[t], _vt, vol_threshold)

            # [P0] ENTROPY_COEFF 상수 사용 (구: 0.05 하드코딩)
            entropy  = -np.sum(probs * np.log(probs + 1e-10))
            td_error = (reward + ENTROPY_COEFF * entropy) + gamma * V[next_state] - V[state]

            V[state] += lr * td_error
            for a in range(n_actions):
                grad = (1.0 if a == action else 0.0) - probs[a]
                theta[state, a] += lr * td_error * grad

            state = next_state

    return theta, V


def _get_static_action(state, theta):
    """Actor logit에서 greedy 행동 선택 (평가용).

    [강화학습] 평가 단계: ε-greedy 없이 argmax(θ[s]) 결정론적 정책 사용.
    theta 크기에 무관하게 동작 (4상태/8상태 공통).
    """
    return int(np.argmax(theta[state]))


# ══════════════════════════════════════════════════════════════════════════════
# Vanilla RL: Q-Learning (비교 기준선)
# ══════════════════════════════════════════════════════════════════════════════

def _train_qlearning_vanilla(returns, prices, emas, lr, gamma, epsilon,
                              train_episodes, n_days, fee_rate):
    """Vanilla RL 훈련 — Tabular Q-Learning (2상태, STATIC 비교 기준선)

    Q(s,a) ← Q(s,a) + lr · [r + γ·max_a' Q(s',a') - Q(s,a)]

    [P0] Q_FLOOR_MARGIN 상수 사용 (구: 0.005 하드코딩)
    """
    n_states, n_actions = 2, 2
    q_table = np.zeros((n_states, n_actions))
    q_table[:, 1] = max(fee_rate * 50, 0.05)

    for ep in range(train_episodes):
        _eps = epsilon * max(1.0, 2.0 - 1.0 * ep / max(train_episodes - 1, 1))
        state = _make_state_vanilla(returns[0])
        prev_action = 1  # BUY 시작 고정 (CASH 편향 방지)

        for t in range(1, n_days):
            if np.random.rand() < _eps:
                action = np.random.randint(0, n_actions)
            else:
                action = int(np.argmax(q_table[state]))

            _fee   = fee_rate if (action == 1 and prev_action == 0) else 0.0
            reward = (returns[t] if action == 1 else 0.0) - _fee
            prev_action = action

            next_state = _make_state_vanilla(returns[t])
            best_next  = int(np.argmax(q_table[next_state]))
            td_target  = reward + gamma * q_table[next_state, best_next]
            q_table[state, action] += lr * (td_target - q_table[state, action])
            state = next_state

    # [P0] Q_FLOOR_MARGIN 상수 사용 (구: 0.005 하드코딩)
    q_table[0, 1] = max(float(q_table[0, 1]), float(q_table[0, 0]) + Q_FLOOR_MARGIN)
    q_table[1, 1] = max(float(q_table[1, 1]), float(q_table[1, 0]) + Q_FLOOR_MARGIN)

    return q_table


# ══════════════════════════════════════════════════════════════════════════════
# [P2] 공개 API: run_rl_simulation_with_log (핵심 함수)
# ──────────────────────────────────────────────────────────────────────────────
# P2: return_policy=True → theta 또는 q_table 추가 반환 (Explainable RL)
# P3: vols, vol_threshold → 8상태 변동성 신호 선택적 활성화
# P4: roll_period → OOS 구간 주기적 재학습 (Rolling Window)
# ══════════════════════════════════════════════════════════════════════════════

def run_rl_simulation_with_log(df, lr=0.01, gamma=0.98, epsilon=0.10, episodes=100,
                                use_static=False, seed=2026, fee_rate=0.0,
                                vols=None, vol_threshold=None,
                                roll_period=None, return_policy=False):
    """훈련 + 평가 실행 → 누적수익 + 일별 행동 로그 반환.

    Parameters (신규 추가)
    ──────────────────────
    vols          : np.ndarray 또는 None. 변동성 값 배열 (data_loader의 Rolling_Std 컬럼).
                    None이면 기존 4상태 동작 유지 (하위 호환).
    vol_threshold : float 또는 None. 변동성 이분 임계값.
                    None이면 훈련 구간 중위수로 자동 산출.
    roll_period   : int 또는 None. [P4] OOS 구간에서 매 roll_period 봉마다 재학습.
                    None이면 기존 고정 학습 동작 유지 (하위 호환).
                    STATIC RL 전용 (Vanilla RL은 구조 단순성 유지).
    return_policy : bool. True이면 (cumulative_return, action_log, policy) 반환.
                    policy = theta(STATIC) 또는 q_table(Vanilla).

    Returns
    ───────
    return_policy=False: (cumulative_return, action_log)           ← 기존 동작 유지
    return_policy=True : (cumulative_return, action_log, policy)
    """
    np.random.seed(seed)
    n_days  = len(df)
    returns = df['Daily_Return'].values
    prices  = df['Close'].values
    emas    = df['EMA_10'].values

    # [P3] 변동성 배열: 외부 인수 우선, 없으면 DataFrame 컬럼에서 자동 추출
    _vols = vols
    if _vols is None and 'Rolling_Std' in df.columns:
        _vols = df['Rolling_Std'].values

    # [P0] TRAIN_RATIO 상수 사용 (구: 0.7 하드코딩)
    n_train = max(int(n_days * TRAIN_RATIO), 20)

    # [P3] 변동성 임계값: None이면 훈련 구간 중위수로 자동 산출 (데이터 적응형)
    _vol_thr = vol_threshold
    if _vols is not None and _vol_thr is None:
        _vol_thr = float(np.median(_vols[:n_train]))

    # ── 학습 ──────────────────────────────────────────────────────────────────
    if use_static:
        _vols_train = _vols[:n_train] if _vols is not None else None
        theta, _ = _train_actor_critic_static(
            returns[:n_train], prices[:n_train], emas[:n_train],
            lr, gamma, epsilon, episodes, n_train, fee_rate,
            vols=_vols_train, vol_threshold=_vol_thr
        )
        _policy = [theta]   # 가변 컨테이너: P4 Rolling 재학습 시 인플레이스 갱신

        def get_action(s):
            return _get_static_action(s, _policy[0])

        def make_state(r, p, e, v=None):
            return _make_state_static(r, p, e, v, _vol_thr)

        _q_table = None
    else:
        q_table = _train_qlearning_vanilla(
            returns[:n_train], prices[:n_train], emas[:n_train],
            lr, gamma, epsilon, episodes, n_train, fee_rate
        )
        _q_table = q_table
        _policy  = None

        def get_action(s):
            return int(np.argmax(_q_table[s]))

        def make_state(r, p, e, v=None):
            return _make_state_vanilla(r)

    # ── 평가 (전체 기간) ──────────────────────────────────────────────────────
    cumulative_return = np.zeros(n_days)
    current_capital   = 1.0
    _v0    = float(_vols[0]) if _vols is not None else None
    state  = make_state(returns[0], prices[0], emas[0], _v0)
    prev_action = 0
    action_log  = []

    for t in range(1, n_days):
        # [P4] Rolling Window 재학습 — STATIC RL 전용
        #   OOS 구간(t ≥ n_train)에서 매 roll_period 봉마다 최근 n_train 봉으로 재학습.
        #   분포 변화(Distribution Shift) 대응: 시장 레짐 변화 시 정책 적응.
        if (use_static and roll_period is not None
                and t >= n_train and (t - n_train) % roll_period == 0):
            win_start  = max(0, t - n_train)
            _vols_win  = _vols[win_start:t] if _vols is not None else None
            _new_theta, _ = _train_actor_critic_static(
                returns[win_start:t], prices[win_start:t], emas[win_start:t],
                lr, gamma, epsilon, episodes, t - win_start, fee_rate,
                vols=_vols_win, vol_threshold=_vol_thr
            )
            _policy[0] = _new_theta  # 인플레이스 갱신 → get_action 자동 반영

        action = get_action(state)
        _fee   = fee_rate if (action == 1 and prev_action == 0) else 0.0
        reward = (returns[t] if action == 1 else 0.0) - _fee
        current_capital  *= (1 + reward)
        cumulative_return[t] = (current_capital - 1) * 100
        action_log.append({
            "Day":             t,
            "Action":          "BUY" if action == 1 else "CASH",
            "Daily_Return(%)": round(reward * 100, 4)
        })
        prev_action = action
        _vt   = float(_vols[t]) if _vols is not None else None
        state = make_state(returns[t], prices[t], emas[t], _vt)

    # [P2] return_policy=True 시 학습된 정책 반환 (Explainable RL 시각화용)
    policy_obj = _policy[0] if use_static else _q_table
    if return_policy:
        return cumulative_return, action_log, policy_obj
    return cumulative_return, action_log


# ══════════════════════════════════════════════════════════════════════════════
# 공개 API: run_rl_simulation (thin wrapper — 중복 제거, P2)
# ──────────────────────────────────────────────────────────────────────────────
# 기존: 훈련+평가 로직 완전 복사 (~60줄 중복)
# 개선: run_rl_simulation_with_log 호출로 위임 → 단일 코드 경로 유지
# ══════════════════════════════════════════════════════════════════════════════

def run_rl_simulation(df, lr=0.01, gamma=0.98, epsilon=0.10, episodes=100,
                      use_static=False, seed=2026, fee_rate=0.0,
                      vols=None, vol_threshold=None, roll_period=None):
    """누적수익률 배열만 반환하는 경량 래퍼.

    [P2] run_rl_simulation_with_log 에 위임하여 중복 코드 제거.
    Parameters: run_rl_simulation_with_log 참조.
    """
    result, _ = run_rl_simulation_with_log(
        df, lr=lr, gamma=gamma, epsilon=epsilon, episodes=episodes,
        use_static=use_static, seed=seed, fee_rate=fee_rate,
        vols=vols, vol_threshold=vol_threshold, roll_period=roll_period
    )
    return result
