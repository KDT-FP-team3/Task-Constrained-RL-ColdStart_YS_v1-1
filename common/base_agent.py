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

# ── 신경망 RL 알고리즘 공통 상수 ─────────────────────────────────────────────
NN_HIDDEN      = 32    # TinyMLP 히든 뉴런 수
N_FEATURES     = 5     # extract_features() 출력 차원
PPO_CLIP_EPS   = 0.2   # PPO 클리핑 ε  (STATIC-H에서 재사용)
PPO_GAE_LAMBDA = 0.95  # GAE λ (Generalized Advantage Estimation)
SAC_ALPHA_LR   = 0.01  # SAC 자동 온도 α 학습률
DDPG_TAU       = 0.005 # DDPG 타겟 네트워크 소프트 갱신 계수

# ── STATIC-H 하이브리드 전용 상수 ─────────────────────────────────────────────
HYBRID_ALPHA_MIN = 0.01   # 적응 온도 α_t 최솟값 (안정 구간 하한)
HYBRID_ALPHA_MAX = 0.15   # 적응 온도 α_t 최댓값 (급변 구간 상한)
HYBRID_FLIP_WIN  = 20     # 행동 전환율 계산 슬라이딩 윈도우 (봉)


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


# ══════════════════════════════════════════════════════════════════════════════
# STATIC-H: Tabular PPO Clipping + Adaptive Temperature (SAC-inspired)
# ══════════════════════════════════════════════════════════════════════════════

def _train_actor_critic_hybrid(returns, prices, emas, lr, gamma, epsilon,
                                train_episodes, n_days, fee_rate,
                                vols=None, vol_threshold=None):
    """STATIC-H 하이브리드 RL 훈련 — Tabular PPO Clipping + Adaptive Temperature

    STATIC Actor-Critic 기반에 두 가지 개선 적용:

    1. PPO Clipping (Tabular):
       π_old = softmax(θ[s])           — 업데이트 전 정책 (현재 step)
       θ_try = θ + lr·δ·grad           — 후보 step 계산
       r_t   = π_try(a|s) / π_old(a|s) — 정책 변화 비율
       A_t > 0 이고 r_t > 1+ε → step_scale = (1+ε)/r_t  (과도한 BUY 쏠림 억제)
       A_t < 0 이고 r_t < 1-ε → step_scale = (1-ε)/r_t  (과도한 CASH 쏠림 억제)
       → PPO_CLIP_EPS=0.2 재사용 / lr이 높아도 단일 상태 π→1 수렴 방지

    2. Adaptive Temperature α_t (SAC-inspired):
       flip_rate = 최근 HYBRID_FLIP_WIN 봉에서 행동 전환 비율 (0~1)
       α_t = clip(ENTROPY_COEFF × (1 + flip_rate), HYBRID_ALPHA_MIN, HYBRID_ALPHA_MAX)
       r_eff = r + α_t × H(π)
       → 행동이 잦게 바뀌는 불안정 구간: α_t ↑ (탐험 강화)
       → 행동이 안정적인 구간:           α_t ↓ (활용 집중, ENTROPY_COEFF 하한)

    Cold-Start 초기화·상태 인코딩·epsilon은 STATIC과 동일.
    """
    use_vol  = (vols is not None and vol_threshold is not None)
    n_states = 8 if use_vol else 4
    n_actions = 2

    # ── Cold-Start 초기화: STATIC과 동일 ────────────────────────────────────
    theta = np.zeros((n_states, n_actions))
    theta[1, 1] = max(0.05, fee_rate * 30)
    theta[2, 1] = max(0.1,  fee_rate * 50)
    theta[3, 1] = max(0.2,  fee_rate * 80)
    if use_vol:
        theta[5, 1] = max(0.05, fee_rate * 30)
        theta[6, 1] = max(0.1,  fee_rate * 50)
        theta[7, 1] = max(0.2,  fee_rate * 80)
    V = np.zeros(n_states)

    def softmax_policy(logits):
        exp_l = np.exp(np.clip(logits - np.max(logits), -30, 30))
        return exp_l / (np.sum(exp_l) + 1e-10)

    # ── 훈련 루프 ────────────────────────────────────────────────────────────
    for _ in range(train_episodes):
        _v0 = float(vols[0]) if use_vol else None
        state = _make_state_static(returns[0], prices[0], emas[0], _v0, vol_threshold)
        prev_action = 0
        action_history = []

        for t in range(1, n_days):
            probs = softmax_policy(theta[state])

            if np.random.rand() < epsilon:
                action = np.random.randint(0, n_actions)
            else:
                action = np.random.choice([0, 1], p=probs)

            # [Adaptive α_t] 최근 HYBRID_FLIP_WIN봉 행동 전환율 계산
            action_history.append(action)
            if len(action_history) > HYBRID_FLIP_WIN:
                action_history = action_history[-HYBRID_FLIP_WIN:]
            if len(action_history) > 1:
                flips = sum(1 for i in range(1, len(action_history))
                            if action_history[i] != action_history[i - 1])
                flip_rate = flips / (len(action_history) - 1)
            else:
                flip_rate = 0.0
            alpha_t = float(np.clip(
                ENTROPY_COEFF * (1.0 + flip_rate), HYBRID_ALPHA_MIN, HYBRID_ALPHA_MAX
            ))

            _fee   = fee_rate if (action == 1 and prev_action == 0) else 0.0
            reward = (returns[t] if action == 1 else 0.0) - _fee
            prev_action = action

            _vt = float(vols[t]) if use_vol else None
            next_state = _make_state_static(returns[t], prices[t], emas[t], _vt, vol_threshold)

            entropy  = -np.sum(probs * np.log(probs + 1e-10))
            td_error = (reward + alpha_t * entropy) + gamma * V[next_state] - V[state]

            V[state] += lr * td_error

            # [PPO Clipping] 후보 step의 정책 비율 검사 후 step_scale 조정
            grad      = np.array([(1.0 if a == action else 0.0) - probs[a]
                                   for a in range(n_actions)])
            theta_try = theta[state] + lr * td_error * grad
            probs_try = softmax_policy(theta_try)
            r_t       = probs_try[action] / (probs[action] + 1e-10)

            step_scale = 1.0
            if td_error > 0 and r_t > 1.0 + PPO_CLIP_EPS:
                step_scale = (1.0 + PPO_CLIP_EPS) / (r_t + 1e-10)
            elif td_error < 0 and r_t < 1.0 - PPO_CLIP_EPS:
                step_scale = (1.0 - PPO_CLIP_EPS) / (r_t + 1e-10)

            theta[state] += step_scale * lr * td_error * grad

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
                                roll_period=None, return_policy=False,
                                algorithm="STATIC"):
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
        _train_fn = (_train_actor_critic_hybrid if algorithm == "STATIC_H"
                     else _train_actor_critic_static)
        theta, _ = _train_fn(
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
            _new_theta, _ = _train_fn(
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


# ══════════════════════════════════════════════════════════════════════════════
# 신경망 RL 알고리즘 — A2C / A3C / PPO / SAC / DDPG  (NumPy 전용)
# ──────────────────────────────────────────────────────────────────────────────
# 공통 구조:
#   Actor:  TinyMLP[N_FEATURES → NN_HIDDEN → n_actions]
#   Critic: TinyMLP[N_FEATURES → NN_HIDDEN → 1]  (또는 Q용 2-output)
#   상태 표현: extract_features() → 5차원 연속 벡터
#   평가:   _eval_neural() 공통 함수 (훈련 후 전체 기간 그리디 실행)
# ══════════════════════════════════════════════════════════════════════════════

def _make_df_vals(df):
    """DataFrame에서 배열 딕셔너리 구성 (반복 호출 최소화)."""
    n = len(df)
    return {
        'returns': df['Daily_Return'].values,
        'prices':  df['Close'].values,
        'emas':    df['EMA_10'].values,
        'vols':    df['Rolling_Std'].values if 'Rolling_Std' in df.columns else np.zeros(n),
    }


def _softmax(logits):
    """수치 안정 softmax."""
    z = logits - np.max(logits)
    e = np.exp(np.clip(z, -30, 30))
    return e / (e.sum() + 1e-10)


def _eval_neural(df, actor, algorithm, fee_rate, seed):
    """신경망 RL 공통 평가 (그리디 정책, 전체 기간).

    Returns
    -------
    cumulative_return : np.ndarray (n_days,)
    action_log        : list[dict]
    actor             : 학습된 Actor 네트워크
    """
    from common.nn_utils import extract_features

    np.random.seed(seed)
    n_days   = len(df)
    df_vals  = _make_df_vals(df)
    returns  = df_vals['returns']

    cumulative_return = np.zeros(n_days)
    current_capital   = 1.0
    prev_action       = 0
    action_log        = []

    for t in range(1, n_days):
        s = extract_features(df_vals, t)
        logits, _, _ = actor.forward(s)

        if algorithm == 'DDPG':
            # 연속 포지션 → 0.5 임계값으로 이진 결정
            position = 1.0 / (1.0 + np.exp(-float(logits[0])))
            action   = 1 if position >= 0.5 else 0
        else:
            probs  = _softmax(logits)
            action = int(np.argmax(probs))

        _fee    = fee_rate if (action == 1 and prev_action == 0) else 0.0
        reward  = (returns[t] if action == 1 else 0.0) - _fee
        current_capital   *= (1 + reward)
        cumulative_return[t] = (current_capital - 1) * 100
        action_log.append({
            "Day":             t,
            "Action":          "BUY" if action == 1 else "CASH",
            "Daily_Return(%)": round(reward * 100, 4),
        })
        prev_action = action

    return cumulative_return, action_log, actor


# ── A2C ──────────────────────────────────────────────────────────────────────

def _train_a2c(df, lr, gamma, epsilon, episodes, fee_rate, seed):
    """A2C — Advantage Actor-Critic (온라인 TD, 단일 에피소드).

    [강화학습] Advantage Actor-Critic
    ──────────────────────────────────
    advantage  A_t = r_eff + γ·V(s') - V(s)    [TD 오차]
    Critic 갱신: ∇_w L_c = -A_t  (MSE: ½A_t² → 기울기 = -A_t)
    Actor  갱신: ∇_θ L_a = -A_t · ∇_θ log π(a|s)
                          = -A_t · (e_a - π)    [score function]
    엔트로피 정규화: r_eff = r + ENTROPY_COEFF · H(π)
    """
    from common.nn_utils import TinyMLP, extract_features

    np.random.seed(seed)
    n_days  = len(df)
    n_train = max(int(n_days * TRAIN_RATIO), 20)
    df_vals = _make_df_vals(df)
    returns = df_vals['returns']

    actor  = TinyMLP([N_FEATURES, NN_HIDDEN, 2], seed=seed,     lr=lr)
    critic = TinyMLP([N_FEATURES, NN_HIDDEN, 1], seed=seed + 1, lr=lr)

    for _ in range(episodes):
        prev_action = 1  # BUY 시작 (Vanilla RL과 동일 관행)
        for t in range(1, n_train):
            s      = extract_features(df_vals, t)
            s_next = extract_features(df_vals, min(t + 1, n_train - 1))

            logits, pre_a, acts_a = actor.forward(s)
            probs = _softmax(logits)

            action = (np.random.randint(0, 2)
                      if np.random.rand() < epsilon
                      else np.random.choice([0, 1], p=probs))

            _fee    = fee_rate if (action == 1 and prev_action == 0) else 0.0
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            reward  = (returns[t] if action == 1 else 0.0) - _fee + ENTROPY_COEFF * entropy
            prev_action = action

            v_s,    pre_c,  acts_c  = critic.forward(s)
            v_next, _pre_n, _acts_n = critic.forward(s_next)
            adv = reward + gamma * float(v_next[0]) - float(v_s[0])

            # Critic: 손실 = ½·adv², ∂/∂v = -adv
            critic.backward_and_update(pre_c, acts_c, np.array([-adv]), lr=lr)

            # Actor: 손실 = -adv·log π(a)  →  기울기 w.r.t. logits = -adv·(e_a - π)
            score        = np.zeros(2)
            score[action] = 1.0
            score        -= probs
            actor.backward_and_update(pre_a, acts_a, -adv * score, lr=lr)

    return actor, critic


# ── A3C ──────────────────────────────────────────────────────────────────────

def _train_a3c(df, lr, gamma, epsilon, episodes, fee_rate, seed, n_steps=5):
    """A3C — Asynchronous Advantage Actor-Critic (단일 스레드, n-step 리턴).

    [강화학습] n-step A3C (단일 스레드 근사)
    ────────────────────────────────────────
    n-step 리턴: R_t = r_t + γ·r_{t+1} + ... + γ^{n-1}·r_{t+n-1} + γ^n·V(s_{t+n})
    advantage  : A_t = R_t - V(s_t)
    동일한 Actor/Critic 갱신 수식 (A2C와 동일, n-step 리턴으로 분산 감소)
    n_steps = 5 (기본)
    """
    from common.nn_utils import TinyMLP, extract_features

    np.random.seed(seed)
    n_days  = len(df)
    n_train = max(int(n_days * TRAIN_RATIO), 20)
    df_vals = _make_df_vals(df)
    returns = df_vals['returns']

    actor  = TinyMLP([N_FEATURES, NN_HIDDEN, 2], seed=seed,     lr=lr)
    critic = TinyMLP([N_FEATURES, NN_HIDDEN, 1], seed=seed + 1, lr=lr)

    for _ in range(episodes):
        prev_action = 1
        t = 1
        while t < n_train:
            # n-step 전이 수집
            transitions = []
            for step in range(n_steps):
                if t + step >= n_train:
                    break
                s = extract_features(df_vals, t + step)

                logits, _, _ = actor.forward(s)
                probs = _softmax(logits)
                action = (np.random.randint(0, 2)
                          if np.random.rand() < epsilon
                          else np.random.choice([0, 1], p=probs))

                _fee    = fee_rate if (action == 1 and prev_action == 0) else 0.0
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                reward  = (returns[t + step] if action == 1 else 0.0) - _fee + ENTROPY_COEFF * entropy
                prev_action = action
                transitions.append((s, action, reward))

            if not transitions:
                break

            # 부트스트랩: 마지막 상태 이후 V
            # R_init = V(s_{t+n}), 역방향 루프에서 R = r_{t+k} + γ·R 적용
            # → 최종 R_t = r_t + γr_{t+1} + ... + γ^{n-1}r_{t+n-1} + γ^n·V(s_{t+n}) ✓
            t_end = t + len(transitions) - 1
            if t_end + 1 < n_train:
                s_final = extract_features(df_vals, t_end + 1)
                v_final, _, _ = critic.forward(s_final)
                R = float(v_final[0])   # γ 미적용: 루프 내 첫 R = r_{t+n-1} + γ·V(s_{t+n})
            else:
                R = 0.0

            # 역방향으로 n-step 리턴 갱신
            for s, action, reward in reversed(transitions):
                R = reward + gamma * R
                # 최신 네트워크로 재순전파 (stale 기울기 방지)
                logits_f, pre_a, acts_a = actor.forward(s)
                v_s_f,    pre_c, acts_c = critic.forward(s)
                adv = R - float(v_s_f[0])

                critic.backward_and_update(pre_c, acts_c, np.array([-adv]), lr=lr)

                probs_f      = _softmax(logits_f)
                score        = np.zeros(2)
                score[action] = 1.0
                score        -= probs_f
                actor.backward_and_update(pre_a, acts_a, -adv * score, lr=lr)

            t += len(transitions)

    return actor, critic


# ── PPO ──────────────────────────────────────────────────────────────────────

def _train_ppo(df, lr, gamma, epsilon, episodes, fee_rate, seed,
               clip_eps=PPO_CLIP_EPS, gae_lambda=PPO_GAE_LAMBDA, n_epochs=4):
    """PPO — Proximal Policy Optimization (클리핑 대리 목적).

    [강화학습] PPO (Clipped Surrogate Objective)
    ──────────────────────────────────────────────
    r_t(θ)  = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
    L_CLIP  = E[min(r_t·Â_t, clip(r_t, 1-ε, 1+ε)·Â_t)]
    GAE Â_t = Σ_{k≥0} (γλ)^k·δ_{t+k}  where δ_t = r_t + γV(s_{t+1}) - V(s_t)
    n_epochs = 4 (미니배치 반복 갱신)
    """
    from common.nn_utils import TinyMLP, extract_features

    np.random.seed(seed)
    n_days      = len(df)
    n_train     = max(int(n_days * TRAIN_RATIO), 20)
    rollout_len = min(64, n_train - 1)
    df_vals     = _make_df_vals(df)
    returns     = df_vals['returns']

    actor  = TinyMLP([N_FEATURES, NN_HIDDEN, 2], seed=seed,     lr=lr)
    critic = TinyMLP([N_FEATURES, NN_HIDDEN, 1], seed=seed + 1, lr=lr)

    for ep in range(episodes):
        # ── 롤아웃 수집 ──
        states_r          = []
        actions_r         = []
        rewards_r         = []
        old_probs_action  = []
        values_r          = []

        prev_action = 1
        for t in range(1, min(rollout_len + 1, n_train)):
            s = extract_features(df_vals, t)

            logits, _, _ = actor.forward(s)
            probs = _softmax(logits)
            action = (np.random.randint(0, 2)
                      if np.random.rand() < epsilon
                      else np.random.choice([0, 1], p=probs))

            _fee    = fee_rate if (action == 1 and prev_action == 0) else 0.0
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            reward  = (returns[t] if action == 1 else 0.0) - _fee + ENTROPY_COEFF * entropy
            prev_action = action

            v_s, _, _ = critic.forward(s)

            states_r.append(s)
            actions_r.append(action)
            rewards_r.append(reward)
            old_probs_action.append(float(probs[action]))
            values_r.append(float(v_s[0]))

        T = len(states_r)
        if T == 0:
            continue

        # ── GAE 어드밴티지 ──
        advantages = np.zeros(T)
        last_adv   = 0.0
        for t in reversed(range(T)):
            if t + 1 < T:
                v_next = values_r[t + 1]
            else:
                s_boot  = extract_features(df_vals, min(rollout_len + 1, n_train - 1))
                vb, _, _ = critic.forward(s_boot)
                v_next  = float(vb[0])
            delta      = rewards_r[t] + gamma * v_next - values_r[t]
            last_adv   = delta + gamma * gae_lambda * last_adv
            advantages[t] = last_adv

        returns_tgt = advantages + np.array(values_r)

        # ── n_epochs 미니배치 갱신 ──
        for _ in range(n_epochs):
            idx = np.random.permutation(T)
            for i in idx:
                s     = states_r[i]
                a     = actions_r[i]
                adv   = float(advantages[i])
                ret_t = float(returns_tgt[i])
                old_p = float(old_probs_action[i])

                logits, pre_a, acts_a = actor.forward(s)
                probs = _softmax(logits)
                new_p = float(probs[a])
                ratio = new_p / (old_p + 1e-10)

                # 클리핑 여부에 따른 Actor 기울기
                clipped = np.clip(ratio, 1 - clip_eps, 1 + clip_eps)
                if (adv >= 0 and ratio < 1 + clip_eps) or (adv < 0 and ratio > 1 - clip_eps):
                    # 미클리핑: ∂r_t/∂logit_i = ratio·(e_a-π)  →  ∂(-L_CLIP)/∂logits = -adv·ratio·(e_a-π)
                    score        = np.zeros(2)
                    score[a]     = 1.0
                    score       -= probs
                    grad_actor   = -adv * ratio * score
                else:
                    grad_actor = np.zeros(2)   # 클리핑 → 기울기 0

                actor.backward_and_update(pre_a, acts_a, grad_actor, lr=lr)

                # Critic: MSE 손실
                v_s, pre_c, acts_c = critic.forward(s)
                critic_err = float(v_s[0]) - ret_t
                critic.backward_and_update(pre_c, acts_c, np.array([critic_err]), lr=lr)

    return actor, critic


# ── SAC ──────────────────────────────────────────────────────────────────────

def _train_sac(df, lr, gamma, epsilon, episodes, fee_rate, seed,
               buffer_size=2000, batch_size=32, target_update_freq=10):
    """SAC — Soft Actor-Critic (이산 행동, 자동 온도 α).

    [강화학습] SAC-Discrete (Haarnoja et al., 2018)
    ─────────────────────────────────────────────────
    소프트 V:    V(s) = Σ_a π(a|s)·[Q(s,a) - α·log π(a|s)]
    Q 타겟:      y    = r + γ·V(s')
    Critic 갱신: Q(s,a) ← min-MSE with y  (단일 Q 사용)
    Actor  갱신: max_π Σ_a π(a|s)·[Q(s,a) - α·log π(a|s)]
                 ∂L_a/∂z_i = π_i·(v_soft - q_adj_i)   where q_adj = Q - α·log π
    자동 α:      J(α) = -α·(log π(a|s) + H_target)
    """
    from common.nn_utils import TinyMLP, ReplayBuffer, extract_features

    np.random.seed(seed)
    n_days  = len(df)
    n_train = max(int(n_days * TRAIN_RATIO), 20)
    df_vals = _make_df_vals(df)
    returns = df_vals['returns']

    actor      = TinyMLP([N_FEATURES, NN_HIDDEN, 2], seed=seed,     lr=lr)
    critic     = TinyMLP([N_FEATURES, NN_HIDDEN, 2], seed=seed + 1, lr=lr)
    critic_tgt = critic.copy()

    log_alpha = np.array([0.0])
    H_target  = -np.log(0.5) * 0.5   # ≈ 0.35 — 목표 엔트로피
    alpha     = float(np.exp(log_alpha[0]))

    buffer = ReplayBuffer(buffer_size, N_FEATURES)

    for _ep in range(episodes):
        prev_action = 1
        for t in range(1, n_train):
            s      = extract_features(df_vals, t)
            s_next = extract_features(df_vals, min(t + 1, n_train - 1))

            # ε-greedy 탐색
            if len(buffer) < batch_size or np.random.rand() < epsilon:
                action = np.random.randint(0, 2)
            else:
                logits, _, _ = actor.forward(s)
                action = int(np.argmax(_softmax(logits)))

            _fee    = fee_rate if (action == 1 and prev_action == 0) else 0.0
            reward  = (returns[t] if action == 1 else 0.0) - _fee
            prev_action = action
            buffer.push(s, action, reward, s_next, int(t == n_train - 1))

            if len(buffer) < batch_size:
                continue

            s_b, a_b, r_b, ns_b, d_b = buffer.sample(batch_size)

            for j in range(batch_size):
                s_j  = s_b[j]
                a_j  = int(a_b[j])
                r_j  = float(r_b[j])
                ns_j = ns_b[j]

                # 소프트 V(s') — 타겟 Q + 현재 정책
                logits_next, _, _ = actor.forward(ns_j)
                probs_next        = _softmax(logits_next)
                q_next, _, _      = critic_tgt.forward(ns_j)
                v_soft_next = float(np.sum(
                    probs_next * (q_next - alpha * np.log(probs_next + 1e-10))
                ))
                y = r_j + gamma * (1.0 - float(d_b[j])) * v_soft_next

                # Critic 갱신 (MSE)
                q_all, pre_c, acts_c = critic.forward(s_j)
                td_err       = float(q_all[a_j]) - y
                grad_q       = np.zeros(2)
                grad_q[a_j]  = td_err
                critic.backward_and_update(pre_c, acts_c, grad_q, lr=lr)

                # Actor 갱신: ∂(-L_a)/∂z_i = π_i·(v_soft - q_adj_i)
                logits_cur, pre_a, acts_a = actor.forward(s_j)
                probs_cur   = _softmax(logits_cur)
                q_all_cur, _, _ = critic.forward(s_j)
                q_adj       = q_all_cur - alpha * np.log(probs_cur + 1e-10)
                v_soft_cur  = float(np.sum(probs_cur * q_adj))
                grad_actor  = probs_cur * (v_soft_cur - q_adj)
                actor.backward_and_update(pre_a, acts_a, grad_actor, lr=lr)

                # 자동 α 갱신
                log_pi_a  = float(np.log(probs_cur[a_j] + 1e-10))
                alpha_grad = -(log_pi_a + H_target)
                log_alpha[0] -= SAC_ALPHA_LR * alpha_grad
                log_alpha[0]  = float(np.clip(log_alpha[0], -5, 2))
                alpha         = float(np.exp(log_alpha[0]))

            if t % target_update_freq == 0:
                critic_tgt.soft_update_from(critic, tau=DDPG_TAU)

    return actor, critic


# ── DDPG ─────────────────────────────────────────────────────────────────────

def _train_ddpg(df, lr, gamma, epsilon, episodes, fee_rate, seed,
                buffer_size=2000, batch_size=32, update_freq=4):
    """DDPG — Deep Deterministic Policy Gradient (연속 포지션 [0,1]).

    [강화학습] DDPG (Lillicrap et al., 2015)
    ─────────────────────────────────────────
    Actor:    μ(s) ∈ [0,1] (sigmoid 출력) — 연속 포지션 비율
    Critic:   Q(s, a) where a = μ(s)  [입력 = s ‖ a, 차원 N_FEATURES+1]
    Q 타겟:   y = r + γ·Q_tgt(s', μ_tgt(s'))
    Critic 갱신: MSE(Q(s,a), y)
    Actor  갱신: ∇_θ J = ∇_a Q(s,a)|_{a=μ(s)} · ∇_θ μ(s)
                 (체인 규칙: critic 역전파로 ∂Q/∂a 획득)
    탐색:    OU noise (θ=0.15, σ=0.2) × epsilon
    타겟망:  θ_tgt ← τ·θ + (1-τ)·θ_tgt  (소프트 갱신)
    """
    from common.nn_utils import TinyMLP, ReplayBuffer, extract_features

    np.random.seed(seed)
    n_days  = len(df)
    n_train = max(int(n_days * TRAIN_RATIO), 20)
    df_vals = _make_df_vals(df)
    returns = df_vals['returns']

    actor      = TinyMLP([N_FEATURES,     NN_HIDDEN, 1],         seed=seed,     lr=lr)
    actor_tgt  = actor.copy()
    critic     = TinyMLP([N_FEATURES + 1, NN_HIDDEN, 1],         seed=seed + 1, lr=lr)
    critic_tgt = critic.copy()

    buffer   = ReplayBuffer(buffer_size, N_FEATURES)
    ou_theta = 0.15
    ou_sigma = 0.2

    for _ep in range(episodes):
        ou_state = np.zeros(1)  # OU noise — 에피소드마다 초기화
        prev_pos = 0.5
        for t in range(1, n_train):
            s      = extract_features(df_vals, t)
            s_next = extract_features(df_vals, min(t + 1, n_train - 1))

            # OU noise 탐색
            ou_state += -ou_theta * ou_state + ou_sigma * np.random.randn(1)
            logit_a, _, _ = actor.forward(s)
            mu      = 1.0 / (1.0 + np.exp(-float(logit_a[0])))
            pos     = float(np.clip(mu + epsilon * float(ou_state[0]), 0.0, 1.0))

            # 연속 포지션 보상: position × return - fee·|Δposition|
            _fee   = fee_rate * abs(pos - prev_pos)
            reward = returns[t] * pos - _fee
            prev_pos = pos
            buffer.push(s, pos, reward, s_next, int(t == n_train - 1))

            if len(buffer) < batch_size:
                continue
            if t % update_freq != 0:   # update_freq 스텝마다 1회 갱신 (속도 최적화)
                continue

            s_b, a_b, r_b, ns_b, d_b = buffer.sample(batch_size)

            for j in range(batch_size):
                s_j  = s_b[j]
                a_j  = float(a_b[j])
                r_j  = float(r_b[j])
                ns_j = ns_b[j]

                # 타겟 Q
                logit_next, _, _ = actor_tgt.forward(ns_j)
                mu_next = 1.0 / (1.0 + np.exp(-float(logit_next[0])))
                sa_next = np.append(ns_j, mu_next)
                q_next, _, _ = critic_tgt.forward(sa_next)
                y = r_j + gamma * (1.0 - float(d_b[j])) * float(q_next[0])

                # Critic 갱신
                sa      = np.append(s_j, a_j)
                q_val, pre_c, acts_c = critic.forward(sa)
                critic_err = float(q_val[0]) - y
                critic.backward_and_update(pre_c, acts_c, np.array([critic_err]), lr=lr)

                # Actor 갱신: ∇_θ J = (∂Q/∂a) · (∂sigmoid/∂logit) — 체인 규칙
                logit_cur, pre_a, acts_a = actor.forward(s_j)
                mu_cur  = 1.0 / (1.0 + np.exp(-float(logit_cur[0])))
                sa_cur  = np.append(s_j, mu_cur)
                _q_cur, pre_c2, acts_c2 = critic.forward(sa_cur)
                # critic 입력에 대한 기울기 (가중치 갱신 없이)
                grad_sa    = critic.get_grad_input(pre_c2, acts_c2, np.array([-1.0]))
                dQ_dmu     = -float(grad_sa[-1])          # ∂Q/∂μ
                dmu_dlogit = mu_cur * (1.0 - mu_cur)      # sigmoid 미분
                actor.backward_and_update(pre_a, acts_a, np.array([-dQ_dmu * dmu_dlogit]), lr=lr)

            # 소프트 갱신 (매 스텝)
            actor_tgt.soft_update_from(actor,   tau=DDPG_TAU)
            critic_tgt.soft_update_from(critic, tau=DDPG_TAU)

    return actor, critic


# ── ACER ─────────────────────────────────────────────────────────────────────

def _train_acer(df, lr, gamma, epsilon, episodes, fee_rate, seed,
                buffer_size=2000, c_clip=10.0, retrace_lambda=0.95):
    """ACER — Actor-Critic with Experience Replay (Wang et al., 2016).

    [강화학습] Retrace(λ) + Truncated Importance Sampling
    ──────────────────────────────────────────────────────
    행동 정책 μ: ε-greedy (on-policy 롤아웃)
    IS 비율    : ρ_t = π(a_t|s_t) / μ(a_t|s_t)
    절단 IS    : c_t = min(c̄, ρ_t) · λ

    Retrace(λ) 역방향 (t=T,...,0):
      Q^ret_T = r_T + γ·V(s_{T+1})
      Q^ret_t = r_t + γ·V(s_{t+1}) + γ·c_{t+1}·(Q^ret_{t+1} − Q(s_{t+1}, a_{t+1}))

    Actor gradient (주항 + 보정항):
      주항:   −min(c̄, ρ_t) · (Q^ret_t − V(s_t)) · (e_{a_t} − π_t)
      보정항: −Σ_a max(0, 1 − c̄/ρ(a|s_t)) · π(a|s_t) · (Q(s_t,a) − V(s_t)) · (e_a − π_t)

    2-action 특성 활용: μ(1−a|s) = 1 − μ(a|s) → 보정항을 한 번의 추가 계산으로 처리
    Trust Region: 생략 (NumPy 구현 범위 내 단순화)
    """
    from common.nn_utils import TinyMLP, ReplayBuffer, extract_features

    np.random.seed(seed)
    n_days  = len(df)
    n_train = max(int(n_days * TRAIN_RATIO), 20)
    df_vals = _make_df_vals(df)
    returns = df_vals['returns']

    actor = TinyMLP([N_FEATURES, NN_HIDDEN, 2], seed=seed,     lr=lr)   # π(a|s)
    q_net = TinyMLP([N_FEATURES, NN_HIDDEN, 2], seed=seed + 1, lr=lr)   # Q(s,·)

    buffer = ReplayBuffer(buffer_size, N_FEATURES)

    def _get_V(s_vec):
        """V(s) = Σ_a π(a|s)·Q(s,a)"""
        q_all, _, _ = q_net.forward(s_vec)
        lg, _, _    = actor.forward(s_vec)
        return float(np.sum(_softmax(lg) * q_all))

    for _ in range(episodes):
        # ── On-policy 롤아웃 수집 ─────────────────────────────────────────────
        traj = []          # (s, a, r, s_next, done, mu_a)
        prev_action = 1

        for t in range(1, n_train):
            s      = extract_features(df_vals, t)
            s_next = extract_features(df_vals, min(t + 1, n_train - 1))

            logits, _, _ = actor.forward(s)
            pi = _softmax(logits)

            # ε-greedy 행동 정책 μ
            # μ(a|s) = ε/2 + (1-ε)·π(a|s)  (2-action ε-greedy 정확한 확률)
            if np.random.rand() < epsilon:
                action = np.random.randint(0, 2)
            else:
                action = np.random.choice([0, 1], p=pi)
            mu_a = epsilon * 0.5 + (1.0 - epsilon) * float(pi[action])

            _fee = fee_rate if (action == 1 and prev_action == 0) else 0.0
            ent  = -np.sum(pi * np.log(pi + 1e-10))
            r    = (returns[t] if action == 1 else 0.0) - _fee + ENTROPY_COEFF * ent
            prev_action = action

            done = int(t == n_train - 1)
            traj.append((s, action, r, s_next, done, mu_a))
            buffer.push(s, action, r, s_next, done, float(np.log(mu_a + 1e-10)))

        if len(traj) < 2:
            continue

        # ── Retrace(λ) Q 타겟 역방향 계산 ────────────────────────────────────
        n      = len(traj)
        q_rets = np.zeros(n)
        R      = _get_V(traj[-1][3]) if not traj[-1][4] else 0.0   # 부트스트랩

        for i in reversed(range(n)):
            s_i, a_i, r_i, s_next_i, done_i, mu_a_i = traj[i]
            V_next = _get_V(s_next_i) if not done_i else 0.0

            if i == n - 1:
                q_rets[i] = r_i + gamma * V_next
            else:
                # c_{i+1} = λ·min(c̄, ρ_{i+1}),  ρ_{i+1} = π(a_{i+1}|s_{i+1}) / μ(a_{i+1})
                s_i1, a_i1, _, _, _, mu_a_i1 = traj[i + 1]
                lg_i1, _, _  = actor.forward(s_i1)
                pi_i1        = _softmax(lg_i1)
                rho_i1       = float(pi_i1[a_i1]) / max(mu_a_i1, 1e-10)
                c_i1         = retrace_lambda * min(c_clip, rho_i1)

                q_all_i1, _, _ = q_net.forward(s_i1)
                Q_i1_a         = float(q_all_i1[a_i1])

                q_rets[i] = r_i + gamma * V_next + gamma * c_i1 * (R - Q_i1_a)

            R = q_rets[i]

        # ── Actor & Q-network 업데이트 ────────────────────────────────────────
        for i in range(n):
            s_i, a_i, _, _, _, mu_a_i = traj[i]
            q_ret = q_rets[i]

            logits_i, pre_a, acts_a = actor.forward(s_i)
            pi_i                    = _softmax(logits_i)
            q_all_i, pre_q, acts_q  = q_net.forward(s_i)
            V_i = float(np.sum(pi_i * q_all_i))

            # Q-network 갱신 (MSE: 선택 행동만)
            q_err       = np.zeros(2)
            q_err[a_i]  = float(q_all_i[a_i]) - q_ret
            q_net.backward_and_update(pre_q, acts_q, q_err, lr=lr)

            # IS 비율 (선택 행동)
            rho_i    = float(pi_i[a_i]) / max(mu_a_i, 1e-10)
            rho_clip = min(c_clip, rho_i)

            # 주항: −min(c̄, ρ)·(Q^ret − V)·(e_a − π)
            score_a        = np.zeros(2)
            score_a[a_i]   = 1.0
            score_a       -= pi_i
            grad_actor     = -rho_clip * (q_ret - V_i) * score_a

            # 보정항: 비선택 행동 (2-action: μ(1−a|s) = 1 − μ(a|s))
            a_other        = 1 - a_i
            mu_other       = max(1.0 - mu_a_i, 1e-10)
            rho_other      = float(pi_i[a_other]) / mu_other
            coeff_other    = max(0.0, 1.0 - c_clip / max(rho_other, 1e-10))
            score_other    = np.zeros(2)
            score_other[a_other] = 1.0
            score_other   -= pi_i
            grad_actor    -= (coeff_other * float(pi_i[a_other])
                              * (float(q_all_i[a_other]) - V_i) * score_other)

            actor.backward_and_update(pre_a, acts_a, grad_actor, lr=lr)

    return actor, q_net


# ══════════════════════════════════════════════════════════════════════════════
# 공개 API: run_neural_rl
# ══════════════════════════════════════════════════════════════════════════════

def run_neural_rl(df, lr=0.01, gamma=0.98, epsilon=0.10, episodes=100,
                  algorithm="A2C", seed=42, fee_rate=0.0):
    """신경망 RL 디스패처 — 알고리즘 선택 후 훈련+평가 실행.

    Parameters
    ----------
    df        : fetch_stock_data() 반환 DataFrame (Close, EMA_10, Daily_Return, Rolling_Std)
    lr        : 학습률
    gamma     : 할인율
    epsilon   : 탐색률 (ε-greedy / OU noise 스케일)
    episodes  : 훈련 에피소드 수
    algorithm : 'A2C' | 'A3C' | 'PPO' | 'ACER' | 'SAC' | 'DDPG'
    seed      : 난수 시드
    fee_rate  : 매매 수수료율

    Returns
    -------
    cumulative_return : np.ndarray (n_days,) — 누적 수익률 (%)
    action_log        : list[dict]           — 일별 행동 로그
    actor             : 학습된 Actor TinyMLP  — (Explainable RL / 시각화용)
    """
    if algorithm == "A2C":
        actor, _ = _train_a2c(df, lr, gamma, epsilon, episodes, fee_rate, seed)
    elif algorithm == "A3C":
        actor, _ = _train_a3c(df, lr, gamma, epsilon, episodes, fee_rate, seed)
    elif algorithm == "PPO":
        actor, _ = _train_ppo(df, lr, gamma, epsilon, episodes, fee_rate, seed)
    elif algorithm == "ACER":
        actor, _ = _train_acer(df, lr, gamma, epsilon, episodes, fee_rate, seed)
    elif algorithm == "SAC":
        actor, _ = _train_sac(df, lr, gamma, epsilon, episodes, fee_rate, seed)
    elif algorithm == "DDPG":
        actor, _ = _train_ddpg(df, lr, gamma, epsilon, episodes, fee_rate, seed)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm!r}. "
                         f"Choose from 'A2C', 'A3C', 'PPO', 'ACER', 'SAC', 'DDPG'.")

    return _eval_neural(df, actor, algorithm, fee_rate, seed)
