# Chainers Master Fund — Claude 작업 지침

## 파일 구조

| 파일 | 줄 수 | 역할 |
|------|-------|------|
| `app.py` | ~2310 | Streamlit UI 전체 + `get_rl_data()` 정의 (수정 빈도 높음) |
| `common/base_agent.py` | 374 | RL 훈련·평가 엔진 (STATIC + Vanilla) |
| `common/heuristic.py` | 405 | PGActorCriticOptimizer |
| `common/evaluator.py` | 57 | softmax 비중, MDD, CTPT |
| `common/data_loader.py` | 67 | yfinance fetch + EMA_10, Rolling_Std |
| `members/member_N/config.py` | — | MEMBER_NAME, TARGET_INDICES, RL_PARAMS |

---

## get_rl_data 시그니처 (9-tuple 반환, 모든 호출에서 9개 언패킹 필수)

```python
get_rl_data(ticker, lr, gamma, epsilon, n_bars, train_episodes, seed,
            v_epsilon=None, fee_rate=0.0, interval="1d",
            use_vol=False, roll_period=None)
→ (df, v_trace, s_trace, real_ret_trace, s_mdd, v_log, s_log, s_theta, v_qtable)
```

---

## RL 알고리즘 상수 (base_agent.py 상단)

```python
TRAIN_RATIO       = 0.7    # 앞 70% 학습 / 30% OOS
ENTROPY_COEFF     = 0.05   # r_eff = r + ENTROPY_COEFF × H(π)  ← 낮추면 Buy&Hold 고착
Q_FLOOR_MARGIN    = 0.005  # Q[s,BUY] ≥ Q[s,CASH] + margin
EMA_SIGNAL_WEIGHT = 2      # state = is_bull×1 + is_above_ema×2
```

## 수정 금지 제약

- STATIC: theta Cold-Start `[1,1]=max(0.05,fee×30)` / `[2,1]=max(0.1,fee×50)` / `[3,1]=max(0.2,fee×80)`, epsilon annealing 없음
- Vanilla: Q init `q[:,1]=max(fee×50,0.05)`, annealing `2ε→ε`, prev_action=1 고정
- Composite Gap = `0.6×(STATIC-Market) + 0.4×(STATIC-max(Vanilla, Market×0.3))`

---

## 상태 인코딩

```
4상태(use_vol=False): state = is_bull + is_above_ema×2  → {0,1,2,3}
8상태(use_vol=True):  state += is_high_vol×4            → {0..7}
vol_threshold=None → 훈련 구간 중위수 자동 산출
```

---

## config.py RL_PARAMS 구조 (9 필드)

```python
RL_PARAMS = {
    TARGET_INDICES[0]: {
        "lr": float, "gamma": float, "epsilon": float, "v_epsilon": float,
        "episodes": int, "train_episodes": int, "seed": int,
        "use_vol": bool, "roll_period": int|None
    },
    "default": { ... }  # fallback
}
```

**오버라이드 우선순위**: 전역 session_state > per-member config

---

## 멤버별 최적 파라미터 (5-2 기준)

| 멤버 | 종목 | lr | gamma | ε(S) | ε(V) | seed | use_vol | roll_period |
|------|------|-----|-------|------|------|------|---------|-------------|
| M1 | SPY | 0.0496 | 0.8863 | 0.1190 | 0.0993 | 42 | False | None |
| M2 | QQQ | 0.0650 | 0.9075 | 0.1005 | 0.1043 | 137 | False | None |
| M3 | KOSPI | 0.0227 | 0.9569 | 0.1386 | 0.1762 | 2024 | False | None |
| M4 | KOSDAQ | 0.0168 | 0.9084 | 0.0863 | 0.1157 | 777 | False | None |
| M5 | NVDA | 0.0497 | 0.9183 | 0.0443 | 0.1055 | 314 | **True** | **15** |
| M6 | TSLA | 0.0364 | 0.8873 | 0.1283 | 0.0842 | 99 | **True** | **20** |

> KOSPI/KOSDAQ: OOS 구조적 한계로 α < Market 정상.

---

## session_state 주요 키

```
sim_result       {hist_key: best_params_dict}
ghost_data       {hist_key: {v_trace, s_trace, params, gap}}
policy_cache     {hist_key: {theta, q_table, n_states}}
stock_trial_history  {hist_key: [trial_dict, ...]}
member_traces    {m_name: {s_trace, dates, stocks}}
fund_temperature float  (기본 1.0, 1.0~5.0)
fund_max_weight  float  (기본 1.0, 0.1~1.0)
use_vol_feature  bool   (전역 8상태 강제)
roll_period_active bool / roll_period_val int
interrupt_requested bool
```

---

## Simulation 루프 핵심 수치

```python
n_iters = max(eff_sim_min, l_auto_runs * eff_sim_mult)
_n_eval = min(4, max(3, l_auto_runs // 2))   # 최소 3 시드 보장
_eval_seeds = [base_seed + j*37 for j in range(_n_eval)]
```

## 알파 판정

| Gap (STATIC vs Market) | 판정 |
|------------------------|------|
| ≥ 1%p | ✅ 목표 달성 |
| ≥ 5%p | 우수 |
| ≥ 25%p | 최고 🏆 |
