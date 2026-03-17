# Chainers Master Fund — Claude 작업 지침

## 파일 구조

| 파일 | 역할 |
|------|------|
| `app.py` | Streamlit UI + `get_rl_data()` 정의 |
| `common/base_agent.py` | RL 훈련·평가 엔진 (STATIC / Vanilla / Neural) |
| `common/nn_utils.py` | TinyMLP, ReplayBuffer, extract_features |
| `common/heuristic.py` | PGActorCriticOptimizer |
| `common/evaluator.py` | softmax 비중, MDD, CTPT |
| `common/data_loader.py` | yfinance fetch + EMA_10, Rolling_Std |
| `common/stock_registry.py` | STOCK_REGISTRY + FEE_REGISTRY |
| `members/member_N/config.py` | MEMBER_NAME, TARGET_INDICES, RL_PARAMS |

---

## get_rl_data 시그니처 (9-tuple 반환, 모든 호출에서 9개 언패킹 필수)

```python
get_rl_data(ticker, lr, gamma, epsilon, n_bars, train_episodes, seed,
            v_epsilon=None, fee_rate=0.0, interval="1d",
            use_vol=False, roll_period=None,
            algorithm="STATIC")
→ (df, v_trace, s_trace, real_ret_trace, s_mdd, v_log, s_log, s_theta, v_qtable)
```

- `algorithm="STATIC"` → 기존 tabular 경로
- 그 외(`A2C`/`A3C`/`PPO`/`SAC`/`DDPG`) → `run_neural_rl()` 호출, `s_theta=None`

---

## RL 알고리즘 상수 (base_agent.py 상단)

```python
TRAIN_RATIO       = 0.7    # 앞 70% 학습 / 30% OOS
ENTROPY_COEFF     = 0.05   # r_eff = r + ENTROPY_COEFF × H(π)  ← 낮추면 Buy&Hold 고착
Q_FLOOR_MARGIN    = 0.005  # Q[s,BUY] ≥ Q[s,CASH] + margin
EMA_SIGNAL_WEIGHT = 2      # state = is_bull×1 + is_above_ema×2
# 신경망 공통
NN_HIDDEN = 32   N_FEATURES = 5
PPO_CLIP_EPS = 0.2   PPO_GAE_LAMBDA = 0.95
SAC_ALPHA_LR = 0.01   DDPG_TAU = 0.005
```

---

## 수정 금지 제약

- **STATIC**: theta Cold-Start `[1,1]=max(0.05,fee×30)` / `[2,1]=max(0.1,fee×50)` / `[3,1]=max(0.2,fee×80)`, epsilon annealing 없음
- **Vanilla**: Q init `q[:,1]=max(fee×50,0.05)`, annealing `2ε→ε`, prev_action=1 고정, 훈련 후 `Q[s,1]≥Q[s,0]+0.005` (전체 상태)
- **Composite Gap** = `0.6×(STATIC-Market) + 0.4×(STATIC-max(Vanilla, Market×0.3))`

---

## 상태 인코딩

```
4상태(use_vol=False): state = is_bull×1 + is_above_ema×2  → {0,1,2,3}
8상태(use_vol=True):  state += is_high_vol×4              → {0..7}
vol_threshold=None → 훈련 구간 중위수 자동 산출
```

---

## config.py RL_PARAMS 구조 (10 필드)

```python
RL_PARAMS = {
    TARGET_INDICES[0]: {
        "lr": float, "gamma": float, "epsilon": float, "v_epsilon": float,
        "episodes": int, "train_episodes": int, "seed": int,
        "use_vol": bool, "roll_period": int|None,
        "algorithm": "STATIC"  # "STATIC"|"A2C"|"A3C"|"PPO"|"SAC"|"DDPG"
    },
    "default": { ... }  # fallback
}
```

**오버라이드 우선순위**: 전역 session_state > per-member config
