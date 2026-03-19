# Chainers Master Fund — Claude 작업 지침

## 파일 구조

| 파일 | 역할 |
|------|------|
| `app.py` | Streamlit UI + `get_rl_data()` (~2500줄) |
| `common/base_agent.py` | RL 훈련·평가 (STATIC/STATIC_H/Vanilla/Neural) |
| `common/nn_utils.py` | TinyMLP, ReplayBuffer, extract_features |
| `common/heuristic.py` | PGActorCriticOptimizer (Cosine σ_max 상한) |
| `common/evaluator.py` | softmax 비중, MDD, CTPT |
| `common/data_loader.py` | yfinance fetch + EMA_10, Rolling_Std |
| `common/stock_registry.py` | STOCK_REGISTRY(12종) + FEE_REGISTRY |
| `members/member_N/config.py` | MEMBER_NAME, TARGET_INDICES, RL_PARAMS |
| `members/member_N/custom_logic.py` | 멤버별 커스텀 로직 (확장용, 현재 미사용) |
| `.streamlit/config.toml` | Streamlit 서버 설정 (업로드 10MB, fastReruns ON) |
| `requirements.txt` | Python 패키지 의존성 |
| `runtime.txt` | Python 버전 지정 (Streamlit Cloud 배포용) |
| `.devcontainer/devcontainer.json` | GitHub Codespaces 개발 환경 설정 |

## get_rl_data 시그니처 (9-tuple, 모든 호출에서 9개 언패킹 필수)

```python
get_rl_data(ticker, lr, gamma, epsilon, n_bars, train_episodes, seed,
            v_epsilon=None, fee_rate=0.0, interval="1d",
            use_vol=False, roll_period=None, algorithm="STATIC")
→ (df, v_trace, s_trace, real_ret_trace, s_mdd, v_log, s_log, s_theta, v_qtable)
```

- `algorithm not in ("STATIC","STATIC_H")` → `run_neural_rl()` 경로, `s_theta=None`

## RL 상수 (base_agent.py 상단 — 직접 수정 금지)

```python
TRAIN_RATIO=0.7  ENTROPY_COEFF=0.05  Q_FLOOR_MARGIN=0.005  EMA_SIGNAL_WEIGHT=2
NN_HIDDEN=32  N_FEATURES=5  PPO_CLIP_EPS=0.2  PPO_GAE_LAMBDA=0.95
SAC_ALPHA_LR=0.01  DDPG_TAU=0.005
HYBRID_ALPHA_MIN=0.01  HYBRID_ALPHA_MAX=0.15  HYBRID_FLIP_WIN=20
```

## 수정 금지 제약

- **STATIC**: theta Cold-Start `[1,1]=max(0.05,fee×30)` / `[2,1]=max(0.1,fee×50)` / `[3,1]=max(0.2,fee×80)`, epsilon annealing 없음
- **STATIC_H**: STATIC과 동일 Cold-Start 초기화. Tabular PPO Clip(`step_scale=0.3`, `clip_eps=PPO_CLIP_EPS=0.2`). Adaptive Temperature `α_t=clip(0.05×(1+flip_rate), 0.01, 0.15)`, `flip_rate` 윈도우=`HYBRID_FLIP_WIN=20`
- **Vanilla**: Q init `q[:,1]=max(fee×50,0.05)`, annealing `2ε→ε`, prev_action=1 고정, 훈련 후 `Q[s,1]≥Q[s,0]+0.005` (전체 상태)
- **Composite Gap** = `0.6×(STATIC-Market) + 0.4×(STATIC-max(Vanilla, Market×0.3))`
- **Gap Penalized** = `gap_mean − 0.5×std(gaps) − 0.3×max(0, avg_mdd−15.0)` (λ_std=0.5, λ_mdd=0.3, MDD_FLOOR=15%)
- **σ_max Cosine 상한**: `σ_max_t = σ_min + 0.5×(σ_max−σ_min)×(1+cos(π×t/T_max))` → `σ=min(σ_reactive, σ_max_t)` (heuristic.py `PGActorCriticOptimizer`, `T_max=n_iters`)

## 상태 인코딩

```
4상태: state = is_bull×1 + is_above_ema×2  → {0,1,2,3}
8상태: state += is_high_vol×4              → {0..7}  (use_vol=True)
vol_threshold=None → 훈련 구간 중위수 자동 산출
```

## config.py RL_PARAMS 구조 (10 필드)

```python
RL_PARAMS = {
    TARGET_INDICES[0]: {
        "lr": float, "gamma": float, "epsilon": float, "v_epsilon": float,
        "episodes": int, "train_episodes": int, "seed": int,
        "use_vol": bool, "roll_period": int|None,
        "algorithm": "STATIC_H"  # "STATIC"|"STATIC_H"|"A2C"|"A3C"|"PPO"|"ACER"|"SAC"|"DDPG"
    },
    "default": { ... }
}
```

**오버라이드 우선순위**: 전역 session_state > per-member config

## Fallback apply_all — session_state 키 패턴

per-member 위젯 키: `f"{prefix}_{m_name}_{stock_name}"`

| 체크키 | prefix | 체크키 | prefix |
|--------|--------|--------|--------|
| timeframe | `tf_` | episodes | `epi_{interval}` |
| frame | `fspd_` | train_epi | `train_epi_` |
| seed | `seed_` | auto | `autoruns_` |
| active | `active_` | lr/gamma/eps/v_eps | 동명 |
| algo | `algo_` | sim_min/sim_mult | `sim_min_`/`sim_mult_` |

apply_all: ① 스냅 → ② 새값 적용 → ③ `st.rerun()`
revert_all: 스냅 복원 → `st.rerun()`

## 현재 성과 (improve 7-4 기준 — 전 종목 STATIC_H)

| M | 종목 | Ticker | STATIC_H | Composite Gap | 파라미터 특이사항 |
|---|------|--------|----------|---------------|------------------|
| 1 | SPY | SPY | 23.68% | 8.96 | seed=42, use_vol=True |
| 2 | QQQ | QQQ | 48.40% | 30.81 | seed=137, 최고성과 🏆 |
| 3 | KOSPI | ^KS11 | 157.06% | 17.53 | seed=2024, use_vol=True, 8-State |
| 4 | KOSDAQ | ^KQ11 | 53.38% | 22.43 | seed=777, use_vol=True, roll=30 |
| 5 | SCHD | SCHD | 28.61% | 2.00 | seed=314, use_vol=True, roll=60 |
| 6 | RGLD | RGLD | 105.17% | 12.17 | seed=100, use_vol=True |

> Composite Gap = 0.6×(STATIC_H−Market) + 0.4×(STATIC_H−max(Vanilla, Market×0.3))

## 저장된 최적 파라미터 (config.py 기준 — improve 7-4)

| M | lr | gamma | epsilon | v_epsilon | episodes | train_epi | seed | use_vol | roll |
|---|----|-------|---------|-----------|----------|-----------|------|---------|------|
| 1 | 0.07268 | 0.938065 | 0.133828 | 0.102277 | 300 | 150 | 42 | True | None |
| 2 | 0.080412 | 0.907645 | 0.119118 | 0.177335 | 500 | 300 | 137 | False | None |
| 3 | 0.049807 | 0.880582 | 0.023155 | 0.160969 | 500 | 300 | 2024 | True | None |
| 4 | 0.038007 | 0.921727 | 0.075960 | 0.061587 | 500 | 300 | 777 | True | 30 |
| 5 | 0.025708 | 0.917577 | 0.037793 | 0.106362 | 500 | 400 | 314 | True | 60 |
| 6 | 0.022200 | 0.950344 | 0.164422 | 0.139944 | 300 | 150 | 100 | True | None |

## 워크플로우 규칙

- **자동 커밋/푸시 금지**: 수정 후 변경 내용만 보여주고, 명시적 요청시만 커밋
