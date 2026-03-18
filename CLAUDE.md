# Chainers Master Fund — Claude 작업 지침

## 파일 구조
| 파일 | 역할 |
|------|------|
| `app.py` | Streamlit UI + `get_rl_data()` |
| `common/base_agent.py` | RL 훈련·평가 (STATIC/Vanilla/Neural) |
| `common/nn_utils.py` | TinyMLP, ReplayBuffer, extract_features |
| `common/heuristic.py` | PGActorCriticOptimizer |
| `common/evaluator.py` | softmax 비중, MDD, CTPT |
| `common/data_loader.py` | yfinance fetch + EMA_10, Rolling_Std |
| `common/stock_registry.py` | STOCK_REGISTRY(12종) + FEE_REGISTRY |
| `members/member_N/config.py` | MEMBER_NAME, TARGET_INDICES, RL_PARAMS |

## get_rl_data 시그니처 (9-tuple)
```python
get_rl_data(ticker, lr, gamma, epsilon, n_bars, train_episodes, seed,
            v_epsilon=None, fee_rate=0.0, interval="1d",
            use_vol=False, roll_period=None, algorithm="STATIC")
→ (df, v_trace, s_trace, real_ret_trace, s_mdd, v_log, s_log, s_theta, v_qtable)
```
`algorithm != "STATIC"` → `run_neural_rl()`, `s_theta=None`

## RL 상수 (base_agent.py — 직접 수정 금지)
```
TRAIN_RATIO=0.7  ENTROPY_COEFF=0.05  Q_FLOOR_MARGIN=0.005  EMA_SIGNAL_WEIGHT=2
NN_HIDDEN=32  N_FEATURES=5  PPO_CLIP_EPS=0.2  PPO_GAE_LAMBDA=0.95
SAC_ALPHA_LR=0.01  DDPG_TAU=0.005
```

## 수정 금지 제약
- **STATIC**: theta Cold-Start `[1,1]=max(0.05,fee×30)` / `[2,1]=max(0.1,fee×50)` / `[3,1]=max(0.2,fee×80)`, epsilon annealing 없음
- **Vanilla**: Q init `q[:,1]=max(fee×50,0.05)`, annealing `2ε→ε`, prev_action=1 고정, 훈련 후 `Q[s,1]≥Q[s,0]+0.005` (전체 상태)
- **Composite Gap** = `0.6×(STATIC-Market) + 0.4×(STATIC-max(Vanilla, Market×0.3))`

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
        "algorithm": "STATIC"  # "STATIC"|"A2C"|"A3C"|"PPO"|"SAC"|"DDPG"
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

## 현재 성과 및 알려진 문제 (improve 7-2 기준)

### 시뮬 결과 요약
| M | 종목 | STATIC | Market | Alpha | Expected α | 비고 |
|---|------|--------|--------|-------|-----------|------|
| 1 | SPY | 23.86% | 14.65% | +9.3% | +8.43%p | Seed↑ 의존성 있음 |
| 2 | QQQ | 46.12% | 16.93% | +29.3% | +28.55%p | 안정 |
| 3 | KOSPI | 133.22% | 122.98% | +10.8% | +10.24%p | MDD -19.24% 주의 |
| 4 | KOSDAQ | 63.79% | 62.96% | +1.2% | **-2.12%p** | ⚠ 구조적 약점 |
| 5 | SCHD | 19.60% | 17.21% | +2.5% | +0.68%p | 배당ETF 신호 약 |
| 6 | RGLD | 105.17% | 92.12% | +13.2% | +13.04%p | 5시드 동일값 |

Team Fund: +85.52% | Weight: M3·M6=28%, M2=16.5%, M4=12.3%, M1=8.3%, M5=6.9%

### 구조적 문제
- **M4 KOSDAQ**: Expected Alpha 음수 → seed=777이 운 좋은 케이스. 어떤 파라미터로도 시장 초과 어려움 (OOS 구간 급등 구조)
- **M5 SCHD**: SCHD 일변동 낮아 EMA/Bull 신호 구분력 부족. STATIC Range 16.56~19.81% (시드 의존)
- **M6 RGLD**: 5시드 모두 완전 동일 결과 → seed가 실질 무효 (탐색 다양성 없음)

### 저장된 최적 파라미터 (config.py)
- M1 SPY: lr=0.0802, γ=0.8732, ε=0.1667, v_ε=0.1347, seed=42
- M2 QQQ: lr=0.072192, γ=0.966283, ε=0.21158, v_ε=0.062005, seed=137
- M3 KOSPI: lr=0.005, γ=0.908675, ε=0.146537, v_ε=0.19471, seed=2024
- M4 KOSDAQ: lr=0.038007, γ=0.921727, ε=0.07596, v_ε=0.061587, seed=777
- M5 SCHD: lr=0.0624, γ=0.9449, ε=0.1708, v_ε=0.0879, seed=314
- M6 RGLD: lr=0.0269, γ=0.9380, ε=0.1509, v_ε=0.1268, seed=99

## 워크플로우 규칙
- **자동 커밋/푸시 금지**: 수정 후 변경 내용만 보여주고, 명시적 요청시만 커밋
