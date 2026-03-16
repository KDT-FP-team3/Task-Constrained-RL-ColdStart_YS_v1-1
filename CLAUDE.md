# Chainers Master Fund — Claude 작업 지침 (improve 5-1 기준)

> Claude Code 자동 로드 파일. 매 대화 시작 시 app.py(2281줄) 전체 재독 없이 작업 가능.
> **수정 전 반드시 이 파일 전체를 먼저 읽을 것.**

---

## 파일 역할 지도 (현행 줄 수)

| 파일 | 줄 수 | 역할 | 수정 빈도 |
|------|-------|------|----------|
| `app.py` | 2295 | Streamlit UI 전체 + `get_rl_data()` 정의 | 높음 |
| `common/base_agent.py` | 374 | RL 훈련·평가 엔진 (STATIC + Vanilla) | 중간 |
| `common/heuristic.py` | 405 | PGActorCriticOptimizer (파라미터 자동 탐색) | 낮음 |
| `common/evaluator.py` | 57 | softmax 비중, MDD, CTPT 코드 계산 | 낮음 |
| `common/data_loader.py` | 67 | yfinance fetch + EMA_10, Rolling_Std 계산 | 낮음 |
| `common/stock_registry.py` | — | STOCK_REGISTRY {0:SPY,1:QQQ,2:^KS11,3:^KQ11,4:NVDA,5:TSLA,...} | 거의 없음 |
| `members/member_N/config.py` | — | MEMBER_NAME, TARGET_INDICES, RL_PARAMS (use_vol/roll_period 포함) | 파라미터 탐색 후 |

---

## 작업 패턴 — 토큰 절약 가이드

| 요청 유형 | 읽어야 할 파일 | 건드리지 않을 파일 |
|----------|--------------|-----------------|
| UI 레이아웃·렌더링 수정 | app.py 해당 구간만 | base_agent.py, heuristic.py |
| RL 알고리즘 파라미터 조정 | base_agent.py 상수 블록(1~6줄) | app.py |
| 팀 펀드 배분 로직 | app.py `draw_top_dashboard()` | base_agent.py |
| 파라미터 탐색 튜닝 | heuristic.py | base_agent.py, app.py UI |
| config.py 최적값 갱신 | members/member_N/config.py | 공통 모듈 전체 |
| 데이터 지표 추가 | data_loader.py `_postprocess_df()` | 나머지 전체 |

---

## 핵심 API 시그니처

### `get_rl_data` (app.py 내부 정의)

```python
get_rl_data(
    ticker, lr, gamma, epsilon, n_bars, train_episodes, seed,
    v_epsilon=None, fee_rate=0.0, interval="1d",
    use_vol=False,      # [P3] True → 8상태 변동성 모드
    roll_period=None    # [P4] int → OOS 주기 재학습 (봉 수)
) → (df, v_trace, s_trace, real_ret_trace, s_mdd,
      v_log, s_log, s_theta, v_qtable)   # 9-tuple
```

> ⚠️ **모든 호출 지점에서 9개 언패킹 필수**
> ```python
> df, v_trace, s_trace, mkt, s_mdd, v_log, s_log, s_theta, v_qtable = get_rl_data(...)
> 불필요한 값: _, _, s_tr, mkt, _, _, _, _, _ = get_rl_data(...)
> ```
> 현재 호출 지점: app.py 3곳 (Eval 루프, Simul 평가, 메인 렌더링)

### `run_rl_simulation_with_log` (base_agent.py 핵심 함수)

```python
run_rl_simulation_with_log(
    df, lr, gamma, epsilon, episodes, use_static, seed, fee_rate,
    vols=None, vol_threshold=None,  # [P3] 변동성 배열·임계값
    roll_period=None,               # [P4] OOS 재학습 주기
    return_policy=False             # [P2] True → (trace, log, policy) 반환
) → (cumulative_return, action_log)          # return_policy=False (기본)
  → (cumulative_return, action_log, policy)  # return_policy=True
```

> `run_rl_simulation`: 위 함수의 thin wrapper, 누적수익만 반환.

### `PGActorCriticOptimizer` (heuristic.py)

```python
PGActorCriticOptimizer(param_bounds, lr_actor=0.12, sigma_init=0.18,
                        sigma_min=0.02, sigma_max=0.30, value_alpha=0.25, seed)
  .suggest_next() → {lr, gamma, epsilon, v_epsilon}
  .update(gap: float)
  .best_params → dict | None
```

### `evaluator.py` 함수들

```python
calculate_softmax_weights(scores, temperature=1.0) → np.ndarray  # 합=1
calculate_mdd(returns_percent_array) → float
calculate_ctpt_and_color(lr, gamma, epsilon) → (code_str, desc, hex_color)
```

---

## RL 알고리즘 상수 블록 (base_agent.py 1~6줄)

```python
TRAIN_RATIO       = 0.7    # 워크포워드 분할: 앞 70% 학습 / 30% OOS
ENTROPY_COEFF     = 0.05   # r_eff = r + ENTROPY_COEFF × H(π)
Q_FLOOR_MARGIN    = 0.005  # Q[s,BUY] ≥ Q[s,CASH] + Q_FLOOR_MARGIN
EMA_SIGNAL_WEIGHT = 2      # state = is_bull×1 + is_above_ema×EMA_SIGNAL_WEIGHT
```

> 알고리즘 수치를 바꿀 때는 **이 블록만 수정**. 나머지 코드는 상수를 참조함.

---

## 상태 인코딩 — 선형 조합 (`_encode_state`)

```
state = Σ(signal_i × 2^i)

4상태 (기본, use_vol=False):
  signals = [is_bull, is_above_ema]
  state ∈ {0(하락+EMA아래), 1(상승+EMA아래), 2(하락+EMA위), 3(상승+EMA위)}

8상태 (P3 활성, use_vol=True):
  signals = [is_bull, is_above_ema, is_high_vol]
  state ∈ {0..3} (저변동) ∪ {4..7} (고변동, bit2=1)
```

> 8상태 사용 시: theta shape (4,2) → (8,2). config.py 파라미터 재탐색 권장.
> `vol_threshold=None` → 훈련 구간 중위수 자동 산출 (데이터 적응형).

---

## 알고리즘 제약 (수정 금지 항목)

### STATIC RL (Actor-Critic)
- **theta Cold-Start 초기화 패턴**: `theta[1,1]=max(0.05,fee×30)`, `[2,1]=max(0.1,fee×50)`, `[3,1]=max(0.2,fee×80)`
- **ENTROPY_COEFF = 0.05** — 4-8에서 0.02→0.05 확정. 낮추면 Buy&Hold 고착.
- **epsilon annealing 없음** — 상수 ε 유지 (annealing 추가 금지)

### Vanilla RL (Q-Learning)
- **Q Cold-Start 초기화**: `q[:,1] = max(fee×50, 0.05)`
- **epsilon annealing**: `2ε → ε` (초반 탐험 강화, 이 스케줄 변경 금지)
- **prev_action=1** (에피소드 시작 BUY 고정 — CASH 편향 방지)
- **Q_FLOOR_MARGIN = 0.005** — 4-9에서 0.001→0.005 확정

### 평가 구조
- Composite Gap = `0.6×(STATIC-Market) + 0.4×(STATIC-max(Vanilla, Market×0.3))`
- 평가: 전체 100% (학습 70% 포함) — 분리 금지

### Rolling Window (P4) — 변경 제약
- STATIC RL 전용 (`use_static=True`일 때만 활성화)
- `roll_period=None` 기본 → 기존 결과와 직접 비교 시 OFF 유지
- 멤버별 기본값은 config.py `RL_PARAMS["roll_period"]`로 관리 (None = 비활성)

---

## 파라미터 허용 범위

| 파라미터 | 탐색 범위 | 비고 |
|---------|----------|------|
| lr (α) | 0.005 ~ 0.10 | PGOptimizer 탐색 |
| gamma (γ) | 0.85 ~ 0.99 | PGOptimizer 탐색 |
| epsilon (STATIC ε) | 0.01 ~ 0.25 | PGOptimizer 탐색 |
| v_epsilon (Vanilla ε) | 0.01 ~ 0.25 | PGOptimizer 탐색 |
| fund_temperature | 1.0 ~ 5.0 | UI 슬라이더, 런타임 조정 가능 |
| fund_max_weight | 0.10 ~ 1.0 | UI 슬라이더, 런타임 조정 가능 |

---

## 멤버별 최적 파라미터 (improve 4-9 기준 + improve 5-2 use_vol/roll_period 추가)

| 멤버 | 종목 | lr | gamma | ε(S) | ε(V) | seed | use_vol | roll_period |
|------|------|-----|-------|------|------|------|---------|-------------|
| M1 | SPY | 0.0496 | 0.8863 | 0.1190 | 0.0993 | 42 | False | None |
| M2 | QQQ | 0.0650 | 0.9075 | 0.1005 | 0.1043 | 137 | False | None |
| M3 | KOSPI | 0.0227 | 0.9569 | 0.1386 | 0.1762 | 2024 | False | None |
| M4 | KOSDAQ | 0.0168 | 0.9084 | 0.0863 | 0.1157 | 777 | False | None |
| M5 | NVDA | 0.0497 | 0.9183 | 0.0443 | 0.1055 | 314 | **True** | **15** |
| M6 | TSLA | 0.0364 | 0.8873 | 0.1283 | 0.0842 | 99 | **True** | **20** |

> KOSPI/KOSDAQ: OOS 구조적 한계로 α < Market 정상. 수정 대상 아님.
> NVDA/TSLA: 고변동성 — 8상태(use_vol=True) + 주기 재학습 활성. 파라미터 재탐색 권장.

---

## config.py RL_PARAMS 전체 키 구조 (improve 5-2 기준)

```python
RL_PARAMS = {
    stock_idx: {
        "lr":            float,   # Actor-Critic 학습률
        "gamma":         float,   # 할인율
        "epsilon":       float,   # STATIC ε
        "v_epsilon":     float,   # Vanilla ε
        "episodes":      int,     # 전체 에피소드 수
        "train_episodes":int,     # 학습 에피소드 수
        "seed":          int,     # 재현성 시드
        "use_vol":       bool,    # [P3] True → 8상태 변동성 신호 (기본 False)
        "roll_period":   int|None # [P4] OOS 재학습 주기 봉 수 (기본 None)
    },
    "default": { ... }   # stock_idx 없을 때 fallback
}
```

> **우선순위**: 전역 session_state(UI 오버라이드) > per-member config 기본값
> - `use_vol_feature=True` (전역 ON) → 전체 강제 8상태
> - `use_vol_feature=False` → 각 멤버 config.py 값 사용 (NVDA/TSLA True, 나머지 False)
> - `roll_period_active=True` (전역 ON) → 전체 `roll_period_val` 적용
> - `roll_period_active=False` → 각 멤버 config.py `roll_period` 사용

---

## app.py — session_state 전체 키 목록

```
# 시스템
master_pbar_pct          float   Analyzing Agents 진행률 (0.0~1.0)
prev_final_contributions list    이전 실행 멤버 결과
prev_episodes_run        int     Real-time Load 게이지용 에피소드 수
run_all_queue            list    [(m_name, stock_name)] Eval All 큐
sim_all_queue            list    [(m_name, stock_name)] Simul All 큐
sim_auto_save            bool    Simul All 자동 저장
interrupt_requested      bool    중단 요청

# 파라미터·상태
member_traces            dict    {m_name: {s_trace, dates, stocks}}
stock_use_fallback       str|None  "ALL" or None
fallback_params          dict    전역 fallback 파라미터 스냅샷
fallback_prev_state      dict    되돌리기용 이전 상태 스냅샷
stocks_reverted          set     Run Evaluation으로 복귀한 종목
stock_trial_history      dict    {hist_key: [trial_dict, ...]}
sim_result               dict    {hist_key: best_params_dict}
ghost_data               dict    {hist_key: {v_trace, s_trace, params, gap}}

# [P1] Fund 배분 설정
fund_temperature         float   Softmax 온도 (기본 1.0, 범위 1.0~5.0)
fund_max_weight          float   단일 종목 최대 비중 (기본 1.0, 범위 0.1~1.0)

# [P2] Explainable RL
policy_cache             dict    {hist_key: {theta, q_table, n_states}}

# [P3/P4] 신규 모드 토글
use_vol_feature          bool    8상태 변동성 모드 (기본 False)
roll_period_active       bool    Rolling Window 재학습 (기본 False)
roll_period_val          int     재학습 주기 봉 수 (기본 20)
```

---

## app.py — UI 레이아웃 구조

```
sidebar:
  System Status
    master_progress_placeholder  ← _render_master_pbar_html() [즉시 렌더 필수]
    gauge_placeholder            ← update_load_bar()
  [▶ Eval. All] [⚙ Simul. All] [■ 중단]
  [🔁 All 적용] [↩ 되돌리기]
  "Fund & Agent Settings" expander  ← [P1/P3/P4 신규]
    Temperature 슬라이더 (1.0~5.0)
    Weight Cap 슬라이더 (10~100%)
    8-State Mode 토글 (P3)
    Rolling Retrain 토글 + Roll Period 입력 (P4)
  "Fallback Parameters" expander
    슬라이더: LR/Gamma/STATIC_ε/Vanilla_ε/Sim_Min/Sim_Mult

main:
  상단 컨테이너 (prev_final_contributions 기반 — draw_top_dashboard)
  ### Portfolio Managers (Independent RL Labs)
    멤버별 루프:
      col_left:
        누적수익 차트 → 지표 카드(3열) → Agent Decision Analysis
        "State Policy Analysis" expander  ← [P2 신규, policy_cache 있을 때]
          STATIC P(BUY|s) 수평 막대 / Vanilla Q-Advantage 차트
      col_right:
        Trial History Statistical Analysis
```

---

## app.py — 렌더링·코딩 규칙

- `st.sidebar.empty()` placeholder → **생성 직후 즉시** 채울 것 (공백 방지)
- 진행률 바: CSS injection 금지, `_render_master_pbar_html(pct, ph)` 사용
  - 100% → `#AAFF00`(연두), 진행 중 → `#1C83E1`(파랑)
- HTML 게이지: `update_load_bar()` 스타일 (inline style + `with placeholder: st.markdown(...)`)
- config.py 저장: `save_config_to_file()` 함수 사용 (app.py 내부)
- 멤버 순서: `sorted_modules` (MEMBER_NAME 기준 정렬)
- **fund_temperature는 런타임 조정 가능** (session_state에서 읽음, 하드코딩 금지)

### 팀 펀드 계산 (draw_top_dashboard 내부)

```python
score_i = avg_return_i / (1 + abs(avg_mdd_i))
weights = calculate_softmax_weights(scores, temperature=fund_temperature)  # session_state
if fund_max_weight < 1.0:                                                   # Weight Cap
    weights = np.minimum(weights, fund_max_weight)
    weights /= weights.sum()
team_curve = np.dot(weights, aligned_traces)
```

### Simulation 루프 주요 계산

```python
n_iters = max(eff_sim_min, l_auto_runs * eff_sim_mult)
_n_eval = min(4, max(3, l_auto_runs // 2))
_eval_seeds = [base_seed + j*37 for j in range(_n_eval)]
composite_gap = 0.6×(STATIC-Market) + 0.4×(STATIC-max(Vanilla, Market×0.3))
```

---

## 개선 이력

| 버전 | 내용 |
|------|------|
| 4-8 | ENTROPY_COEFF 0.02→0.05, theta[2,1] 초기화 강화 |
| 4-9 | Q_FLOOR_MARGIN 0.001→0.005, SPY/QQQ/TSLA 파라미터 재탐색 |
| 5-1 | P0 상수화 / P1 Temperature+WeightCap UI / P2 Explainable RL / P3 8상태+_encode_state / P4 Rolling Window |
| 5-2 | P3/P4 멤버별 분리 — config.py RL_PARAMS에 use_vol/roll_period 추가, 전역 session_state 오버라이드 구조, Eval/Simul 루프 동기화 |

---

## linter 무시 목록 (false positive)

| 위치 | 경고 | 이유 |
|------|------|------|
| `app.py` (Simul 루프 내) | `col_right is not defined` | `st.rerun()`이 먼저 종료하므로 도달 불가 |
| `base_agent.py` `_make_state_vanilla` | `price`, `ema` not accessed | 기본값 `None` 파라미터, 호출 패턴 호환용 |

---

## 알파 판정 기준

| Gap (STATIC − Vanilla) | 판정 |
|------------------------|------|
| ≥ 1%p | 목표 달성 |
| ≥ 5%p | 우수 달성 |
| ≥ 25%p | 최고 달성 🏆 |
