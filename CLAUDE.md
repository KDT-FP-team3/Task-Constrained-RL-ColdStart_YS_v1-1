# Chainers Master Fund — Claude 작업 지침

> Claude Code가 자동으로 읽는 프로젝트 컨텍스트 파일입니다.
> 수정 작업 전 이 파일을 먼저 참조하여 app.py 전체 재독을 피하세요.

---

## 프로젝트 개요

Task-Constrained RL Cold-Start 조건에서 STATIC RL(Actor-Critic, 4상태)과
Vanilla RL(Q-Learning, 2상태)의 성과를 비교하는 멀티 에이전트 트레이딩 시뮬레이터.
멤버 6명이 각 담당 종목(SPY/QQQ/KOSPI/KOSDAQ/NVDA/TSLA)에 에이전트를 배치하여
팀 펀드 수익률을 측정한다.

---

## 파일 역할 지도

| 파일 | 줄 수 | 역할 |
|------|-------|------|
| `app.py` | ~2093 | Streamlit UI 전체 (수정 시 반드시 Read 먼저) |
| `common/base_agent.py` | ~396 | STATIC RL 훈련·평가 + Vanilla RL 훈련·평가 |
| `common/heuristic.py` | ~405 | PGActorCriticOptimizer (파라미터 자동 탐색) |
| `common/evaluator.py` | ~57 | softmax 비중, MDD, CTPT 코드 계산 |
| `common/data_loader.py` | — | yfinance 데이터 fetch, EMA 계산 |
| `common/stock_registry.py` | — | STOCK_REGISTRY {0:SPY, 1:QQQ, 2:^KS11, 3:^KQ11, 4:NVDA, 5:TSLA, ...} |
| `members/member_N/config.py` | — | MEMBER_NAME, TARGET_INDICES, RL_PARAMS |
| `Logic.txt` | — | 알고리즘 설명 (비정형) |

---

## 핵심 함수 시그니처

```python
# base_agent.py
get_rl_data(ticker, interval, n_bars, train_episodes, lr, gamma,
            epsilon, v_epsilon, seed, fee_rate=0.001)
  → dict: {
      's_return': float,   # STATIC RL 최종 수익률(%)
      'v_return': float,   # Vanilla RL 최종 수익률(%)
      's_mdd': float,      # STATIC RL MDD(%)
      's_trace': list,     # STATIC 에쿼티 곡선
      'v_trace': list,
      'm_trace': list,     # 시장 Buy&Hold 곡선
      'dates': index,
      'market_return': float
    }

# heuristic.py
PGActorCriticOptimizer(param_bounds, sigma_max=0.30, sigma_init=0.18)
  .suggest_next() → {lr, gamma, epsilon, v_epsilon}  # 정규화 공간 후보
  .update(gap: float)  # composite gap으로 정책 업데이트

# evaluator.py
calculate_softmax_weights(scores, temperature=1.0) → np.ndarray  # 합=1
calculate_mdd(returns_percent_array) → float
calculate_ctpt_and_color(lr, gamma, epsilon) → (code, desc, color)
```

---

## 알고리즘 제약 (수정 금지)

아래는 성능 검증 완료된 설계 결정으로, 임의 변경 금지.

### STATIC RL (Actor-Critic)
- **theta init**: `theta[1,1]=max(0.05,fee*30)`, `[2,1]=max(0.1,fee*50)`, `[3,1]=max(0.2,fee*80)` — fee 비례 BUY 선호
- **엔트로피 정규화 계수**: `0.05` — `r_eff = r + 0.05·H(π)` (Buy&Hold 고착 방지, 4-8에서 0.02→0.05 확정)
- **epsilon annealing 없음** — 상수 epsilon 유지

### Vanilla RL (Q-Learning)
- **Q init**: `q[:,1] = max(fee_rate*50, 0.05)` — BUY 초기 우위
- **epsilon annealing**: `2ε → ε` (학습 후반 수렴)
- **prev_action = 1** (BUY 시작 고정 — CASH 편향 방지)
- **훈련 후 보정 margin**: `0.005` — `q[s,BUY] ≥ q[s,CASH] + 0.005` (4-9 확정, 0.001→0.005)

### 평가 구조
- 워크포워드: 첫 **70%** 학습 / 전체 **100%** 평가
- Composite Gap = `0.6 × (STATIC - Market) + 0.4 × (STATIC - max(Vanilla, Market×0.3))`

---

## 파라미터 허용 범위

| 파라미터 | 범위 | 탐색 공간 |
|---------|------|----------|
| lr (α) | 0.005 ~ 0.10 | PGOptimizer 탐색 |
| gamma (γ) | 0.85 ~ 0.99 | PGOptimizer 탐색 |
| epsilon (STATIC ε) | 0.01 ~ 0.25 | PGOptimizer 탐색 |
| v_epsilon (Vanilla ε) | 0.01 ~ 0.25 | PGOptimizer 탐색 |
| sigma_max | 0.30 | 고정 |
| sigma_init | 0.18 | 고정 |
| temperature (softmax) | 1.0 | 고정 |

---

## 멤버별 최적 파라미터 (improve 4-9 기준)

| 멤버 | 종목 | lr | gamma | ε | v_ε | seed |
|------|------|-----|-------|---|-----|------|
| M1 | SPY | 0.0496 | 0.8863 | 0.1190 | 0.0993 | 42 |
| M2 | QQQ | 0.0650 | 0.9075 | 0.1005 | 0.1043 | 137 |
| M3 | KOSPI | 0.0227 | 0.9569 | 0.1386 | 0.1762 | 2024 |
| M4 | KOSDAQ | 0.0168 | 0.9084 | 0.0863 | 0.1157 | 777 |
| M5 | NVDA | 0.0497 | 0.9183 | 0.0443 | 0.1055 | 314 |
| M6 | TSLA | 0.0364 | 0.8873 | 0.1283 | 0.0842 | 99 |

> KOSPI/KOSDAQ: 워크포워드 OOS 구조상 α < market 정상 — 수정 대상 아님

---

## app.py 핵심 구조

### session_state 주요 키
```
master_pbar_pct          float       Analyzing Agents 진행률 (0.0~1.0)
prev_final_contributions list        이전 실행 멤버 결과
prev_episodes_run        int         Real-time Load 게이지용 에피소드 수
run_all_queue            list        [(m_name, stock_name), ...] Eval All 큐
sim_all_queue            list        [(m_name, stock_name), ...] Simul All 큐
sim_auto_save            bool        Simul All 자동 저장 플래그
interrupt_requested      bool        중단 요청 플래그
member_traces            dict        {m_name: {s_trace, dates, stocks}}
stock_use_fallback       str|None    "ALL" or None
fallback_params          dict        전역 fallback 파라미터 스냅샷
```

### UI 레이아웃 구조
```
sidebar:
  System Status
    master_progress_placeholder  ← _render_master_pbar_html() 즉시 렌더 필수
    gauge_placeholder            ← update_load_bar()
  [▶ Eval. All] [⚙ Simul. All] [■ 중단]
  [🔁 All 적용] [↩ 되돌리기]
  Fallback Parameters (expander)
    슬라이더: LR/Gamma/STATIC_ε/Vanilla_ε/Sim_Min_Steps/Sim_Step_Mult

main:
  상단 컨테이너 (prev_final_contributions 기반 차트/테이블)
  ### Portfolio Managers (Independent RL Labs)
    멤버별 루프 → 종목별 차트 렌더
  Team Fund (Softmax 비중)
```

### 렌더링 주의사항
- `st.sidebar.empty()` placeholder는 **생성 직후 즉시** 채울 것 (incremental rendering 공백 방지)
- CSS injection 대신 **inline style HTML** 사용 (Streamlit emotion CSS 충돌)
- `_render_master_pbar_html(pct, placeholder)`: 100%→`#AAFF00`(연두), 진행중→`#1C83E1`(파랑)
- `update_load_bar(episodes_run, placeholder)`: Real-time Load 그라디언트 바

### 주요 계산
```python
n_iters = max(eff_sim_min, l_auto_runs * eff_sim_mult)
_n_eval  = min(4, max(3, l_auto_runs // 2))
_eval_seeds = [base_seed + j*37 for j in range(_n_eval)]
score_i = avg_return_i / (1 + abs(avg_mdd_i))   # evaluator.py
team_curve = np.dot(softmax_weights, aligned_traces)
```

---

## 코딩 규칙

- 새 HTML 바/게이지는 `update_load_bar` 스타일(inline style, `with placeholder: st.markdown(...)`) 따를 것
- config.py 저장 시 `save_config_to_file()` 함수 사용 (app.py 내부)
- 멤버 순서: `sorted_modules` (MEMBER_NAME 기준 정렬)
- 팀 펀드 softmax temperature = **1.0** 고정 (변경 금지)

---

## linter 무시 목록 (false positive)

| 위치 | 경고 | 이유 |
|------|------|------|
| `app.py` ~1396 | `col_right is not defined` | `st.rerun()`이 먼저 실행되어 도달 불가 |
| `base_agent.py` L17 | `price`, `ema` not accessed | `_make_state_vanilla` 시그니처 통일 목적 |

---

## 알파 판정 기준

| Gap (STATIC − Vanilla) | 판정 |
|------------------------|------|
| ≥ 1%p | 목표 달성 |
| ≥ 5%p | 우수 달성 |
| ≥ 25%p | 최고 달성 🏆 |
