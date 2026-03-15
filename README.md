# Chainers Master Fund — Task-Constrained RL Cold-Start

---

## 요약 보고서

### 프로젝트 개요

본 프로젝트는 **Task-Constrained RL Cold-Start** 조건에서 EMA 기반 4-상태 Actor-Critic 에이전트(STATIC RL)와 2-상태 Q-Learning 에이전트(Vanilla RL)의 성과를 비교하는 멀티 에이전트 트레이딩 시뮬레이터다. 6명의 팀원이 각자 담당 종목에 에이전트를 배치하여 팀 펀드 수익률을 측정한다.

**핵심 연구 질문**

사전 데이터 없이 Cold-Start 조건에서, EMA 기반 4-상태 Actor-Critic 에이전트는 2-상태 Q-Learning 에이전트 대비 얼마나 높은 누적 수익률을 달성하는가?

**Alpha Gap** = STATIC RL 최종 수익률 − Vanilla RL 최종 수익률

- Gap ≥ 1%p : 목표 달성
- Gap ≥ 5%p : 우수 달성
- Gap ≥ 25%p : 최고 달성 🏆

---

### 알고리즘 비교 (improve 4-7 기준)

| 항목 | STATIC RL | Vanilla RL |
|------|-----------|------------|
| 알고리즘 | Actor-Critic (Policy Gradient Theorem) | Tabular Q-Learning |
| 상태 수 | 4 (추세 × EMA 위치) | 2 (추세만) |
| 정책 표현 | Softmax 확률 정책 π_θ(a\|s) | argmax Q(s, a) |
| Baseline | TD Critic V(s) (분산 감소) | 없음 |
| 초기화 | theta[s,1] = fee 비례 BUY 선호 | Q[:,1] = max(fee×50, 0.05) |
| 엔트로피 정규화 | r_eff = r + 0.02·H(π) (Buy&Hold 고착 방지) | 없음 |
| 탐험 | 상수 epsilon-greedy | epsilon annealing (2ε→ε) |
| 에피소드 시작 | prev_action=0 | prev_action=1 (BUY 고정, CASH 편향 제거) |
| 훈련 후 보정 | 없음 | Q[1,BUY] ≥ Q[1,CASH]+0.001 (상대 우위 하한) |
| 훈련 데이터 | 전체의 첫 70% (워크포워드) | 전체의 첫 70% (워크포워드) |
| 역할 | 평가 대상 | 비교 기준선 |

---

### 팀 구성 및 담당 종목

| 멤버 | 담당 종목 | Ticker | 시드 | 최적 파라미터 (improve 4-7) |
|------|-----------|--------|------|---------------------------|
| Member 1 | S&P 500 ETF | SPY | 42 | lr=0.0577, γ=0.8938, ε=0.1624, v_ε=0.1706 |
| Member 2 | Nasdaq 100 ETF | QQQ | 137 | lr=0.0627, γ=0.8805, ε=0.1028, v_ε=0.1640 |
| Member 3 | KOSPI 지수 | ^KS11 | 2024 | lr=0.0227, γ=0.9569, ε=0.1386, v_ε=0.1762 |
| Member 4 | KOSDAQ 지수 | ^KQ11 | 777 | lr=0.0168, γ=0.9084, ε=0.0863, v_ε=0.1157 |
| Member 5 | NVIDIA | NVDA | 314 | lr=0.0497, γ=0.9183, ε=0.0443, v_ε=0.1055 |
| Member 6 | Tesla | TSLA | 99 | lr=0.0539, γ=0.9083, ε=0.1322, v_ε=0.1596 |

추가 지원 종목: GOOGL, MSFT, 삼성전자(005930.KS), SK하이닉스(000660.KS)

---

### 최신 시뮬레이션 성과 요약 (improve 4-7 기준)

| 종목 | STATIC RL | Market | Vanilla RL | Alpha Gap | σ | 비고 |
|------|-----------|--------|------------|-----------|---|------|
| SPY | +32.62% | +32.70% | -4.80% | +9.07%p | 0.098 | ✅ 수렴 |
| QQQ | +74.63% | +38.34% | -16.90% | +47.03%p | 0.150 | 🏆 최고 Alpha |
| KOSPI | +47.21% | +103.72% | +6.55% | -27.47%p | 0.089 | ⚠️ OOS 구조 한계 |
| KOSDAQ | +2.45% | +29.56% | -24.70% | -18.83%p | 0.233 | ⚠️ OOS 구조 한계 |
| NVDA | +195.85% | +105.33% | 불안정 | +120.01%p | 0.177 | 🏆 최대 수익 |
| TSLA | +138.94% | +139.16% | 0.00% | +38.74%p | 0.150 | ⚠️ Vanilla 0% |

> KOSPI/KOSDAQ는 워크포워드 구조적 한계: 학습 구간(2024~2025 상반기) 횡보/하락, OOS(2025 하반기~2026) 급등. 어떤 파라미터로도 OOS 수익률 초과 어려움.

---

### 거래 수수료

| 시장 | 매수 | 매도 | 왕복 합계 |
|------|------|------|-----------|
| 미국 주식·ETF | 0.05% | 0.05% | 0.10% |
| 국내 주식·지수 | 0.015% | 0.215% | 0.23% |

---

### 주요 기능

- **Run Evaluation**: 현재 파라미터로 RL 에이전트를 평가하고 Trial History를 축적한다.
- **Simulation**: PG Actor-Critic Optimizer로 하이퍼파라미터를 자동 탐색하여 복합 Gap을 극대화한다.
- **Fallback Parameters**: 체크박스로 선택한 파라미터만 모든 종목에 일괄 적용하거나 이전 상태로 복원한다.
- **Trial History Statistical Analysis**: 반복 평가 결과를 박스 플롯, 추이 차트, 통계 요약으로 표시한다.
- **Ghost Line**: Simulation에서 발견된 최적 파라미터의 수익 곡선을 현재 차트에 점선으로 병렬 표시한다.
- **팀 포트폴리오 대시보드**: Softmax 가중 배분으로 전체 펀드 성과를 집계한다.

---

### 설치 및 실행

```
Python 3.9+
pip install streamlit yfinance numpy pandas plotly
streamlit run app.py
```

---

## 전체 상세 설명

### 목차

1. [시스템 아키텍처](#1-시스템-아키텍처)
2. [강화학습 알고리즘](#2-강화학습-알고리즘)
3. [워크포워드 검증 — Train/Test 분리](#3-워크포워드-검증--traintest-분리)
4. [하이퍼파라미터 탐색 — PG Actor-Critic Optimizer](#4-하이퍼파라미터-탐색--pg-actor-critic-optimizer)
5. [하이퍼파라미터 상세](#5-하이퍼파라미터-상세)
6. [랜덤 시드의 역할](#6-랜덤-시드의-역할)
7. [데이터 파이프라인](#7-데이터-파이프라인)
8. [포트폴리오 평가 및 성과 지표](#8-포트폴리오-평가-및-성과-지표)
9. [UI 기능 상세](#9-ui-기능-상세)
10. [파일 구조](#10-파일-구조)
11. [Simulation 단계별 연산 흐름](#11-simulation-단계별-연산-흐름)
12. [개선 이력](#12-개선-이력)

---

## 1. 시스템 아키텍처

```
app.py  (Streamlit 웹 UI, ~1900줄)
 |
 +-- 사이드바
 |    +-- Eval. All / Simul. All 버튼
 |    +-- Fallback Parameters (항목별 체크박스 + 일괄 적용/되돌리기)
 |    +-- System Status (Cloud/Local, GPU/CPU 표시)
 |
 +-- 팀 포트폴리오 대시보드
 |    +-- All Members 누적 수익 차트 + Team Fund
 |    +-- 멤버별 성과 테이블 (STATIC, Vanilla, Alpha Gap, MDD, Score, Weight%)
 |
 +-- 멤버별 탭 (6개)
      +-- 종목별 파라미터 패널
      |    +-- [1행] Timeframe / Trading Days / Train Episodes / Frame Speed / Base Seed / Auto Run Count / Active Agents
      |    +-- [2행] LR, Gamma, STATIC ε, Vanilla ε, Sim Min Steps, Sim Step Mult.
      +-- Run Evaluation / Simulation 버튼
      +-- 좌측 패널
      |    +-- 누적 수익 차트 (Ghost Line 포함)
      |    +-- Final Cumulative Return 카드 (Vanilla / STATIC / Market)
      |    +-- Agent Decision Analysis
      |         +-- STATIC Action Frequency 막대 차트
      |         +-- 일별 행동 로그 테이블
      +-- 우측 패널
           +-- Trial-by-Trial Return 추이 차트
           +-- Return Distribution 박스 플롯
           +-- Statistics Summary (Mean±σ, Range)
           +-- Trial 데이터 테이블 (헤더 줄바꿈, 전체 열 표시)

common/
 +-- base_agent.py      RL 훈련 및 평가 (Actor-Critic / Q-Learning)
 +-- heuristic.py       하이퍼파라미터 탐색 (PGActorCriticOptimizer)
 +-- evaluator.py       성과 지표 계산 (MDD, Softmax 비중, CTPT 코드)
 +-- data_loader.py     yfinance 데이터 로드 (다봉 지원, 캐시 1시간)
 +-- stock_registry.py  종목 정보 및 수수료 테이블

members/member_N/
 +-- config.py          멤버별 담당 종목 + RL 하이퍼파라미터 (v_epsilon 포함)
```

---

## 2. 강화학습 알고리즘

### 2.1 STATIC RL — Actor-Critic

**파일:** `common/base_agent.py` — `_train_actor_critic_static()`

Policy Gradient Theorem + REINFORCE with baseline을 온라인 TD 방식으로 구현한다.

#### 마르코프 결정 과정 (MDP) 정의

```
상태 공간 S = {0, 1, 2, 3}  (4개 상태)
행동 공간 A = {0(CASH), 1(BUY)}
전이 확률 P(s'|s, a)  = 시장 가격 변동 (외생 확률 과정)
보상 함수 R(s, a, s') = daily_return × 1[a=BUY] − fee + 0.02·H(π)
할인 인수 γ           = 종목별 최적값 (0.88~0.96)
```

#### 벨만 최적 방정식 (Policy Gradient 기반)

```
V*(s) = max_π E[Σ γ^t · r_t | s_0 = s, π]

온라인 TD(0) 근사:
  δ_t = r_eff + γ · V(s_{t+1}) − V(s_t)   ← TD 오차 (Advantage 근사)
  r_eff = reward + 0.02 · H(π)             ← 엔트로피 정규화 (Buy&Hold 고착 방지)
  V(s_t) += lr · δ_t                       ← Critic 업데이트
```

#### Policy Gradient Theorem

```
∇J(θ) = E_π[∇ log π_θ(a|s) · A(s, a)]

Actor (Softmax 정책):
  π_θ(a|s) = softmax(θ[s, :])
  ∇ log π(a|s) = 1[a == action] − π(·|s)   (score function)

Actor 업데이트:
  θ[s, a] += lr · δ_t · ∇ log π(a|s)
```

#### 초기화 및 훈련 설정

```python
# fee 비례 BUY 선호 초기화 (cold-start 수수료 장벽 완화)
theta = np.zeros((4, 2))
theta[1, 1] = max(0.05, fee_rate * 30)   # EMA아래+상승: 미세 BUY 선호
theta[2, 1] = max(0.10, fee_rate * 50)   # EMA위+하락:  BUY 선호
theta[3, 1] = max(0.20, fee_rate * 80)   # EMA위+상승:  BUY 선호 강화
V = np.zeros(4)                          # Critic 가치함수

# 탐험: 상수 epsilon-greedy
if np.random.rand() < epsilon:
    action = np.random.randint(0, 2)
else:
    action = np.random.choice([0, 1], p=softmax(theta[state]))

# 엔트로피 정규화: Buy&Hold 고착 방지, 정책 다양성 유지
entropy = -sum(probs * log(probs + 1e-10))
r_eff   = reward + 0.02 * entropy

# 단일 학습률 (actor/critic 동일 lr)
V[state]        += lr · δ_t
theta[state, a] += lr · δ_t · grad_log_pi
```

---

### 2.2 Vanilla RL — Q-Learning (비교 기준선)

**파일:** `common/base_agent.py` — `_train_qlearning_vanilla()`

2-상태 Tabular Q-Learning으로 구현된 비교 기준선 에이전트다.

#### 벨만 방정식 (Q-Learning)

```
Q*(s, a) = E[r + γ · max_{a'} Q*(s', a') | s, a]

TD 업데이트:
  Q(s, a) += lr · [r + γ · max_{a'} Q(s', a') − Q(s, a)]
             ← Bellman Residual 최소화
```

#### Q-테이블 초기화 및 CASH 편향 해소

```python
q_table = np.zeros((2, 2))
q_table[:, 1] = max(fee_rate * 50, 0.05)  # fee 비례 BUY 선호 (수수료가 높을수록 강화)
```

> fee_rate=0.001(미국) → Q[:,1]=0.05, fee_rate=0.0023(국내) → Q[:,1]=0.115

#### epsilon annealing

```python
for ep in range(train_episodes):
    _eps = epsilon * max(1.0, 2.0 - ep / (train_episodes - 1))
    # 탐험: 2ε (ep=0) → ε (ep=마지막)  — 초반 강탐험, 후반 기본 탐험 유지
    prev_action = 1   # BUY 시작 고정 (에피소드 첫 step 수수료 편향 제거)
```

#### 훈련 후 보정 — 상대 우위 하한 (improve 4-7)

```python
# bull state: BUY가 반드시 CASH보다 우세하도록 보정
# 문제: Q[1,CASH]가 학습으로 높아지면 절대 하한(0.002)을 초과 → BUY 영구 패배
# 해결: 상대 우위 하한으로 고변동성(TSLA) 0% 고착 방지
q_table[1, 1] = max(float(q_table[1, 1]), float(q_table[1, 0]) + 0.001)
```

> **이전 방식 (improve 3-2-6)**: `Q[1,1] = max(Q[1,1], 0.002)` — 절대 하한
> Q[1,CASH]가 학습으로 0.05 이상이 되면 여전히 CASH 선택 → TSLA 0% 지속
>
> **현재 방식 (improve 4-7)**: `Q[1,1] = max(Q[1,1], Q[1,0] + 0.001)` — 상대 우위
> Q[1,CASH]가 얼마나 크든 bull state에서 BUY 우세 보장

---

### 2.3 상태 공간 설계

```
STATIC RL — 4개 상태 (MDP):

  State 0: 하락 (ret ≤ 0) + EMA 아래 (price < EMA_10)   → 가장 보수적
  State 1: 상승 (ret > 0)  + EMA 아래                    → 주의 단계
  State 2: 하락             + EMA 위 (price ≥ EMA_10)    → 중립
  State 3: 상승             + EMA 위                     → 가장 강한 매수 신호

Vanilla RL — 2개 상태:

  State 0: 하락 (ret ≤ 0)
  State 1: 상승 (ret > 0)
```

EMA_10 (10일/봉 지수이동평균)은 `data_loader.py`에서 계산된다.

```python
df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
```

---

### 2.4 행동 및 보상 함수

| 행동 | 코드 | 의미 |
|------|------|------|
| CASH | 0 | 현금 보유 (수익률 0%) |
| BUY | 1 | 매수·보유 (당일 수익률 반영) |

> **SELL 액션 없음**: 행동 공간은 BUY(1)/CASH(0) 2개뿐이다.
> BUY→CASH 전환이 암묵적 청산(매도)이며, 청산 시 수수료는 부과되지 않는다.
> 수수료는 **CASH→BUY 진입 시에만** 1회 부과된다.

```python
# 학습 및 평가 단계 보상 (클리핑 없음)
fee    = fee_rate if (action == BUY and prev_action == CASH) else 0
reward = daily_return − fee   # 실제 수익률 그대로

# 포트폴리오 누적 수익
current_capital     *= (1 + reward)
cumulative_return_% = (current_capital − 1) × 100
```

---

## 3. 워크포워드 검증 — Train/Test 분리

**파일:** `common/base_agent.py` — `run_rl_simulation()`, `run_rl_simulation_with_log()`

### Trading Days vs Train Episodes 구분

두 파라미터는 용도가 다르며 독립적으로 설정한다.

| 파라미터 | 의미 | 기본값 |
|----------|------|--------|
| Trading Days (`n_bars`) | yfinance에서 가져올 데이터 봉 수 (데이터 창 크기) | 500 (일봉 약 2년) |
| Train Episodes | 훈련 데이터를 반복 학습하는 횟수 (epoch 수) | 300 |

```
Trading Days = 500봉:  전체 데이터 창
                  │
      ┌───────────┴─────────────┐
      │                         │
  학습 구간 (70%)           평가 전체 구간 (100%)
  n_train = max(int(500×0.7), 20) = 350봉   500봉
                              │
                    ┌─────────┴──────────┐
                    │                    │
                인샘플 (70%)       OOS 30% ✅
                (0 ~ 350봉)      (350 ~ 500봉)

Train Episodes = 300:  위 350봉 데이터를 300번 반복 학습
```

```python
n_train = max(int(n_days * 0.7), 20)   # 최소 20봉 학습 보장

# 학습: 첫 70% 데이터만 사용
theta, _ = _train_actor_critic_static(
    returns[:n_train], prices[:n_train], emas[:n_train],
    lr, gamma, epsilon, train_episodes, n_train, fee_rate
)

# 평가: 전체 기간 (후반 30%가 진짜 OOS 검증)
for t in range(1, n_days):
    action = get_action(state)
    reward = (returns[t] if action == 1 else 0.0) − fee   # 클리핑 없음
    current_capital *= (1 + reward)
```

| 기간 | 일봉 500일 |
|------|-----------|
| 학습 구간 | 350일 (~1년 5개월) |
| 평가 전체 | 500일 (~2년) |
| OOS 구간 | 마지막 150일 (~6개월) |

### 워크포워드 구조적 한계 (KOSPI/KOSDAQ)

KOSPI·KOSDAQ는 학습 구간(2024~2025 상반기)이 횡보/하락이고, OOS 구간(2025 하반기~2026)에 급등이 집중된다. 이 경우 어떤 파라미터도 OOS 수익률을 추월하기 어렵다.

---

## 4. 하이퍼파라미터 탐색 — PG Actor-Critic Optimizer

**파일:** `common/heuristic.py` — `PGActorCriticOptimizer`

하이퍼파라미터 `(lr, gamma, epsilon_static, epsilon_vanilla)`를 자동 탐색하여
복합 Gap 기대값을 극대화한다.

### 복합 Gap 목표 함수

```python
gap_vs_market  = STATIC_final − Market_final          # 시장 초과수익 (60% 가중)
V_floor        = Market_final × 0.3                   # Vanilla 하한: 시장의 30%
V_adj          = max(Vanilla_final, V_floor)          # 역유인 방지 (Vanilla 의도적 0% 방지)
gap_vs_vanilla = STATIC_final − V_adj                 # Vanilla 대비 우위 (40% 가중)

composite_gap  = 0.6 × gap_vs_market + 0.4 × gap_vs_vanilla
```

> **역유인 방지 (improve 4-2)**: Vanilla가 0%일 때 gap이 최대 → 옵티마이저가 Vanilla를
> 의도적으로 망가뜨리는 구조를 V_floor로 차단. Vanilla ≥ Market×30%를 하한으로 보정.

### 이론 구조

```
탐색 공간: {lr, gamma, epsilon_static, epsilon_vanilla} → 정규화 공간 [0,1]^4

1. Policy (Actor, Gaussian):
   x = clip(μ + σ × ε, 0, 1),   ε ~ N(0, 1)
   → 다음 하이퍼파라미터 후보 제안

2. 기대값 (Expected Composite Gap):
   복수 평가 시드(_n_eval = 3~4)로 gap 측정 후 평균
   expected_gap = mean[composite_gap(seed_i) for seed_i in eval_seeds]

3. Advantage (REINFORCE with baseline):
   V      += value_alpha × (gap − V)          (Critic: EMA baseline 갱신)
   A       = gap − V                          (raw advantage)
   A_norm  = tanh(A / 10)                     ([−1,1] 정규화)

4. Actor 업데이트 (Policy Gradient):
   pg_dir = clip(Δ / σ, L2≤1)                (방향 벡터)
   μ      += lr_actor × A_norm × pg_dir

5. σ 자동 스케줄링:
   A > 0  →  σ × 0.96  (수렴: 좋은 방향 집중)
   A ≤ 0  →  σ × 1.04  (탐험: 더 넓게 재탐색)
```

### 탐색 페이즈

| 단계 | 조건 | 설명 |
|------|------|------|
| PG Exploring | step < n_iters / 4 | 광역 탐험으로 파라미터 공간 초기 파악 |
| PG Actor-Critic | σ_mean > 0.12 | Policy Gradient 업데이트로 유망 방향 탐색 |
| PG Converging | σ_mean ≤ 0.12 | 수렴 단계, 최적 파라미터 정밀 탐색 |

### 반복 횟수 계산

```python
n_iters = max(Sim_Min_Steps, Auto_Run_Count × Sim_Step_Mult)
          # 기본: max(30, 6 × 10) = 60 iterations
_n_eval = min(4, max(3, Auto_Run_Count // 2))
          # 기본: min(4, max(3, 3)) = 3 eval seeds
총 평가 = n_iters × _n_eval   # 기본: 180회 RL 평가
```

| UI 파라미터 | 기본값 | 역할 |
|------------|--------|------|
| Sim Min Steps | 30 | n_iters 하한 (최소 탐색 step 수) |
| Sim Step Mult. | 10 | Auto Run Count 배수 (n_iters = max(Min, Count×Mult)) |

### Optimizer 초기값

```python
optimizer = PGActorCriticOptimizer(
    bounds      = param_bounds,
    lr_actor    = 0.12,     # Actor 정책 업데이트 속도
    sigma_init  = 0.18,     # 초기 탐험 폭
    sigma_min   = 0.02,     # 최소 σ (정밀 수렴 단계)
    sigma_max   = 0.30,     # 최대 σ (재탐험 단계, improve 4-2에서 0.45→0.30)
    value_alpha = 0.25,     # Critic EMA 업데이트 속도
    seed        = base_seed,
)
```

---

## 5. 하이퍼파라미터 상세

### RL 학습 파라미터

| 파라미터 | 탐색 범위 | 역할 |
|----------|-----------|------|
| lr (α) | 0.005 ~ 0.10 | Actor / Q-Table 업데이트 속도 |
| gamma (γ) | 0.85 ~ 0.99 | 미래 보상 할인율 |
| epsilon_static (ε) | 0.01 ~ 0.25 | STATIC RL 탐험율 |
| v_epsilon | 0.01 ~ 0.25 | Vanilla RL 전용 탐험율 (독립 최적화) |

> **gamma 하한 0.85 이유**: gamma < 0.85는 지나치게 근시안적 정책을 유발한다.
> gamma ≥ 0.85는 최소 6~7일 이상의 보유 수익을 고려하여 더 현실적인 정책을 학습한다.

> **epsilon 상한 0.25 이유**: epsilon > 0.25는 경계값(0.01, 0.25) 고착 현상 발생.
> 탐험율이 너무 높으면 학습된 정책보다 랜덤에 가까워져 수렴이 느려진다.

### 시스템 파라미터

| 파라미터 | 기본값 | 역할 |
|----------|--------|------|
| Trading Days | 500 (일봉 약 2년) | yfinance 데이터 봉 수 |
| Train Episodes | 300 | 훈련 데이터 반복 학습 횟수 (epoch) |
| seed | 멤버별 상이 | 훈련 재현성 고정 |
| Auto Run Count | 6 | Run Evaluation 자동 반복 횟수 |
| Sim Min Steps | 30 | 시뮬레이션 최소 탐색 step |
| Sim Step Mult. | 10 | 시뮬레이션 step 배수 |
| Timeframe | 일봉 (1d) | 데이터 봉 단위 |
| fee_rate | 종목별 | 매수 진입 시 1회 수수료 |

### Fallback Parameters

사이드바의 Fallback Parameters 패널에서 체크박스로 선택한 파라미터만 전체 종목에 일괄 적용할 수 있다.

| 대상 파라미터 | 적용 방식 |
|-------------|---------|
| LR / Gamma / ε(S) / ε(V) / Days / Episodes / Seed | 체크 시 글로벌 기본값으로 대체 |
| Active Agents | 체크 시 전체 종목 에이전트 구성 통일 |
| Sim Min Steps / Sim Step Mult. | 체크 시 시뮬레이션 반복 횟수 일괄 변경 |

### CTPT 성향 코드

RL 파라미터 조합으로 에이전트 투자 성향을 3자리 코드로 분류한다.

```
1번째 자리: lr ≥ 0.01  → A(Aggressive),  < 0.01 → P(Passive)
2번째 자리: gamma ≥ 0.95 → L(Long-term), < 0.95 → S(Short-term)
3번째 자리: epsilon ≥ 0.10 → V(Volatile), < 0.10 → R(Reserved)
```

| 코드 | 성향 |
|------|------|
| ALV | 적응형 모험가 |
| ALR | 안정적 성장형 |
| ASV | 단기 모험형 |
| ASR | 단기 민첩형 |
| PLV | 유연한 장기형 |
| PLR | 신중한 장기형 |
| PSV | 탐색형 |
| PSR | 보수형 |

---

## 6. 랜덤 시드의 역할

### 훈련 시드 (Base Seed)

```python
np.random.seed(seed)  # 훈련 시작 전 고정
```

epsilon-greedy 탐험 경로를 고정하여 동일 시드에서 항상 동일한 훈련 궤적이 재현된다.

| 종목 | 시드 | 선택 근거 |
|------|------|-----------|
| SPY | 42 | 안정 지수에 적합한 수렴성 |
| QQQ | 137 | 기술주 고분산 환경에서 안정 수렴 확인 |
| KOSPI | 2024 | 국내 시장 리듬과 친화적인 연도 기반 시드 |
| KOSDAQ | 777 | 고변동성 시장, 탐험 다양성 확보 |
| NVDA | 314 | 수학적 다양성(π 근사), 반도체 고변동 환경 |
| TSLA | 99 | 넓은 탐험 범위, 최고 변동성 대응 |

### 복수 평가 시드

```python
_n_eval     = min(4, max(3, Auto_Run_Count // 2))   # 기본 3개 시드
_eval_seeds = [base_seed + j * 37 for j in range(_n_eval)]
expected_gap = mean([composite_gap(seed_i) for seed_i in _eval_seeds])
```

Simulation 탐색 시 동일 파라미터를 여러 시드로 평가하여 특정 시드의 우연에 의존하지 않는
일반화된 기대값을 산출한다.

### Trial 시드

```python
# ×37 소수 간격 (시드 독립성 강화, improve 4-1)
trial_seed = base_seed + (len(trials) + run_i) × 37
```

> 소수(prime) 간격 이유: np.random의 LCG 특성상 +1 간격 시드들은 내부 상태가 유사하여
> 학습 궤적도 거의 동일해진다. ×37 간격은 Trial마다 실질적으로 다른 탐험 경로를 보장한다.

---

## 7. 데이터 파이프라인

**파일:** `common/data_loader.py`

### 지원 봉 단위

| 봉 단위 | interval | 최대 기간 | 기본 Trading Days |
|---------|----------|-----------|-------------------|
| 15분봉 | 15m | 60일 | 80 |
| 1시간봉 | 1h | 730일 | 120 |
| 일봉 | 1d | 2년 | **500** |
| 주봉 | 1wk | 10년 | 105 |
| 월봉 | 1mo | 10년 | 24 |

### 전처리 흐름

```
yf.download()  →  (실패 시) yf.Ticker().history()
    │
    ↓
MultiIndex 컬럼 정리 + 중복 제거
    │
    ↓
인덱스 처리: 인트라데이(15m/1h) → datetime  │  일봉 이상 → .date
    │
    ↓
EMA_10  = Close.ewm(span=10, adjust=False).mean()
    │
    ↓
Daily_Return = Close.pct_change()
    │
    ↓
dropna() 최종 정리
```

`@st.cache_data(ttl=3600)` — 동일 티커 + 봉 단위 조합을 1시간 동안 캐싱한다.

---

## 8. 포트폴리오 평가 및 성과 지표

### 개별 종목 지표

| 지표 | 계산 방법 |
|------|-----------|
| Final Return (%) | 누적 수익률 배열의 마지막 값 |
| Alpha Gap (%p) | STATIC RL − Vanilla RL 최종 수익률 차이 |
| MDD (%) | min((wealth_index − running_peak) / running_peak) × 100 |
| Volatility | 누적 수익률 배열의 표준편차 |

### 팀 포트폴리오 비중 — Softmax 가중 배분

**파일:** `common/evaluator.py`

```python
score_i  = avg_return_i / (1 + abs(avg_mdd_i))   # 위험 조정 수익
weight_i = softmax(scores)[i]                      # 성과 비례 자본 배분
```

온도 파라미터(`temperature=1.0`): 낮을수록 최고 성과 멤버에 집중, 높을수록 균등 배분.

### Ghost Line (최적 파라미터 투영)

Simulation에서 발견된 최적 파라미터로 산출한 수익 곡선을 점선(Ghost)으로 현재 차트에
병렬 표시한다. 현재 파라미터와 최적 파라미터 간의 성과 차이를 직관적으로 비교할 수 있다.

---

## 9. UI 기능 상세

### 9.1 사이드바

| 기능 | 동작 |
|------|------|
| Eval. All | 전체 멤버·종목을 현재 파라미터로 순차 평가 |
| Simul. All | 전체 멤버·종목의 최적 파라미터 자동 탐색 후 저장 및 평가 실행 |
| All 적용 | 체크된 Fallback Parameters를 모든 종목에 일괄 적용 |
| 되돌리기 | 체크된 파라미터를 이전 상태로 복원 |
| System Status | Cloud/Local 환경 + GPU/CPU 상태 표시 (☁️ / 🖥️, ⚡ / 🔲) |

### 9.2 Run Evaluation

```
1. 현재 파라미터로 RL 에이전트 평가 실행
2. Trial History에 결과 추가
   trial_seed = base_seed + (n_accumulated + run_i) × 37
3. Auto Run Count만큼 자동 반복
4. 각 Trial의 최종 수익률, Alpha Gap, MDD를 통계 분석 패널에 표시
```

### 9.3 Simulation

```
1. PG Actor-Critic Optimizer가 n_iters 반복으로 하이퍼파라미터 탐색
   n_iters = max(Sim Min Steps, Auto Run Count × Sim Step Mult.)
2. 각 iteration마다 복수 시드(3~4)로 RL 평가
   → 복합 Gap = 0.6 × (STATIC−Market) + 0.4 × (STATIC−max(Vanilla, Market×0.3))
3. Policy Gradient로 탐색 정책 μ 업데이트, σ 자동 스케줄링
4. 수렴 차트 실시간 표시
   - 파라미터 정규화 추이 차트 (α/γ/ε_S/ε_V)
   - Alpha vs Market 수렴 차트 (목표선 +1%p/+5%p 표시)
5. 탐색 완료 후 저장 여부 선택
```

### 9.4 Agent Decision Analysis

- 좌측: STATIC RL의 BUY/CASH 행동 빈도 막대 차트
- 우측: 일별 행동 로그 테이블 (BUY: 파란색, CASH: 빨간색)

### 9.5 Trial History Statistical Analysis

- **Trial-by-Trial Return Progression**: 반복 평가별 수익률 추이 및 Mean/Max/Min 기준선
- **Return Distribution across Trials**: Vanilla RL과 STATIC RL의 박스 플롯 비교
- **Statistics Summary**: Vanilla/STATIC의 Mean(σ), Range를 항목별 표시
- **Trial 데이터 테이블**: 열 줄바꿈 헤더로 공간 최적화, 전체 열(Trial/Seed/Vanilla/STATIC/Market) 표시

### 9.6 팀 포트폴리오 대시보드

- All Members STATIC RL Cumulative Returns + Team Fund 차트
- 멤버별 성과 테이블: Member, Stocks, Persona(CTPT), Capital($), STATIC(%), Vanilla(%), Alpha Gap, MDD, Score, Weight%
- Team Fund: Softmax 가중 배분 기준 팀 전체 수익 곡선

---

## 10. 파일 구조

```
Task-Constrained-RL-ColdStart_YS_v1-1/
│
├── app.py                          Streamlit 메인 앱 (~1900줄)
│
├── common/
│   ├── base_agent.py               Actor-Critic / Q-Learning 훈련·평가
│   ├── heuristic.py                PGActorCriticOptimizer
│   ├── evaluator.py                MDD, Softmax 비중, CTPT 코드
│   ├── data_loader.py              yfinance 데이터 로드 (다봉, 캐시)
│   └── stock_registry.py           종목 정보 + 수수료 테이블
│
├── members/
│   ├── member_1/config.py          Member 1 — SPY    (seed=42,  최적 lr=0.0577)
│   ├── member_2/config.py          Member 2 — QQQ    (seed=137, 최적 lr=0.0627)
│   ├── member_3/config.py          Member 3 — KOSPI  (seed=2024,최적 lr=0.0227)
│   ├── member_4/config.py          Member 4 — KOSDAQ (seed=777, 최적 lr=0.0168)
│   ├── member_5/config.py          Member 5 — NVDA   (seed=314, 최적 lr=0.0497)
│   └── member_6/config.py          Member 6 — TSLA   (seed=99,  최적 lr=0.0539)
│
└── README.md
```

---

## 11. Simulation 단계별 연산 흐름

**파일:** `app.py` (sim_clicked 블록) + `common/heuristic.py` (PGActorCriticOptimizer)

### STEP 1 — 탐색 공간 및 반복 횟수 결정

```python
n_iters = max(eff_sim_min, l_auto_runs * eff_sim_mult)
          # eff_*: Fallback Parameters 체크 시 글로벌 기본값 적용

param_bounds = {
    "lr":        (0.005, 0.10),   # LR 하한 0.005 (극단적 저학습률 방지)
    "gamma":     (0.85,  0.99),   # 하한 0.85: 단기 편향 방지
    "epsilon":   (0.01,  0.25),   # 상한 0.25: 경계값 고착 방지
    "v_epsilon": (0.01,  0.25),
}
```

### STEP 2 — PGActorCriticOptimizer 초기화

```python
optimizer = PGActorCriticOptimizer(
    bounds      = param_bounds,
    lr_actor    = 0.12,
    sigma_init  = 0.18,
    sigma_min   = 0.02,
    sigma_max   = 0.30,     # 0.45→0.30: 전역 진동 방지 (improve 4-2)
    value_alpha = 0.25,
    seed        = base_seed,
)
```

### STEP 3 — 복수 평가 시드 준비

```python
_n_eval     = min(4, max(3, l_auto_runs // 2))   # 최소 3 보장
_eval_seeds = [base_seed + j * 37 for j in range(_n_eval)]
```

### STEP 4 — 탐색 페이즈 판정

```
_explore_end = max(6, n_iters // 4)

i < _explore_end  →  🔴 PG Exploring    (초기 광역 탐험)
σ_mean > 0.12     →  🟡 PG Actor-Critic  (정책 업데이트 중)
σ_mean ≤ 0.12     →  🟢 PG Converging    (수렴 단계)
```

### STEP 5 — Actor: 파라미터 후보 샘플링

```python
Δ     = rng.normal(0, σ)
x_new = clip(μ + Δ, 0, 1)

candidate["lr"]        = 0.005 + x_new[0] × (0.10 − 0.005)
candidate["gamma"]     = 0.85  + x_new[1] × (0.99 − 0.85)
candidate["epsilon"]   = 0.01  + x_new[2] × (0.25 − 0.01)
candidate["v_epsilon"] = 0.01  + x_new[3] × (0.25 − 0.01)
```

### STEP 6 — 복수 시드로 RL 에이전트 평가

```python
for seed_i in _eval_seeds:
    _, v_trace, s_trace, mkt_trace, ... = get_rl_data(...)
    gap_vs_market  = s_trace[-1] − mkt_trace[-1]
    V_floor        = mkt_trace[-1] × 0.3
    V_adj          = max(v_trace[-1], V_floor)       # 역유인 방지
    gap_vs_vanilla = s_trace[-1] − V_adj
    composite_gap  = 0.6 × gap_vs_market + 0.4 × gap_vs_vanilla
    gaps.append(composite_gap)
```

### STEP 7 ~ 12 — Critic → Actor → σ 스케줄링

```
[7]  expected_gap = mean(gaps)

[8]  V += value_alpha × (expected_gap − V)            # Critic EMA

[9]  A_norm = tanh((expected_gap − V) / 10)           # Advantage → [−1,1]

[10] pg_dir = Δ / σ  (if L2>1: 정규화)
     μ = clip(μ + lr_actor × A_norm × pg_dir, 0, 1)  # Actor

[11] A_norm > 0 → σ × 0.96 (수렴)
     A_norm ≤ 0 → σ × 1.04 (재탐험)

[12] best 갱신, 수렴 차트 실시간 업데이트
```

### 전체 흐름 요약

```
[Simulation 클릭]
      │
      ↓
STEP 1: n_iters = max(Sim_Min, AutoRun×Mult), param_bounds 결정
      │
      ↓
STEP 2: PGActorCriticOptimizer 초기화 (σ_max=0.30)
      │
      ↓
  ┌── for i in range(n_iters): ──────────────────────────────────┐
  │                                                               │
  │  STEP 4: 페이즈 판정 (Exploring / Actor-Critic / Converging) │
  │  STEP 5: μ + Δ 샘플링 → candidate 파라미터 (4차원)           │
  │  STEP 6: 복수 시드 RL 평가 (3~4 seeds)                       │
  │    • STATIC AC 훈련(첫 70%, train_eps회) → 전체 평가          │
  │    • Vanilla QL 훈련(첫 70%, train_eps회) → 전체 평가         │
  │    • composite_gap = 0.6×(S-M) + 0.4×(S-max(V, M×0.3))     │
  │  STEP 7~12: Critic/Advantage/Actor/σ 갱신                   │
  └───────────────────────────────────────────────────────────────┘
      │
      ↓
[n_iters 완료]
      │
      ↓
Simul. All 모드  →  best 파라미터 config.py 자동 저장 → Run Evaluation 실행
수동 모드        →  [저장 및 반영] / [반영 취소] 버튼 표시
```

---

## 12. 개선 이력

### improve 4-7 (현재) — Vanilla bull-state 상대 우위 하한

| # | 문제 | 원인 | 수정 내용 |
|---|------|------|-----------|
| V1 | TSLA Vanilla 0% 고착 | Q[1,CASH]가 학습으로 높아져 절대 하한(0.002)을 초과, BUY 영구 패배 | Q[1,BUY] ≥ Q[1,CASH]+0.001 상대 우위 하한 (base_agent.py) |
| V2 | 6종목 config 최적값 미반영 | 시뮬레이션 발견 파라미터가 session_state에만 저장, 재시작 시 소실 | 각 멤버 config.py에 lr/gamma/epsilon/v_epsilon 최적값 저장 |

### improve 4-5~4-6 — fee 비례 초기화 + 환경 감지 + 범용 개선

| # | 변경 내용 |
|---|-----------|
| 4-5-A | STATIC/Vanilla Q init fee 비례 (수수료 높을수록 BUY 학습 난이도 보정) |
| 4-5-B | `_IS_CLOUD` (HOME==/home/appuser) + `_HAS_CUDA` 환경 자동 감지 |
| 4-5-C | Cloud 환경 시 슬라이더 기본값 자동 축소 (auto_runs=3, 부하 감소) |
| 4-5-D | 사이드바 System Status 배지 (☁️/🖥️, ⚡ GPU/🔲 CPU) |
| 4-6-A | STATIC: 엔트로피 정규화 r_eff = r + 0.02·H(π) |
| 4-6-B | Vanilla: epsilon annealing 2ε→ε (초반 강탐험, 후반 기본 탐험) |
| 4-6-C | Vanilla: prev_action=1 (BUY 시작 고정, CASH 편향 완전 제거) |

### improve 4-1~4-4 — 시뮬레이션 안정화

| # | 변경 내용 |
|---|-----------|
| 4-1 | Trial seed 간격 ×13 → ×37 소수 간격 (시드 독립성 강화) |
| 4-1 | Vanilla Q init q[:,1]=max(fee×50, 0.05) (fee 비례 BUY 선호) |
| 4-2 | 역유인 제거: V_floor = Market×0.3 (Vanilla=0% 의도 방지) |
| 4-2 | LR 탐색 하한 0.001→0.005 (극단적 저학습률 방지) |
| 4-2 | sigma_max 0.45→0.30 (전역 진동 방지) |
| 4-2 | _n_eval 최소 3 보장 |
| 4-3 | epsilon 탐색 상한 0.5→0.25 (경계값 고착 방지) |
| 4-3 | _eval_seeds 간격 ×37 소수 강화 |
| 4-4 | Sim Min Steps / Sim Step Mult. UI 파라미터 추가 (n_iters 세밀 제어) |
| 4-4 | Sim Min Steps / Sim Step Mult. Fallback Parameters 지원 |

### improve 3-2 — Trading Days / Train Episodes 분리 + RL 구조 단순화

| # | 변경 내용 |
|---|-----------|
| E1 | `episodes` → Trading Days(n_bars)와 Train Episodes로 분리 |
| E2 | 일봉 기본 Trading Days: 80 → 500 (약 2년) |
| E3 | Train Episodes 슬라이더 추가 (기본값 300) |
| E4 | theta 초기화 편향 제거 → fee 비례 선호로 재설계 |
| E5 | 엔트로피 정규화 제거 후 재도입 (0.02·H(π) — improve 4-6) |
| E6 | lr_actor/lr_critic 분리 제거 (단일 lr) |
| E7 | 보상 클리핑 제거 (실제 수익률 직접 반영) |
| E8 | 워크포워드 검증: n_train = max(int(n_days×0.7), 20) |

### improve 2-7~2-9 — 알고리즘 전환 및 기초 구조

- Actor-Critic으로 알고리즘 전환 (기존 단순 Q-Learning → Policy Gradient 기반)
- BayesianOptimizer → PGActorCriticOptimizer 전환
- 복합 Gap 목표 함수 도입 (0.6×Market + 0.4×Vanilla)
- Run Eval bug: try/except/finally 큐 팝 보장
- found 기준: gap ≥ 5%p → gap ≥ 1%p (목표), gap ≥ 25%p (🏆)
- 테이블 우측 정렬, 열 헤더 줄바꿈 (전체 열 표시)
