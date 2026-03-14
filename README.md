# Chainers Master Fund — Task-Constrained RL Cold-Start

---

## 요약 보고서

### 프로젝트 개요

본 프로젝트는 **Task-Constrained RL Cold-Start** 조건에서 EMA 기반 4-상태 Actor-Critic 에이전트(STATIC RL)와 2-상태 Q-Learning 에이전트(Vanilla RL)의 성과를 비교하는 멀티 에이전트 트레이딩 시뮬레이터다. 6명의 팀원이 각자 담당 종목에 에이전트를 배치하여 팀 펀드 수익률을 측정한다.

**핵심 연구 질문**

사전 데이터 없이 Cold-Start 조건에서, EMA 기반 4-상태 Actor-Critic 에이전트는 2-상태 Q-Learning 에이전트 대비 얼마나 높은 누적 수익률을 달성하는가?

**Alpha Gap** = STATIC RL 최종 수익률 − Vanilla RL 최종 수익률

- Gap ≥ 1% : 목표 달성
- Gap ≥ 25% : 최고 달성 🏆

---

### 알고리즘 비교 (현재 기준)

| 항목 | STATIC RL | Vanilla RL |
|------|-----------|------------|
| 알고리즘 | Actor-Critic (Policy Gradient) | Tabular Q-Learning |
| 상태 수 | 4 (추세 × EMA 위치) | 2 (추세만) |
| 정책 표현 | Softmax 확률 정책 | argmax Q(s, a) |
| Baseline | TD Critic V(s) | 없음 |
| 초기화 | theta = 0 (편향 없음) | Q[:, 1] = 0.01 (BUY 미세 선호) |
| 엔트로피 정규화 | 없음 | 없음 |
| 학습률 분리 | 없음 (단일 lr) | 없음 |
| 보상 클리핑 | 없음 (실제 수익 반영) | 없음 |
| 탐험 | 상수 epsilon-greedy | 랜덤 시작 포지션 + 상수 epsilon |
| 훈련 데이터 | 전체의 첫 70% (워크포워드) | 전체의 첫 70% (워크포워드) |
| 역할 | 평가 대상 | 비교 기준선 |

---

### 팀 구성 및 담당 종목

| 멤버 | 담당 종목 | Ticker | 시드 |
|------|-----------|--------|------|
| Member 1 | S&P 500 ETF | SPY | 42 |
| Member 2 | Nasdaq 100 ETF | QQQ | 137 |
| Member 3 | KOSPI 지수 | ^KS11 | 2024 |
| Member 4 | KOSDAQ 지수 | ^KQ11 | 777 |
| Member 5 | NVIDIA | NVDA | 314 |
| Member 6 | Tesla | TSLA | 99 |

추가 지원 종목: GOOGL, MSFT, 삼성전자(005930.KS), SK하이닉스(000660.KS)

---

### 거래 수수료

| 시장 | 매수 | 매도 | 왕복 합계 |
|------|------|------|-----------|
| 미국 주식·ETF | 0.05% | 0.05% | 0.10% |
| 국내 주식·지수 | 0.015% | 0.215% | 0.23% |

---

### 주요 기능

- **Run Evaluation**: 현재 파라미터로 RL 에이전트를 평가하고 Trial History를 축적한다.
- **Simulation**: PG Actor-Critic Optimizer로 하이퍼파라미터를 자동 탐색하여 복합 Gap(Market 대비 60% + Vanilla 대비 40%)을 극대화한다.
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
 |
 +-- 팀 포트폴리오 대시보드
 |    +-- All Members 누적 수익 차트 + Team Fund
 |    +-- 멤버별 성과 테이블 (STATIC, Vanilla, Alpha Gap, MDD, Score, Weight%)
 |
 +-- 멤버별 탭 (6개)
      +-- 종목별 파라미터 패널
      |    +-- Timeframe / Trading Days / Train Episodes
      |    +-- LR, Gamma, STATIC ε, Vanilla ε
      |    +-- Frame Speed, Base Seed, Auto Run Count, Active Agents
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
           +-- Statistics Summary (Mean, Range)
           +-- Trial 데이터 테이블

common/
 +-- base_agent.py      RL 훈련 및 평가 (Actor-Critic / Q-Learning)
 +-- heuristic.py       하이퍼파라미터 탐색 (PGActorCriticOptimizer)
 +-- evaluator.py       성과 지표 계산 (MDD, Softmax 비중, CTPT 코드)
 +-- data_loader.py     yfinance 데이터 로드 (다봉 지원, 캐시 1시간)
 +-- stock_registry.py  종목 정보 및 수수료 테이블

members/member_N/
 +-- config.py          멤버별 담당 종목 + RL 하이퍼파라미터
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
보상 함수 R(s, a, s') = daily_return × 1[a=BUY] − fee
할인 인수 γ           = 0.93 (기본값, 단기 거래 최적값)
```

#### 벨만 최적 방정식 (Policy Gradient 기반)

```
V*(s) = max_π E[Σ γ^t · r_t | s_0 = s, π]

온라인 TD(0) 근사:
  δ_t = r_t + γ · V(s_{t+1}) − V(s_t)   ← TD 오차 (Advantage 근사)
  V(s_t) += lr · δ_t                     ← Critic 업데이트
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
theta = np.zeros((4, 2))   # 편향 없는 초기화 — 학습으로만 정책 형성
V     = np.zeros(4)        # Critic 가치함수

# 탐험: 상수 epsilon-greedy (어닐링 없음)
if np.random.rand() < epsilon:
    action = np.random.randint(0, 2)
else:
    action = np.random.choice([0, 1], p=softmax(theta[state]))

# 보상: 클리핑 없음 — 실제 시장 수익률 그대로 반영
fee    = fee_rate if (action == BUY and prev_action == CASH) else 0
reward = daily_return − fee   # 클리핑 없음

# 단일 학습률 (actor/critic 분리 없음)
V[state]        += lr · δ_t
theta[state, a] += lr · δ_t · grad_log_pi
```

> **설계 원칙**: 초기화 편향, 엔트로피 정규화, lr 분리, 보상 클리핑 등 복잡한 장치를
> 모두 제거하여 파라미터 변화가 결과에 직접·투명하게 반영되도록 했다.
> 추가 기법은 학습 불안정성을 야기할 수 있으며, 현재의 단순한 구조가
> Cold-Start 조건에서 더 안정적인 수렴을 보인다.

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
q_table[:, 1] = 0.01   # BUY 초기값 = 0.01 (CASH 편향 강제 해소)
```

> **CASH 고착 문제**: Q[CASH]=0 은 안전 하한이지만, 초기 BUY 시도 시 수수료
> 패널티로 Q[BUY]가 즉시 음수가 되면 이후 BUY를 영구 회피하게 된다.
> 0.01 초기값은 수수료(0.1%)보다 10배 크므로 초기 BUY 탐험을 보장한다.

#### 에피소드 시작 포지션 랜덤화

```python
for ep in range(train_episodes):
    prev_action = int(np.random.randint(0, 2))  # CASH 또는 BUY 랜덤 시작
```

> **문제**: `prev_action = 0` (항상 CASH)으로 시작하면 300 에피소드 × 매 에피소드 첫
> BUY마다 수수료가 부과된다. 이 300번의 패널티가 Q[BUY]를 체계적으로 낮춰 항상
> 직선(CASH 0%) 수렴을 유발한다.
>
> **수정**: 랜덤 시작 포지션으로 수수료 발생을 평균 50%로 줄여 편향을 해소한다.

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
fee        = fee_rate if (action == BUY and prev_action == CASH) else 0
reward     = daily_return − fee   # 실제 수익률 그대로

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
  n_train = int(500 × 0.7) = 350봉    500봉
                              │
                    ┌─────────┴──────────┐
                    │                    │
                인샘플 (70%)       OOS 30% ✅
                (0 ~ 350봉)      (350 ~ 500봉)

Train Episodes = 300:  위 350봉 데이터를 300번 반복 학습
```

```python
n_train = max(int(n_days * 0.7), 20)   # 최소 20봉 학습 보장

# 학습: 첫 70% 데이터만 사용, train_episodes 횟수 반복
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

---

## 4. 하이퍼파라미터 탐색 — PG Actor-Critic Optimizer

**파일:** `common/heuristic.py` — `PGActorCriticOptimizer`

하이퍼파라미터 `(lr, gamma, epsilon_static, epsilon_vanilla)`를 자동 탐색하여
복합 Gap 기대값을 극대화한다.

### 복합 Gap 목표 함수

```python
gap_vs_market  = STATIC_final − Market_final   # 시장 초과수익
gap_vs_vanilla = STATIC_final − Vanilla_final  # Vanilla 대비 우위 (Portfolio Alpha)

composite_gap  = 0.6 × gap_vs_market + 0.4 × gap_vs_vanilla
```

> 0.6/0.4 가중치 이유: Market benchmark를 더 중시하면서도(60%) Portfolio Alpha 지표
> (STATIC-Vanilla)와 최적화 방향을 정렬(40%)한다.

### 이론 구조

```
탐색 공간: {lr, gamma, epsilon_static, epsilon_vanilla} → 정규화 공간 [0,1]^4

1. Policy (Actor, Gaussian):
   x = clip(μ + σ × ε, 0, 1),   ε ~ N(0, 1)
   → 다음 하이퍼파라미터 후보 제안

2. 기대값 (Expected Composite Gap):
   복수 평가 시드(_n_eval = 2~3)로 gap 측정 후 평균
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
n_iters = max(20, Auto_Run_Count × 8)           # 기본: 6 × 8 = 48 iterations
_n_eval = min(3, max(2, Auto_Run_Count // 3))    # 기본: 2 eval seeds
총 평가 = n_iters × _n_eval                      # 기본: 96회 RL 평가
```

---

## 5. 하이퍼파라미터 상세

### RL 학습 파라미터

| 파라미터 | 기본값 | 탐색 범위 | 역할 |
|----------|--------|-----------|------|
| lr (α) | 0.03 | 0.001 ~ 0.1 | Actor / Q-Table 업데이트 속도 |
| gamma (γ) | 0.93 | **0.85 ~ 0.99** | 미래 보상 할인율 |
| epsilon (ε) | 0.15 | 0.01 ~ 0.5 | STATIC RL 탐험율 |
| v_epsilon | = ε | 0.01 ~ 0.5 | Vanilla RL 전용 탐험율 (독립 최적화) |

> **gamma 하한 0.85 이유**: gamma < 0.85는 지나치게 근시안적 정책을 유발한다.
> 일봉 기준으로 gamma=0.74(이전 최적값)는 약 3일 후의 수익만 고려하여
> 진입 수수료 대비 수익성이 낮게 평가되고 Buy-and-Hold보다 열등한 정책을 찾는다.
> gamma ≥ 0.85는 최소 6~7일 이상의 보유 수익을 고려하여 더 현실적인 정책을 학습한다.

`gamma = 0.93`을 기본값으로 사용하는 이유: 일간 단기 거래에서 `gamma = 0.98`은 지나치게
먼 미래를 고려하여 TD 오차에 노이즈가 증가한다. `gamma = 0.93`은 단기 피드백을
효과적으로 반영한다.

### 시스템 파라미터

| 파라미터 | 기본값 | 역할 |
|----------|--------|------|
| Trading Days | **500** (일봉 약 2년) | yfinance 데이터 봉 수 (데이터 창 크기) |
| Train Episodes | **300** | 훈련 데이터 반복 학습 횟수 (epoch) |
| seed | 멤버별 상이 | 훈련 재현성 고정 |
| Auto Run Count | 6 | Run Evaluation 자동 반복 횟수 |
| Timeframe | 일봉 (1d) | 데이터 봉 단위 (15분봉 ~ 월봉) |
| fee_rate | 종목별 | 매수 진입 시 1회 수수료 부과율 |

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
_eval_seeds  = [base_seed + j for j in range(_n_eval)]
expected_gap = mean([composite_gap(seed_i) for seed_i in _eval_seeds])
```

Simulation 탐색 시 동일 파라미터를 여러 시드로 평가하여 특정 시드의 우연에 의존하지 않는
일반화된 기대값을 산출한다.

### Trial 시드

```python
# ×37 소수 간격 (시드 독립성 강화)
trial_seed = base_seed + (len(trials) + run_i) × 37
```

> 소수(prime) 간격을 사용하는 이유: np.random의 LCG 특성상 +1 간격 시드들은
> 내부 상태가 매우 유사하여 학습 궤적도 거의 동일해진다.
> ×37 간격은 Trial마다 실질적으로 다른 탐험 경로를 보장한다.

```
예시 (seed=42, 6회 Trial):
  기존 ×13: 42, 55, 68,  81,  94,  107  (상관성 높음)
  현재 ×37: 42, 79, 116, 153, 190, 227  (더 독립적 분포)
```

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
| Alpha Gap (%) | STATIC RL − Vanilla RL 최종 수익률 차이 |
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

### 9.1 사이드바 버튼

| 버튼 | 동작 |
|------|------|
| Eval. All | 전체 멤버·종목을 현재 파라미터로 순차 평가 |
| Simul. All | 전체 멤버·종목의 최적 파라미터 자동 탐색 후 저장 및 평가 실행 |
| All 적용 | 체크된 Fallback Parameters를 모든 종목에 일괄 적용 |
| 되돌리기 | 체크된 파라미터를 이전 상태로 복원 |

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
2. 각 iteration마다 복수 시드로 RL 평가
   → 복합 Gap = 0.6 × (STATIC−Market) + 0.4 × (STATIC−Vanilla)
3. Policy Gradient로 탐색 정책 μ 업데이트, σ 자동 스케줄링
4. 수렴 차트 실시간 표시 (파라미터 정규화 추이 + Gap 수렴 추이)
5. 탐색 완료 후 저장 여부 선택
```

### 9.4 Agent Decision Analysis

- 좌측: STATIC RL의 BUY/CASH 행동 빈도 막대 차트
- 우측: 일별 행동 로그 테이블 (BUY: 파란색, CASH: 빨간색)

### 9.5 Trial History Statistical Analysis

- **Trial-by-Trial Return Progression**: 반복 평가별 수익률 추이 및 Mean/Max/Min 기준선
- **Return Distribution across Trials**: Vanilla RL과 STATIC RL의 박스 플롯 비교
- **Statistics Summary**: Vanilla/STATIC의 Mean(σ), Range를 항목별 표시
- **Trial 데이터 테이블**: Trial, Seed, Vanilla Final %, STATIC Final %, Market Final %

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
│   ├── member_1/config.py          Member 1 — SPY    (seed=42)
│   ├── member_2/config.py          Member 2 — QQQ    (seed=137)
│   ├── member_3/config.py          Member 3 — KOSPI  (seed=2024)
│   ├── member_4/config.py          Member 4 — KOSDAQ (seed=777)
│   ├── member_5/config.py          Member 5 — NVDA   (seed=314)
│   └── member_6/config.py          Member 6 — TSLA   (seed=99)
│
└── README.md
```

---

## 11. Simulation 단계별 연산 흐름

**파일:** `app.py` (sim_clicked 블록) + `common/heuristic.py` (PGActorCriticOptimizer)

### STEP 1 — 탐색 공간 및 반복 횟수 결정

```python
n_iters = max(20, Auto_Run_Count × 8)   # 예: 6 × 8 = 48

param_bounds = {
    "lr":        (0.001, 0.1),
    "gamma":     (0.85,  0.99),   # 하한 0.85: 단기 편향 방지
    "epsilon":   (0.01,  0.5),
    "v_epsilon": (0.01,  0.5),
}
```

### STEP 2 — PGActorCriticOptimizer 초기화

```python
optimizer = PGActorCriticOptimizer(
    bounds      = param_bounds,
    lr_actor    = 0.12,
    sigma_init  = 0.18,     # 초기 탐험 폭
    sigma_min   = 0.02,     # 최소 σ (정밀 수렴 단계)
    sigma_max   = 0.45,     # 최대 σ (재탐험 단계)
    value_alpha = 0.25,     # Critic EMA 속도
    seed        = l_seed,
)
```

| 변수 | 초기값 | 의미 |
|------|--------|------|
| μ | [0.5, 0.5, 0.5, 0.5] | 정규화 공간 정책 평균 (탐색 중심점) |
| σ | [0.18, 0.18, 0.18, 0.18] | 파라미터별 탐험 폭 |
| V | 0.0 | Critic baseline |
| best_score | −∞ | 현재까지 최고 복합 Gap |

### STEP 3 — 복수 평가 시드 준비

```python
_n_eval     = min(3, max(2, Auto_Run_Count // 3))
_eval_seeds = [base_seed + j for j in range(_n_eval)]
```

### STEP 4 — 탐색 페이즈 판정

```
_explore_end = max(6, n_iters // 4)

i < _explore_end  →  PG Exploring    (초기 광역 탐험)
σ_mean > 0.12     →  PG Actor-Critic  (정책 업데이트 중)
σ_mean ≤ 0.12     →  PG Converging    (수렴 단계)
```

### STEP 5 — Actor: 파라미터 후보 샘플링

```python
Δ     = rng.normal(0, σ)
x_new = clip(μ + Δ, 0, 1)

candidate["lr"]        = 0.001 + x_new[0] × (0.1  − 0.001)
candidate["gamma"]     = 0.85  + x_new[1] × (0.99 − 0.85)   # ← 하한 0.85
candidate["epsilon"]   = 0.01  + x_new[2] × (0.5  − 0.01)
candidate["v_epsilon"] = 0.01  + x_new[3] × (0.5  − 0.01)
```

### STEP 6 — 복수 시드로 RL 에이전트 평가

```python
for seed_i in _eval_seeds:
    _, v_trace, s_trace, mkt_trace, ... = get_rl_data(
        ticker, candidate["lr"], candidate["gamma"],
        candidate["epsilon"], n_bars, train_episodes, seed_i,
        v_epsilon=candidate["v_epsilon"], ...
    )
    gap_vs_market  = s_trace[-1] − mkt_trace[-1]
    gap_vs_vanilla = s_trace[-1] − v_trace[-1]
    composite_gap  = 0.6 × gap_vs_market + 0.4 × gap_vs_vanilla
    gaps.append(composite_gap)
```

`get_rl_data` 내부 동작:

```
(1) 주가 데이터 로드 (data_loader.fetch_stock_data, 캐시 TTL=1h)
    → n_bars(Trading Days)봉 취득

(2) n_train = max(int(n_days × 0.7), 20)   ← 워크포워드 분리

(3) STATIC RL 훈련: Actor-Critic, train_episodes 에피소드
    - 첫 70% 데이터(returns[:n_train])로만 학습
    - 단일 lr (actor/critic 분리 없음)
    - 상수 epsilon-greedy (어닐링 없음)
    - 보상 클리핑 없음

(4) Vanilla RL 훈련: Q-Learning, train_episodes 에피소드
    - 첫 70% 데이터로만 학습
    - 랜덤 시작 포지션 (CASH 편향 해소)
    - Q[:, 1] = 0.01 초기화
    - 동일 lr, gamma / v_epsilon 적용

(5) 평가: 전체 기간 (0~n_days), 클리핑 없음
    - 학습된 정책으로 greedy 행동 선택
    - 실제 수익률로 누적 수익 계산
```

### STEP 7 ~ 12 — Critic 업데이트 → Actor 업데이트 → σ 스케줄링

```
[7] expected_gap = mean(gaps)   # 복수 시드 평균

[8] V += value_alpha × (expected_gap − V)   # Critic

[9] A_norm = tanh((expected_gap − V) / 10)  # Advantage → [−1,1]

[10] pg_dir = Δ / σ  (if L2>1: 정규화)
     μ = clip(μ + lr_actor × A_norm × pg_dir, 0, 1)   # Actor

[11] A_norm > 0 → σ × 0.96 (수렴)
     A_norm ≤ 0 → σ × 1.04 (재탐험)

[12] best 갱신, 수렴 차트 실시간 업데이트
```

### 전체 흐름 요약

```
[Simulation 클릭]
      │
      ↓
STEP 1: n_iters, param_bounds 결정 (gamma 하한=0.85)
      │
      ↓
STEP 2: PGActorCriticOptimizer 초기화
      │
      ↓
  ┌── for i in range(n_iters): ──────────────────────────┐
  │                                                       │
  │  STEP 4: 페이즈 판정                                 │
  │  STEP 5: μ + Δ 샘플링 → candidate 파라미터           │
  │  STEP 6: 복수 시드 RL 평가                            │
  │    • STATIC AC 훈련(첫 70%, train_episodes회) → 평가  │
  │    • Vanilla QL 훈련(첫 70%, train_episodes회) → 평가 │
  │    • composite_gap = 0.6×(S-M) + 0.4×(S-V)          │
  │  STEP 7~12: Critic/Advantage/Actor/σ 갱신            │
  └───────────────────────────────────────────────────────┘
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

### improve 4-1 (현재)

**Vanilla RL CASH 고착 근본 수정 + 시뮬레이션 안정화**

| # | 문제 | 원인 | 수정 내용 |
|---|------|------|-----------|
| V1 | Vanilla 0% 직선 | 에피소드 항상 CASH 시작 → 300번 수수료 패널티 체계적 누적 | `prev_action = np.random.randint(0, 2)` 랜덤 시작 |
| V2 | Q-table 초기값 미흡 | `fee_rate=0.001` 너무 작아 첫 BUY 패널티 극복 불가 | `q_table[:, 1] = 0.01` (고정값, fee_rate 10배) |
| S1 | STATIC 이분화 (9.53% 집중) | gamma=0.74 수렴 → 단기 편향, OOS에서 수익 포기 | gamma 탐색 범위 (0.5→0.85, 0.99) 하한 상향 |
| S2 | Trial 시드 다양성 부족 | ×13 간격으로 상관된 정책 수렴 | Trial seed 간격 ×13 → ×37 소수 강화 |

### improve 3-2 (이전)

**Trading Days / Train Episodes 분리 + RL 구조 단순화**

| # | 변경 내용 |
|---|-----------|
| E1 | `episodes` 변수를 Trading Days(n_bars)와 Train Episodes로 분리 |
| E2 | 일봉 기본 Trading Days: 80 → 500 (약 2년) |
| E3 | Train Episodes 별도 슬라이더 추가 (기본값 100→300) |
| E4 | theta 초기화 편향 제거 (−1.5/−0.8/0.1/0.4 → 0으로 통일) |
| E5 | 엔트로피 정규화 제거 (entropy_coeff=0.005 삭제) |
| E6 | lr_actor/lr_critic 분리 제거 (단일 lr 적용) |
| E7 | eps_start 어닐링 제거 (상수 epsilon 유지) |
| E8 | 보상 클리핑 제거 (실제 수익률 직접 반영) |
| E9 | episodes 배수 오버라이드 제거 (max×3, max×2 → 직접 전달) |

### improve 2-7~2-9 (초기)

- Actor-Critic으로 알고리즘 전환 (기존 Q-Learning → Actor-Critic)
- BayesianOptimizer → PGActorCriticOptimizer 전환
- 복합 Gap 목표 함수 도입 (0.6×Market + 0.4×Vanilla)
- can_buy 하드 차단 → 소프트 theta 편향으로 전환
- Run Eval bug: try/except/finally 큐 팝 보장
- found 기준: gap ≥ 5% → gap ≥ 1% (목표), gap ≥ 25% (🏆)
