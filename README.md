# Chainers Master Fund — Task-Constrained RL Cold-Start

> **강화학습 기반 멀티 에이전트 주식 트레이딩 시뮬레이터**
> 6명의 팀원이 각자 담당 종목에 Actor-Critic 강화학습 에이전트를 배치하여 팀 펀드 수익률을 극대화합니다.

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [시스템 아키텍처](#2-시스템-아키텍처)
3. [담당 종목 및 멤버 구성](#3-담당-종목-및-멤버-구성)
4. [강화학습 알고리즘](#4-강화학습-알고리즘)
5. [기대값 산출 — PG Actor-Critic Optimizer](#5-기대값-산출--pg-actor-critic-optimizer)
6. [하이퍼파라미터 상세](#6-하이퍼파라미터-상세)
7. [랜덤 시드의 역할](#7-랜덤-시드의-역할)
8. [데이터 파이프라인](#8-데이터-파이프라인)
9. [포트폴리오 평가 및 성과 지표](#9-포트폴리오-평가-및-성과-지표)
10. [파일 구조](#10-파일-구조)
11. [설치 및 실행](#11-설치-및-실행)

---

## 1. 프로젝트 개요

이 프로젝트는 **Task-Constrained RL Cold-Start** 문제를 실전 주식 시장에 적용한 멀티 에이전트 트레이딩 시뮬레이터입니다.

**핵심 연구 질문:**
> 사전 데이터 없이 Cold-Start 조건에서, EMA(지수이동평균) 기반 상태 공간을 갖춘 Actor-Critic 에이전트(STATIC RL)는 단순 Q-Learning 에이전트(Vanilla RL)에 비해 얼마나 높은 누적 수익률을 달성할 수 있는가?

**Alpha Gap** = `STATIC RL 최종 수익률 − Vanilla RL 최종 수익률`

- Gap ≥ 1% → `✅ 목표 달성`
- Gap ≥ 25% → `🏆 최고 달성`

---

## 2. 시스템 아키텍처

```
app.py (Streamlit 웹 UI)
 ├── 사이드바: Eval. All / Simul. All / Fallback Parameters
 ├── 멤버별 탭
 │    ├── 누적 수익 차트 (Ghost Line 포함)
 │    ├── Agent Decision Analysis (행동 빈도 + 로그 테이블)
 │    └── Trial History: Statistical Analysis
 │         ├── Trial-by-Trial Return 추이 차트
 │         ├── Return Distribution (Box Plot)
 │         └── Statistics Summary
 └── 팀 포트폴리오 대시보드 (All Members 차트 + 성과 테이블)

common/
 ├── base_agent.py      ← RL 훈련 및 평가 (Actor-Critic / Q-Learning)
 ├── heuristic.py       ← 하이퍼파라미터 탐색 (PGActorCriticOptimizer / BayesianOptimizer)
 ├── evaluator.py       ← 성과 지표 계산 (MDD, Softmax 비중, CTPT 코드)
 ├── data_loader.py     ← yfinance 데이터 로드 (다봉 지원)
 └── stock_registry.py  ← 종목 정보 및 수수료 테이블

members/member_N/
 └── config.py          ← 멤버별 담당 종목 + RL 하이퍼파라미터
```

---

## 3. 담당 종목 및 멤버 구성

| 멤버 | 담당 종목 | Ticker | 시드 | 시장 특성 |
|------|-----------|--------|------|-----------|
| Member 1 | S&P 500 ETF | SPY | 42 | 미국 대형주 지수, 저변동성·안정형 |
| Member 2 | Nasdaq 100 ETF | QQQ | 137 | 미국 기술주 지수, 중간 변동성 |
| Member 3 | KOSPI 지수 | ^KS11 | 2024 | 한국 대형주 지수, 국내 시장 리듬 |
| Member 4 | KOSDAQ 지수 | ^KQ11 | 777 | 한국 소형·성장주, 고변동성 |
| Member 5 | NVIDIA | NVDA | 314 | 반도체·AI, 매우 고변동성 |
| Member 6 | Tesla | TSLA | 99 | EV·기술주, 최고 변동성 |

> 추가 지원 종목: GOOGL, MSFT, 삼성전자(005930.KS), SK하이닉스(000660.KS)

### 거래 수수료

| 시장 | 매수 | 매도 | 왕복 합계 |
|------|------|------|-----------|
| 미국 주식·ETF | 0.05% | 0.05% | 0.10% |
| 국내 주식·지수 | 0.015% | 0.215% (위탁+거래세) | 0.23% |

---

## 4. 강화학습 알고리즘

### 4.1 STATIC RL — Actor-Critic

**파일:** `common/base_agent.py` — `_train_actor_critic_static()`

STATIC RL은 **Policy Gradient Theorem + REINFORCE with baseline**을 온라인 TD 방식으로 구현합니다.

#### 수식

```
Actor (Softmax 정책):
  π_θ(a|s) = softmax(θ[s, :])
  ∇log π(a|s) = 1[a == action] − π(·|s)   ← score function

Critic (TD(0) 가치 함수 V):
  δ = r + γ·V(s') − V(s)                   ← TD 오차 = Advantage 근사

업데이트:
  V(s)     += lr_critic · δ                 ← Critic 업데이트
  θ[s, a]  += lr_actor  · δ · ∇log π(a|s) ← Actor 업데이트
```

#### Actor Logit 초기화 (소프트 편향)

하드 매수 금지 대신 초기 logit 편향으로 EMA 위치 선호도를 표현하고, Policy Gradient가 직접 학습합니다.

| 상태 | 설명 | θ[s, BUY] 초기값 |
|------|------|-------------------|
| 0 | 하락 + EMA 아래 | −1.5 (강한 비선호) |
| 1 | 상승 + EMA 아래 | −0.8 (비선호) |
| 2 | 하락 + EMA 위 | +0.5 (가능) |
| 3 | 상승 + EMA 위 | +1.2 (핵심 매수 신호) |

#### 훈련 설정

- **훈련 에피소드:** `max(episodes × 3, 500)` — Cold-Start 보완을 위한 충분한 반복 학습
- **Actor 학습률:** `lr × 1.0`
- **Critic 학습률:** `lr × 1.5` (안정적 baseline 추정)
- **ε 초기값:** `min(ε × 2.5, 0.60)` → 에피소드 진행에 따라 최종 ε까지 선형 감소

---

### 4.2 Vanilla RL — Q-Learning (비교 기준선)

**파일:** `common/base_agent.py` — `_train_qlearning_vanilla()`

Tabular Q-Learning으로 2상태(하락/상승)만 사용하는 단순 기준선 에이전트입니다.

```
Q(s, a) += lr · [r + γ · max_a' Q(s', a') − Q(s, a)]   ← Bellman 업데이트
```

- 상태 수: 2 (0: 하락, 1: 상승)
- EMA 정보 없음 → 시장 추세만 반영
- 초기 Q 편향: `Q[1, BUY] = 0.05` (상승 시 매수 약간 선호)
- ε 초기값: `min(ε × 3.0, 0.6)`

---

### 4.3 상태 공간 설계

```
STATIC RL — 4개 상태 (is_bull + 2 × is_above_ema):

  State 0: 하락 (ret ≤ 0) + EMA 아래 (price < EMA_10)
  State 1: 상승 (ret > 0) + EMA 아래
  State 2: 하락           + EMA 위 (price ≥ EMA_10)
  State 3: 상승           + EMA 위            ← 가장 강한 매수 신호

Vanilla RL — 2개 상태:

  State 0: 하락 (ret ≤ 0)
  State 1: 상승 (ret > 0)
```

**EMA_10** (10일 지수이동평균)은 `data_loader.py`에서 종가에 대해 계산됩니다.

```python
df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
```

---

### 4.4 행동 및 보상

| 행동 | 코드 | 의미 |
|------|------|------|
| CASH | 0 | 현금 보유 (수익률 0%) |
| BUY | 1 | 매수·보유 (당일 수익률 반영) |

**보상 계산:**
```python
fee = fee_rate if (action == BUY and prev_action == CASH) else 0  # 신규 진입 시 1회 부과
raw_reward = daily_return − fee
reward = clip(raw_reward, −0.025, +0.030)  # 극단값 클리핑으로 학습 안정화
```

**누적 수익률 (평가 단계):**
```python
current_capital *= (1 + reward)
cumulative_return_pct = (current_capital − 1) × 100
```

---

## 5. 기대값 산출 — PG Actor-Critic Optimizer

**파일:** `common/heuristic.py` — `PGActorCriticOptimizer`

하이퍼파라미터 `(lr, γ, ε_static, ε_vanilla)`를 자동 탐색하여 **Alpha Gap 기대값을 극대화**합니다.

### 이론 구조

```
탐색 공간: {lr, γ, ε_static, ε_vanilla} → 정규화 공간 [0,1]^4

1. Policy (Actor, Gaussian):
   x = clip(μ + σ·ε, 0, 1),   ε ~ N(0, 1)
   → 다음 하이퍼파라미터 후보 제안

2. 기대값 (Expected Gap):
   복수 평가 시드(_n_eval = 2~3)로 gap 측정 후 평균
   expected_gap = mean[STATIC_final(seed_i) − Vanilla_final(seed_i)]
   → 특정 시드에 과적합되지 않은 일반화 성능 추정

3. Advantage (REINFORCE with baseline):
   V += value_alpha · (gap − V)     ← Critic: EMA 방식 baseline 갱신
   A = gap − V                      ← 현재 gap이 기대치보다 좋은지 측정

4. Actor 업데이트 (Policy Gradient):
   μ += lr_actor · A · (Δ − μ) / σ²   ← 좋은 파라미터 방향으로 정책 평균 이동

5. σ 자동 스케줄링:
   A > 0 → σ × 0.96  (수렴: 좋은 방향 집중)
   A < 0 → σ × 1.04  (탐험: 더 넓게 재탐색)
```

### 탐색 페이즈

| 단계 | 조건 | 설명 |
|------|------|------|
| 🔴 PG Exploring | `step < n_iters / 4` | 넓은 탐험으로 파라미터 공간 초기 파악 |
| 🟡 PG Actor-Critic | `σ_mean > 0.12` | Actor-Critic 업데이트로 유망 방향 탐색 |
| 🟢 PG Converging | `σ_mean ≤ 0.12` | 수렴 단계, 최적 파라미터 정밀 탐색 |

### 반복 횟수 계산

```python
n_iters  = max(20, Auto_Run_Count × 8)          # 기본: 6 × 8 = 48 iterations
_n_eval  = min(3, max(2, Auto_Run_Count // 3))   # 기본: 2 eval seeds
총 평가  = n_iters × _n_eval                     # 기본: 96회 RL 평가
```

---

## 6. 하이퍼파라미터 상세

### RL 학습 파라미터

| 파라미터 | 기본값 | 탐색 범위 | 역할 |
|----------|--------|-----------|------|
| `lr` (α) | 0.03 | 0.001 ~ 0.1 | Actor / Q-Table 업데이트 속도 |
| `gamma` (γ) | 0.93 | 0.5 ~ 0.99 | 미래 보상 할인율. 낮을수록 단기 거래 최적화 |
| `epsilon` (ε) | 0.15 | 0.01 ~ 0.5 | STATIC RL ε-greedy 탐험율 |
| `v_epsilon` | `= ε` | 0.01 ~ 0.5 | Vanilla RL 전용 탐험율 (독립 최적화) |
| `episodes` | 80 | 10 ~ 500 | 평가 윈도우 크기 (데이터 봉 수) |

**gamma = 0.93 선택 근거:** 일간 단기 거래에서 `γ = 0.98`은 지나치게 먼 미래를 고려하여 TD 오차에 노이즈가 증가합니다. `γ = 0.93`은 단기 피드백을 효과적으로 반영합니다.

### 시스템 파라미터

| 파라미터 | 기본값 | 역할 |
|----------|--------|------|
| `seed` | 멤버별 상이 | 훈련 재현성 고정 |
| `Auto Run Count` | 6 | Run Evaluation 자동 반복 횟수 |
| `Timeframe` | 일봉(1d) | 데이터 봉 단위 (15분봉 ~ 월봉) |
| `fee_rate` | 종목별 | 매수 진입 시 1회 수수료 부과율 |

### CTPT 성향 코드

RL 파라미터 조합으로 에이전트 투자 성향을 3자리 코드로 분류합니다.

```
1번째 자리: lr ≥ 0.01 → A(Aggressive), < 0.01 → P(Passive)
2번째 자리: γ ≥ 0.95 → L(Long-term),  < 0.95 → S(Short-term)
3번째 자리: ε ≥ 0.10 → V(Volatile),   < 0.10 → R(Reserved)
```

| 코드 | 성향 | 색상 |
|------|------|------|
| ALV | 적응형 모험가 | 파랑 |
| ALR | 안정적 성장형 | 초록 |
| ASV | 단기 모험형 | 황금 |
| ASR | 단기 민첩형 | 빨강 |
| PLV | 유연한 장기형 | 분홍 |
| PLR | 신중한 장기형 | 남색 |
| PSV | 탐색형 | 주황 |
| PSR | 보수형 | 회색 |

---

## 7. 랜덤 시드의 역할

랜덤 시드는 **훈련 재현성**과 **평가 일반화** 두 가지 목적으로 사용됩니다.

### 훈련 시드 (Base Seed)

```python
np.random.seed(seed)  # 훈련 시작 전 고정
```

- ε-greedy 탐험 시 무작위 행동 선택 경로를 고정합니다.
- 동일 시드에서는 항상 동일한 훈련 궤적이 재현됩니다.
- **종목별로 다른 시드**를 사용하여 각 시장 특성에 맞는 탐험 패턴을 부여합니다.

| 종목 | 시드 | 선택 근거 |
|------|------|-----------|
| SPY | 42 | 클래식 시드, 안정 지수에 적합한 수렴성 |
| QQQ | 137 | 기술주 고분산 환경에서 안정 수렴 확인 |
| KOSPI | 2024 | 국내 시장 리듬과 친화적인 연도 기반 시드 |
| KOSDAQ | 777 | 고변동성 시장, 탐험 다양성 확보 |
| NVDA | 314 | 수학적 다양성(π), 반도체 고변동 환경 적합 |
| TSLA | 99 | 단순하고 넓은 탐험 범위, 최고 변동성 대응 |

### 복수 평가 시드 (Multi-Seed Evaluation)

```python
_eval_seeds = [base_seed + j for j in range(_n_eval)]
expected_gap = mean([gap(seed_i) for seed_i in _eval_seeds])
```

시뮬레이션 탐색 시 **동일 파라미터를 여러 시드로 평가**하여 특정 시드 우연에 의존하지 않는 일반화된 기대값을 산출합니다.

### Trial 시드 (Run Evaluation)

```python
trial_seed = base_seed + len(trials) + run_i
```

Run Evaluation 반복마다 다른 시드를 사용하여 **독립적인 Trial**을 생성합니다. 이렇게 축적된 Trial History를 통해 성과의 통계적 분포(평균, 분산, 범위)를 파악합니다.

---

## 8. 데이터 파이프라인

**파일:** `common/data_loader.py`

### 지원 봉 단위 (Timeframe)

| 봉 단위 | yfinance interval | 최대 조회 기간 | 비고 |
|---------|-------------------|----------------|------|
| 15분봉 | `15m` | 60일 | 단기 고빈도, datetime 인덱스 유지 |
| 1시간봉 | `1h` | 730일 (~2년) | 중기, datetime 인덱스 유지 |
| 일봉 | `1d` | 2년 | **기본값**, date 인덱스 |
| 주봉 | `1wk` | 10년 | 장기 추세 |
| 월봉 | `1mo` | 10년 | 장기 포트폴리오 |

### 전처리 흐름

```
yf.download() → (실패 시) yf.Ticker().history()
    ↓
MultiIndex 컬럼 정리 + 중복 제거
    ↓
인덱스 처리: 인트라데이(15m/1h) → datetime | 일봉 이상 → .date
    ↓
EMA_10 = Close.ewm(span=10, adjust=False).mean()
    ↓
Daily_Return = Close.pct_change()
    ↓
dropna() 최종 정리
```

### 캐시

`@st.cache_data(ttl=3600)` — 동일 티커+봉 단위 조합을 1시간 동안 캐싱하여 반복 호출 최소화.

---

## 9. 포트폴리오 평가 및 성과 지표

### 개별 종목 지표

| 지표 | 계산 방법 |
|------|-----------|
| Final Return (%) | 누적 수익률 배열의 마지막 값 |
| Alpha Gap (%) | STATIC RL − Vanilla RL 최종 수익률 차이 |
| MDD (%) | `min((wealth_index − running_peak) / running_peak) × 100` |
| Volatility | 누적 수익률 배열의 표준편차 |

### 팀 포트폴리오 비중 — Softmax 가중 배분

**파일:** `common/evaluator.py`

```python
score_i  = avg_return_i / (1 + |avg_mdd_i|)   # 위험 조정 수익
weight_i = softmax(scores)[i]                   # 성과 비례 자본 배분
```

온도 파라미터(`temperature=1.0`): 낮을수록 최고 성과 멤버에 집중 배분, 높을수록 균등 배분.

### Ghost Line (최적 파라미터 투영)

시뮬레이션에서 발견된 최적 파라미터로 산출한 수익 곡선을 **점선(Ghost)**으로 현재 차트에 함께 표시합니다. 현재 파라미터와 최적 파라미터 간의 성과 차이를 직관적으로 비교할 수 있습니다.

---

## 10. 파일 구조

```
Task-Constrained-RL-ColdStart_YS_v1-1/
│
├── app.py                          # Streamlit 메인 앱 (~1700줄)
│
├── common/
│   ├── base_agent.py               # Actor-Critic / Q-Learning 훈련·평가
│   ├── heuristic.py                # PGActorCriticOptimizer / BayesianOptimizer
│   ├── evaluator.py                # MDD, Softmax 비중, CTPT 코드
│   ├── data_loader.py              # yfinance 데이터 로드 (다봉)
│   └── stock_registry.py           # 종목 정보 + 수수료 테이블
│
├── members/
│   ├── member_1/config.py          # Member 1 — SPY    (seed=42)
│   ├── member_2/config.py          # Member 2 — QQQ    (seed=137)
│   ├── member_3/config.py          # Member 3 — KOSPI  (seed=2024)
│   ├── member_4/config.py          # Member 4 — KOSDAQ (seed=777)
│   ├── member_5/config.py          # Member 5 — NVDA   (seed=314)
│   └── member_6/config.py          # Member 6 — TSLA   (seed=99)
│
└── README.md
```

---

## 11. 설치 및 실행

### 요구사항

```
Python 3.9+
streamlit
yfinance
numpy
pandas
plotly
```

### 설치

```bash
pip install streamlit yfinance numpy pandas plotly
```

### 실행

```bash
streamlit run app.py
```

### 주요 워크플로우

```
1. [사이드바] Timeframe 및 Fallback Parameters 설정
2. [Run Evaluation] 현재 파라미터로 RL 에이전트 평가 (n회 반복, Trial History 축적)
3. [Simulation] PG Actor-Critic Optimizer로 최적 파라미터 자동 탐색
   → 탐색 완료 후 저장 여부 선택 (config.py에 영구 저장 가능)
4. [Eval. All] 전체 6개 종목 순차 평가 (현재 파라미터 그대로 적용)
5. [Simul. All] 전체 6개 종목 순차 시뮬레이션 + 최적 파라미터 자동 저장
6. [팀 대시보드] 전체 포트폴리오 성과 및 멤버별 기여도 확인
```

---

## 12. Simulation 단계별 상세 연산 흐름

**파일:** `app.py` (sim_clicked 블록) + `common/heuristic.py` (PGActorCriticOptimizer)

Simulation 버튼을 누르면 PG Actor-Critic Optimizer가 하이퍼파라미터 공간을 자동 탐색하여 Alpha Gap(STATIC RL − Vanilla RL 수익률 차이)을 극대화하는 최적 조합을 찾습니다. 전체 흐름은 아래 12단계로 구성됩니다.

---

### STEP 1 — 탐색 공간 정의 및 반복 횟수 결정

```python
n_iters = max(20, Auto_Run_Count × 8)   # 예: 6 × 8 = 48 iterations

param_bounds = {
    "lr":        (0.001, 0.1),   # Actor / Q-Table 학습률
    "gamma":     (0.5,   0.99),  # 미래 보상 할인율
    "epsilon":   (0.01,  0.5),   # STATIC RL ε-greedy 탐험율
    "v_epsilon": (0.01,  0.5),   # Vanilla RL 전용 탐험율
}
```

4개 파라미터가 탐색 대상이며, 모두 연속 실수 범위입니다.
반복 횟수(`n_iters`)는 Auto Run Count 슬라이더 값에 비례하여 결정됩니다.

---

### STEP 2 — PGActorCriticOptimizer 초기화

```python
optimizer = PGActorCriticOptimizer(
    bounds     = param_bounds,
    lr_actor   = 0.12,    # Actor μ 업데이트 속도
    sigma_init = 0.18,    # 초기 탐험 폭 σ (정규화 공간 [0,1] 기준)
    sigma_min  = 0.02,    # σ 최소값 (수렴 하한)
    sigma_max  = 0.45,    # σ 최대값 (탐험 상한)
    value_alpha= 0.25,    # Critic EMA 업데이트 속도
    seed       = l_seed,  # 멤버별 재현성 시드
)
```

내부 상태 초기화:

| 변수 | 초기값 | 의미 |
|------|--------|------|
| `μ` | `[0.5, 0.5, 0.5, 0.5]` | 정규화 공간에서 정책 평균 (탐색 중심점) |
| `σ` | `[0.18, 0.18, 0.18, 0.18]` | 각 파라미터 차원별 탐험 폭 |
| `V` | `0.0` | Critic 기준값 (baseline, 첫 평가 후 초기화) |
| `best_score` | `-∞` | 현재까지 최고 Gap |

---

### STEP 3 — 복수 평가 시드 준비

```python
_n_eval     = min(3, max(2, Auto_Run_Count // 3))   # 예: 6 // 3 = 2 seeds
_eval_seeds = [base_seed + j for j in range(_n_eval)]
# 예 (SPY, seed=42): [42, 43]
```

동일 파라미터를 **여러 시드로 평가**하여 특정 시드 우연에 의존하지 않는
일반화된 기대값(expected gap)을 산출합니다.

---

### STEP 4 — 탐색 페이즈 판정 (매 반복마다)

```
_explore_end = max(6, n_iters // 4)   # 예: 48 // 4 = 12

iteration < _explore_end  →  🔴 PG Exploring   (초기 광역 탐험)
σ_mean > 0.12            →  🟡 PG Actor-Critic  (정책 업데이트 중)
σ_mean ≤ 0.12            →  🟢 PG Converging    (수렴 단계)
```

페이즈는 수렴 차트의 색상과 진행률 바 레이블로 실시간 표시됩니다.

---

### STEP 5 — Actor: 파라미터 후보 샘플링 (`suggest_next`)

```python
# Gaussian 정책에서 변화량 Δ 샘플링
Δ = rng.normal(0, σ)                      # Δ ∈ R^4 (4개 파라미터)

# 정규화 공간 [0,1]에서 후보 계산
x_new = clip(μ + Δ, 0, 1)

# 원본 파라미터 공간으로 역정규화
candidate["lr"]        = 0.001 + x_new[0] × (0.1   − 0.001)
candidate["gamma"]     = 0.5   + x_new[1] × (0.99  − 0.5)
candidate["epsilon"]   = 0.01  + x_new[2] × (0.5   − 0.01)
candidate["v_epsilon"] = 0.01  + x_new[3] × (0.5   − 0.01)
```

정책 평균 `μ`를 중심으로 `σ` 폭의 정규분포에서 후보를 제안합니다.
`σ`가 클수록 넓게 탐험, 작을수록 `μ` 주변을 집중 탐색합니다.

---

### STEP 6 — 복수 시드로 RL 에이전트 평가

각 평가 시드에 대해 독립적으로 RL 훈련 및 평가를 실행합니다.

```python
for seed_i in _eval_seeds:
    vanilla_trace, static_trace = get_rl_data(
        ticker   = ticker,
        lr       = candidate["lr"],
        gamma    = candidate["gamma"],
        epsilon  = candidate["epsilon"],     # STATIC RL 탐험율
        episodes = l_epi,
        seed     = seed_i,
        v_epsilon= candidate["v_epsilon"],   # Vanilla RL 탐험율 (독립)
        fee_rate = fee_rate,
        interval = l_interval,               # 봉 단위 (15m / 1h / 1d / 1wk / 1mo)
    )
    gap_i = static_trace[-1] − vanilla_trace[-1]   # 해당 시드의 Alpha Gap
    gaps.append(gap_i)
```

`get_rl_data` 내부 동작:

```
① 주가 데이터 로드 (data_loader.fetch_stock_data, 캐시 TTL=1h)
② STATIC RL 훈련: Actor-Critic, max(episodes×3, 500) 에피소드
   - lr_actor = lr × 1.0
   - lr_critic = lr × 1.5
   - ε_start = min(ε × 2.5, 0.60) → 선형 감소 → 최종 ε
   - 4-상태 공간 (EMA 위/아래 × 상승/하락)
③ Vanilla RL 훈련: Q-Learning, episodes 에피소드
   - 동일 lr, gamma 적용
   - v_epsilon으로 독립 탐험
   - 2-상태 공간 (상승/하락)
④ STATIC / Vanilla 평가 (최근 episodes 봉 데이터로)
   - 각 봉에서 정책 π(a|s) 실행
   - 거래 수수료 적용 (신규 매수 진입 시 1회)
   - 누적 수익률 배열 반환
```

---

### STEP 7 — Alpha Gap 기대값 계산

```python
expected_gap = mean(gaps)                    # 복수 시드 평균 → 일반화 기대값
candidate["s_final"] = mean(static_finals)   # STATIC 평균 최종 수익률
candidate["v_final"] = mean(vanilla_finals)  # Vanilla 평균 최종 수익률
```

단일 시드 결과가 아닌 **복수 시드 평균**을 사용함으로써,
특정 시드의 운에 의한 과적합을 방지합니다.

---

### STEP 8 — Critic 업데이트 (baseline 갱신)

```python
# 첫 번째 관측: V = gap (직접 초기화)
# 이후: 지수이동평균(EMA)으로 점진적 갱신
V += value_alpha × (expected_gap − V)
# value_alpha = 0.25 → 현재 관측 25%, 이전 추정 75% 가중
```

`V`는 지금까지의 평균 기대값 수준을 추적하는 **기준선(baseline)** 역할을 합니다.
이를 통해 현재 파라미터가 평균 대비 얼마나 좋은지를 측정할 수 있습니다.

---

### STEP 9 — Advantage 계산 (REINFORCE with baseline)

```python
A = expected_gap − V
# A > 0: 현재 파라미터가 평균보다 좋음 → μ를 이 방향으로 이동
# A < 0: 현재 파라미터가 평균보다 나쁨 → μ를 이 방향에서 멀어지게 이동
```

기준선 차감(`baseline subtraction`)은 **분산을 줄여 학습 안정성**을 높입니다.
Gap 절대값이 아닌 "평균 대비 상대적 우위"로 업데이트하므로,
초반의 낮은 Gap에서도 의미 있는 방향 학습이 가능합니다.

---

### STEP 10 — Actor 업데이트 (Policy Gradient)

```python
# Score function (Gaussian 정책의 log 기울기)
∇log π(Δ|μ,σ) = Δ / σ²

# Actor μ 업데이트 (Policy Gradient Theorem 적용)
μ += lr_actor × A × (Δ / σ²)
μ = clip(μ, 0, 1)   # 정규화 공간 경계 유지

# 직관:
# A > 0이고 Δ > 0이면 → μ가 양의 방향으로 이동 (이 방향이 좋았으니 더 가자)
# A > 0이고 Δ < 0이면 → μ가 음의 방향으로 이동
# A < 0 이면 반대 방향으로 이동
```

정책 평균 `μ`가 좋은 파라미터 방향으로 점진적으로 이동합니다.
`lr_actor = 0.12`는 이 이동의 보폭을 결정합니다.

---

### STEP 11 — σ 자동 스케줄링 (탐험-수렴 균형)

```python
if A > 0:
    σ = max(σ × 0.96, σ_min)   # 좋은 방향 발견 → σ 축소 (수렴)
else:
    σ = min(σ × 1.04, σ_max)   # 나쁜 방향 → σ 확대 (재탐험)
```

| 상황 | σ 변화 | 효과 |
|------|--------|------|
| 연속으로 A > 0 | 단조 감소 → σ_min(0.02) | μ 주변 정밀 탐색 |
| 연속으로 A < 0 | 단조 증가 → σ_max(0.45) | 전역 재탐험 |
| A 부호 교번 | σ 진동 유지 | 탐험-수렴 균형 |

---

### STEP 12 — Best 갱신 및 수렴 차트 업데이트

```python
if expected_gap > best["gap"]:
    best = candidate.copy()              # 최고 파라미터 기록 갱신
    _best_s_trace = static_trace         # Ghost Line용 최고 수익 곡선 저장

gap_history.append(best["gap"])          # Best Gap 누적 추이 (우측 차트)
gap_iter_history.append(expected_gap)    # 각 iteration 기대값 (수렴 패턴)

# μ 정규화 추이 기록 (중간 차트)
for k in param_bounds:
    mu_hist_norm[k].append( (μ[k] − lo_k) / (hi_k − lo_k) )
```

수렴 차트는 **실시간** 갱신됩니다:

- **중간 차트 (Parameter Convergence):** 4개 파라미터의 정책 평균 `μ`가 [0-1]로 정규화된 값의 추이. 직선으로 수렴할수록 최적값이 안정적으로 결정된 것
- **우측 차트 (Expected Value → Target Convergence):** 매 iteration의 기대 Gap(회색)과 누적 Best Gap(파랑) 추이. 1% 점선(목표), 25% 점선(최고) 기준선 포함

---

### 전체 흐름 요약 다이어그램

```
[Simulation 클릭]
      │
      ▼
STEP 1: n_iters, param_bounds 결정
      │
      ▼
STEP 2: PGActorCriticOptimizer 초기화
        (μ=[0.5,…], σ=[0.18,…], V=0)
      │
      ▼
STEP 3: 복수 평가 시드 생성
        (_eval_seeds = [seed, seed+1, …])
      │
      ▼
      ┌─────────────────────────────────────┐
      │  for i in range(n_iters):           │
      │                                     │
      │  STEP 4: 페이즈 판정                │
      │    (Exploring / Actor-Critic /      │
      │     Converging)                     │
      │          │                          │
      │          ▼                          │
      │  STEP 5: suggest_next()             │
      │    Δ ~ N(0, σ)                      │
      │    candidate = denorm(clip(μ+Δ,0,1))│
      │          │                          │
      │          ▼                          │
      │  STEP 6: RL 평가 (복수 시드)        │
      │    STATIC AC훈련 → 평가             │
      │    Vanilla QL훈련 → 평가            │
      │    gap_i = S_final - V_final         │
      │          │                          │
      │          ▼                          │
      │  STEP 7: expected_gap = mean(gaps)  │
      │          │                          │
      │          ▼                          │
      │  STEP 8: V += α·(gap − V)  [Critic]│
      │          │                          │
      │          ▼                          │
      │  STEP 9: A = gap − V  [Advantage]  │
      │          │                          │
      │          ▼                          │
      │  STEP 10: μ += lr·A·(Δ/σ²) [Actor] │
      │          │                          │
      │  STEP 11: σ 스케줄링               │
      │    A>0 → σ×0.96 (수렴)             │
      │    A<0 → σ×1.04 (탐험)             │
      │          │                          │
      │  STEP 12: best 갱신, 차트 업데이트 │
      └─────────────────────────────────────┘
      │
      ▼
[n_iters 완료]
      │
      ▼
Simul. All 모드  →  best 파라미터 자동 저장 → config.py 갱신 → Run Evaluation 자동 실행
수동 모드        →  [저장 및 반영] / [반영 취소] 버튼 표시
```

---

### 파라미터 수렴 수식 전체 정리

```
────────────────────────────────────────────────────────
[표기]
  μ  : 정책 평균 (정규화 공간 [0,1]^4)
  σ  : 탐험 폭 (파라미터별 독립)
  Δ  : 이번 iteration의 μ 대비 편차 = x_new − μ (정규화 전)
  V  : Critic baseline (EMA 방식 평균 gap 추정)
  A  : Advantage = gap − V
  α_c: value_alpha = 0.25  (Critic 학습률)
  α_a: lr_actor   = 0.12   (Actor 학습률)
────────────────────────────────────────────────────────

[1] 샘플링
  Δ      ~ N(0, σ)
  x_new  = clip(μ + Δ, 0, 1)

[2] Critic (TD(0) style EMA)
  V ← V + α_c · (gap − V)

[3] Advantage
  A = gap − V

[4] Actor (Policy Gradient)
  μ ← clip(μ + α_a · A · Δ/σ², 0, 1)

[5] σ 스케줄
  σ ← max(σ · 0.96, σ_min)   if A > 0
  σ ← min(σ · 1.04, σ_max)   if A ≤ 0
────────────────────────────────────────────────────────
```

---

## 알고리즘 비교 요약

| 항목 | STATIC RL (Ours) | Vanilla RL (Baseline) |
|------|-----------------|----------------------|
| 알고리즘 | Actor-Critic (Policy Gradient) | Tabular Q-Learning |
| 상태 수 | 4 (추세 × EMA 위치) | 2 (추세만) |
| 정책 표현 | Softmax π_θ(a\|s) | argmax Q(s, a) |
| Baseline | TD Critic V(s) | 없음 |
| EMA 정보 활용 | 소프트 초기 편향 + 학습 | 미사용 |
| 훈련 에피소드 | `max(episodes × 3, 500)` | `episodes` |
| Cold-Start 전략 | logit 초기값 편향 | Q-Table 0 초기화 |
| 목표 | Alpha Gap ≥ 1% (이상: ≥ 25%) | 비교 기준선 |
