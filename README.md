# Chainers Master Fund — Task-Constrained RL Cold-Start

---

## 요약 보고서

### 프로젝트 개요

본 프로젝트는 Task-Constrained RL Cold-Start 조건에서 EMA 기반 상태 공간을 갖춘 Actor-Critic 에이전트(STATIC RL)와 단순 Q-Learning 에이전트(Vanilla RL)의 성과를 비교하는 멀티 에이전트 트레이딩 시뮬레이터다. 6명의 팀원이 각자 담당 종목에 에이전트를 배치하여 팀 펀드 수익률을 측정한다.

**핵심 연구 질문**

사전 데이터 없이 Cold-Start 조건에서, EMA 기반 4-상태 Actor-Critic 에이전트는 2-상태 Q-Learning 에이전트 대비 얼마나 높은 누적 수익률을 달성하는가.

**Alpha Gap** = STATIC RL 최종 수익률 - Vanilla RL 최종 수익률

- Gap >= 1% : 목표 달성
- Gap >= 25% : 최고 달성

---

### 알고리즘 비교

| 항목          | STATIC RL                      | Vanilla RL         |
| ------------- | ------------------------------ | ------------------ |
| 알고리즘      | Actor-Critic (Policy Gradient) | Tabular Q-Learning |
| 상태 수       | 4 (추세 x EMA 위치)            | 2 (추세만)         |
| 정책 표현     | Softmax 확률 정책              | argmax Q(s, a)     |
| Baseline      | TD Critic V(s)                 | 없음               |
| EMA 활용      | 초기 logit 편향 + 학습         | 미사용             |
| 훈련 에피소드 | max(episodes x 3, 500)         | episodes           |
| 역할          | 평가 대상                      | 비교 기준선        |

---

### 팀 구성 및 담당 종목

| 멤버     | 담당 종목      | Ticker | 시드 |
| -------- | -------------- | ------ | ---- |
| Member 1 | S&P 500 ETF    | SPY    | 42   |
| Member 2 | Nasdaq 100 ETF | QQQ    | 137  |
| Member 3 | KOSPI 지수     | ^KS11  | 2024 |
| Member 4 | KOSDAQ 지수    | ^KQ11  | 777  |
| Member 5 | NVIDIA         | NVDA   | 314  |
| Member 6 | Tesla          | TSLA   | 99   |

추가 지원 종목: GOOGL, MSFT, 삼성전자(005930.KS), SK하이닉스(000660.KS)

---

### 거래 수수료

| 시장           | 매수   | 매도   | 왕복 합계 |
| -------------- | ------ | ------ | --------- |
| 미국 주식·ETF  | 0.05%  | 0.05%  | 0.10%     |
| 국내 주식·지수 | 0.015% | 0.215% | 0.23%     |

---

### 주요 기능

- Run Evaluation: 현재 파라미터로 RL 에이전트를 평가하고 Trial History를 축적한다.
- Simulation: PG Actor-Critic Optimizer로 하이퍼파라미터를 자동 탐색하여 Alpha Gap을 극대화하는 최적 조합을 찾는다.
- Fallback Parameters: 체크박스로 선택한 파라미터만 모든 종목에 일괄 적용하거나 이전 상태로 복원한다.
- Trial History Statistical Analysis: 반복 평가 결과를 박스 플롯, 추이 차트, 통계 요약으로 표시한다.
- Ghost Line: Simulation에서 발견된 최적 파라미터의 수익 곡선을 현재 차트에 점선으로 병렬 표시한다.
- 팀 포트폴리오 대시보드: Softmax 가중 배분으로 전체 펀드 성과를 집계한다.

---

### 설치 및 실행

```
Python 3.9+
pip install streamlit yfinance numpy pandas plotly
streamlit run app.py
```

---

---

## 전체 상세 설명

### 목차

1. [시스템 아키텍처](#1-시스템-아키텍처)
2. [강화학습 알고리즘](#2-강화학습-알고리즘)
3. [하이퍼파라미터 탐색 — PG Actor-Critic Optimizer](#3-하이퍼파라미터-탐색--pg-actor-critic-optimizer)
4. [하이퍼파라미터 상세](#4-하이퍼파라미터-상세)
5. [랜덤 시드의 역할](#5-랜덤-시드의-역할)
6. [데이터 파이프라인](#6-데이터-파이프라인)
7. [포트폴리오 평가 및 성과 지표](#7-포트폴리오-평가-및-성과-지표)
8. [UI 기능 상세](#8-ui-기능-상세)
9. [파일 구조](#9-파일-구조)
10. [Simulation 단계별 연산 흐름](#10-simulation-단계별-연산-흐름)

---

## 1. 시스템 아키텍처

```
app.py  (Streamlit 웹 UI, 약 1900줄)
 |
 +-- 사이드바
 |    +-- Eval. All / Simul. All 버튼
 |    +-- Fallback Parameters (항목별 체크박스 + 일괄 적용/되돌리기)
 |
 +-- 팀 포트폴리오 대시보드
 |    +-- All Members 누적 수익 차트
 |    +-- 멤버별 성과 테이블 (STATIC, Vanilla, Alpha Gap, MDD, CTPT 코드)
 |
 +-- 멤버별 탭 (6개)
      +-- 종목별 파라미터 패널 (Timeframe, 에피소드, LR, Gamma, epsilon 등)
      +-- Run Evaluation / Simulation 버튼
      +-- 좌측 패널
      |    +-- 누적 수익 차트 (Ghost Line 포함)
      |    +-- Final Cumulative Return 카드 (Vanilla / STATIC / Market)
      |    +-- Agent Decision Analysis
      |         +-- STATIC Action Frequency 막대 차트
      |         +-- 일별 행동 로그 테이블 (우측 정렬, 2줄 헤더)
      +-- 우측 패널
           +-- Trial-by-Trial Return 추이 차트
           +-- Return Distribution 박스 플롯
           +-- Statistics Summary (Mean, Range 항목별 2줄 표시)
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

Policy Gradient Theorem과 REINFORCE with baseline을 온라인 TD 방식으로 구현한다.

#### 수식

```
Actor (Softmax 정책):
  pi_theta(a|s) = softmax(theta[s, :])
  grad log pi(a|s) = 1[a == action] - pi(·|s)   (score function)

Critic (TD(0) 가치 함수 V):
  delta = r + gamma * V(s') - V(s)               (TD 오차 = Advantage 근사)

업데이트:
  V(s)       += lr_critic * delta                 (Critic 업데이트)
  theta[s,a] += lr_actor  * delta * grad log pi   (Actor 업데이트)
```

#### Actor Logit 초기화 (소프트 편향)

하드 매수 금지 대신 초기 logit 편향으로 EMA 위치 선호도를 표현하고, Policy Gradient가 학습을 통해 조정한다.

| 상태 | 설명            | theta[s, BUY] 초기값                            |
| ---- | --------------- | ----------------------------------------------- |
| 0    | 하락 + EMA 아래 | -1.5 (강한 비선호)                              |
| 1    | 상승 + EMA 아래 | -0.8 (비선호)                                   |
| 2    | 하락 + EMA 위   | +0.3 (약한 선호, 강세장 buy-and-hold 고착 방지) |
| 3    | 상승 + EMA 위   | +0.7 (매수 선호)                                |

#### 훈련 설정

- 훈련 에피소드: `max(episodes x 3, 500)` — Cold-Start 보완을 위한 반복 학습
- Actor 학습률: `lr x 1.0`
- Critic 학습률: `lr x 1.5` (안정적 baseline 추정)
- 초기 탐험율: `min(epsilon x 2.5, 0.60)` — 에피소드 진행에 따라 최종 epsilon까지 선형 감소

---

### 2.2 Vanilla RL — Q-Learning (비교 기준선)

**파일:** `common/base_agent.py` — `_train_qlearning_vanilla()`

2-상태 Tabular Q-Learning으로 구현된 단순 비교 기준선 에이전트다.

```
Q(s, a) += lr * [r + gamma * max_a' Q(s', a') - Q(s, a)]   (Bellman 업데이트)
```

- 상태 수: 2 (0: 하락, 1: 상승)
- EMA 정보 미사용
- 초기 Q 편향: `Q[0, BUY] = 0.02` (하락 시 CASH 고착 방지), `Q[1, BUY] = 0.05` (상승 시 매수 선호)
- 초기 탐험율: `min(epsilon x 3.0, 0.6)`

---

### 2.3 상태 공간 설계

```
STATIC RL — 4개 상태:

  State 0: 하락 (ret <= 0) + EMA 아래 (price < EMA_10)
  State 1: 상승 (ret > 0)  + EMA 아래
  State 2: 하락            + EMA 위 (price >= EMA_10)
  State 3: 상승            + EMA 위   (가장 강한 매수 신호)

Vanilla RL — 2개 상태:

  State 0: 하락 (ret <= 0)
  State 1: 상승 (ret > 0)
```

EMA_10 (10일 지수이동평균)은 `data_loader.py`에서 계산된다.

```python
df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
```

---

### 2.4 행동 및 보상

| 행동 | 코드 | 의미                         |
| ---- | ---- | ---------------------------- |
| CASH | 0    | 현금 보유 (수익률 0%)        |
| BUY  | 1    | 매수·보유 (당일 수익률 반영) |

```python
fee        = fee_rate if (action == BUY and prev_action == CASH) else 0
raw_reward = daily_return - fee
reward     = clip(raw_reward, -0.025, +0.030)   # 극단값 클리핑

current_capital     *= (1 + reward)
cumulative_return_% = (current_capital - 1) * 100
```

신규 매수 진입 시 수수료를 1회 부과하며, 보상을 클리핑하여 학습 안정성을 확보한다.

---

## 3. 하이퍼파라미터 탐색 — PG Actor-Critic Optimizer

**파일:** `common/heuristic.py` — `PGActorCriticOptimizer`

하이퍼파라미터 `(lr, gamma, epsilon_static, epsilon_vanilla)`를 자동 탐색하여 Alpha Gap 기대값을 극대화한다.

### 이론 구조

```
탐색 공간: {lr, gamma, epsilon_static, epsilon_vanilla} -> 정규화 공간 [0,1]^4

1. Policy (Actor, Gaussian):
   x = clip(mu + sigma * eps, 0, 1),   eps ~ N(0, 1)
   -> 다음 하이퍼파라미터 후보 제안

2. 기대값 (Expected Gap):
   복수 평가 시드(_n_eval = 2~3)로 gap 측정 후 평균
   expected_gap = mean[STATIC_final(seed_i) - Vanilla_final(seed_i)]

3. Advantage (REINFORCE with baseline):
   V      += value_alpha * (gap - V)           (Critic: EMA 방식 baseline 갱신)
   A       = gap - V                           (raw advantage)
   A_norm  = tanh(A / 10)                      ([-1,1] 정규화, 10% gap ≈ tanh(1) 포화)

4. Actor 업데이트 (Policy Gradient):
   pg_dir = clip(Delta / sigma, L2≤1)          (방향 벡터, 스텝 크기 상한 보장)
   mu    += lr_actor * A_norm * pg_dir

5. sigma 자동 스케줄링:
   A > 0  ->  sigma * 0.96  (수렴: 좋은 방향 집중)
   A <= 0 ->  sigma * 1.04  (탐험: 더 넓게 재탐색)
```

### 탐색 페이즈

| 단계            | 조건               | 설명                                      |
| --------------- | ------------------ | ----------------------------------------- |
| PG Exploring    | step < n_iters / 4 | 광역 탐험으로 파라미터 공간 초기 파악     |
| PG Actor-Critic | sigma_mean > 0.12  | Policy Gradient 업데이트로 유망 방향 탐색 |
| PG Converging   | sigma_mean <= 0.12 | 수렴 단계, 최적 파라미터 정밀 탐색        |

### 반복 횟수 계산

```python
n_iters = max(20, Auto_Run_Count * 8)          # 기본: 6 * 8 = 48 iterations
_n_eval = min(3, max(2, Auto_Run_Count // 3))   # 기본: 2 eval seeds
총 평가 = n_iters * _n_eval                     # 기본: 96회 RL 평가
```

---

## 4. 하이퍼파라미터 상세

### RL 학습 파라미터

| 파라미터   | 기본값    | 탐색 범위   | 역할                                 |
| ---------- | --------- | ----------- | ------------------------------------ |
| lr (alpha) | 0.03      | 0.001 ~ 0.1 | Actor / Q-Table 업데이트 속도        |
| gamma      | 0.93      | 0.5 ~ 0.99  | 미래 보상 할인율                     |
| epsilon    | 0.15      | 0.01 ~ 0.5  | STATIC RL 탐험율                     |
| v_epsilon  | = epsilon | 0.01 ~ 0.5  | Vanilla RL 전용 탐험율 (독립 최적화) |
| episodes   | 80        | 10 ~ 500    | 평가 윈도우 크기 (데이터 봉 수)      |

`gamma = 0.93`을 사용하는 이유: 일간 단기 거래에서 `gamma = 0.98`은 지나치게 먼 미래를 고려하여 TD 오차에 노이즈가 증가한다. `gamma = 0.93`은 단기 피드백을 효과적으로 반영한다.

### 시스템 파라미터

| 파라미터       | 기본값      | 역할                           |
| -------------- | ----------- | ------------------------------ |
| seed           | 멤버별 상이 | 훈련 재현성 고정               |
| Auto Run Count | 6           | Run Evaluation 자동 반복 횟수  |
| Timeframe      | 일봉 (1d)   | 데이터 봉 단위 (15분봉 ~ 월봉) |
| fee_rate       | 종목별      | 매수 진입 시 1회 수수료 부과율 |

### CTPT 성향 코드

RL 파라미터 조합으로 에이전트 투자 성향을 3자리 코드로 분류한다.

```
1번째 자리: lr >= 0.01  -> A(Aggressive),  < 0.01 -> P(Passive)
2번째 자리: gamma >= 0.95 -> L(Long-term), < 0.95 -> S(Short-term)
3번째 자리: epsilon >= 0.10 -> V(Volatile), < 0.10 -> R(Reserved)
```

| 코드 | 성향          |
| ---- | ------------- |
| ALV  | 적응형 모험가 |
| ALR  | 안정적 성장형 |
| ASV  | 단기 모험형   |
| ASR  | 단기 민첩형   |
| PLV  | 유연한 장기형 |
| PLR  | 신중한 장기형 |
| PSV  | 탐색형        |
| PSR  | 보수형        |

---

## 5. 랜덤 시드의 역할

### 훈련 시드 (Base Seed)

```python
np.random.seed(seed)  # 훈련 시작 전 고정
```

epsilon-greedy 탐험 경로를 고정하여 동일 시드에서 항상 동일한 훈련 궤적이 재현된다. 종목별로 다른 시드를 사용하여 각 시장 특성에 맞는 탐험 패턴을 부여한다.

| 종목   | 시드 | 선택 근거                                  |
| ------ | ---- | ------------------------------------------ |
| SPY    | 42   | 안정 지수에 적합한 수렴성                  |
| QQQ    | 137  | 기술주 고분산 환경에서 안정 수렴 확인      |
| KOSPI  | 2024 | 국내 시장 리듬과 친화적인 연도 기반 시드   |
| KOSDAQ | 777  | 고변동성 시장, 탐험 다양성 확보            |
| NVDA   | 314  | 수학적 다양성(pi 근사), 반도체 고변동 환경 |
| TSLA   | 99   | 넓은 탐험 범위, 최고 변동성 대응           |

### 복수 평가 시드

```python
_eval_seeds  = [base_seed + j for j in range(_n_eval)]
expected_gap = mean([gap(seed_i) for seed_i in _eval_seeds])
```

Simulation 탐색 시 동일 파라미터를 여러 시드로 평가하여 특정 시드의 우연에 의존하지 않는 일반화된 기대값을 산출한다.

### Trial 시드

```python
trial_seed = base_seed + len(trials) + run_i
```

Run Evaluation 반복마다 다른 시드를 사용하여 독립적인 Trial을 생성한다. 축적된 Trial History로 성과의 통계적 분포(평균, 표준편차, 범위)를 파악한다.

---

## 6. 데이터 파이프라인

**파일:** `common/data_loader.py`

### 지원 봉 단위

| 봉 단위 | yfinance interval | 최대 조회 기간 |
| ------- | ----------------- | -------------- |
| 15분봉  | 15m               | 60일           |
| 1시간봉 | 1h                | 730일 (~2년)   |
| 일봉    | 1d                | 2년 (기본값)   |
| 주봉    | 1wk               | 10년           |
| 월봉    | 1mo               | 10년           |

### 전처리 흐름

```
yf.download()  ->  (실패 시) yf.Ticker().history()
    |
    v
MultiIndex 컬럼 정리 + 중복 제거
    |
    v
인덱스 처리: 인트라데이(15m/1h) -> datetime  |  일봉 이상 -> .date
    |
    v
EMA_10  = Close.ewm(span=10, adjust=False).mean()
    |
    v
Daily_Return = Close.pct_change()
    |
    v
dropna() 최종 정리
```

`@st.cache_data(ttl=3600)` — 동일 티커 + 봉 단위 조합을 1시간 동안 캐싱하여 반복 호출을 최소화한다.

---

## 7. 포트폴리오 평가 및 성과 지표

### 개별 종목 지표

| 지표             | 계산 방법                                                |
| ---------------- | -------------------------------------------------------- |
| Final Return (%) | 누적 수익률 배열의 마지막 값                             |
| Alpha Gap (%)    | STATIC RL - Vanilla RL 최종 수익률 차이                  |
| MDD (%)          | min((wealth_index - running_peak) / running_peak) \* 100 |
| Volatility       | 누적 수익률 배열의 표준편차                              |

### 팀 포트폴리오 비중 — Softmax 가중 배분

**파일:** `common/evaluator.py`

```python
score_i  = avg_return_i / (1 + abs(avg_mdd_i))   # 위험 조정 수익
weight_i = softmax(scores)[i]                      # 성과 비례 자본 배분
```

온도 파라미터(`temperature=1.0`): 낮을수록 최고 성과 멤버에 집중 배분하고, 높을수록 균등 배분한다.

### Ghost Line (최적 파라미터 투영)

Simulation에서 발견된 최적 파라미터로 산출한 수익 곡선을 점선(Ghost)으로 현재 차트에 병렬 표시한다. 현재 파라미터와 최적 파라미터 간의 성과 차이를 직관적으로 비교할 수 있다.

---

## 8. UI 기능 상세

### 8.1 사이드바 버튼

| 버튼                        | 동작                                                                      |
| --------------------------- | ------------------------------------------------------------------------- |
| Eval. All                   | 전체 멤버·종목을 현재 파라미터로 순차 평가                                |
| Simul. All                  | 전체 멤버·종목의 최적 파라미터 자동 탐색 후 config.py에 저장 및 평가 실행 |
| Fallback 적용 중 / All 적용 | 체크된 Fallback Parameters를 모든 종목에 일괄 적용                        |
| 되돌리기                    | 체크된 파라미터를 이전 상태로 복원                                        |

### 8.2 Fallback Parameters

사이드바 Fallback Parameters 창에서 항목별 체크박스를 통해 적용 범위를 제어한다.

**파라미터 목록 및 체크박스 동작:**

| 파라미터                | 설명                                            |
| ----------------------- | ----------------------------------------------- |
| Timeframe               | 데이터 봉 단위                                  |
| Trading Weeks/Days 등   | 평가 윈도우 크기                                |
| Frame Speed (sec)       | 차트 갱신 속도                                  |
| Base Seed               | 훈련 재현성 시드                                |
| Auto Run Count          | Run Evaluation 자동 반복 횟수                   |
| Active Agents           | 활성화할 에이전트 선택 (Vanilla RL / STATIC RL) |
| Learning Rate (alpha)   | RL 학습률                                       |
| Discount Factor (gamma) | 미래 보상 할인율                                |
| STATIC epsilon          | STATIC RL 탐험율                                |
| Vanilla epsilon         | Vanilla RL 전용 탐험율                          |

- 체크한 항목만 "All 적용" 버튼 클릭 시 모든 종목에 일괄 적용된다.
- "되돌리기" 버튼 클릭 시 적용 전 체크된 항목의 값만 이전 상태로 복원된다.
- 체크되지 않은 항목은 종목별 개별 설정값을 유지한다.

### 8.3 Run Evaluation

```
1. 현재 파라미터로 RL 에이전트 평가 실행
2. Trial History에 결과 추가 (trial_seed = base_seed + trial_index + run_index)
3. Auto Run Count만큼 자동 반복
4. 각 Trial의 최종 수익률, Alpha Gap, MDD를 통계 분석 패널에 표시
```

### 8.4 Simulation

```
1. PG Actor-Critic Optimizer가 n_iters 반복으로 하이퍼파라미터 탐색
2. 각 iteration마다 복수 시드로 RL 평가 -> Alpha Gap 기대값 산출
3. Policy Gradient로 탐색 정책 mu 업데이트, sigma 자동 스케줄링
4. 수렴 차트 실시간 표시 (파라미터 정규화 추이 + Gap 수렴 추이)
5. 탐색 완료 후 저장 여부 선택 (Simul. All 모드에서는 자동 저장)
```

### 8.5 Agent Decision Analysis

- 좌측: STATIC RL의 BUY/CASH 행동 빈도 막대 차트
- 우측: 일별 행동 로그 테이블 (HTML 기반, 우측 정렬, 2줄 헤더)
  - BUY: 파란색, CASH: 빨간색, 음수 수익: 빨간색
  - 스크롤 가능 (max-height 253px)

### 8.6 Trial History Statistical Analysis

- Trial-by-Trial Return Progression: 반복 평가별 수익률 추이 및 Mean/Max/Min 기준선
- Return Distribution across Trials: Vanilla RL과 STATIC RL의 박스 플롯 비교 (Mean, Median, Market 기준선 포함)
- Statistics Summary: Vanilla/STATIC의 Mean(sigma), Range를 항목별 2줄로 표시
- Trial 데이터 테이블: 전체 Trial 결과 (Trial, Seed, Vanilla Final %, STATIC Final %, Market Final %)

### 8.7 팀 포트폴리오 대시보드

모든 멤버의 평가 완료 후 상단에 자동 생성된다.

- All Members STATIC RL Cumulative Returns + Team Fund 차트
- 멤버별 성과 테이블: Member, Stocks, Persona(CTPT), Capital($), STATIC(%), Vanilla(%), Alpha Gap, MDD
- Team Fund: Softmax 가중 배분 기준 팀 전체 수익 곡선

### 8.8 Sticky Header

멤버별 섹션 헤더가 스크롤 시 화면 상단에 고정되어 현재 위치를 파악할 수 있다.

---

## 9. 파일 구조

```
Task-Constrained-RL-ColdStart_YS_v1-1/
|
+-- app.py                          Streamlit 메인 앱 (~1800줄)
|
+-- common/
|   +-- base_agent.py               Actor-Critic / Q-Learning 훈련·평가
|   +-- heuristic.py                PGActorCriticOptimizer
|   +-- evaluator.py                MDD, Softmax 비중, CTPT 코드
|   +-- data_loader.py              yfinance 데이터 로드 (다봉, 캐시)
|   +-- stock_registry.py           종목 정보 + 수수료 테이블
|
+-- members/
|   +-- member_1/config.py          Member 1 — SPY    (seed=42)
|   +-- member_2/config.py          Member 2 — QQQ    (seed=137)
|   +-- member_3/config.py          Member 3 — KOSPI  (seed=2024)
|   +-- member_4/config.py          Member 4 — KOSDAQ (seed=777)
|   +-- member_5/config.py          Member 5 — NVDA   (seed=314)
|   +-- member_6/config.py          Member 6 — TSLA   (seed=99)
|
+-- README.md
```

---

## 10. Simulation 단계별 연산 흐름

**파일:** `app.py` (sim_clicked 블록) + `common/heuristic.py` (PGActorCriticOptimizer)

---

### STEP 1 — 탐색 공간 정의 및 반복 횟수 결정

```python
n_iters = max(20, Auto_Run_Count * 8)   # 예: 6 * 8 = 48 iterations

param_bounds = {
    "lr":        (0.001, 0.1),
    "gamma":     (0.5,   0.99),
    "epsilon":   (0.01,  0.5),
    "v_epsilon": (0.01,  0.5),
}
```

---

### STEP 2 — PGActorCriticOptimizer 초기화

```python
optimizer = PGActorCriticOptimizer(
    bounds      = param_bounds,
    lr_actor    = 0.12,
    sigma_init  = 0.18,
    sigma_min   = 0.02,
    sigma_max   = 0.45,
    value_alpha = 0.25,
    seed        = l_seed,
)
```

| 변수       | 초기값                   | 의미                                    |
| ---------- | ------------------------ | --------------------------------------- |
| mu         | [0.5, 0.5, 0.5, 0.5]     | 정규화 공간에서 정책 평균 (탐색 중심점) |
| sigma      | [0.18, 0.18, 0.18, 0.18] | 파라미터별 탐험 폭                      |
| V          | 0.0                      | Critic 기준값 (baseline)                |
| best_score | -inf                     | 현재까지 최고 Gap                       |

---

### STEP 3 — 복수 평가 시드 준비

```python
_n_eval     = min(3, max(2, Auto_Run_Count // 3))
_eval_seeds = [base_seed + j for j in range(_n_eval)]
```

---

### STEP 4 — 탐색 페이즈 판정

```
_explore_end = max(6, n_iters // 4)

iteration < _explore_end  ->  PG Exploring    (초기 광역 탐험)
sigma_mean > 0.12         ->  PG Actor-Critic  (정책 업데이트 중)
sigma_mean <= 0.12        ->  PG Converging    (수렴 단계)
```

---

### STEP 5 — Actor: 파라미터 후보 샘플링

```python
Delta = rng.normal(0, sigma)
x_new = clip(mu + Delta, 0, 1)

candidate["lr"]        = 0.001 + x_new[0] * (0.1  - 0.001)
candidate["gamma"]     = 0.5   + x_new[1] * (0.99 - 0.5)
candidate["epsilon"]   = 0.01  + x_new[2] * (0.5  - 0.01)
candidate["v_epsilon"] = 0.01  + x_new[3] * (0.5  - 0.01)
```

---

### STEP 6 — 복수 시드로 RL 에이전트 평가

```python
for seed_i in _eval_seeds:
    vanilla_trace, static_trace = get_rl_data(
        ticker    = ticker,
        lr        = candidate["lr"],
        gamma     = candidate["gamma"],
        epsilon   = candidate["epsilon"],
        episodes  = l_epi,
        seed      = seed_i,
        v_epsilon = candidate["v_epsilon"],
        fee_rate  = fee_rate,
        interval  = l_interval,
    )
    gap_i = static_trace[-1] - vanilla_trace[-1]
    gaps.append(gap_i)
```

`get_rl_data` 내부 동작:

```
(1) 주가 데이터 로드 (data_loader.fetch_stock_data, 캐시 TTL=1h)
(2) STATIC RL 훈련: Actor-Critic, max(episodes*3, 500) 에피소드
    - lr_actor = lr * 1.0
    - lr_critic = lr * 1.5
    - epsilon_start = min(epsilon * 2.5, 0.60) -> 선형 감소 -> 최종 epsilon
    - 4-상태 공간 (EMA 위/아래 x 상승/하락)
(3) Vanilla RL 훈련: Q-Learning, episodes 에피소드
    - 동일 lr, gamma 적용
    - v_epsilon으로 독립 탐험
    - 2-상태 공간 (상승/하락)
(4) STATIC / Vanilla 평가 (최근 episodes 봉 데이터)
    - 각 봉에서 정책 pi(a|s) 실행
    - 거래 수수료 적용 (신규 매수 진입 시 1회)
    - 누적 수익률 배열 반환
```

---

### STEP 7 — Alpha Gap 기대값 계산

```python
expected_gap = mean(gaps)
```

단일 시드가 아닌 복수 시드 평균을 사용하여 특정 시드의 우연에 의한 과적합을 방지한다.

---

### STEP 8 — Critic 업데이트

```python
V += value_alpha * (expected_gap - V)
# value_alpha = 0.25 -> 현재 관측 25%, 이전 추정 75% 가중
```

`V`는 현재까지의 평균 기대값 수준을 추적하는 baseline 역할을 한다.

---

### STEP 9 — Advantage 계산 및 정규화

```python
raw_advantage = expected_gap - V
# raw_advantage > 0: 현재 파라미터가 기대 baseline보다 좋음
# raw_advantage < 0: 현재 파라미터가 기대 baseline보다 나쁨

# tanh 정규화: gap% 단위를 [0,1] 파라미터 공간과 통일 (10% gap ≈ tanh(1)=0.76 포화)
A_norm = tanh(raw_advantage / 10.0)
```

baseline 차감은 분산을 줄이고, tanh 정규화는 gap 스케일과 파라미터 스케일의 단위 불일치를 해소한다.

---

### STEP 10 — Actor 업데이트

```python
# Gaussian 정책의 score function: Δ/σ (L2 노름 ≤ 1 클립으로 스텝 크기 상한 보장)
pg_dir = Delta / sigma
if L2_norm(pg_dir) > 1.0:
    pg_dir = pg_dir / L2_norm(pg_dir)

# Actor mu 업데이트
mu += lr_actor * A_norm * pg_dir
mu  = clip(mu, 0, 1)
```

`Δ/σ`는 Gaussian log 정책의 기울기 방향이며, L2 노름 클립으로 과도한 스텝을 방지한다.

---

### STEP 11 — sigma 자동 스케줄링

```python
if A_norm > 0:
    sigma = max(sigma * 0.96, sigma_min)   # 좋은 방향 발견 -> 수렴
else:
    sigma = min(sigma * 1.04, sigma_max)   # 나쁜 방향 -> 재탐험
```

| 상황                | sigma 변화                   | 효과              |
| ------------------- | ---------------------------- | ----------------- |
| 연속으로 A_norm > 0 | 단조 감소 -> sigma_min(0.02) | mu 주변 정밀 탐색 |
| 연속으로 A_norm < 0 | 단조 증가 -> sigma_max(0.45) | 전역 재탐험       |
| A_norm 부호 교번    | sigma 진동 유지              | 탐험-수렴 균형    |

---

### STEP 12 — Best 갱신 및 차트 업데이트

```python
if expected_gap > best["gap"]:
    best = candidate.copy()
    _best_s_trace = static_trace    # Ghost Line용 최고 수익 곡선

gap_history.append(best["gap"])         # Best Gap 누적 추이
gap_iter_history.append(expected_gap)   # 각 iteration 기대값

for k in param_bounds:
    mu_hist_norm[k].append((mu[k] - lo_k) / (hi_k - lo_k))
```

수렴 차트:

- Parameter Convergence (중간 차트): 4개 파라미터의 정책 평균 mu가 [0,1] 정규화된 값의 추이. 직선으로 수렴할수록 최적값이 안정적으로 결정된 상태다.
- Expected Value Convergence (우측 차트): 매 iteration의 기대 Gap(회색)과 누적 Best Gap(파랑) 추이. Gap >= 1% 목표선과 Gap >= 25% 최고 기준선을 포함한다.

---

### 전체 흐름 요약

```
[Simulation 클릭]
      |
      v
STEP 1: n_iters, param_bounds 결정
      |
      v
STEP 2: PGActorCriticOptimizer 초기화
        (mu=[0.5,...], sigma=[0.18,...], V=0)
      |
      v
STEP 3: 복수 평가 시드 생성
      |
      v
  +--- for i in range(n_iters): -------------------+
  |                                                 |
  |  STEP 4: 페이즈 판정                           |
  |          (Exploring / Actor-Critic / Converging)|
  |          |                                      |
  |          v                                      |
  |  STEP 5: suggest_next()                         |
  |    Delta ~ N(0, sigma)                          |
  |    candidate = denorm(clip(mu + Delta, 0, 1))   |
  |          |                                      |
  |          v                                      |
  |  STEP 6: RL 평가 (복수 시드)                   |
  |    STATIC AC 훈련 -> 평가                       |
  |    Vanilla QL 훈련 -> 평가                      |
  |    gap_i = S_final - V_final                    |
  |          |                                      |
  |          v                                      |
  |  STEP 7: expected_gap = mean(gaps)              |
  |          |                                      |
  |          v                                      |
  |  STEP 8: V += alpha * (gap - V)        [Critic]  |
  |          |                                      |
  |          v                                      |
  |  STEP 9: A = gap - V                            |
  |          A_norm = tanh(A / 10)  [Advantage]     |
  |          |                                      |
  |          v                                      |
  | STEP 10: pg_dir = clip(Delta/sigma, L2≤1)       |
  |          mu += lr * A_norm * pg_dir    [Actor]  |
  |          |                                      |
  | STEP 11: sigma 스케줄링                         |
  |   A_norm > 0  ->  sigma * 0.96  (수렴)          |
  |   A_norm <= 0 ->  sigma * 1.04  (탐험)          |
  |          |                                      |
  | STEP 12: best 갱신, 차트 업데이트               |
  +---+---------------------------------------------+
      |
      v
[n_iters 완료]
      |
      v
Simul. All 모드  ->  best 파라미터 자동 저장 -> config.py 갱신 -> Run Evaluation 자동 실행
수동 모드        ->  [저장 및 반영] / [반영 취소] 버튼 표시
```

---

### 파라미터 수렴 수식 정리

```
[표기]
  mu  : 정책 평균 (정규화 공간 [0,1]^4)
  sigma : 탐험 폭 (파라미터별 독립)
  Delta : x_new - mu (정규화 공간 편차)
  V   : Critic baseline (EMA 방식 평균 gap 추정)
  A   : Advantage = gap - V
  alpha_c : value_alpha = 0.25
  alpha_a : lr_actor   = 0.12

[1] 샘플링
  Delta  ~ N(0, sigma)
  x_new  = clip(mu + Delta, 0, 1)

[2] Critic (EMA 방식)
  V <- V + alpha_c * (gap - V)

[3] Advantage
  A      = gap - V
  A_norm = tanh(A / 10)              (gap% → [-1,1] 정규화)

[4] Actor (Policy Gradient)
  pg_dir = Delta / sigma             (Gaussian score function 방향)
  if L2(pg_dir) > 1: pg_dir /= L2(pg_dir)   (스텝 크기 상한 보장)
  mu <- clip(mu + alpha_a * A_norm * pg_dir, 0, 1)

[5] sigma 스케줄
  sigma <- max(sigma * 0.96, sigma_min)   if A_norm > 0
  sigma <- min(sigma * 1.04, sigma_max)   if A_norm <= 0
```

---

### 주요 워크플로우

```
1. [사이드바] Fallback Parameters에서 적용할 항목에 체크 후 파라미터 설정
2. [All 적용] 버튼으로 체크된 파라미터를 전체 종목에 일괄 적용
3. [Run Evaluation] 현재 파라미터로 RL 에이전트 평가 (n회 반복, Trial History 축적)
4. [Simulation] PG Actor-Critic Optimizer로 최적 파라미터 자동 탐색
   -> 탐색 완료 후 저장 여부 선택 (config.py에 영구 저장 가능)
5. [Eval. All] 전체 6개 종목 순차 평가
6. [Simul. All] 전체 6개 종목 순차 시뮬레이션 + 최적 파라미터 자동 저장
7. [팀 대시보드] 전체 포트폴리오 성과 및 멤버별 기여도 확인
```
