# Chainers Master Fund — Task-Constrained RL Cold-Start

## KDT Team Project : 팀 Chainers 🫡

---

## 요약 보고서

### 프로젝트 개요

본 프로젝트는 **Task-Constrained RL Cold-Start** 조건에서 복수의 강화학습 알고리즘을 비교하는 멀티 에이전트 트레이딩 시뮬레이터다. 6명의 팀원이 각자 담당 종목에 에이전트를 배치하여 개인 수익률과 팀 포트폴리오 수익률을 측정한다. 핵심 비교 대상은 EMA 기반 4-상태 **STATIC Actor-Critic** (Policy Gradient) 과 2-상태 **Vanilla Q-Learning** 이며, 추가로 심층 강화학습 5종 (**A2C / A3C / PPO / SAC / DDPG**, NumPy 전용 TinyMLP)을 지원한다. 하이퍼파라미터는 **PG Actor-Critic Optimizer**가 복합 Gap 목표를 극대화하는 방향으로 자동 탐색한다. 모든 평가는 **워크포워드 검증**(앞 70% 학습 / 뒤 30% OOS)으로 과적합을 방지한다.

**핵심 연구 질문**: 사전 데이터 없이 Cold-Start 조건에서, EMA 기반 4-상태 Actor-Critic 에이전트는 2-상태 Q-Learning 에이전트 대비 얼마나 높은 누적 수익률을 달성하는가?

**Alpha Gap** = STATIC RL 최종 수익률 − Vanilla RL 최종 수익률

| Gap (STATIC vs Market) | 판정 |
|------------------------|------|
| ≥ 1%p | ✅ 목표 달성 |
| ≥ 5%p | 우수 달성 |
| ≥ 25%p | 최고 달성 🏆 |

---

### 팀 구성 및 담당 종목 (improve 7-2 기준)

| 멤버 | 담당 종목 | Ticker | 시드 | 최적 파라미터 | 8-State | Roll Period | Gap |
|------|----------|--------|------|--------------|---------|------------|-----|
| Member 1 | S&P 500 ETF | SPY | 42 | lr=0.0802, γ=0.8732, ε=0.1667, v_ε=0.1347 | O | — | +8.66%p |
| Member 2 | Nasdaq 100 ETF | QQQ | 137 | lr=0.0722, γ=0.9663, ε=0.2116, v_ε=0.0620 | — | — | +35.62%p 🏆 |
| Member 3 | KOSPI 지수 | ^KS11 | 2024 | lr=0.0050, γ=0.9087, ε=0.1465, v_ε=0.1947 | O | — | — ⚠️ |
| Member 4 | KOSDAQ 지수 | ^KQ11 | 777 | lr=0.0380, γ=0.9217, ε=0.0760, v_ε=0.0616 | O | 30봉 | +21.36%p |
| Member 5 | 미국배당다우존스 ETF | SCHD | 314 | lr=0.0624, γ=0.9449, ε=0.1708, v_ε=0.0879 | O | 60봉 | +0.57%p |
| Member 6 | 로열 골드 | RGLD | 99 | lr=0.0269, γ=0.9380, ε=0.1509, v_ε=0.1268 | O | — | +17.81%p |

추가 지원 종목: NVDA, TSLA, GOOGL, MSFT, 삼성전자(005930.KS), SK하이닉스(000660.KS)

> KOSPI/KOSDAQ: 워크포워드 구조적 OOS 한계 — 학습 구간 횡보/하락, OOS 급등. 어떤 파라미터로도 시장 수익률 초과 어려움 (정상 결과).

---

### 알고리즘 비교 요약

#### RL Algorithm 선택 순서 및 핵심 특징

```
── On-policy (버퍼 없음) ───────────────────────┬── On+Off-policy ──┬── Off-policy (버퍼 있음) ──
  STATIC → STATIC_H → A2C → A3C → PPO          │      ACER          │    SAC → DDPG
  Tabular  Hybrid(PPO  1-step n-step Clip+GAE   │  Retrace+IS(이산)  │  Entropy(이산)  Det.(연속)
           Clip+AdapT)
```

| 순서 | 알고리즘 | 핵심 특징 | 행동 공간 | 버퍼 |
|-----|---------|---------|---------|------|
| 1 | **STATIC** | Tabular Actor-Critic, 4/8-상태 EMA×추세 이산화 | 이산 | ❌ |
| 1H | **STATIC_H** | STATIC + Tabular PPO Clipping + Adaptive Temperature (Hybrid) | 이산 | ❌ |
| 2 | **A2C** | 신경망 AC, 1-step TD Advantage | 이산 | ❌ |
| 3 | **A3C** | A2C + n-step return, 편향↓ | 이산 | ❌ |
| 4 | **PPO** | A3C + Clip + GAE, 안정성↑ | 이산 | ❌ |
| 5 | **ACER** | PPO + Retrace(λ) + Truncated IS, 샘플 효율↑ | 이산 | ✅ |
| 6 | **SAC** | Off-policy + 자동 엔트로피 온도 α | 이산 | ✅ |
| 7 | **DDPG** | Off-policy + 연속 포지션 [0,1] + OU noise | **연속** | ✅ |

#### 상세 비교

| 항목 | STATIC | STATIC_H | Vanilla | A2C | A3C | PPO | ACER | SAC | DDPG |
|------|--------|----------|---------|-----|-----|-----|------|-----|------|
| 알고리즘 | PG Actor-Critic (Tabular) | STATIC + PPO Clip + Adaptive-α | Q-Learning | AC (신경망) | AC n-step | Clipped Surrogate | Retrace+IS | Soft AC | Deterministic PG |
| 상태 표현 | 4/8개 이산 | 4/8개 이산 | 2개 이산 | 5차원 연속 | 5차원 연속 | 5차원 연속 | 5차원 연속 | 5차원 연속 | 5차원 연속 |
| Q 추정 | TD V(s) | TD V(s)+PPO step | max Q | 1-step TD | n-step | GAE λ=0.95 | **Retrace(λ)** | Soft-V | Bellman |
| 탐험 | ε-greedy | ε-greedy+α_t 적응 | ε annealing | 확률적 π | 확률적 π | 확률적 π | ε-greedy+IS | 자동 α | OU noise |
| 행동 공간 | 이산 | 이산 | 이산 | 이산 | 이산 | 이산 | 이산 | 이산 | **연속** |
| 버퍼 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| 역할 | 주 평가 대상 | Tabular 하이브리드 | 비교 기준선 | 신경망 비교 | 신경망 비교 | 신경망 비교 | 신경망 비교 | 신경망 비교 | 신경망 비교 |

---

### 거래 수수료

| 시장 | 매수 | 매도 | 왕복 합계 |
|------|------|------|----------|
| 미국 주식·ETF | 0.05% | 0.05% | 0.10% |
| 국내 주식·지수 | 0.015% | 0.215% | 0.23% |

---

### 주요 기능

- **Run Evaluation**: 현재 파라미터로 RL 에이전트를 평가하고 Trial History를 축적한다.
- **Simulation**: PG Actor-Critic Optimizer로 하이퍼파라미터를 자동 탐색하여 복합 Gap을 극대화한다.
- **RL Algorithm 선택**: STATIC / STATIC_H / A2C / A3C / PPO / ACER / SAC / DDPG 중 선택 (System Parameters 패널).
- **Per-stock 위젯**: 종목별 8-State(vol) 사용 여부, Roll Period를 개별 체크박스·셀렉트박스로 설정한다.
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

## 저작권

본 저장소에 포함된 코드 및 모든 출력 이미지 결과물은 저작권법에 의해 보호됩니다.

저작권자의 명시적 허가 없이 본 자료의 전부 또는 일부를 복제, 배포, 수정, 상업적으로 이용하는 행위를 금합니다.

© 2026. All rights reserved.
Contact : sjowun@gmail.com

---

---

## 전체 상세 설명

### 목차

1. [시스템 아키텍처](#1-시스템-아키텍처)
2. [강화학습 알고리즘 — STATIC / STATIC_H / Vanilla](#2-강화학습-알고리즘--static--static_h--vanilla)
3. [강화학습 알고리즘 — 신경망 (A2C / A3C / PPO / ACER / SAC / DDPG)](#3-강화학습-알고리즘--신경망-a2c--a3c--ppo--acer--sac--ddpg)
4. [상태 공간 설계](#4-상태-공간-설계)
5. [워크포워드 검증 — Train/Test 분리](#5-워크포워드-검증--traintest-분리)
6. [하이퍼파라미터 탐색 — PG Actor-Critic Optimizer](#6-하이퍼파라미터-탐색--pg-actor-critic-optimizer)
7. [웹 UI 파라미터 상세](#7-웹-ui-파라미터-상세)
8. [랜덤 시드의 역할](#8-랜덤-시드의-역할)
9. [데이터 파이프라인](#9-데이터-파이프라인)
10. [포트폴리오 평가 및 성과 지표](#10-포트폴리오-평가-및-성과-지표)
11. [UI 기능 상세](#11-ui-기능-상세)
12. [파일 구조](#12-파일-구조)
13. [Simulation 단계별 연산 흐름](#13-simulation-단계별-연산-흐름)
14. [개선 이력](#14-개선-이력)
15. [RL Algorithm 시뮬레이션 학습 가이드](#15-rl-algorithm-시뮬레이션-학습-가이드)

---

## 1. 시스템 아키텍처

```
app.py  (Streamlit 웹 UI, ~2300줄)
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
      |    +-- [2행] RL Algorithm / LR / Gamma / STATIC ε / Vanilla ε / Sim Min Steps / Sim Step Mult.
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
           +-- Trial 데이터 테이블

common/
 +-- base_agent.py      RL 훈련 및 평가 (STATIC / Vanilla / Neural)
 +-- nn_utils.py        TinyMLP, ReplayBuffer, extract_features
 +-- heuristic.py       하이퍼파라미터 탐색 (PGActorCriticOptimizer)
 +-- evaluator.py       성과 지표 계산 (MDD, Softmax 비중, CTPT 코드)
 +-- data_loader.py     yfinance 데이터 로드 (다봉 지원, 캐시 1시간)
 +-- stock_registry.py  종목 정보 및 수수료 테이블 (12종목)

members/member_N/
 +-- config.py          멤버별 담당 종목 + RL 하이퍼파라미터 (10 필드)
```

---

## 2. 강화학습 알고리즘 — STATIC / STATIC_H / Vanilla

### 2.1 STATIC RL — Actor-Critic

**파일:** `common/base_agent.py` — `_train_actor_critic_static()`

Policy Gradient Theorem + REINFORCE with baseline을 온라인 TD 방식으로 구현한다.

#### 마르코프 결정 과정 (MDP)

```
상태 공간 S = {0, 1, 2, 3}  (4개 상태)
행동 공간 A = {0(CASH), 1(BUY)}
전이 확률 P(s'|s, a)  = 시장 가격 변동 (외생 확률 과정)
보상 함수 R(s, a, s') = daily_return × 1[a=BUY] − fee + ENTROPY_COEFF · H(π)
할인 인수 γ           = 종목별 최적값 (0.87~0.97)
```

#### TD Critic — 벨만 방정식 근사

TD(0) 방식으로 상태 가치함수 V(s)를 온라인 갱신한다.

```
δ_t  = r_eff + γ · V(s_{t+1}) − V(s_t)     ← TD 오차 (Advantage 근사값)

여기서:
  r_eff = reward + 0.05 · H(π)              ← 엔트로피 정규화 보상
  H(π)  = −Σ π(a|s) · log π(a|s)           ← 정책 엔트로피 (불확실성 척도)

  V(s_t) += lr · δ_t                        ← Critic 가치함수 갱신
```

**δ_t (TD 오차)의 의미**: 현재 상태의 실제 보상 + 다음 상태 가치 추정값이 현재 가치 추정값보다 얼마나 높은지. 양수면 현재 행동이 예상보다 좋았음 → Actor를 강화.

**엔트로피 정규화 H(π)**: 정책이 CASH와 BUY를 비슷한 확률로 선택할수록 엔트로피가 높아진다. 0.05 계수를 보상에 더해 에이전트가 다양한 행동을 유지하도록 유도한다 (Buy&Hold 고착 방지).

#### Actor — Policy Gradient Theorem

```
∇J(θ) = E_π[∇ log π_θ(a|s) · A(s, a)]

  A(s, a) ≈ δ_t                            ← TD 오차를 Advantage 대용

Softmax 정책:
  π_θ(a|s) = exp(θ[s, a]) / Σ exp(θ[s, :])

Score function (∇ log π):
  ∇ log π(a|s) = 1[a == chosen] − π(·|s)   ← 선택된 행동 강화, 나머지 약화

Actor 업데이트:
  θ[s, a] += lr · δ_t · ∇ log π(a|s)
```

**Policy Gradient 직관**: δ_t > 0 이면 현재 행동이 기대보다 좋았으므로 해당 행동의 확률을 높인다. δ_t < 0 이면 기대보다 나빴으므로 확률을 낮춘다.

#### 초기화 (Cold-Start 수수료 장벽 완화)

```python
theta = np.zeros((4, 2))
theta[1, 1] = max(0.05, fee_rate * 30)   # EMA아래+상승: 미세 BUY 선호
theta[2, 1] = max(0.10, fee_rate * 50)   # EMA위+하락:  BUY 선호
theta[3, 1] = max(0.20, fee_rate * 80)   # EMA위+상승:  BUY 선호 강화
V = np.zeros(4)                          # Critic 가치함수 초기값 = 0
```

수수료가 높을수록 BUY 선호 초기화값을 크게 설정하여 초반 CASH 고착을 방지한다.

---

### 2.2 Vanilla RL — Q-Learning (비교 기준선)

**파일:** `common/base_agent.py` — `_train_qlearning_vanilla()`

#### 벨만 최적 방정식 (Q-Learning)

```
Q*(s, a) = E[r + γ · max_{a'} Q*(s', a') | s, a]

TD 업데이트 (off-policy):
  Q(s, a) += lr · [r + γ · max_{a'} Q(s', a') − Q(s, a)]

여기서:
  r + γ · max_{a'} Q(s', a')  = TD 목표값 (Bellman Target)
  Q(s, a)                      = 현재 추정값
  차이                          = Bellman Residual (최소화 대상)
```

**max_{a'} Q(s', a')의 의미**: 다음 상태에서 최적 행동을 취했을 때의 가치. Q-Learning은 행동 선택(ε-greedy)과 업데이트(max) 정책이 다른 off-policy 알고리즘이다.

#### Q-테이블 초기화 및 CASH 편향 해소

```python
q_table = np.zeros((2, 2))
q_table[:, 1] = max(fee_rate * 50, 0.05)  # fee 비례 BUY 선호 초기화
# fee_rate=0.001(미국) → Q[:,1]=0.05
# fee_rate=0.0023(국내) → Q[:,1]=0.115
```

#### epsilon annealing (탐험률 점진 감소)

```python
for ep in range(train_episodes):
    _eps = epsilon * max(1.0, 2.0 - ep / (train_episodes - 1))
    # 에피소드 시작: 2ε (강한 탐험)  →  에피소드 종료: ε (기본 탐험 유지)

prev_action = 1   # BUY 시작 고정 (에피소드 첫 step 수수료 편향 제거)
```

#### 훈련 후 보정 — Q-floor margin

```python
# 훈련 노이즈로 인한 CASH 고착 방지
# 전체 상태(bear=0, bull=1)에서 BUY 상대 우위 보장
q_table[0, 1] = max(float(q_table[0, 1]), float(q_table[0, 0]) + 0.005)
q_table[1, 1] = max(float(q_table[1, 1]), float(q_table[1, 0]) + 0.005)
```

---

### 2.3 STATIC-H — Hybrid Tabular RL (Tabular PPO Clipping + Adaptive Temperature)

**파일:** `common/base_agent.py` — `_train_actor_critic_hybrid()`

STATIC Actor-Critic 위에 두 가지 안정화 메커니즘을 추가한 하이브리드 Tabular RL이다.
- **Tabular PPO Clipping**: 구 정책 대비 신 정책 비율을 제한하여 급격한 정책 변화 억제
- **Adaptive Temperature**: 최근 행동 전환 빈도(flip_rate)에 따라 엔트로피 계수 α_t를 자동 조정

#### 신규 상수 (base_agent.py 상단)

```python
HYBRID_ALPHA_MIN = 0.01   # 적응 온도 α_t 최솟값
HYBRID_ALPHA_MAX = 0.15   # 적응 온도 α_t 최댓값
HYBRID_FLIP_WIN  = 20     # 행동 전환율 윈도우 (봉)
```

#### Adaptive Temperature — α_t 산출

```
flip_rate = (최근 HYBRID_FLIP_WIN 봉에서 행동이 바뀐 횟수) / HYBRID_FLIP_WIN
α_t       = clip(0.05 × (1 + flip_rate), HYBRID_ALPHA_MIN, HYBRID_ALPHA_MAX)

  flip_rate = 0.0 (행동 변화 없음) → α_t = 0.05 (최솟값에 가까운 약한 탐험)
  flip_rate = 1.0 (매 봉마다 전환) → α_t = 0.10 (탐험 강화)
  → BUY/CASH 전환이 잦을수록 엔트로피 보상을 높여 탐험 다양성 유지
```

**α_t (적응 온도)의 의미**: 에이전트가 한 행동에 고착되면 flip_rate가 낮아지고 α_t도 낮아져 엔트로피 보상이 줄어든다. 반대로 과도하게 전환하면 α_t가 높아져 탐험을 장려한다. STATIC의 고정 ENTROPY_COEFF=0.05 대신 시장 상황에 맞게 자동 조정된다.

#### Tabular PPO Clipping — 정책 업데이트 제한

```
1단계: 구 정책 저장
  π_old[state] = softmax(θ[state])     ← 업데이트 전 BUY/CASH 확률

2단계: 시험 업데이트 (theta_try)
  theta_try = θ[state] + lr × δ_t × ∇log π(a|s)

3단계: Importance Sampling 비율 산출
  π_new[state] = softmax(theta_try[state])
  r_t = π_new[a] / π_old[a]            ← 신/구 정책 비율

4단계: step_scale 결정
  if r_t < 1 − clip_eps or r_t > 1 + clip_eps:
      step_scale = 0.3                  ← 급격한 변화 억제 (30% 스텝)
  else:
      step_scale = 1.0                  ← 정상 업데이트

5단계: 최종 업데이트
  θ[state] += step_scale × lr × δ_t × ∇log π(a|s)

여기서 clip_eps = PPO_CLIP_EPS = 0.2 (base_agent.py 상수)
```

**Tabular PPO Clipping의 의미**: 신경망 PPO와 달리 Tabular 정책에서 구 정책 대비 신 정책 비율이 `1 ± 0.2` 범위를 벗어나면 업데이트 폭을 30%로 줄인다. BUY→CASH 또는 CASH→BUY로 정책이 한 번에 크게 뒤집히는 것을 방지하여 수렴 안정성을 높인다.

#### 엔트로피 보상 통합

```
r_eff = r + α_t × H(π)               ← 적응 온도 적용 (STATIC은 고정 0.05)
H(π)  = −Σ π(a|s) × log π(a|s)      ← 정책 엔트로피
```

#### Cold-Start 초기화

STATIC과 동일한 fee 비례 BUY 선호 초기화를 사용한다.

```python
theta[1, 1] = max(0.05, fee_rate * 30)
theta[2, 1] = max(0.10, fee_rate * 50)
theta[3, 1] = max(0.20, fee_rate * 80)
```

#### STATIC vs STATIC_H 비교

| 항목 | STATIC | STATIC_H |
|------|--------|----------|
| 엔트로피 계수 | 고정 0.05 | 동적 α_t (0.01~0.15) |
| 정책 업데이트 | 직접 θ += lr·δ·∇logπ | PPO Clip step_scale 적용 |
| 급격한 정책 변화 | 제어 없음 | 비율 1±0.2 초과 시 0.3배 억제 |
| 행동 고착 대응 | ENTROPY_COEFF=0.05 고정 | flip_rate 낮으면 α_t 자동 유지 |
| 상태 분리 | 상태별 독립 학습 | 상태별 독립 + step_scale 공통 |

---

## 3. 강화학습 알고리즘 — 신경망 (A2C / A3C / PPO / ACER / SAC / DDPG)

모든 신경망 알고리즘은 **NumPy 전용 TinyMLP** [5→32→n_out] 구조를 사용한다.
PyTorch/TensorFlow 의존성 없음.

### 3.1 특징 벡터 (공통, 5차원)

```
s_t = [ret, ema_ratio, vol, momentum, trend]

  ret       = Daily_Return[t]                        # 당일 수익률
  ema_ratio = Close[t] / EMA_10[t] - 1               # EMA 괴리율
  vol       = Rolling_Std[t]                         # 변동성
  momentum  = Σ Daily_Return[t-5:t]                  # 5일 모멘텀
  trend     = sign(Close[t] - Close[t-5])            # 5일 추세 방향
```

### 3.2 TinyMLP 구조

**파일:** `common/nn_utils.py`

```
입력(5) → [Dense(32) + ReLU] → 출력(n_out, 선형)

초기화: He 초기화  W ~ N(0, sqrt(2/fan_in))
최적화: Adam  (β₁=0.9, β₂=0.999, ε=1e-8)
역전파: backward_and_update(x, grad_out, lr)
```

#### 순전파 수식

```
z₁ = x @ W₁ + b₁         [n_h=32]   ← 선형 변환
h₁ = max(0, z₁)           [n_h=32]   ← ReLU 활성화
z₂ = h₁ @ W₂ + b₂        [n_out]    ← 선형 출력 (softmax/sigmoid는 호출자 적용)
```

#### 역전파 수식

```
입력: grad_out  (∂L/∂z₂, n_out 차원)

출력층:
  ∂L/∂W₂ = h₁ᵀ ⊗ grad_out    ← outer product
  ∂L/∂b₂ = grad_out
  δ₁     = (grad_out @ W₂ᵀ) · 1[z₁ > 0]   ← W₂ 역전파 × ReLU 미분

은닉층:
  ∂L/∂W₁ = xᵀ ⊗ δ₁
  ∂L/∂b₁ = δ₁

입력 기울기 (DDPG 체인 규칙용):
  ∂L/∂x  = δ₁ @ W₁ᵀ
```

#### Adam 갱신 수식

```
t       += 1
m_t      = β₁·m_{t-1} + (1-β₁)·g_t          [1차 모멘텀]
v_t      = β₂·v_{t-1} + (1-β₂)·g_t²         [2차 모멘텀]
m̂_t     = m_t / (1 - β₁ᵗ)                    [편향 보정]
v̂_t     = v_t / (1 - β₂ᵗ)
W       -= lr · m̂_t / (√v̂_t + ε)

타겟 네트워크 소프트 갱신 (SAC/DDPG 공통):
  θ_tgt ← τ·θ + (1-τ)·θ_tgt,  τ=0.005
```

**He 초기화의 의미**: ReLU 활성화 함수에서 층을 통과할수록 분산이 유지되도록 설계된 초기화. fan_in이 클수록 초기 가중치를 작게 설정하여 기울기 폭발/소실 방지.

**Adam 최적화기**: 1차 모멘텀(이동평균 기울기)과 2차 모멘텀(기울기 제곱 이동평균)을 동시에 추적하여 파라미터별 학습률을 자동 조정한다. 희소한 보상 환경에서 SGD보다 안정적으로 수렴한다.

---

### 3.3 A2C — Advantage Actor-Critic

**파일:** `common/base_agent.py` — `_train_a2c()`

온라인 1-step TD Advantage로 Actor와 Critic을 동시에 학습하는 가장 단순한 신경망 Actor-Critic.

#### 수식

```
1단계: Critic (가치함수 V 학습)
  δ_t  = r_t + γ · V(s_{t+1}) − V(s_t)         ← 1-step TD 오차
  ∂L_critic/∂W_c = −δ_t                          ← MSE 손실의 기울기

2단계: Actor (정책 π_θ 학습)
  π_θ(a|s)  = softmax(Actor(s_t))
  A_t       = δ_t                                ← Advantage = TD 오차
  ∂L_actor  = −A_t · (1[a=chosen] − π_θ(·|s))   ← policy gradient
```

**Advantage A_t의 의미**: 선택된 행동이 평균(V(s))보다 얼마나 좋은지를 나타낸다. A_t > 0이면 그 행동 확률을 높이고, A_t < 0이면 낮춘다. TD 오차를 Advantage로 사용함으로써 Monte Carlo 방식보다 분산이 낮다.

---

### 3.4 A3C — Asynchronous Advantage Actor-Critic (n-step)

**파일:** `common/base_agent.py` — `_train_a3c()`

A2C의 n-step 리턴 버전. 단일 스레드로 근사 구현 (비동기 워커 생략).

#### 수식

```
n-step Return (n=5):
  R_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... + γ^(n-1)·r_{t+n-1} + γ^n·V(s_{t+n})

역방향 누적:
  R ← V(s_{t+n})                 (부트스트랩 초기값, γ 미적용)
  for k = n-1, n-2, ..., 0:
    R ← r_{t+k} + γ · R         (역방향으로 γ 누적)

A_t = R_t − V(s_t)              ← n-step Advantage
```

**n-step의 의미**: 1-step(A2C)은 즉각 보상만 반영하여 편향이 크다. n-step은 n개 미래 보상을 직접 합산하여 편향을 줄이되, Monte Carlo(전체 에피소드)보다 분산이 낮다. n=5는 편향-분산 균형점.

---

### 3.5 PPO — Proximal Policy Optimization

**파일:** `common/base_agent.py` — `_train_ppo()`

정책 업데이트 폭을 클리핑으로 제한하여 학습 안정성을 높인 알고리즘.

#### 수식

```
1단계: GAE (Generalized Advantage Estimation, λ=0.95)
  δ_t = r_t + γ·V(s_{t+1}) − V(s_t)             ← 1-step TD 오차

  역방향 GAE 누적:
  Â_t = δ_t + γλ·Â_{t+1}                         ← GAE Advantage

2단계: Importance Sampling 비율
  r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)      ← 구정책 대비 신정책 비율

3단계: Clipped Surrogate Objective
  L_CLIP = E[min(r_t · Â_t,  clip(r_t, 1-ε, 1+ε) · Â_t)]

4단계: n_epochs=4 미니배치 업데이트 (rollout_len=64 스텝)
```

**GAE λ의 의미**: λ=0이면 1-step TD (고편향, 저분산), λ=1이면 Monte Carlo (저편향, 고분산). λ=0.95는 미래 δ들을 지수 감쇠 가중치로 합산하여 편향-분산 균형을 최적화한다.

**Clip의 역할**: r_t(θ)가 1 ± ε 범위를 벗어나면 기울기를 0으로 만들어 정책이 한 번에 너무 많이 변하지 않도록 제한한다. 이를 통해 학습이 갑자기 발산하는 것을 방지한다.

---

### 3.6 ACER — Actor-Critic with Experience Replay

**파일:** `common/base_agent.py` — `_train_acer()`

On-policy 롤아웃을 경험 재생 버퍼에 저장하고, Retrace(λ)로 오프-정책 Q 타겟을 보정한다. Truncated IS로 정책 기울기의 분산을 제어하며, 2-action 특성을 이용해 선택/비선택 행동 양쪽에 편향 보정항을 적용한다.

#### 수식

```
행동 정책 μ: ε-greedy  →  μ(a|s) = ε·½ + (1-ε)·π(a|s)
IS 비율    : ρ_t = π(a_t|s_t) / μ(a_t|s_t)
절단 IS    : c_t = λ · min(c̄, ρ_t),   c̄=10, λ=0.95

Retrace(λ) 역방향 (t=T,...,0):
  Q^ret_T = r_T + γ·V(s_{T+1})                                        [부트스트랩]
  Q^ret_t = r_t + γ·V(s_{t+1}) + γ·c_{t+1}·(Q^ret_{t+1} − Q(s_{t+1}, a_{t+1}))

  V(s) = Σ_a π(a|s)·Q(s,a)    [Q-network 2-output으로 계산]

Actor gradient (주항 + 보정항):
  주항:   −min(c̄, ρ_t) · (Q^ret_t − V_t) · (e_{a_t} − π_t)
  보정항: −max(0, 1 − c̄/ρ(ā|s_t)) · π(ā|s_t) · (Q(s_t,ā) − V_t) · (e_ā − π_t)
  (ā = 1 − a_t,  μ(ā|s_t) = 1 − μ(a_t|s_t) — 2-action 특성 활용)

Q-network 갱신 (MSE, 선택 행동만):
  ∂L_Q/∂Q[a_t] = Q(s_t, a_t) − Q^ret_t
```

**Retrace(λ)의 의미**: 오프-정책 전이에서 Q 타겟을 계산할 때, IS 비율을 λ·min(c̄, ρ)로 절단하여 분산을 억제하면서도 편향 없이 다단계 미래 보상을 반영한다. ρ가 1에 가까울수록(현재 정책과 행동 정책이 비슷할수록) Monte Carlo에 가까워지고, 멀수록 1-step TD에 가까워진다.

**편향 보정항의 의미**: min(c̄, ρ_t) 절단으로 인해 생기는 편향을, 모든 행동에 대한 가중 합으로 보정한다. 이를 통해 불균형한 IS 절단의 영향을 상쇄한다.

---

### 3.7 SAC — Soft Actor-Critic (이산 행동)

**파일:** `common/base_agent.py` — `_train_sac()`

엔트로피 최대화 목표를 명시적으로 포함하고, 온도 파라미터 α를 자동 학습하는 오프-정책 알고리즘.

#### 수식

```
1단계: Soft Q-Target
  Q_tgt(s, a) = r + γ · [V_soft(s')]

  V_soft(s') = Σ_{a'} π(a'|s') · [Q(s', a') − α · log π(a'|s')]
             = E_π[Q(s',a') − α·H(π)]             ← 엔트로피 포함 가치

2단계: Q-Network 업데이트
  L_Q = (Q(s,a) − Q_tgt(s,a))²                   ← MSE

3단계: Actor 업데이트 (엔트로피 최대화)
  L_actor = Σ_a π(a|s) · [α·log π(a|s) − Q(s,a)]
  = −E_π[Q(s,a) − α·log π(a|s)]                  ← Q 최대화 + 엔트로피 보상

4단계: 자동 온도 α 업데이트
  J(α) = −α · [log π(a|s) + H_target]
  α 업데이트 방향: log_alpha -= lr · (log_pi + H_target)
  H_target = log(n_actions) × 0.5                 ← 목표 엔트로피
```

**α (온도 파라미터)의 의미**: α가 크면 엔트로피(탐험)에 더 높은 보상을 부여 → 정책이 더 확산적(random)이 된다. α=0이면 일반 Actor-Critic. 자동 α는 실제 엔트로피가 목표보다 낮으면 α를 높이고, 높으면 낮춰 탐험/착취의 균형을 자동 조정한다.

**타겟 Q 네트워크**: 학습 안정성을 위해 Q_tgt는 별도 네트워크로 유지하고, 매 에피소드마다 소프트 업데이트: `θ_tgt ← τ·θ + (1-τ)·θ_tgt` (τ=0.005).

---

### 3.8 DDPG — Deep Deterministic Policy Gradient

**파일:** `common/base_agent.py` — `_train_ddpg()`

결정론적 정책으로 **연속 포지션 [0, 1]** 을 직접 출력하는 오프-정책 알고리즘.

#### 수식

```
Actor (결정론적 정책):
  μ(s) = sigmoid(Actor(s)) ∈ [0, 1]            ← 보유 비율

Critic (행동-가치 함수):
  Q(s, a) = Critic([s, a])                      ← 6차원 입력 (5 상태 + 1 행동)

Q-Target 업데이트:
  Q_tgt(s,a) = r + γ · Q_tgt(s', μ_tgt(s'))

Actor 업데이트 (Policy Gradient):
  ∇_θ J = E[∂Q/∂a · ∂μ/∂θ]

  연쇄 법칙:
    ∂Q/∂a  = Critic 입력에 대한 기울기 (마지막 입력 성분)
    ∂μ/∂θ  = sigmoid'(logit) · ∂Actor/∂θ

타겟 네트워크 소프트 갱신 (매 스텝):
  θ_actor_tgt  ← τ · θ_actor  + (1-τ) · θ_actor_tgt
  θ_critic_tgt ← τ · θ_critic + (1-τ) · θ_critic_tgt
  τ = 0.005 (DDPG_TAU)

탐험: OU (Ornstein-Uhlenbeck) Noise
  X_{t+1} = X_t − θ_ou·X_t + σ_ou·N(0,1)
  θ_ou=0.15, σ_ou=0.2
  pos_t = clip(μ(s_t) + ε · X_t, 0, 1)     ← 탐험 포지션

연속 포지션 보상 (훈련 중):
  r_t = Daily_Return[t] × pos_t − fee_rate × |pos_t − pos_{t-1}|
  → 포지션 비율에 비례한 수익 − 포지션 변동 비례 수수료

평가 시 이진 변환 (ε·X 없음):
  μ(s_t) = sigmoid(Actor(s_t))
  a_t = BUY(1)   if μ(s_t) ≥ 0.5
        CASH(0)  otherwise
```

**결정론적 정책의 의미**: DDPG는 확률적 정책 대신 단일 행동값을 직접 출력한다. 연속 포지션 μ(s) ∈ [0,1]은 보유 비율(0=완전 현금, 1=완전 매수)을 의미하며, 평가 시 0.5 임계값으로 BUY/CASH 이진 결정을 한다.

**경험 재생 버퍼 (ReplayBuffer)**: 과거 (s, a, r, s', done) 전이를 원형 버퍼에 저장하고 미니배치(batch_size=64)로 무작위 샘플링하여 학습한다. 이를 통해 연속된 경험들의 시간 상관성을 제거한다.

---

## 4. 상태 공간 설계

```
STATIC RL — 4개 이산 상태:

  state = is_bull × 1 + is_above_ema × 2

  State 0: 하락 (ret ≤ 0) + EMA 아래 (price < EMA_10)   → 가장 보수적
  State 1: 상승 (ret > 0) + EMA 아래                    → 주의 단계
  State 2: 하락            + EMA 위 (price ≥ EMA_10)   → 중립
  State 3: 상승            + EMA 위                    → 가장 강한 매수 신호

확장 — 8개 이산 상태 (use_vol=True):
  state += is_high_vol × 4   (rolling_std > 훈련구간 중위수)
  → {0..7}: 변동성 레짐이 높을 때 에이전트가 별도 정책 학습

Vanilla RL — 2개 이산 상태:
  State 0: 하락 (ret ≤ 0)
  State 1: 상승 (ret > 0)

신경망 RL — 연속 상태:
  s_t = [ret, ema_ratio, vol, momentum, trend]   ← 5차원 벡터
```

EMA_10 (10일 지수이동평균): `df['Close'].ewm(span=10, adjust=False).mean()`

---

## 5. 워크포워드 검증 — Train/Test 분리

**파일:** `common/base_agent.py`

### Trading Days vs Train Episodes

| 파라미터 | 의미 | 기본값 |
|---------|------|--------|
| Trading Days (`n_bars`) | yfinance 데이터 봉 수 (창 크기) | 500 (일봉 약 2년) |
| Train Episodes | 훈련 데이터 반복 학습 횟수 | 300 |

```
Trading Days = 500봉 전체 창
      │
      ├── 학습 구간: 앞 70% = 350봉 (Train Episodes = 300회 반복)
      │
      └── 평가 전체: 500봉 (인샘플 70% + OOS 30%)

n_train = max(int(n_days * 0.7), 20)   # 최소 20봉 학습 보장
```

| 기간 | 일봉 500일 |
|------|-----------|
| 학습 구간 | 350일 (~1년 5개월) |
| 평가 전체 | 500일 (~2년) |
| OOS 구간 | 마지막 150일 (~6개월) |

### 워크포워드 구조적 한계 (KOSPI/KOSDAQ)

KOSPI·KOSDAQ는 학습 구간(2024~2025 상반기)이 횡보/하락이고, OOS 구간(2025 하반기~2026)에 급등이 집중된다. 이 경우 에이전트는 조용한 시장을 학습했으므로 급등 OOS를 추월하기 어렵다. 이는 알고리즘 문제가 아닌 구조적 한계다.

---

## 6. 하이퍼파라미터 탐색 — PG Actor-Critic Optimizer

**파일:** `common/heuristic.py` — `PGActorCriticOptimizer`

`(lr, gamma, epsilon_static, epsilon_vanilla)` 4차원 파라미터를 자동 탐색하여 복합 Gap 기대값을 극대화한다.

### 복합 Gap 목표 함수

```python
gap_vs_market  = STATIC_final − Market_final          # 시장 초과수익 (60% 가중)
V_floor        = Market_final × 0.3                   # Vanilla 하한 (역유인 방지)
V_adj          = max(Vanilla_final, V_floor)
gap_vs_vanilla = STATIC_final − V_adj                 # Vanilla 대비 우위 (40% 가중)

composite_gap  = 0.6 × gap_vs_market + 0.4 × gap_vs_vanilla
```

**V_floor의 역할 (역유인 방지)**: Vanilla가 0%일 때 gap_vs_vanilla가 최대 → 옵티마이저가 Vanilla를 의도적으로 망가뜨리는 구조를 차단. Vanilla ≥ Market×30%를 하한으로 보정.

### 알고리즘 이론

```
탐색 공간: {lr, gamma, epsilon_static, epsilon_vanilla} → 정규화 공간 [0,1]^4

1. 다음 후보 파라미터 샘플링 (Actor, Gaussian):
   Δ = N(0, σ),  x_new = clip(μ + Δ, 0, 1)

2. 복수 평가 시드로 composite_gap 측정:
   expected_gap = mean([composite_gap(seed_i) for seed_i in eval_seeds])

3. Critic (EMA baseline):
   V += value_alpha × (expected_gap − V)

4. Advantage 정규화:
   A_norm = tanh((expected_gap − V) / 10)   → [−1, 1]

5. Actor (Policy Gradient):
   pg_dir = clip(Δ / σ, L2≤1)
   μ = clip(μ + lr_actor × A_norm × pg_dir, 0, 1)

6. σ 자동 스케줄링:
   A_norm > 0 → σ × 0.96  (좋은 방향으로 수렴)
   A_norm ≤ 0 → σ × 1.04  (더 넓게 재탐색)
```

### 탐색 페이즈

| 단계 | 조건 | 설명 |
|------|------|------|
| PG Exploring | step < n_iters/4 | 광역 탐험 |
| PG Actor-Critic | σ_mean > 0.12 | Policy Gradient 업데이트 |
| PG Converging | σ_mean ≤ 0.12 | 정밀 수렴 단계 |

---

## 7. 웹 UI 파라미터 상세

아래 모든 파라미터는 웹 UI에서 조절 가능하며, Fallback Parameters 패널에서 전체 종목에 일괄 적용할 수 있다.

### 7.1 RL Algorithm 선택 (System Parameters 2행 첫 번째)

| 옵션 | 설명 |
|------|------|
| STATIC | EMA 기반 4/8-상태 Tabular Actor-Critic (기본) |
| STATIC_H | STATIC + Tabular PPO Clipping + Adaptive Temperature (Hybrid) |
| A2C | 신경망 Actor-Critic (1-step TD Advantage) |
| A3C | 신경망 Actor-Critic (n-step Return, n=5) |
| PPO | Clipped Surrogate + GAE (clip_eps=0.2, λ=0.95) |
| ACER | Retrace(λ) 오프-정책 + Truncated IS + 편향 보정 (ReplayBuffer) |
| SAC | Soft Actor-Critic + 자동 온도 α (이산 행동) |
| DDPG | 결정론적 정책 + OU noise + ReplayBuffer (연속 행동) |

- STATIC / STATIC_H 선택 시 State Analysis 탭 사용 가능 (θ 행렬 시각화)
- 신경망 알고리즘 선택 시 State Analysis 탭 비활성화 (`s_theta=None`)
- STATIC / STATIC_H 선택 시 Per-stock 위젯(8-State, Roll Period) 활성화; 신경망 알고리즘 선택 시 비활성화

### 7.2 RL 학습 파라미터

| 파라미터 | 탐색 범위 | 증가 효과 | 감소 효과 |
|---------|----------|---------|---------|
| **LR (α)** | 0.005 ~ 0.10 | 빠른 학습, 진동 위험 | 안정적이지만 느린 수렴 |
| **Gamma (γ)** | 0.85 ~ 0.99 | 장기 보상 중시, 수렴 느림 | 단기 편향, 빠른 수렴 |
| **STATIC ε** | 0.01 ~ 0.25 | 더 많은 탐험, 낮은 착취 | 더 많은 착취, 초반 고착 위험 |
| **Vanilla ε** | 0.01 ~ 0.25 | Vanilla 탐험 증가 | Vanilla 착취 증가 |

**LR**: 너무 크면 Q 값/θ가 진동하여 수렴 불가. 너무 작으면 수렴까지 에피소드 수 과다.

**Gamma**: γ=0.99는 약 100스텝 미래 보상까지 고려. γ=0.85는 약 7스텝. 주식처럼 수익이 며칠에 걸쳐 실현되는 환경에서는 γ ≥ 0.85가 적합.

**Epsilon**: 상한 0.25 초과 시 탐험이 너무 많아 학습된 정책보다 랜덤에 가까워져 수렴이 지연된다.

### 7.3 시스템 파라미터

| 파라미터 | 기본값 | 역할 | 증가 효과 | 감소 효과 |
|---------|--------|------|---------|---------|
| **Trading Days** | 500 (일봉) | yfinance 데이터 봉 수 | 더 긴 역사 학습 | 빠른 실행, 최신 데이터만 |
| **Train Episodes** | 300 | 훈련 데이터 반복 횟수 | 더 충분한 학습 (과적합 위험) | 빠른 실행, 과소 학습 위험 |
| **Base Seed** | 멤버별 | 재현성 난수 시드 | — | — |
| **Auto Run Count** | 6 | Run Evaluation 자동 반복 | 더 많은 Trial 축적 | 빠른 단일 실행 |
| **Sim Min Steps** | 30 | Simulation n_iters 하한 | 더 많은 탐색 스텝 | 빠른 시뮬레이션 |
| **Sim Step Mult.** | 10 | n_iters = max(Min, Count × Mult) | 탐색 밀도 증가 | 빠른 시뮬레이션 |
| **Timeframe** | 1d (일봉) | 데이터 봉 단위 | — | — |
| **Frame Speed** | 설정값 | 차트 애니메이션 속도 | 느린 시각화 | 빠른 시각화 |

**Trading Days 증가**: 학습 구간이 길어지므로 에이전트가 더 다양한 시장 국면을 학습한다. 단, 오래된 데이터가 현재 시장 패턴과 다를 수 있다.

**Train Episodes 증가**: 동일 데이터를 더 많이 반복 학습하여 정책 수렴도 향상. 과도하면 훈련 데이터에 과적합될 수 있다.

**Sim Min Steps + Sim Step Mult**: 총 시뮬레이션 반복 횟수 = `max(Sim_Min, Auto_Run_Count × Sim_Step_Mult)`. 기본 `max(30, 6×10)=60회`, 각 회차마다 3~4 시드 평가 → 총 180~240회 RL 실행.

### 7.4 Fallback Parameters (사이드바)

| 파라미터 | 체크 시 동작 |
|---------|------------|
| RL Algorithm | 선택된 알고리즘을 모든 종목에 일괄 적용 |
| LR / Gamma / ε(S) / ε(V) | 글로벌 기본값으로 전체 종목 대체 |
| Trading Days / Train Episodes / Seed | 데이터/학습 설정 일괄 변경 |
| Active Agents | STATIC/Vanilla 에이전트 구성 통일 |
| Sim Min Steps / Sim Step Mult. | 시뮬레이션 반복 횟수 일괄 변경 |

### 7.5 팀 포트폴리오 파라미터 (Dashboard)

| 파라미터 | 기본값 | 역할 | 증가 효과 | 감소 효과 |
|---------|--------|------|---------|---------|
| **Fund Temperature** | 1.0 | Softmax 온도 (1.0~5.0) | 균등 배분에 가까워짐 | 최고 성과 멤버에 집중 |
| **Max Weight** | 1.0 | 단일 멤버 최대 비중 (0.1~1.0) | 집중 허용 | 분산 강제 |

**Fund Temperature**: Softmax 가중치 = `exp(score/T) / Σ exp(score_i/T)`. T가 작을수록 최고 score 멤버에 자본이 집중된다. T=1.0(기본)은 score 차이를 그대로 반영.

### 7.6 CTPT 성향 코드

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

### 7.7 Per-stock 위젯 (종목별 개별 설정)

**파일:** `app.py` — 멤버별 탭 내 System Parameters 하단 위젯 행

System Parameters 2행 아래에 종목별 상태 공간 설정을 위한 전용 위젯이 추가되었다.

| 위젯 | 종류 | 세션키 패턴 | 설명 |
|------|------|------------|------|
| 8-State (Vol) | 체크박스 | `use_vol_{m_name}_{stock_name}` | 체크 시 변동성 레짐 추가 → 8-상태 (4-상태 기본) |
| States | 표시 | — | 현재 상태 수 표시 (4 또는 8) |
| Roll Period (봉) | 셀렉트박스 | `roll_period_{m_name}_{stock_name}` | Rolling_Std 계산 창 (None/10/20/30/60/90봉) |

- **STATIC / STATIC_H 선택 시**: 위젯 활성화, 종목별 독립적 설정 가능
- **신경망 알고리즘 선택 시**: 위젯 `disabled=True` (tabular 전용 기능이므로)
- **Fallback Parameters**: 전역 `use_vol_feature` / `roll_period` 세션 상태로 전체 종목 일괄 오버라이드

#### use_vol (8-State) 동작

```
use_vol = False (기본):
  state = is_bull×1 + is_above_ema×2  → {0, 1, 2, 3}  (4개 상태)

use_vol = True:
  vol_threshold = 훈련 구간 Rolling_Std 중위수  (자동 산출)
  is_high_vol   = Rolling_Std[t] > vol_threshold
  state = is_bull×1 + is_above_ema×2 + is_high_vol×4  → {0..7}  (8개 상태)
```

변동성 레짐(고/저)을 별도 차원으로 분리하여 에이전트가 고변동성 구간에서 다른 정책을 학습할 수 있다.

#### roll_period 동작

```
roll_period = None (기본):
  데이터 로더(data_loader.py)의 기본 Rolling_Std 사용

roll_period = N (10/20/30/60/90봉):
  Rolling_Std = Close.pct_change().rolling(N, min_periods=2).std()
  → 짧은 창: 최근 변동성에 민감 / 긴 창: 평활화된 장기 변동성
```

#### 현재 멤버별 설정 (improve 7-2 기준)

| 멤버 | 종목 | use_vol | roll_period |
|------|------|---------|------------|
| M1 | SPY | O (8-State) | None |
| M2 | QQQ | — (4-State) | None |
| M3 | KOSPI | O (8-State) | None |
| M4 | KOSDAQ | O (8-State) | 30봉 |
| M5 | SCHD | O (8-State) | 60봉 |
| M6 | RGLD | O (8-State) | None |

---

## 8. 랜덤 시드의 역할

### 훈련 시드 (Base Seed)

```python
np.random.seed(seed)  # 훈련 시작 전 고정
```

epsilon-greedy 탐험 경로를 고정하여 동일 시드에서 항상 동일한 훈련 궤적이 재현된다.

| 종목 | 시드 | 선택 근거 |
|------|------|---------|
| SPY | 42 | 안정 지수에 적합한 수렴성 |
| QQQ | 137 | 기술주 고분산 환경에서 안정 수렴 확인 |
| KOSPI | 2024 | 국내 시장 리듬 친화적 연도 기반 시드 |
| KOSDAQ | 777 | 고변동성 시장, 탐험 다양성 확보 |
| SCHD | 314 | 배당 ETF 방어적 성향 |
| RGLD | 99 | 넓은 탐험 범위 |

### 복수 평가 시드 (Simulation)

```python
_n_eval     = min(4, max(3, Auto_Run_Count // 2))   # 기본 3개 시드
_eval_seeds = [base_seed + j * 37 for j in range(_n_eval)]
expected_gap = mean([composite_gap(seed_i) for seed_i in _eval_seeds])
```

특정 시드의 우연에 의존하지 않는 일반화된 기대값을 산출한다.

### Trial 시드

```python
trial_seed = base_seed + (n_accumulated + run_i) * 37
```

소수(prime) ×37 간격: np.random LCG 특성상 +1 간격 시드들은 내부 상태가 유사하여 학습 궤적이 거의 동일해진다. ×37 간격은 Trial마다 실질적으로 다른 탐험 경로를 보장한다.

---

## 9. 데이터 파이프라인

**파일:** `common/data_loader.py`

### 지원 봉 단위

| 봉 단위 | interval | 최대 기간 | 기본 Trading Days |
|--------|---------|---------|-----------------|
| 15분봉 | 15m | 60일 | 80 |
| 1시간봉 | 1h | 730일 | 120 |
| 일봉 | 1d | 2년 | **500** |
| 주봉 | 1wk | 10년 | 105 |
| 월봉 | 1mo | 10년 | 24 |

### 전처리 흐름

```
yf.download() → (실패 시) yf.Ticker().history()
    │
    ↓
MultiIndex 컬럼 정리 + 중복 제거
    │
    ↓
인덱스 처리: 인트라데이(15m/1h) → datetime  │  일봉 이상 → .date
    │
    ↓
EMA_10          = Close.ewm(span=10, adjust=False).mean()
Rolling_Std     = Close.pct_change().rolling(roll_period, min_periods=2).std()
Daily_Return    = Close.pct_change()
    │
    ↓
dropna() 최종 정리
```

`@st.cache_data(ttl=3600)` — 동일 티커 + 봉 단위 조합을 1시간 동안 캐싱한다.

---

## 10. 포트폴리오 평가 및 성과 지표

### 개별 종목 지표

| 지표 | 계산 방법 |
|------|---------|
| Final Return (%) | 누적 수익률 배열의 마지막 값 |
| Alpha Gap (%p) | STATIC RL − Vanilla RL 최종 수익률 차이 |
| MDD (%) | min((wealth_index − running_peak) / running_peak) × 100 |
| Volatility | 누적 수익률 배열의 표준편차 |

### 팀 포트폴리오 — Softmax 가중 배분

**파일:** `common/evaluator.py`, `app.py`

```python
# 위험 조정 점수
score_i  = avg_return_i / (1 + abs(avg_mdd_i))   # MDD 패널티 적용 수익률

# Softmax 가중치 (수치 안정: z -= z.max())
weight_i = softmax(scores, temperature=T)[i]

# 팀 펀드 수익 곡선
team_curve = np.dot(weights_arr, aligned_traces)   # (n_members,) · (n_members, n_days)
```

### 행동 및 보상 함수

| 행동 | 코드 | 의미 |
|------|------|------|
| CASH | 0 | 현금 보유 (수익률 0%) |
| BUY | 1 | 매수·보유 (당일 수익률 반영) |

- SELL 액션 없음. BUY→CASH 전환이 암묵적 청산.
- 수수료는 **CASH→BUY 진입 시에만** 1회 부과.

```python
fee    = fee_rate if (action == BUY and prev_action == CASH) else 0
reward = daily_return − fee   # 클리핑 없음, 실제 수익률 그대로
current_capital *= (1 + reward)
```

### Ghost Line

Simulation에서 발견된 최적 파라미터 수익 곡선을 점선으로 현재 차트에 병렬 표시한다.

---

## 11. UI 기능 상세

### 11.1 사이드바

| 기능 | 동작 |
|------|------|
| Eval. All | 전체 멤버·종목을 현재 파라미터로 순차 평가 |
| Simul. All | 전체 멤버·종목 최적 파라미터 자동 탐색 후 저장 및 평가 |
| All 적용 | 체크된 Fallback Parameters를 모든 종목에 일괄 적용 |
| 되돌리기 | 체크된 파라미터를 이전 상태로 복원 |
| System Status | Cloud/Local 환경 + GPU/CPU 상태 표시 |

### 11.2 Run Evaluation

```
1. 현재 파라미터로 RL 에이전트 평가
2. Trial History에 결과 추가 (trial_seed = base_seed + n*37)
3. Auto Run Count만큼 자동 반복
4. 각 Trial의 수익률, Alpha Gap, MDD 통계 분석 패널에 표시
```

#### 평가 루프 전체 수식

```
초기화:
  C_0 = 1.0       ← 정규화 자본 (1 = 100%)
  prev_action = 0 (CASH)

t = 1, 2, ..., n_days − 1:

  ① 행동 결정 (greedy, ε 없음):
       STATIC  : a_t = argmax_a θ[s_t, a]
       Vanilla : a_t = argmax_a Q[s_t, a]
       A2C/A3C/PPO/SAC : a_t = argmax softmax(Actor(s_t))
       DDPG    : a_t = 1[sigmoid(Actor(s_t)) ≥ 0.5]

  ② 수수료:
       fee_t = fee_rate   if (a_t == BUY and prev_a == CASH)
             = 0          otherwise

  ③ 보상:
       r_t = Daily_Return[t] − fee_t   (a_t == BUY)
           = 0                          (a_t == CASH)

  ④ 자본 갱신:
       C_{t+1} = C_t × (1 + r_t)

  ⑤ 누적 수익률:
       cumulative_return[t] = (C_t − 1) × 100    [%]

성과 지표:
  Alpha_Gap = STATIC_final − Vanilla_final         [%p]
  MDD       = min_t { (C_t − max_{k≤t} C_k) / max_{k≤t} C_k } × 100   [%]
```

### 11.3 Simulation

```
1. PGActorCriticOptimizer가 n_iters 반복 탐색
   n_iters = max(Sim Min Steps, Auto Run Count × Sim Step Mult.)
2. 각 iteration: 복수 시드(3~4)로 RL 평가
   → composite_gap = 0.6×(STATIC-Market) + 0.4×(STATIC-max(Vanilla, Market×0.3))
3. Policy Gradient로 탐색 정책 μ 업데이트, σ 자동 스케줄링
4. 수렴 차트 실시간 표시 (파라미터 추이 + Alpha vs Market)
5. 탐색 완료 후 저장 여부 선택 → config.py 자동 저장 + 모듈 리로드
```

### 11.4 Agent Decision Analysis

- 좌측: STATIC RL의 BUY/CASH 행동 빈도 막대 차트
- 우측: 일별 행동 로그 테이블 (BUY: 파란색, CASH: 빨간색)
- 신경망 알고리즘(A2C 등) 선택 시 State Analysis 탭 비활성화

### 11.5 Trial History Statistical Analysis

- **Trial-by-Trial Return Progression**: 반복 평가별 수익률 추이
- **Return Distribution across Trials**: Vanilla RL vs STATIC RL 박스 플롯
- **Statistics Summary**: Mean(σ), Range 항목별 표시
- **Trial 데이터 테이블**: 열 줄바꿈 헤더, 전체 열(Trial/Seed/Vanilla/STATIC/Market) 표시

### 11.6 팀 포트폴리오 대시보드

- All Members STATIC RL Cumulative Returns + Team Fund 차트
- 멤버별 성과 테이블: Member, Stocks, Persona(CTPT), Capital($), STATIC(%), Vanilla(%), Alpha Gap, MDD, Score, Weight%
- Master Fund Contribution (Donut): Softmax score 기반 자본 비중
- Profit Comparison (Bar): 멤버별 Vanilla RL vs STATIC RL 손익

---

## 12. 파일 구조

```
Task-Constrained-RL-ColdStart_YS_v1-1/
│
├── app.py                          Streamlit 메인 앱 (~2300줄)
│
├── common/
│   ├── base_agent.py               STATIC/Vanilla/Neural RL 훈련·평가
│   ├── nn_utils.py                 TinyMLP (NumPy), ReplayBuffer, extract_features
│   ├── heuristic.py                PGActorCriticOptimizer
│   ├── evaluator.py                MDD, Softmax 비중, CTPT 코드
│   ├── data_loader.py              yfinance 데이터 로드 (다봉, 캐시)
│   └── stock_registry.py           종목 정보(12종) + 수수료 테이블
│
├── members/
│   ├── member_1/config.py          Member 1 — SPY    (seed=42,  lr=0.0802, gap=8.66%)
│   ├── member_2/config.py          Member 2 — QQQ    (seed=137, lr=0.0722, gap=35.62% 🏆)
│   ├── member_3/config.py          Member 3 — KOSPI  (seed=2024,lr=0.0050, 구조적 OOS)
│   ├── member_4/config.py          Member 4 — KOSDAQ (seed=777, lr=0.0380, gap=21.36%)
│   ├── member_5/config.py          Member 5 — SCHD   (seed=314, lr=0.0624, gap=0.57%)
│   └── member_6/config.py          Member 6 — RGLD   (seed=99,  lr=0.0269, gap=17.81%)
│
└── README.md
```

---

## 13. Simulation 단계별 연산 흐름

```
[Simulation 클릭]
      │
STEP 1: n_iters = max(Sim_Min, AutoRun×Mult), param_bounds 결정
      │  lr:(0.005,0.10) / gamma:(0.85,0.99) / epsilon:(0.01,0.25) / v_epsilon:(0.01,0.25)
      ↓
STEP 2: PGActorCriticOptimizer 초기화
      │  sigma_init=0.18, sigma_max=0.30, lr_actor=0.12, value_alpha=0.25
      ↓
STEP 3: _n_eval = min(4, max(3, auto_runs//2)),  eval_seeds = [seed + j*37]
      │
  ┌── for i in range(n_iters): ──────────────────────────────────────┐
  │  STEP 4: 페이즈 판정 (Exploring / Actor-Critic / Converging)      │
  │  STEP 5: Δ ~ N(0,σ),  x_new = clip(μ+Δ, 0,1) → candidate 4개   │
  │  STEP 6: 복수 시드 RL 평가 (3~4 seeds)                           │
  │    • STATIC AC 훈련(첫 70%, 300회) → 전체 기간 평가              │
  │    • Vanilla QL 훈련(첫 70%, 150회) → 전체 기간 평가              │
  │    • composite_gap = 0.6×(S-M) + 0.4×(S-max(V, M×0.3))          │
  │  STEP 7: expected_gap = mean(gaps)                                │
  │  STEP 8: V += value_alpha × (expected_gap − V)    (Critic EMA)   │
  │  STEP 9: A_norm = tanh((expected_gap − V) / 10)                  │
  │  STEP 10: pg_dir = Δ/σ (L2≤1);  μ += lr_actor × A_norm × pg_dir │
  │  STEP 11: A_norm > 0 → σ×0.96;  A_norm ≤ 0 → σ×1.04            │
  │  STEP 12: best 갱신, 수렴 차트 실시간 업데이트                    │
  └────────────────────────────────────────────────────────────────────┘
      │
Simul. All 모드 → best 파라미터 config.py 자동 저장 → 모듈 리로드 → Run Evaluation 실행
수동 모드       → [저장 및 반영] / [반영 취소] 버튼 표시
```

---

## 14. 개선 이력

### improve 7-2 (현재) — STATIC-H 하이브리드 알고리즘 + 차트 폰트 개선

| # | 변경 내용 |
|---|---------|
| H1 | `common/base_agent.py` 신규 상수: `HYBRID_ALPHA_MIN=0.01`, `HYBRID_ALPHA_MAX=0.15`, `HYBRID_FLIP_WIN=20` |
| H2 | `common/base_agent.py` 신규: `_train_actor_critic_hybrid()` — Tabular PPO Clipping + Adaptive Temperature α_t |
| H3 | `common/base_agent.py` `run_rl_simulation_with_log()` 시그니처 변경: `algorithm="STATIC"` 파라미터 추가 |
| H4 | `common/base_agent.py` 훈련 라우팅 분기: `algorithm=="STATIC_H"` → `_train_actor_critic_hybrid`, 나머지 → `_train_actor_critic_static` |
| H5 | `app.py` `_ALGO_OPTS` 업데이트: `["STATIC", "STATIC_H", "A2C", "A3C", "PPO", "ACER", "SAC", "DDPG"]` |
| H6 | `app.py` 카드 레이블: `STATIC_H` → "HYBRID RL" / "HYBRID" 표시 |
| H7 | `app.py` 알고리즘 분기: `algorithm in ("STATIC", "STATIC_H")` → `run_rl_simulation_with_log(..., algorithm=algorithm)` |
| H8 | `app.py` `_is_static_algo` 조건: `l_algorithm in ("STATIC", "STATIC_H")` → Per-stock 위젯 활성화 여부 결정 |
| C1 | 누적 수익 차트 폰트 확대: title 18→22, axis 14→16, tick 14(신규), legend 12→14 |
| C2 | Parameter Convergence 차트 폰트 확대: title 12→15, axis/tick 14/13, legend 12→13 |
| C3 | Alpha vs Market 차트 폰트 확대: title 12→15, axis/tick 14/13, legend 11→13, annotation 10→12 |

---

### improve 7-1 — Per-stock UI 위젯 (8-State, States 표시, Roll Period)

| # | 변경 내용 |
|---|---------|
| W1 | `app.py` System Parameters 2행 아래 위젯 행 신규 추가 (3열 구성) |
| W2 | `app.py` 위젯: `8-State (Vol)` 체크박스 (`use_vol_{m_name}_{stock_name}`) |
| W3 | `app.py` 위젯: `States` 표시 (4 또는 8, STATIC/STATIC_H 전용) |
| W4 | `app.py` 위젯: `Roll Period (봉)` 셀렉트박스 (`roll_period_{m_name}_{stock_name}`, 옵션: None/10/20/30/60/90) |
| W5 | `app.py` per-member config 초기값 읽기: `_per_use_vol`, `_per_roll` 세션 상태 키 패턴 추가 |
| W6 | `app.py` `_is_static_algo` 조건: 신경망 알고리즘 선택 시 위젯 `disabled=True` |
| W7 | `members/member_N/config.py` (M1~M6): `use_vol`, `roll_period` 필드 최종값 반영 |

---

### improve 6-1 — 신경망 RL 알고리즘 추가 + 버그 수정

| # | 변경 내용 |
|---|---------|
| N1 | `common/nn_utils.py` 신규: TinyMLP (He+Adam), ReplayBuffer, extract_features (5차원 특징) |
| N2 | `common/base_agent.py` 신규: `_train_a2c()`, `_train_a3c()`, `_train_ppo()`, `_train_sac()`, `_train_ddpg()` |
| N2a | `common/nn_utils.py` ReplayBuffer 확장: `_log_probs` 필드, `push(log_prob=0.0)`, `sample_with_logp()` (ACER용) |
| N2b | `common/base_agent.py` 신규: `_train_acer()` — Retrace(λ) + Truncated IS + 편향 보정 |
| N2c | `app.py` RL Algorithm 순서 재배치: STATIC→A2C→A3C→PPO→**ACER**→SAC→DDPG (온-정책→오프-정책 흐름) |
| N3 | `common/base_agent.py` 신규: `run_neural_rl()` 공개 API |
| N4 | `app.py` RL Algorithm selectbox 추가 (System Parameters 2행, LR 왼쪽) |
| N5 | `app.py` Fallback Parameters에 RL Algorithm 체크박스 추가 |
| N6 | `get_rl_data()` algorithm 파라미터 추가, 신경망 분기 처리 |
| N7 | M5 → SCHD (미국배당다우존스 ETF, index=10) 으로 변경 |
| N8 | M6 → RGLD (로열 골드, index=11) 으로 변경 |
| B1 | A3C bootstrap γ 오류 수정: `R = gamma * V(s_n)` → `R = V(s_n)` |
| B2 | `_ALL_CHK_KEYS`에 `"algo"`, `"sim_min"`, `"sim_mult"` 누락 → 추가 |
| B3 | `sim_pending` algo 세션 키 누락 → Simulation 후 Algorithm selectbox 복원 수정 |
| B4 | `_save_sim_params_to_config()` 저장 후 `importlib.reload(m_config)` 추가 (캐시 갱신) |
| C1 | M1 SPY config 갱신: lr=0.0802, γ=0.8732, ε=0.1667, v_ε=0.1347 (gap=8.66%) |
| C2 | M5 SCHD config 초기 최적화: lr=0.0624, γ=0.9449, ε=0.1708, v_ε=0.0879 (gap=0.57%) |
| C3 | M6 RGLD config 초기 최적화: lr=0.0269, γ=0.9380, ε=0.1509, v_ε=0.1268 (gap=17.81%) |

---

### improve 4-9 — Q-floor margin 강화 + config 최적값 갱신

| # | 문제 | 수정 내용 |
|---|------|---------|
| V1 | Vanilla CASH 고착 지속 (margin=0.001 < 훈련 노이즈) | margin 0.001→0.005 강화 |
| C1 | M1 SPY config 미갱신 | lr=0.0496, γ=0.8863, ε=0.1190, v_ε=0.0993 |
| C2 | M2 QQQ config 미갱신 | lr=0.0650, γ=0.9075, ε=0.1005, v_ε=0.1043 |

---

### improve 4-8 — Vanilla 전체 상태 Q-floor + STATIC entropy 강화

| # | 문제 | 수정 내용 |
|---|------|---------|
| V1 | Vanilla bear-state(0) OOS 전 구간 CASH 고착 | Q[0,BUY] ≥ Q[0,CASH]+0.001 추가 |
| V2 | STATIC 과결정론적 수렴 (entropy_coeff=0.02 부족) | entropy_coeff 0.02→0.05 |

---

### improve 4-5~4-7 — fee 비례 초기화 + 환경 감지 + 안정화

| # | 변경 내용 |
|---|---------|
| 4-5 | STATIC/Vanilla fee 비례 BUY 선호 초기화 |
| 4-5 | Cloud/Local 환경 자동 감지 (`_IS_CLOUD`) + System Status 배지 |
| 4-6 | STATIC: 엔트로피 정규화 `r_eff = r + 0.02·H(π)` (→4-8에서 0.05로 강화) |
| 4-6 | Vanilla: epsilon annealing 2ε→ε + prev_action=1 고정 |
| 4-7 | Vanilla: Q[1,BUY] ≥ Q[1,CASH]+0.001 상대 우위 하한 |

---

### improve 4-1~4-4 — 시뮬레이션 안정화

| # | 변경 내용 |
|---|---------|
| 4-1 | Trial seed 간격 ×13→×37 소수 간격 (시드 독립성) |
| 4-2 | 역유인 제거: V_floor = Market×0.3 |
| 4-2 | LR 탐색 하한 0.001→0.005, sigma_max 0.45→0.30 |
| 4-3 | epsilon 탐색 상한 0.5→0.25 (경계값 고착 방지) |
| 4-4 | Sim Min Steps / Sim Step Mult. UI 파라미터 추가 |

---

### improve 2~3 — 알고리즘 전환 및 기초 구조

- Actor-Critic 알고리즘 전환 (단순 Q-Learning → Policy Gradient 기반)
- BayesianOptimizer → PGActorCriticOptimizer 전환
- 복합 Gap 목표 함수 도입 (0.6×Market + 0.4×Vanilla)
- 워크포워드 검증: n_train = max(int(n_days×0.7), 20)
- Trading Days / Train Episodes 독립 파라미터 분리
- Trial History Statistical Analysis 패널 추가

---

## 15. RL Algorithm 시뮬레이션 학습 가이드

각 알고리즘을 시뮬레이터에서 어떻게 적용하고 관찰해야 강화학습의 핵심 개념을 체감할 수 있는지 정리한다.

### 공통 관찰 포인트

Run Evaluation 실행 후 아래 세 가지를 반드시 확인한다.

| 관찰 항목 | 의미 |
|---------|------|
| **누적 수익 곡선** | STATIC/신경망 vs Market vs Vanilla 3선 비교 |
| **Action Frequency** | BUY/CASH 비율 — 한쪽 고착 여부 확인 |
| **Alpha Gap** | 양수 = Vanilla 대비 우위 달성, 음수 = 고착/미수렴 |

---

### 1. STATIC (Tabular Actor-Critic) — 기준 알고리즘 / 1H. STATIC_H (Hybrid) — 안정화 확장

**학습 원리**: 시장을 4개 상태(EMA 위/아래 × 상승/하락)로 단순화하여 상태별 BUY/CASH 확률(θ)을 Policy Gradient로 학습한다.

**실험 방법**:
- lr을 0.01 → 0.05 → 0.10으로 높이면서 수익 곡선 안정성 변화 관찰
- epsilon을 0.05 → 0.20으로 높이면 Action Frequency에서 CASH 비율 증가 확인

**성공 지표**: Gap ≥ 1%p (시장 초과), BUY:CASH 비율 90:10 ~ 70:30 범위

**핵심 개념 확인**: State Analysis 탭에서 θ[3,1](EMA위+상승 BUY)이 θ[0,1](EMA아래+하락 BUY)보다 크게 학습됨을 확인 → 상태별 정책 분화

#### 1H. STATIC_H (Hybrid) — Tabular PPO Clipping + Adaptive Temperature

**학습 원리**: STATIC과 동일한 4/8-상태 Tabular AC 구조에 Tabular PPO Clipping(정책 급변 억제)과 Adaptive Temperature(행동 전환 빈도 기반 α_t)를 추가하여 안정성을 높인다.

**실험 방법**:
- STATIC과 동일 파라미터로 STATIC_H 실행 → 수렴 안정성 비교 (특히 KOSPI처럼 BUY 95% 고착이 발생하던 종목)
- 8-State(Vol) 체크박스를 활성화하면 변동성 레짐별 별도 정책 학습 여부 확인
- Roll Period를 None → 30봉 → 60봉으로 변경하여 Rolling_Std 창 크기가 8-State 분류에 미치는 영향 관찰

**STATIC 대비 차이**: 단일 상태에서 정책이 BUY 100%로 수렴하는 것을 Clip이 억제한다. flip_rate가 낮아지면 α_t도 낮아져 안정적 착취 위주로 전환된다.

**핵심 개념 확인**: Tabular PPO step_scale=0.3 적용 시 파라미터 수렴 차트에서 θ 값이 STATIC보다 완만하게 변화하는 것을 확인 → 정책 안정화

**Per-stock 위젯 활용 팁**:
- `8-State (Vol)` 체크 → States가 4→8로 변경됨을 즉시 확인
- `Roll Period` 변경 시 Rolling_Std 계산 창이 바뀌어 고변동성/저변동성 경계가 달라짐
- 신경망 알고리즘(A2C 등) 선택 시 위젯이 자동 비활성화됨을 확인

---

### 2. Vanilla (Q-Learning) — 비교 기준선

**학습 원리**: 2상태(상승/하락)에서 Q 테이블로 BUY vs CASH 기대수익을 직접 추정한다.

**실험 방법**:
- STATIC과 동일 파라미터 적용 후 수익 곡선 차이 비교
- Vanilla가 음수면 CASH 고착 → v_epsilon을 높여 탐험 강제

**성공 지표**: Vanilla > 0%, STATIC > Vanilla (Alpha Gap 양수)

**핵심 개념 확인**: 4상태(STATIC)가 2상태(Vanilla)보다 우위인 이유 — EMA 신호가 상태 공간에 추가 정보를 제공하여 더 정교한 정책 학습 가능

---

### 3. A2C (Advantage Actor-Critic) — 신경망 진입점

**학습 원리**: STATIC과 동일한 Actor-Critic 구조이지만, 상태를 이산화하지 않고 5차원 연속 벡터로 표현한다. TinyMLP가 비선형 패턴을 직접 학습한다.

**실험 방법**:
- Trading Days를 200 → 500 → 1000으로 늘리면서 성능 변화 관찰
- Train Episodes를 100 → 300으로 늘리면 수렴 안정성 향상 확인

**STATIC 대비 차이**: 상태 공간이 연속이라 시장 패턴이 복잡할수록 유리하다. 단순 지수(SPY, SCHD)에서는 STATIC과 비슷하거나 낮을 수 있다.

**핵심 개념 확인**: 이산 상태 vs 연속 특징 벡터의 표현력 차이 — 신경망이 더 많은 데이터를 요구하는 대신 더 풍부한 패턴 학습 가능

---

### 4. A3C (n-step Return) — A2C의 편향 개선

**학습 원리**: 1-step TD(A2C) 대신 5-step 미래 보상을 직접 합산하여 Advantage를 추정한다. 단기 편향을 줄인다.

**실험 방법**:
- A2C와 동일 파라미터로 실행하여 수익 곡선 비교
- 변동성 큰 종목(RGLD)에서 A2C vs A3C 차이가 더 뚜렷하게 나타남
- gamma를 높이면(≥ 0.95) n-step 효과 극대화

**A2C 대비 차이**: 보상이 며칠에 걸쳐 발생하는 주식 환경에서 n-step이 유리하다.

**핵심 개념 확인**: 편향(bias)–분산(variance) 트레이드오프

| 방식 | 편향 | 분산 |
|------|------|------|
| 1-step TD (A2C) | 고편향 | 저분산 |
| n-step (A3C, n=5) | 중간 | 중간 |
| Monte Carlo (전체) | 저편향 | 고분산 |

---

### 5. PPO (Proximal Policy Optimization) — 안정성 최고

**학습 원리**: 정책 업데이트 폭을 clip(1-ε, 1+ε)로 제한하여 한 번에 너무 크게 변하지 않도록 제어한다. 신경망 RL 중 가장 안정적이다.

**실험 방법**:
- Train Episodes를 적게(100) 설정해도 수렴 안정성이 A2C보다 높음을 확인
- lr을 0.005 → 0.05로 높여도 발산하지 않는 범위 확인

**신경망 RL 입문 추천 순서**: A2C → PPO → SAC → DDPG

**핵심 개념 확인**: Importance Sampling 비율 r_t(θ)가 클수록(구 정책과 차이가 클수록) Clip이 기울기를 차단 → 점진적 안전한 학습

---

### 6. ACER (Actor-Critic with Experience Replay) — 오프-정책 안정성

**학습 원리**: 온-정책 롤아웃을 ReplayBuffer에 저장하고, Retrace(λ)로 오프-정책 Q 추정값을 계산한다. Truncated Importance Sampling(IS 비율 상한 c̄=10)으로 분산을 제어하고, 편향 보정 항으로 절단으로 인한 편향을 보정한다.

**실험 방법**:
- Trading Days를 300 이상, ReplayBuffer가 충분히 채워진 뒤 성능 비교
- PPO 대비 동일 종목에서 안정성 비교 (PPO는 온-정책, ACER는 오프-정책)
- Retrace λ=0.95 vs λ=0 (TD(0)) 차이를 epsilon 값 변화로 간접 확인

**핵심 개념 확인**: IS 비율 ρ_t=π(a|s)/μ(a|s)가 c̄=10을 초과하면 절단 → 분산 감소; 편향 보정 항이 절단된 부분을 보상. PPO의 clip과 유사하지만 경험 재생을 병행한다는 점이 다르다.

---

### 7. SAC (Soft Actor-Critic) — 탐험 자동 조절

**학습 원리**: 보상에 엔트로피 항을 명시적으로 추가하고, 온도 α를 자동 학습한다. 탐험/착취 균형을 파라미터 없이 자동 조정한다.

**실험 방법**:
- epsilon 값을 달리 설정해도 결과가 비슷함을 확인 → α가 자동으로 탐험 조절
- 변동성 큰 환경(RGLD)에서 PPO 대비 비교
- 동일 파라미터로 여러 종목에 적용하여 범용성 확인

**핵심 개념 확인**: 엔트로피 최대화 목표 `max E[Q(s,a) − α·log π(a|s)]` — Q만 최대화하는 것보다 정책이 더 다양해지며, α가 자동으로 탐험/착취 균형을 찾음

---

### 8. DDPG — 연속 행동 공간

**학습 원리**: BUY/CASH 이진 결정이 아닌 **포지션 비율 [0, 1]** 을 직접 출력한다. 결정론적 정책으로 0.5 임계값 기준 BUY/CASH를 결정한다.

**실험 방법**:
- Trading Days를 충분히 길게(500 이상) 설정하여 ReplayBuffer가 채워진 뒤 안정적 학습 확인
- Train Episodes를 300 이상으로 설정 권장
- OU Noise의 역할 확인: 다른 알고리즘과 달리 epsilon 없이도 탐험이 이루어짐

**다른 알고리즘 대비 차이**: 연속 정책이라 BUY/CASH 경계를 더 섬세하게 학습하지만, 버퍼 충전 전 초반 성능이 낮을 수 있다.

**핵심 개념 확인**: 결정론적 정책에서는 탐험이 자동으로 없으므로 OU Noise가 필수. OU Noise 없이는 항상 동일한 행동을 반복하여 학습 불가.

---

### 권장 실험 순서

```
1단계:  STATIC vs Vanilla       기본 구조 이해, State Analysis 탭 활용
1H단계: STATIC_H vs STATIC      동일 파라미터로 PPO Clip + Adaptive-α 효과 비교
        8-State(Vol) 체크박스 활성화 → States: 8 확인 + Roll Period 조정
2단계:  A2C                     연속 특징 벡터 전환, STATIC과 성능 비교
3단계:  PPO                     안정적 학습 확인, lr 크게 설정해도 발산 없음 확인
4단계:  ACER                    오프-정책 재생 + Retrace, PPO와 동일 조건 비교
5단계:  SAC                     epsilon 무관한 자동 탐험 조절 확인
6단계:  A3C vs A2C              동일 종목에서 n-step vs 1-step 차이 비교
7단계:  DDPG                    연속 포지션 개념, Trading Days 충분히 확보 후 실행
```

### 알고리즘별 적합 환경 요약

| 알고리즘 | 데이터 요구량 | 변동성 환경 | 안정성 | 해석 가능성 | 특징 |
|---------|------------|-----------|-------|-----------|-----|
| STATIC | 낮음 | 보통 | 높음 | 높음 (State Analysis) | 기준 알고리즘 |
| STATIC_H | 낮음 | 보통~높음 | 매우 높음 | 높음 (State Analysis) | Tabular 하이브리드, 고착 방지 |
| Vanilla | 낮음 | 낮음 | 중간 | 높음 | 비교 기준선 |
| A2C | 중간 | 중간 | 중간 | 낮음 | 신경망 AC 기본 |
| A3C | 중간 | 높음 | 중간 | 낮음 | n-step 편향 개선 |
| PPO | 중간 | 중간 | 매우 높음 | 낮음 | 신경망 RL 입문 추천 |
| ACER | 높음 | 높음 | 높음 | 낮음 | 오프-정책 재생 |
| SAC | 중간 | 높음 | 높음 | 낮음 | 자동 온도 탐험 |
| DDPG | 높음 | 높음 | 중간 | 낮음 | 연속 포지션 |

> STATIC / STATIC_H은 해석 가능성이 높고 데이터 요구량이 낮다. STATIC_H는 행동 고착 문제가 있는 종목(KOSPI 등)에서 STATIC 대비 수렴 안정성이 개선된다. 신경망 알고리즘들은 더 복잡한 패턴을 학습하지만 파라미터와 데이터 양에 더 민감하다.
