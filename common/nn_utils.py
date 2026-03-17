"""
common/nn_utils.py  — NumPy 전용 신경망 유틸리티
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TinyMLP    : 순수 NumPy 2층 완전 연결 신경망 (Adam 최적화기 내장)
ReplayBuffer: DDPG/SAC용 경험 재생 버퍼
extract_features: DataFrame → 5차원 연속 특징 벡터

특징 벡터 (5개):
  [0] ret        = Daily_Return[t]
  [1] ema_ratio  = Close[t] / EMA_10[t] - 1
  [2] vol        = Rolling_Std[t]
  [3] momentum   = sum(Daily_Return[t-5:t])
  [4] trend      = sign(Close[t] - Close[t-5])
"""

import numpy as np
import copy


# ══════════════════════════════════════════════════════════════════════════════
# TinyMLP — 순수 NumPy MLP (He 초기화 + Adam)
# ══════════════════════════════════════════════════════════════════════════════

class TinyMLP:
    """2층(Hidden + Output) 완전 연결 신경망.

    구조: input → [Hidden: ReLU] → [Output: 선형]
    출력에 softmax / sigmoid 적용은 호출자 책임.

    Adam 최적화기 내장, 역전파 시 가중치 자동 갱신.
    """

    def __init__(self, layer_sizes: list, seed: int = 42, lr: float = 0.001):
        """
        Parameters
        ----------
        layer_sizes : [n_in, n_hidden, n_out]  (정확히 3개)
        seed        : 가중치 초기화 난수 시드
        lr          : Adam 기본 학습률
        """
        np.random.seed(seed)
        self.layer_sizes = layer_sizes
        self.lr = lr

        # He 초기화
        self.layers = []  # [[W, b], ...]
        for i in range(len(layer_sizes) - 1):
            fan_in  = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            W = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / max(fan_in, 1))
            b = np.zeros(fan_out)
            self.layers.append([W, b])

        # Adam 모멘텀 상태
        self.beta1     = 0.9
        self.beta2     = 0.999
        self.eps_adam  = 1e-8
        self.t_adam    = 0
        self.m_W = [np.zeros_like(l[0]) for l in self.layers]
        self.v_W = [np.zeros_like(l[0]) for l in self.layers]
        self.m_b = [np.zeros_like(l[1]) for l in self.layers]
        self.v_b = [np.zeros_like(l[1]) for l in self.layers]

    # ── 순전파 ──────────────────────────────────────────────────────────────

    def forward(self, x: np.ndarray):
        """순전파.

        Parameters
        ----------
        x : 1D 입력 벡터 (n_in,)

        Returns
        -------
        output    : 최종 선형 출력 (n_out,)  — softmax/sigmoid는 호출자 적용
        pre_acts  : 각 층 선형 합 z_i 리스트  (역전파용 캐시)
        acts      : 활성화 출력 리스트 (acts[0]=입력, acts[1]=hidden, acts[2]=output)
        """
        x = np.asarray(x, dtype=float).ravel()
        pre_acts = []
        acts = [x]

        for i, (W, b) in enumerate(self.layers):
            z = acts[-1] @ W + b
            pre_acts.append(z)
            if i < len(self.layers) - 1:
                h = np.maximum(0.0, z)   # ReLU
            else:
                h = z                     # 선형 출력
            acts.append(h)

        return acts[-1].copy(), pre_acts, acts

    # ── 역전파 (내부) ────────────────────────────────────────────────────────

    def _backprop(self, pre_acts, acts, grad_out):
        """역전파로 각 층 가중치 기울기 계산.

        Returns
        -------
        grad_input : 입력에 대한 기울기 (n_in,)
        grads_W    : 각 층 W 기울기 리스트
        grads_b    : 각 층 b 기울기 리스트
        """
        delta    = np.asarray(grad_out, dtype=float).ravel()
        grads_W  = [None] * len(self.layers)
        grads_b  = [None] * len(self.layers)

        for i in reversed(range(len(self.layers))):
            a_in        = acts[i]
            grads_W[i]  = np.outer(a_in, delta)
            grads_b[i]  = delta.copy()
            if i > 0:
                # 이전 층으로 기울기 전달 (ReLU 미분 적용)
                delta = (delta @ self.layers[i][0].T) * (pre_acts[i - 1] > 0)

        # 입력에 대한 기울기: delta(grad w.r.t. pre_acts[0]) @ W_0.T → shape (n_in,)
        grad_input = delta @ self.layers[0][0].T
        return grad_input, grads_W, grads_b

    # ── Adam 가중치 갱신 ────────────────────────────────────────────────────

    def _adam_update(self, grads_W, grads_b, lr):
        self.t_adam += 1
        t = self.t_adam
        for i in range(len(self.layers)):
            # 1차·2차 모멘텀 갱신
            self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * grads_W[i]
            self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * grads_W[i] ** 2
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grads_b[i]
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * grads_b[i] ** 2
            # 편향 보정
            mhat_W = self.m_W[i] / (1 - self.beta1 ** t)
            vhat_W = self.v_W[i] / (1 - self.beta2 ** t)
            mhat_b = self.m_b[i] / (1 - self.beta1 ** t)
            vhat_b = self.v_b[i] / (1 - self.beta2 ** t)
            # 가중치 갱신
            self.layers[i][0] -= lr * mhat_W / (np.sqrt(vhat_W) + self.eps_adam)
            self.layers[i][1] -= lr * mhat_b / (np.sqrt(vhat_b) + self.eps_adam)

    # ── 공개 메서드 ──────────────────────────────────────────────────────────

    def backward_and_update(self, pre_acts, acts, grad_out, lr=None):
        """역전파 + Adam 가중치 갱신.

        Parameters
        ----------
        pre_acts  : forward() 반환 pre_acts 캐시
        acts      : forward() 반환 acts 캐시
        grad_out  : 손실 함수의 최종 출력에 대한 기울기 (n_out,)
        lr        : 학습률 (None이면 self.lr 사용)

        Returns
        -------
        grad_input : 입력에 대한 기울기 (DDPG 체인 규칙용)
        """
        if lr is None:
            lr = self.lr
        grad_input, grads_W, grads_b = self._backprop(pre_acts, acts, grad_out)
        self._adam_update(grads_W, grads_b, lr)
        return grad_input

    def get_grad_input(self, pre_acts, acts, grad_out):
        """가중치 갱신 없이 입력 기울기만 반환 (DDPG actor 체인 규칙용)."""
        grad_input, _, _ = self._backprop(pre_acts, acts, grad_out)
        return grad_input

    def copy(self):
        """깊은 복사 (타겟 네트워크 생성용)."""
        new = TinyMLP.__new__(TinyMLP)
        new.layer_sizes = self.layer_sizes[:]
        new.lr          = self.lr
        new.beta1       = self.beta1
        new.beta2       = self.beta2
        new.eps_adam    = self.eps_adam
        new.t_adam      = self.t_adam
        new.layers = [[W.copy(), b.copy()] for W, b in self.layers]
        new.m_W = [m.copy() for m in self.m_W]
        new.v_W = [v.copy() for v in self.v_W]
        new.m_b = [m.copy() for m in self.m_b]
        new.v_b = [v.copy() for v in self.v_b]
        return new

    def soft_update_from(self, source: "TinyMLP", tau: float):
        """타겟 네트워크 소프트 갱신: θ_self ← τ·θ_src + (1-τ)·θ_self"""
        for i in range(len(self.layers)):
            self.layers[i][0] = tau * source.layers[i][0] + (1 - tau) * self.layers[i][0]
            self.layers[i][1] = tau * source.layers[i][1] + (1 - tau) * self.layers[i][1]


# ══════════════════════════════════════════════════════════════════════════════
# ReplayBuffer — 경험 재생 버퍼 (DDPG / SAC용)
# ══════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    """고정 용량 원형 버퍼 (Circular Buffer).

    transition: (state, action, reward, next_state, done)
    """

    def __init__(self, capacity: int, n_features: int):
        self.capacity   = capacity
        self.n_features = n_features
        self._states      = np.zeros((capacity, n_features), dtype=np.float32)
        self._actions     = np.zeros(capacity, dtype=np.float32)
        self._rewards     = np.zeros(capacity, dtype=np.float32)
        self._next_states = np.zeros((capacity, n_features), dtype=np.float32)
        self._dones       = np.zeros(capacity, dtype=np.float32)
        self._ptr  = 0
        self._size = 0

    def push(self, s, a, r, ns, done):
        self._states[self._ptr]      = s
        self._actions[self._ptr]     = float(a)
        self._rewards[self._ptr]     = float(r)
        self._next_states[self._ptr] = ns
        self._dones[self._ptr]       = float(done)
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.choice(self._size, size=min(batch_size, self._size), replace=False)
        return (
            self._states[idx],
            self._actions[idx],
            self._rewards[idx],
            self._next_states[idx],
            self._dones[idx],
        )

    def __len__(self):
        return self._size


# ══════════════════════════════════════════════════════════════════════════════
# 특징 추출기 — 5차원 연속 벡터
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(df_vals: dict, t: int, lookback: int = 5) -> np.ndarray:
    """시장 데이터 → 5차원 연속 특징 벡터.

    Parameters
    ----------
    df_vals  : {'returns', 'prices', 'emas', 'vols'} — 각 항목은 np.ndarray
    t        : 현재 시간 인덱스
    lookback : 모멘텀·추세 계산 기간 (기본 5)

    Returns
    -------
    np.ndarray shape (5,):
      [0] ret       = Daily_Return[t]
      [1] ema_ratio = Close[t] / EMA_10[t] - 1
      [2] vol       = Rolling_Std[t]
      [3] momentum  = sum(Daily_Return[t-lookback:t])
      [4] trend     = sign(Close[t] - Close[t-lookback])
    """
    returns = df_vals['returns']
    prices  = df_vals['prices']
    emas    = df_vals['emas']
    vols    = df_vals['vols']

    ret       = float(returns[t])
    ema_ratio = float(prices[t]) / max(float(emas[t]), 1e-10) - 1.0
    vol       = float(vols[t])

    t_start  = max(0, t - lookback)
    momentum = float(np.sum(returns[t_start:t])) if t > 0 else 0.0

    if t >= lookback:
        trend = float(np.sign(prices[t] - prices[t - lookback]))
    else:
        trend = 0.0

    return np.array([ret, ema_ratio, vol, momentum, trend], dtype=float)
