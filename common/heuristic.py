"""
Bayesian Optimization Engine — numpy 전용 GP + UCB Acquisition
================================================================
의존성: numpy 만 사용 (scipy 불필요)
Python 3.9+ 호환: from __future__ import annotations 로 타입 힌트 지연 평가

알고리즘 개요
-----------
1. 초기 탐색(n_random_start 회): 균등 랜덤 샘플링
2. 이후: RBF 커널 GP를 관측값에 피팅 →
   UCB(Upper Confidence Bound) 획득 함수로 다음 후보 제안
3. 모든 파라미터는 내부적으로 [0,1] 정규화 후 연산

수식
----
  K(x,x') = σ²·exp(-||x-x'||² / (2·l²))    [RBF 커널]
  GP posterior: μ* = k*ᵀ(K+σₙ²I)⁻¹y
                σ*² = k** - k*ᵀ(K+σₙ²I)⁻¹k*
  UCB: α(x) = μ(x) + κ·σ(x)               [κ = 2.0]
"""

from __future__ import annotations  # Python 3.9 호환: dict|None 등 타입 힌트 지연 평가

import numpy as np


class BayesianOptimizer:
    """
    파라미터 탐색을 위한 Bayesian Optimization 엔진.

    Parameters
    ----------
    bounds : dict
        {param_name: (lo, hi)} 형식의 파라미터 범위.
    n_random_start : int
        GP 피팅 전 순수 랜덤 탐색 횟수 (기본 8).
    kappa : float
        UCB 탐험-활용 균형 계수 (기본 2.0).
        클수록 탐험 강화, 작을수록 활용 강화.
    length_scale : float
        GP 커널의 길이 스케일 (정규화 공간 기준, 기본 0.3).
    noise : float
        GP 관측 노이즈 σₙ² (수치 안정성, 기본 1e-4).
    n_restarts : int
        UCB 최대화 시 랜덤 시작점 수 (기본 20).
    """

    def __init__(
        self,
        bounds: dict,
        n_random_start: int = 8,
        kappa: float = 2.0,
        length_scale: float = 0.3,
        noise: float = 1e-4,
        n_restarts: int = 20,
    ):
        self.bounds = bounds                          # {name: (lo, hi)}
        self.param_names = list(bounds.keys())
        self.n_random_start = n_random_start
        self.kappa = kappa
        self.length_scale = length_scale
        self.noise = noise
        self.n_restarts = n_restarts

        # 관측 저장 (정규화 공간)
        self._X: list[np.ndarray] = []               # 각 원소: shape (n_params,)
        self._y: list[float] = []                    # 관측된 스코어 (gap %)

        self._best_params: dict | None = None
        self._best_score: float = -np.inf

    # ------------------------------------------------------------------
    # 공개 API
    # ------------------------------------------------------------------

    def suggest_next(self) -> dict:
        """다음 탐색 후보를 제안 (원본 파라미터 공간으로 반환)."""
        n_obs = len(self._X)

        if n_obs < self.n_random_start:
            # ─ 초기 랜덤 탐색: Latin-Hypercube 스타일 ─
            x_norm = self._latin_hypercube_sample(n_obs, len(self.param_names))
        else:
            # ─ GP UCB 안내 탐색 ─
            x_norm = self._ucb_next()

        return self._denormalize(x_norm)

    def update(self, params: dict, score: float) -> None:
        """관측 결과를 옵티마이저에 추가."""
        x_norm = self._normalize(params)
        self._X.append(x_norm)
        self._y.append(float(score))

        if score > self._best_score:
            self._best_score = score
            self._best_params = {k: float(v) for k, v in params.items()}

    @property
    def best_params(self) -> dict | None:
        """현재까지 가장 높은 스코어의 파라미터 반환."""
        return self._best_params

    @property
    def best_score(self) -> float:
        return self._best_score

    @property
    def n_observations(self) -> int:
        return len(self._X)

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _normalize(self, params: dict) -> np.ndarray:
        """파라미터 dict → [0,1] 정규화 벡터."""
        lo_hi = [self.bounds[k] for k in self.param_names]
        vals  = [params[k] for k in self.param_names]
        return np.array([
            (v - lo) / (hi - lo) if hi != lo else 0.5
            for v, (lo, hi) in zip(vals, lo_hi)
        ], dtype=float)

    def _denormalize(self, x_norm: np.ndarray) -> dict:
        """[0,1] 벡터 → 원본 파라미터 공간 dict."""
        result = {}
        for i, k in enumerate(self.param_names):
            lo, hi = self.bounds[k]
            result[k] = float(np.clip(lo + x_norm[i] * (hi - lo), lo, hi))
        return result

    def _latin_hypercube_sample(self, idx: int, n_dims: int) -> np.ndarray:
        """
        n_random_start 구간을 n_dims 차원으로 균등 분할하는
        Latin Hypercube 방식 단일 샘플.
        idx: 현재 관측 수 (0 ~ n_random_start-1).
        """
        rng = np.random.default_rng(seed=idx * 31 + 7)  # 재현성 있는 랜덤
        segment = 1.0 / self.n_random_start
        lower   = idx * segment
        return rng.uniform(lower, lower + segment, size=n_dims)

    # ------------------------------------------------------------------
    # GP 커널 및 예측
    # ------------------------------------------------------------------

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        RBF (squared exponential) 커널.
        X1: (n, d), X2: (m, d) → (n, m)
        """
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        diffs   = X1[:, None, :] - X2[None, :, :]     # (n, m, d)
        sq_dist = np.sum(diffs ** 2, axis=-1)          # (n, m)
        return np.exp(-sq_dist / (2.0 * self.length_scale ** 2))

    def _gp_predict(self, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        현재 관측값으로 피팅된 GP의 사후 예측.
        Returns: (mu, sigma) — 각각 shape (n_test,)
        """
        X_obs = np.array(self._X)   # (n_obs, d)
        y_obs = np.array(self._y)   # (n_obs,)
        n_obs = len(X_obs)

        K_obs = self._rbf_kernel(X_obs, X_obs) + self.noise * np.eye(n_obs)
        K_test_obs = self._rbf_kernel(X_test, X_obs)          # (n_test, n_obs)
        K_test_test_diag = np.ones(len(X_test))               # 커널 대각 (RBF(x,x)=1)

        try:
            L = np.linalg.cholesky(K_obs)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_obs))
            mu = K_test_obs @ alpha

            v = np.linalg.solve(L, K_test_obs.T)              # (n_obs, n_test)
            sigma2 = K_test_test_diag - np.sum(v ** 2, axis=0)
            sigma  = np.sqrt(np.maximum(sigma2, 1e-9))
        except np.linalg.LinAlgError:
            # Cholesky 실패 시 최소한의 예측 반환
            mu    = np.full(len(X_test), float(np.mean(y_obs)))
            sigma = np.ones(len(X_test)) * 0.1

        return mu, sigma

    # ------------------------------------------------------------------
    # UCB 최대화
    # ------------------------------------------------------------------

    def _ucb_acquisition(self, X_test: np.ndarray) -> np.ndarray:
        """UCB 획득 함수값 계산. X_test: (n_test, d)"""
        mu, sigma = self._gp_predict(X_test)
        return mu + self.kappa * sigma

    def _ucb_next(self) -> np.ndarray:
        """
        랜덤 다중 시작점으로 UCB 최대화 → 최적 정규화 후보 반환.
        """
        n_dims = len(self.param_names)
        rng    = np.random.default_rng()

        # n_restarts 개의 후보를 동시에 평가
        X_candidates = rng.uniform(0.0, 1.0, size=(self.n_restarts * 50, n_dims))
        acq_values   = self._ucb_acquisition(X_candidates)
        best_idx     = int(np.argmax(acq_values))
        return X_candidates[best_idx]
