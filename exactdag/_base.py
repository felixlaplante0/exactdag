"""Exact score-based DAG learning via subset dynamic programming."""

from __future__ import annotations

from typing import Self, cast

import numpy as np
from numba import njit, prange  # type: ignore
from sklearn.base import BaseEstimator  # type: ignore
from sklearn.utils.validation import validate_data  # type: ignore


@njit(cache=True, inline="always")  # type: ignore
def _get_A_b_num(
    cov_matrix: np.ndarray, pmask: int, target: int, d: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Builds the packed lower-triangular system matrix A and right-hand side b.

    Args:
        cov_matrix (np.ndarray): Covariance matrix.
        pmask (int): Bitmask encoding the predictor variables.
        target (int): Index of the response variable.
        d (int): Number of variables.

    Returns:
        tuple[np.ndarray, np.ndarray, int]: Packed lower-triangular matrix, right-hand
            side vector, and number of parents.
    """
    m = d - 1
    parents = np.empty(m, dtype=np.int32)
    A = np.empty(m * (m + 1) // 2, dtype=np.float64)
    b = np.empty(m, dtype=np.float64)
    i = k = 0

    while pmask:
        if pmask & 1:
            pos = i if i < target else i + 1
            b[k] = cov_matrix[target, pos]
            kk = k * (k + 1) // 2
            for j in range(k):
                A[kk + j] = cov_matrix[pos, parents[j]]
            A[kk + k] = cov_matrix[pos, pos]
            parents[k] = pos
            k += 1
        pmask >>= 1
        i += 1

    return A, b, k


@njit(cache=True, inline="always")
def _cholesky_solve_norm_inplace(A: np.ndarray, b: np.ndarray, n: int):
    """Solves via in-place Cholesky decomposition and returns the squared norm.

    Args:
        A (np.ndarray): Packed lower-triangular matrix, modified in-place.
        b (np.ndarray): Right-hand side vector, overwritten with the solution.
        n (int): Size of the system.

    Returns:
        float: Squared norm of the solution vector.
    """
    sq_norm = 0.0

    for i in range(n):
        ii = i * (i + 1) // 2
        for j in range(i + 1):
            ij = ii + j
            for k in range(j):
                A[ij] -= A[ii + k] * A[j * (j + 1) // 2 + k]
            if i == j:
                A[ij] = np.sqrt(A[ij])
            else:
                A[ij] /= A[j * (j + 1) // 2 + j]
                b[i] -= A[ij] * b[j]
        b[i] /= A[ii + i]
        sq_norm += b[i] ** 2

    return sq_norm


@njit(cache=True, inline="always")  # type: ignore
def _score(
    cov_matrix: np.ndarray, target: int, pmask: int, d: int, penalty: float
) -> float:
    """Solve least squares for target regressed on the variables in pmask.

    Args:
        cov_matrix (np.ndarray): cov_matrixariance matrix.
        target (int): Index of the response variable.
        pmask (int): Bitmask encoding the predictor variables.
        d (int): Number of variables.
        penalty (float): Regularization penalty per parent.

    Returns:
        float: The score for the given target, pmask, and penalty.
    """
    A, b, k = _get_A_b_num(cov_matrix, pmask, target, d)
    sq_norm = _cholesky_solve_norm_inplace(A, b, k)
    return cov_matrix[target, target] - sq_norm + penalty * k


@njit(cache=True, parallel=True)  # type: ignore
def _parents_dp(
    cov_matrix: np.ndarray, d: int, penalty: float
) -> tuple[np.ndarray, np.ndarray]:
    """Find the best subset as parents set for each node and candidate set.

    Args:
        cov_matrix (np.ndarray): cov_matrixariance matrix.
        target (int): Index of the response variable.
        d (int): Number of variables.
        penalty (float): Regularization penalty per parent.

    Returns:
        tuple[np.ndarray, np.ndarray]: Best scores and best parent sets.
    """
    n = 1 << (d - 1)
    best_scores = np.empty((d, n), dtype=np.float32)
    best_parents_sets = np.empty((d, n), dtype=np.int32)

    for j in prange(d):
        best_scores[j, 0] = cov_matrix[j, j]
        best_parents_sets[j, 0] = 0
        for pmask in range(1, n):
            cur_best_score = _score(cov_matrix, j, pmask, d, penalty)
            cur_best_set = bits = pmask
            while bits:
                lsb = bits & -bits
                sub_pmask = pmask ^ lsb
                if best_scores[j, sub_pmask] < cur_best_score:
                    cur_best_score = best_scores[j, sub_pmask]
                    cur_best_set = best_parents_sets[j, sub_pmask]
                bits ^= lsb
            best_scores[j, pmask] = cur_best_score
            best_parents_sets[j, pmask] = cur_best_set

    return best_scores, best_parents_sets


@njit(cache=True, inline="always")
def pack_mask(mask: int, j: int) -> int:
    """Removes bit j from mask.

    Args:
        mask (int): Bitmask encoding a subset of variables.
        j (int): Index of the target bit.
    """
    lower = mask & ((1 << j) - 1)
    upper = mask >> (j + 1)
    return lower | (upper << j)


@njit(cache=True)  # type: ignore
def _sink_dp(best_scores: np.ndarray, d: int) -> tuple[np.ndarray, float]:
    """Find the optimal sink node for every subset via dynamic programming.

    Args:
        best_score (np.ndarray): Best scores.
        d (int): Number of variables.

    Returns:
        tuple(np.ndarray, float): Optimal sink index for each subset and inertia.
    """
    n = 1 << d
    H = np.zeros(n, dtype=np.float32)
    sinks = np.full(n, -1, dtype=np.int32)

    for mask in range(1, n):
        cur_best_score = np.inf
        cur_best_sink = -1
        bits = mask
        s = 0
        while bits:
            if bits & 1:
                prev_mask = mask ^ (1 << s)
                prev_pmask = pack_mask(prev_mask, s)
                score = H[prev_mask] + best_scores[s, prev_pmask]
                if score < cur_best_score:
                    cur_best_score = score
                    cur_best_sink = s
            bits >>= 1
            s += 1
        H[mask] = cur_best_score
        sinks[mask] = cur_best_sink

    return sinks, H[n - 1]


def _causal_order(sinks: np.ndarray, d: int) -> np.ndarray:
    """Recovers the causal order from source to sink.

    Args:
        sinks (np.ndarray): Optimal sink index for each subset.
        d (int): Number of variables.

    Returns:
        np.ndarray: Causal order from source to sink.
    """
    order = np.empty(d, dtype=np.int32)
    mask = (1 << d) - 1

    for i in range(d):
        s = sinks[mask]
        order[i] = s
        mask ^= 1 << s

    return order[::-1]


def _ols_weights(
    order: np.ndarray, best_parents_sets: np.ndarray, cov_matrix: np.ndarray, d: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recovers regression edge weights given the causal order and parent sets.

    Args:
        order (np.ndarray): Causal order.
        best_parents_sets (np.ndarray): Optimal parents set table.
        cov_matrix (np.ndarray): cov_matrixariance matrix.
        d (int): Number of variables.

    Returns:
        np.ndarray: Weight matrix.
    """
    W = np.zeros((d, d), dtype=np.float64)
    mask = 0

    for t in range(d):
        j = order[t]
        parents_mask = best_parents_sets[j, mask]
        if parents_mask:
            parents = np.where([(parents_mask >> i) & 1 for i in range(d)])[0]
            W[j, parents] = np.linalg.solve(
                cov_matrix[np.ix_(parents, parents)], cov_matrix[parents, j]
            )
        mask |= 1 << j

    return W


class ExactDAG(BaseEstimator):
    """Exact dynamic programming-based causal discovery.

    Attributes:
        fit_intercept (bool): Whether to center the data.
        penalty (float): Regularization penalty per parent.
        causal_order_ (np.ndarray): Causal order from source to sink.
        adjacency_matrix_ (np.ndarray): Causal weight matrix.
        intercept_ (np.ndarray): Intercept of the regression models.
        inertia_ (float): Residual sum of squares of the learned DAG.
    """

    fit_intercept: bool
    penalty: float
    causal_order_: np.ndarray
    adjacency_matrix_: np.ndarray
    intercept_: np.ndarray
    inertia_: np.ndarray

    def __init__(self, fit_intercept: bool = True, penalty: float = 0) -> None:
        """Initialize ExactDAG.

        Args:
            fit_intercept (bool, optional): Whether to center the data. Defaults to
                True.
            penalty (float, optional): Regularization penalty per parent. Defaults to 0.
        """
        self.fit_intercept = fit_intercept
        self.penalty = penalty

    def fit(self, X: np.typing.ArrayLike, y: None = None) -> Self:  # noqa: ARG002
        """Fits the ExactDAG algorithm.

        Args:
            X (np.typing.ArrayLike): Input data of shape (n_samples, d).

        Returns:
            ExactDAG: The fitted estimator.
        """
        X = cast(np.ndarray, validate_data(self, X, dtype=np.float64))  # type: ignore
        d = X.shape[1]

        if self.fit_intercept:
            shift = X.mean(axis=0)
            X = X - shift  # type: ignore

        cov_matrix = cast(np.ndarray, X.T @ X)  # type: ignore

        best_scores, best_parents_sets = _parents_dp(cov_matrix, d, self.penalty)
        sinks, self.inertia_ = _sink_dp(best_scores, d)

        self.causal_order_ = _causal_order(sinks, d)
        self.adjacency_matrix_ = _ols_weights(
            self.causal_order_, best_parents_sets, cov_matrix, d
        )

        if self.fit_intercept:
            self.intercept_ = shift - self.adjacency_matrix_ @ shift

        return self
