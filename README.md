# ExactDAG

Exact score-based causal DAG learning via subset dynamic programming.

ExactDAG recovers a **Directed Acyclic Graph (DAG)** from observational data by solving the structure learning problem exactly. No heuristics, no greedy search. It uses a dynamic programming algorithm over subsets of variables to find the globally optimal causal order and parent sets under a penalized least-squares score.

## Algorithm

Given $n$ samples over $d$ variables, ExactDAG:

1. **Scores table** — computes the local score for every `(node, parent set)` pair using the empirical covariance matrix. The score for node $j$ with parent set $S$ is the OLS residual variance plus a regularization term `penalty * |S|`.
2. **Parent DP** — for each node and each candidate ancestor set, selects the best parent subset in $O(d \cdot 3^d)$ time.
3. **Sink DP** — finds the optimal causal ordering by choosing the best sink node for every subset of variables in $O(d^2 \cdot 2^d)$ time.

After the ordering is found, OLS regression recovers the edge weights.

The implementation uses [Numba](https://numba.pydata.org/) JIT compilation for speed, making it practical on real data up to ~20 variables.

## Installation

```bash
pip install git+https://github.com/felixlaplante0/exactdag.git
```

**Dependencies:** `numpy`, `numba`, `scikit-learn`

## Quickstart

```python
import numpy as np

from exactdag import ExactDAG

# Generate synthetic data from a linear SEM
d = 5
B = np.tril(np.ones((d, d)), k=-1)   # true adjacency matrix
noise = np.random.uniform(-1, 1, size=(10_000, d))
X = noise @ np.linalg.inv(np.eye(d) - B).T

# Fit
model = ExactDAG(penalty=1e-3)
model.fit(X)

print("Causal order:", model.causal_order_)
print("Estimated adjacency matrix:\n", model.adjacency_matrix_.round(2))
```

## License

See [LICENSE](LICENSE).
