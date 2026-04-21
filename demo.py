import time

import numpy as np
from exactdag import ExactDAG

rng = np.random.seed(42)

n, d = 50000, 10

B = np.tril(np.ones((d, d)), k=-1)

noise = np.random.uniform(-1, 1, size=(n, d))
X = noise @ np.linalg.inv(np.eye(d) - B).T

model = ExactDAG(penalty=1e-3)
model.fit(X)  # Removes compile overhead

t0 = time.perf_counter()
model.fit(X)
elapsed = time.perf_counter() - t0

B_hat = model.adjacency_matrix_
shd = int(np.sum((B != 0) != (B_hat != 0)))

print(f"fit time: {elapsed:.3f}s")
print(f"inertia: {model.inertia_:.4f}")
print(f"SHD: {shd}")
print("order:", model.causal_order_)
print("estimated B:\n", B_hat.round(2))
