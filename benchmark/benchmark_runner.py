import time
import numpy as np
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.ensemble import RandomForestClassifier

rng = np.random.default_rng(42)
X = rng.standard_normal((300, 4))
y = (X[:, 0] + X[:, 1] > 0).astype(int)
chunks = [(X[i:i+30], y[i:i+30]) for i in range(0, 300, 30)]

t0 = time.perf_counter()
tree = StreamTrainer(model=DecisionTreeClassifier(max_depth=4))
for X_chunk, y_chunk in chunks:
    tree.fit_chunk(X_chunk, y_chunk)
tree_time = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
forest = StreamTrainer(model=RandomForestClassifier(n_estimators=5, max_depth=4))
for X_chunk, y_chunk in chunks:
    forest.fit_chunk(X_chunk, y_chunk)
forest_time = (time.perf_counter() - t0) * 1000

print(f"Decision Tree  : {tree_time:.2f} ms")
print(f"Random Forest  : {forest_time:.2f} ms")
print(f"Forest is {forest_time/tree_time:.1f}x slower")