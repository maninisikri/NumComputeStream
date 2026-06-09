# NumCompute Stream

A streaming, decision tree-based machine learning framework built with plain Python and NumPy only.

Extends the NumCompute toolkit with incremental learning support, ensemble methods, and real-time visualisation. All components support chunk-wise updates through `.partial_fit()` methods, simulating a real-world online learning scenario.

> **Course:** COMP-5004 — Programming for AI
> **University:** University of Adelaide
> **Author:** Manini Sikri

---

## Project Structure

- numcompute_stream/
  - io.py — CSV loading with chunking and missing value handling
  - preprocessing.py — StandardScaler, MinMaxScaler, OneHotEncoder, SimpleImputer with partial_fit
  - stats.py — Descriptive stats and StreamingStats with update_stats
  - metrics.py — Classification metrics with streaming support
  - tree.py — DecisionTreeClassifier with Gini impurity and partial_fit
  - ensemble.py — RandomForestClassifier with streaming support
  - stream.py — StreamTrainer for chunk-based incremental learning
  - pipeline.py — Pipeline and FeatureUnion with partial_fit
  - visualise.py — Matplotlib plots for streaming metrics
- tests/ — 30 unit tests
- demo/stream_demo.ipynb — End-to-end streaming demo
- benchmark/
- README.md

---

## Installation

```bash
git clone https://github.com/maninisikri/NumComputeStream.git
cd NumComputeStream
pip3 install numpy matplotlib pytest
```

---

## Quick Start

```python
import numpy as np
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.ensemble import RandomForestClassifier
from numcompute_stream.preprocessing import StandardScaler
from numcompute_stream.stream import StreamTrainer

trainer = StreamTrainer(
    model=RandomForestClassifier(n_estimators=5),
    preprocessor=StandardScaler()
)

for X_chunk, y_chunk in chunks:
    trainer.fit_chunk(X_chunk, y_chunk)

trainer.summary()
```

---

## Module Overview

### `tree.py` — Decision Tree
| Method | Description |
|--------|-------------|
| `fit(X, y)` | Train on full dataset |
| `partial_fit(X, y)` | Update with new chunk |
| `predict(X)` | Predict class labels |

### `ensemble.py` — Random Forest
| Method | Description |
|--------|-------------|
| `fit(X, y)` | Train all trees on bootstrap samples |
| `partial_fit(X, y)` | Update each tree with new chunk |
| `predict(X)` | Majority vote across all trees |
| `score(X, y)` | Accuracy on given data |

### `stream.py` — Stream Trainer
| Method | Description |
|--------|-------------|
| `fit_chunk(X, y)` | Train and log metrics for one chunk |
| `score_chunk(X, y)` | Evaluate on a chunk |
| `reset()` | Clear all logs |
| `summary()` | Print current metrics |

### `visualise.py` — Plots
| Function | Description |
|----------|-------------|
| `plot_metric_over_time(values, title, ylabel)` | Plot metric across chunks |
| `compare_models(metric1, metric2, labels)` | Compare two models |
| `plot_predictions_vs_ground_truth(y_true, y_pred)` | Predictions vs actuals |

### `preprocessing.py` — Scalers
All classes support `fit()`, `transform()`, `fit_transform()` and `partial_fit()`.

| Class | Description |
|-------|-------------|
| `StandardScaler` | Z-score standardisation with Welford streaming |
| `MinMaxScaler` | Scale to [0,1] with running min/max |
| `OneHotEncoder` | Incremental category expansion |
| `SimpleImputer` | Running mean/median for NaN filling |

### `metrics.py` — Streaming Metrics
| Class | Description |
|-------|-------------|
| `StreamingAccuracy` | Running accuracy with optional window |
| `StreamingConfusionMatrix` | Accumulated confusion matrix |
| `StreamingMetrics` | Accuracy, precision, recall, F1 together |

### `stats.py` — Streaming Stats
| Class | Description |
|-------|-------------|
| `StreamingStats` | Running mean, variance, quantiles, histogram |

---

## Running Tests

```bash
pytest tests/ -v
```

30 unit tests covering all modules including edge cases like empty chunks, NaN values, unfitted models, and streaming updates.

---

## Benchmark Results

Running on Apple M-series, Python 3.14.3, NumPy 2.4.3:

| Model | Time (ms) | Accuracy |
|-------|-----------|----------|
| Decision Tree | 51.24 | 100% |
| Random Forest (5 trees) | 110.40 | 97.67% |

Forest is 2.2x slower but more robust on unseen data.

---

## Design Principles

- **Streaming first** — every component supports `.partial_fit()` for chunk-wise updates
- **No external ML libraries** — only NumPy and matplotlib
- **Numerical stability** — NaN handling, zero variance protection, Welford algorithm
- **Consistent API** — all models follow fit/partial_fit/predict pattern