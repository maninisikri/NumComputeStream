import numpy as np
import pytest
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.preprocessing import StandardScaler

X = np.array([[1,2],[3,4],[5,6],[7,8],[1,3],[6,7]], dtype=float)
y = np.array([0,0,1,1,0,1])

def test_fit_chunk_logs_accuracy():
    clf = DecisionTreeClassifier(max_depth=3)
    trainer = StreamTrainer(model=clf)
    trainer.fit_chunk(X[:3], y[:3])
    assert len(trainer.accuracy_log_) == 1

def test_chunk_count_increments():
    clf = DecisionTreeClassifier(max_depth=3)
    trainer = StreamTrainer(model=clf)
    trainer.fit_chunk(X[:3], y[:3])
    trainer.fit_chunk(X[3:], y[3:])
    assert trainer.chunk_count_ == 2

def test_reset_clears_logs():
    clf = DecisionTreeClassifier(max_depth=3)
    trainer = StreamTrainer(model=clf)
    trainer.fit_chunk(X, y)
    trainer.reset()
    assert trainer.accuracy_log_ == []
    assert trainer.chunk_count_ == 0

def test_with_preprocessor():
    clf = DecisionTreeClassifier(max_depth=3)
    scaler = StandardScaler()
    trainer = StreamTrainer(model=clf, preprocessor=scaler)
    trainer.fit_chunk(X, y)
    assert len(trainer.accuracy_log_) == 1

def test_summary_runs_without_error():
    clf = DecisionTreeClassifier(max_depth=3)
    trainer = StreamTrainer(model=clf)
    trainer.fit_chunk(X, y)
    trainer.summary()

def test_score_chunk_returns_dict():
    clf = DecisionTreeClassifier(max_depth=3)
    trainer = StreamTrainer(model=clf)
    trainer.fit_chunk(X, y)
    result = trainer.score_chunk(X, y)
    assert "accuracy" in result
    assert "f1" in result