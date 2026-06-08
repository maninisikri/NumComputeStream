import numpy as np
import pytest
from numcompute_stream.ensemble import RandomForestClassifier

X = np.array([[1,2],[3,4],[5,6],[7,8],[1,3],[6,7]], dtype=float)
y = np.array([0,0,1,1,0,1])

def test_fit_predict_basic():
    rf = RandomForestClassifier(n_estimators=3, random_state=42)
    rf.fit(X, y)
    preds = rf.predict(X)
    assert len(preds) == len(y)

def test_predict_before_fit_raises():
    rf = RandomForestClassifier()
    with pytest.raises(RuntimeError):
        rf.predict(X)

def test_partial_fit_updates_trees():
    rf = RandomForestClassifier(n_estimators=3, random_state=42)
    rf.fit(X, y)
    rf.partial_fit(X, y)
    assert len(rf.trees_) == 3

def test_score_perfect():
    rf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0)
    rf.fit(X, y)
    assert rf.score(X, y) >= 0.5

def test_n_estimators():
    rf = RandomForestClassifier(n_estimators=7, random_state=0)
    rf.fit(X, y)
    assert len(rf.trees_) == 7

def test_sqrt_max_features():
    rf = RandomForestClassifier(n_estimators=3, max_features="sqrt", random_state=0)
    rf.fit(X, y)
    preds = rf.predict(X)
    assert len(preds) == len(y)