import numpy as np
import pytest
from numcompute_stream.tree import DecisionTreeClassifier

X = np.array([[1,2],[3,4],[5,6],[7,8],[1,3],[6,7]], dtype=float)
y = np.array([0,0,1,1,0,1])

def test_fit_predict_basic():
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert len(preds) == len(y)

def test_predict_before_fit_raises():
    clf = DecisionTreeClassifier()
    with pytest.raises(RuntimeError):
        clf.predict(X)

def test_partial_fit_returns_self():
    clf = DecisionTreeClassifier(max_depth=3)
    result = clf.partial_fit(X, y)
    assert result is clf

def test_perfect_fit_simple_data():
    clf = DecisionTreeClassifier(max_depth=5)
    clf.fit(X, y)
    assert np.array_equal(clf.predict(X), y)

def test_max_depth_one():
    clf = DecisionTreeClassifier(max_depth=1)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert len(preds) == len(y)

def test_single_class_data():
    X2 = np.array([[1,2],[3,4],[5,6]], dtype=float)
    y2 = np.array([0,0,0])
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X2, y2)
    assert np.all(clf.predict(X2) == 0)

def test_shape_mismatch_raises():
    clf = DecisionTreeClassifier()
    with pytest.raises(ValueError):
        clf.fit(X, np.array([0,1]))

def test_max_features_subset():
    clf = DecisionTreeClassifier(max_depth=3, max_features=1)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert len(preds) == len(y)
