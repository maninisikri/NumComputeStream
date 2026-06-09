import numpy as np
import pytest
from numcompute_stream.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, SimpleImputer

def test_standard_scaler_partial_fit_updates():
    scaler = StandardScaler()
    X1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    X2 = np.array([[5.0, 6.0], [7.0, 8.0]])
    scaler.partial_fit(X1)
    scaler.partial_fit(X2)
    assert scaler.mean_ is not None
    assert scaler.scale_ is not None

def test_minmax_scaler_partial_fit_expands_range():
    scaler = MinMaxScaler()
    X1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    X2 = np.array([[0.0, 10.0], [5.0, 1.0]])
    scaler.partial_fit(X1)
    scaler.partial_fit(X2)
    assert scaler.data_min_[0] == 0.0
    assert scaler.data_max_[1] == 10.0

def test_onehot_partial_fit_adds_categories():
    enc = OneHotEncoder()
    X1 = np.array([[0, 1], [1, 1]])
    X2 = np.array([[2, 3], [0, 2]])
    enc.partial_fit(X1)
    enc.partial_fit(X2)
    assert 2 in enc.categories_[0]
    assert 3 in enc.categories_[1]

def test_imputer_partial_fit_mean():
    imp = SimpleImputer(strategy="mean")
    X1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    imp.partial_fit(X1)
    assert imp.statistics_ is not None

def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        StandardScaler().transform(np.ones((3, 2)))