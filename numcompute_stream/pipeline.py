from __future__ import annotations
from typing import Any, Protocol, runtime_checkable
import numpy as np

@runtime_checkable
class Transformer(Protocol):
    """Anything that can fit, transform and fit_transform can go in a Pipeline."""
    def fit(self, X: np.ndarray, /) -> Any: ...
    def transform(self, X: np.ndarray, /) -> np.ndarray: ...
    def fit_transform(self, X: np.ndarray, /) -> np.ndarray: ...

def _validate_transformer(name: str, obj: object) -> None:
    """Check that a step has all three required methods before we accept it."""
    for attr in ("fit", "transform", "fit_transform"):
        if not callable(getattr(obj, attr, None)):
            raise ValueError(f"Step '{name}' does not implement required method '{attr}'.")

class Pipeline:
    """
    Run a sequence of transformers one after another.
    All steps except the last call fit_transform during fit.
    The last step only calls fit so it can be a model rather
    than a transformer if needed.
    Supports both batch training via fit() and incremental
    updates via partial_fit() for streaming data.
    """
    def __init__(self, steps: list[tuple[str, Transformer]]) -> None:
        if not steps:
            raise ValueError("Pipeline requires at least one step.")
        names = [s[0] for s in steps]
        if len(set(names)) != len(names):
            raise ValueError("Pipeline step names must be unique.")
        for name, est in steps:
            _validate_transformer(name, est)
        self.steps = steps
        self._fitted = False
    def fit(self, X: np.ndarray) -> "Pipeline":
        """
        Fit all steps in order on the full dataset.
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data.
        Returns
        -------
        self
        Raises
        ------
        RuntimeError
            If any step is missing a required method.
        """
        Xt = np.asarray(X)
        for name, est in self.steps[:-1]:
            Xt = est.fit_transform(Xt)
        self.steps[-1][1].fit(Xt)
        self._fitted = True
        return self
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Pass X through every step in order.
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Data to transform.
        Returns
        -------
        np.ndarray
            Transformed data.
        Raises
        ------
        RuntimeError
            If called before fit or partial_fit.
        """
        if not self._fitted:
            raise RuntimeError("This Pipeline instance is not fitted yet; call fit first.")
        Xt = np.asarray(X)
        for _, est in self.steps:
            Xt = est.transform(Xt)
        return Xt
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)
    def partial_fit(self, X: np.ndarray, y: np.ndarray = None) -> "Pipeline":
        """
        Update each step incrementally with a new chunk of data.
        Works through the steps in order — each transformer gets
        partial_fit called on it then transforms the data before
        passing it along to the next step. The last step gets
        partial_fit with labels too if it is a model.
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            New chunk of features.
        y : np.ndarray or None
            Labels for the chunk. Only needed if the last step is a model.
        Returns
        -------
        self
        Raises
        ------
        RuntimeError
            If a step does not support partial_fit.
        """
        Xt = np.asarray(X, dtype=float)
        for name, est in self.steps[:-1]:
            if not hasattr(est, "partial_fit"):
                raise RuntimeError(f"Step '{name}' does not support partial_fit.")
            est.partial_fit(Xt)
            Xt = est.transform(Xt)
        # last step gets the labels if this is a model
        last_name, last_est = self.steps[-1]
        if not hasattr(last_est, "partial_fit"):
            raise RuntimeError(f"Step '{last_name}' does not support partial_fit.")
        if y is not None:
            last_est.partial_fit(Xt, y)
        else:
            last_est.partial_fit(Xt)
        self._fitted = True
        return self

class FeatureUnion:
    """
    Apply several transformers to the same input and join their outputs side by side.
    Useful when you want to combine different feature representations
    into one wider feature matrix before passing to a model.
    """
    def __init__(self, transformers: list[tuple[str, Transformer]]) -> None:
        if not transformers:
            raise ValueError("FeatureUnion requires at least one transformer.")
        names = [t[0] for t in transformers]
        if len(set(names)) != len(names):
            raise ValueError("FeatureUnion names must be unique.")
        for name, est in transformers:
            _validate_transformer(name, est)
        self.transformers = transformers
        self._fitted = False
    def fit(self, X: np.ndarray) -> "FeatureUnion":
        """
        Fit all transformers on the same X.
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data.
        Returns
        -------
        self
        """
        X = np.asarray(X)
        for _, est in self.transformers:
            est.fit(X)
        self._fitted = True
        return self
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform X with each transformer and concatenate the results.
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Data to transform.
        Returns
        -------
        np.ndarray
            Combined output from all transformers side by side.
        Raises
        ------
        RuntimeError
            If called before fit.
        """
        if not self._fitted:
            raise RuntimeError("This FeatureUnion instance is not fitted yet; call fit first.")
        X = np.asarray(X)
        parts = [est.transform(X) for _, est in self.transformers]
        return np.hstack(parts)
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        X = np.asarray(X)
        parts = [est.fit_transform(X) for _, est in self.transformers]
        self._fitted = True
        return np.hstack(parts)
    def partial_fit(self, X: np.ndarray) -> "FeatureUnion":
        """
        Update all transformers incrementally with a new chunk.
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            New chunk of data.
        Returns
        -------
        self
        Raises
        ------
        RuntimeError
            If any transformer does not support partial_fit.
        """
        X = np.asarray(X, dtype=float)
        for name, est in self.transformers:
            if not hasattr(est, "partial_fit"):
                raise RuntimeError(f"Transformer '{name}' does not support partial_fit.")
            est.partial_fit(X)
        self._fitted = True
        return self