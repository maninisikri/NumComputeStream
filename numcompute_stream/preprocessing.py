from __future__ import annotations
import warnings
import numpy as np


class StandardScaler:
    """
    Standardise features by removing the mean and scaling to unit variance.

    Supports both batch fitting via fit() and incremental updates via
    partial_fit() for streaming data scenarios.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> StandardScaler:
        """
        Compute mean and standard deviation from the full dataset.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If X is not 2-D.

        Time complexity: O(n_samples * n_features)
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D with shape (n_samples, n_features).")
        self.mean_ = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        if np.any(std == 0):
            warnings.warn(
                "Zero standard deviation detected; scale set to 1 for those features.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.scale_ = np.where(std == 0, 1.0, std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Standardise X using the mean and scale learned during fit.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        np.ndarray
            Standardised array, same shape as X.

        Raises
        ------
        RuntimeError
            If called before fit or partial_fit.

        Time complexity: O(n_samples * n_features)
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler must be fitted before transform.")
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def partial_fit(self, X: np.ndarray) -> "StandardScaler":
        """
        Update the scaler with a new chunk of data.

        Uses Welford's online algorithm to update the mean and variance
        one chunk at a time so we never have to hold the full dataset
        in memory.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            New chunk of data to learn from.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If X is not 2-D.

        Time complexity: O(n_samples * n_features) per chunk
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D.")

        if self.mean_ is None:
            # first chunk — initialise accumulators
            self.mean_ = np.zeros(X.shape[1])
            self._var = np.zeros(X.shape[1])
            self._count = 0

        # update running mean and variance for each new row
        for x in X:
            self._count += 1
            delta = x - self.mean_
            self.mean_ += delta / self._count
            delta2 = x - self.mean_
            self._var += delta * delta2

        std = np.sqrt(self._var / max(self._count - 1, 1))
        self.scale_ = np.where(std == 0, 1.0, std)
        return self


class MinMaxScaler:
    """
    Scale features to a fixed range, default [0, 1].

    Supports both batch fitting and incremental updates for streaming.
    The range expands automatically as new chunks arrive with unseen values.
    """

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)) -> None:
        lo, hi = feature_range
        if hi <= lo:
            raise ValueError("feature_range must satisfy min < max.")
        self.feature_range = (float(lo), float(hi))
        self.data_min_: np.ndarray | None = None
        self.data_max_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        """
        Compute per-feature min and max from the full dataset.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If X is not 2-D.

        Time complexity: O(n_samples * n_features)
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D with shape (n_samples, n_features).")
        self.data_min_ = np.nanmin(X, axis=0)
        self.data_max_ = np.nanmax(X, axis=0)
        span = self.data_max_ - self.data_min_
        self.scale_ = np.where(span == 0.0, 1.0, span)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale X to the target feature range.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        np.ndarray
            Scaled array, same shape as X.

        Raises
        ------
        RuntimeError
            If called before fit or partial_fit.

        Time complexity: O(n_samples * n_features)
        """
        if self.data_min_ is None or self.data_max_ is None or self.scale_ is None:
            raise RuntimeError("MinMaxScaler must be fitted before transform.")
        X = np.asarray(X, dtype=float)
        lo, hi = self.feature_range
        X_std = (X - self.data_min_) / self.scale_
        X_scaled = X_std * (hi - lo) + lo
        constant = self.data_max_ == self.data_min_
        if np.any(constant):
            X_scaled[:, constant] = 0.0
        return X_scaled

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def partial_fit(self, X: np.ndarray) -> "MinMaxScaler":
        """
        Update the scaler with a new chunk of data.

        Each time a new chunk arrives, we check if it contains values
        outside the range seen so far and expand the min/max accordingly.
        This keeps the scaler accurate without needing to revisit old data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            New chunk of data to learn from.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If X is not 2-D.

        Time complexity: O(n_samples * n_features) per chunk
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D.")

        chunk_min = np.nanmin(X, axis=0)
        chunk_max = np.nanmax(X, axis=0)

        if self.data_min_ is None:
            # first chunk — take the min and max directly
            self.data_min_ = chunk_min
            self.data_max_ = chunk_max
        else:
            # expand the range if this chunk goes beyond what we have seen
            self.data_min_ = np.minimum(self.data_min_, chunk_min)
            self.data_max_ = np.maximum(self.data_max_, chunk_max)

        span = self.data_max_ - self.data_min_
        self.scale_ = np.where(span == 0.0, 1.0, span)
        return self


class OneHotEncoder:
    """
    Encode categorical columns as binary indicator columns.

    Supports incremental updates so new categories discovered in later
    chunks are automatically added to the encoding.
    """

    def __init__(self) -> None:
        self.categories_: list[np.ndarray] | None = None

    def fit(self, X: np.ndarray) -> OneHotEncoder:
        """
        Learn the unique categories for each feature column.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data with categorical columns.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If X is not 2-D.

        Time complexity: O(n_samples * n_features)
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2-D with shape (n_samples, n_features).")
        self.categories_ = [np.unique(X[:, j]) for j in range(X.shape[1])]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Expand each categorical column into binary indicator columns.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Data to encode.

        Returns
        -------
        np.ndarray, shape (n_samples, total_unique_categories)
            Binary encoded array.

        Raises
        ------
        RuntimeError
            If called before fit or partial_fit.

        Time complexity: O(n_samples * total_categories)
        """
        if self.categories_ is None:
            raise RuntimeError("OneHotEncoder must be fitted before transform.")
        X = np.asarray(X)
        blocks: list[np.ndarray] = []
        for j, cats in enumerate(self.categories_):
            col = X[:, j]
            blocks.append((col[:, None] == cats[None, :]).astype(float))
        return np.hstack(blocks) if blocks else np.empty((X.shape[0], 0))

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def partial_fit(self, X: np.ndarray) -> "OneHotEncoder":
        """
        Update the encoder with new categories seen in this chunk.

        If a category appears that we haven't seen before, it gets added
        to the known list for that feature. The output width grows over
        time as the encoder discovers new values in the stream.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            New chunk of data. Each column is a categorical feature.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If X is not 2-D.

        Time complexity: O(n_samples * n_features) per chunk
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2-D.")

        if self.categories_ is None:
            # first chunk — discover initial categories per feature
            self.categories_ = [np.unique(X[:, j]) for j in range(X.shape[1])]
        else:
            # expand categories if this chunk has values we haven't seen
            for j in range(X.shape[1]):
                new_cats = np.unique(X[:, j])
                self.categories_[j] = np.unique(
                    np.concatenate([self.categories_[j], new_cats])
                )
        return self


class SimpleImputer:
    """
    Replace missing values with a per-feature statistic or constant.

    Supports incremental updates so the fill values stay accurate
    as new chunks of data arrive in a streaming setting.
    """

    def __init__(self, strategy: str = "mean", fill_value: float = 0.0) -> None:
        if strategy not in ("mean", "median", "constant"):
            raise ValueError("strategy must be 'mean', 'median', or 'constant'.")
        self.strategy = strategy
        self.fill_value = float(fill_value)
        self.statistics_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "SimpleImputer":
        """
        Compute fill values from the full dataset.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data, may contain NaN values.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If X is not 2-D.

        Time complexity: O(n_samples * n_features)
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D with shape (n_samples, n_features).")
        if self.strategy == "mean":
            stats = np.nanmean(X, axis=0)
        elif self.strategy == "median":
            stats = np.nanmedian(X, axis=0)
        else:
            stats = np.full(X.shape[1], self.fill_value, dtype=float)
        self.statistics_ = np.where(np.isnan(stats), 0.0, stats)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Replace NaN values in X with the learned fill values.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Data to impute.

        Returns
        -------
        np.ndarray
            Imputed array, same shape as X.

        Raises
        ------
        RuntimeError
            If called before fit or partial_fit.

        Time complexity: O(n_samples * n_features)
        """
        if self.statistics_ is None:
            raise RuntimeError("SimpleImputer must be fitted before transform.")
        X = np.asarray(X, dtype=float).copy()
        nan_mask = np.isnan(X)
        if np.any(nan_mask):
            X[nan_mask] = np.take(self.statistics_, np.where(nan_mask)[1])
        return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def partial_fit(self, X: np.ndarray) -> "SimpleImputer":
        """
        Update fill values with a new chunk of data.

        We keep a running count and sum per feature so the fill value
        stays accurate as more chunks arrive. The constant strategy
        ignores new data since the fill value never changes.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            New chunk of data, may contain NaN values.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If X is not 2-D.

        Time complexity: O(n_samples * n_features) per chunk
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D.")

        if self.strategy == "constant":
            # constant fill value never changes — nothing to update
            if self.statistics_ is None:
                self.statistics_ = np.full(X.shape[1], self.fill_value)
            return self

        if not hasattr(self, "_chunk_count"):
            # first chunk — initialise running accumulators
            self._chunk_count = np.zeros(X.shape[1])
            self._chunk_sum = np.zeros(X.shape[1])

        # only count and sum the non-NaN values per feature
        valid_mask = ~np.isnan(X)
        self._chunk_count += valid_mask.sum(axis=0)
        self._chunk_sum += np.where(valid_mask, X, 0).sum(axis=0)

        if self.strategy == "mean":
            # running mean = total sum divided by total count
            safe_count = np.where(self._chunk_count == 0, 1, self._chunk_count)
            stats = self._chunk_sum / safe_count
        elif self.strategy == "median":
            # median cannot be computed exactly in a streaming way
            # so we use the current chunk as the best approximation
            stats = np.nanmedian(X, axis=0)

        # fall back to 0 for any feature that was all NaN
        self.statistics_ = np.where(np.isnan(stats), 0.0, stats)
        return self