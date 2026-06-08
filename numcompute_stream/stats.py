from __future__ import annotations
import numpy as np

def mean(arr, axis=None):
    """Average value across the array, skipping any NaN entries.
    Parameters
    ----------
    arr : array-like
        Input data of any shape.
    axis : int or None
        Axis to reduce along. None flattens the whole array first.
    Returns
    -------
    np.ndarray or float
        Mean value(s). Shape is arr.shape with axis removed.
    Time complexity: O(n)
    """
    arr = np.asarray(arr, dtype=float)
    return np.nanmean(arr, axis=axis)

def median(arr, axis=None):
    """Middle value of the array, skipping any NaN entries.
    Parameters
    ----------
    arr : array-like
        Input data of any shape.
    axis : int or None
        Axis to reduce along.
    Returns
    -------
    np.ndarray or float
    Time complexity: O(n log n)
    """
    arr = np.asarray(arr, dtype=float)
    return np.nanmedian(arr, axis=axis)

def std(arr, axis=None, ddof=0):
    """How spread out the values are, skipping NaNs.
    Parameters
    ----------
    arr : array-like
        Input data of any shape.
    axis : int or None
        Axis to reduce along.
    ddof : int
        Degrees of freedom. 0 for the whole population, 1 for a sample.
    Returns
    -------
    np.ndarray or float
    Time complexity: O(n)
    """
    arr = np.asarray(arr, dtype=float)
    return np.nanstd(arr, axis=axis, ddof=ddof)

def var(arr, axis=None, ddof=0):
    """Variance of the array, skipping NaN entries.
    Parameters
    ----------
    arr : array-like
        Input data of any shape.
    axis : int or None
        Axis to reduce along.
    ddof : int
        Degrees of freedom.
    Returns
    -------
    np.ndarray or float
    Time complexity: O(n)
    """
    arr = np.asarray(arr, dtype=float)
    return np.nanvar(arr, axis=axis, ddof=ddof)

def minimum(arr, axis=None):
    """Smallest value in the array, ignoring NaNs.
    Parameters
    ----------
    arr : array-like
        Input data.
    axis : int or None
        Axis to reduce along.
    Returns
    -------
    np.ndarray or float
    Time complexity: O(n)
    """
    arr = np.asarray(arr, dtype=float)
    return np.nanmin(arr, axis=axis)

def maximum(arr, axis=None):
    """Largest value in the array, ignoring NaNs.
    Parameters
    ----------
    arr : array-like
        Input data.
    axis : int or None
        Axis to reduce along.
    Returns
    -------
    np.ndarray or float
    Time complexity: O(n)
    """
    arr = np.asarray(arr, dtype=float)
    return np.nanmax(arr, axis=axis)

def summary(arr, axis=None):
    """Quick snapshot of the key descriptive stats in one go.
    Parameters
    ----------
    arr : array-like
        Input data.
    axis : int or None
        Axis to reduce along. None reduces the whole array.
    Returns
    -------
    dict
        Keys: mean, median, std, min, max, count_nan.
    Time complexity: O(n)
    """
    arr = np.asarray(arr, dtype=float)
    return {
        "mean":      np.nanmean(arr, axis=axis),
        "median":    np.nanmedian(arr, axis=axis),
        "std":       np.nanstd(arr, axis=axis),
        "min":       np.nanmin(arr, axis=axis),
        "max":       np.nanmax(arr, axis=axis),
        "count_nan": int(np.sum(np.isnan(arr))),
    }

def welford_update(existing_aggregate, new_value):
    """One step of Welford's online algorithm for running mean and variance.
    Takes the current state and one new value, returns the updated state.
    This lets us compute variance in a single pass without keeping all
    the raw data around.
    Parameters
    ----------
    existing_aggregate : tuple of (int, float, float)
        Current state as (count, mean, M2).
    new_value : float
        The next value to incorporate.
    Returns
    -------
    tuple of (int, float, float)
        Updated (count, mean, M2).
    Time complexity: O(1)
    """
    count, mean_val, m2 = existing_aggregate
    count += 1
    delta = new_value - mean_val
    mean_val += delta / count
    delta2 = new_value - mean_val
    m2 += delta * delta2
    return count, mean_val, m2

def welford_finalize(count, mean_val, m2, ddof=0):
    """Turn the Welford accumulator into a usable mean and variance.
    Parameters
    ----------
    count : int
        Total number of values processed.
    mean_val : float
        Running mean accumulated so far.
    m2 : float
        Sum of squared deviations accumulated so far.
    ddof : int
        0 for population variance, 1 for sample variance.
    Returns
    -------
    tuple of (float, float)
        (mean, variance). Returns (nan, nan) if count is too small.
    Time complexity: O(1)
    """
    if count == 0 or count <= ddof:
        return float("nan"), float("nan")
    variance = m2 / (count - ddof)
    return mean_val, variance

def histogram(arr, bins=10, range=None):
    """Count how many values fall into each bin, ignoring NaNs.
    Parameters
    ----------
    arr : array-like
        Input data, flattened internally.
    bins : int
        Number of equal-width bins. Must be at least 1.
    range : tuple of (float, float) or None
        Lower and upper bin boundaries. None uses the data range.
    Returns
    -------
    counts : np.ndarray, shape (bins,)
    bin_edges : np.ndarray, shape (bins + 1,)
    Raises
    ------
    ValueError
        If bins is less than 1.
    Time complexity: O(n)
    """
    if bins < 1:
        raise ValueError(f"bins must be >= 1, got {bins}.")
    arr = np.asarray(arr, dtype=float)
    clean = arr[~np.isnan(arr)].ravel()
    return np.histogram(clean, bins=bins, range=range)

def quantile(arr, q, axis=None):
    """Value below which q fraction of the data falls, ignoring NaNs.
    Parameters
    ----------
    arr : array-like
        Input data.
    q : float or array-like
        Quantile(s) between 0 and 1. For example 0.5 gives the median.
    axis : int or None
        Axis to reduce along.
    Returns
    -------
    np.ndarray or float
    Raises
    ------
    ValueError
        If any value in q is outside [0, 1].
    Time complexity: O(n log n)
    """
    arr = np.asarray(arr, dtype=float)
    q = np.asarray(q, dtype=float)
    if np.any((q < 0) | (q > 1)):
        raise ValueError("All quantile values must be in [0, 1].")
    return np.nanpercentile(arr, q * 100, axis=axis)

class StreamingStats:
    """
    Track descriptive statistics incrementally as data arrives in chunks.
    Instead of storing every value we have seen, we keep running totals
    that get updated each time a new chunk comes in. Memory usage stays
    constant no matter how much data has been processed.
    """
    def __init__(self, window_size: int | None = None) -> None:
        """
        Parameters
        ----------
        window_size : int or None
            If given, only the last window_size chunks are used
            to compute stats. None means use everything seen so far.
        """
        self.window_size = window_size
        self._count = 0
        self._mean = None
        self._m2 = None
        self._chunks: list = []
        self._max_chunks = window_size
    def update_stats(self, X_chunk: np.ndarray) -> "StreamingStats":
        """
        Feed in a new chunk and update all running statistics.
        Uses Welford's algorithm for mean and variance so we never
        need to store raw values from previous chunks.
        Parameters
        ----------
        X_chunk : array-like, shape (n_samples,) or (n_samples, n_features)
            New chunk of data to incorporate.
        Returns
        -------
        self
        """
        X_chunk = np.asarray(X_chunk, dtype=float).ravel()
        self._chunks.append(X_chunk)
        if self._max_chunks is not None and len(self._chunks) > self._max_chunks:
            # drop the oldest chunk when the window is full
            self._chunks.pop(0)
        # update running mean and variance one value at a time
        for val in X_chunk:
            if np.isnan(val):
                continue
            self._count += 1
            if self._mean is None:
                self._mean = val
                self._m2 = 0.0
            else:
                delta = val - self._mean
                self._mean += delta / self._count
                delta2 = val - self._mean
                self._m2 += delta * delta2
        return self
    def result(self) -> dict:
        """
        Return all statistics from data seen so far.
        Returns
        -------
        dict
            Keys: mean, variance, std, min, max, count.
            Returns NaN values if no data has been fed in yet.
        """
        if self._count == 0 or self._mean is None:
            return {
                "mean": float("nan"),
                "variance": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "count": 0,
            }
        variance = self._m2 / self._count if self._count > 1 else 0.0
        all_data = np.concatenate(self._chunks)
        return {
            "mean": self._mean,
            "variance": variance,
            "std": float(np.sqrt(variance)),
            "min": float(np.nanmin(all_data)),
            "max": float(np.nanmax(all_data)),
            "count": self._count,
        }
    def quantile(self, q: float) -> float:
        """
        Compute a quantile from all chunks stored so far.
        Parameters
        ----------
        q : float
            Quantile between 0 and 1. For example 0.5 gives the median.
        Returns
        -------
        float
        Raises
        ------
        ValueError
            If no data yet or q is out of range.
        """
        if not self._chunks:
            raise ValueError("No data yet — call update_stats first.")
        if q < 0 or q > 1:
            raise ValueError("q must be between 0 and 1.")
        all_data = np.concatenate(self._chunks)
        return float(np.nanpercentile(all_data, q * 100))
    def histogram(self, bins: int = 10):
        """
        Compute a histogram from all chunks stored so far.
        Parameters
        ----------
        bins : int
            Number of bins. Must be at least 1.
        Returns
        -------
        counts : np.ndarray
        bin_edges : np.ndarray
        Raises
        ------
        ValueError
            If no data yet or bins is less than 1.
        """
        if not self._chunks:
            raise ValueError("No data yet — call update_stats first.")
        if bins < 1:
            raise ValueError("bins must be at least 1.")
        all_data = np.concatenate(self._chunks)
        clean = all_data[~np.isnan(all_data)]
        return np.histogram(clean, bins=bins)
    def reset(self) -> "StreamingStats":
        """Clear everything and start from scratch."""
        self._count = 0
        self._mean = None
        self._m2 = None
        self._chunks = []
        return self