from __future__ import annotations
from collections import deque
import numpy as np

def _validate_inputs(y_true, y_pred):
    """Make sure both arrays are 1-D and the same length before we do anything."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError(
            f"y_true and y_pred must be 1-D. "
            f"Got shapes {y_true.shape} and {y_pred.shape}."
        )
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"Length mismatch: {y_true.shape[0]} vs {y_pred.shape[0]}."
        )
    return y_true, y_pred

def accuracy(y_true, y_pred):
    """Fraction of predictions that matched the true label.
    Parameters
    ----------
    y_true : array-like, shape (n,)
        Ground-truth labels.
    y_pred : array-like, shape (n,)
        Predicted labels.
    Returns
    -------
    float
        A number between 0.0 and 1.0.
    Raises
    ------
    ValueError
        If inputs are not 1-D or have different lengths.
    Time complexity: O(n)
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    return float(np.mean(y_true == y_pred))

def confusion_matrix(y_true, y_pred):
    """Build a class-by-class count of where predictions landed.
    Each row is the true class, each column is the predicted class.
    The diagonal holds the correct predictions.
    Parameters
    ----------
    y_true : array-like, shape (n,)
        Ground-truth labels.
    y_pred : array-like, shape (n,)
        Predicted labels.
    Returns
    -------
    np.ndarray, shape (C, C)
        Integer matrix where entry [i, j] counts samples with true
        label i predicted as j.
    Raises
    ------
    ValueError
        If inputs are not 1-D or have different lengths.
    Time complexity: O(n + C^2)
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    true_idx = np.array([label_to_idx[l] for l in y_true])
    pred_idx = np.array([label_to_idx[l] for l in y_pred])
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (true_idx, pred_idx), 1)
    return cm

def precision(y_true, y_pred, average="macro"):
    """How many of the predicted positives were actually correct.
    Computed as TP / (TP + FP) per class, then averaged across classes.
    Parameters
    ----------
    y_true : array-like, shape (n,)
        Ground-truth labels.
    y_pred : array-like, shape (n,)
        Predicted labels.
    average : str
        'macro' for unweighted mean across classes.
        'binary' for the positive class only (needs exactly 2 classes).
    Returns
    -------
    float
    Raises
    ------
    ValueError
        If inputs are invalid or 'binary' is used with more than 2 classes.
    Time complexity: O(n * C)
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    if average == "binary":
        if len(classes) > 2:
            raise ValueError(
                f"average='binary' requires exactly 2 classes, found {len(classes)}."
            )
        pos = classes[-1]
        tp = float(np.sum((y_pred == pos) & (y_true == pos)))
        fp = float(np.sum((y_pred == pos) & (y_true != pos)))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    per_class = np.zeros(len(classes))
    for i, cls in enumerate(classes):
        tp = float(np.sum((y_pred == cls) & (y_true == cls)))
        fp = float(np.sum((y_pred == cls) & (y_true != cls)))
        per_class[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return float(np.mean(per_class))

def recall(y_true, y_pred, average="macro"):
    """How many of the actual positives the model managed to find.
    Computed as TP / (TP + FN) per class, then averaged.
    Parameters
    ----------
    y_true : array-like, shape (n,)
        Ground-truth labels.
    y_pred : array-like, shape (n,)
        Predicted labels.
    average : str
        'macro' or 'binary'. Same rules as precision().
    Returns
    -------
    float
    Raises
    ------
    ValueError
        If inputs are invalid or 'binary' is used with more than 2 classes.
    Time complexity: O(n * C)
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    if average == "binary":
        if len(classes) > 2:
            raise ValueError(
                f"average='binary' requires exactly 2 classes, found {len(classes)}."
            )
        pos = classes[-1]
        tp = float(np.sum((y_pred == pos) & (y_true == pos)))
        fn = float(np.sum((y_pred != pos) & (y_true == pos)))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    per_class = np.zeros(len(classes))
    for i, cls in enumerate(classes):
        tp = float(np.sum((y_pred == cls) & (y_true == cls)))
        fn = float(np.sum((y_pred != cls) & (y_true == cls)))
        per_class[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(np.mean(per_class))

def f1(y_true, y_pred, average="macro"):
    """Harmonic mean of precision and recall.
    A good single number when you care about both false positives
    and false negatives. Returns 0.0 when both are zero.
    Parameters
    ----------
    y_true : array-like, shape (n,)
        Ground-truth labels.
    y_pred : array-like, shape (n,)
        Predicted labels.
    average : str
        'macro' or 'binary'.
    Returns
    -------
    float
    Time complexity: O(n * C)
    """
    p = precision(y_true, y_pred, average=average)
    r = recall(y_true, y_pred, average=average)
    return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0

def mse(y_true, y_pred):
    """Average squared difference between predictions and ground truth.
    Parameters
    ----------
    y_true : array-like, shape (n,)
        Ground-truth values.
    y_pred : array-like, shape (n,)
        Predicted values.
    Returns
    -------
    float
    Raises
    ------
    ValueError
        If inputs are not 1-D or have different lengths.
    Time complexity: O(n)
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))

def roc_curve(y_true_binary, y_scores):
    """Compute the ROC curve for a binary classifier.
    Sweeps through score thresholds from high to low and records
    the false positive rate and true positive rate at each step.
    Parameters
    ----------
    y_true_binary : array-like, shape (n,)
        Binary ground-truth labels (0 or 1).
    y_scores : array-like, shape (n,)
        Confidence scores for the positive class.
    Returns
    -------
    fpr : np.ndarray
        False positive rates, starting at 0.
    tpr : np.ndarray
        True positive rates, starting at 0.
    thresholds : np.ndarray
        Threshold values in decreasing order.
    Raises
    ------
    ValueError
        If inputs are not 1-D, different lengths, or not binary.
    Time complexity: O(n log n)
    """
    y_true_binary, y_scores = _validate_inputs(y_true_binary, y_scores)
    unique_labels = np.unique(y_true_binary)
    if len(unique_labels) != 2:
        raise ValueError(
            f"roc_curve requires exactly 2 classes, found {len(unique_labels)}."
        )
    desc_idx = np.argsort(-y_scores)
    y_sorted = y_true_binary[desc_idx]
    thresholds = y_scores[desc_idx]
    total_pos = float(np.sum(y_true_binary == 1))
    total_neg = float(np.sum(y_true_binary == 0))
    tp_cumsum = np.cumsum(y_sorted == 1).astype(float)
    fp_cumsum = np.cumsum(y_sorted == 0).astype(float)
    tpr = tp_cumsum / total_pos if total_pos > 0 else np.zeros_like(tp_cumsum)
    fpr = fp_cumsum / total_neg if total_neg > 0 else np.zeros_like(fp_cumsum)
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    thresholds = np.concatenate([[thresholds[0] + 1], thresholds])
    return fpr, tpr, thresholds

def auc(fpr, tpr):
    """Area under the ROC curve using the trapezoidal rule.
    Parameters
    ----------
    fpr : array-like, shape (n,)
        False positive rates on the x-axis.
    tpr : array-like, shape (n,)
        True positive rates on the y-axis.
    Returns
    -------
    float
        1.0 means perfect, around 0.5 means no better than random.
    Time complexity: O(n)
    """
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    return float(np.trapezoid(tpr, fpr))

class StreamingAccuracy:
    """
    Track accuracy incrementally as predictions arrive chunk by chunk.
    Instead of recomputing from scratch each time, we just keep a
    running count of correct predictions and total predictions seen.
    Call update() after each chunk, result() to get the current
    accuracy, and reset() to start fresh.
    """
    def __init__(self, window_size: int | None = None) -> None:
        """
        Parameters
        ----------
        window_size : int or None
            If given, only the last window_size chunks contribute
            to the result. None means accumulate everything.
        """
        self.window_size = window_size
        self._correct = 0
        self._total = 0
        self._window: deque = deque(maxlen=window_size)
    def update(self, y_true, y_pred) -> "StreamingAccuracy":
        """
        Feed in the latest chunk of predictions.
        Parameters
        ----------
        y_true : array-like, shape (n,)
            True labels for this chunk.
        y_pred : array-like, shape (n,)
            Predicted labels for this chunk.
        Returns
        -------
        self
        """
        y_true, y_pred = _validate_inputs(y_true, y_pred)
        chunk_correct = int(np.sum(y_true == y_pred))
        chunk_total = len(y_true)
        if self.window_size is not None:
            # if window is full the oldest chunk gets dropped automatically
            if len(self._window) == self.window_size:
                old_correct, old_total = self._window[0]
                self._correct -= old_correct
                self._total -= old_total
            self._window.append((chunk_correct, chunk_total))
        self._correct += chunk_correct
        self._total += chunk_total
        return self
    def result(self) -> float:
        """
        Return the current accuracy across all chunks seen so far.
        Returns
        -------
        float
            Accuracy in [0.0, 1.0]. Returns 0.0 if no data yet.
        """
        if self._total == 0:
            return 0.0
        return self._correct / self._total
    def reset(self) -> "StreamingAccuracy":
        """Clear all accumulated data and start from scratch."""
        self._correct = 0
        self._total = 0
        self._window.clear()
        return self

class StreamingConfusionMatrix:
    """
    Accumulate a confusion matrix across streaming chunks.
    Useful when you want to see the full picture of where the model
    is making mistakes after processing many chunks of data.
    """
    def __init__(self) -> None:
        self._cm: np.ndarray | None = None
        self._classes: np.ndarray | None = None
    def update(self, y_true, y_pred) -> "StreamingConfusionMatrix":
        """
        Add the latest chunk's predictions to the running matrix.
        Parameters
        ----------
        y_true : array-like, shape (n,)
            True labels for this chunk.
        y_pred : array-like, shape (n,)
            Predicted labels for this chunk.
        Returns
        -------
        self
        """
        y_true, y_pred = _validate_inputs(y_true, y_pred)
        chunk_cm = confusion_matrix(y_true, y_pred)
        new_classes = np.unique(np.concatenate([y_true, y_pred]))
        if self._cm is None:
            # first chunk — just store the matrix directly
            self._cm = chunk_cm
            self._classes = new_classes
        else:
            # expand matrix if new classes appeared in this chunk
            all_classes = np.unique(np.concatenate([self._classes, new_classes]))
            n = len(all_classes)
            expanded = np.zeros((n, n), dtype=int)
            old_idx = np.searchsorted(all_classes, self._classes)
            expanded[np.ix_(old_idx, old_idx)] += self._cm
            new_idx = np.searchsorted(all_classes, new_classes)
            expanded[np.ix_(new_idx, new_idx)] += chunk_cm
            self._cm = expanded
            self._classes = all_classes
        return self
    def result(self) -> np.ndarray:
        """Return the accumulated confusion matrix."""
        if self._cm is None:
            return np.zeros((1, 1), dtype=int)
        return self._cm
    def reset(self) -> "StreamingConfusionMatrix":
        """Clear the matrix and start fresh."""
        self._cm = None
        self._classes = None
        return self

class StreamingMetrics:
    """
    Track accuracy, precision, recall and F1 together across chunks.
    This is the main streaming metrics class to use in the demo
    and pipeline. Call update() after each chunk and result() to
    get all four metrics at once.
    """
    def __init__(self, window_size: int | None = None) -> None:
        """
        Parameters
        ----------
        window_size : int or None
            Rolling window size. None means accumulate everything.
        """
        self.window_size = window_size
        self._y_true_all: list = []
        self._y_pred_all: list = []
        self._window: deque = deque(maxlen=window_size)
    def update(self, y_true, y_pred) -> "StreamingMetrics":
        """
        Feed in the latest chunk of predictions.
        Parameters
        ----------
        y_true : array-like, shape (n,)
            True labels for this chunk.
        y_pred : array-like, shape (n,)
            Predicted labels for this chunk.
        Returns
        -------
        self
        """
        y_true, y_pred = _validate_inputs(y_true, y_pred)
        if self.window_size is not None:
            self._window.append((y_true, y_pred))
            all_true = np.concatenate([c[0] for c in self._window])
            all_pred = np.concatenate([c[1] for c in self._window])
        else:
            self._y_true_all.append(y_true)
            self._y_pred_all.append(y_pred)
            all_true = np.concatenate(self._y_true_all)
            all_pred = np.concatenate(self._y_pred_all)
        self._all_true = all_true
        self._all_pred = all_pred
        return self
    def result(self) -> dict:
        """
        Get all four metrics computed on everything seen so far.
        Returns
        -------
        dict
            Keys: 'accuracy', 'precision', 'recall', 'f1'.
            Returns zeros if no data has been fed in yet.
        """
        if not hasattr(self, "_all_true"):
            return {"accuracy": 0.0, "precision": 0.0,
                    "recall": 0.0, "f1": 0.0}
        return {
            "accuracy":  accuracy(self._all_true, self._all_pred),
            "precision": precision(self._all_true, self._all_pred),
            "recall":    recall(self._all_true, self._all_pred),
            "f1":        f1(self._all_true, self._all_pred),
        }
    def reset(self) -> "StreamingMetrics":
        """Clear everything and start from scratch."""
        self._y_true_all = []
        self._y_pred_all = []
        self._window.clear()
        if hasattr(self, "_all_true"):
            del self._all_true
            del self._all_pred
        return self