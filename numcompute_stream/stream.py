from __future__ import annotations
import numpy as np
from numcompute_stream.metrics import StreamingMetrics

class StreamTrainer:
    """
    Manages the full streaming training loop in one place.
    Handles feeding chunks to the model and preprocessor,
    scoring after each chunk, and keeping a log of how the
    metrics change over time.
    Parameters
    ----------
    model : object
        Any classifier with fit(), partial_fit() and predict() methods.
        Works with DecisionTreeClassifier or RandomForestClassifier.
    preprocessor : object or None
        Optional transformer with partial_fit() and transform() methods.
        Pass None if you don't need preprocessing.
    window_size : int or None
        Rolling window for metric tracking. None accumulates everything.
    """
    def __init__(self, model, preprocessor=None, window_size=None):
        self.model = model
        self.preprocessor = preprocessor
        self.metrics_ = StreamingMetrics(window_size=window_size)
        # logs store one value per chunk so we can plot progress later
        self.accuracy_log_: list[float] = []
        self.precision_log_: list[float] = []
        self.recall_log_: list[float] = []
        self.f1_log_: list[float] = []
        self.chunk_count_ = 0
    def fit_chunk(self, X, y):
        """
        Train on one chunk of data and log the resulting metrics.
        Preprocesses the chunk first if a preprocessor was provided,
        then calls partial_fit on the model so it learns incrementally.
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix for this chunk.
        y : np.ndarray, shape (n_samples,)
            Labels for this chunk.
        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if self.preprocessor is not None:
            # update the preprocessor's stats then transform
            self.preprocessor.partial_fit(X)
            X = self.preprocessor.transform(X)
        self.model.partial_fit(X, y)
        # score on the same chunk we just trained on
        self.score_chunk(X, y, already_transformed=True)
        self.chunk_count_ += 1
        return self
    def score_chunk(self, X, y, already_transformed=False):
        """
        Evaluate the model on a chunk and update the metric logs.
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Features to evaluate on.
        y : np.ndarray, shape (n_samples,)
            True labels.
        already_transformed : bool
            Set to True if X has already been through the preprocessor.
            This avoids transforming twice inside fit_chunk.
        Returns
        -------
        dict
            The four metrics for this chunk.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if self.preprocessor is not None and not already_transformed:
            X = self.preprocessor.transform(X)
        y_pred = self.model.predict(X)
        self.metrics_.update(y, y_pred)
        result = self.metrics_.result()
        # append to logs so we can plot them later
        self.accuracy_log_.append(result["accuracy"])
        self.precision_log_.append(result["precision"])
        self.recall_log_.append(result["recall"])
        self.f1_log_.append(result["f1"])
        return result
    def reset(self):
        """
        Clear all logs and metric accumulators.
        Does not reset the model — call fit() on the model directly
        if you want to start training from scratch.
        Returns
        -------
        self
        """
        self.metrics_.reset()
        self.accuracy_log_ = []
        self.precision_log_ = []
        self.recall_log_ = []
        self.f1_log_ = []
        self.chunk_count_ = 0
        return self
    def summary(self):
        """
        Print a quick summary of training progress so far.
        Shows how many chunks have been processed and the
        current value of each metric.
        """
        result = self.metrics_.result()
        print(f"Chunks processed : {self.chunk_count_}")
        print(f"Accuracy         : {result['accuracy']:.4f}")
        print(f"Precision        : {result['precision']:.4f}")
        print(f"Recall           : {result['recall']:.4f}")
        print(f"F1               : {result['f1']:.4f}")