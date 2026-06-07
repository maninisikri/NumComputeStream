from __future__ import annotations
import numpy as np
from numcompute_stream.tree import DecisionTreeClassifier

class RandomForestClassifier:
    """
    A collection of decision trees that vote together on predictions.
    Each tree is trained on a random bootstrap sample of the data and
    considers only a random subset of features at each split. This
    randomness makes the trees different from each other, and when
    they vote together the ensemble tends to do better than any
    single tree would on its own.
    Parameters
    ----------
    n_estimators : int
        How many trees to grow. More trees generally means better
        accuracy but slower training and prediction.
    max_depth : int
        Maximum depth for each individual tree.
    min_samples_split : int
        Minimum samples needed at a node before splitting.
    max_features : int or None
        Features considered at each split. None uses all features.
        'sqrt' is a common choice for classification tasks.
    random_state : int or None
        Seed for reproducibility. None means different results each run.
    """
    def __init__(self, n_estimators=10, max_depth=5,
                 min_samples_split=2, max_features=None,
                 random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees_: list[DecisionTreeClassifier] = []
    def fit(self, X, y):
        """
        Train all trees on bootstrap samples of the full dataset.
        Each tree sees a different random sample of the data with
        replacement, which is what makes the trees diverse.
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training features.
        y : np.ndarray, shape (n_samples,)
            Training labels.
        Returns
        -------
        self
        Time complexity: O(n_estimators * n * d * log n)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]
        # work out how many features each tree should consider
        if self.max_features == "sqrt":
            max_feat = max(1, int(np.sqrt(X.shape[1])))
        elif self.max_features is None:
            max_feat = X.shape[1]
        else:
            max_feat = int(self.max_features)
        self.trees_ = []
        for _ in range(self.n_estimators):
            # bootstrap sample — same size as original but with replacement
            indices = rng.integers(0, n_samples, size=n_samples)
            X_boot = X[indices]
            y_boot = y[indices]
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_feat,
            )
            tree.fit(X_boot, y_boot)
            self.trees_.append(tree)
        return self
    def partial_fit(self, X, y):
        """
        Update the forest with a new chunk of data.
        Each tree gets refitted on a bootstrap sample drawn from
        the new chunk. This keeps the forest roughly up to date
        without needing to store all the old data.
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            New chunk of training features.
        y : np.ndarray, shape (n_samples,)
            New chunk of training labels.
        Returns
        -------
        self
        """
        if not self.trees_:
            # no trees yet — do a full fit on this chunk
            return self.fit(X, y)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]
        if self.max_features == "sqrt":
            max_feat = max(1, int(np.sqrt(X.shape[1])))
        elif self.max_features is None:
            max_feat = X.shape[1]
        else:
            max_feat = int(self.max_features)
        for tree in self.trees_:
            # give each tree a fresh bootstrap sample from the new chunk
            indices = rng.integers(0, n_samples, size=n_samples)
            tree.partial_fit(X[indices], y[indices])
        return self
    def predict(self, X):
        """
        Each tree votes and the majority class wins.
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Samples to classify.
        Returns
        -------
        np.ndarray, shape (n_samples,)
            Predicted class labels by majority vote.
        Raises
        ------
        RuntimeError
            If called before fit or partial_fit.
        Time complexity: O(n_estimators * n * max_depth)
        """
        if not self.trees_:
            raise RuntimeError("Forest must be fitted before calling predict.")
        X = np.asarray(X, dtype=float)
        # collect predictions from every tree
        all_preds = np.array([tree.predict(X) for tree in self.trees_])
        # majority vote for each sample
        result = []
        for sample_preds in all_preds.T:
            values, counts = np.unique(sample_preds, return_counts=True)
            result.append(values[np.argmax(counts)])
        return np.array(result)
    def score(self, X, y):
        """
        Fraction of samples the forest predicted correctly.
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Test features.
        y : np.ndarray, shape (n_samples,)
            True labels.
        Returns
        -------
        float
            Accuracy between 0.0 and 1.0.
        """
        return float(np.mean(self.predict(X) == np.asarray(y)))
    