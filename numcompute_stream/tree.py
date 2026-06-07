from __future__ import annotations
import numpy as np

class DecisionNode:
    """A single node in the decision tree.
    Leaf nodes store a predicted class value.
    Internal nodes store the feature and threshold used to split the data.
    """
    def __init__(self, feature=None, threshold=None,
                 left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    def is_leaf(self):
        """Returns True if this node makes a prediction rather than a split."""
        return self.value is not None

class DecisionTreeClassifier:
    """
    A depth-limited decision tree that splits on Gini impurity.
    Supports both batch training via fit() and incremental updates
    via partial_fit() for streaming data settings.
    Parameters
    ----------
    max_depth : int
        How deep the tree is allowed to grow. Deeper trees fit
        training data more closely but can overfit.
    min_samples_split : int
        Minimum number of samples needed at a node before we
        try to split it further.
    max_features : int or None
        How many features to consider at each split. None means
        use all of them. Setting this lower speeds things up and
        adds randomness — useful for random forests.
    """
    def __init__(self, max_depth=5, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root = None
    def fit(self, X, y):
        """
        Build the tree from scratch on a full dataset.
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training features.
        y : np.ndarray, shape (n_samples,)
            Training labels.
        Returns
        -------
        self
        Raises
        ------
        ValueError
            If X and y have different numbers of rows.
        Time complexity: O(n * d * log n) where d is number of features
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        self.root = self._grow(X, y, depth=0)
        return self
    def partial_fit(self, X, y):
        """
        Update the tree with a new chunk of data.
        For now this refits the tree on the new chunk. In a real
        streaming system you would want to update only the affected
        branches, but refitting is a reasonable approximation that
        keeps the implementation tractable.
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
        return self.fit(X, y)
    def predict(self, X):
        """
        Predict the class label for each sample in X.
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Samples to classify.
        Returns
        -------
        np.ndarray, shape (n_samples,)
            Predicted class labels.
        Raises
        ------
        RuntimeError
            If called before fit or partial_fit.
        Time complexity: O(n * max_depth)
        """
        if self.root is None:
            raise RuntimeError("Tree must be fitted before calling predict.")
        X = np.asarray(X, dtype=float)
        return np.array([self._traverse(x, self.root) for x in X])
    def transform(self, X):
        """
        Return predictions as a column vector.
        Allows the tree to sit inside a Pipeline as the last step.
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Samples to classify.
        Returns
        -------
        np.ndarray, shape (n_samples, 1)
        """
        return self.predict(X).reshape(-1, 1)
    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        if y is not None:
            return self.fit(X, y).transform(X)
        return self.transform(X)
    def _gini(self, y):
        """Gini impurity — 0 means perfectly pure, higher means more mixed."""
        if len(y) == 0:
            return 0.0
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1.0 - float(np.sum(probs ** 2))
    def _best_split(self, X, y):
        """Find the feature and threshold that reduces impurity the most."""
        n_samples, n_features = X.shape
        best_gain = -1
        best_feature = None
        best_threshold = None
        parent_gini = self._gini(y)
        features = np.arange(n_features)
        if self.max_features is not None:
            features = np.random.choice(
                n_features,
                size=min(self.max_features, n_features),
                replace=False
            )
        for feature in features:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                left_gini = self._gini(y[left_mask])
                right_gini = self._gini(y[right_mask])
                n_left = left_mask.sum()
                n_right = right_mask.sum()
                weighted_gini = (n_left * left_gini + n_right * right_gini) / n_samples
                gain = parent_gini - weighted_gini
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold
    def _grow(self, X, y, depth):
        """Recursively build the tree by splitting nodes until we hit a stopping condition."""
        n_samples = len(y)
        if (depth >= self.max_depth or
                n_samples < self.min_samples_split or
                len(np.unique(y)) == 1):
            return DecisionNode(value=self._most_common(y))
        feature, threshold = self._best_split(X, y)
        if feature is None:
            return DecisionNode(value=self._most_common(y))
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        left = self._grow(X[left_mask], y[left_mask], depth + 1)
        right = self._grow(X[right_mask], y[right_mask], depth + 1)
        return DecisionNode(feature=feature, threshold=threshold,
                            left=left, right=right)
    def _most_common(self, y):
        """Return whichever class label appears most often in y."""
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]
    def _traverse(self, x, node):
        """Walk down the tree for a single sample until we hit a leaf."""
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)