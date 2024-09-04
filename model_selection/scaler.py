import numpy as np

class Standardizer:
    """Standardization tool.

    Standardizes datasets by first getting mean, standard deviation of one set
    which can then be used to transform other datasets.

    Formula:
    (x - x_mean)/x_stdev

    Parameters
    ----------
    X : ndarray, optional
        The data used for initialization.

    Attributes
    ----------
    fitted : bool
        Whether the standardizer has been fitted.
    X_means : ndarray
        Array of means for each feature.
    X_std : ndarray
        Array of standard deviations of each feature.
    n_feat : int
        The number of features of the fitted dataset.

    """
    def __init__(self, X = None):
        self.fitted = False

        self.X_mean = None
        self.X_std = None
        self.n_feat = None

        if X is not None:
            self.fit(X)

    def fit(self, X):
        """Store mean, stdeviation, number of feat of X."""
        self.X_mean =  X.mean(axis=0)
        self.X_std = X.std(axis=0)
        self.n_feat = X.shape[1]
        self.fitted = True

    def transform(self, X):
        """Standardize X if fitted and the shape match."""
        assert self.fitted
        assert self.n_feat == X.shape[1]
        return (X - self.X_mean)/self.X_std

    def fit_transform(self, X):
        """Fit X and return its transformation."""
        self.fit(X)
        return self.transform(X)
