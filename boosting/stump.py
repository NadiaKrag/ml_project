import numpy as np

class DecisionStump:
    """
    Decision stump classifier.

    Parameters
    ----------
    feature : integer
        The feature along which the best stump is found.
    
    threshold : float
        The threshold of self.feature that provides the best stump.
    
    sign : -1 or 1
        The sign of the stump classifier; determines the class below and above
        the threshold.
    
    References:
    Page 17, figure 2.3, in
    https://users.lal.in2p3.fr/kegl/teaching/stages/notes/tutorial?fbclid=IwAR1jY7wmct5fbQNlIn_BpKmDyb56-cn9wDlG4MoIHYSXyfvgVGJ1uBS1sgo
    """

    def __init__(self):
        self.feature = None
        self.threshold = None
        self.sign = None
        self.class_labels = np.array([-1,1])

    def fit(self, X, y, sample_weight=None):
        """Fit the stump according to the given training data.
        
        Attributes
        ----------
        X : ndarray, shape (n_samples, n_features)
            The input (training) data.
        y : array, shape (n_samples)
            Target data relative to X.
        sample_weight : ndarray, shape (n_samples)
            Individual sample weights.
        """
        N, M = X.shape
        self.class_labels = np.unique(y)
        if self.class_labels.shape[0] == 1:
            self.is_constant = True
            self.sign = -1
            return
        y = np.where(y == self.class_labels[1], 1, -1)

        if sample_weight is None:
            sample_weight = np.ones(N)

        constant = np.dot(sample_weight, y)
        best = constant
        for m in range(M):
            sort_idx = np.argsort(X[:,m])
            current = constant
            for n in range(2, N):
                current -= 2 * np.dot(sample_weight[sort_idx[n-1]], y[sort_idx[n-1]])
                if X[sort_idx[n-1],m] != X[sort_idx[n],m]:
                    if np.abs(current) > np.abs(best):
                        best = current
                        self.feature = m
                        self.threshold = (X[sort_idx[n-1], m] + X[sort_idx[n], m]) / 2
        self.sign = np.sign(best)

        self.is_constant = best == constant

    def predict(self, X):
        """Predict classes with the stump according to the given test data.
        
        Attributes
        ----------
        X : ndarray, shape (n_samples, n_features)
            The input (training) data.
        
        Returns
        -------
        y : array, shape (n_samples)
            Returns the predicted class of each sample in X.
        """
        if self.is_constant:
            preds = self.sign * np.ones((X.shape[0]))
        else:
            preds = np.ones(X[:,self.feature].shape)
            preds[X[:,self.feature] < self.threshold] = -1
            preds *= self.sign
        return np.where(preds == -1, *self.class_labels)