import numpy as np
from scipy.special import expit as sigmoid

class GaussianGenerative:
    """
    This is a binary classifier implemented as a Gaussian generative model as
    described by Bishop in sections 4.2.1 and 4.2.2.
    """
    def __init__(self):
        self.fitted = False

    def fit(self, X, y):
        N = X.shape[0]
        self.class_labels, (N1,N2) = np.unique(y, return_counts=True)

        # Find probabilities and means
        p_C1, p_C2 = N1/N, N2/N
        mu1 = np.mean(X[y == 1], axis = 0)
        mu2 = np.mean(X[y == 2], axis = 0)

        # Calculate each class covar and combine them
        s1 = (N1/N)*(X[y == 1] - mu1).T @ (X[y == 1] - mu1)
        s2 = (N2/N)*(X[y == 2] - mu2).T @ (X[y == 2] - mu2)
        covar = (s1 + s2)/N
        covar_inv= np.linalg.inv(covar)

        # Calculate the weight and the interception w_0
        self.w = covar_inv @ (mu1 - mu2)
        self.w_0 = -(1/2)*mu1.T @ covar_inv @ mu1 + (1/2)*mu2.T @ covar_inv @ mu2 + np.log(p_C1/p_C2)
        self.fitted = True
        
        # For plotting
        self.mu1 = mu1
        self.mu2 = mu2
        self.covar = covar

    def predict_prob(self, X):
        assert self.fitted
        probs = np.dot(X, self.w) + self.w_0
        return sigmoid(probs)

    def predict(self, X):
        probs = self.predict_prob(X)
        return np.where(probs >= 0.5, *self.class_labels)



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    X = np.loadtxt(fname='../data/olive_train.csv',skiprows=1,delimiter=',',usecols=(1,2))
    y = np.loadtxt(fname='../data/olive_train.csv',skiprows=1,delimiter=',',usecols=(0))
    X_test = np.loadtxt(fname='../data/olive_test.csv',skiprows=1,delimiter=',',usecols=(1,2))
    y_test = np.loadtxt(fname='../data/olive_test.csv',skiprows=1,delimiter=',',usecols=(0))
    
    clf = GaussianGenerative()
    clf.fit(X,y)
    preds = clf.predict(X_test)

    train_preds = clf.predict(X)
    test_preds = clf.predict(X_test)
    train_acc = (train_preds == y).mean()
    test_acc = (test_preds == y_test).mean()

    print("Train accuracy: {:>6}".format(train_acc))
    print("Test accuracy: {:>7}".format(test_acc))

    ## Plot decision boundaries
    def draw_ellipse(position, covariance, ax=None, **kwargs):
        ax = ax or plt.gca()

        if covariance.shape == (2, 2):
            U, s, Vt = np.linalg.svd(covariance)
            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
            width, height = 2 * np.sqrt(s)
        else:
            angle = 0
            width, height = 2 * np.sqrt(covariance)

        for nsig in range(1, 4):
            ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                                angle, **kwargs))

    def cf(x):
        if x == 1:
            return "magenta"
        else:
            return "turquoise"

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X[:,0],X[:,1],c=np.vectorize(cf)(y))
    draw_ellipse(clf.mu1,clf.covar,ax,color="magenta",alpha=0.2)
    draw_ellipse(clf.mu2,clf.covar,ax,color="turquoise",alpha=0.2)
    plt.show()
