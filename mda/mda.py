import numpy as np
from scipy.stats import multivariate_normal
from mda.gmm import GaussianMixture

class MixtureDiscriminant:
    """
    Mixture discriminant analysis (MDA) model.

    The MDA is a generative model for classification that assumes that each
    class follows a Gaussian mixture model (GMM). It takes use of the
    GaussianMixture class.

    Parameters
    ----------
    fitted : bool
        Whether the model has been fitted to training data or not.

    Attributes
    ----------
    Ks : ndarray
        Number of components per class (ordered).
    seed : integer, float, default: None
        Randomization parameter for initialization of GMMs.
    max_iterations : integer, default: 100
        The maximum number of iterations of each GMM.
    tol : integer, float, default: 1e-05
        The tolerance of which each GMM has converged.
    verbose : int, default: 0
        Do you want some prints or not, mate?
    """

    def __init__(self,Ks=np.array([2,2]),seed=None,max_iterations=100,tol=1e-05,verbose=0):
        self.Ks = Ks
        self.fitted = False

        self.seed = seed
        self.max_iterations = max_iterations
        self.tol = tol
        self.verbose = verbose

    def fit(self,X,y):
        """Fit the model according to the given training data.
        
        Attributes
        ----------
        X : ndarray, shape (n_samples, n_features)
            The input (training) data.
        y : array, shape (n_samples)
            Target data relative to X.
        """
        self.Cs, freq = np.unique(y,return_counts=True)
        C = len(self.Cs)
        assert len(self.Ks) == len(self.Cs)

        self.proportions = freq / X.shape[0]
        self.gmms = []
        for i in range(C):
            if self.verbose:
                print('Fitting class {} with {} Gaussians'.format(self.Cs[i],self.Ks[i]))
                print('\t',end='')
            gmm = GaussianMixture(self.Ks[i],seed=self.seed,max_iterations=self.max_iterations,tol=self.tol,verbose=self.verbose)
            gmm.fit(X[y == self.Cs[i]])
            self.gmms.append(gmm)
        self.fitted = True


    def predict_prob(self,X):
        """Predict probabilities with the model according to the given test data.
        
        Attributes
        ----------
        X : ndarray, shape (n_samples, n_features)
            The input (training) data.
        
        Returns
        -------
        y : array, shape (n_samples, n_classes)
            Returns the probability of each sample in X for each class.
        """
        assert self.fitted
        class_probs = np.zeros((X.shape[0], len(self.Cs)))
        for c in range(len(self.Cs)):
            class_probs[:,[c]] = self.gmms[c]._p_x(X)
        return class_probs / class_probs.sum(axis=1,keepdims=True)


    def predict(self,X):
        """Predict classes with the model according to the given test data.
        
        Attributes
        ----------
        X : ndarray, shape (n_samples, n_features)
            The input (training) data.
        
        Returns
        -------
        y : array, shape (n_samples)
            Returns the most probable class of each sample in X.
        """
        return np.take(self.Cs, self.predict_prob(X).argmax(axis=1))

    def _sample(self):
        _c = np.random.choice(np.arange(len(self.Cs)), p=self.proportions)
        _gmm = self.gmms[_c]
        _ck = np.random.choice(np.arange(_gmm.K), p=_gmm.weights)
        return np.append( self.Cs[_c],multivariate_normal(_gmm.means[_ck],_gmm.covars[_ck]).rvs())

    def sample(self,n_samples):
        assert self.fitted
        sample = np.array([self._sample() for _ in range(n_samples)])
        sample_X = sample[:,1:]
        sample_y = sample[:,0]
        return sample_X, sample_y

if __name__ == "__main__":
    X = np.loadtxt(fname="../data/olive_train.csv",skiprows=1,delimiter=",",usecols=(1,2))
    y = np.loadtxt(fname="../data/olive_train.csv",skiprows=1,delimiter=",",usecols=(0),dtype=int)

    Ks = np.array([3,5])

    mda = MixtureDiscriminant(Ks=Ks,verbose=0)
    mda.fit(X,y)
    y_pred = mda.predict(X)
    print('Train Accuracy: ', sum(y == y_pred)/len(y))

    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    def draw_ellipse(position, covariance, ax=None, **kwargs):
        """Draw an ellipse with a given position and covariance"""
        ax = ax or plt.gca()

        # Convert covariance to principal axes
        if covariance.shape == (2, 2):
            U, s, Vt = np.linalg.svd(covariance)
            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
            width, height = 2 * np.sqrt(s)
        else:
            angle = 0
            width, height = 2 * np.sqrt(covariance)

        # Draw the Ellipse
        for nsig in range(1, 4):
            ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                                angle, **kwargs))

    def cf(x):
        if x == 1:
            return "magenta"
        else:
            return "turquoise"

    vcf = np.vectorize(cf)

    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111)
    ax.scatter(X[:,0],X[:,1],c=vcf(y))
    for i,K in enumerate(Ks):
        for j in range(K):
            gmm = mda.gmms[i]
            draw_ellipse(gmm.means[j],gmm.covars[j],ax,color=cf(i+1),alpha=0.2)
    plt.show()
