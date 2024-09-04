import numpy as np
from scipy.stats import multivariate_normal

class GaussianMixture:
    """Gaussian Mixture Model class.

    Implementation of the EM algorithm featured in Bishop p. 438-439.

    Parameters
    ----------
    fitted : bool
        Whether the model has been fitted to training data or not.
    means : array
        Array of current means for each Gaussian.
    covars : array
        Array of current covariance matrices for each Gaussian.
    weights : array
        Array of current weights for each Gaussian.


    Attributes
    ----------
    K : int
        Number of Gaussians.
    init_means : array, optional
        Array of initial means for Gaussian w. len(init_means) == K.
    seed : int, default: None
        Random seed passed to numpy.random when initating random means.
    max_iterations : int, default: 100
        Maximum number of iterations for the EM steps.
    tol : float, default: 1e-05
        Convergence tolerance between loglikelihoods.
    verbose : int, default: 0
        1: Print stats while fitting. 0: No prints.

    """
    def __init__(self, K,init_means=None,seed=None,max_iterations=100,tol=1e-05,verbose=0):
        self.fitted = False
        self.K = K

        self.means = init_means
        self.covars = None
        self.weights = None

        self.seed = seed
        self.max_iterations = max_iterations
        self.tol = tol
        self.verbose = verbose

    def _initialize(self, X):
        """Initiates the means, covariance matrices, and weights.

        If no means where given at creation, this function sets initial means
        to K random locations with uniform probability in min-max range of
        each feature.
        The covariance matrices are all initialized to the covariance of the
        X.
        The Gaussian weights are uniformly initiated to 1/K.

        Attributes
        ----------
        X : ndarray
            The data used for initialization.

        """
        if self.means is None:
            np.random.seed(self.seed)
            self.means = np.random.uniform(X.min(axis=0),
                                           X.max(axis=0),
                                          (self.K, X.shape[1]))

        cov = np.cov(X,rowvar=False)
        self.covars = np.full((self.K, *cov.shape), cov)

        self.weights = np.full(self.K, 1/self.K)

    def _loglike(self, X):
        """Return loglikelihood of X given current means, covariance, matrices.

        Attributes
        ----------
        X : ndarray
            The data to evaluate.

        Returns
        -------
        float
            The loglikelihood.

        """
        ll = []
        for k in range(self.K):
            ll.append(multivariate_normal(self.means[k],self.covars[k]).pdf(X))
        ll = np.column_stack(ll) * self.weights
        ll = np.log(ll.sum(axis=1)).sum()
        return ll

    def _eval_gaussians(self, X):
        """Compute NxK array with evaluated weighted Gaussians of X."""
        p = np.zeros((X.shape[0], self.K))
        for k in range(self.K):
            p[:,k] = self.weights[k] * multivariate_normal(self.means[k],self.covars[k]).pdf(X)
        return p

    def _p_x(self,X):
        """Compute p(x) - the superposition of K Gaussians for each x.

        Attributes
        ----------
        X : ndarray
            The data to evaluate.

        Returns
        -------
        ndarray
            Nx1 array of p(x) for each sample in X.

        """
        return self._eval_gaussians(X).sum(axis=1,keepdims=True)

    def _expectation(self, X):
        """The expectation step of EM algorithm. Returns gammas.

        Attributes
        ----------
        X : ndarray
            The data to evaluate.

        Returns
        -------
        ndarray
            NxK gamma array.

        """
        gaussians = self._eval_gaussians(X)
        return gaussians / gaussians.sum(axis=1,keepdims=True)

    def _maximization(self, X, gammas):
        """The maximization step of EM algorithm.

        Update weights, means, and covariance matrices using
        gammas from expectation step.

        Attributes
        ----------
        X : ndarray
            The data to evaluate.
        gammas : ndarray
            Output from _expectation function.

        """
        for k in range(self.K):
            self.weights[k] = gammas[:,k].sum() / X.shape[0]
            self.means[k] = np.sum(gammas[:,[k]] * X, axis=0) / gammas[:,k].sum()

            diff = (X - self.means[[k]])
            self.covars[k] = np.dot( diff.T, diff * gammas[:,[k]])
            self.covars[k] /= gammas[:,k].sum()

            # Squeeze to remove dimensions when data only have 1 feature
            self.covars[k] = self.covars[k].squeeze()

    def fit(self, X):
        """Run the EM algorithm and fit model.

        Starts by initializing the Gaussians before running the algorithm.
        Runs a maximum of max_iterations but stops if the loglikelihood 
        converges by the tolerance tol.  
        If the covariance matrix becomes singular, the function calls
        itself and initiate new means.
        Prints stats after fitting if verbose.

        Attributes
        ----------
        X : ndarray
            The data to fit.

        """
        self._initialize(X)

        converged = False
        ll_old = np.inf
        for i in range(self.max_iterations):
            try:
                ll = self._loglike(X)

                # Compare log likelihood with previous value to see if converged
                if np.allclose(ll_old, ll, atol=self.tol):
                    converged = True
                    break
                ll_old = ll

                gammas = self._expectation(X)
                self._maximization(X, gammas)
            except np.linalg.LinAlgError as err:
                if 'singular matrix' in str(err):
                    if self.verbose:
                        print('Singular matrix! Restarting with new means.')
                    self.means = None
                    self.fit(X)
                    return
                else:
                    raise

        self.fitted = True
        if self.verbose:
            print('Iterations: {:4d}, LogL: {:4.2f}, Converged: {}'.format(i+1,ll,converged))

    def predict(self, X):
        """Predict X if fitted."""
        assert self.fitted
        return self._expectation(X,self.means,self.covars,self.weights).argmax(axis=1)

if __name__ == "__main__":
    data = np.loadtxt(fname='../data/olive_train.csv',skiprows=1,delimiter=',',usecols=(1,2))
    gmm = GaussianMixture(2,max_iterations=1000)
    gmm.fit(data)
    covars = gmm.covars
    means = gmm.means
    weights = gmm.weights
    print('Means: ', means)
    print('Covariances: ', covars)
    print('Weights: ', weights)


    ### PLOT STUFF
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

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

    w_factor = 0.2 / weights.max()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data[:,0],data[:,1])
    for i in range(len(means)):
        draw_ellipse(means[i],covars[i],ax,alpha=weights[i] * w_factor)
    plt.show()
