import numpy as np

class LogisticRegression:
    def __init__(self, basis_name = "simple_basis", M_array = np.array([7,3]), s = None):
        self.basis_name = basis_name
        self.basis = None

        self.M_array = M_array
        self.s = s
        self.fitted = False

    def sigmoid(self,a):
        return 1/(1 + np.exp(-a))

    def set_basis(self,X):
        # Only initialises a new basis object if the model has not been fitted,
        # this is to ensure the same intervals for both the training and the predictions
        if self.basis_name == "simple_basis":
            self.basis = simple_basis(X)
            self.M = X.shape[1] + 1

        elif self.basis_name == "rbf":
            self.basis = rbf(X, self.M_array, self.s)
            self.M = np.prod(self.M_array) + 1

        elif self.basis_name == "sbf":
            self.basis = sbf(X, self.M_array, self.s)
            self.M = np.prod(self.M_array) + 1

    def calc_Phi(self, X):
        """
        Creates a NxM matrix called the design matrix Phi where 
        Phi_{nj} = phi_j(x_n) and wher phi_0(x) = 1.
        The function also adds an ekstra coulm of ones for the intersection.
        """
        Phi_matrix = np.ones((X.shape[0],self.M))
        for i in range(X.shape[0]):
            for j in range(1,self.M):
                Phi_matrix[i,j] = self.basis.calc(X[i], j-1)
        return Phi_matrix

    def fit(self,X,y):
        """
        This implementation uses Newton-Raphson update formula to assign the new weights 
        4.99 page 208 bishop
        """
        N = X.shape[0]

        self.class_labels = np.unique(y)
        new_y = np.where(y == self.class_labels[1], 1, 0)

        # Calculates the initial weights
        self.set_basis(X)
        Phi = self.calc_Phi(X)

        Phi_sword = np.linalg.inv(Phi.T @ Phi) @ Phi.T
        w_first = np.dot(Phi_sword, new_y)

        # Uses the initial weights two calculate the new weights
        y = self.sigmoid(np.dot(Phi, w_first.T))
        R_y = np.multiply(np.eye(N), y)
        R = np.multiply(R_y, (1 - R_y))
        R_inv = np.linalg.inv(R)
        z = np.dot(Phi, w_first.T) - np.dot(R_inv, (y - new_y))
        inverse = np.linalg.inv(Phi.T @ R @ Phi) @ Phi.T
        RZ = np.dot(R, z)
        self.w = inverse @ RZ

        self.fitted = True

    def predict(self, X):
        Phi = self.calc_Phi(X)
        preds = self.sigmoid(np.dot(Phi, self.w.T))
        return np.where(preds < 0.5, *self.class_labels)
    
    def predict_prob(self, X):
        Phi = self.calc_Phi(X)
        probs = self.sigmoid(np.dot(Phi, self.w.T))
        return np.column_stack([probs,1-probs])

#-----------------------------------------------
# Basis functions
#-----------------------------------------------

class simple_basis:
    """
    Simple radial basis function
    """
    def __init__(self, X):
        self.X = X

    def calc(self,x,j):
        """
        Calculates the value in Phi_{nj} = phi_j(x_n)
        """
        return x[j]


class rbf:
    """
    Radial basis function.
    Here M need to be a numpy array, where each val represents the number of intervals on that X dimention.
    Ex: M = np.array([2,3]) # is 2 intervals on X[0] and 3 interval on X[1]
    """
    def __init__(self, X, M_array=None,s=None):
        self.M_array = M_array
        self.s = s
        self._initialize(X)

    def _initialize(self, X):
        # Initialize s
        if self.s is None:
            self.s = np.linalg.norm((X.max(0) - X.min(0)) / np.prod(self.M_array), ord = 2) 

        # Initialize the intervals
        intervals = []
        f_interval = np.linspace(X[:,0].min(), X[:,0].max(), self.M_array[0]+1, endpoint=False)[1:]
        s_interval = np.linspace(X[:,1].min(), X[:,1].max(), self.M_array[1]+1, endpoint=False)[1:]
        for i in range(self.M_array[0]):
            first_val = f_interval[i]
            for k in range(self.M_array[1]):
                intervals.append([f_interval[i],s_interval[k]])
        self.intervals = intervals


    def calc(self,x,j):
        """
        Calculates the value in Phi_{nj} = phi_j(x_n)
        """
        return np.exp(-(np.linalg.norm(x-self.intervals[j], ord = 2)**2)/(2*self.s**2))



class sbf:
    """
    Sigmoidial basis function.
    Here M need to be a numpy array, where each val represents the number of intervals on that X dimention.
    Ex: M = np.array([2,3]) # is 2 intervals on X[0] and 3 interval on X[1]
    """
    def __init__(self, X, M_array=None, s=None):
        self.M_array = M_array
        self.s = s
        self._initialize(X)

    def _initialize(self,X):
        # Initialize s
        if self.s is None:
            self.s = np.linalg.norm((X.max(0) - X.min(0)) / np.prod(self.M_array), ord = 2) # <------- M-2 or (M-2) ???

        # Initialize the intervals
        intervals = []
        f_interval = np.linspace(X[:,0].min(), X[:,0].max(), self.M_array[0]+1, endpoint=False)[1:]
        s_interval = np.linspace(X[:,1].min(), X[:,1].max(), self.M_array[1]+1, endpoint=False)[1:]
        for i in range(self.M_array[0]):
            first_val = f_interval[i]
            for k in range(self.M_array[1]):
                intervals.append([f_interval[i],s_interval[k]])
        self.intervals = intervals

    def sigmoid(self,a):
        return 1/(1 + np.exp(-a))

    def calc(self,x,j):
        """
        Calculates the value in Phi_{nj} = phi_j(x_n)
        """
        return self.sigmoid(np.linalg.norm(x-self.intervals[j], ord = 2)/self.s)
