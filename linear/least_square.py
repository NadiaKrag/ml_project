import numpy as np

class LSClassification:
    def __init__(self):
        self.fitted = False

    def fit(self, X, y, sample_weight=None):
        N = X.shape[0]
        self.class_labels = np.unique(y)
        y = np.where(y == self.class_labels[1], 1, -1)

        # Scale input data by sample weights
        if sample_weight is not None:
            X = np.multiply(X.T,sample_weight).T

        # Add extra column of ones for the bias coefficient, making the design matrix
        phi = np.column_stack((np.ones(N), X))
        phi_pseudo_inv = np.linalg.inv(phi.T @ phi) @ phi.T

        self.coefs = phi_pseudo_inv @ y
        self.fitted = True

    def predict(self, X):
        assert self.fitted
        X = np.column_stack((np.ones(len(X)), X))
        preds = np.sign(X @ self.coefs)
        return np.where(preds == -1, *self.class_labels)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    X = np.loadtxt(fname='../data/olive_train.csv',skiprows=1,delimiter=',',usecols=(1,2))
    y = np.loadtxt(fname='../data/olive_train.csv',skiprows=1,delimiter=',',usecols=(0))
    X_test = np.loadtxt(fname='../data/olive_test.csv',skiprows=1,delimiter=',',usecols=(1,2))
    y_test = np.loadtxt(fname='../data/olive_test.csv',skiprows=1,delimiter=',',usecols=(0))
    
    clf = LSClassification()
    clf.fit(X,y)

    train_preds = clf.predict(X)
    test_preds = clf.predict(X_test)
    train_acc = (train_preds == y).mean()
    test_acc = (test_preds == y_test).mean()

    print("Train accuracy: {:>6}".format(train_acc))
    print("Test accuracy: {:>7}".format(test_acc))

    ## Plot decision boundary
    x_1 = np.linspace(X.min(),X.max(),50)
    x_2 = -(clf.coefs[0] + clf.coefs[1] * x_1) / clf.coefs[2]

    fig,ax = plt.subplots(figsize=(13,8))
    ax.plot(x_1,x_2,linestyle="--",c="grey",alpha=0.675)
    ax.scatter(x=X[:,0],y=X[:,1],c=y)
    ax.set_title("Train set")
    plt.show()
