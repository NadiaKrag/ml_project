import numpy as np
from boosting.stump import DecisionStump
from metrics.metrics import accuracy

class AdaBoost:
    """
    The boosting algorithm Adaboost following the algorithm from Bishop page 658.
    """
    def __init__(self, weak_learner="stump", m_estimators=100):
        self.m = m_estimators
        self.weak_learner = weak_learner

    def fit(self, X, y, sample_weight=None):
        N = X.shape[0]
        self.alphas = []
        self.weak_learners = []
        self.class_labels = np.unique(y)
        y = np.where(y == self.class_labels[1], 1, -1)

        if sample_weight is None:
            sample_weight = np.full(N, 1/N)

        for m in range(self.m):
            if self.weak_learner == "stump":
                weak_learner = DecisionStump()
            weak_learner.fit(X, y, sample_weight=sample_weight)
            self.weak_learners.append(weak_learner)

            indicator = np.where(weak_learner.predict(X) == y, 0, 1)
            error = np.dot(sample_weight, indicator) / sample_weight.sum()
            alpha = np.log((1 - error) / error)
            sample_weight = np.multiply(sample_weight, np.exp(alpha*indicator))
            self.alphas.append(alpha)

    def predict(self, X, M=None):
        if M == None:
            M = self.m
        prod = [self.alphas[m] * self.weak_learners[m].predict(X) for m in range(M)]
        preds = np.sign(np.sum(prod,axis=0))
        return np.where(preds == -1, *self.class_labels)

    def find_M_max(self, X_train, y_train, X_test, y_test, draw_plot = False):
        train_results = []
        test_results = []

        # Predicts using m models and save accuracy.
        for m in range(1,self.m + 1):
            train_pred = self.predict(X_train,m)
            test_pred = self.predict(X_test,m)
            train_accuracy = accuracy(train_pred,y_train)
            test_accuracy = accuracy(test_pred,y_test)
            train_results.append(train_accuracy)
            test_results.append(test_accuracy)

        # Find the best scores indices
        max_test_idx = np.argmax(test_results)
        max_test = np.max(test_results)

        if draw_plot:
            self.draw_plot(train_results, test_results,max_test_idx, max_test)
        return max_test_idx