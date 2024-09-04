import numpy as np
from model_selection.splitter import KFold
from model_selection.scaler import Standardizer
from metrics.metrics import accuracy
from collections import defaultdict

class GridSearch:
    """
    Exhaustive grid search class utilizing k-fold cross-validation for each
    combination of parameters.

    Grid search is a method for optimizing hyperparamters.

    Parameters
    ----------
    param_combos : list
        All the combinations of parameter values.
    fitted : bool
        Whether the grid search has run or not.

    Attributes
    ----------
    model : class
        The model to optimize hyperparameters for.
    params : dict
        A dictionary containing the parameter names as keys and the
        range of values to test as values of the keys.
    k_splits : integer
        The number of folds in the k-fold cross-validation.
    n_repeats : integer
        The number of times to repeat the k-fold cross-validation.
    verbose : 0, 1 or 2
        Verbosity. Prints or not, mate.
    seed : integer
        Randomization of the shuffle of each cross-validation.
    """

    def __init__(self,model,params,k_splits=5,n_repeats=1,verbose=1,seed=None):
        self.model = model.__class__
        self.params = params
        self.k_splits = k_splits
        self.n_repeats = n_repeats

        self.param_combos = list()

        self.fitted = False
        self.verbose = verbose
        self.seed = seed

    def give_info(self):
        """Return the information gained from the last run grid search.

        Returns
        -------
        info : dict
            Dictionary containing the total out-of-fold (oof) accuracy, std,
            and var, as well as the training std, var, and fold accuracy.
            That is:
            - parameters
            - oof accuracy total
            - oof accuracy std
            - oof accuracy var
            - oof accuracy per fold
            - train accuracy std
            - train accuracy var
            - train accuracy per fold
        """
        assert self.fitted
        return self.info

    def give_combos(self):
        """Return the combinations of parameters from the last run grid search.

        Returns
        -------
        param_combos : list
            List containing the parameter combinations in the same order as
            the info dictionary.
        """
        assert self.fitted
        return self.param_combos

    def fit(self,X,y):
        """Run exhaustive grid search on the given model with the parameters
        using the training data (X) and target data (y).
        """
        # We find all combinations of the parameters (backtracking method)
        combos = [[]]
        for vals in self.params.values():
            cur_combo = list()
            for combo in combos:
                for val in vals:
                    cur_combo.append(combo + [val])
            combos = cur_combo

        self.info = {}
        info_keys = ["total_oof_acc", "oof_std", "oof_var", "train_std", "train_var", "oof_acc", "train_acc"]
        for key in info_keys:
            self.info[key] = [0] * len(combos)

        for i in range(self.n_repeats):
            # We go through all combinations of parameters
            for idx,combo in enumerate(combos):
                if self.verbose == 1 and (idx+1) % 10 == 0:
                    print("Running combo #{}, repeat #{}..".format(idx+1, i+1))

                params = dict(zip(self.params.keys(),combo))

                # Then we run cross-validation on the combination
                try:
                    data = self._kfoldcv(params,X,y)

                    for j, info_key in enumerate(info_keys):
                        self.info[info_key][idx] += data[j]
                except RuntimeError:
                    print("Failed fitting model for this combo - continuing..", params)
                    for j, info_key in enumerate(info_keys):
                        self.info[info_key][idx] = -1

        # Normalize each metric by number of iterations
        for key, val in self.info.items():
            for i,v in enumerate(val):
                if v is None:
                    self.info[key][i] = -1
                else:
                    self.info[key][i] = v / self.n_repeats

        if self.verbose > 1:
            for key in self.info.keys():
                vals = list()
                for val in self.info[key]:
                    vals.append(val)
                print("{}: {}".format(key,vals))

        self.fitted = True

    def _kfoldcv(self,params,X,y):
        """Run a k-fold cross-validation on one set of parameters with the
        given training data (X) and target data (y).
        """
        folds = KFold(k_splits=self.k_splits,seed=self.seed)
        data_list = list()

        if self.verbose > 1:
            print("Parameters are {}".format(params))

        oof_pred = np.zeros(X.shape[0])
        oof_accs = list()
        train_accs = list()

        for idx_train,idx_val in folds.stratified_split(X,y):
            scaler = Standardizer()
            X_train = scaler.fit_transform(X[idx_train])
            X_val = scaler.transform(X[idx_val])

            try:
                clf = self.model(**params)
                clf.fit(X_train,y[idx_train])

                oof_pred[idx_val] = clf.predict(X_val)
                oof_acc = accuracy(oof_pred[idx_val],y[idx_val])

                train_pred = clf.predict(X_train)
                train_acc = accuracy(train_pred,y[idx_train])

                oof_accs.append(oof_acc)
                train_accs.append(train_acc)
            except:
                self.param_combos.append(params)
                raise RuntimeError

        data_list.append(accuracy(oof_pred,y))
        data_list.append(np.std(oof_accs))
        data_list.append(np.var(oof_accs))
        data_list.append(np.std(train_accs))
        data_list.append(np.var(train_accs))
        data_list.append(np.array(oof_accs))
        data_list.append(np.array(train_accs))

        self.param_combos.append(params)

        return data_list

if __name__ == "__main__":
    pass
