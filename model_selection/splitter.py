import numpy as np

class KFold():
    """
    Provides indices for train and validation for k splits. 
    """
    def __init__(self, k_splits=2, seed=None):
        self.k_splits = k_splits
        self.seed = seed

    def stratified_split(self, X, y):
        """
        Return the partition indices of the data, stratified.
        """
        c1, c2 = np.unique(y)

        # Find the idx for both clases and the split sizes for each class
        c1_mask = y == c1
        idx = np.arange(len(X))
        c1_idx = list(idx[c1_mask])
        c2_idx = list(idx[~c1_mask])

        np.random.seed(self.seed)
        np.random.shuffle(c1_idx)
        np.random.seed(self.seed)
        np.random.shuffle(c2_idx)

        c1_split_size = sum(c1_mask)//self.k_splits
        c2_split_size = sum(~c1_mask)//self.k_splits

        # Create partitions
        for i in range(self.k_splits):
            c1_lo = i * c1_split_size
            c1_hi = (i + 1) * c1_split_size
            c2_lo = i * c2_split_size
            c2_hi = (i + 1) * c2_split_size

            if i+1 == self.k_splits:
                test_split_idx = c1_idx[c1_lo:] + c2_idx[c2_lo:]
                train_split_idx = c1_idx[:c1_lo] + c2_idx[:c2_lo]
            else:
                test_split_idx = c1_idx[c1_lo:c1_hi] + c2_idx[c2_lo:c2_hi]
                train_split_idx = c1_idx[:c1_lo] + c1_idx[c1_hi:] + c2_idx[:c2_lo] + c2_idx[c2_hi:]

            # Shuffle the data from all classes
            np.random.seed(self.seed)
            np.random.shuffle(test_split_idx)
            np.random.seed(self.seed)
            np.random.shuffle(train_split_idx)
            yield train_split_idx, test_split_idx

    def split(self, X, y):
        """
        Returns indices for the train and validation sets.
        """
        assert self._size_check(X,y)

        N = y.shape[0]
        np.random.seed(self.seed)
        idx = np.random.permutation(N)
        split_size = int(N/self.k_splits)

        # Yields the train_idx, and val_idx for each k in k_splits
        for k in range(self.k_splits):
            if k+1 == self.k_splits:
                val_idx = idx[k*split_size:]
            else:
                val_idx = idx[k*split_size:(k+1)*split_size]
            train_idx = np.array([k for k in idx if k not in val_idx])
            yield train_idx,val_idx

    def _size_check(self, X, y):
        """
        Makes sure the input and target data has the same number of points.
        """
        return X.shape[0] == y.shape[0]


if __name__ == "__main__":
    ## KFold test
    X = np.loadtxt(fname="../data/olive_train.csv",skiprows=1,delimiter=",",usecols=(1,2))
    y = np.loadtxt(fname="../data/olive_train.csv",skiprows=1,delimiter=",",usecols=(0))

    CV = KFold(k_splits=5, seed=2408)
    
    all_idx = np.zeros(X.shape[0])
    for fold,(train_idx,val_idx) in enumerate(CV.stratified_split(X,y)):
        all_idx[train_idx] = 1
    print(sum(all_idx))
    exit()


    for fold,(train_idx,val_idx) in enumerate(CV.stratified_split(X,y)):
        print("\nFold #{}".format(fold))
        print("Number of training and validation indices: {}, {}".format(len(train_idx),len(val_idx)))
        print("Amount of validation indices in training indices: {}".format(np.isin(train_idx,val_idx).sum()))
