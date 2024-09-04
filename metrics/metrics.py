import numpy as np

def accuracy(preds,true):
    """Calculates accuracy of predictions given targets.
        
    Attributes
    ----------
    preds : array, shape (n_samples)
        The predictions.
    y : array, shape (n_samples)
        Target labels.

    Returns
    -------
    float
        Accuracy.
    """
    return (preds == true).mean()

def recall(preds,true):
    """Calculates recall of predictions given targets.
        
    Attributes
    ----------
    preds : array, shape (n_samples)
        The predictions.
    y : array, shape (n_samples)
        Target labels.

    Returns
    -------
    float
        Recall.
    """
    class_labels = np.unique(true)
    class_0 = np.isin(
        np.flatnonzero(true == class_labels[0]),
        np.flatnonzero(preds == class_labels[0])
    )
    class_1 = np.isin(
        np.flatnonzero(true == class_labels[1]),
        np.flatnonzero(preds == class_labels[1])
    )
    return class_0.mean(),class_1.mean()

def precision(preds,true):
    """Calculates precision of predictions given targets.
        
    Attributes
    ----------
    preds : array, shape (n_samples)
        The predictions.
    y : array, shape (n_samples)
        Target labels.

    Returns
    -------
    float
        Precision.
    """
    class_labels = np.unique(true)
    class_0 = np.isin(
        np.flatnonzero(preds == class_labels[0]),
        np.flatnonzero(true == class_labels[0])
    )
    class_1 = np.isin(
        np.flatnonzero(preds == class_labels[1]),
        np.flatnonzero(true == class_labels[1])
    )
    return class_0.mean(),class_1.mean()

def f1_score(preds,true):
    """Calculates f1-score of predictions given targets.
        
    Attributes
    ----------
    preds : array, shape (n_samples)
        The predictions.
    y : array, shape (n_samples)
        Target labels.

    Returns
    -------
    float
        F1-score.
    """
    numerator = np.multiply(precision(preds,true),recall(preds,true))
    denominator = np.array(precision(preds,true)) + np.array(recall(preds,true))
    return 2 * numerator / denominator



if __name__ == "__main__":
    pass

