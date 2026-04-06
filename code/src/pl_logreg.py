from pl_algorithms import Model
from sklearn.linear_model import LogisticRegression
import numpy as np
from src.data_prep import convert_to_pl 


def use_saute(X: np.ndarray,y: np.ndarray, n_vars: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Uses Saute Partial Label algorithm for feature selection and assign most probable label for missing labels.
    
    Parameters
    -------
    X : np.ndarray
    y: np.ndarray
        Labels with missing values {0, 1, -1}.
    n_vars: int
        Number of features to select

    Returns
    -------
    tuple with indicies:
        0 - np.ndarray: X only with selected features,
        1 - np.ndarray: y Binary labels obtained from saute algorithm.
    """

    # convert y with missing labels to partially labeled y
    y_pl = convert_to_pl(y)

    # perform saute algorithm 
    saute = Model(X,y_pl)
    X_updated = saute.select_saute(n_vars)
    y_updated = np.argmax(saute.y_confidence, axis=1)

    return X_updated, y_updated

def logreg_with_saute(X: np.ndarray,y: np.ndarray, n_vars: int, use_selection: bool) -> LogisticRegression:
    """
    Logistic regression with data binarized by SAUTE algorithm.


    Parameters
    -------
    X : np.ndarray
    y: np.ndarray
        Labels with missing values {0, 1, -1}.
    n_vars: int
        Number of features to select
    use_selection: bool
        Wheter to use full X or subset of X with selected features by saute algorithm

    Returns
    -------
    LogisticRegression: fitted model
        
    """
    # get X and y updated by saute algorithm
    X_updated, y_updated = use_saute(X,y, n_vars)
    
    # use logistic regression on data updated by saute
    logreg = LogisticRegression()
    X_to_fit = X if not(use_selection) else X_updated
    logreg.fit(X_to_fit, y_updated)

    return logreg