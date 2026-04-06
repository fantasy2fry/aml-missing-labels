import numpy as np
from sklearn.linear_model import LogisticRegression

def novel_logreg(X: np.ndarray,y: np.ndarray, n_iter: int) -> LogisticRegression:

    """
    Logistic regression model combining semi-supervised learning and active learning for missing labels.

    Parameters
    -------
    X : np.ndarray
    y: np.ndarray
        Labels with missing values {0, 1, -1}.
    n_iter: int
        Number of iterations, used to calculate step size 

    Returns
    -------
    LogisticRegression: fitted model
    """

    # copy y for training purposes
    y_copied = y.copy()

    labeled_idx = np.where(y != -1)[0]
    unlabeled_idx = np.where(y == -1)[0]

    unlabeled_mask = (y == -1)

    labeled_X = X[labeled_idx]
    labeled_y = y[labeled_idx] 

    unlabeled_X = X[unlabeled_idx]
    unlabeled_y = y[unlabeled_idx]

    training_X = labeled_X
    training_y = labeled_y
    
    model = LogisticRegression()
    model.fit(training_X, training_y)

    for i in range(n_iter):

        step_size = (i+1)/n_iter

        predicted_y = model.predict_proba(X)[:,0]
        binarize_y = predicted_y < 0.5

        # Build the Active Learning set 
        mask_AL = ((unlabeled_mask) &  (predicted_y < 0.5 + step_size) & (predicted_y >0.5+step_size))
        AL_set = np.where(mask_AL)[0]

        # Build the Self Learning set 
        mask_SSL = ((unlabeled_mask) & ((predicted_y < step_size) | (predicted_y > 1 -step_size)))
        SSL_set = np.where(mask_SSL)[0]
        
        y_copied[AL_set] = binarize_y[AL_set]
        y_copied[SSL_set] = binarize_y[SSL_set]

        # Add Active Learning and Self learning sets to training set
        training_idx = np.concatenate([labeled_idx, AL_set, SSL_set]) 
        
        # Identify the false pseudo-labeled samples
        model = LogisticRegression()
        training_X = X[training_idx]
        training_y = y_copied[training_idx]
        model.fit(training_X, training_y)
        new_predicted_y = model.predict_proba(X)[:,0]
        new_binarize_y = new_predicted_y < 0.5
        mask = np.where(binarize_y != new_binarize_y)[0]

        y_copied[AL_set] = new_binarize_y[AL_set]
        y_copied[SSL_set] = new_binarize_y[SSL_set]

        # Remove pseudo-label observations from training set
        changed_SSL = np.intersect1d(SSL_set, mask)
        SSL_set = np.setdiff1d(SSL_set, changed_SSL)

        training_idx = np.concatenate([labeled_idx, AL_set, SSL_set]) 
    
    training_X = X[training_idx]
    training_y = y_copied[training_idx]
    model = LogisticRegression()
    model.fit(training_X, training_y)
    return model




