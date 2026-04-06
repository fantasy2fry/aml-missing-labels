import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, precision_score, f1_score, balanced_accuracy_score, roc_auc_score, average_precision_score

class FistaLogisticRegression:
    """
    Logistic Regression with L1 Penalty (Lasso) optimized using Fast Iterative 
    Shrinkage-Thresholding Algorithm (FISTA).
    """
    def __init__(self, lambdas=None, max_iter=1000, tol=1e-4):
        """
        Parameters
        ----------
        lambdas : array-like or None
            A list of lambda values to evaluate. If None, a default exponentially 
            spaced range is used.
        max_iter : int
            Maximum number of FISTA iterations per lambda.
        tol : float
            Tolerance for early stopping criterion based on weight differences.
        """
        if lambdas is None:
            self.lambdas = np.logspace(-4, 2, 50)
        else:
            self.lambdas = np.array(lambdas)
            
        self.max_iter = max_iter
        self.tol = tol
        
        # Internal state to hold training results
        self.coefs_path_ = {}        # Dictionary mapping: lambda -> weights array
        self.validation_scores_ = {} # Dictionary mapping: lambda -> validation score
        
        # Best model parameters found via validation
        self.best_lambda_ = None
        self.coef_ = None            # Feature weights
        self.intercept_ = None       # Bias term
        self.last_measure_ = None

    def _sigmoid(self, z):
        # Clip 'z' to avoid numerical overflow when performing exp(-z)
        z = np.clip(z, -250, 250)
        return 1.0 / (1.0 + np.exp(-z))

    def _soft_threshold(self, w, threshold):
        """
        Soft-Thresholding Operator (Proximal step for L1 norm).
        We don't apply shrinkage to the intercept term (w[0]).
        """
        w_new = np.zeros_like(w)
        w_new[0] = w[0] # Intercept is entirely unpenalized!
        
        w_rest = w[1:]
        # Proximal operator for L1 definition: sign(x) * max(|x| - alpha, 0)
        w_new[1:] = np.sign(w_rest) * np.maximum(np.abs(w_rest) - threshold, 0.0)
        return w_new

    def fit(self, X_train, y_train):
        """
        Fits the logistic regression using FISTA for ALL lambdas in self.lambdas.
        This constructs the full regularization path (warm-started) in a single run.
        """
        n_samples, n_features = X_train.shape
        
        # Extend X with a column of 1s to learn the intercept
        X_ext = np.hstack((np.ones((n_samples, 1)), X_train))
        
        # Compute the Lipschitz constant (L) of the negative log-likelihood gradient.
        # For logistic regression, it's bounded by lambda_max(X^T X) / (4 * N).
        # We compute this accurately using the spectral norm of X_ext.
        # 1/L is the biggest possible step size we can take without risking divergence
        L = (np.linalg.norm(X_ext, ord=2) ** 2) / (4.0 * n_samples)
        step_size = 1.0 / L
        
        # Initialize weights with zeros, +1 for the intercept 
        w = np.zeros(n_features + 1)
        
        # Sorting lambdas descending makes our "warm starts" much more effective 
        # (models with high lambdas drop features out easily and start training the next faster).
        # (so we start with the most regularized model and then decrease lambda, using the previous solution as a warm start for the next one)
        sorted_lambdas = np.sort(self.lambdas)[::-1]
        
        for lmbda in sorted_lambdas:
            y_k = w.copy()
            t_k = 1.0  # parameter for momentum step
            shrinkage_threshold = lmbda * step_size
            
            for i in range(self.max_iter):
                w_prev = w.copy()
                
                # 1. Gradient descent step
                # gradient = X^T(p - y) / N
                logits = X_ext.dot(y_k)
                probas = self._sigmoid(logits)
                grad = X_ext.T.dot(probas - y_train) / n_samples
                
                z = y_k - step_size * grad
                
                # 2. Proximal step
                w = self._soft_threshold(z, shrinkage_threshold)
                
                # 3. Check parameter convergence
                if np.linalg.norm(w - w_prev, ord=2) < self.tol:
                    break
                    
                # 4. Momentum step (FISTA defining characteristic)
                t_next = (1.0 + np.sqrt(1.0 + 4.0 * (t_k ** 2))) / 2.0  # formula is strictly defined by the FISTA algorithm
                y_k = w + ((t_k - 1.0) / t_next) * (w - w_prev)
                t_k = t_next
                
            # Keep trained coefficients mapped to this specific lambda
            self.coefs_path_[lmbda] = w.copy()
            
        return self

    def validate(self, X_valid, y_valid, measure="recall"):
        """
        Calculates the specified measure across all previously fitted lambda 
        models and selects the optimal lambda.
        
        Supported measures: 'recall', 'precision', 'f-measure', 
        'balanced_accuracy', 'roc_auc', 'pr_auc'.
        """
        self.last_measure_ = measure
        self.validation_scores_ = {}
        
        n_samples = X_valid.shape[0]
        X_ext = np.hstack((np.ones((n_samples, 1)), X_valid))
        
        best_score = -np.inf
        best_lmbda = None
        
        for lmbda, w in self.coefs_path_.items():
            logits = X_ext.dot(w)
            probas = self._sigmoid(logits)
            
            # Binary predictions for threshold 0.5
            preds = (probas >= 0.5).astype(int)
            
            if measure == "recall":
                score = recall_score(y_valid, preds, zero_division=0)
            elif measure == "precision":
                score = precision_score(y_valid, preds, zero_division=0)
            elif measure == "f-measure":
                score = f1_score(y_valid, preds, zero_division=0)
            elif measure == "balanced_accuracy":
                score = balanced_accuracy_score(y_valid, preds)
            elif measure == "roc_auc":
                score = roc_auc_score(y_valid, probas)
            elif measure == "pr_auc":
                score = average_precision_score(y_valid, probas)
            else:
                raise ValueError(f"Unknown measure: {measure}")
                
            self.validation_scores_[lmbda] = score
            
            # Maximize score
            if score > best_score:
                best_score = score
                best_lmbda = lmbda
        
        # Save the single best scenario as the primary choice for the algorithm output.
        self.best_lambda_ = best_lmbda
        
        best_w = self.coefs_path_[best_lmbda]
        self.intercept_ = best_w[0]
        self.coef_ = best_w[1:] # feature parameters only
        
        return best_lmbda, best_score

    def predict_proba(self, X_test):
        """
        Predict probability estimates using the validated (optimal) lambda's coefficients.
        """
        if self.coef_ is None:
            raise ValueError("Algorithm hasn't been validated yet. Call validate() to find best lambda before predicting.")
            
        logits = np.dot(X_test, self.coef_) + self.intercept_
        probas = self._sigmoid(logits)
        
        # Scikit-learn outputs an array of shape (n_samples, 2)
        # where the first column is probability of class 0, and second of class 1.
        return np.vstack([1.0 - probas, probas]).T
        
    def predict(self, X_test):
        # Standard helper utilizing 0.5 decision threshold
        probas = self.predict_proba(X_test)[:, 1]
        return (probas >= 0.5).astype(int)

    def plot(self):
        """
        Produces a plot showing how the given evaluation measure changes with the lambda parameter.
        """
        if not self.validation_scores_:
            raise ValueError("Validation scores not found. Run validate() first.")
            
        # Ensure we plot lambdas in increasing order on X-axis
        sorted_items = sorted(self.validation_scores_.items())
        lmbdas_sorted = [x[0] for x in sorted_items]
        scores_sorted = [x[1] for x in sorted_items]
        
        plt.figure(figsize=(8, 5))
        plt.plot(lmbdas_sorted, scores_sorted, marker='o')
        plt.xscale('log')
        plt.xlabel('Lambda (L1 Penalty)')
        
        measure_label = self.last_measure_.replace('_', ' ').title()
        plt.ylabel(f'{measure_label} Score')
        plt.title('Validation Score vs Lambda parameter')
        plt.axvline(self.best_lambda_, color='red', linestyle='--', label=rf'Optimal $\lambda$ = {self.best_lambda_:.4f}')
        plt.legend()
        plt.grid(True)
        # return the figure so it can be saved without displaying
        return plt.gcf()
        
    def plot_coefficients(self):
        """
        Produces a plot showing the coefficient values as a function of the lambda parameter.
        """
        if not self.coefs_path_:
            raise ValueError("Coefficient paths not found. Run fit() first.")
            
        # Get increasing lambdas for proper plotting
        sorted_items = sorted(self.coefs_path_.items())
        lmbdas_sorted = [x[0] for x in sorted_items]
        
        # Strip out the first parameter mapping which contains the bias term, leaving only model weights
        coefs_matrix = np.array([x[1][1:] for x in sorted_items])
        
        plt.figure(figsize=(10, 6))
        plt.plot(lmbdas_sorted, coefs_matrix)
        plt.xscale('log')
        plt.xlabel('Lambda (L1 Penalty)')
        plt.ylabel('Coefficients values (weights)')
        plt.title('Lasso Paths (Coefficients evolution vs Lambda)')
        plt.axvline(self.best_lambda_, color='black', linestyle='--', label=rf'Optimal $\lambda$ = {self.best_lambda_:.4f}')
        # We purposely don't inject a massive legend here since high-dimension sets (like Spambase) 
        # have 57 features (it would obstruct the graph).
        plt.grid(True)
        # return the figure so it can be saved without displaying
        return plt.gcf()
