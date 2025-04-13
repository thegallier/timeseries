import numpy as np
import cvxpy as cp

class AsymmetricRidgeRegression:
    def __init__(self, alpha=None):
        """
        Initialize the model. 
        alpha: shrinkage factor in [0,1]. If None, it will be determined via 5-fold CV.
        """
        self.alpha = alpha        # user-specified alpha (None means "auto")
        self.alpha_ = None        # optimal alpha found
        self.coef_ = None         # fitted weights (coefficients)
        self.intercept_ = None    # fitted intercept
        # Saved matrices for inspection:
        self.R_ = None            # original correlation matrix
        self.R_shrunk_ = None     # shrunk correlation matrix
        self.S_ = None            # original covariance matrix
        self.S_shrunk_ = None     # shrunk covariance matrix
        # Cross-validation history:
        self.cv_alphas_ = None    # array of alphas tried
        self.cv_scores_ = None    # corresponding MSE for each alpha
    
    def _compute_shrunk_covariance(self, X_centered, alpha):
        """
        Compute shrunk covariance S' = D * [(1-alpha)R + alpha I] * D for centered data.
        Returns S_shrunk, R, R_shrunk.
        """
        n, p = X_centered.shape
        # Standard deviations
        std = np.sqrt(np.sum(X_centered**2, axis=0) / n)
        std[std < 1e-12] = 1e-12  # guard against zero std
        # Empirical covariance (using population formula 1/n)
        S = (X_centered.T @ X_centered) / n
        # Correlation matrix
        R = S / np.outer(std, std)
        np.fill_diagonal(R, 1.0)
        # Shrinkage: convex combo of R and identity
        R_shrunk = (1 - alpha) * R + alpha * np.eye(p)
        # Reconstruct shrunk covariance
        D = np.diag(std)
        S_shrunk = D.dot(R_shrunk).dot(D)
        # Ensure symmetry and PSD
        S_shrunk = (S_shrunk + S_shrunk.T) / 2.0
        eigvals = np.linalg.eigvalsh(S_shrunk)
        if np.min(eigvals) < -1e-8:
            # Clip tiny negative eigenvalues to 0
            eigvals_clipped = np.clip(eigvals, 0, None)
            eigvecs = np.linalg.eigh(S_shrunk)[1]
            S_shrunk = eigvecs.dot(np.diag(eigvals_clipped)).dot(eigvecs.T)
        return S_shrunk, R, R_shrunk
    
    def fit(self, X, y):
        """
        Fit the model on training data X, y. Determines optimal alpha if not set.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n, p = X.shape
        # Center the data (for intercept handling)
        X_mean = X.mean(axis=0)
        y_mean = y.mean()
        X_centered = X - X_mean
        y_centered = y - y_mean
        
        if self.alpha is not None:
            # Use specified alpha
            alpha_use = float(np.clip(self.alpha, 0.0, 1.0))
            # Compute shrunk covariance on full data
            S_shrunk, R, R_shrunk = self._compute_shrunk_covariance(X_centered, alpha_use)
            # Solve S_shrunk * w = Cov(X,y)
            c = (X_centered.T @ y_centered) / n
            w = None
            # Solve via CVXPY (quadratic program) for numerical stability
            w_var = cp.Variable(p)
            obj = cp.Minimize(0.5 * cp.quad_form(w_var, S_shrunk) - c.T @ w_var)
            cp.Problem(obj).solve(solver=cp.SCS, verbose=False)
            if w_var.value is not None:
                w = w_var.value
            else:
                # Fallback to direct solve if solver fails
                try:
                    w = np.linalg.solve(S_shrunk, c)
                except np.linalg.LinAlgError:
                    w = np.linalg.pinv(S_shrunk).dot(c)
            # Compute intercept and store everything
            b = y_mean - X_mean.dot(w)
            self.alpha_ = alpha_use
            self.coef_ = w
            self.intercept_ = b
            self.R_ = R
            self.R_shrunk_ = R_shrunk
            self.S_ = (X_centered.T @ X_centered) / n
            self.S_shrunk_ = S_shrunk
            self.cv_alphas_ = None
            self.cv_scores_ = None
            return self
        
        # If alpha is None, perform 5-fold cross-validation to find optimal alpha
        k = 5
        indices = np.arange(n)
        np.random.shuffle(indices)
        fold_sizes = [n//k + (1 if i < n % k else 0) for i in range(k)]
        folds = []
        start = 0
        for fs in fold_sizes:
            end = start + fs
            folds.append(indices[start:end])
            start = end
        
        alphas_grid = np.linspace(0, 1, 21)
        cv_scores = []
        for alpha_val in alphas_grid:
            mse_folds = []
            for i in range(k):
                # Split into training and validation folds
                val_idx = folds[i]
                train_idx = np.setdiff1d(indices, val_idx)
                X_train = X[train_idx]; y_train = y[train_idx]
                X_val = X[val_idx];   y_val = y[val_idx]
                # Center training data and apply same shift to validation features
                X_train_mean = X_train.mean(axis=0)
                y_train_mean = y_train.mean()
                X_train_c = X_train - X_train_mean
                y_train_c = y_train - y_train_mean
                X_val_c = X_val - X_train_mean
                # Shrink covariance on training fold
                S_shrunk_train, _, _ = self._compute_shrunk_covariance(X_train_c, alpha_val)
                c_train = (X_train_c.T @ y_train_c) / X_train_c.shape[0]
                # Solve for w on training fold
                w_train = None
                w_var = cp.Variable(p)
                obj = cp.Minimize(0.5 * cp.quad_form(w_var, S_shrunk_train) - c_train.T @ w_var)
                cp.Problem(obj).solve(solver=cp.SCS, verbose=False)
                if w_var.value is not None:
                    w_train = w_var.value
                else:
                    try:
                        w_train = np.linalg.solve(S_shrunk_train, c_train)
                    except np.linalg.LinAlgError:
                        w_train = np.linalg.pinv(S_shrunk_train).dot(c_train)
                b_train = y_train_mean - X_train_mean.dot(w_train)
                # Validation predictions and MSE
                y_pred_val = X_val.dot(w_train) + b_train
                mse_folds.append(np.mean((y_val - y_pred_val)**2))
            cv_scores.append(np.mean(mse_folds))
        cv_scores = np.array(cv_scores)
        best_idx = int(np.argmin(cv_scores))
        alpha_opt = alphas_grid[best_idx]
        
        # Refit on full data with optimal alpha
        S_shrunk, R, R_shrunk = self._compute_shrunk_covariance(X_centered, alpha_opt)
        c_full = (X_centered.T @ y_centered) / n
        w_full = None
        w_var = cp.Variable(p)
        obj = cp.Minimize(0.5 * cp.quad_form(w_var, S_shrunk) - c_full.T @ w_var)
        cp.Problem(obj).solve(solver=cp.SCS, verbose=False)
        if w_var.value is not None:
            w_full = w_var.value
        else:
            try:
                w_full = np.linalg.solve(S_shrunk, c_full)
            except np.linalg.LinAlgError:
                w_full = np.linalg.pinv(S_shrunk).dot(c_full)
        b_full = y_mean - X_mean.dot(w_full)
        # Store final results
        self.alpha_ = alpha_opt
        self.coef_ = w_full
        self.intercept_ = b_full
        self.R_ = R
        self.R_shrunk_ = R_shrunk
        self.S_ = (X_centered.T @ X_centered) / n
        self.S_shrunk_ = S_shrunk
        self.cv_alphas_ = alphas_grid
        self.cv_scores_ = cv_scores
        return self
    
    def predict(self, X):
        """
        Make predictions on new data.
        """
        X = np.asarray(X)
        return X.dot(self.coef_) + self.intercept_
