import numpy as np
import cvxpy as cp
from numpy.linalg import eigvalsh
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import KFold

class AsymmetricRidgeRegression:
    def __init__(self, lambda_pos=1.0, lambda_neg=10.0, alpha=None, cv=5, random_state=None):
        """
        Initialize the model.
        
        Args:
            lambda_pos (float): L2 penalty multiplier for positive weights.
            lambda_neg (float): L2 penalty multiplier for negative weights.
            alpha (float or None): Shrinkage factor in [0,1] for the correlation matrix. 
                                   If None, it will be determined via cross-validation.
            cv (int): Number of folds in cross-validation.
            random_state (int or None): Seed for reproducibility.
        """
        self.lambda_pos = lambda_pos
        self.lambda_neg = lambda_neg
        self.alpha = alpha  # If set, use user-specified alpha; otherwise, choose via CV.
        self.cv = cv
        self.random_state = random_state
        
        # To be determined at fit time:
        self.alpha_ = None            # optimal alpha selected via CV.
        self.coef_ = None             # fitted coefficients (weights)
        self.intercept_ = None        # fitted intercept
        self.X_mean_ = None           # mean of training features
        self.y_mean_ = None           # mean of training target
        
        # Saved matrices for inspection/plotting:
        self.R_ = None                # original correlation matrix (full data)
        self.R_shrunk_ = None         # shrunk correlation matrix (with optimal alpha)
        self.S_ = None                # original covariance matrix (from centered X)
        self.S_shrunk_ = None         # shrunk covariance matrix (with optimal alpha)
        
        # CV history:
        self.cv_alphas_ = None        # alphas grid tested
        self.cv_scores_ = None        # corresponding CV MSE values

    def _compute_shrunk_covariance(self, X_centered, alpha):
        """
        Given centered data, compute the shrunk covariance matrix S' = D * R' * D,
        where R' = (1-alpha)*R + alpha*I and R is the correlation matrix.
        
        Returns:
            S_shrunk (np.ndarray): The shrunk covariance matrix.
            R (np.ndarray): The original correlation matrix.
            R_shrunk (np.ndarray): The shrunk correlation matrix.
        """
        n, p = X_centered.shape
        # Standard deviations (using population scaling: divide by n)
        std = np.sqrt(np.sum(X_centered**2, axis=0) / n)
        # Avoid division by zero:
        std[std < 1e-12] = 1e-12
        
        # Compute sample covariance S (using 1/n)
        S = (X_centered.T @ X_centered) / n
        # Construct the correlation matrix R:
        R = S / np.outer(std, std)
        np.fill_diagonal(R, 1.0)
        
        # Shrink correlation: R' = (1-alpha)*R + alpha*I
        R_shrunk = (1 - alpha) * R + alpha * np.eye(p)
        # Reconstruct shrunk covariance: S' = D * R_shrunk * D
        D = np.diag(std)
        S_shrunk = D @ R_shrunk @ D
        # Make sure S_shrunk is symmetric.
        S_shrunk = (S_shrunk + S_shrunk.T) / 2.0
        
        # Optional: enforce PSD by clipping tiny negative eigenvalues.
        eigvals = eigvalsh(S_shrunk)
        if np.min(eigvals) < -1e-8:
            eigvals_clipped = np.clip(eigvals, 0, None)
            # Reconstruct S_shrunk with the clipped eigenvalues
            eigvecs = np.linalg.eigh(S_shrunk)[1]
            S_shrunk = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
        
        return S_shrunk, R, R_shrunk

    def _solve_qp(self, X, y, S_shrunk):
        """
        Solve the convex optimization problem:
        
           minimize   0.5*||X*w - y||^2 + 0.5*w^T S_shrunk w 
                      + lambda_pos * ||cp.pos(w)||^2 + lambda_neg * ||cp.neg(w)||^2
        
        Args:
            X (np.ndarray): Centered design matrix.
            y (np.ndarray): Centered target vector.
            S_shrunk (np.ndarray): Shrunk covariance matrix.
            
        Returns:
            w (np.ndarray): Coefficient vector.
        """
        p = X.shape[1]
        w_var = cp.Variable(p)
        # Data fidelity term:
        loss = 0.5 * cp.sum_squares(X @ w_var - y)
        # Covariance penalty term:
        penalty_cov = 0.5 * cp.quad_form(w_var, S_shrunk)
        # Asymmetric L2 penalties: penalize positive and negative parts differently.
        penalty_asym = self.lambda_pos * cp.sum_squares(cp.pos(w_var)) + \
                       self.lambda_neg * cp.sum_squares(cp.neg(w_var))
        objective = cp.Minimize(loss + penalty_cov + penalty_asym)
        prob = cp.Problem(objective)
        prob.solve(solver=cp.SCS, verbose=False)
        if w_var.value is None:
            # Fall back to pseudo-inverse if needed
            w = np.zeros(p)
        else:
            w = w_var.value
        return w

    def fit(self, X, y):
        """
        Fit the asymmetric ridge regression model on the data.
        
        If alpha was not provided at initialization, this method will select 
        the best alpha in [0,1] using 5-fold cross-validation (minimizing MSE).
        
        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features).
            y (np.ndarray): Target vector of shape (n_samples,).
            
        Returns:
            self
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n, p = X.shape
        
        # Set random state if provided:
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Center the data (for intercept handling):
        self.X_mean_ = X.mean(axis=0)
        self.y_mean_ = y.mean()
        X_centered = X - self.X_mean_
        y_centered = y - self.y_mean_
        
        if self.alpha is not None:
            # Use user-specified alpha (ensure it lies in [0,1])
            alpha_use = float(np.clip(self.alpha, 0.0, 1.0))
            self.alpha_ = alpha_use
            S_shrunk, R, R_shrunk = self._compute_shrunk_covariance(X_centered, alpha_use)
            self.R_ = R
            self.R_shrunk_ = R_shrunk
            self.S_ = (X_centered.T @ X_centered) / n
            self.S_shrunk_ = S_shrunk
            # Solve for weights using the full dataset.
            w = self._solve_qp(X_centered, y_centered, S_shrunk)
            self.coef_ = w
            self.intercept_ = self.y_mean_ - self.X_mean_.dot(w)
            self.cv_alphas_ = np.array([alpha_use])
            self.cv_scores_ = np.array([np.mean((y_centered - X_centered.dot(w))**2)])
            return self
        
        # Otherwise, select alpha via cross-validation.
        alphas_grid = np.linspace(0, 1, 21)  # e.g., 0.0, 0.05, …, 1.0
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        cv_scores = []
        
        # For each candidate alpha, perform CV:
        for alpha_val in alphas_grid:
            mse_folds = []
            for train_idx, val_idx in kf.split(X_centered):
                X_train, X_val = X_centered[train_idx], X_centered[val_idx]
                y_train, y_val = y_centered[train_idx], y_centered[val_idx]
                # Compute shrunk covariance for training data:
                S_shrunk_train, _, _ = self._compute_shrunk_covariance(X_train, alpha_val)
                # Solve for weights on training fold:
                w_train = self._solve_qp(X_train, y_train, S_shrunk_train)
                # Compute predictions on validation fold:
                y_pred = X_val.dot(w_train) + (y_train.mean() - X_train.mean(axis=0).dot(w_train))
                mse = np.mean((y_val - y_pred)**2)
                mse_folds.append(mse)
            cv_scores.append(np.mean(mse_folds))
        cv_scores = np.array(cv_scores)
        best_idx = int(np.argmin(cv_scores))
        alpha_opt = alphas_grid[best_idx]
        self.cv_alphas_ = alphas_grid
        self.cv_scores_ = cv_scores
        
        # Refit on the full data with the optimal alpha:
        self.alpha_ = alpha_opt
        S_shrunk, R, R_shrunk = self._compute_shrunk_covariance(X_centered, alpha_opt)
        self.R_ = R
        self.R_shrunk_ = R_shrunk
        self.S_ = (X_centered.T @ X_centered) / n
        self.S_shrunk_ = S_shrunk
        
        w = self._solve_qp(X_centered, y_centered, S_shrunk)
        self.coef_ = w
        self.intercept_ = self.y_mean_ - self.X_mean_.dot(w)
        return self
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X (np.ndarray): New data of shape (n_samples, n_features).
            
        Returns:
            y_pred (np.ndarray): Predicted targets.
        """
        X = np.asarray(X)
        return X.dot(self.coef_) + self.intercept_
    
    def plot_cv_results(self):
        """
        Plot CV results using Plotly:
         - Plot of CV MSE vs. alpha.
         - Heatmaps of the original correlation matrix R and the shrunk correlation matrix R_shrunk.
        """
        if self.cv_alphas_ is None or self.cv_scores_ is None:
            print("No CV history available. Fit with alpha=None for cross-validation.")
            return
        
        # Plot CV MSE vs. alpha:
        fig1 = px.line(x=self.cv_alphas_, y=self.cv_scores_,
                       labels={'x': "Alpha", 'y': "CV MSE"},
                       title="5-Fold CV MSE vs. Shrinkage Factor (alpha)")
        fig1.add_scatter(x=[self.alpha_], y=[self.cv_scores_[np.argmin(self.cv_scores_)]],
                         mode='markers', marker=dict(size=12, color='red'),
                         name="Selected alpha")
        fig1.show()
        
        # Plot original vs shrunk correlation matrices as heatmaps.
        # (We assume self.R_ and self.R_shrunk_ are computed from the full dataset.)
        fig2 = go.Figure()
        fig2.add_trace(go.Heatmap(z=self.R_,
                                  x=[f"F{i}" for i in range(1, self.R_.shape[0]+1)],
                                  y=[f"F{i}" for i in range(1, self.R_.shape[0]+1)],
                                  colorscale='Viridis',
                                  zmin=-1, zmax=1))
        fig2.update_layout(title="Original Correlation Matrix", xaxis_title="Features", yaxis_title="Features")
        fig2.show()
        
        fig3 = go.Figure()
        fig3.add_trace(go.Heatmap(z=self.R_shrunk_,
                                  x=[f"F{i}" for i in range(1, self.R_shrunk_.shape[0]+1)],
                                  y=[f"F{i}" for i in range(1, self.R_shrunk_.shape[0]+1)],
                                  colorscale='Viridis',
                                  zmin=-1, zmax=1))
        fig3.update_layout(title=f"Shrunk Correlation Matrix (alpha={self.alpha_:.2f})", 
                           xaxis_title="Features", yaxis_title="Features")
        fig3.show()
/--
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
