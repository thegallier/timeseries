import jax
import jax.numpy as jnp
from jax.scipy.stats import t as student_t

def batched_rolling_trend_jax(coef_tensor: jnp.ndarray, window: int = 50, alpha: float = 0.05):
    """
    Detects local linear trends in a tensor of coefficient series using JAX.
    
    Args:
        coef_tensor: jnp.ndarray of shape (n_series, n_time)
        window: sliding window length
        alpha: significance level for trend detection
    
    Returns:
        dict with keys: 'slope', 't_stat', 'p_val', 'is_trending'
        Each value is a jnp.ndarray of shape (n_series, n_windows)
    """
    n_series, n_time = coef_tensor.shape
    n_windows = n_time - window + 1

    # Prepare fixed design matrix: X = [1, t]
    t_vec = jnp.arange(window)
    X = jnp.stack([jnp.ones(window), t_vec], axis=1)             # shape: (window, 2)
    XtX_inv = jnp.linalg.inv(X.T @ X)                            # shape: (2, 2)
    Xt = X.T                                                     # shape: (2, window)
    
    # Get strided rolling windows (n_series, n_windows, window)
    def sliding_windows(x, window):
        i = jnp.arange(window)[None, None, :]
        j = jnp.arange(n_windows)[None, :, None]
        return x[:, j + i]                                       # shape: (n_series, n_windows, window)

    Y = sliding_windows(coef_tensor, window)                     # shape: (n_series, n_windows, window)

    # Compute beta = (X'X)^(-1) X'Y for all windows and series
    beta = jnp.einsum('ij,jk,nwk->nwi', XtX_inv, Xt, Y)          # shape: (n_series, n_windows, 2)
    slopes = beta[:, :, 1]                                       # (n_series, n_windows)

    # Predicted values and residuals
    Y_hat = jnp.einsum('ij,nwj->nwi', X, beta)                   # (n_series, n_windows, window)
    residuals = Y - Y_hat                                        # (n_series, n_windows, window)
    
    dof = window - 2
    sigma_sq = jnp.sum(residuals ** 2, axis=-1) / dof            # (n_series, n_windows)
    se_slope = jnp.sqrt(sigma_sq * XtX_inv[1, 1])                # scalar multiplier

    t_stat = slopes / se_slope                                   # (n_series, n_windows)
    p_val = 2 * student_t.sf(jnp.abs(t_stat), df=dof)            # (n_series, n_windows)
    is_trending = p_val < alpha                                  # (n_series, n_windows)

    return {
        'slope': slopes,
        't_stat': t_stat,
        'p_val': p_val,
        'is_trending': is_trending
    }

/// old

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

def detect_trend_segments(coef_series, window=50, alpha=0.05):
    trend_slopes = []
    trend_pvals = []
    indices = []

    y = coef_series.dropna().values
    n = len(y)

    for start in range(n - window + 1):
        end = start + window
        sub_y = y[start:end]
        t = np.arange(window)
        X = sm.add_constant(t)
        model = sm.OLS(sub_y, X).fit()
        slope = model.params[1]
        pval = model.pvalues[1]
        trend_slopes.append(slope)
        trend_pvals.append(pval)
        indices.append(start + window // 2)  # center point of window

    # Create results DataFrame
    trend_df = pd.DataFrame({
        'center_index': indices,
        'slope': trend_slopes,
        'pval': trend_pvals
    })
    trend_df['is_trending'] = trend_df['pval'] < alpha

    return trend_df

# Example usage
trend_df = detect_trend_segments(beta_series['x1'], window=50)

# Find last trend
latest_trend = trend_df[trend_df['is_trending']].tail(1)
if not latest_trend.empty:
    start_idx = latest_trend['center_index'].values[0]
    print(f"Latest trend started around index {start_idx}")
else:
    print("No significant trend detected in any window.")

# Plot for inspection
plt.figure(figsize=(12,6))
plt.plot(beta_series['x1'].reset_index(drop=True), label="Coefficient")
plt.scatter(trend_df['center_index'], trend_df['slope'],
            c=trend_df['is_trending'].map({True: 'red', False: 'gray'}),
            label="Trend slope", marker='x')
plt.axvline(start_idx, color='red', linestyle='--', label="Latest trend start")
plt.legend()
plt.title("Rolling Coefficient and Detected Trends")
plt.show()
