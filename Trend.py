import jax.numpy as jnp
from jax.scipy.stats import t as student_t
from statsmodels.tsa.stattools import adfuller, kpss
import numpy as np
import matplotlib.pyplot as plt

def analyze_coef_trend_and_stationarity(coef_df, window=50, alpha=0.05, plot=False):
    """
    Analyze each coefficient series for trend (rolling slope), ADF, KPSS.
    
    Args:
        coef_df: pd.DataFrame of shape (n_time, n_series)
        window: rolling window size
        alpha: significance level
        plot: if True, plots each coefficient series with trend line
        
    Returns:
        dict: per-series results, including ADF, KPSS, trend info
    """
    coef_tensor = jnp.array(coef_df.T.values)  # shape (n_series, n_time)
    n_series, n_time = coef_tensor.shape
    n_windows = n_time - window + 1

    # ---- 1. Rolling trend detection with JAX ----
    t_vec = jnp.arange(window)
    X = jnp.stack([jnp.ones(window), t_vec], axis=1)
    XtX_inv = jnp.linalg.inv(X.T @ X)
    Xt = X.T

    def sliding_windows(x, window):
        i = jnp.arange(window)[None, None, :]
        j = jnp.arange(n_windows)[None, :, None]
        return x[:, j + i]

    Y = sliding_windows(coef_tensor, window)  # (n_series, n_windows, window)
    beta = jnp.einsum('ij,jk,nwk->nwi', XtX_inv, Xt, Y)  # (n_series, n_windows, 2)
    slopes = beta[:, :, 1]
    y_hat = jnp.einsum('ij,nwj->nwi', X, beta)
    residuals = Y - y_hat
    dof = window - 2
    sigma_sq = jnp.sum(residuals ** 2, axis=-1) / dof
    se_slope = jnp.sqrt(sigma_sq * XtX_inv[1, 1])
    t_stat = slopes / se_slope
    p_val = 2 * student_t.sf(jnp.abs(t_stat), df=dof)
    is_trending = p_val < alpha

    # ---- 2. ADF + KPSS tests per series (not vectorized) ----
    results = {}
    for i, name in enumerate(coef_df.columns):
        series = coef_df[name].dropna().values

        # ADF
        try:
            adf_stat, adf_p, _, _, _, _ = adfuller(series)
            adf_res = {'stat': adf_stat, 'p_value': adf_p, 'is_stationary': adf_p < alpha}
        except Exception as e:
            adf_res = {'error': str(e)}

        # KPSS
        try:
            kpss_stat, kpss_p, _, _ = kpss(series, regression='c', nlags='auto')
            kpss_res = {'stat': kpss_stat, 'p_value': kpss_p, 'is_stationary': kpss_p > alpha}
        except Exception as e:
            kpss_res = {'error': str(e)}

        # Trend slope & flags from JAX
        result = {
            'adf': adf_res,
            'kpss': kpss_res,
            'rolling_trend': {
                'slope': np.array(slopes[i]),
                't_stat': np.array(t_stat[i]),
                'p_val': np.array(p_val[i]),
                'is_trending': np.array(is_trending[i])
            }
        }

        if plot:
            fig, ax = plt.subplots(figsize=(10, 3))
            t = np.arange(len(series))
            ax.plot(t, series, label=name)
            # OLS trend line
            try:
                coeffs = np.polyfit(t, series, 1)
                trend_line = np.polyval(coeffs, t)
                ax.plot(t, trend_line, linestyle='--', color='red', label='Trend')
                ax.set_title(f"{name} — slope: {coeffs[0]:.4f}")
            except Exception as e:
                ax.set_title(f"{name} (trend error: {str(e)})")
            ax.legend()
            plt.tight_layout()
            plt.show()

        results[name] = result

    return results

results = analyze_coef_trend_and_stationarity(beta_series, window=50, alpha=0.05, plot=True)

# Access results
results['x1']['adf']
results['x1']['kpss']
results['x1']['rolling_trend']['is_trending']


#== kalman filters

import numpy as np
import pandas as pd
import jax.numpy as jnp
from jax.scipy.stats import t as student_t
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter
from statsmodels.tsa.statespace.structural import UnobservedComponents
import matplotlib.pyplot as plt

def analyze_coef_trend_adf_kpss_kalman(coef_df, window=50, alpha=0.05, plot=False):
    coef_tensor = jnp.array(coef_df.T.values)  # shape: (n_series, n_time)
    n_series, n_time = coef_tensor.shape
    n_windows = n_time - window + 1

    # --- JAX rolling slope detection ---
    t_vec = jnp.arange(window)
    X = jnp.stack([jnp.ones(window), t_vec], axis=1)
    XtX_inv = jnp.linalg.inv(X.T @ X)
    Xt = X.T

    def sliding_windows(x, window):
        i = jnp.arange(window)[None, None, :]
        j = jnp.arange(n_windows)[None, :, None]
        return x[:, j + i]

    Y = sliding_windows(coef_tensor, window)  # (n_series, n_windows, window)
    beta = jnp.einsum('ij,jk,nwk->nwi', XtX_inv, Xt, Y)
    slopes = beta[:, :, 1]
    y_hat = jnp.einsum('ij,nwj->nwi', X, beta)
    residuals = Y - y_hat
    dof = window - 2
    sigma_sq = jnp.sum(residuals ** 2, axis=-1) / dof
    se_slope = jnp.sqrt(sigma_sq * XtX_inv[1, 1])
    t_stat = slopes / se_slope
    p_val = 2 * student_t.sf(jnp.abs(t_stat), df=dof)
    is_trending = p_val < alpha

    results = {}
    for i, name in enumerate(coef_df.columns):
        series = coef_df[name].dropna().values
        t = np.arange(len(series))

        # --- ADF ---
        try:
            adf_stat, adf_p, _, _, _, _ = adfuller(series)
            adf_res = {
                'stat': adf_stat,
                'p_value': adf_p,
                'reject_H0': adf_p < alpha,
                'conclusion': 'Stationary' if adf_p < alpha else 'Non-stationary (unit root)'
            }
        except Exception as e:
            adf_res = {'error': str(e)}

        # --- KPSS ---
        try:
            kpss_stat, kpss_p, _, _ = kpss(series, regression='c', nlags='auto')
            kpss_res = {
                'stat': kpss_stat,
                'p_value': kpss_p,
                'reject_H0': kpss_p < alpha,
                'conclusion': 'Non-stationary (trending)' if kpss_p < alpha else 'Stationary'
            }
        except Exception as e:
            kpss_res = {'error': str(e)}

        # --- Kalman filter only if both tests confirm stationarity ---
        kalman_path = None
        if adf_res.get('reject_H0') and not kpss_res.get('reject_H0'):
            try:
                mod = UnobservedComponents(series, level='local level')
                res = mod.fit(disp=False)
                kalman_path = {
                    'filtered': res.filtered_state[0],
                    'smoothed': res.smoothed_state[0],
                    'stderr': res.smoothed_state_cov[0, 0, :].clip(0)**0.5
                }
            except Exception as e:
                kalman_path = {'error': str(e)}

        # --- Optional plot ---
        if plot:
            plt.figure(figsize=(10, 3))
            plt.plot(t, series, label=name)
            if kalman_path and isinstance(kalman_path, dict) and 'smoothed' in kalman_path:
                plt.plot(t, kalman_path['smoothed'], label='Kalman', color='green')
                plt.fill_between(
                    t,
                    kalman_path['smoothed'] - 2 * kalman_path['stderr'],
                    kalman_path['smoothed'] + 2 * kalman_path['stderr'],
                    color='green', alpha=0.2, label='±2σ'
                )
            plt.title(f"{name} — ADF: {adf_res.get('conclusion')}, KPSS: {kpss_res.get('conclusion')}")
            plt.legend()
            plt.tight_layout()
            plt.show()

        # --- Compile result ---
        results[name] = {
            'adf': adf_res,
            'kpss': kpss_res,
            'rolling_trend': {
                'slope': np.array(slopes[i]),
                't_stat': np.array(t_stat[i]),
                'p_val': np.array(p_val[i]),
                'is_trending': np.array(is_trending[i])
            },
            'kalman': kalman_path
        }

    return results

results = analyze_coef_trend_adf_kpss_kalman(beta_series, window=50, alpha=0.05, plot=True)

# Example
results['x1']['adf']['conclusion']
results['x1']['kpss']['conclusion']
results['x1']['kalman']['smoothed']  # If applicable


#=== 
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
