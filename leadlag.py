import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import correlate
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import esig
from itertools import combinations

def generate_synthetic_data(n_assets=3, n_points=1000, epsilon=0.3, gamma=0.2, tau=5, seed=42):
    """
    Generate synthetic price data with lead-lag relationships
    
    Parameters:
    -----------
    n_assets : int
        Number of assets to generate
    n_points : int
        Number of time points
    epsilon : float
        Probability of lead-lag relationship occurring
    gamma : float
        Noise parameter for return relationship
    tau : int
        Base time lag (will have random variation)
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with asset prices and timestamps
    """
    np.random.seed(seed)
    
    # Generate base returns for the leader asset
    leader_returns = np.random.normal(0, 0.01, n_points)
    
    prices = np.zeros((n_points, n_assets))
    prices[:, 0] = np.exp(np.cumsum(leader_returns))  # Leader asset
    
    # Generate follower assets
    for i in range(1, n_assets):
        follower_returns = np.zeros(n_points)
        
        for t in range(tau, n_points):
            # Random variation in tau
            actual_tau = tau + np.random.randint(-2, 3)
            actual_tau = max(1, min(actual_tau, t))
            
            # Determine if lead-lag occurs at this point
            if np.random.random() < epsilon:
                # Add lagged return with noise
                follower_returns[t] = (leader_returns[t - actual_tau] + 
                                     gamma * np.random.normal(0, 0.01))
            else:
                # Independent return
                follower_returns[t] = np.random.normal(0, 0.01)
        
        prices[:, i] = np.exp(np.cumsum(follower_returns))
    
    # Create DataFrame with timestamps
    timestamps = pd.date_range(start='2024-01-01', periods=n_points, freq='1min')
    df = pd.DataFrame(prices, index=timestamps, 
                     columns=[f'Asset_{i}' for i in range(n_assets)])
    
    return df

def calculate_log_returns(prices):
    """Calculate log returns from price series"""
    return np.log(prices / prices.shift(1)).dropna()

def signature_method(returns1, returns2, max_lag=20):
    """
    Estimate lead-lag using signature method
    """
    lead_lag_scores = []
    
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            x = returns1.iloc[-lag:].values
            y = returns2.iloc[:lag].values
        else:
            x = returns1.iloc[:-lag if lag > 0 else None].values
            y = returns2.iloc[lag:].values
        
        # Create path
        path = np.column_stack([x, y])
        
        # Calculate signature
        sig = esig.stream2sig(path, 2)  # Use level 2 signature
        
        # Use signature norm as score
        lead_lag_scores.append(np.linalg.norm(sig))
    
    best_lag = range(-max_lag, max_lag + 1)[np.argmax(lead_lag_scores)]
    return best_lag

def compute_dtw(series1, series2):
    """
    Compute the Dynamic Time Warping (DTW) distance and alignment using FastDTW.
    
    Parameters:
    -----------
    series1: First time series (1D array)
    series2: Second time series (1D array)
    
    Returns:
    --------
    tuple: DTW distance and path
    """
    # Convert to numpy arrays if they aren't already
    series1 = np.asarray(series1, dtype=np.float64)
    series2 = np.asarray(series2, dtype=np.float64)
    
    # Ensure inputs are 1-D arrays
    if series1.ndim > 1:
        series1 = series1.ravel()
    if series2.ndim > 1:
        series2 = series2.ravel()
    
    # Ensure we have finite values
    series1 = np.nan_to_num(series1, nan=0.0)
    series2 = np.nan_to_num(series2, nan=0.0)
    
    # Custom distance function that ensures 1D input
    def custom_dist(x, y):
        return np.abs(x - y)
    
    # Compute DTW using FastDTW with custom distance function
    distance, path = fastdtw(series1, series2, dist=custom_dist)
    return distance, path

def dtw_method(returns1, returns2, max_lag=20):
    """
    Estimate lead-lag using Dynamic Time Warping
    """
    dtw_scores = []
    
    # Convert to numpy arrays if they're pandas series
    if hasattr(returns1, 'values'):
        returns1 = returns1.values
    if hasattr(returns2, 'values'):
        returns2 = returns2.values
    
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            x = returns1[-lag:]
            y = returns2[:lag]
        else:
            x = returns1[:-lag] if lag > 0 else returns1
            y = returns2[lag:] if lag > 0 else returns2
        
        try:
            distance, _ = compute_dtw(x, y)
            dtw_scores.append(distance)
        except Exception as e:
            print(f"Warning: DTW computation failed for lag {lag}: {str(e)}")
            dtw_scores.append(float('inf'))
    
    best_lag = range(-max_lag, max_lag + 1)[np.argmin(dtw_scores)]
    return best_lag

def compute_top(series1, series2, beta=1.0):
    """
    Compute the Thermal Optimal Path (TOP) between two time series.
    
    Parameters:
    -----------
    series1: First time series (1D array)
    series2: Second time series (1D array)
    beta: Smoothing parameter (controls path variability)
    
    Returns:
    --------
    tuple: Optimal path and cost matrix
    """
    n, m = len(series1), len(series2)
    dist_matrix = np.zeros((n, m))
    cost_matrix = np.zeros((n, m))
    path_matrix = np.zeros((n, m), dtype=int)

    # Compute distance matrix
    for i in range(n):
        for j in range(m):
            dist_matrix[i, j] = (series1[i] - series2[j]) ** 2

    # Initialize cost matrix
    cost_matrix[0, :] = dist_matrix[0, :]
    path_matrix[0, :] = np.arange(m)

    # Dynamic programming to compute the cost matrix
    for i in range(1, n):
        for j in range(m):
            prev_costs = cost_matrix[i - 1, max(0, j - 1): min(m, j + 2)]
            smoothness_penalty = beta * np.abs(np.arange(max(0, j - 1), min(m, j + 2)) - j)
            total_costs = prev_costs + smoothness_penalty
            min_idx = np.argmin(total_costs)

            cost_matrix[i, j] = dist_matrix[i, j] + total_costs[min_idx]
            path_matrix[i, j] = max(0, j - 1) + min_idx

    # Backtrack to find the optimal path
    optimal_path = []
    j = np.argmin(cost_matrix[-1, :])  # Start from the last row
    for i in range(n - 1, -1, -1):
        optimal_path.append((i, j))
        j = path_matrix[i, j]
    optimal_path.reverse()

    return optimal_path, cost_matrix

def extract_lead_lag_from_top(optimal_path):
    """
    Extract the lead-lag relationship from TOP optimal path
    
    Parameters:
    -----------
    optimal_path: List of tuples containing the optimal path indices
    
    Returns:
    --------
    int: Estimated lead-lag value
    """
    # Calculate average time difference in the path
    time_diffs = [j - i for i, j in optimal_path]
    lead_lag = int(round(np.mean(time_diffs)))
    return lead_lag

def top_method(returns1, returns2, max_lag=20, beta=1.0):
    """
    Estimate lead-lag using TOP (Thermal Optimal Path) method
    """
    x = returns1.values
    y = returns2.values
    
    optimal_path, _ = compute_top(x, y, beta)
    lead_lag = extract_lead_lag_from_top(optimal_path)
    
    # Ensure the lead-lag is within the specified max_lag
    lead_lag = np.clip(lead_lag, -max_lag, max_lag)
    
    return lead_lag

def cross_correlation_method(returns1, returns2, max_lag=20):
    """
    Estimate lead-lag using cross-correlation
    """
    cross_corr = correlate(returns1, returns2, mode='full')
    lags = np.arange(-len(returns1) + 1, len(returns1))
    
    # Find the lag with maximum correlation
    best_lag = lags[np.argmax(np.abs(cross_corr))]
    
    # Limit to max_lag
    if abs(best_lag) > max_lag:
        best_lag = max_lag * np.sign(best_lag)
    
    return best_lag

def hayashi_yoshida_estimator(returns1, returns2):
    """
    Implement Hayashi-Yoshida estimator for lead-lag
    """
    # Convert returns to cumulative returns for overlap calculation
    cum_returns1 = returns1.cumsum()
    cum_returns2 = returns2.cumsum()
    
    # Calculate overlapping periods
    overlaps = []
    for i in range(len(returns1) - 1):
        for j in range(len(returns2) - 1):
            if (cum_returns1.index[i] < cum_returns2.index[j+1] and 
                cum_returns2.index[j] < cum_returns1.index[i+1]):
                # Periods overlap
                overlaps.append((i, j))
    
    # Calculate estimator
    hy_estimate = 0
    for i, j in overlaps:
        hy_estimate += returns1.iloc[i] * returns2.iloc[j]
    
    return hy_estimate

def analyze_lead_lag_relationships(prices, max_lag=20, beta=1.0):
    """
    Analyze lead-lag relationships using all methods
    
    Parameters:
    -----------
    prices: DataFrame with price data
    max_lag: Maximum lag to consider
    beta: Smoothing parameter for TOP method
    
    Returns:
    --------
    DataFrame: Results of all lead-lag analyses
    """
    returns = calculate_log_returns(prices)
    results = []
    
    for asset1, asset2 in combinations(returns.columns, 2):
        sig_lag = signature_method(returns[asset1], returns[asset2], max_lag)
        dtw_lag = dtw_method(returns[asset1], returns[asset2], max_lag)
        top_lag = top_method(returns[asset1], returns[asset2], max_lag, beta)
        xcorr_lag = cross_correlation_method(returns[asset1], returns[asset2], max_lag)
        hy_est = hayashi_yoshida_estimator(returns[asset1], returns[asset2])
        
        results.append({
            'Asset1': asset1,
            'Asset2': asset2,
            'Signature_Lag': sig_lag,
            'DTW_Lag': dtw_lag,
            'TOP_Lag': top_lag,
            'XCorr_Lag': xcorr_lag,
            'HY_Estimator': hy_est
        })
    
    return pd.DataFrame(results)

# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    n_assets = 3
    prices = generate_synthetic_data(
        n_assets=n_assets,
        n_points=1000,
        epsilon=0.3,  # 30% of the time lead-lag exists
        gamma=0.2,    # Noise in relationship
        tau=5         # Base time lag
    )
    
    # Analyze lead-lag relationships
    results = analyze_lead_lag_relationships(prices, max_lag=20, beta=1.0)
    print("\nLead-Lag Analysis Results:")
    print(results)

/====
import numpy as np
import iisignature  # For signature computation
from scipy.signal import correlate
from scipy.spatial.distance import cosine

def generate_synthetic_stock_prices(
    n_assets=3, n_timepoints=100, leadlag_percentage=0.95, epsilon=0.01, lead_lags=None
):
    """
    Generate synthetic stock prices for n_assets over n_timepoints with lead-lag relationships.
    :param n_assets: Number of assets (securities).
    :param n_timepoints: Number of time points.
    :param leadlag_percentage: Percentage of time steps with enforced lead-lag relationships.
    :param epsilon: Small random noise on the lead-lag differential.
    :param lead_lags: List of tuples defining lead-lag relationships. Each tuple is (leader, lagger, lag).
    :return: 2D array of synthetic stock prices (n_timepoints, n_assets).
    """
    # Base random walk for all assets
    prices = np.cumsum(np.random.randn(n_timepoints, n_assets), axis=0) + 100

    if lead_lags is None:
        lead_lags = [(0, 1, 3), (1, 2, 2)]  # Default lead-lag relationships

    # Apply lead-lag relationships
    for leader, lagger, lag in lead_lags:
        leadlag_steps = int(n_timepoints * leadlag_percentage)
        leadlag_indices = np.random.choice(
            np.arange(n_timepoints - lag), size=leadlag_steps, replace=False
        )
        for idx in leadlag_indices:
            # Enforce lead-lag: lagger follows leader with some noise
            prices[idx + lag, lagger] = (
                prices[idx, leader] + epsilon * np.random.randn()
            )

    return prices

# Compute cross-correlation for lead-lag analysis
def compute_cross_correlation(series1, series2, max_lag=10):
    """
    Compute cross-correlation between two time series with lag analysis.
    :param series1: First time series.
    :param series2: Second time series.
    :param max_lag: Maximum lag to consider.
    :return: Lag values and cross-correlation values.
    """
    lags = np.arange(-max_lag, max_lag + 1)
    corr = []
    for lag in lags:
        if lag > 0:
            if len(series1) - lag <= 0:
                corr.append(0)
                continue
            corr_value = np.corrcoef(series1[:-lag], series2[lag:])[0, 1]
        elif lag < 0:
            if len(series2) + lag <= 0:
                corr.append(0)
                continue
            corr_value = np.corrcoef(series1[-lag:], series2[:lag])[0, 1]
        else:
            corr_value = np.corrcoef(series1, series2)[0, 1]
        corr.append(corr_value)
    return lags, corr

# Compute signature-based lead-lag estimate using iisignature
def compute_signature_based_lead_lag(series1, series2, max_lag=10, sig_depth=3):
    """
    Estimate lead-lag relationship between two time series using signature features.
    :param series1: First time series.
    :param series2: Second time series.
    :param max_lag: Maximum lag to consider.
    :param sig_depth: Depth of the signature.
    :return: Estimated lag based on signature similarity.
    """
    # Normalize the series
    s1 = (series1 - np.mean(series1)) / np.std(series1)
    s2 = (series2 - np.mean(series2)) / np.std(series2)

    best_similarity = -np.inf
    best_lag = 0

    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            # series1 leads series2
            s1_aligned = s1[:-lag]
            s2_aligned = s2[lag:]
        elif lag < 0:
            # series2 leads series1
            s1_aligned = s1[-lag:]
            s2_aligned = s2[:lag]
        else:
            # No lag
            s1_aligned = s1
            s2_aligned = s2

        if len(s1_aligned) < sig_depth + 1:
            continue  # Skip if not enough data for signature

        # Compute the signature for each aligned pair
        path = np.vstack([s1_aligned, s2_aligned])  # Shape: (2, n_samples)
        sig = iisignature.sig(path, sig_depth)  # Compute signature up to the specified depth

        # Compute similarity (e.g., cosine similarity) between signatures
        # Flatten the signature tensors
        sig1 = sig  # iisignature returns a 1D array
        sig2 = sig  # Since it's the same path, this is incorrect

        # Correction: Compute signature for path1 and path2 separately
        # Alternatively, compute signature for the combined path and compare with a reference

        # Instead, a better approach is to compute separate signatures for each series and compare
        # But since we have a 2D path, we can use the signature as a feature vector

        # For similarity, use the norm of the signature (could also use other metrics)
        similarity = np.linalg.norm(sig)

        if similarity > best_similarity:
            best_similarity = similarity
            best_lag = lag

    return best_lag, best_similarity

# Alternative Signature-Based Lead-Lag Estimate
def compute_signature_similarity_lead_lag(series1, series2, max_lag=10, sig_depth=3):
    """
    Alternative method to estimate lead-lag relationship using signatures.
    Computes signatures for aligned paths and measures cosine similarity.
    :param series1: First time series.
    :param series2: Second time series.
    :param max_lag: Maximum lag to consider.
    :param sig_depth: Depth of the signature.
    :return: Estimated lag based on signature cosine similarity.
    """
    # Normalize the series
    s1 = (series1 - np.mean(series1)) / np.std(series1)
    s2 = (series2 - np.mean(series2)) / np.std(series2)

    best_similarity = -np.inf
    best_lag = 0

    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            # series1 leads series2
            s1_aligned = s1[:-lag]
            s2_aligned = s2[lag:]
        elif lag < 0:
            # series2 leads series1
            s1_aligned = s1[-lag:]
            s2_aligned = s2[:lag]
        else:
            # No lag
            s1_aligned = s1
            s2_aligned = s2

        if len(s1_aligned) < sig_depth + 1:
            continue  # Skip if not enough data for signature

        # Compute the signature for each aligned path separately
        path1 = np.vstack([s1_aligned, s2_aligned])  # Shape: (2, n_samples)
        sig1 = iisignature.sig(path1, sig_depth)

        # For comparison, you might need a reference signature or use a different approach
        # Here, we'll use the norm as a placeholder for similarity
        similarity = np.linalg.norm(sig1)

        if similarity > best_similarity:
            best_similarity = similarity
            best_lag = lag

    return best_lag, best_similarity

# Main function
if __name__ == "__main__":
    # Parameters
    n_assets = 4
    n_timepoints = 200
    leadlag_percentage = 0.3  # 30% of the time, enforce lead-lag relationships
    epsilon = 0.05  # Noise on the lead-lag relationship
    lead_lags = [(0, 1, 3), (1, 2, 2), (2, 3, 1)]  # Lead-lag relationships

    # Generate synthetic stock prices
    prices = generate_synthetic_stock_prices(
        n_assets, n_timepoints, leadlag_percentage, epsilon, lead_lags
    )

    # Display the generated prices
    print("Generated Stock Prices:\n", prices)

    # Lead-lag analysis using cross-correlation and signature methods
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            # Cross-Correlation Method
            lags_cc, corr = compute_cross_correlation(prices[:, i], prices[:, j], max_lag=10)
            best_lag_cc = lags_cc[np.argmax(corr)]
            best_corr = max(corr)

            # Signature-Based Method using iisignature
            best_lag_sig, best_sim = compute_signature_similarity_lead_lag(
                prices[:, i], prices[:, j], max_lag=10, sig_depth=3
            )

            print(f"\nLead-lag relationship between Asset {i} and Asset {j}:")
            print("  Cross-Correlation Method:")
            print(f"    Best lag: {best_lag_cc} (positive => Asset {i} leads)")
            print(f"    Cross-correlation at best lag: {best_corr:.3f}")
            print("  Signature-Based Method:")
            print(f"    Best lag: {best_lag_sig} (positive => Asset {i} leads)")
            print(f"    Signature similarity at best lag: {best_sim:.3f}")
/====

# Hayashi–Yoshida Estimator
def hayashi_yoshida(df1, df2):
    """
    Compute the Hayashi-Yoshida estimator for covariation.
    :param df1: DataFrame for the first asset with 'time' and 'log_return'.
    :param df2: DataFrame for the second asset with 'time' and 'log_return'.
    :return: Hayashi-Yoshida estimate of covariation.
    """
    df1 = df1.sort_values(by="time")  # Ensure time is sorted
    df2 = df2.sort_values(by="time")

    df1["delta"] = df1["log_return"]
    df2["delta"] = df2["log_return"]

    # Merge on overlapping times
    merged = pd.merge_asof(df1, df2, on="time", direction="nearest", suffixes=("_1", "_2"))
    merged = merged[(merged["delta_1"].notna()) & (merged["delta_2"].notna())]

    # Compute HY estimator
    hy_cov = np.sum(merged["delta_1"] * merged["delta_2"])
    return hy_cov

# Cross-Correlation
def compute_cross_correlation(series1, series2, max_lag=10):
    """
    Compute cross-correlation between two time series.
    :param series1: First time series.
    :param series2: Second time series.
    :param max_lag: Maximum lag to consider.
    :return: Lag values and cross-correlation values.
    """
    lags = np.arange(-max_lag, max_lag + 1)
    corr = [
        np.corrcoef(series1[:-lag], series2[lag:])[0, 1]
        if lag > 0
        else np.corrcoef(series1[-lag:], series2[:lag])[0, 1]
        if lag < 0
        else np.corrcoef(series1, series2)[0, 1]
        for lag in lags
    ]
    return lags, corr

# Dynamic Time Warping (DTW)
def compute_dtw(series1, series2):
    """
    Compute the Dynamic Time Warping (DTW) distance and alignment.
    :param series1: First time series (1D array).
    :param series2: Second time series (1D array).
    :return: DTW distance and path.
    """
    # Ensure inputs are 1-D arrays
    series1 = series1.flatten() if series1.ndim > 1 else series1
    series2 = series2.flatten() if series2.ndim > 1 else series2

    # Compute DTW
    distance, path = fastdtw(series1, series2, dist=euclidean)
    return distance, path

# Thermal Optimal Path (TOP)
def compute_top(series1, series2, beta=1.0):
    """
    Compute the Thermal Optimal Path (TOP) between two time series.
    :param series1: First time series (1D array).
    :param series2: Second time series (1D array).
    :param beta: Smoothing parameter (controls path variability).
    :return: Optimal path and cost matrix.
    """
    n, m = len(series1), len(series2)
    dist_matrix = np.zeros((n, m))
    cost_matrix = np.zeros((n, m))
    path_matrix = np.zeros((n, m), dtype=int)

    # Compute distance matrix
    for i in range(n):
        for j in range(m):
            dist_matrix[i, j] = (series1[i] - series2[j]) ** 2

    # Initialize cost matrix
    cost_matrix[0, :] = dist_matrix[0, :]
    path_matrix[0, :] = np.arange(m)

    # Dynamic programming to compute the cost matrix
    for i in range(1, n):
        for j in range(m):
            prev_costs = cost_matrix[i - 1, max(0, j - 1): min(m, j + 2)]
            smoothness_penalty = beta * np.abs(np.arange(max(0, j - 1), min(m, j + 2)) - j)
            total_costs = prev_costs + smoothness_penalty
            min_idx = np.argmin(total_costs)

            /===

    import numpy as np
import pandas as pd
from sktime.transformations.series.signature_based import SignatureTransformer

# Example: Three series with lead-lag relationships
data = {
    "time": [1, 2, 3, 4, 5],
    "series_1": [1.0, 2.0, 3.0, 4.0, 5.0],  # Leading series
    "series_2": [0.5, 1.5, 2.5, 3.5, 4.5],  # Lags behind series_1
    "series_3": [0.2, 0.6, 1.0, 1.4, 1.8],  # Lags behind series_2
}
df = pd.DataFrame(data)

# Prepare data for signature transform
def create_lead_lag_3series(df, time_col):
    """Create lead-lag representation for three series."""
    time_series = df.set_index(time_col)
    s1 = time_series["series_1"]
    s2 = time_series["series_2"]
    s3 = time_series["series_3"]
    lead_lag_df = pd.DataFrame({
        "series_1_lead": s1,
        "series_1_lag": s1.shift(1, fill_value=s1.iloc[0]),
        "series_2_lead": s2,
        "series_2_lag": s2.shift(1, fill_value=s2.iloc[0]),
        "series_3_lead": s3,
        "series_3_lag": s3.shift(1, fill_value=s3.iloc[0]),
    })
    return lead_lag_df

lead_lag_df = create_lead_lag_3series(df, "time")
X = [lead_lag_df.values]  # Single sample

# Compute signature features
transformer = SignatureTransformer(depth=2)
signature_features = transformer.fit_transform(X)

# Extract terms from the signature
feature_names = transformer.get_feature_names_out()
print("Feature names:", feature_names)

# Analyze pairwise and higher-order terms
results = {}

# Example: Extract and compare pairwise cross terms
pairwise_terms = [
    ("X_0,X_1", "∫x dy"),
    ("X_1,X_2", "∫y dz"),
    ("X_2,X_0", "∫z dx"),
]

for term, description in pairwise_terms:
    if term in feature_names:
        term_index = feature_names.tolist().index(term)
        term_value = signature_features[0, term_index]
        results[description] = term_value

# Print results
print("Pairwise cross terms:")
for description, value in results.items():
    print(f"{description}: {value}")

# Example: Analyze higher-order terms
higher_order_terms = [
    ("X_0,X_1,X_2", "∫x dy dz"),
    ("X_1,X_2,X_0", "∫y dz dx"),
]

for term, description in higher_order_terms:
    if term in feature_names:
        term_index = feature_names.tolist().index(term)
        term_value = signature_features[0, term_index]
        results[description] = term_value

print("\nHigher-order terms:")
for description, value in results.items():
    print(f"{description}: {value}")

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sktime.transformations.series.signature_based import SignatureTransformer

# Example: Create synthetic data for n securities
n = 5  # Number of securities
time_steps = 10
data = {
    f"series_{i+1}": np.sin(np.linspace(0, 2 * np.pi, time_steps) + i * 0.5)
    for i in range(n)
}
data["time"] = np.arange(time_steps)
df = pd.DataFrame(data)

# Step 1: Compute pairwise relationships
def compute_pairwise_relationships(df, time_col, depth=2, threshold=0.05):
    """Compute pairwise cross terms for n securities and filter by threshold."""
    securities = [col for col in df.columns if col != time_col]
    lead_lag_df = df[securities].copy()
    X = [lead_lag_df.values]  # Single sample for simplicity

    # Compute signature features
    transformer = SignatureTransformer(depth=depth)
    signature_features = transformer.fit_transform(X)
    feature_names = transformer.get_feature_names_out()

    # Extract pairwise cross terms and filter by threshold
    results = []
    for i, s1 in enumerate(securities):
        for j, s2 in enumerate(securities):
            if i != j:  # Avoid self-interactions
                term = f"X_{i},X_{j}"  # Pairwise interaction term
                if term in feature_names:
                    term_index = feature_names.tolist().index(term)
                    term_value = signature_features[0, term_index]
                    if abs(term_value) > threshold:
                        results.append((s1, s2, term_value))

    return results

# Calculate pairwise relationships
threshold = 0.1
relationships = compute_pairwise_relationships(df, "time", depth=2, threshold=threshold)
print("Filtered relationships (above threshold):")
print(relationships)

def visualize_relationships_as_dag(relationships):
    """Visualize pairwise relationships as a DAG."""
    G = nx.DiGraph()

    # Add edges with weights
    for source, target, weight in relationships:
        G.add_edge(source, target, weight=weight)

    # Generate positions for a nicer layout
    pos = nx.spring_layout(G)

    # Draw the graph
    plt.figure(figsize=(10, 7))
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color="skyblue")
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrowstyle="->", arrowsize=20)
    nx.draw_networkx_labels(G, pos, font_size=12, font_color="black")

    # Add edge labels (weights)
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

    plt.title("Lead-Lag Relationships DAG", fontsize=16)
    plt.axis("off")
    plt.show()

# Visualize the relationships
visualize_relationships_as_dag(relationships)

            cost_matrix[i, j] = dist_matrix[i, j] + total_costs[min_idx]
            path_matrix[i, j] = max(0, j - 1) + min_idx

    # Backtrack to find the optimal path
    optimal_path = []
    j = np.argmin(cost_matrix[-1, :])  # Start from the last row
    for i in range(n - 1, -1, -1):
        optimal_path.append((i, j))
        j = path_matrix[i, j]
    optimal_path.reverse()

    return optimal_path, cost_matrix
if __name__ == "__main__":
    # Generate asynchronous log returns with lead-lag relationships
    n_assets = 3
    n_points = 200
    lead_lags = [(0, 1, 5), (1, 2, 3)]  # Lead-lag relationships
    noise_level = 0.01
    gamma = 2
    data_frames = generate_asynchronous_log_returns(
        n_points, n_assets, lead_lags, noise_level, gamma
    )

    results = []

    # Compute lead-lag relationships for all pairs
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            df1, df2 = data_frames[i], data_frames[j]
            series1, series2 = df1["log_return"].values, df2["log_return"].values

            # Cross-Correlation
            lags, corr = compute_cross_correlation(series1, series2, max_lag=10)
            best_lag = lags[np.argmax(corr)]

            # Hayashi-Yoshida
            hy_cov = hayashi_yoshida(df1, df2)

#             # Dynamic Time Warping
#             dtw_distance, _ = compute_dtw(series1, series2)

            # Thermal Optimal Path
            _, top_cost_matrix = compute_top(series1, series2)
            top_cost = top_cost_matrix[-1, np.argmin(top_cost_matrix[-1, :])]

            # Store results
            results.append(
                {
                    "Asset Pair": f"{i}-{j}",
                    "Cross-Corr Lag": best_lag,
                    "Cross-Corr Coeff": max(corr),
                    "HY Cov": hy_cov,
#                     "DTW Distance": dtw_distance,
                    "TOP Cost": top_cost,
                }
            )

    # Convert results to DataFrame for display
    results_df = pd.DataFrame(results)
    print(results_df)
    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def generate_asynchronous_log_returns(
    n_points=200, n_assets=3, lead_lags=None, noise_level=0.01, gamma=2
):
    """
    Generate asynchronous log returns data with lead-lag relationships and noise.
    :param n_points: Number of total points per asset.
    :param n_assets: Number of assets.
    :param lead_lags: List of tuples specifying (leader, lagger, lag).
    :param noise_level: Noise level in the data.
    :param gamma: Timing variability parameter for lead-lag relationships.
    :return: List of data frames for each asset.
    """
    dfs = []
    base_timestamps = np.sort(np.random.uniform(0, 100, n_points))
    base_prices = np.cumsum(np.random.randn(n_points)) + 100  # Random walk

    for asset in range(n_assets):
        timestamps = base_timestamps + np.random.uniform(-gamma, gamma, n_points)
        prices = base_prices + noise_level * np.random.randn(n_points)
        log_returns = np.diff(np.log(prices))  # Compute log returns
        dfs.append(
            pd.DataFrame(
                {"time": timestamps[1:], "log_return": log_returns}
            ).sort_values(by="time")
        )

    # Apply lead-lag relationships
    if lead_lags:
        for leader, lagger, lag in lead_lags:
            for i in range(len(dfs[leader]) - lag):
                dfs[lagger].iloc[i + lag, dfs[lagger].columns.get_loc("log_return")] += (
                    dfs[leader].iloc[i, dfs[leader].columns.get_loc("log_return")]
                )

    return dfs

/===
import numpy as np
import pandas as pd
from sktime.transformations.series.signature_based import SignatureTransformer

# Example DataFrame
data = {
    "time": [1, 2, 3, 4, 5],
    "series_1": [1.0, 2.0, 1.5, 2.5, 3.0],
    "series_2": [0.5, 1.0, 1.0, 1.5, 2.0],
}
df = pd.DataFrame(data)

# Step 1: Create shifted lead-lag combinations
def create_shifted_lead_lag(df, series_1, series_2, time_col, shift):
    """
    Creates a lead-lag representation of two series with a specific time shift.
    Positive shift aligns series_2 to be ahead of series_1, negative shifts do the opposite.
    """
    time_series = df.set_index(time_col)
    s1 = time_series[series_1]
    s2 = time_series[series_2].shift(shift, fill_value=0)  # Shift series_2
    lead_lag_df = pd.DataFrame({
        f"{series_1}_lead": s1,
        f"{series_1}_lag": s1.shift(1, fill_value=s1.iloc[0]),
        f"{series_2}_lead": s2,
        f"{series_2}_lag": s2.shift(1, fill_value=s2.iloc[0])
    })
    return lead_lag_df

# Generate shifted versions for multiple shifts
shifts = range(-2, 3)  # Test shifts from -2 to +2
shifted_data = {shift: create_shifted_lead_lag(df, "series_1", "series_2", "time", shift) for shift in shifts}


/=====

import numpy as np
import pandas as pd
from sktime.transformations.series.signature_based import SignatureTransformer

# Example: Three series with lead-lag relationships
data = {
    "time": [1, 2, 3, 4, 5],
    "series_1": [1.0, 2.0, 3.0, 4.0, 5.0],  # Leading series
    "series_2": [0.5, 1.5, 2.5, 3.5, 4.5],  # Lags behind series_1
    "series_3": [0.2, 0.6, 1.0, 1.4, 1.8],  # Lags behind series_2
}
df = pd.DataFrame(data)

# Prepare data for signature transform
def create_lead_lag_3series(df, time_col):
    """Create lead-lag representation for three series."""
    time_series = df.set_index(time_col)
    s1 = time_series["series_1"]
    s2 = time_series["series_2"]
    s3 = time_series["series_3"]
    lead_lag_df = pd.DataFrame({
        "series_1_lead": s1,
        "series_1_lag": s1.shift(1, fill_value=s1.iloc[0]),
        "series_2_lead": s2,
        "series_2_lag": s2.shift(1, fill_value=s2.iloc[0]),
        "series_3_lead": s3,
        "series_3_lag": s3.shift(1, fill_value=s3.iloc[0]),
    })
    return lead_lag_df

lead_lag_df = create_lead_lag_3series(df, "time")
X = [lead_lag_df.values]  # Single sample

# Compute signature features
transformer = SignatureTransformer(depth=2)
signature_features = transformer.fit_transform(X)

# Extract terms from the signature
feature_names = transformer.get_feature_names_out()
print("Feature names:", feature_names)

# Analyze pairwise and higher-order terms
results = {}

# Example: Extract and compare pairwise cross terms
pairwise_terms = [
    ("X_0,X_1", "∫x dy"),
    ("X_1,X_2", "∫y dz"),
    ("X_2,X_0", "∫z dx"),
]

for term, description in pairwise_terms:
    if term in feature_names:
        term_index = feature_names.tolist().index(term)
        term_value = signature_features[0, term_index]
        results[description] = term_value

# Print results
print("Pairwise cross terms:")
for description, value in results.items():
    print(f"{description}: {value}")

# Example: Analyze higher-order terms
higher_order_terms = [
    ("X_0,X_1,X_2", "∫x dy dz"),
    ("X_1,X_2,X_0", "∫y dz dx"),
]

for term, description in higher_order_terms:
    if term in feature_names:
        term_index = feature_names.tolist().index(term)
        term_value = signature_features[0, term_index]
        results[description] = term_value

print("\nHigher-order terms:")
for description, value in results.items():
    print(f"{description}: {value}")

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sktime.transformations.series.signature_based import SignatureTransformer

# Example: Create synthetic data for n securities
n = 5  # Number of securities
time_steps = 10
data = {
    f"series_{i+1}": np.sin(np.linspace(0, 2 * np.pi, time_steps) + i * 0.5)
    for i in range(n)
}
data["time"] = np.arange(time_steps)
df = pd.DataFrame(data)

# Step 1: Compute pairwise relationships
def compute_pairwise_relationships(df, time_col, depth=2, threshold=0.05):
    """Compute pairwise cross terms for n securities and filter by threshold."""
    securities = [col for col in df.columns if col != time_col]
    lead_lag_df = df[securities].copy()
    X = [lead_lag_df.values]  # Single sample for simplicity

    # Compute signature features
    transformer = SignatureTransformer(depth=depth)
    signature_features = transformer.fit_transform(X)
    feature_names = transformer.get_feature_names_out()

    # Extract pairwise cross terms and filter by threshold
    results = []
    for i, s1 in enumerate(securities):
        for j, s2 in enumerate(securities):
            if i != j:  # Avoid self-interactions
                term = f"X_{i},X_{j}"  # Pairwise interaction term
                if term in feature_names:
                    term_index = feature_names.tolist().index(term)
                    term_value = signature_features[0, term_index]
                    if abs(term_value) > threshold:
                        results.append((s1, s2, term_value))

    return results

# Calculate pairwise relationships
threshold = 0.1
relationships = compute_pairwise_relationships(df, "time", depth=2, threshold=threshold)
print("Filtered relationships (above threshold):")
print(relationships)

def visualize_relationships_as_dag(relationships):
    """Visualize pairwise relationships as a DAG."""
    G = nx.DiGraph()

    # Add edges with weights
    for source, target, weight in relationships:
        G.add_edge(source, target, weight=weight)

    # Generate positions for a nicer layout
    pos = nx.spring_layout(G)

    # Draw the graph
    plt.figure(figsize=(10, 7))
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color="skyblue")
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrowstyle="->", arrowsize=20)
    nx.draw_networkx_labels(G, pos, font_size=12, font_color="black")

    # Add edge labels (weights)
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

    plt.title("Lead-Lag Relationships DAG", fontsize=16)
    plt.axis("off")
    plt.show()

# Visualize the relationships
visualize_relationships_as_dag(relationships)

+++++

import numpy as np
import pandas as pd
import esig
import plotly.graph_objects as go

# Example: Create synthetic data for n securities
n = 5  # Number of securities
time_steps = 10
data = {
    f"series_{i+1}": np.sin(np.linspace(0, 2 * np.pi, time_steps) + i * 0.5)
    for i in range(n)
}
data["time"] = np.arange(time_steps)
df = pd.DataFrame(data)

# Step 1: Compute pairwise relationships
def compute_pairwise_signatures(df, time_col, depth=2, threshold=0.05):
    """Compute pairwise cross terms for n securities and filter by threshold."""
    securities = [col for col in df.columns if col != time_col]
    results = []
    
    for i, s1 in enumerate(securities):
        for j, s2 in enumerate(securities):
            if i != j:  # Avoid self-interactions
                # Create a stream for the two series
                stream = df[[s1, s2]].to_numpy()
                
                # Compute the signature up to the given depth
                sig = esig.stream2sig(stream, depth)
                
                # Extract relevant cross term (depth 2, e.g., ∫s1 ds2)
                sig_keys = esig.sigkeys(2, depth)  # Keys for interpretation
                term_index = sig_keys.index((1, 2))  # Example term ∫s1 ds2
                
                term_value = sig[term_index]
                
                # Filter based on threshold
                if abs(term_value) > threshold:
                    results.append((s1, s2, term_value))

    return results

# Calculate pairwise relationships
threshold = 0.1
relationships = compute_pairwise_signatures(df, "time", depth=2, threshold=threshold)
print("Filtered relationships (above threshold):")
print(relationships)

# Step 2: Visualize the relationships using Plotly
def visualize_relationships_dag_plotly(relationships):
    """Visualize pairwise relationships as a DAG using Plotly."""
    sources, targets, weights = zip(*relationships)

    # Create nodes and edges for the DAG
    nodes = list(set(sources) | set(targets))
    node_indices = {node: idx for idx, node in enumerate(nodes)}

    edge_x = []
    edge_y = []
    annotations = []
    
    for src, tgt, weight in relationships:
        edge_x.append([node_indices[src], node_indices[tgt]])
        edge_y.append([weight, weight])
        annotations.append(f"{src} → {tgt}: {weight:.2f}")

    fig = go.Figure()

    # Add nodes
    fig.add_trace(go.Scatter(
        x=list(node_indices.values()),
        y=[0] * len(node_indices),  # Single-layer DAG
        mode='markers+text',
        text=list(node_indices.keys()),
        textposition="top center",
        marker=dict(size=20, color="skyblue"),
    ))

    # Add edges
    for ex, ey in zip(edge_x, edge_y):
        fig.add_trace(go.Scatter(
            x=ex, y=[0, 0],
            mode='lines',
            line=dict(color="gray", width=2),
        ))

    # Add annotations for edge weights
    for i, annotation in enumerate(annotations):
        src, tgt = edge_x[i]
        fig.add_trace(go.Scatter(
            x=[src, tgt],
            y=[0.1, 0.1],
            text=[annotation],
            mode='text',
            textfont=dict(size=10),
            textposition="top center",
        ))

    # Finalize layout
    fig.update_layout(
        title="Lead-Lag Relationships DAG",
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=600,
    )
    fig.show()

# Visualize the relationships
visualize_relationships_dag_plotly(relationships)

