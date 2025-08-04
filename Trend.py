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
