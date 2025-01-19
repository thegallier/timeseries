import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Optional libraries
# ---------------------------
try:
    from esig import tosig  # for signature
    ESIG_AVAILABLE = True
except ImportError:
    ESIG_AVAILABLE = False

try:
    import pywt  # for wavelet
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False

try:
    from dtaidistance import dtw  # for DTW
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False

from scipy.stats import wasserstein_distance  # for Wasserstein
import math

# ------------------------------------------------------------------
# Thermal Optimal Path (TOP) Code - adapted / integrated
# ------------------------------------------------------------------

def default_error(a, b):
    """
    Default error function for the partition function.
    You can adapt or import from your 'thermal_optimal_path.error_models'.
    """
    return (a - b)**2

def iter_lattice(n, exclude_boundary=True):
    """
    Generator for the partition function integer coordinates, adapted from your snippet.
    """
    start_time = 1 if exclude_boundary else 0
    for t in range(start_time, 2 * n):
        offset = 1 if exclude_boundary else 0
        start = max(t - 2 * n + 1, -t + offset)
        end = min(2 * n - t - 1, t - offset)
        if (start + t) % 2:
            start += 1
        for x in range(start, end + 1, 2):
            t_a = (t - x) // 2
            t_b = (t + x) // 2
            yield x, t, t_a, t_b

def partition_function(series_a, series_b, temperature, error_func=None):
    """
    Compute the partition function given two time series and the temperature parameter,
    per Sornette & Zhou (2004).
    """
    if error_func is None:
        error_func = default_error
    return _partition_function_impl(series_a, series_b, temperature, error_func)

def _partition_function_impl(series_a, series_b, temperature, error_func):
    size_a = len(series_a)
    size_b = len(series_b)
    if size_a != size_b:
        raise NotImplementedError("TOP code here only handles same-length series for simplicity.")

    # G matrix
    g = np.full((size_a, size_b), np.nan)

    # boundary conditions
    g[0, :] = 0
    g[:, 0] = 0
    g[0, 0] = 1
    if size_a > 1:
        g[0, 1] = 1
        g[1, 0] = 1

    # Fill
    for x, t, t_a, t_b in iter_lattice(size_a):
        g_sum = g[t_a, t_b - 1] + g[t_a - 1, t_b] + g[t_a - 1, t_b - 1]
        cost = error_func(series_a[t_a], series_b[t_b])
        val = g_sum * math.exp(-cost / temperature)
        g[t_a, t_b] = val

    return g

def average_path(partition_func):
    """
    Compute the average path from the partition function.
    Returns an array of length 2*n - 1 in the new coordinates.
    For direct interpretation in original time, typically you sample every-other point.
    """
    n = partition_func.shape[0]
    total = np.full(2*n - 1, np.nan)
    avg = np.zeros(2*n - 1)
    for x, t, t_a, t_b in iter_lattice(n, exclude_boundary=False):
        if np.isnan(total[t]):
            total[t] = 0
        total[t] += partition_func[t_a, t_b]
        avg[t] += x * partition_func[t_a, t_b]
    return avg / total

def compute_top_features(series_a, series_b, temperature=1.0):
    """
    Wrapper: compute the TOP partition function & average path, 
    then produce a few useful derived results as a feature vector or dictionary.
    
    - partition_func: 2D array
    - final_value: partition_func[-1, -1], i.e. 'end' accumulative measure
    - avg_path: 1D array (length=2N-1)
    
    For convenience, we return a dictionary with these items.
    """
    pf = partition_function(series_a, series_b, temperature=temperature)
    avg_p = average_path(pf)
    final_val = pf[-1, -1]  # the bottom-right corner might be an interesting measure

    return {
        "partition_function": pf,
        "average_path": avg_p,
        "final_value": final_val
    }

# ------------------------------------------------------------------
# Other Tools (Signatures, Wavelets, DTW, HY, etc.)
# ------------------------------------------------------------------

def compute_signature(sub_df, column='price', level=2):
    if not ESIG_AVAILABLE:
        return np.array([])
    arr = sub_df[column].values.reshape(-1, 1)
    return tosig.stream2sig(arr, level)

def compute_wavelet_features(sub_df, column='price', wavelet='morl', scales=range(1,6)):
    if not PYWT_AVAILABLE:
        return np.array([])
    arr = sub_df[column].values
    coeffs, freqs = pywt.cwt(arr, scales, wavelet)
    return np.sum(np.abs(coeffs), axis=1)  # sum across time

def dtw_distance_subpaths(sub_df1, sub_df2, column='price'):
    if not DTW_AVAILABLE:
        return np.nan
    x = sub_df1[column].values
    y = sub_df2[column].values
    dist = dtw.distance(x, y)
    return dist

def hayashi_yoshida_covar(dfA, dfB, shift_ms=0):
    if shift_ms != 0:
        dfB = dfB.copy()
        dfB['time'] += shift_ms
    dfA = dfA.sort_values('time')
    dfB = dfB.sort_values('time')
    timesA, pxA = dfA['time'].values, dfA['price'].values
    timesB, pxB = dfB['time'].values, dfB['price'].values

    # increments
    A_incr = []
    for i in range(1, len(pxA)):
        dX = pxA[i] - pxA[i-1]
        t0A, t1A = timesA[i-1], timesA[i]
        A_incr.append((t0A, t1A, dX))

    B_incr = []
    for j in range(1, len(pxB)):
        dY = pxB[j] - pxB[j-1]
        t0B, t1B = timesB[j-1], timesB[j]
        B_incr.append((t0B, t1B, dY))

    hy_sum = 0.0
    for (A0, A1, dX) in A_incr:
        for (B0, B1, dY) in B_incr:
            if A1 > B0 and B1 > A0:  # overlap
                hy_sum += dX * dY
    return hy_sum

def hayashi_yoshida_correlation(dfA, dfB, shift_ms=0):
    covAB = hayashi_yoshida_covar(dfA, dfB, shift_ms=shift_ms)
    varA = hayashi_yoshida_covar(dfA, dfA, shift_ms=0)
    varB = hayashi_yoshida_covar(dfB, dfB, shift_ms=0)
    if varA <= 0 or varB <= 0:
        return np.nan
    return covAB / np.sqrt(varA * varB)

# ------------------------------------------------------------------
# Simple Data Pipeline (Generation, Distribution Comparison)
# ------------------------------------------------------------------

def create_validation_dataset(n=200, seed=42):
    np.random.seed(seed)
    times = np.arange(n) * 10
    securities = np.random.choice(['AAPL','GOOG','TSLA'], size=n, p=[0.4,0.3,0.3])
    sides = np.random.choice(['buy','sell'], size=n, p=[0.6,0.4])

    amount_list = []
    for sec, side in zip(securities, sides):
        base = 100 if sec=='AAPL' else 80 if sec=='GOOG' else 50
        scale = 1.2 if side=='buy' else 0.8
        amt_val = np.random.gamma(shape=2.0, scale=base*scale/2.0)
        amount_list.append(amt_val)

    data1_list, data2_list, price_list = [],[],[]
    for sec in securities:
        if sec=='AAPL':
            d1 = np.random.normal(150,5)
            d2 = np.random.normal(200,10)
            base_price = 150
        elif sec=='GOOG':
            d1 = np.random.normal(100,3)
            d2 = np.random.normal(300,20)
            base_price = 100
        else:
            d1 = np.random.normal(180,8)
            d2 = np.random.normal(400,30)
            base_price = 180
        p = base_price + 0.1*d1 + 0.05*d2 + np.random.normal(0,2)
        data1_list.append(d1)
        data2_list.append(d2)
        price_list.append(p)

    df = pd.DataFrame({
        'time': times,
        'security': securities,
        'side': sides,
        'amount': amount_list,
        'price': price_list,
        'data1': data1_list,
        'data2': data2_list
    })
    return df

def generate_synthetic_data(n=150, reference_df=None, use_features_from_reference=True, seed=999):
    np.random.seed(seed)
    if reference_df is None or not use_features_from_reference:
        times = np.arange(n)*5
        secs = np.random.choice(['AAPL','GOOG','TSLA'], size=n)
        sds = np.random.choice(['buy','sell'], size=n)
        amt = np.random.gamma(shape=2.0, scale=70, size=n)
        d1 = np.random.normal(150,10,size=n)
        d2 = np.random.normal(300,50,size=n)
        px = 100 + 0.2*d1 + 0.1*d2 + np.random.normal(0,5,size=n)
        return pd.DataFrame({
            'time': times,'security':secs,'side':sds,
            'amount':amt,'price':px,'data1':d1,'data2':d2
        })
    else:
        groups = reference_df.groupby(['security','side'])
        group_keys = list(groups.groups.keys())
        group_sizes = [len(groups.get_group(k)) for k in group_keys]
        total = sum(group_sizes)
        probs = [gs/total for gs in group_sizes]
        rows = []
        for i in range(n):
            t = i*6
            choice_key = np.random.choice(group_keys, p=probs)
            sub_df = groups.get_group(choice_key)
            row = sub_df.sample(n=1).iloc[0]
            rows.append((t, choice_key[0], choice_key[1],
                         row['amount'], row['price'], row['data1'], row['data2']))
        return pd.DataFrame(rows, columns=['time','security','side','amount','price','data1','data2'])

def compare_distributions_wasserstein(df_real, df_synth, columns=('amount','price','data1','data2')):
    results={}
    for col in columns:
        rvals = df_real[col].dropna().values
        svals = df_synth[col].dropna().values
        wdist = wasserstein_distance(rvals,svals)
        results[col]=wdist
    return results

def compare_conditional_distributions_wasserstein(df_real, df_synth, group_cols=('security','side'),
                                                 columns=('amount','price','data1','data2')):
    real_groups = df_real.groupby(list(group_cols))
    synth_groups= df_synth.groupby(list(group_cols))
    all_keys = set(real_groups.groups.keys()).union(synth_groups.groups.keys())
    cond_results={}
    for gk in all_keys:
        if gk not in real_groups.groups or gk not in synth_groups.groups:
            continue
        sub_r = real_groups.get_group(gk)
        sub_s = synth_groups.get_group(gk)
        d_sub={}
        for col in columns:
            rv = sub_r[col].dropna().values
            sv = sub_s[col].dropna().values
            if len(rv)>1 and len(sv)>1:
                wd=wasserstein_distance(rv,sv)
            else:
                wd=np.nan
            d_sub[col]=wd
        cond_results[gk]=d_sub
    return cond_results

# ------------------------------------------------------------------
# Main demonstration
# ------------------------------------------------------------------
def main():
    # 1) Real dataset
    df_validation = create_validation_dataset(n=200, seed=42)
    df_validation.sort_values('time', inplace=True)

    # 2) Example: pick two sub-series (AAPL vs GOOG) for Thermal Optimal Path
    df_aapl = df_validation[df_validation['security']=='AAPL'].copy()
    df_goog = df_validation[df_validation['security']=='GOOG'].copy()

    # Ensure same length if we want to do TOT directly. For illustration, let's truncate the longer one.
    min_len = min(len(df_aapl), len(df_goog))
    df_aapl = df_aapl.iloc[:min_len].reset_index(drop=True)
    df_goog = df_goog.iloc[:min_len].reset_index(drop=True)

    # Extract the 'price' series from each
    price_a = df_aapl['price'].values
    price_g = df_goog['price'].values

    # 3) Compute Thermal Optimal Path
    top_result = compute_top_features(price_a, price_g, temperature=1.0)
    pf = top_result["partition_function"]
    avg_p = top_result["average_path"]
    final_val = top_result["final_value"]
    print(f"\n-- Thermal Optimal Path Results (AAPL vs GOOG) --\n"
          f"Partition Function shape: {pf.shape}\n"
          f"Final Value (pf[-1,-1]): {final_val:.4f}\n"
          f"Average Path shape: {avg_p.shape}\n")

    # 4) Compare with other methods on the same sub-series
    # a) DTW
    if DTW_AVAILABLE:
        dtw_dist = dtw.distance(price_a, price_g)
        print(f"DTW distance (AAPL vs GOOG prices): {dtw_dist:.4f}")
    else:
        print("DTW library not installed.")

    # b) Hayashi–Yoshida Cov/Correlation
    hy_cov = hayashi_yoshida_covar(df_aapl, df_goog, shift_ms=0)
    hy_corr= hayashi_yoshida_correlation(df_aapl, df_goog, shift_ms=0)
    print(f"Hayashi–Yoshida Covariance: {hy_cov:.4f}, Correlation: {hy_corr:.4f}")

    # 5) Synthetic data generation & distribution comparison
    df_synth_uncond = generate_synthetic_data(n=150, reference_df=None, use_features_from_reference=False)
    df_synth_cond   = generate_synthetic_data(n=150, reference_df=df_validation, use_features_from_reference=True)

    # Unconditional distribution
    dist_uncond = compare_distributions_wasserstein(df_validation, df_synth_uncond)
    dist_cond   = compare_distributions_wasserstein(df_validation, df_synth_cond)

    print("\n-- Wasserstein Distances (Unconditional) --")
    for k,v in dist_uncond.items():
        print(f"  {k}: {v:.4f}")

    print("\n-- Wasserstein Distances (Conditional) --")
    for k,v in dist_cond.items():
        print(f"  {k}: {v:.4f}")

    # 6) Chunking + Signatures/Wavelets on the real data
    #    (just to show integration)
    chunk_size = 20
    subpaths = []
    for sym in df_validation['security'].unique():
        df_sym = df_validation[df_validation['security']==sym].copy()
        df_sym.sort_values('time', inplace=True)
        sp = chunk_by_fixed_size(df_sym, chunk_size=chunk_size)
        for idx, s in enumerate(sp):
            subpaths.append((sym, idx, s))

    # Example: compute signature & wavelet features on the first subpath
    if subpaths:
        (sym, idx, first_sub) = subpaths[0]
        print(f"\nExample chunk: security={sym}, chunk_index={idx}, shape={first_sub.shape}")

        if ESIG_AVAILABLE:
            sig_vec = compute_signature(first_sub, column='price', level=2)
            print(f"Signature vector (price, level=2): {sig_vec}")

        if PYWT_AVAILABLE:
            wave_feats = compute_wavelet_features(first_sub, column='price', wavelet='morl', scales=range(1,4))
            print(f"Wavelet features (price) shape: {wave_feats.shape}, values={wave_feats}")

    # 7) Simple plot of the partition function or the average path
    plt.figure(figsize=(6,5))
    plt.imshow(pf, aspect='auto', origin='lower', cmap='inferno')
    plt.title("Thermal Optimal Path - Partition Function (AAPL vs GOOG)")
    plt.colorbar(label='Partition Function Value')

    plt.figure(figsize=(8,3))
    plt.plot(avg_p, label='Average Path (thermal coords)')
    plt.title("Thermal Optimal Path - Average Path (2*N - 1 points)")
    plt.legend()
    plt.show()

if __name__=="__main__":
    main()
