import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
from scipy.interpolate import UnivariateSpline, PchipInterpolator
from scipy.special import comb
import matplotlib.pyplot as plt
import warnings

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class SmoothingConfig:
    """Configuration for smoothing methods."""
    tenor_method: str = 'pspline'  
    time_method: str = 'ewma'  
    tenor_params: Dict[str, Any] = None
    time_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tenor_params is None:
            self.tenor_params = {}
        if self.time_params is None:
            self.time_params = {}

# ============================================================================
# TENOR SMOOTHING
# ============================================================================

class TenorSmoother:
    """Handles cross-sectional smoothing across tenors."""
    
    def __init__(self, knot_positions: List[float] = None):
        self.knot_positions = knot_positions or [2, 5, 10, 30]
    
    def pspline(self, tenors: np.ndarray, values: np.ndarray, 
                smoothing_param: float = None, degree: int = 3) -> np.ndarray:
        """P-spline smoothing."""
        if np.all(np.isnan(values)):
            return values
        
        mask = ~np.isnan(values)
        n_valid = np.sum(mask)
        
        if n_valid < max(degree + 1, 2):
            if n_valid < 2:
                return values
            return np.interp(tenors, tenors[mask], values[mask])
        
        valid_tenors = tenors[mask]
        valid_values = values[mask]
        
        # Handle duplicates
        unique_tenors, inverse_indices = np.unique(valid_tenors, return_inverse=True)
        if len(unique_tenors) < len(valid_tenors):
            unique_values = np.zeros(len(unique_tenors))
            for i in range(len(unique_tenors)):
                unique_values[i] = np.mean(valid_values[inverse_indices == i])
            valid_tenors = unique_tenors
            valid_values = unique_values
        
        if len(valid_tenors) < degree + 1:
            degree = max(1, len(valid_tenors) - 1)
        
        try:
            if smoothing_param is None or smoothing_param == 0:
                spline = UnivariateSpline(valid_tenors, valid_values, k=degree, s=0)
            else:
                # IMPORTANT: For your data scale, we need larger s values
                # s represents the sum of squared residuals allowed
                s_value = smoothing_param * len(valid_values) * np.var(valid_values)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    spline = UnivariateSpline(valid_tenors, valid_values, k=degree, s=s_value)
            
            result = spline(tenors)
            return result
            
        except Exception as e:
            return np.interp(tenors, valid_tenors, valid_values)
    
    def bernstein_polynomial(self, tenors: np.ndarray, values: np.ndarray,
                           degree: int = 5) -> np.ndarray:
        """Bernstein polynomial smoothing."""
        if np.any(np.isnan(values)):
            mask = ~np.isnan(values)
            if np.sum(mask) < 2:
                return values
            valid_tenors = tenors[mask]
            valid_values = values[mask]
        else:
            valid_tenors = tenors
            valid_values = values
        
        if len(valid_values) < degree + 1:
            return np.interp(tenors, valid_tenors, valid_values)
        
        t_min, t_max = np.min(valid_tenors), np.max(valid_tenors)
        if t_max - t_min < 1e-10:
            return values
        
        t_norm_valid = (valid_tenors - t_min) / (t_max - t_min)
        t_norm_all = (tenors - t_min) / (t_max - t_min)
        
        n = degree
        B_valid = np.column_stack([
            comb(n, i) * (t_norm_valid ** i) * ((1 - t_norm_valid) ** (n - i))
            for i in range(n + 1)
        ])
        
        regularization = 1e-8 * np.eye(n + 1)
        try:
            coeffs = np.linalg.solve(
                B_valid.T @ B_valid + regularization,
                B_valid.T @ valid_values
            )
        except:
            coeffs, _, _, _ = np.linalg.lstsq(B_valid, valid_values, rcond=None)
        
        B_all = np.column_stack([
            comb(n, i) * (t_norm_all ** i) * ((1 - t_norm_all) ** (n - i))
            for i in range(n + 1)
        ])
        
        return B_all @ coeffs
    
    def cubic_spline(self, tenors: np.ndarray, values: np.ndarray,
                     tension: float = 0.0) -> np.ndarray:
        """Cubic spline interpolation."""
        if np.all(np.isnan(values)):
            return values
        
        mask = ~np.isnan(values)
        n_valid = np.sum(mask)
        
        if n_valid < 2:
            return values
        if n_valid < 4:
            return np.interp(tenors, tenors[mask], values[mask])
        
        try:
            valid_tenors = tenors[mask]
            valid_values = values[mask]
            
            if len(np.unique(valid_tenors)) < len(valid_tenors):
                unique_tenors = np.unique(valid_tenors)
                unique_values = np.array([np.mean(valid_values[valid_tenors == t]) 
                                        for t in unique_tenors])
                valid_tenors = unique_tenors
                valid_values = unique_values
            
            spline = UnivariateSpline(valid_tenors, valid_values, k=3, s=tension)
            return spline(tenors)
        except:
            return np.interp(tenors, tenors[mask], values[mask])
    
    def monotonic_spline(self, tenors: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Monotonic spline (PCHIP)."""
        if np.any(np.isnan(values)):
            mask = ~np.isnan(values)
            if np.sum(mask) < 2:
                return values
            valid_tenors = tenors[mask]
            valid_values = values[mask]
            interp = PchipInterpolator(valid_tenors, valid_values)
            return interp(tenors)
        
        interp = PchipInterpolator(tenors, values)
        return interp(tenors)
    
    def smooth(self, tenors: np.ndarray, values: np.ndarray,
               method: str, **params) -> np.ndarray:
        """Apply specified smoothing method."""
        if method == 'none':
            return values
        elif method == 'pspline':
            return self.pspline(tenors, values, **params)
        elif method == 'cubic_spline':
            return self.cubic_spline(tenors, values, **params)
        elif method == 'monotonic':
            return self.monotonic_spline(tenors, values)
        elif method == 'bernstein':
            return self.bernstein_polynomial(tenors, values, **params)
        else:
            raise ValueError(f"Unknown tenor smoothing method: {method}")

# ============================================================================
# TIME SMOOTHING
# ============================================================================

# ============================================================================
# TIME SMOOTHING (EXTENDED WITH ALL METHODS)
# ============================================================================

class TimeSmoother:
    """Handles time-series smoothing - all methods are causal (backward-looking only)."""
    
    @staticmethod
    def ewma(values: np.ndarray, decay: float = 0.94) -> np.ndarray:
        """Exponentially weighted moving average - causal filter."""
        n = len(values)
        smoothed = np.zeros(n)
        
        init_idx = 0
        while init_idx < n and np.isnan(values[init_idx]):
            init_idx += 1
        
        if init_idx >= n:
            return values
        
        smoothed[init_idx] = values[init_idx]
        
        for i in range(init_idx + 1, n):
            if np.isnan(values[i]):
                smoothed[i] = smoothed[i - 1]
            else:
                smoothed[i] = decay * smoothed[i - 1] + (1 - decay) * values[i]
        
        smoothed[:init_idx] = smoothed[init_idx]
        return smoothed
    
    @staticmethod
    def double_ewma(values: np.ndarray, decay: float = 0.94) -> np.ndarray:
        """Double EWMA for trend-preserving smoothing - causal."""
        first_smooth = TimeSmoother.ewma(values, decay)
        second_smooth = TimeSmoother.ewma(first_smooth, decay)
        return second_smooth
    
    @staticmethod
    def kalman_filter(values: np.ndarray, 
                      process_noise: float = 0.01,
                      measurement_noise: float = 0.1) -> np.ndarray:
        """
        Simple Kalman filter for 1D time series - causal by design.
        Only uses observations up to current time.
        """
        n = len(values)
        filtered = np.zeros(n)
        
        # Find first non-NaN value for initialization
        init_idx = 0
        while init_idx < n and np.isnan(values[init_idx]):
            init_idx += 1
        
        if init_idx >= n:
            return values
        
        # Initialize state and covariance
        x = values[init_idx]
        P = 1.0
        
        filtered[:init_idx + 1] = x
        
        for i in range(init_idx + 1, n):
            # Predict (time update)
            x_pred = x
            P_pred = P + process_noise
            
            if not np.isnan(values[i]):
                # Update (measurement update)
                K = P_pred / (P_pred + measurement_noise)
                x = x_pred + K * (values[i] - x_pred)
                P = (1 - K) * P_pred
            else:
                # No update for NaN
                x = x_pred
                P = P_pred
            
            filtered[i] = x
        
        return filtered
    
    @staticmethod
    def gaussian_process(values: np.ndarray, 
                        length_scale: float = 10.0,
                        noise_level: float = 0.1) -> np.ndarray:
        """
        Causal Gaussian Process regression.
        For each point, only uses historical data for prediction.
        """
        n = len(values)
        smoothed = np.zeros(n)
        
        # Find first non-NaN
        init_idx = 0
        while init_idx < n and np.isnan(values[init_idx]):
            init_idx += 1
        
        if init_idx >= n:
            return values
        
        smoothed[:init_idx + 1] = values[init_idx]
        
        # RBF kernel function
        def rbf_kernel(t1, t2, length_scale, signal_var=1.0):
            t1 = np.atleast_1d(t1)
            t2 = np.atleast_1d(t2)
            dist = np.abs(t1[:, None] - t2[None, :])
            return signal_var * np.exp(-0.5 * (dist / length_scale) ** 2)
        
        # Process each point using only historical data
        for i in range(init_idx + 1, n):
            # Use all non-NaN historical points up to i-1
            hist_mask = ~np.isnan(values[:i])
            if np.sum(hist_mask) == 0:
                smoothed[i] = smoothed[i - 1]
                continue
            
            t_hist = np.where(hist_mask)[0]
            y_hist = values[hist_mask]
            
            # Limit history to recent past for efficiency
            max_lookback = min(100, len(t_hist))
            if len(t_hist) > max_lookback:
                t_hist = t_hist[-max_lookback:]
                y_hist = y_hist[-max_lookback:]
            
            try:
                # Compute kernel matrices
                K_hist = rbf_kernel(t_hist, t_hist, length_scale) + \
                        noise_level * np.eye(len(t_hist))
                k_pred = rbf_kernel(np.array([i]), t_hist, length_scale).flatten()
                
                # GP prediction
                L = np.linalg.cholesky(K_hist + 1e-6 * np.eye(len(t_hist)))
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_hist))
                smoothed[i] = np.dot(k_pred, alpha)
                
                # If current observation is not NaN, blend it with prediction
                if not np.isnan(values[i]):
                    obs_weight = 1 / (1 + noise_level)
                    smoothed[i] = obs_weight * values[i] + (1 - obs_weight) * smoothed[i]
                    
            except np.linalg.LinAlgError:
                smoothed[i] = smoothed[i - 1]
        
        return smoothed
    
    @staticmethod
    def moving_average(values: np.ndarray, window: int = 10) -> np.ndarray:
        """Simple moving average - causal implementation."""
        n = len(values)
        smoothed = np.zeros(n)
        
        for i in range(n):
            # Look back only (causal)
            start_idx = max(0, i - window + 1)
            window_data = values[start_idx:i+1]
            
            # Handle NaN values
            valid_data = window_data[~np.isnan(window_data)]
            if len(valid_data) > 0:
                smoothed[i] = np.mean(valid_data)
            else:
                smoothed[i] = np.nan if i == 0 else smoothed[i-1]
        
        return smoothed
    
    @staticmethod
    def savitzky_golay(values: np.ndarray, window: int = 11, polyorder: int = 3) -> np.ndarray:
        """
        Savitzky-Golay filter - modified to be causal.
        Uses one-sided window looking only backward.
        """
        n = len(values)
        smoothed = np.zeros(n)
        
        for i in range(n):
            if i < window:
                # Not enough history, use simple average
                valid_data = values[:i+1][~np.isnan(values[:i+1])]
                smoothed[i] = np.mean(valid_data) if len(valid_data) > 0 else values[i]
            else:
                # Apply SG filter on backward window
                window_data = values[i-window+1:i+1]
                
                # Handle NaN by interpolation
                if np.any(np.isnan(window_data)):
                    valid_idx = ~np.isnan(window_data)
                    if np.sum(valid_idx) > polyorder:
                        x = np.arange(len(window_data))
                        window_data = np.interp(x, x[valid_idx], window_data[valid_idx])
                    else:
                        smoothed[i] = smoothed[i-1]
                        continue
                
                # Fit polynomial and evaluate at last point
                try:
                    x = np.arange(len(window_data))
                    coeffs = np.polyfit(x, window_data, polyorder)
                    smoothed[i] = np.polyval(coeffs, len(window_data)-1)
                except:
                    smoothed[i] = window_data[-1]
        
        return smoothed
    
    @staticmethod
    def hodrick_prescott(values: np.ndarray, lamb: float = 1600) -> np.ndarray:
        """
        Hodrick-Prescott filter - causal implementation.
        Standard HP filter is not causal, so we apply it in a rolling window fashion.
        """
        n = len(values)
        smoothed = np.zeros(n)
        
        # Minimum window for HP filter
        min_window = 20
        
        for i in range(n):
            if i < min_window:
                # Not enough data for HP filter
                valid_data = values[:i+1][~np.isnan(values[:i+1])]
                smoothed[i] = np.mean(valid_data) if len(valid_data) > 0 else values[i]
            else:
                # Apply HP filter on historical window
                window_size = min(i+1, 100)  # Limit window for efficiency
                window_data = values[max(0, i-window_size+1):i+1].copy()
                
                # Handle NaN
                nan_mask = np.isnan(window_data)
                if np.any(nan_mask):
                    if np.sum(~nan_mask) > 2:
                        x = np.arange(len(window_data))
                        window_data[nan_mask] = np.interp(x[nan_mask], x[~nan_mask], window_data[~nan_mask])
                    else:
                        smoothed[i] = smoothed[i-1]
                        continue
                
                # Apply HP filter
                try:
                    T = len(window_data)
                    # Create second difference matrix
                    I = np.eye(T)
                    D2 = np.diff(I, n=2, axis=0)
                    
                    # HP filter: minimize (y-trend)'(y-trend) + lambda * trend'D2'D2*trend
                    trend = np.linalg.solve(I + lamb * D2.T @ D2, window_data)
                    smoothed[i] = trend[-1]
                except:
                    smoothed[i] = window_data[-1]
        
        return smoothed
    
    @staticmethod
    def smooth(values: np.ndarray, method: str, **params) -> np.ndarray:
        """Apply specified time smoothing method - all are causal."""
        if method == 'none':
            return values
        elif method == 'ewma':
            return TimeSmoother.ewma(values, **params)
        elif method == 'double_ewma':
            return TimeSmoother.double_ewma(values, **params)
        elif method == 'kalman':
            return TimeSmoother.kalman_filter(values, **params)
        elif method == 'gp' or method == 'gaussian_process':
            return TimeSmoother.gaussian_process(values, **params)
        elif method == 'ma' or method == 'moving_average':
            return TimeSmoother.moving_average(values, **params)
        elif method == 'savgol' or method == 'savitzky_golay':
            return TimeSmoother.savitzky_golay(values, **params)
        elif method == 'hp' or method == 'hodrick_prescott':
            return TimeSmoother.hodrick_prescott(values, **params)
        else:
            raise ValueError(f"Unknown time smoothing method: {method}")

# ============================================================================
# MAIN FRAMEWORK
# ============================================================================

class YieldSmoothingFramework:
    """Main framework for smoothing yield curve data."""
    
    def __init__(self, knot_positions: List[float] = None):
        self.tenor_smoother = TenorSmoother(knot_positions)
        self.time_smoother = TimeSmoother()
    
    def smooth(self, df: pd.DataFrame, config: SmoothingConfig) -> pd.DataFrame:
        """Apply smoothing to yield curve data."""
        result = df.copy()
        
        if isinstance(result.index, pd.MultiIndex):
            result = result.sort_index()
            self._smooth_multiindex(result, config)
        
        return result
    
    def _smooth_multiindex(self, df: pd.DataFrame, config: SmoothingConfig):
        """Handle MultiIndex format (date, country, tenor)."""
        dates = df.index.get_level_values(0).unique()
        countries = df.index.get_level_values(1).unique()
        
        # First: smooth across tenors for each country-date
        if config.tenor_method != 'none':
            for date in dates:
                for country in countries:
                    try:
                        mask = (df.index.get_level_values(0) == date) & \
                               (df.index.get_level_values(1) == country)
                        
                        if mask.any():
                            data_slice = df.loc[mask]
                            
                            if len(data_slice) > 1:
                                tenors = data_slice.index.get_level_values(2).values.astype(float)
                                values = data_slice.values.flatten()
                                
                                unique_tenors = np.unique(tenors)
                                avg_values = np.array([np.mean(values[tenors == t]) 
                                                      for t in unique_tenors])
                                
                                if not np.all(np.isnan(avg_values)):
                                    smoothed_unique = self.tenor_smoother.smooth(
                                        unique_tenors, avg_values, 
                                        config.tenor_method, 
                                        **config.tenor_params
                                    )
                                    
                                    smoothed_all = np.zeros(len(tenors))
                                    for i, unique_tenor in enumerate(unique_tenors):
                                        smoothed_all[tenors == unique_tenor] = smoothed_unique[i]
                                    
                                    df.loc[mask, df.columns[0]] = smoothed_all
                    except:
                        continue
        
        # Second: smooth across time for each country-tenor
        if config.time_method != 'none':
            for country in countries:
                unique_tenors = df.index.get_level_values(2).unique()
                
                for tenor in unique_tenors:
                    try:
                        mask = (df.index.get_level_values(1) == country) & \
                               (df.index.get_level_values(2) == tenor)
                        
                        if mask.any():
                            time_series_data = df.loc[mask]
                            time_series_data = time_series_data.sort_index()
                            values = time_series_data.values.flatten()
                            
                            if not np.all(np.isnan(values)):
                                smoothed = self.time_smoother.smooth(
                                    values,
                                    config.time_method,
                                    **config.time_params
                                )
                                
                                if not np.all(np.isnan(smoothed)):
                                    df.loc[mask, df.columns[0]] = smoothed
                    except:
                        continue

# ============================================================================
# DEMO AND VISUALIZATION
# ============================================================================

def create_sample_data_with_noise() -> pd.DataFrame:
    """Create sample yield curve data with realistic noise."""
    np.random.seed(42)
    
    dates = pd.date_range('2023-01-01', periods=100, freq='B')
    countries = ['US']
    tenors = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30])
    
    data = []
    for date_idx, date in enumerate(dates):
        for country in countries:
            # Create a realistic yield curve shape
            base_curve = 0.02 + 0.03 * (1 - np.exp(-tenors/5))  # Normal curve
            
            # Add systematic daily changes
            daily_shift = 0.001 * np.sin(date_idx / 10)
            
            # Add noise that increases with tenor
            noise = np.random.randn(len(tenors)) * (0.002 + 0.001 * np.sqrt(tenors))
            
            # Add jump on specific date
            if date_idx == 50:
                noise += 0.01  # 100bp jump
            
            values = base_curve + daily_shift + noise
            
            for tenor, value in zip(tenors, values):
                data.append({
                    'date': date,
                    'country': country,
                    'tenor': tenor,
                    'yield_change': value
                })
    
    df = pd.DataFrame(data)
    df = df.set_index(['date', 'country', 'tenor'])
    return df

def visualize_smoothing_effects(df: pd.DataFrame, 
                               sample_date: pd.Timestamp = None,
                               sample_tenor: float = 5.0):
    """Visualize the effects of different smoothing methods."""
    
    framework = YieldSmoothingFramework()
    
    if sample_date is None:
        sample_date = df.index.get_level_values(0).unique()[50]
    
    # Define smoothing configurations
    configs = {
        'Original': SmoothingConfig(tenor_method='none', time_method='none'),
        
        'Light P-Spline': SmoothingConfig(
            tenor_method='pspline',
            time_method='none',
            tenor_params={'smoothing_param': 0.5}  # Light smoothing
        ),
        
        'Heavy P-Spline': SmoothingConfig(
            tenor_method='pspline',
            time_method='none',
            tenor_params={'smoothing_param': 5.0}  # Heavy smoothing
        ),
        
        'Bernstein': SmoothingConfig(
            tenor_method='bernstein',
            time_method='none',
            tenor_params={'degree': 4}
        ),
        
        'EWMA Time': SmoothingConfig(
            tenor_method='none',
            time_method='ewma',
            time_params={'decay': 0.94}
        ),
        
        'Combined': SmoothingConfig(
            tenor_method='pspline',
            time_method='ewma',
            tenor_params={'smoothing_param': 1.0},
            time_params={'decay': 0.94}
        )
    }
    
    # Apply smoothing
    results = {}
    for name, config in configs.items():
        results[name] = framework.smooth(df, config)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Tenor curve for a specific date
    ax1 = axes[0, 0]
    country = 'US'
    
    for name, result_df in results.items():
        if 'Time' not in name and 'Combined' not in name:  # Skip time-only smoothing
            data = result_df.loc[(sample_date, country)]
            tenors = data.index.get_level_values(0).values
            values = data.values.flatten()
            ax1.plot(tenors, values * 100, marker='o', label=name, alpha=0.7)
    
    ax1.set_xlabel('Tenor (years)')
    ax1.set_ylabel('Yield (%)')
    ax1.set_title(f'Tenor Smoothing Effects ({sample_date.date()}, {country})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Time series for a specific tenor
    ax2 = axes[0, 1]
    
    for name in ['Original', 'EWMA Time', 'Combined']:
        result_df = results[name]
        mask = (result_df.index.get_level_values(1) == country) & \
               (result_df.index.get_level_values(2) == sample_tenor)
        time_series = result_df.loc[mask].sort_index()
        dates = time_series.index.get_level_values(0)
        values = time_series.values.flatten()
        ax2.plot(dates, values * 100, label=name, alpha=0.7)
    
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Yield (%)')
    ax2.set_title(f'Time Smoothing Effects ({country}, {sample_tenor}Y tenor)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Roughness comparison
    ax3 = axes[1, 0]
    
    roughness_data = []
    for name, result_df in results.items():
        # Calculate roughness across tenors
        sample_data = result_df.loc[(sample_date, country)]
        values = sample_data.values.flatten()
        if len(values) > 2:
            second_diff = np.diff(values, n=2)
            roughness = np.sqrt(np.mean(second_diff ** 2))
            roughness_data.append((name, roughness * 10000))  # Scale for visibility
    
    names, roughness_values = zip(*roughness_data)
    ax3.bar(names, roughness_values)
    ax3.set_ylabel('Roughness (scaled)')
    ax3.set_title('Curve Roughness Comparison')
    ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Numerical comparison
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate statistics
    stats_text = "Smoothing Statistics:\n\n"
    original = results['Original']
    
    for name, result_df in results.items():
        if name != 'Original':
            # Calculate RMSE vs original
            diff = (result_df.values - original.values)
            rmse = np.sqrt(np.mean(diff ** 2))
            
            # Calculate smoothness improvement
            orig_values = original.loc[(sample_date, country)].values.flatten()
            smooth_values = result_df.loc[(sample_date, country)].values.flatten()
            
            if len(orig_values) > 2 and len(smooth_values) > 2:
                orig_rough = np.mean(np.diff(orig_values, n=2) ** 2)
                smooth_rough = np.mean(np.diff(smooth_values, n=2) ** 2)
                rough_reduction = (1 - smooth_rough/orig_rough) * 100 if orig_rough > 0 else 0
            else:
                rough_reduction = 0
            
            stats_text += f"{name}:\n"
            stats_text += f"  RMSE from original: {rmse*100:.3f}%\n"
            stats_text += f"  Roughness reduction: {rough_reduction:.1f}%\n\n"
    
    ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, 
             fontsize=10, verticalalignment='center', fontfamily='monospace')
    
    plt.tight_layout()
    plt.show()
    
    return results

def print_numerical_comparison(df: pd.DataFrame, results: Dict[str, pd.DataFrame],
                              sample_date: pd.Timestamp = None):
    """Print numerical comparison of smoothing methods."""
    
    if sample_date is None:
        sample_date = df.index.get_level_values(0).unique()[50]
    
    country = 'US'
    
    print("\n" + "="*70)
    print(f"NUMERICAL COMPARISON - {sample_date.date()}, {country}")
    print("="*70)
    
    # Get original values
    original_data = results['Original'].loc[(sample_date, country)]
    tenors = original_data.index.get_level_values(0).values
    
    print(f"\n{'Tenor':<6}", end='')
    for name in results.keys():
        print(f"{name[:12]:<13}", end='')
    print()
    print("-" * (6 + 13 * len(results)))
    
    # Print values for each tenor
    for tenor in tenors[:7]:  # First 7 tenors for brevity
        print(f"{tenor:<6.2f}", end='')
        for name, result_df in results.items():
            value = result_df.loc[(sample_date, country, tenor), 'yield_change']
            print(f"{value*100:>12.4f}%", end=' ')
        print()
    
    # Print smoothness metrics
    print("\n" + "="*70)
    print("SMOOTHNESS METRICS")
    print("="*70)
    
    for name, result_df in results.items():
        data = result_df.loc[(sample_date, country)]
        values = data.values.flatten()
        
        if len(values) > 2:
            # First differences (volatility)
            first_diff = np.diff(values)
            volatility = np.std(first_diff)
            
            # Second differences (roughness)
            second_diff = np.diff(values, n=2)
            roughness = np.sqrt(np.mean(second_diff ** 2))
            
            print(f"\n{name}:")
            print(f"  Volatility (std of 1st diff): {volatility*100:.5f}%")
            print(f"  Roughness (RMS of 2nd diff):  {roughness*100:.5f}%")

# ============================================================================
# CUSTOM SMOOTHING FUNCTIONS
# ============================================================================

def nelson_siegel_smoother(tenors: np.ndarray, values: np.ndarray, tau: float = 2.0) -> np.ndarray:
    """
    Nelson-Siegel model for yield curve smoothing.
    Y(m) = β0 + β1*(1-exp(-m/τ))/(m/τ) + β2*((1-exp(-m/τ))/(m/τ) - exp(-m/τ))
    """
    n = len(tenors)
    
    # Avoid division by zero
    safe_tenors = np.maximum(tenors, 0.01)
    
    # Create basis functions
    ones = np.ones(n)
    factor1 = (1 - np.exp(-safe_tenors/tau)) / (safe_tenors/tau)
    factor2 = factor1 - np.exp(-safe_tenors/tau)
    
    # Stack basis
    X = np.column_stack([ones, factor1, factor2])
    
    # Fit coefficients using least squares
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, values, rcond=None)
        return X @ coeffs
    except:
        return values

def svensson_smoother(tenors: np.ndarray, values: np.ndarray, 
                     tau1: float = 2.0, tau2: float = 5.0) -> np.ndarray:
    """
    Svensson model (extension of Nelson-Siegel with 4 factors).
    Adds an additional hump term for better fit at medium tenors.
    """
    n = len(tenors)
    safe_tenors = np.maximum(tenors, 0.01)
    
    # Create basis functions
    ones = np.ones(n)
    factor1 = (1 - np.exp(-safe_tenors/tau1)) / (safe_tenors/tau1)
    factor2 = factor1 - np.exp(-safe_tenors/tau1)
    factor3 = (1 - np.exp(-safe_tenors/tau2)) / (safe_tenors/tau2) - np.exp(-safe_tenors/tau2)
    
    # Stack basis
    X = np.column_stack([ones, factor1, factor2, factor3])
    
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, values, rcond=None)
        return X @ coeffs
    except:
        return nelson_siegel_smoother(tenors, values, tau1)

def polynomial_ridge_smoother(tenors: np.ndarray, values: np.ndarray, 
                             degree: int = 5, alpha: float = 0.1) -> np.ndarray:
    """
    Polynomial smoothing with ridge regularization to prevent overfitting.
    """
    # Create polynomial features
    X = np.column_stack([tenors**i for i in range(degree + 1)])
    
    # Add ridge regularization
    XtX = X.T @ X
    XtX_ridge = XtX + alpha * np.eye(degree + 1)
    Xty = X.T @ values
    
    try:
        coeffs = np.linalg.solve(XtX_ridge, Xty)
        return X @ coeffs
    except:
        # Fall back to lower degree
        if degree > 2:
            return polynomial_ridge_smoother(tenors, values, degree-1, alpha)
        return values

def local_regression_smoother(tenors: np.ndarray, values: np.ndarray, 
                             bandwidth: float = 5.0) -> np.ndarray:
    """
    Local regression (LOESS-like) smoothing.
    Fits local polynomials with Gaussian weights.
    """
    n = len(tenors)
    smoothed = np.zeros(n)
    
    for i, t in enumerate(tenors):
        # Calculate weights based on distance
        distances = np.abs(tenors - t)
        weights = np.exp(-(distances**2) / (2 * bandwidth**2))
        
        # Weighted polynomial regression (degree 2)
        X = np.column_stack([np.ones(n), tenors, tenors**2])
        W = np.diag(weights)
        
        try:
            # Weighted least squares
            coeffs = np.linalg.solve(X.T @ W @ X, X.T @ W @ values)
            # Evaluate at current point
            smoothed[i] = coeffs[0] + coeffs[1] * t + coeffs[2] * t**2
        except:
            smoothed[i] = np.average(values, weights=weights)
    
    return smoothed

# ============================================================================
# EXTENDED TENOR SMOOTHER
# ============================================================================

class TenorSmootherExtended(TenorSmoother):
    """Extended TenorSmoother with custom model support."""
    
    def custom_model(self, tenors: np.ndarray, values: np.ndarray,
                    model_func: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> np.ndarray:
        """Apply a user-supplied custom model function."""
        # Handle NaN values
        mask = ~np.isnan(values)
        if np.sum(mask) < 2:
            return values
        
        valid_tenors = tenors[mask]
        valid_values = values[mask]
        
        # Handle duplicates
        unique_tenors, inverse_indices = np.unique(valid_tenors, return_inverse=True)
        if len(unique_tenors) < len(valid_tenors):
            unique_values = np.zeros(len(unique_tenors))
            for i in range(len(unique_tenors)):
                unique_values[i] = np.mean(valid_values[inverse_indices == i])
            
            # Apply model to unique points
            try:
                smoothed_unique = model_func(unique_tenors, unique_values)
                # Map back to all points
                result = np.full_like(values, np.nan)
                result[mask] = smoothed_unique[inverse_indices]
                return result
            except Exception as e:
                print(f"Custom model failed: {e}")
                return values
        else:
            # No duplicates
            try:
                smoothed = model_func(valid_tenors, valid_values)
                result = np.full_like(values, np.nan)
                result[mask] = smoothed
                return result
            except Exception as e:
                print(f"Custom model failed: {e}")
                return values
    
    def smooth(self, tenors: np.ndarray, values: np.ndarray,
               method: str, **params) -> np.ndarray:
        """Apply specified smoothing method including custom."""
        if method == 'custom':
            if 'model_func' not in params:
                raise ValueError("Custom method requires 'model_func' parameter")
            return self.custom_model(tenors, values, params['model_func'])
        else:
            return super().smooth(tenors, values, method, **params)

# Update the main framework to use extended smoother
class YieldSmoothingFrameworkExtended(YieldSmoothingFramework):
    """Extended framework with custom model support."""
    
    def __init__(self, knot_positions: List[float] = None):
        self.tenor_smoother = TenorSmootherExtended(knot_positions)
        self.time_smoother = TimeSmoother()

# ============================================================================
# LARGE DATASET CREATION AND VISUALIZATION
# ============================================================================

def create_large_multicountry_dataset() -> pd.DataFrame:
    """
    Create a large dataset with 13 countries, 14 tenors, ~1500 rows per country.
    Total: ~270,000 data points
    """
    np.random.seed(42)
    
    # Configuration
    dates = pd.date_range('2020-01-01', periods=115, freq='B')  # ~6 months of business days
    countries = ['US', 'UK', 'DE', 'FR', 'JP', 'CN', 'AU', 'CA', 'CH', 'SE', 'NO', 'NZ', 'SG']
    tenors = np.array([0.25, 0.5, 1, 2, 3, 4, 5, 7, 10, 12, 15, 20, 25, 30])
    
    print(f"Creating dataset: {len(dates)} dates × {len(countries)} countries × {len(tenors)} tenors")
    print(f"Total data points: {len(dates) * len(countries) * len(tenors):,}")
    
    data = []
    
    # Country-specific characteristics
    country_params = {
        'US': {'base': 0.03, 'vol': 0.003},
        'UK': {'base': 0.025, 'vol': 0.004},
        'DE': {'base': 0.01, 'vol': 0.002},
        'FR': {'base': 0.012, 'vol': 0.0025},
        'JP': {'base': -0.001, 'vol': 0.001},
        'CN': {'base': 0.035, 'vol': 0.005},
        'AU': {'base': 0.028, 'vol': 0.0035},
        'CA': {'base': 0.027, 'vol': 0.003},
        'CH': {'base': -0.005, 'vol': 0.0015},
        'SE': {'base': 0.008, 'vol': 0.002},
        'NO': {'base': 0.015, 'vol': 0.0025},
        'NZ': {'base': 0.032, 'vol': 0.004},
        'SG': {'base': 0.02, 'vol': 0.003}
    }
    
    for date_idx, date in enumerate(dates):
        # Global factor (affects all countries)
        global_factor = 0.002 * np.sin(date_idx / 20)
        
        for country in countries:
            params = country_params[country]
            
            # Country-specific yield curve
            base_curve = params['base'] + 0.02 * (1 - np.exp(-tenors/7))
            
            # Add country-specific trend
            country_trend = 0.001 * np.sin(date_idx / 15 + hash(country) % 10)
            
            # Add noise
            noise = np.random.randn(len(tenors)) * params['vol']
            
            # Tenor-specific noise (more noise at long end)
            tenor_noise = np.random.randn(len(tenors)) * params['vol'] * np.sqrt(tenors/10)
            
            # Combine all factors
            values = base_curve + global_factor + country_trend + noise + tenor_noise
            
            # Add occasional jumps (5% chance)
            if np.random.random() < 0.05:
                values += np.random.randn() * params['vol'] * 3
            
            for tenor, value in zip(tenors, values):
                data.append({
                    'date': date,
                    'country': country,
                    'tenor': tenor,
                    'yield_change': value
                })
    
    df = pd.DataFrame(data)
    df = df.set_index(['date', 'country', 'tenor'])
    return df

def visualize_time_smoothing_comparison(df: pd.DataFrame):
    """
    Create comprehensive visualization comparing all time smoothing methods.
    """
    framework = YieldSmoothingFrameworkExtended()
    
    # Select sample country and tenor for time series analysis
    sample_country = 'US'
    sample_tenor = 5.0
    
    # Extract time series
    mask = (df.index.get_level_values(1) == sample_country) & \
           (df.index.get_level_values(2) == sample_tenor)
    time_series_data = df.loc[mask].sort_index()
    dates = time_series_data.index.get_level_values(0)
    original_values = time_series_data.values.flatten()
    
    # Define all time smoothing methods with different parameters
    time_methods = {
        'Original': ('none', {}),
        'EWMA (λ=0.94)': ('ewma', {'decay': 0.94}),
        'EWMA (λ=0.90)': ('ewma', {'decay': 0.90}),
        'Double EWMA': ('double_ewma', {'decay': 0.94}),
        'Kalman (Low Noise)': ('kalman', {'process_noise': 0.001, 'measurement_noise': 0.01}),
        'Kalman (High Noise)': ('kalman', {'process_noise': 0.01, 'measurement_noise': 0.1}),
        'Gaussian Process (LS=10)': ('gp', {'length_scale': 10.0, 'noise_level': 0.1}),
        'Gaussian Process (LS=20)': ('gp', {'length_scale': 20.0, 'noise_level': 0.05}),
        'Moving Average (W=10)': ('ma', {'window': 10}),
        'Moving Average (W=20)': ('ma', {'window': 20}),
        'Savitzky-Golay': ('savgol', {'window': 11, 'polyorder': 3}),
        'Hodrick-Prescott': ('hp', {'lamb': 1600})
    }
    
    # Apply all smoothing methods
    smoothed_results = {}
    print("\nApplying time smoothing methods:")
    for name, (method, params) in time_methods.items():
        print(f"  Processing: {name}")
        smoothed = TimeSmoother.smooth(original_values.copy(), method, **params)
        smoothed_results[name] = smoothed
    
    # Create visualization
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    fig.suptitle('Comprehensive Time-Series Smoothing Comparison', fontsize=16, y=1.02)
    
    # Plot each method
    method_groups = [
        ['Original', 'EWMA (λ=0.94)', 'EWMA (λ=0.90)', 'Double EWMA'],
        ['Original', 'Kalman (Low Noise)', 'Kalman (High Noise)'],
        ['Original', 'Gaussian Process (LS=10)', 'Gaussian Process (LS=20)'],
        ['Original', 'Moving Average (W=10)', 'Moving Average (W=20)'],
        ['Original', 'Savitzky-Golay', 'Hodrick-Prescott']
    ]
    
    # First 5 subplots: method comparisons
    for idx, method_group in enumerate(method_groups):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        for method_name in method_group:
            values = smoothed_results[method_name]
            alpha = 0.4 if method_name == 'Original' else 0.8
            linewidth = 1 if method_name == 'Original' else 1.5
            ax.plot(dates, values * 100, label=method_name, alpha=alpha, linewidth=linewidth)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Yield (%)')
        ax.set_title(f'{method_group[1].split("(")[0].strip()} Methods')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Subplot 6: Lag analysis
    ax = axes[1, 2]
    lags_to_test = ['EWMA (λ=0.94)', 'Kalman (Low Noise)', 'Gaussian Process (LS=10)']
    
    for method_name in lags_to_test:
        values = smoothed_results[method_name]
        # Calculate lag correlation with original
        max_lag = 20
        correlations = []
        for lag in range(max_lag):
            if lag == 0:
                corr = np.corrcoef(original_values[:-1] if len(original_values) > 1 else original_values, 
                                  values[:-1] if len(values) > 1 else values)[0, 1]
            else:
                corr = np.corrcoef(original_values[:-lag] if lag < len(original_values) else original_values,
                                  values[lag:] if lag < len(values) else values)[0, 1]
            correlations.append(corr)
        
        ax.plot(range(max_lag), correlations, marker='o', label=method_name, markersize=4)
    
    ax.set_xlabel('Lag (periods)')
    ax.set_ylabel('Correlation')
    ax.set_title('Lag Analysis (Correlation with Original)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Subplot 7: Smoothness comparison (bar chart)
    ax = axes[2, 0]
    
    smoothness_data = []
    for name, values in smoothed_results.items():
        if len(values) > 1:
            # Calculate first difference standard deviation
            volatility = np.std(np.diff(values))
            smoothness_data.append((name, volatility * 10000))
    
    names, volatilities = zip(*smoothness_data[:8])  # First 8 methods
    colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
    bars = ax.bar(range(len(names)), volatilities, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Volatility (scaled)')
    ax.set_title('Smoothness Comparison (Lower = Smoother)')
    
    # Subplot 8: Response to jumps
    ax = axes[2, 1]
    
    # Create synthetic jump
    jump_series = original_values.copy()
    jump_idx = len(jump_series) // 2
    jump_series[jump_idx] += 0.02  # Add 200bp jump
    
    # Apply smoothing to jump series
    jump_methods = ['EWMA (λ=0.94)', 'Kalman (Low Noise)', 'Moving Average (W=10)']
    
    for method_name in jump_methods:
        method, params = time_methods[method_name]
        smoothed_jump = TimeSmoother.smooth(jump_series.copy(), method, **params)
        
        # Plot around jump
        plot_range = slice(jump_idx - 20, jump_idx + 30)
        plot_dates = dates[plot_range]
        plot_values = smoothed_jump[plot_range]
        
        ax.plot(range(len(plot_values)), plot_values * 100, 
               label=method_name, marker='o', markersize=3)
    
    ax.axvline(x=20, color='red', linestyle='--', alpha=0.5, label='Jump')
    ax.set_xlabel('Time (relative to jump)')
    ax.set_ylabel('Yield (%)')
    ax.set_title('Response to 200bp Jump')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Subplot 9: Computational metrics
    ax = axes[2, 2]
    ax.axis('off')
    
    metrics_text = "Performance Metrics:\n" + "="*30 + "\n\n"
    
    for name in ['EWMA (λ=0.94)', 'Kalman (Low Noise)', 'Gaussian Process (LS=10)', 
                 'Moving Average (W=10)', 'Savitzky-Golay']:
        values = smoothed_results[name]
        
        # Calculate metrics
        rmse = np.sqrt(np.mean((values - original_values) ** 2))
        max_dev = np.max(np.abs(values - original_values))
        smooth_metric = np.std(np.diff(values)) if len(values) > 1 else 0
        
        metrics_text += f"{name}:\n"
        metrics_text += f"  RMSE: {rmse*100:.4f}%\n"
        metrics_text += f"  Max Dev: {max_dev*100:.4f}%\n"
        metrics_text += f"  Smoothness: {smooth_metric*10000:.2f}\n\n"
    
    ax.text(0.1, 0.5, metrics_text, transform=ax.transAxes, 
           fontsize=9, verticalalignment='center', fontfamily='monospace')
    
    # Subplots 10-12: Combined tenor and time smoothing
    combined_configs = [
        ('P-Spline + EWMA', 'pspline', 'ewma', 
         {'smoothing_param': 2.0}, {'decay': 0.94}),
        ('Nelson-Siegel + Kalman', 'custom', 'kalman',
         {'model_func': nelson_siegel_smoother}, {'process_noise': 0.01, 'measurement_noise': 0.1}),
        ('Bernstein + GP', 'bernstein', 'gp',
         {'degree': 5}, {'length_scale': 15.0, 'noise_level': 0.1})
    ]
    
    for idx, (name, tenor_method, time_method, tenor_params, time_params) in enumerate(combined_configs):
        ax = axes[3, idx]
        
        # Apply combined smoothing
        config = SmoothingConfig(
            tenor_method=tenor_method,
            time_method=time_method,
            tenor_params=tenor_params,
            time_params=time_params
        )
        
        smoothed_df = framework.smooth(df, config)
        combined_series = smoothed_df.loc[mask].sort_index()
        combined_values = combined_series.values.flatten()
        
        # Plot original vs combined smoothing
        ax.plot(dates, original_values * 100, label='Original', alpha=0.4, linewidth=1)
        ax.plot(dates, combined_values * 100, label=name, alpha=0.8, linewidth=1.5)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Yield (%)')
        ax.set_title(f'Combined: {name}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return smoothed_results

def compare_all_smoothing_methods(df: pd.DataFrame):
    """
    Comprehensive comparison of all smoothing methods with statistics.
    """
    framework = YieldSmoothingFrameworkExtended()
    
    # Get sample data
    sample_date = df.index.get_level_values(0).unique()[60]
    sample_country = 'US'
    sample_tenor = 5.0
    
    print("\n" + "="*80)
    print("COMPREHENSIVE SMOOTHING METHOD COMPARISON")
    print("="*80)
    
    # Test all combinations
    all_configs = {
        # Tenor only
        'P-Spline Only': SmoothingConfig(
            tenor_method='pspline', time_method='none',
            tenor_params={'smoothing_param': 2.0}
        ),
        'Bernstein Only': SmoothingConfig(
            tenor_method='bernstein', time_method='none',
            tenor_params={'degree': 5}
        ),
        'Nelson-Siegel Only': SmoothingConfig(
            tenor_method='custom', time_method='none',
            tenor_params={'model_func': nelson_siegel_smoother}
        ),
        
        # Time only
        'EWMA Only': SmoothingConfig(
            tenor_method='none', time_method='ewma',
            time_params={'decay': 0.94}
        ),
        'Kalman Only': SmoothingConfig(
            tenor_method='none', time_method='kalman',
            time_params={'process_noise': 0.01, 'measurement_noise': 0.1}
        ),
        'GP Only': SmoothingConfig(
            tenor_method='none', time_method='gp',
            time_params={'length_scale': 15.0, 'noise_level': 0.1}
        ),
        
        # Combined
        'P-Spline + EWMA': SmoothingConfig(
            tenor_method='pspline', time_method='ewma',
            tenor_params={'smoothing_param': 2.0},
            time_params={'decay': 0.94}
        ),
        'Nelson-Siegel + Kalman': SmoothingConfig(
            tenor_method='custom', time_method='kalman',
            tenor_params={'model_func': nelson_siegel_smoother},
            time_params={'process_noise': 0.01, 'measurement_noise': 0.1}
        ),
        'Bernstein + GP': SmoothingConfig(
            tenor_method='bernstein', time_method='gp',
            tenor_params={'degree': 5},
            time_params={'length_scale': 15.0, 'noise_level': 0.1}
        ),
        'Svensson + Double EWMA': SmoothingConfig(
            tenor_method='custom', time_method='double_ewma',
            tenor_params={'model_func': svensson_smoother},
            time_params={'decay': 0.92}
        )
    }
    
    # Apply all configs and calculate metrics
    results = []
    original_df = framework.smooth(df, SmoothingConfig(tenor_method='none', time_method='none'))
    
    for name, config in all_configs.items():
        print(f"Processing: {name}")
        smoothed_df = framework.smooth(df, config)
        
        # Calculate metrics for tenor dimension (single date)
        orig_tenor = original_df.loc[(sample_date, sample_country)].values.flatten()
        smooth_tenor = smoothed_df.loc[(sample_date, sample_country)].values.flatten()
        
        tenor_roughness = np.sqrt(np.mean(np.diff(smooth_tenor, n=2) ** 2)) if len(smooth_tenor) > 2 else 0
        tenor_rmse = np.sqrt(np.mean((smooth_tenor - orig_tenor) ** 2))
        
        # Calculate metrics for time dimension (single tenor)
        mask = (original_df.index.get_level_values(1) == sample_country) & \
               (original_df.index.get_level_values(2) == sample_tenor)
        
        orig_time = original_df.loc[mask].values.flatten()
        smooth_time = smoothed_df.loc[mask].values.flatten()
        
        time_volatility = np.std(np.diff(smooth_time)) if len(smooth_time) > 1 else 0
        time_rmse = np.sqrt(np.mean((smooth_time - orig_time) ** 2))
        
        results.append({
            'Method': name,
            'Tenor Roughness': tenor_roughness * 10000,
            'Tenor RMSE': tenor_rmse * 100,
            'Time Volatility': time_volatility * 10000,
            'Time RMSE': time_rmse * 100
        })
    
    # Print results table
    results_df = pd.DataFrame(results)
    print("\n" + "-"*80)
    print("SMOOTHING METRICS SUMMARY")
    print("-"*80)
    print(results_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    
    return results_df
    """
    Create comprehensive visualization for multi-country dataset.
    Shows effects across different countries and smoothing methods.
    """
    framework = YieldSmoothingFrameworkExtended()
    
    # Select sample date and countries for visualization
    sample_date = df.index.get_level_values(0).unique()[60]  # Middle of dataset
    sample_countries = ['US', 'DE', 'JP', 'CN']  # Diverse set
    
    # Define smoothing configurations including custom
    configs = {
        'Original': SmoothingConfig(
            tenor_method='none', 
            time_method='none'
        ),
        
        'P-Spline': SmoothingConfig(
            tenor_method='pspline',
            time_method='none',
            tenor_params={'smoothing_param': 2.0}
        ),
        
        'Nelson-Siegel': SmoothingConfig(
            tenor_method='custom',
            time_method='none',
            tenor_params={'model_func': nelson_siegel_smoother}
        ),
        
        'Svensson': SmoothingConfig(
            tenor_method='custom',
            time_method='none',
            tenor_params={'model_func': svensson_smoother}
        ),
        
        'Local Regression': SmoothingConfig(
            tenor_method='custom',
            time_method='none',
            tenor_params={'model_func': lambda t, v: local_regression_smoother(t, v, bandwidth=3.0)}
        ),
        
        'Combined (P-Spline + EWMA)': SmoothingConfig(
            tenor_method='pspline',
            time_method='ewma',
            tenor_params={'smoothing_param': 1.5},
            time_params={'decay': 0.94}
        )
    }
    
    # Apply smoothing (this will take a moment with 270k points)
    print("\nApplying smoothing methods to large dataset...")
    results = {}
    for name, config in configs.items():
        print(f"  Processing: {name}...")
        results[name] = framework.smooth(df, config)
    
    # Create visualization
    fig = plt.figure(figsize=(18, 12))
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.25)
    
    # Plot 1: Compare methods for one country
    ax1 = fig.add_subplot(gs[0, :])
    country = 'US'
    colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))
    
    for (name, result_df), color in zip(results.items(), colors):
        data = result_df.loc[(sample_date, country)]
        tenors = data.index.get_level_values(0).values
        values = data.values.flatten()
        ax1.plot(tenors, values * 100, marker='o', label=name, alpha=0.7, color=color)
    
    ax1.set_xlabel('Tenor (years)')
    ax1.set_ylabel('Yield (%)')
    ax1.set_title(f'Smoothing Methods Comparison ({country}, {sample_date.date()})')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plots 2-5: Different countries with same method
    for idx, country in enumerate(sample_countries):
        row = 1 + idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])
        
        # Show original vs smoothed
        original_data = results['Original'].loc[(sample_date, country)]
        smoothed_data = results['Nelson-Siegel'].loc[(sample_date, country)]
        
        tenors = original_data.index.get_level_values(0).values
        orig_values = original_data.values.flatten()
        smooth_values = smoothed_data.values.flatten()
        
        ax.plot(tenors, orig_values * 100, 'o-', label='Original', alpha=0.5)
        ax.plot(tenors, smooth_values * 100, 's-', label='Nelson-Siegel', alpha=0.8)
        
        ax.set_xlabel('Tenor (years)')
        ax.set_ylabel('Yield (%)')
        ax.set_title(f'{country} Yield Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 6: Smoothness comparison across countries
    ax6 = fig.add_subplot(gs[1, 2])
    
    smoothness_data = []
    for country in df.index.get_level_values(1).unique()[:8]:  # First 8 countries
        for method_name in ['Original', 'P-Spline', 'Nelson-Siegel']:
            data = results[method_name].loc[(sample_date, country)]
            values = data.values.flatten()
            if len(values) > 2:
                roughness = np.sqrt(np.mean(np.diff(values, n=2) ** 2))
                smoothness_data.append({
                    'Country': country,
                    'Method': method_name,
                    'Roughness': roughness * 10000
                })
    
    smoothness_df = pd.DataFrame(smoothness_data)
    pivot_df = smoothness_df.pivot(index='Country', columns='Method', values='Roughness')
    pivot_df.plot(kind='bar', ax=ax6)
    ax6.set_ylabel('Roughness (scaled)')
    ax6.set_title('Roughness by Country')
    ax6.tick_params(axis='x', rotation=45)
    ax6.legend(title='Method', loc='best')
    
    # Plot 7: Time series smoothing example
    ax7 = fig.add_subplot(gs[2, 2])
    
    country = 'UK'
    tenor = 5.0
    
    for method_name in ['Original', 'Combined (P-Spline + EWMA)']:
        result_df = results[method_name]
        mask = (result_df.index.get_level_values(1) == country) & \
               (result_df.index.get_level_values(2) == tenor)
        time_series = result_df.loc[mask].sort_index()
        dates = time_series.index.get_level_values(0)
        values = time_series.values.flatten()
        
        label = 'Original' if 'Original' in method_name else 'Smoothed (Time + Tenor)'
        alpha = 0.5 if 'Original' in method_name else 0.8
        ax7.plot(dates, values * 100, label=label, alpha=alpha)
    
    ax7.set_xlabel('Date')
    ax7.set_ylabel('Yield (%)')
    ax7.set_title(f'Time Smoothing ({country}, {tenor}Y)')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    plt.suptitle('Multi-Country Yield Curve Smoothing Analysis', fontsize=14, y=0.98)
    plt.show()
    
    return results

def print_multicountry_statistics(df: pd.DataFrame, results: Dict[str, pd.DataFrame]):
    """Print comprehensive statistics for the multi-country dataset."""
    
    print("\n" + "="*80)
    print("MULTI-COUNTRY SMOOTHING STATISTICS")
    print("="*80)
    
    sample_date = df.index.get_level_values(0).unique()[60]
    
    # Overall statistics
    print("\nDataset Summary:")
    print(f"  Total rows: {len(df):,}")
    print(f"  Countries: {df.index.get_level_values(1).nunique()}")
    print(f"  Tenors: {df.index.get_level_values(2).nunique()}")
    print(f"  Date range: {df.index.get_level_values(0).min()} to {df.index.get_level_values(0).max()}")
    
    # Method comparison
    print("\n" + "-"*80)
    print("METHOD COMPARISON (averaged across all countries on sample date)")
    print("-"*80)
    
    method_stats = []
    for method_name, result_df in results.items():
        all_roughness = []
        all_rmse = []
        
        for country in df.index.get_level_values(1).unique():
            try:
                # Get data for this country
                orig_data = results['Original'].loc[(sample_date, country)].values.flatten()
                smooth_data = result_df.loc[(sample_date, country)].values.flatten()
                
                # Calculate roughness
                if len(smooth_data) > 2:
                    roughness = np.sqrt(np.mean(np.diff(smooth_data, n=2) ** 2))
                    all_roughness.append(roughness)
                
                # Calculate RMSE from original
                if method_name != 'Original':
                    rmse = np.sqrt(np.mean((smooth_data - orig_data) ** 2))
                    all_rmse.append(rmse)
            except:
                continue
        
        avg_roughness = np.mean(all_roughness) if all_roughness else 0
        avg_rmse = np.mean(all_rmse) if all_rmse else 0
        
        method_stats.append({
            'Method': method_name,
            'Avg Roughness': avg_roughness * 10000,
            'Avg RMSE': avg_rmse * 100
        })
    
    stats_df = pd.DataFrame(method_stats)
    print("\n", stats_df.to_string(index=False))
    
    # Country-specific effects
    print("\n" + "-"*80)
    print("SMOOTHING EFFECT BY COUNTRY (Nelson-Siegel vs Original)")
    print("-"*80)
    
    country_effects = []
    for country in df.index.get_level_values(1).unique():
        orig_data = results['Original'].loc[(sample_date, country)].values.flatten()
        smooth_data = results['Nelson-Siegel'].loc[(sample_date, country)].values.flatten()
        
        orig_rough = np.sqrt(np.mean(np.diff(orig_data, n=2) ** 2))
        smooth_rough = np.sqrt(np.mean(np.diff(smooth_data, n=2) ** 2))
        reduction = (1 - smooth_rough/orig_rough) * 100 if orig_rough > 0 else 0
        
        country_effects.append({
            'Country': country,
            'Original Rough': orig_rough * 10000,
            'Smoothed Rough': smooth_rough * 10000,
            'Reduction %': reduction
        })
    
    effects_df = pd.DataFrame(country_effects)
    print("\n", effects_df.to_string(index=False))
