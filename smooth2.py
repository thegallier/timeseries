import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
from scipy.interpolate import UnivariateSpline, PchipInterpolator
from scipy.special import comb
from functools import partial


@dataclass
class SmoothingConfig:
    """Configuration for smoothing methods."""
    tenor_method: str = 'pspline'  # 'pspline', 'cubic_spline', 'monotonic', 'bernstein', 'none'
    time_method: str = 'ewma'  # 'ewma', 'kalman', 'gp', 'none'
    tenor_params: Dict[str, Any] = None
    time_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tenor_params is None:
            self.tenor_params = {}
        if self.time_params is None:
            self.time_params = {}


class TenorSmoother:
    """Handles cross-sectional smoothing across tenors."""
    
    def __init__(self, knot_positions: List[float] = None):
        self.knot_positions = knot_positions or [2, 5, 10, 30]
    
    def bernstein_basis(self, x: np.ndarray, n: int, i: int) -> np.ndarray:
        """Compute Bernstein basis polynomial."""
        # Normalize x to [0, 1]
        x_min, x_max = np.min(x), np.max(x)
        if x_max - x_min < 1e-10:
            return np.ones_like(x)
        t = (x - x_min) / (x_max - x_min)
        return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
    
    def bernstein_polynomial(self, tenors: np.ndarray, values: np.ndarray,
                           degree: int = 5) -> np.ndarray:
        """Fit Bernstein polynomial to tenor curve."""
        if np.any(np.isnan(values)):
            mask = ~np.isnan(values)
            if np.sum(mask) < 2:
                return values
            # For Bernstein, we need the actual valid points
            valid_tenors = tenors[mask]
            valid_values = values[mask]
        else:
            valid_tenors = tenors
            valid_values = values
            mask = np.ones(len(tenors), dtype=bool)
        
        # Need at least degree+1 points
        if len(valid_values) < degree + 1:
            # Just interpolate
            return np.interp(tenors, valid_tenors, valid_values)
        
        # Normalize tenors to [0, 1]
        t_min, t_max = np.min(valid_tenors), np.max(valid_tenors)
        if t_max - t_min < 1e-10:
            return values  # All same tenor
        
        # Normalize for Bernstein basis
        t_norm_valid = (valid_tenors - t_min) / (t_max - t_min)
        t_norm_all = (tenors - t_min) / (t_max - t_min)
        
        # Create Bernstein basis matrix for valid points
        n = degree
        B_valid = np.column_stack([
            comb(n, i) * (t_norm_valid ** i) * ((1 - t_norm_valid) ** (n - i))
            for i in range(n + 1)
        ])
        
        # Fit coefficients using least squares with regularization
        # Add small regularization to prevent overfitting
        regularization = 1e-8 * np.eye(n + 1)
        try:
            # Solve (B'B + reg)c = B'y
            coeffs = np.linalg.solve(
                B_valid.T @ B_valid + regularization,
                B_valid.T @ valid_values
            )
        except:
            # If that fails, use pseudoinverse
            coeffs, _, _, _ = np.linalg.lstsq(B_valid, valid_values, rcond=None)
        
        # Create basis for all points
        B_all = np.column_stack([
            comb(n, i) * (t_norm_all ** i) * ((1 - t_norm_all) ** (n - i))
            for i in range(n + 1)
        ])
        
        # Reconstruct smooth curve
        result = B_all @ coeffs
        
        # Diagnostic check
        if np.ptp(result) < np.ptp(valid_values) * 0.01:  # Lost 99% of variation
            print(f"Warning: Bernstein over-smoothing (degree={degree}), falling back to interpolation")
            return np.interp(tenors, valid_tenors, valid_values)
        
        return result
    
    def pspline(self, tenors: np.ndarray, values: np.ndarray, 
                smoothing_param: float = None, degree: int = 3) -> np.ndarray:
        """P-spline smoothing with automatic smoothing parameter selection."""
        # Check for all NaN
        if np.all(np.isnan(values)):
            return values
        
        # Handle missing values
        mask = ~np.isnan(values)
        n_valid = np.sum(mask)
        
        # Need at least degree+1 points for spline
        if n_valid < max(degree + 1, 2):
            if n_valid < 2:
                return values
            # Fall back to linear interpolation
            return np.interp(tenors, tenors[mask], values[mask])
        
        # Get valid points
        valid_tenors = tenors[mask]
        valid_values = values[mask]
        
        try:
            # Check if tenors are unique and sorted
            if len(np.unique(valid_tenors)) < len(valid_tenors):
                # Average duplicate tenor values
                unique_tenors, indices = np.unique(valid_tenors, return_inverse=True)
                unique_values = np.zeros(len(unique_tenors))
                for i, tenor in enumerate(unique_tenors):
                    unique_values[i] = np.mean(valid_values[valid_tenors == tenor])
                valid_tenors = unique_tenors
                valid_values = unique_values
            
            # Ensure we have enough unique points
            if len(valid_tenors) < degree + 1:
                degree = max(1, len(valid_tenors) - 1)
            
            # Create spline
            if smoothing_param is None:
                # Use interpolating spline by default - no smoothing
                spline = UnivariateSpline(valid_tenors, valid_values, k=degree, s=0)
            elif smoothing_param == 0:
                # Explicitly interpolating spline
                spline = UnivariateSpline(valid_tenors, valid_values, k=degree, s=0)
            else:
                # For data in the range -0.08 to 0.08, we need much larger smoothing values
                # The 's' parameter is the sum of squared residuals that's acceptable
                # For your data scale, this should be in the range of 0.001 to 0.1
                
                # Scale by the sum of squared values to make parameter intuitive
                data_scale = np.sum(valid_values**2)
                
                # smoothing_param is now a fraction of acceptable error (0 to 1)
                # 0.01 means we accept 1% of the total squared variation as error
                scaled_smooth = smoothing_param * data_scale
                
                spline = UnivariateSpline(valid_tenors, valid_values, k=degree, s=scaled_smooth)
            
            # Evaluate at all tenor points
            result = spline(tenors)
            
            # Diagnostic check - if all values are nearly identical, something went wrong
            if np.ptp(result) < np.ptp(valid_values) * 0.1:  # Lost 90% of variation
                print(f"Warning: Over-smoothing detected (range reduced from {np.ptp(valid_values):.6f} to {np.ptp(result):.6f})")
                print(f"  Data scale: {data_scale:.6f}, Smoothing param: {smoothing_param}, Scaled: {scaled_smooth:.6f}")
                # Fall back to interpolation
                return np.interp(tenors, valid_tenors, valid_values)
            
            return result
            
        except Exception as e:
            # If spline fitting fails completely, fall back to linear interpolation
            print(f"Warning: P-spline fitting failed ({str(e)}), using linear interpolation")
            return np.interp(tenors, valid_tenors, valid_values)
    
    def cubic_spline(self, tenors: np.ndarray, values: np.ndarray,
                     tension: float = 0.0) -> np.ndarray:
        """Standard cubic spline interpolation."""
        # Check for all NaN
        if np.all(np.isnan(values)):
            return values
            
        mask = ~np.isnan(values)
        n_valid = np.sum(mask)
        
        if n_valid < 2:
            return values
        
        if n_valid < 4:
            # Need at least 4 points for cubic spline, fall back to linear
            return np.interp(tenors, tenors[mask], values[mask])
        
        try:
            valid_tenors = tenors[mask]
            valid_values = values[mask]
            
            # Handle duplicate tenors
            if len(np.unique(valid_tenors)) < len(valid_tenors):
                unique_tenors, indices = np.unique(valid_tenors, return_inverse=True)
                unique_values = np.zeros(len(unique_tenors))
                for i, tenor in enumerate(unique_tenors):
                    unique_values[i] = np.mean(valid_values[valid_tenors == tenor])
                valid_tenors = unique_tenors
                valid_values = unique_values
            
            spline = UnivariateSpline(valid_tenors, valid_values, k=3, s=tension)
            result = spline(tenors)
            
            # Check for NaN in result
            if np.any(np.isnan(result)):
                nan_mask = np.isnan(result)
                result[nan_mask] = np.interp(tenors[nan_mask], valid_tenors, valid_values)
            
            return result
            
        except Exception as e:
            print(f"Warning: Cubic spline failed ({str(e)}), using linear interpolation")
            return np.interp(tenors, tenors[mask], values[mask])
    
    def monotonic_spline(self, tenors: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Monotonic spline using PCHIP interpolation."""
        if np.any(np.isnan(values)):
            mask = ~np.isnan(values)
            if np.sum(mask) < 2:
                return values
            # For monotonic, we need to handle NaNs carefully
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


class TimeSmoother:
    """Handles time-series smoothing - all methods are causal (backward-looking only)."""
    
    @staticmethod
    def ewma(values: np.ndarray, decay: float = 0.94) -> np.ndarray:
        """
        Exponentially weighted moving average - causal filter.
        Only uses past values to compute current smoothed value.
        """
        n = len(values)
        smoothed = np.zeros(n)
        
        # Initialize with first non-NaN value
        init_idx = 0
        while init_idx < n and np.isnan(values[init_idx]):
            init_idx += 1
        
        if init_idx >= n:
            return values  # All NaN
        
        smoothed[init_idx] = values[init_idx]
        
        # Forward pass - each point only depends on previous points
        for i in range(init_idx + 1, n):
            if np.isnan(values[i]):
                smoothed[i] = smoothed[i - 1]  # Carry forward last value
            else:
                smoothed[i] = decay * smoothed[i - 1] + (1 - decay) * values[i]
        
        # Fill initial NaNs
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
        
        # Initialize
        x = values[init_idx]
        P = 1.0
        
        # Fill initial values
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
                # No update for NaN - just propagate prediction
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
            
            # Limit history to recent past for computational efficiency
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
                    # Weight by relative noise level
                    obs_weight = 1 / (1 + noise_level)
                    smoothed[i] = obs_weight * values[i] + (1 - obs_weight) * smoothed[i]
                    
            except np.linalg.LinAlgError:
                # Fallback to last value if numerical issues
                smoothed[i] = smoothed[i - 1]
        
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
        elif method == 'gp':
            return TimeSmoother.gaussian_process(values, **params)
        else:
            raise ValueError(f"Unknown time smoothing method: {method}")


class YieldSmoothingFramework:
    """Main framework for smoothing yield curve data."""
    
    def __init__(self, knot_positions: List[float] = None):
        self.tenor_smoother = TenorSmoother(knot_positions)
        self.time_smoother = TimeSmoother()
    
    def smooth(self, df: pd.DataFrame, config: SmoothingConfig) -> pd.DataFrame:
        """
        Apply smoothing to yield curve data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with MultiIndex (date, country, tenor) or similar structure
        config : SmoothingConfig
            Configuration specifying methods and parameters
        
        Returns:
        --------
        pd.DataFrame
            Smoothed dataframe with same structure as input
        """
        # Copy dataframe to avoid modifying original
        result = df.copy()
        
        # Ensure index is sorted for performance
        if isinstance(result.index, pd.MultiIndex):
            result = result.sort_index()
        
        # Detect dataframe structure
        if isinstance(df.index, pd.MultiIndex):
            self._smooth_multiindex(result, config)
        else:
            self._smooth_wide_format(result, config)
        
        return result
    
    def _smooth_multiindex(self, df: pd.DataFrame, config: SmoothingConfig):
        """Handle MultiIndex format (date, country, tenor)."""
        # Get unique dates and countries
        dates = df.index.get_level_values(0).unique()
        countries = df.index.get_level_values(1).unique()
        
        # First: smooth across tenors for each country-date
        if config.tenor_method != 'none':
            for date in dates:
                for country in countries:
                    try:
                        slice_idx = (date, country)
                        if slice_idx in df.index:
                            data = df.loc[slice_idx]
                            
                            # Handle both Series and DataFrame
                            if isinstance(data, pd.Series):
                                if len(data) > 1:
                                    tenors = np.array(data.index).astype(float)
                                    values = data.values
                                    
                                    # Check if we have enough valid data
                                    if not np.all(np.isnan(values)):
                                        smoothed = self.tenor_smoother.smooth(
                                            tenors, values, 
                                            config.tenor_method, 
                                            **config.tenor_params
                                        )
                                        
                                        # Only update if smoothing succeeded
                                        if not np.all(np.isnan(smoothed)):
                                            df.loc[slice_idx] = smoothed
                                        
                            elif isinstance(data, pd.DataFrame) and len(data) > 1:
                                tenors = data.index.get_level_values(-1).values.astype(float)
                                values = data.values.flatten()
                                
                                if not np.all(np.isnan(values)):
                                    smoothed = self.tenor_smoother.smooth(
                                        tenors, values, 
                                        config.tenor_method, 
                                        **config.tenor_params
                                    )
                                    
                                    if not np.all(np.isnan(smoothed)):
                                        df.loc[slice_idx] = smoothed.reshape(-1, 1)
                                        
                    except Exception as e:
                        print(f"Warning: Smoothing failed for {date}, {country}: {str(e)}")
                        continue
        
        # Second: smooth across time for each country-tenor (causal)
        if config.time_method != 'none':
            for country in countries:
                for tenor in df.index.get_level_values(2).unique():
                    try:
                        # Use loc with tuple indexing for MultiIndex
                        # Select all dates for this country-tenor combination
                        idx = pd.IndexSlice
                        time_series_data = df.loc[idx[:, country, tenor], :]
                        
                        if len(time_series_data) > 1:
                            # Sort by date to ensure causality
                            time_series_data = time_series_data.sort_index()
                            values = time_series_data.values.flatten()
                            
                            if not np.all(np.isnan(values)):
                                smoothed = self.time_smoother.smooth(
                                    values,
                                    config.time_method,
                                    **config.time_params
                                )
                                
                                if not np.all(np.isnan(smoothed)):
                                    # Update using the same indexing
                                    df.loc[idx[:, country, tenor], :] = smoothed.reshape(-1, 1)
                                    
                    except Exception as e:
                        print(f"Warning: Time smoothing failed for {country}, tenor {tenor}: {str(e)}")
                        continue
    
    def _smooth_wide_format(self, df: pd.DataFrame, config: SmoothingConfig):
        """Handle wide format where columns represent different series."""
        result = df.copy()
        
        # Ensure time index is sorted for causality
        result = result.sort_index()
        
        # Assume columns are MultiIndex (country, tenor) or similar
        if isinstance(df.columns, pd.MultiIndex):
            countries = df.columns.get_level_values(0).unique()
            
            # First: smooth across tenors for each country-date
            if config.tenor_method != 'none':
                for idx, date in enumerate(df.index):
                    for country in countries:
                        country_cols = df.columns[df.columns.get_level_values(0) == country]
                        tenors = country_cols.get_level_values(1).values.astype(float)
                        values = df.loc[date, country_cols].values
                        
                        if not np.all(np.isnan(values)) and len(values) > 1:
                            smoothed = self.tenor_smoother.smooth(
                                tenors, values,
                                config.tenor_method,
                                **config.tenor_params
                            )
                            result.loc[date, country_cols] = smoothed
            
            # Second: smooth across time for each series (causal)
            if config.time_method != 'none':
                for col in df.columns:
                    values = df[col].values
                    if not np.all(np.isnan(values)):
                        smoothed = self.time_smoother.smooth(
                            values,
                            config.time_method,
                            **config.time_params
                        )
                        result[col] = smoothed
        
        return result


# Example usage and testing functions
def create_sample_data() -> pd.DataFrame:
    """Create sample yield curve data for testing - scaled to match user's data."""
    np.random.seed(42)
    
    dates = pd.date_range('2020-01-01', periods=252*5, freq='B')
    countries = ['US', 'UK', 'DE', 'JP']
    tenors = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
    
    # Create MultiIndex
    index = pd.MultiIndex.from_product(
        [dates, countries, tenors],
        names=['date', 'country', 'tenor']
    )
    
    # Generate synthetic yield changes with structure (SCALED TO USER'S RANGE)
    n = len(index)
    
    # Create structured yield changes in the range -0.08 to 0.08
    data = []
    for date_idx, date in enumerate(dates):
        for country in countries:
            # Base level for this country (larger scale)
            level = np.random.randn() * 0.02  # 2% moves
            
            for tenor_idx, tenor in enumerate(tenors):
                # Add term structure
                slope_effect = -0.005 * np.log(tenor)
                value = level + slope_effect + np.random.randn() * 0.01  # 1% noise
                data.append(value)
    
    df = pd.DataFrame(
        data,
        index=index,
        columns=['yield_change']
    )
    
    # Add some NaNs randomly (1% of data)
    mask = np.random.random(len(df)) < 0.01
    df.loc[mask, 'yield_change'] = np.nan
    
    # Sort index for performance
    df = df.sort_index()
    
    return df


def compare_methods(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Compare different smoothing method combinations."""
    framework = YieldSmoothingFramework()
    
    # Define method combinations to test
    configs = {
        'original': SmoothingConfig(tenor_method='none', time_method='none'),
        
        'pspline_only': SmoothingConfig(
            tenor_method='pspline',
            time_method='none',
            tenor_params={'smoothing_param': None}
        ),
        
        'bernstein_only': SmoothingConfig(
            tenor_method='bernstein',
            time_method='none',
            tenor_params={'degree': 5}
        ),
        
        'ewma_only': SmoothingConfig(
            tenor_method='none',
            time_method='ewma',
            time_params={'decay': 0.94}
        ),
        
        'pspline_ewma': SmoothingConfig(
            tenor_method='pspline',
            time_method='ewma',
            tenor_params={'smoothing_param': None},
            time_params={'decay': 0.94}
        ),
        
        'bernstein_kalman': SmoothingConfig(
            tenor_method='bernstein',
            time_method='kalman',
            tenor_params={'degree': 6},
            time_params={'process_noise': 0.01, 'measurement_noise': 0.1}
        ),
        
        'cubic_kalman': SmoothingConfig(
            tenor_method='cubic_spline',
            time_method='kalman',
            tenor_params={'tension': 0.0},
            time_params={'process_noise': 0.01, 'measurement_noise': 0.1}
        ),
        
        'monotonic_gp': SmoothingConfig(
            tenor_method='monotonic',
            time_method='gp',
            time_params={'length_scale': 20.0, 'noise_level': 0.05}
        ),
    }
    
    results = {}
    for name, config in configs.items():
        print(f"Processing: {name}")
        results[name] = framework.smooth(df, config)
    
    return results


# Metrics for evaluation
def calculate_smoothness_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate metrics to evaluate smoothness and signal preservation."""
    metrics = {}
    
    # Get values and remove NaNs
    values = df.values.flatten()
    values = values[~np.isnan(values)]
    
    if len(values) > 2:
        # Roughness: sum of squared second differences
        second_diff = np.diff(values, n=2)
        metrics['roughness'] = np.mean(second_diff ** 2)
        
        # Variance
        metrics['variance'] = np.var(values)
        
        # Mean absolute change (to measure smoothness)
        first_diff = np.diff(values)
        metrics['mean_abs_change'] = np.mean(np.abs(first_diff))
        
        # Effective sample size
        metrics['n_valid'] = len(values)
        
        # Autocorrelation at lag 1 (higher = smoother)
        if len(values) > 10:
            mean = np.mean(values)
            c0 = np.sum((values - mean) ** 2) / len(values)
            c1 = np.sum((values[:-1] - mean) * (values[1:] - mean)) / (len(values) - 1)
            metrics['autocorr_lag1'] = c1 / c0 if c0 > 0 else 0
    
    return metrics


def validate_causality(df_original: pd.DataFrame, df_smoothed: pd.DataFrame,
                      sample_dates: int = 10) -> bool:
    """
    Validate that smoothing is causal (only uses past data).
    This checks that changing future values doesn't affect past smoothed values.
    """
    # Take a sample of dates to test
    dates = df_original.index.get_level_values(0).unique()
    test_dates = np.random.choice(dates[len(dates)//2:], 
                                 min(sample_dates, len(dates)//4), 
                                 replace=False)
    
    framework = YieldSmoothingFramework()
    config = SmoothingConfig(
        tenor_method='pspline',
        time_method='ewma',
        tenor_params={'smoothing_param': None},
        time_params={'decay': 0.94}
    )
    
    for test_date in test_dates:
        # Create two versions: one normal, one with modified future
        df_test1 = df_original.copy()
        df_test2 = df_original.copy()
        
        # Modify future values in test2
        future_mask = df_test2.index.get_level_values(0) > test_date
        df_test2.loc[future_mask] = df_test2.loc[future_mask] * 2 + 0.01
        
        # Smooth both
        smoothed1 = framework.smooth(df_test1, config)
        smoothed2 = framework.smooth(df_test2, config)
        
        # Check that past values are identical
        past_mask = smoothed1.index.get_level_values(0) <= test_date
        past_values1 = smoothed1.loc[past_mask].values
        past_values2 = smoothed2.loc[past_mask].values
        
        if not np.allclose(past_values1, past_values2, rtol=1e-10):
            print(f"Causality violation detected at date {test_date}")
            return False
    
    return True


def grid_search_smoothing(df: pd.DataFrame, 
                         smoothing_params: List[float],
                         method: str = 'pspline') -> Dict[float, Dict[str, float]]:
    """
    Grid search to find optimal smoothing parameter.
    
    Returns dict of smoothing_param -> metrics
    """
    framework = YieldSmoothingFramework()
    results = {}
    
    for smooth_param in smoothing_params:
        config = SmoothingConfig(
            tenor_method=method,
            time_method='none',  # Isolate tenor smoothing
            tenor_params={'smoothing_param': smooth_param} if method == 'pspline' 
                        else {'degree': int(smooth_param)} if method == 'bernstein'
                        else {}
        )
        
        smoothed = framework.smooth(df, config)
        
        # Calculate fit vs smoothness metrics
        orig_values = df.values.flatten()
        smooth_values = smoothed.values.flatten()
        
        mask = ~(np.isnan(orig_values) | np.isnan(smooth_values))
        if np.sum(mask) > 0:
            # Mean squared error (fit)
            mse = np.mean((orig_values[mask] - smooth_values[mask]) ** 2)
            
            # Roughness (smoothness)
            if len(smooth_values) > 2:
                second_diff = np.diff(smooth_values[mask], n=2)
                roughness = np.mean(second_diff ** 2) if len(second_diff) > 0 else 0
            else:
                roughness = 0
            
            results[smooth_param] = {
                'mse': mse,
                'roughness': roughness,
                'fit_smooth_ratio': mse / (roughness + 1e-10)  # Balance metric
            }
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Creating sample data...")
    df = create_sample_data()
    
    print("\nInitializing framework...")
    framework = YieldSmoothingFramework(knot_positions=[2, 5, 10, 30])
    
    # Test with NO smoothing first to see the data
    print("\nTesting with NO smoothing to verify data structure...")
    no_smooth_config = SmoothingConfig(
        tenor_method='none',
        time_method='none'
    )
    
    original = framework.smooth(df, no_smooth_config)
    
    # Check a sample date/country
    sample_date = df.index.get_level_values(0).unique()[100]
    sample_country = 'US'
    sample_data = df.loc[(sample_date, sample_country)]
    
    print(f"\nSample data for {sample_date}, {sample_country}:")
    print(f"Values: {sample_data.values.flatten()}")
    print(f"Range: {np.ptp(sample_data.values):.6f}")
    print(f"Mean: {np.mean(sample_data.values):.6f}")
    print(f"Std: {np.std(sample_data.values):.6f}")
    
    print("\n" + "="*50)
    print("Testing Different Smoothing Approaches for Large-Scale Data")
    print("="*50)
    
    # For data scaled like yours (-0.08 to 0.08), we need different parameters
    
    print("\n1. INTERPOLATION only (no smoothing)...")
    interp_config = SmoothingConfig(
        tenor_method='pspline',
        time_method='none',
        tenor_params={'smoothing_param': 0, 'degree': 3}  # Pure interpolation
    )
    
    interpolated = framework.smooth(df, interp_config)
    interp_sample = interpolated.loc[(sample_date, sample_country)]
    
    print(f"Interpolated values: {interp_sample.values.flatten()[:5]}...")  # Show first 5
    print(f"Range preserved: {np.ptp(interp_sample.values):.6f}")
    
    print("\n2. VERY LIGHT smoothing (1% error tolerance)...")
    light_config = SmoothingConfig(
        tenor_method='pspline',
        time_method='none',
        tenor_params={'smoothing_param': 0.01, 'degree': 3}  # 1% of squared variation
    )
    
    light_smoothed = framework.smooth(df, light_config)
    light_sample = light_smoothed.loc[(sample_date, sample_country)]
    
    print(f"Light smoothed values: {light_sample.values.flatten()[:5]}...")
    print(f"Range after smoothing: {np.ptp(light_sample.values):.6f}")
    print(f"Range reduction: {(1 - np.ptp(light_sample.values)/np.ptp(sample_data.values))*100:.1f}%")
    
    print("\n3. MODERATE smoothing (5% error tolerance)...")
    moderate_config = SmoothingConfig(
        tenor_method='pspline',
        time_method='none',
        tenor_params={'smoothing_param': 0.05, 'degree': 3}  # 5% of squared variation
    )
    
    moderate_smoothed = framework.smooth(df, moderate_config)
    moderate_sample = moderate_smoothed.loc[(sample_date, sample_country)]
    
    print(f"Moderate smoothed values: {moderate_sample.values.flatten()[:5]}...")
    print(f"Range after smoothing: {np.ptp(moderate_sample.values):.6f}")
    print(f"Range reduction: {(1 - np.ptp(moderate_sample.values)/np.ptp(sample_data.values))*100:.1f}%")
    
    print("\n4. MONOTONIC interpolation...")
    monotonic_config = SmoothingConfig(
        tenor_method='monotonic',
        time_method='none'
    )
    
    monotonic = framework.smooth(df, monotonic_config)
    mono_sample = monotonic.loc[(sample_date, sample_country)]
    
    print(f"Monotonic values: {mono_sample.values.flatten()[:5]}...")
    print(f"Range: {np.ptp(mono_sample.values):.6f}")
    
    print("\n" + "="*50)
    print("Full Pipeline Test with Time Smoothing")
    print("="*50)
    
    # Now test the full pipeline with both tenor and time smoothing
    full_config = SmoothingConfig(
        tenor_method='pspline',
        time_method='ewma',
        tenor_params={'smoothing_param': 0.01, 'degree': 3},  # Light tenor smoothing
        time_params={'decay': 0.94}  # Time smoothing
    )
    
    smoothed = framework.smooth(df, full_config)
    
    print(f"\nOriginal shape: {df.shape}")
    print(f"Smoothed shape: {smoothed.shape}")
    print(f"NaN count original: {df.isna().sum().sum()}")
    print(f"NaN count smoothed: {smoothed.isna().sum().sum()}")
    
    # Calculate metrics
    orig_metrics = calculate_smoothness_metrics(df)
    smooth_metrics = calculate_smoothness_metrics(smoothed)
    
    print(f"\nSmoothing Metrics:")
    print(f"Original roughness: {orig_metrics.get('roughness', 0):.6f}")
    print(f"Smoothed roughness: {smooth_metrics.get('roughness', 0):.6f}")
    print(f"Roughness reduction: {(1 - smooth_metrics.get('roughness', 0)/orig_metrics.get('roughness', 1)) * 100:.1f}%")
    
    print(f"\nOriginal mean abs change: {orig_metrics.get('mean_abs_change', 0):.6f}")
    print(f"Smoothed mean abs change: {smooth_metrics.get('mean_abs_change', 0):.6f}")
    
    print(f"\nOriginal autocorr(1): {orig_metrics.get('autocorr_lag1', 0):.3f}")
    print(f"Smoothed autocorr(1): {smooth_metrics.get('autocorr_lag1', 0):.3f}")
    
    # Validate causality
    print("\nValidating causality of smoothing methods...")
    is_causal = validate_causality(df, smoothed, sample_dates=5)
    print(f"Causality check: {'PASSED' if is_causal else 'FAILED'}")
    
    # Compare all methods
    
    print("\n" + "="*50)
    print("Grid Search for Optimal Smoothing:")
    print("="*50)
    
    # Test different smoothing parameters
    smoothing_params = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    grid_results = grid_search_smoothing(df, smoothing_params, method='pspline')
    
    print("\nP-Spline Smoothing Parameter Grid Search:")
    print("-" * 60)
    print(f"{'Smooth Param':<15} {'MSE':<12} {'Roughness':<12} {'Fit/Smooth':<12}")
    print("-" * 60)
    
    for param, metrics in grid_results.items():
        print(f"{param:<15.4f} {metrics['mse']:.6e}  {metrics['roughness']:.6e}  {metrics['fit_smooth_ratio']:.3f}")
    
    # Find best parameter (minimize fit_smooth_ratio)
    best_param = min(grid_results.keys(), key=lambda x: grid_results[x]['fit_smooth_ratio'])
    print(f"\nBest smoothing parameter: {best_param:.4f}")
    
    print("\n" + "="*50)
    print("Comparing Different Smoothing Strengths:")
    print("="*50)
    results = compare_methods(df)
    
    print("\nMethod Comparison Summary:")
    print("-" * 60)
    print(f"{'Method':<20} {'Roughness':<12} {'Mean Abs Δ':<12} {'Autocorr(1)':<10}")
    print("-" * 60)
    
    for name, result_df in results.items():
        metrics = calculate_smoothness_metrics(result_df)
        print(f"{name:<20} {metrics.get('roughness', 0):.6f}    "
              f"{metrics.get('mean_abs_change', 0):.6f}    "
              f"{metrics.get('autocorr_lag1', 0):.3f}")
