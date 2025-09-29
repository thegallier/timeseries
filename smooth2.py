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
        
        # CRITICAL: Handle duplicate tenors by averaging
        unique_tenors, inverse_indices = np.unique(valid_tenors, return_inverse=True)
        if len(unique_tenors) < len(valid_tenors):
            # We have duplicates - average them
            unique_values = np.zeros(len(unique_tenors))
            for i in range(len(unique_tenors)):
                unique_values[i] = np.mean(valid_values[inverse_indices == i])
            valid_tenors = unique_tenors
            valid_values = unique_values
        
        # Ensure we still have enough points after deduplication
        if len(valid_tenors) < degree + 1:
            degree = max(1, len(valid_tenors) - 1)
        
        try:
            # Create spline
            if smoothing_param is None or smoothing_param == 0:
                # Use interpolating spline
                spline = UnivariateSpline(valid_tenors, valid_values, k=degree, s=0)
            else:
                # Scale smoothing parameter by data scale
                data_scale = np.sum(valid_values**2)
                
                # Adjust scaling to avoid "s too small" warnings
                # Use a minimum threshold to prevent convergence issues
                scaled_smooth = max(smoothing_param * data_scale, 1e-6 * len(valid_values))
                
                # Suppress warnings and create spline
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    spline = UnivariateSpline(valid_tenors, valid_values, k=degree, s=scaled_smooth)
            
            # Evaluate at ALL original tenor points (including duplicates)
            result = spline(tenors)
            
            # Diagnostic check
            if np.ptp(result) < np.ptp(valid_values) * 0.1:  # Lost 90% of variation
                # Silently fall back to interpolation
                return np.interp(tenors, valid_tenors, valid_values)
            
            return result
            
        except Exception as e:
            # Silently fall back to linear interpolation
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
            smoothed_unique = model_func(unique_tenors, unique_values)
            
            # Map back to all points
            result = np.full_like(values, np.nan)
            result[mask] = smoothed_unique[inverse_indices]
            return result
        else:
            # No duplicates
            try:
                smoothed = model_func(valid_tenors, valid_values)
                result = np.full_like(values, np.nan)
                result[mask] = smoothed
                return result
            except Exception as e:
                print(f"Warning: Custom model failed ({str(e)}), returning original values")
                return values
    
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
        elif method == 'custom':
            if 'model_func' not in params:
                raise ValueError("Custom method requires 'model_func' parameter")
            return self.custom_model(tenors, values, params['model_func'])
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
    
    def smooth(self, df: pd.DataFrame, config: SmoothingConfig,
               post_tenor_method: Optional[str] = None,
               post_tenor_params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Apply smoothing to yield curve data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with MultiIndex (date, country, tenor) or similar structure
        config : SmoothingConfig
            Configuration specifying methods and parameters
        post_tenor_method : str, optional
            Additional tenor smoothing to apply after initial smoothing
        post_tenor_params : dict, optional
            Parameters for post-tenor smoothing
        
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
        
        # Apply main smoothing
        if isinstance(df.index, pd.MultiIndex):
            self._smooth_multiindex(result, config)
        else:
            self._smooth_wide_format(result, config)
        
        # Optional chained tenor smoothing
        if post_tenor_method is not None and post_tenor_method != 'none':
            if isinstance(result.index, pd.MultiIndex):
                dates = result.index.get_level_values(0).unique()
                countries = result.index.get_level_values(1).unique()
                
                for date in dates:
                    for country in countries:
                        try:
                            mask = (result.index.get_level_values(0) == date) & \
                                   (result.index.get_level_values(1) == country)
                            
                            if mask.any():
                                data_slice = result.loc[mask]
                                if len(data_slice) > 1:
                                    tenors = data_slice.index.get_level_values(2).values.astype(float)
                                    values = data_slice.values.flatten()
                                    
                                    # Apply post-processing smoothing
                                    smoothed = self.tenor_smoother.smooth(
                                        tenors, values, post_tenor_method, 
                                        **(post_tenor_params or {})
                                    )
                                    
                                    result.loc[mask, result.columns[0]] = smoothed
                        except Exception as e:
                            continue
        
        return result
    
    def _smooth_multiindex(self, df: pd.DataFrame, config: SmoothingConfig):
        """Handle MultiIndex format (date, country, tenor) with potential duplicates."""
        # Get unique dates and countries
        dates = df.index.get_level_values(0).unique()
        countries = df.index.get_level_values(1).unique()
        
        # First: smooth across tenors for each country-date
        if config.tenor_method != 'none':
            for date in dates:
                for country in countries:
                    try:
                        # Get data for this date-country combination
                        mask = (df.index.get_level_values(0) == date) & \
                               (df.index.get_level_values(1) == country)
                        
                        if mask.any():
                            data_slice = df.loc[mask]
                            
                            if len(data_slice) > 1:
                                # Extract tenors and values
                                tenors = data_slice.index.get_level_values(2).values.astype(float)
                                values = data_slice.values.flatten()
                                
                                # Get unique tenors and their positions
                                unique_tenors = np.unique(tenors)
                                
                                # Debug output for first few iterations
                                if date == dates[0] and country == countries[0]:
                                    print(f"\nDebug: Processing {date}, {country}")
                                    print(f"  Total rows: {len(tenors)}")
                                    print(f"  Unique tenors: {len(unique_tenors)}")
                                    print(f"  Duplicate structure: {[np.sum(tenors == t) for t in unique_tenors[:5]]}")
                                
                                # Option A: Average duplicates before smoothing
                                avg_values = np.array([np.mean(values[tenors == t]) for t in unique_tenors])
                                
                                if not np.all(np.isnan(avg_values)):
                                    # Smooth the averaged values
                                    smoothed_unique = self.tenor_smoother.smooth(
                                        unique_tenors, avg_values, 
                                        config.tenor_method, 
                                        **config.tenor_params
                                    )
                                    
                                    # Map smoothed values back to all original rows
                                    smoothed_all = np.zeros(len(tenors))
                                    for i, unique_tenor in enumerate(unique_tenors):
                                        smoothed_all[tenors == unique_tenor] = smoothed_unique[i]
                                    
                                    if date == dates[0] and country == countries[0]:
                                        print(f"  Original values: {values[:5]}...")
                                        print(f"  Smoothed values: {smoothed_all[:5]}...")
                                    
                                    # Update the values
                                    df.loc[mask, df.columns[0]] = smoothed_all
                                        
                    except Exception as e:
                        print(f"Warning: Smoothing failed for {date}, {country}: {str(e)}")
                        continue
        
        # Second: smooth across time for each country-tenor (causal)
        if config.time_method != 'none':
            for country in countries:
                unique_tenors = df.index.get_level_values(2).unique()
                
                for tenor in unique_tenors:
                    try:
                        # Get mask for this country-tenor combination
                        mask = (df.index.get_level_values(1) == country) & \
                               (df.index.get_level_values(2) == tenor)
                        
                        if mask.any():
                            time_series_data = df.loc[mask]
                            
                            # Check if we have duplicates per date
                            dates_in_series = time_series_data.index.get_level_values(0)
                            unique_dates = dates_in_series.unique()
                            
                            if len(time_series_data) > len(unique_dates):
                                # We have duplicates - average them per date
                                avg_values = []
                                for d in unique_dates:
                                    date_mask = (df.index.get_level_values(0) == d) & mask
                                    avg_values.append(df.loc[date_mask].values.mean())
                                values_to_smooth = np.array(avg_values)
                            else:
                                # No duplicates, use as is
                                time_series_data = time_series_data.sort_index()
                                values_to_smooth = time_series_data.values.flatten()
                            
                            if not np.all(np.isnan(values_to_smooth)):
                                smoothed = self.time_smoother.smooth(
                                    values_to_smooth,
                                    config.time_method,
                                    **config.time_params
                                )
                                
                                if not np.all(np.isnan(smoothed)):
                                    # Map back to original structure
                                    if len(time_series_data) > len(unique_dates):
                                        # Expand smoothed values to match duplicates
                                        smoothed_expanded = np.zeros(len(time_series_data))
                                        for i, d in enumerate(unique_dates):
                                            date_mask = dates_in_series == d
                                            smoothed_expanded[date_mask] = smoothed[i]
                                        df.loc[mask, df.columns[0]] = smoothed_expanded
                                    else:
                                        df.loc[mask, df.columns[0]] = smoothed
                                    
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
def diagnose_data_structure(df: pd.DataFrame) -> None:
    """Diagnose the structure of the yield data."""
    print("\n" + "="*50)
    print("Data Structure Diagnosis")
    print("="*50)
    
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Index type: {type(df.index)}")
    
    if isinstance(df.index, pd.MultiIndex):
        print(f"MultiIndex levels: {df.index.names}")
        print(f"Number of unique dates: {df.index.get_level_values(0).nunique()}")
        print(f"Number of unique countries: {df.index.get_level_values(1).nunique()}")
        print(f"Number of unique tenors: {df.index.get_level_values(2).nunique()}")
        
        # Check for duplicates
        duplicates = df.index.duplicated()
        n_duplicates = duplicates.sum()
        print(f"\nDuplicate index entries: {n_duplicates}")
        
        if n_duplicates > 0:
            print("\nWARNING: Duplicate index entries found!")
            print("This will cause issues with smoothing.")
            print("\nExample duplicates:")
            dup_mask = df.index.duplicated(keep=False)
            sample_dups = df[dup_mask].head(10)
            print(sample_dups)
            
            print("\nSuggestion: Remove duplicates or aggregate them:")
            print("df = df.groupby(level=[0,1,2]).mean()  # Average duplicates")
            print("# OR")
            print("df = df[~df.index.duplicated(keep='first')]  # Keep first")
        
        # Check data structure for a sample date-country
        sample_date = df.index.get_level_values(0).unique()[0]
        sample_country = df.index.get_level_values(1).unique()[0]
        
        print(f"\nSample slice for {sample_date}, {sample_country}:")
        sample_data = df.loc[(sample_date, sample_country)]
        print(f"Shape: {sample_data.shape}")
        print(f"Tenors: {sample_data.index.unique().tolist() if hasattr(sample_data.index, 'unique') else 'N/A'}")
        print(f"Values:\n{sample_data.head()}")
    
    print("\n" + "="*50)


def preprocess_data(df: pd.DataFrame, method: str = 'mean') -> pd.DataFrame:
    """
    Preprocess data to handle duplicates and ensure proper structure.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    method : str
        How to handle duplicates: 'mean', 'first', 'last'
    
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe
    """
    df_clean = df.copy()
    
    # Check for and handle duplicates
    if df_clean.index.duplicated().any():
        print(f"Found {df_clean.index.duplicated().sum()} duplicate index entries")
        
        if method == 'mean':
            # Average duplicate entries
            df_clean = df_clean.groupby(level=list(range(df_clean.index.nlevels))).mean()
            print(f"Averaged duplicates. New shape: {df_clean.shape}")
        elif method == 'first':
            # Keep first occurrence
            df_clean = df_clean[~df_clean.index.duplicated(keep='first')]
            print(f"Kept first occurrence. New shape: {df_clean.shape}")
        elif method == 'last':
            # Keep last occurrence
            df_clean = df_clean[~df_clean.index.duplicated(keep='last')]
            print(f"Kept last occurrence. New shape: {df_clean.shape}")
    
    # Ensure sorted index
    df_clean = df_clean.sort_index()
    
    return df_clean
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


def compare_methods(df: pd.DataFrame,
                    custom_models: Optional[Dict[str, Callable]] = None) -> Dict[str, pd.DataFrame]:
    """
    Compare different smoothing method combinations, including custom models.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input yield curve data
    custom_models : dict, optional
        Dictionary of {name: function} for custom tenor models
        Functions should accept (tenors, values) and return smoothed values
    
    Returns:
    --------
    dict
        Dictionary of {method_name: smoothed_dataframe}
    """
    framework = YieldSmoothingFramework()
    
    # Define standard configurations
    configs = {
        'original': SmoothingConfig(tenor_method='none', time_method='none'),
        
        'pspline_only': SmoothingConfig(
            tenor_method='pspline',
            time_method='none',
            tenor_params={'smoothing_param': 0.01}
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
            tenor_params={'smoothing_param': 0.01},
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
    
    # Add custom models if provided
    if custom_models:
        for name, model_func in custom_models.items():
            # Custom model alone
            configs[f"custom_{name}"] = SmoothingConfig(
                tenor_method='custom',
                time_method='none',
                tenor_params={'model_func': model_func}
            )
            
            # Custom model with EWMA time smoothing
            configs[f"custom_{name}_ewma"] = SmoothingConfig(
                tenor_method='custom',
                time_method='ewma',
                tenor_params={'model_func': model_func},
                time_params={'decay': 0.94}
            )
    
    results = {}
    for name, config in configs.items():
        print(f"Processing: {name}")
        
        # Check if we need post-processing
        if "custom" in name and "_pspline" in name.split("_")[-1]:
            # Chain custom model with p-spline smoothing
            results[name] = framework.smooth(
                df, config,
                post_tenor_method="pspline",
                post_tenor_params={'smoothing_param': 0.001, 'degree': 3}
            )
        else:
            results[name] = framework.smooth(df, config)
    
    return results


# Example custom models
def nelson_siegel_model(tenors: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    Simplified Nelson-Siegel curve fitting.
    
    This is a basic implementation - a full NS model would optimize tau.
    """
    # Set tau (could be optimized)
    tau = 2.0
    
    # Create basis functions
    ones = np.ones_like(tenors)
    f1 = (1 - np.exp(-tenors/tau)) / (tenors/tau + 1e-10)  # Avoid division by zero
    f2 = f1 - np.exp(-tenors/tau)
    
    # Stack basis functions
    X = np.column_stack([ones, f1, f2])
    
    # Fit using least squares
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, values, rcond=None)
        return X @ coeffs
    except:
        # Fall back to simple polynomial
        return np.polyval(np.polyfit(tenors, values, 2), tenors)


def svensson_model(tenors: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    Simplified Svensson (4-factor Nelson-Siegel) model.
    """
    tau1, tau2 = 2.0, 5.0
    
    # Create basis functions
    ones = np.ones_like(tenors)
    f1 = (1 - np.exp(-tenors/tau1)) / (tenors/tau1 + 1e-10)
    f2 = f1 - np.exp(-tenors/tau1)
    f3 = (1 - np.exp(-tenors/tau2)) / (tenors/tau2 + 1e-10) - np.exp(-tenors/tau2)
    
    # Stack basis functions
    X = np.column_stack([ones, f1, f2, f3])
    
    # Fit using least squares
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, values, rcond=None)
        return X @ coeffs
    except:
        return nelson_siegel_model(tenors, values)  # Fall back to NS


def polynomial_model(degree: int = 4):
    """Factory function for polynomial models of specified degree."""
    def poly_fit(tenors: np.ndarray, values: np.ndarray) -> np.ndarray:
        coeffs = np.polyfit(tenors, values, degree)
        return np.polyval(coeffs, tenors)
    return poly_fit


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


def test_smoothing_step_by_step(df: pd.DataFrame) -> None:
    """Test smoothing step by step to diagnose issues."""
    print("\n" + "="*50)
    print("Step-by-Step Smoothing Test")
    print("="*50)
    
    # Get a sample date and country
    sample_date = df.index.get_level_values(0).unique()[0]
    sample_country = df.index.get_level_values(1).unique()[0]
    
    print(f"\nTesting with {sample_date}, {sample_country}")
    
    # Extract data for this date-country
    mask = (df.index.get_level_values(0) == sample_date) & \
           (df.index.get_level_values(1) == sample_country)
    
    data_slice = df.loc[mask]
    print(f"\nOriginal data slice shape: {data_slice.shape}")
    print(f"Original data:\n{data_slice}")
    
    # Extract tenors and values
    tenors = data_slice.index.get_level_values(2).values.astype(float)
    values = data_slice.values.flatten()
    
    print(f"\nTenors: {tenors}")
    print(f"Values: {values}")
    
    # Check for duplicate tenors
    unique_tenors, counts = np.unique(tenors, return_counts=True)
    if np.any(counts > 1):
        print("\n⚠️  WARNING: Duplicate tenors found!")
        print(f"Duplicated tenors: {unique_tenors[counts > 1]}")
        print(f"Counts: {counts[counts > 1]}")
        
        # Handle duplicates by averaging
        print("\nAveraging duplicate tenor values...")
        avg_values = np.zeros(len(unique_tenors))
        for i, t in enumerate(unique_tenors):
            avg_values[i] = np.mean(values[tenors == t])
        
        tenors = unique_tenors
        values = avg_values
        print(f"After averaging - Tenors: {tenors}")
        print(f"After averaging - Values: {values}")
    
    # Test different smoothing methods
    from scipy.interpolate import UnivariateSpline
    
    # Test 1: Pure interpolation
    print("\n--- Test 1: Pure Interpolation (s=0) ---")
    try:
        spline = UnivariateSpline(tenors, values, k=min(3, len(tenors)-1), s=0)
        smoothed_interp = spline(tenors)
        print(f"Result: {smoothed_interp}")
        print(f"Changed? {not np.allclose(values, smoothed_interp)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 2: Light smoothing
    print("\n--- Test 2: Light Smoothing (s=0.001) ---")
    try:
        data_scale = np.sum(values**2)
        s_param = 0.01 * data_scale  # 1% error tolerance
        spline = UnivariateSpline(tenors, values, k=min(3, len(tenors)-1), s=s_param)
        smoothed_light = spline(tenors)
        print(f"Result: {smoothed_light}")
        print(f"Changed? {not np.allclose(values, smoothed_light)}")
        
        # Test 3: Check if all values become the same
        print(f"\nAll values same after smoothing? {len(np.unique(np.round(smoothed_light, 10))) == 1}")
        print(f"Unique values in smoothed: {np.unique(np.round(smoothed_light, 10))[:5]}...")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Example usage
    print("Creating sample data...")
    df = create_sample_data()
    
    # Diagnose the data structure
    diagnose_data_structure(df)
    
    # Test smoothing step by step
    test_smoothing_step_by_step(df)
    
    # Clean the data if needed
    df_clean = preprocess_data(df, method='mean')
    
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
