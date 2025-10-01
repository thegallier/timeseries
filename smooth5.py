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

class TimeSmoother:
    """Handles time-series smoothing - all methods are causal."""
    
    @staticmethod
    def ewma(values: np.ndarray, decay: float = 0.94) -> np.ndarray:
        """Exponentially weighted moving average."""
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
    def smooth(values: np.ndarray, method: str, **params) -> np.ndarray:
        """Apply specified time smoothing method."""
        if method == 'none':
            return values
        elif method == 'ewma':
            return TimeSmoother.ewma(values, **params)
        else:
            return values

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

def visualize_multicountry_smoothing(df: pd.DataFrame):
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
