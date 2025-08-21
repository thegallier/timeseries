import numpy as np
import jax.numpy as jnp
from typing import List, Tuple, Dict, Union, Optional
from scipy.interpolate import interp1d, CubicSpline
import pandas as pd


def get_interpolated_hedge_ratios(
    queries: List[Tuple[str, float]],
    results: Dict,
    method: str = 'linear',
    extrapolate: str = 'nearest',
    return_dataframe: bool = True,
    verbose: bool = True
) -> Union[Dict, pd.DataFrame]:
    """
    Get interpolated hedge ratios for any country-tenor combination.
    
    Args:
        queries: List of (country, tenor) tuples where tenor is a float
                 e.g., [('BEL', 2.0), ('BEL', 7.5), ('FRA', 10.0), ('DEU', 25.0)]
        results: Results from compute_all_countries_hedge_ratios_batch or similar
        method: Interpolation method
                - 'linear': Linear interpolation (default)
                - 'cubic': Cubic spline interpolation (smoother)
                - 'nearest': Nearest neighbor (no interpolation)
                - 'quadratic': Quadratic interpolation
                - 'log-linear': Linear in log space (good for long tenors)
        extrapolate: How to handle tenors outside [2, 50] range
                - 'nearest': Use nearest available tenor (default)
                - 'linear': Linear extrapolation
                - 'zero': Return zeros
                - 'error': Raise error
        return_dataframe: If True, return pandas DataFrame; else return dict
        verbose: Print information about interpolation
        
    Returns:
        Dictionary or DataFrame with interpolated hedge ratios for each query
        
    Example:
        queries = [('BEL', 7.5), ('FRA', 25.0), ('DEU', 3.5)]
        hedges = get_interpolated_hedge_ratios(queries, results, method='cubic')
    """
    
    # Initialize builder for tenor mapping
    builder = HedgeConstraintBuilder()
    
    # Standard tenor points in years
    standard_tenors = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 50], dtype=float)
    tenor_names = builder.tenors
    hedge_names = builder.hedges
    n_hedges = len(hedge_names)
    
    # Extract W matrix from results
    if 'batch_results' in results and 'W_avg' in results['batch_results']:
        W_full = results['batch_results']['W_avg']
    elif 'W_avg' in results:
        W_full = results['W_avg']
    elif 'W_processed' in results:
        W_full = results['W_processed']
    else:
        raise ValueError("No coefficient matrix found in results")
    
    # Convert to numpy for interpolation
    W_full = np.array(W_full)
    
    if verbose:
        print("="*60)
        print("HEDGE RATIO INTERPOLATION")
        print("="*60)
        print(f"Interpolation method: {method}")
        print(f"Extrapolation: {extrapolate}")
        print(f"Queries: {len(queries)} points")
        print(f"W matrix shape: {W_full.shape}")
    
    # Process each query
    interpolated_results = []
    
    for country, tenor_query in queries:
        if country not in builder.countries:
            if verbose:
                print(f"\nWarning: Country {country} not found, skipping")
            continue
        
        country_idx = builder.countries.index(country)
        
        # Extract coefficients for this country (shape: n_hedges × n_tenors)
        start_idx = country_idx * len(standard_tenors)
        end_idx = (country_idx + 1) * len(standard_tenors)
        W_country = W_full[:, start_idx:end_idx]
        
        # Initialize result for this query
        query_result = {
            'country': country,
            'tenor': tenor_query,
            'interpolated': method != 'nearest' and tenor_query not in standard_tenors
        }
        
        # Check if exact tenor exists
        if tenor_query in standard_tenors:
            # Exact match - no interpolation needed
            tenor_idx = np.where(standard_tenors == tenor_query)[0][0]
            for hedge_idx, hedge_name in enumerate(hedge_names):
                value = float(W_country[hedge_idx, tenor_idx])
                query_result[hedge_name] = value
            
            if verbose:
                print(f"\n{country} {tenor_query}yr: Exact match (no interpolation)")
        
        else:
            # Need interpolation
            if verbose:
                print(f"\n{country} {tenor_query}yr: Interpolating using {method}")
            
            # Check bounds
            if tenor_query < standard_tenors[0] or tenor_query > standard_tenors[-1]:
                if extrapolate == 'error':
                    raise ValueError(f"Tenor {tenor_query} outside range [{standard_tenors[0]}, {standard_tenors[-1]}]")
                elif verbose:
                    print(f"  Note: Extrapolating ({extrapolate} method)")
            
            # Interpolate for each hedge
            for hedge_idx, hedge_name in enumerate(hedge_names):
                hedge_values = W_country[hedge_idx, :]
                
                # Skip if all zeros (no need to interpolate)
                if np.all(np.abs(hedge_values) < 1e-10):
                    query_result[hedge_name] = 0.0
                    continue
                
                # Create interpolation function based on method
                if method == 'linear':
                    interp_func = create_linear_interpolator(
                        standard_tenors, hedge_values, extrapolate
                    )
                elif method == 'cubic':
                    interp_func = create_cubic_interpolator(
                        standard_tenors, hedge_values, extrapolate
                    )
                elif method == 'nearest':
                    interp_func = create_nearest_interpolator(
                        standard_tenors, hedge_values
                    )
                elif method == 'quadratic':
                    interp_func = create_quadratic_interpolator(
                        standard_tenors, hedge_values, extrapolate
                    )
                elif method == 'log-linear':
                    interp_func = create_log_linear_interpolator(
                        standard_tenors, hedge_values, extrapolate
                    )
                else:
                    raise ValueError(f"Unknown interpolation method: {method}")
                
                # Get interpolated value
                value = float(interp_func(tenor_query))
                query_result[hedge_name] = value
        
        interpolated_results.append(query_result)
    
    # Convert to desired output format
    if return_dataframe:
        df = pd.DataFrame(interpolated_results)
        
        # Reorder columns for better readability
        base_cols = ['country', 'tenor', 'interpolated']
        hedge_cols = [col for col in df.columns if col not in base_cols]
        df = df[base_cols + hedge_cols]
        
        # Round for display
        for col in hedge_cols:
            df[col] = df[col].round(6)
        
        if verbose:
            print("\n" + "="*60)
            print("INTERPOLATION RESULTS")
            print("="*60)
            print(df.to_string(index=False))
        
        return df
    else:
        return interpolated_results


def create_linear_interpolator(x, y, extrapolate='nearest'):
    """Create linear interpolation function."""
    if extrapolate == 'linear':
        fill_value = 'extrapolate'
    elif extrapolate == 'zero':
        fill_value = 0.0
    else:  # nearest
        fill_value = (y[0], y[-1])
    
    return interp1d(x, y, kind='linear', bounds_error=False, fill_value=fill_value)


def create_cubic_interpolator(x, y, extrapolate='nearest'):
    """Create cubic spline interpolation function."""
    if len(x) < 4:
        # Need at least 4 points for cubic, fall back to linear
        return create_linear_interpolator(x, y, extrapolate)
    
    cs = CubicSpline(x, y, extrapolate=False)
    
    def interpolator(x_new):
        if np.isscalar(x_new):
            x_new = np.array([x_new])
        
        result = cs(x_new)
        
        # Handle extrapolation
        if extrapolate == 'nearest':
            result[x_new < x[0]] = y[0]
            result[x_new > x[-1]] = y[-1]
        elif extrapolate == 'zero':
            result[x_new < x[0]] = 0.0
            result[x_new > x[-1]] = 0.0
        elif extrapolate == 'linear':
            # Use linear extrapolation beyond bounds
            mask_low = x_new < x[0]
            mask_high = x_new > x[-1]
            
            if np.any(mask_low):
                slope_low = (y[1] - y[0]) / (x[1] - x[0])
                result[mask_low] = y[0] + slope_low * (x_new[mask_low] - x[0])
            
            if np.any(mask_high):
                slope_high = (y[-1] - y[-2]) / (x[-1] - x[-2])
                result[mask_high] = y[-1] + slope_high * (x_new[mask_high] - x[-1])
        
        return result[0] if result.shape[0] == 1 else result
    
    return interpolator


def create_nearest_interpolator(x, y):
    """Create nearest neighbor interpolation function."""
    def interpolator(x_new):
        if np.isscalar(x_new):
            idx = np.argmin(np.abs(x - x_new))
            return y[idx]
        else:
            return np.array([y[np.argmin(np.abs(x - xn))] for xn in x_new])
    return interpolator


def create_quadratic_interpolator(x, y, extrapolate='nearest'):
    """Create quadratic interpolation function."""
    if extrapolate == 'linear' or extrapolate == 'quadratic':
        fill_value = 'extrapolate'
    elif extrapolate == 'zero':
        fill_value = 0.0
    else:  # nearest
        fill_value = (y[0], y[-1])
    
    return interp1d(x, y, kind='quadratic', bounds_error=False, fill_value=fill_value)


def create_log_linear_interpolator(x, y, extrapolate='nearest'):
    """Create log-linear interpolation (good for yield curves)."""
    # Handle zeros and negative values
    y_positive = np.maximum(y, 1e-10)
    log_y = np.log(y_positive)
    
    # Create linear interpolator in log space
    if extrapolate == 'linear':
        fill_value = 'extrapolate'
    elif extrapolate == 'zero':
        fill_value = np.log(1e-10)
    else:  # nearest
        fill_value = (log_y[0], log_y[-1])
    
    log_interp = interp1d(x, log_y, kind='linear', bounds_error=False, fill_value=fill_value)
    
    def interpolator(x_new):
        log_result = log_interp(x_new)
        result = np.exp(log_result)
        # Restore original sign
        if np.isscalar(x_new):
            if x_new <= x[0] and y[0] < 0:
                result = -result
            elif x_new >= x[-1] and y[-1] < 0:
                result = -result
        return result
    
    return interpolator


def plot_interpolated_hedge_ratios(
    queries: List[Tuple[str, float]],
    results: Dict,
    methods: List[str] = ['linear', 'cubic'],
    hedges_to_plot: List[str] = None,
    figsize: Tuple[int, int] = (15, 10)
):
    """
    Visualize interpolated hedge ratios across different methods.
    
    Args:
        queries: List of (country, tenor) queries
        results: Results dictionary
        methods: List of interpolation methods to compare
        hedges_to_plot: Specific hedges to plot (default: all non-zero)
        figsize: Figure size
        
    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    builder = HedgeConstraintBuilder()
    standard_tenors = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 50])
    
    # Get unique countries from queries
    countries = list(set([q[0] for q in queries]))
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(len(countries), 2, figure=fig, hspace=0.3, wspace=0.3)
    
    for c_idx, country in enumerate(countries):
        # Get country's hedge ratios
        country_idx = builder.countries.index(country)
        
        if 'batch_results' in results and 'W_avg' in results['batch_results']:
            W_full = results['batch_results']['W_avg']
        else:
            W_full = results['W_avg']
        
        start_idx = country_idx * len(standard_tenors)
        end_idx = (country_idx + 1) * len(standard_tenors)
        W_country = W_full[:, start_idx:end_idx]
        
        # Determine which hedges to plot
        if hedges_to_plot is None:
            # Plot non-zero hedges
            hedges_to_plot_idx = []
            for h_idx in range(len(builder.hedges)):
                if np.any(np.abs(W_country[h_idx, :]) > 1e-10):
                    hedges_to_plot_idx.append(h_idx)
        else:
            hedges_to_plot_idx = [builder.hedges.index(h) for h in hedges_to_plot 
                                  if h in builder.hedges]
        
        # Plot original points and interpolation
        ax = fig.add_subplot(gs[c_idx, 0])
        
        # Create fine grid for smooth interpolation curves
        tenor_grid = np.linspace(2, 50, 200)
        
        for h_idx in hedges_to_plot_idx:
            hedge_name = builder.hedges[h_idx]
            hedge_values = W_country[h_idx, :]
            
            # Plot original points
            ax.scatter(standard_tenors, hedge_values, label=f'{hedge_name} (data)', 
                      alpha=0.7, s=30)
            
            # Plot interpolation
            for method in methods[:2]:  # Show first 2 methods
                interp_values = []
                for t in tenor_grid:
                    result = get_interpolated_hedge_ratios(
                        [(country, t)], results, method=method, 
                        return_dataframe=False, verbose=False
                    )
                    interp_values.append(result[0][hedge_name])
                
                ax.plot(tenor_grid, interp_values, '--', alpha=0.5, 
                       label=f'{hedge_name} ({method})', linewidth=1)
        
        # Mark query points
        country_queries = [q for q in queries if q[0] == country]
        query_tenors = [q[1] for q in country_queries]
        
        for qt in query_tenors:
            ax.axvline(x=qt, color='red', linestyle=':', alpha=0.3)
        
        ax.set_xlabel('Tenor (years)')
        ax.set_ylabel('Hedge Ratio')
        ax.set_title(f'{country} - Interpolation Methods')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Plot comparison table
        ax2 = fig.add_subplot(gs[c_idx, 1])
        ax2.axis('off')
        
        # Create comparison table for this country's queries
        if country_queries:
            table_data = []
            for cq in country_queries:
                row = [f"{cq[1]:.1f}yr"]
                for method in methods:
                    result = get_interpolated_hedge_ratios(
                        [cq], results, method=method, 
                        return_dataframe=False, verbose=False
                    )
                    # Get the most significant hedge
                    hedge_vals = {k: abs(v) for k, v in result[0].items() 
                                 if k not in ['country', 'tenor', 'interpolated']}
                    if hedge_vals:
                        max_hedge = max(hedge_vals, key=hedge_vals.get)
                        row.append(f"{result[0][max_hedge]:.4f}")
                table_data.append(row)
            
            table = ax2.table(cellText=table_data,
                            colLabels=['Tenor'] + methods,
                            cellLoc='center',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            ax2.set_title(f'{country} Query Results', fontsize=10)
    
    plt.suptitle('Hedge Ratio Interpolation Analysis', fontsize=14)
    return fig


# Example usage function
def example_interpolation():
    """
    Example showing how to use the interpolation function.
    """
    
    print("="*80)
    print("HEDGE RATIO INTERPOLATION EXAMPLE")
    print("="*80)
    
    # Generate sample data and results
    np.random.seed(42)
    n_samples = 1000
    X = jnp.array(np.random.randn(n_samples, 7))
    Y = jnp.array(np.random.randn(n_samples, 168))
    
    # Run batch computation
    country_rules = {
        'BEL': {'allowed_countries': ['DEU', 'FRA'], 'use_adjacent_only': False},
        'FRA': {'allowed_countries': ['FRA'], 'use_adjacent_only': False},
        'DEU': {'allowed_countries': ['DEU'], 'use_adjacent_only': False}
    }
    
    print("\n1. Running batch computation...")
    results = compute_all_countries_hedge_ratios_batch(
        X=X, Y=Y,
        country_rules=country_rules,
        window_size=200,
        stride=150,
        method='jax',
        constraint_method='penalty',
        verbose=False
    )
    
    print("   Computation complete.")
    
    # Define queries for non-standard tenors
    queries = [
        ('BEL', 2.0),    # Exact match (no interpolation)
        ('BEL', 2.5),    # Between 2yr and 3yr
        ('BEL', 7.5),    # Between 7yr and 8yr
        ('BEL', 11.0),   # Between 10yr and 12yr
        ('BEL', 25.0),   # Between 20yr and 30yr
        ('BEL', 45.0),   # Between 30yr and 50yr
        ('FRA', 3.5),    # France 3.5yr
        ('FRA', 10.0),   # France 10yr (exact)
        ('DEU', 8.25),   # Germany 8.25yr
        ('DEU', 35.0),   # Germany 35yr
    ]
    
    print(f"\n2. Interpolating hedge ratios for {len(queries)} queries...")
    print("   Queries:", queries)
    
    # Test different interpolation methods
    print("\n3. Testing different interpolation methods...")
    print("-"*60)
    
    methods_to_test = ['linear', 'cubic', 'nearest', 'log-linear']
    
    all_results = {}
    for method in methods_to_test:
        print(f"\n{method.upper()} INTERPOLATION:")
        df = get_interpolated_hedge_ratios(
            queries=queries,
            results=results,
            method=method,
            extrapolate='nearest',
            return_dataframe=True,
            verbose=False
        )
        all_results[method] = df
        
        # Show sample results
        print(df[['country', 'tenor', 'DU', 'RX', 'OAT']].head(5).to_string(index=False))
    
    # Compare methods for specific query
    print("\n4. Method comparison for BEL 7.5yr:")
    print("-"*60)
    
    comparison_data = []
    test_query = [('BEL', 7.5)]
    
    for method in methods_to_test:
        result = get_interpolated_hedge_ratios(
            test_query, results, method=method, 
            return_dataframe=False, verbose=False
        )
        
        row = {'Method': method}
        for hedge in ['DU', 'OE', 'RX', 'UB', 'OAT']:
            row[hedge] = result[0].get(hedge, 0.0)
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Test extrapolation
    print("\n5. Testing extrapolation (tenors outside [2, 50] range):")
    print("-"*60)
    
    extreme_queries = [
        ('BEL', 1.0),    # Below minimum (2yr)
        ('BEL', 60.0),   # Above maximum (50yr)
    ]
    
    for extrapolate_method in ['nearest', 'linear', 'zero']:
        print(f"\nExtrapolation method: {extrapolate_method}")
        df_extreme = get_interpolated_hedge_ratios(
            queries=extreme_queries,
            results=results,
            method='linear',
            extrapolate=extrapolate_method,
            return_dataframe=True,
            verbose=False
        )
        print(df_extreme[['country', 'tenor', 'DU', 'RX']].to_string(index=False))
    
    # Create visualization
    print("\n6. Creating visualization...")
    fig = plot_interpolated_hedge_ratios(
        queries=queries,
        results=results,
        methods=['linear', 'cubic', 'nearest'],
        hedges_to_plot=['DU', 'RX', 'OAT']
    )
    
    if fig:
        plt.savefig('interpolation_example.png', dpi=150, bbox_inches='tight')
        print("   Saved to interpolation_example.png")
    
    return all_results


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Run example
    all_results = example_interpolation()
    
    print("\n" + "="*80)
    print("INTERPOLATION EXAMPLE COMPLETED")
    print("="*80)
    print("\nKey features demonstrated:")
    print("1. Linear interpolation for smooth transitions")
    print("2. Cubic spline for smoother curves")
    print("3. Nearest neighbor for step-wise changes")
    print("4. Log-linear for exponential behavior")
    print("5. Extrapolation handling (nearest, linear, zero)")
    print("6. Batch processing of multiple queries")
    print("\nUse cases:")
    print("- Get hedge ratios for non-standard tenors (e.g., 7.5yr)")
    print("- Smooth transitions between tenor points")
    print("- Handle exotic tenor requests")
    print("- Create continuous hedge ratio curves")
