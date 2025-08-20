import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns


def compute_true_r_squared(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    W: jnp.ndarray,
    window_size: int,
    stride: int,
    verbose: bool = True
) -> Dict:
    """
    Compute true R² without penalties using actual coefficients on actual data.
    
    Args:
        X: Input data (n_samples, n_features)
        Y: Output data (n_samples, n_outputs)
        W: Coefficients (n_features, n_outputs) or (n_windows, n_features, n_outputs)
        window_size: Window size used
        stride: Stride used
        verbose: Print details
        
    Returns:
        Dictionary with R² metrics
    """
    
    n_samples = X.shape[0]
    n_windows = (n_samples - window_size) // stride + 1
    
    # Handle both averaged and windowed coefficients
    if W.ndim == 2:
        # W is averaged across windows - expand it
        W_expanded = jnp.expand_dims(W, axis=0)
        W_expanded = jnp.repeat(W_expanded, n_windows, axis=0)
    else:
        # W is already (n_windows, n_features, n_outputs)
        W_expanded = W
    
    # Create windows
    indices = np.arange(n_windows)[:, None] * stride + np.arange(window_size)[None, :]
    X_wins = X[indices]  # (n_windows, window_size, n_features)
    Y_wins = Y[indices]  # (n_windows, window_size, n_outputs)
    
    # Compute predictions for each window
    Y_pred_wins = jnp.einsum('wij,wjk->wik', X_wins, W_expanded)
    
    # Compute R² for each window and output
    r2_per_window_output = []
    
    for w in range(n_windows):
        r2_outputs = []
        for o in range(Y.shape[1]):
            y_true = Y_wins[w, :, o]
            y_pred = Y_pred_wins[w, :, o]
            
            ss_res = jnp.sum((y_true - y_pred) ** 2)
            ss_tot = jnp.sum((y_true - jnp.mean(y_true)) ** 2)
            
            r2 = 1 - ss_res / (ss_tot + 1e-10)
            r2_outputs.append(float(r2))
        
        r2_per_window_output.append(r2_outputs)
    
    r2_per_window_output = jnp.array(r2_per_window_output)  # (n_windows, n_outputs)
    
    # Compute overall R² (all windows and outputs combined)
    Y_pred_flat = Y_pred_wins.reshape(-1)
    Y_true_flat = Y_wins.reshape(-1)
    
    ss_res_total = jnp.sum((Y_true_flat - Y_pred_flat) ** 2)
    ss_tot_total = jnp.sum((Y_true_flat - jnp.mean(Y_true_flat)) ** 2)
    r2_overall = 1 - ss_res_total / (ss_tot_total + 1e-10)
    
    # Compute R² per window (averaged across outputs)
    r2_per_window = jnp.mean(r2_per_window_output, axis=1)
    
    # Compute R² per output (averaged across windows)
    r2_per_output = jnp.mean(r2_per_window_output, axis=0)
    
    # Additional metrics
    mse_per_window = jnp.mean((Y_wins - Y_pred_wins) ** 2, axis=(1, 2))
    rmse_per_window = jnp.sqrt(mse_per_window)
    
    # Mean absolute error
    mae_per_window = jnp.mean(jnp.abs(Y_wins - Y_pred_wins), axis=(1, 2))
    
    results = {
        'r2_overall': float(r2_overall),
        'r2_per_window': np.array(r2_per_window),
        'r2_per_output': np.array(r2_per_output),
        'r2_per_window_output': r2_per_window_output,
        'r2_mean': float(jnp.mean(r2_per_window)),
        'r2_std': float(jnp.std(r2_per_window)),
        'r2_min': float(jnp.min(r2_per_window)),
        'r2_max': float(jnp.max(r2_per_window)),
        'mse_per_window': np.array(mse_per_window),
        'rmse_per_window': np.array(rmse_per_window),
        'mae_per_window': np.array(mae_per_window),
        'n_windows': n_windows,
        'n_outputs': Y.shape[1]
    }
    
    if verbose:
        print("="*60)
        print("TRUE R² ANALYSIS (without penalties)")
        print("="*60)
        print(f"Overall R²: {r2_overall:.4f}")
        print(f"Mean R² across windows: {results['r2_mean']:.4f} ± {results['r2_std']:.4f}")
        print(f"R² range: [{results['r2_min']:.4f}, {results['r2_max']:.4f}]")
        print(f"Mean RMSE: {jnp.mean(rmse_per_window):.4f}")
        print(f"Mean MAE: {jnp.mean(mae_per_window):.4f}")
    
    return results


def compute_r_squared_by_country_tenor(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    W_full: jnp.ndarray,
    window_size: int,
    stride: int,
    country: str = None,
    tenor: str = None,
    verbose: bool = True
) -> Dict:
    """
    Compute R² for specific country/tenor combinations.
    
    Args:
        X: Input data (n_samples, n_hedges)
        Y: Full output data (n_samples, n_countries * n_tenors)
        W_full: Full coefficients (n_hedges, n_countries * n_tenors) or with window dim
        window_size: Window size
        stride: Stride
        country: Optional specific country
        tenor: Optional specific tenor
        verbose: Print details
        
    Returns:
        Dictionary with detailed R² by country/tenor
    """
    
    builder = HedgeConstraintBuilder()
    n_countries = len(builder.countries)
    n_tenors = len(builder.tenors)
    n_samples = X.shape[0]
    n_windows = (n_samples - window_size) // stride + 1
    
    results = {}
    
    if country:
        # Specific country
        country_idx = builder.countries.index(country)
        start_idx = country_idx * n_tenors
        end_idx = (country_idx + 1) * n_tenors
        
        Y_country = Y[:, start_idx:end_idx]
        
        if W_full.ndim == 2:
            W_country = W_full[:, start_idx:end_idx]
        else:
            W_country = W_full[:, :, start_idx:end_idx]
        
        if tenor:
            # Specific tenor
            tenor_idx = builder.tenors.index(tenor)
            Y_specific = Y_country[:, tenor_idx:tenor_idx+1]
            
            if W_country.ndim == 2:
                W_specific = W_country[:, tenor_idx:tenor_idx+1]
            else:
                W_specific = W_country[:, :, tenor_idx:tenor_idx+1]
            
            r2_results = compute_true_r_squared(
                X, Y_specific, W_specific, 
                window_size, stride, verbose=False
            )
            
            results[f"{country}_{tenor}"] = r2_results
            
            if verbose:
                print(f"\n{country} - {tenor}:")
                print(f"  R²: {r2_results['r2_overall']:.4f}")
                print(f"  Mean window R²: {r2_results['r2_mean']:.4f}")
        else:
            # All tenors for country
            r2_results = compute_true_r_squared(
                X, Y_country, W_country,
                window_size, stride, verbose=False
            )
            
            results[country] = r2_results
            
            # Also compute per tenor
            results[f"{country}_by_tenor"] = {}
            for t_idx, t_name in enumerate(builder.tenors):
                Y_t = Y_country[:, t_idx:t_idx+1]
                if W_country.ndim == 2:
                    W_t = W_country[:, t_idx:t_idx+1]
                else:
                    W_t = W_country[:, :, t_idx:t_idx+1]
                
                r2_t = compute_true_r_squared(
                    X, Y_t, W_t,
                    window_size, stride, verbose=False
                )
                results[f"{country}_by_tenor"][t_name] = r2_t['r2_overall']
            
            if verbose:
                print(f"\n{country}:")
                print(f"  Overall R²: {r2_results['r2_overall']:.4f}")
                print(f"  Best tenor: {max(results[f'{country}_by_tenor'].items(), key=lambda x: x[1])}")
                print(f"  Worst tenor: {min(results[f'{country}_by_tenor'].items(), key=lambda x: x[1])}")
    else:
        # All countries
        for c_idx, c_name in enumerate(builder.countries):
            start_idx = c_idx * n_tenors
            end_idx = (c_idx + 1) * n_tenors
            
            Y_c = Y[:, start_idx:end_idx]
            if W_full.ndim == 2:
                W_c = W_full[:, start_idx:end_idx]
            else:
                W_c = W_full[:, :, start_idx:end_idx]
            
            r2_c = compute_true_r_squared(
                X, Y_c, W_c,
                window_size, stride, verbose=False
            )
            
            results[c_name] = {
                'r2_overall': r2_c['r2_overall'],
                'r2_mean': r2_c['r2_mean'],
                'r2_std': r2_c['r2_std']
            }
        
        if verbose:
            print("\nR² by Country:")
            print("-" * 40)
            for c_name, metrics in results.items():
                if isinstance(metrics, dict) and 'r2_overall' in metrics:
                    print(f"{c_name:8s}: R²={metrics['r2_overall']:.4f} "
                          f"(mean={metrics['r2_mean']:.4f} ± {metrics['r2_std']:.4f})")
    
    return results


def add_r_squared_to_results(
    results: Dict,
    X: jnp.ndarray,
    Y: jnp.ndarray,
    verbose: bool = True
) -> Dict:
    """
    Add proper R² calculations to existing results.
    
    Args:
        results: Results from compute_all_countries_hedge_ratios_batch
        X: Original input data
        Y: Original output data
        verbose: Print details
        
    Returns:
        Updated results dictionary with R² metrics
    """
    
    if 'batch_results' not in results:
        print("No batch_results found")
        return results
    
    batch_results = results['batch_results']
    config = batch_results.get('config', {})
    window_size = config.get('window_size', 200)
    stride = config.get('stride', 150)
    
    # Get coefficients
    if 'W_all' in batch_results:
        W = batch_results['W_all']  # (n_windows, n_features, n_outputs)
    elif 'W_avg' in batch_results:
        W = batch_results['W_avg']  # (n_features, n_outputs)
    else:
        print("No coefficients found")
        return results
    
    # Compute true R² without penalties
    r2_metrics = compute_true_r_squared(X, Y, W, window_size, stride, verbose=verbose)
    
    # Add to results
    results['r2_metrics'] = r2_metrics
    batch_results['r2_true'] = r2_metrics
    
    # Compute R² by country
    builder = HedgeConstraintBuilder()
    country_r2 = {}
    
    for country in results.get('country_results', {}).keys():
        country_metrics = compute_r_squared_by_country_tenor(
            X, Y, W, window_size, stride,
            country=country, verbose=False
        )
        country_r2[country] = country_metrics[country]
        
        # Add to country results
        if 'country_results' in results and country in results['country_results']:
            results['country_results'][country]['r2_metrics'] = country_metrics[country]
            results['country_results'][country]['r2_by_tenor'] = country_metrics.get(f"{country}_by_tenor", {})
    
    results['r2_by_country'] = country_r2
    
    if verbose:
        print("\n" + "="*60)
        print("R² SUMMARY BY COUNTRY")
        print("="*60)
        
        # Create summary DataFrame
        summary_data = []
        for country, metrics in country_r2.items():
            summary_data.append({
                'Country': country,
                'R² Overall': metrics['r2_overall'],
                'R² Mean': metrics['r2_mean'],
                'R² Std': metrics['r2_std']
            })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            df = df.sort_values('R² Overall', ascending=False)
            print(df.to_string(index=False))
    
    return results


def plot_r_squared_analysis(
    results: Dict,
    figsize: Tuple[int, int] = (15, 10),
    method_name: str = None
) -> plt.Figure:
    """
    Create comprehensive R² visualization.
    
    Args:
        results: Results with R² metrics
        figsize: Figure size
        method_name: Optional method name for title
        
    Returns:
        Matplotlib figure
    """
    
    if 'r2_metrics' not in results:
        print("No R² metrics found. Run add_r_squared_to_results first.")
        return None
    
    r2_metrics = results['r2_metrics']
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot 1: R² per window
    ax1 = fig.add_subplot(gs[0, :])
    window_indices = np.arange(r2_metrics['n_windows'])
    ax1.plot(window_indices, r2_metrics['r2_per_window'], 'b-', linewidth=2)
    ax1.fill_between(window_indices, 
                     r2_metrics['r2_per_window'] - np.std(r2_metrics['r2_per_window_output'], axis=1),
                     r2_metrics['r2_per_window'] + np.std(r2_metrics['r2_per_window_output'], axis=1),
                     alpha=0.3)
    ax1.axhline(y=r2_metrics['r2_overall'], color='r', linestyle='--', 
                label=f'Overall R²={r2_metrics["r2_overall"]:.4f}')
    ax1.set_xlabel('Window Index')
    ax1.set_ylabel('R²')
    title = 'R² Evolution Across Windows'
    if method_name:
        title += f' ({method_name})'
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: R² distribution
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(r2_metrics['r2_per_window'], bins=20, edgecolor='black', alpha=0.7)
    ax2.axvline(x=r2_metrics['r2_mean'], color='r', linestyle='--', 
                label=f'Mean={r2_metrics["r2_mean"]:.4f}')
    ax2.set_xlabel('R²')
    ax2.set_ylabel('Frequency')
    ax2.set_title('R² Distribution Across Windows')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: RMSE per window
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(window_indices, r2_metrics['rmse_per_window'], 'g-', linewidth=2)
    ax3.set_xlabel('Window Index')
    ax3.set_ylabel('RMSE')
    ax3.set_title('RMSE Across Windows')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: R² vs RMSE scatter
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.scatter(r2_metrics['r2_per_window'], r2_metrics['rmse_per_window'], alpha=0.6)
    ax4.set_xlabel('R²')
    ax4.set_ylabel('RMSE')
    ax4.set_title('R² vs RMSE Trade-off')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: R² by country (if available)
    ax5 = fig.add_subplot(gs[2, :2])
    if 'r2_by_country' in results:
        countries = list(results['r2_by_country'].keys())
        r2_values = [results['r2_by_country'][c]['r2_overall'] for c in countries]
        
        bars = ax5.bar(range(len(countries)), r2_values, alpha=0.7)
        
        # Color bars by performance
        for bar, r2 in zip(bars, r2_values):
            if r2 > 0.8:
                bar.set_color('green')
            elif r2 > 0.6:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        ax5.set_xticks(range(len(countries)))
        ax5.set_xticklabels(countries, rotation=45, ha='right')
        ax5.set_ylabel('R²')
        ax5.set_title('R² by Country')
        ax5.axhline(y=0.5, color='k', linestyle='--', alpha=0.3)
        ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Summary statistics
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    
    summary_text = "R² Summary Statistics\n" + "="*25 + "\n\n"
    summary_text += f"Overall R²: {r2_metrics['r2_overall']:.4f}\n"
    summary_text += f"Mean R²: {r2_metrics['r2_mean']:.4f}\n"
    summary_text += f"Std R²: {r2_metrics['r2_std']:.4f}\n"
    summary_text += f"Min R²: {r2_metrics['r2_min']:.4f}\n"
    summary_text += f"Max R²: {r2_metrics['r2_max']:.4f}\n\n"
    summary_text += f"Mean RMSE: {np.mean(r2_metrics['rmse_per_window']):.4f}\n"
    summary_text += f"Mean MAE: {np.mean(r2_metrics['mae_per_window']):.4f}\n"
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
            verticalalignment='top', fontsize=10, family='monospace')
    
    plt.suptitle('R² Performance Analysis (True Fit Without Penalties)', fontsize=14, y=1.02)
    
    return fig


def compare_r_squared_across_methods(
    comparison_results: Dict,
    X: jnp.ndarray,
    Y: jnp.ndarray,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compare true R² across different methods.
    
    Args:
        comparison_results: Output from compare_regression_methods
        X: Input data
        Y: Output data
        verbose: Print comparison
        
    Returns:
        DataFrame with R² comparison
    """
    
    comparison_data = []
    
    for method_name, method_data in comparison_results.items():
        if 'results' not in method_data:
            continue
        
        # Add R² metrics to each method's results
        updated_results = add_r_squared_to_results(
            method_data['results'], X, Y, verbose=False
        )
        
        if 'r2_metrics' in updated_results:
            r2_metrics = updated_results['r2_metrics']
            
            comparison_data.append({
                'Method': method_name,
                'R² Overall': r2_metrics['r2_overall'],
                'R² Mean': r2_metrics['r2_mean'],
                'R² Std': r2_metrics['r2_std'],
                'R² Min': r2_metrics['r2_min'],
                'R² Max': r2_metrics['r2_max'],
                'RMSE Mean': np.mean(r2_metrics['rmse_per_window']),
                'MAE Mean': np.mean(r2_metrics['mae_per_window']),
                'Time (s)': method_data.get('computation_time', np.nan)
            })
    
    df = pd.DataFrame(comparison_data)
    
    if verbose and not df.empty:
        print("="*80)
        print("R² COMPARISON ACROSS METHODS (True Fit Without Penalties)")
        print("="*80)
        
        # Sort by R² Overall
        df_sorted = df.sort_values('R² Overall', ascending=False)
        
        # Format for display
        display_df = df_sorted.copy()
        for col in ['R² Overall', 'R² Mean', 'R² Std', 'R² Min', 'R² Max']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
        for col in ['RMSE Mean', 'MAE Mean']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
        if 'Time (s)' in display_df.columns:
            display_df['Time (s)'] = display_df['Time (s)'].apply(lambda x: f"{x:.3f}")
        
        print(display_df.to_string(index=False))
        
        # Best method
        best_method = df_sorted.iloc[0]['Method']
        print(f"\nBest R² Performance: {best_method}")
        
        # Check if differences are significant
        if len(df) > 1:
            r2_range = df['R² Overall'].max() - df['R² Overall'].min()
            print(f"R² Range across methods: {r2_range:.4f}")
            
            if r2_range < 0.01:
                print("Note: All methods achieve very similar R² performance")
            elif r2_range > 0.1:
                print("Warning: Large differences in R² across methods")
    
    return df


def create_r_squared_report(
    results: Dict,
    X: jnp.ndarray,
    Y: jnp.ndarray,
    filename: str = 'r2_report.xlsx'
) -> None:
    """
    Create Excel report with R² analysis.
    
    Args:
        results: Results from batch computation
        X: Input data
        Y: Output data
        filename: Output filename
    """
    
    # Ensure R² metrics are computed
    if 'r2_metrics' not in results:
        results = add_r_squared_to_results(results, X, Y, verbose=False)
    
    builder = HedgeConstraintBuilder()
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        
        # Sheet 1: Overall summary
        summary_data = {
            'Metric': ['Overall R²', 'Mean R²', 'Std R²', 'Min R²', 'Max R²', 
                      'Mean RMSE', 'Mean MAE'],
            'Value': [
                results['r2_metrics']['r2_overall'],
                results['r2_metrics']['r2_mean'],
                results['r2_metrics']['r2_std'],
                results['r2_metrics']['r2_min'],
                results['r2_metrics']['r2_max'],
                np.mean(results['r2_metrics']['rmse_per_window']),
                np.mean(results['r2_metrics']['mae_per_window'])
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        # Sheet 2: R² by window
        window_data = pd.DataFrame({
            'Window': np.arange(results['r2_metrics']['n_windows']),
            'R²': results['r2_metrics']['r2_per_window'],
            'RMSE': results['r2_metrics']['rmse_per_window'],
            'MAE': results['r2_metrics']['mae_per_window']
        })
        window_data.to_excel(writer, sheet_name='By Window', index=False)
        
        # Sheet 3: R² by country
        if 'r2_by_country' in results:
            country_data = []
            for country, metrics in results['r2_by_country'].items():
                country_data.append({
                    'Country': country,
                    'R² Overall': metrics['r2_overall'],
                    'R² Mean': metrics['r2_mean'],
                    'R² Std': metrics['r2_std']
                })
            pd.DataFrame(country_data).to_excel(writer, sheet_name='By Country', index=False)
        
        # Sheet 4: R² by country-tenor
        if 'country_results' in results:
            ct_data = []
            for country, country_results in results['country_results'].items():
                if 'r2_by_tenor' in country_results:
                    for tenor, r2 in country_results['r2_by_tenor'].items():
                        ct_data.append({
                            'Country': country,
                            'Tenor': tenor,
                            'R²': r2
                        })
            
            if ct_data:
                df_ct = pd.DataFrame(ct_data)
                pivot = df_ct.pivot(index='Tenor', columns='Country', values='R²')
                pivot.to_excel(writer, sheet_name='Country-Tenor Matrix')
    
    print(f"R² report saved to {filename}")


# Example usage
def example_r_squared_analysis():
    """
    Example showing comprehensive R² analysis.
    """
    
    print("EXAMPLE: R² Analysis Without Penalties")
    print("="*80)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1256
    n_hedges = 7
    n_countries = 12
    n_tenors = 14
    
    # Create more realistic data with actual relationships
    X = np.random.randn(n_samples, n_hedges)
    
    # Create true coefficients
    true_W = np.random.randn(n_hedges, n_countries * n_tenors) * 0.3
    
    # Generate Y with noise
    Y = X @ true_W + np.random.randn(n_samples, n_countries * n_tenors) * 0.5
    
    X = jnp.array(X)
    Y = jnp.array(Y)
    
    country_rules = {
        'BEL': {
            'allowed_countries': ['DEU', 'FRA'],
            'use_adjacent_only': True,
            'sign_constraints': {'RX': 'negative'}
        },
        'DEU': {
            'allowed_countries': ['DEU'],
            'use_adjacent_only': False,
            'sign_constraints': {}
        }
    }
    
    # Run batch computation
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
    
    # Add true R² calculations
    print("\n2. Computing true R² (without penalties)...")
    print("-"*60)
    results = add_r_squared_to_results(results, X, Y, verbose=True)
    
    # Analyze specific country
    print("\n3. R² analysis for Belgium...")
    print("-"*60)
    bel_r2 = compute_r_squared_by_country_tenor(
        X, Y, 
        results['batch_results']['W_avg'],
        window_size=200,
        stride=150,
        country='BEL',
        verbose=True
    )
    
    # Compare methods
    print("\n4. Comparing R² across methods...")
    print("-"*60)
    
    comparison = compare_regression_methods(
        X=X, Y=Y,
        country_rules=country_rules,
        window_size=200,
        stride=150,
        methods_to_test=['jax', 'vectorized'],
        verbose=False
    )
    
    r2_comparison = compare_r_squared_across_methods(
        comparison, X, Y, verbose=True
    )
    
    # Create visualizations
    print("\n5. Creating R² visualizations...")
    fig = plot_r_squared_analysis(results, method_name='JAX Penalty')
    if fig:
        plt.savefig('r2_analysis.png', dpi=150, bbox_inches='tight')
        print("  Saved to r2_analysis.png")
    
    # Create report
    print("\n6. Creating R² report...")
    create_r_squared_report(results, X, Y, 'r2_report.xlsx')
    
    return results, r2_comparison


if __name__ == "__main__":
    results, r2_comparison = example_r_squared_analysis()
    
    print("\n" + "="*80)
    print("R² ANALYSIS COMPLETED")
    print("="*80)
    print("\nKey insights:")
    print("- True R² computed without penalty terms")
    print("- R² tracked across all windows and outputs")
    print("- Performance compared across methods")
    print("\nAccess R² metrics via:")
    print("  results['r2_metrics']['r2_overall'] - Overall R²")
    print("  results['r2_metrics']['r2_per_window'] - R² for each window")
    print("  results['r2_by_country'][country] - R² for specific country")
    print("\nGenerated files:")
    print("  - r2_analysis.png (visualization)")
    print("  - r2_report.xlsx (detailed report)")
