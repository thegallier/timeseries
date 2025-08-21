import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def apply_post_processing(
    W: jnp.ndarray,
    post_zero_threshold: float = 1e-6,
    post_processing_config: Dict = None,
    verbose: bool = True
) -> jnp.ndarray:
    """
    Apply post-processing to coefficients including zeroing out small values.
    
    Args:
        W: Coefficient matrix (n_features, n_outputs) or (n_windows, n_features, n_outputs)
        post_zero_threshold: Threshold below which coefficients are set to zero
        post_processing_config: Additional post-processing options:
            - 'zero_threshold': Threshold for zeroing (default: 1e-6)
            - 'relative_threshold': Zero if below x% of max in same output
            - 'sparsity_target': Target sparsity percentage
            - 'keep_top_k': Keep only top k coefficients per output
            - 'enforce_sign_consistency': Zero out if sign changes across windows
        verbose: Print post-processing statistics
        
    Returns:
        Post-processed coefficient matrix
    """
    
    if post_processing_config is None:
        post_processing_config = {}
    
    # Use config threshold if provided, otherwise use parameter
    zero_threshold = post_processing_config.get('zero_threshold', post_zero_threshold)
    
    W_processed = W.copy()
    original_nonzeros = jnp.sum(jnp.abs(W) > 1e-10)
    
    # Step 1: Zero out small absolute values
    if zero_threshold > 0:
        mask = jnp.abs(W_processed) > zero_threshold
        W_processed = W_processed * mask
        
        if verbose:
            zeroed = original_nonzeros - jnp.sum(mask)
            print(f"Post-processing: Zeroed {zeroed} coefficients below {zero_threshold:.2e}")
    
    # Step 2: Relative threshold (zero if below x% of max in same output)
    if 'relative_threshold' in post_processing_config:
        rel_threshold = post_processing_config['relative_threshold']
        
        if W.ndim == 2:
            # For each output, find max absolute value
            max_per_output = jnp.max(jnp.abs(W_processed), axis=0, keepdims=True)
            threshold_per_output = rel_threshold * max_per_output
            mask = jnp.abs(W_processed) > threshold_per_output
            W_processed = W_processed * mask
        else:
            # For windowed coefficients
            for w in range(W.shape[0]):
                max_per_output = jnp.max(jnp.abs(W_processed[w]), axis=0, keepdims=True)
                threshold_per_output = rel_threshold * max_per_output
                mask = jnp.abs(W_processed[w]) > threshold_per_output
                W_processed = W_processed.at[w].set(W_processed[w] * mask)
        
        if verbose:
            new_nonzeros = jnp.sum(jnp.abs(W_processed) > 1e-10)
            print(f"Post-processing: Relative threshold zeroed {original_nonzeros - new_nonzeros} more coefficients")
    
    # Step 3: Keep only top k coefficients per output
    if 'keep_top_k' in post_processing_config:
        k = post_processing_config['keep_top_k']
        
        if W.ndim == 2:
            n_features, n_outputs = W.shape
            for j in range(n_outputs):
                col = W_processed[:, j]
                # Get indices of top k by absolute value
                top_k_indices = jnp.argsort(jnp.abs(col))[-k:]
                mask = jnp.zeros(n_features)
                mask = mask.at[top_k_indices].set(1.0)
                W_processed = W_processed.at[:, j].set(col * mask)
        else:
            # For windowed coefficients
            for w in range(W.shape[0]):
                for j in range(W.shape[2]):
                    col = W_processed[w, :, j]
                    top_k_indices = jnp.argsort(jnp.abs(col))[-k:]
                    mask = jnp.zeros(W.shape[1])
                    mask = mask.at[top_k_indices].set(1.0)
                    W_processed = W_processed.at[w, :, j].set(col * mask)
        
        if verbose:
            print(f"Post-processing: Kept only top {k} coefficients per output")
    
    # Step 4: Enforce sign consistency across windows (for windowed coefficients)
    if 'enforce_sign_consistency' in post_processing_config and W.ndim == 3:
        if post_processing_config['enforce_sign_consistency']:
            # Check sign consistency across windows
            signs = jnp.sign(W_processed)
            mean_signs = jnp.mean(signs, axis=0)
            
            # Zero out if sign is not consistent (mean close to 0)
            inconsistent_mask = jnp.abs(mean_signs) < 0.5
            
            # Apply mask to all windows
            for w in range(W.shape[0]):
                W_processed = W_processed.at[w].set(W_processed[w] * (1 - inconsistent_mask))
            
            if verbose:
                n_inconsistent = jnp.sum(inconsistent_mask)
                print(f"Post-processing: Zeroed {n_inconsistent} coefficients with inconsistent signs")
    
    # Step 5: Target sparsity
    if 'sparsity_target' in post_processing_config:
        target_sparsity = post_processing_config['sparsity_target']
        current_sparsity = 1 - jnp.sum(jnp.abs(W_processed) > 1e-10) / W_processed.size
        
        if current_sparsity < target_sparsity:
            # Need to zero out more coefficients
            all_values = jnp.abs(W_processed).flatten()
            all_values = all_values[all_values > 0]  # Only non-zero values
            
            if len(all_values) > 0:
                # Find threshold that achieves target sparsity
                n_to_keep = int((1 - target_sparsity) * W_processed.size)
                if n_to_keep < len(all_values):
                    threshold = jnp.sort(all_values)[-(n_to_keep+1)]
                    mask = jnp.abs(W_processed) > threshold
                    W_processed = W_processed * mask
            
            if verbose:
                final_sparsity = 1 - jnp.sum(jnp.abs(W_processed) > 1e-10) / W_processed.size
                print(f"Post-processing: Achieved sparsity of {final_sparsity:.1%} (target: {target_sparsity:.1%})")
    
    # Final statistics
    if verbose:
        final_nonzeros = jnp.sum(jnp.abs(W_processed) > 1e-10)
        final_sparsity = 1 - final_nonzeros / W_processed.size
        print(f"Post-processing complete: {final_nonzeros} non-zero coefficients ({final_sparsity:.1%} sparsity)")
    
    return W_processed


def compute_true_r_squared_with_post_processing(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    W_raw: jnp.ndarray,
    window_size: int,
    stride: int,
    post_processing_config: Dict = None,
    verbose: bool = True
) -> Dict:
    """
    Compute true R² using post-processed coefficients (no penalties).
    
    Args:
        X: Input data (n_samples, n_features)
        Y: Output data (n_samples, n_outputs)
        W_raw: Raw coefficients from optimization
        window_size: Window size used
        stride: Stride used
        post_processing_config: Post-processing configuration
        verbose: Print details
        
    Returns:
        Dictionary with R² metrics computed on post-processed coefficients
    """
    
    # Apply post-processing
    W_processed = apply_post_processing(
        W_raw, 
        post_processing_config=post_processing_config,
        verbose=verbose
    )
    
    # Now compute R² using the post-processed coefficients
    n_samples = X.shape[0]
    n_windows = (n_samples - window_size) // stride + 1
    
    # Handle both averaged and windowed coefficients
    if W_processed.ndim == 2:
        W_expanded = jnp.expand_dims(W_processed, axis=0)
        W_expanded = jnp.repeat(W_expanded, n_windows, axis=0)
    else:
        W_expanded = W_processed
    
    # Create windows
    indices = np.arange(n_windows)[:, None] * stride + np.arange(window_size)[None, :]
    X_wins = X[indices]
    Y_wins = Y[indices]
    
    # Compute predictions using POST-PROCESSED coefficients
    Y_pred_wins = jnp.einsum('wij,wjk->wik', X_wins, W_expanded)
    
    # Compute R² metrics
    Y_pred_flat = Y_pred_wins.reshape(-1)
    Y_true_flat = Y_wins.reshape(-1)
    
    ss_res_total = jnp.sum((Y_true_flat - Y_pred_flat) ** 2)
    ss_tot_total = jnp.sum((Y_true_flat - jnp.mean(Y_true_flat)) ** 2)
    r2_overall = 1 - ss_res_total / (ss_tot_total + 1e-10)
    
    # Per window R²
    r2_per_window = []
    for w in range(n_windows):
        y_true = Y_wins[w].flatten()
        y_pred = Y_pred_wins[w].flatten()
        
        ss_res = jnp.sum((y_true - y_pred) ** 2)
        ss_tot = jnp.sum((y_true - jnp.mean(y_true)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)
        r2_per_window.append(float(r2))
    
    r2_per_window = jnp.array(r2_per_window)
    
    # Additional metrics
    mse = jnp.mean((Y_true_flat - Y_pred_flat) ** 2)
    rmse = jnp.sqrt(mse)
    mae = jnp.mean(jnp.abs(Y_true_flat - Y_pred_flat))
    
    results = {
        'r2_overall': float(r2_overall),
        'r2_per_window': np.array(r2_per_window),
        'r2_mean': float(jnp.mean(r2_per_window)),
        'r2_std': float(jnp.std(r2_per_window)),
        'r2_min': float(jnp.min(r2_per_window)),
        'r2_max': float(jnp.max(r2_per_window)),
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'W_processed': W_processed,
        'n_nonzero': int(jnp.sum(jnp.abs(W_processed) > 1e-10)),
        'sparsity': float(1 - jnp.sum(jnp.abs(W_processed) > 1e-10) / W_processed.size)
    }
    
    if verbose:
        print("\n" + "="*60)
        print("TRUE R² (Post-Processed Coefficients, No Penalties)")
        print("="*60)
        print(f"Overall R²: {r2_overall:.4f}")
        print(f"Mean R²: {results['r2_mean']:.4f} ± {results['r2_std']:.4f}")
        print(f"R² range: [{results['r2_min']:.4f}, {results['r2_max']:.4f}]")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"Sparsity: {results['sparsity']:.1%}")
    
    return results


def example_post_processing_and_r_squared():
    """
    Example showing post-processing options and true R² computation.
    """
    
    print("="*80)
    print("POST-PROCESSING AND TRUE R² COMPUTATION")
    print("="*80)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1256
    n_hedges = 7
    n_countries = 12
    n_tenors = 14
    
    # Create data with some structure
    X = np.random.randn(n_samples, n_hedges)
    true_W = np.random.randn(n_hedges, n_countries * n_tenors) * 0.3
    
    # Make true W sparse
    true_W[np.abs(true_W) < 0.2] = 0
    
    Y = X @ true_W + np.random.randn(n_samples, n_countries * n_tenors) * 0.3
    
    X = jnp.array(X)
    Y = jnp.array(Y)
    
    # Define country rules
    country_rules = {
        'BEL': {
            'allowed_countries': ['DEU', 'FRA'],
            'use_adjacent_only': False,
            'sign_constraints': {'RX': 'negative'}
        }
    }
    
    print("\n1. TESTING DIFFERENT POST-PROCESSING CONFIGURATIONS")
    print("-"*60)
    
    # Different post-processing configurations to test
    post_processing_configs = {
        'minimal': {
            'zero_threshold': 1e-10  # Almost no post-processing
        },
        'standard': {
            'zero_threshold': 1e-6   # Standard threshold
        },
        'aggressive': {
            'zero_threshold': 1e-4,  # Higher threshold
            'relative_threshold': 0.05  # Also use relative threshold
        },
        'top_k': {
            'zero_threshold': 1e-6,
            'keep_top_k': 3  # Keep only top 3 hedges per output
        },
        'target_sparsity': {
            'zero_threshold': 1e-6,
            'sparsity_target': 0.8  # Target 80% sparsity
        }
    }
    
    # For CVXPY, we can also set post_zero_threshold in the config
    cvxpy_configs = {
        'no_post': {
            'loss': 'squared',
            'post_zero_threshold': 0  # No post-processing
        },
        'standard_post': {
            'loss': 'squared',
            'post_zero_threshold': 1e-6  # Standard post-processing
        },
        'aggressive_post': {
            'loss': 'squared',
            'post_zero_threshold': 1e-4  # Aggressive post-processing
        },
        'with_tc': {
            'loss': 'squared',
            'transaction_costs': np.ones(n_hedges) * 0.01,
            'tc_lambda': 0.1,
            'post_zero_threshold': 1e-6  # Combine with transaction costs
        }
    }
    
    print("Post-processing configurations to test:")
    for name, config in post_processing_configs.items():
        print(f"  {name}: {config}")
    
    # Run regression
    print("\n2. RUNNING REGRESSION WITH JAX")
    print("-"*60)
    
    results = compute_all_countries_hedge_ratios_batch(
        X=X, Y=Y,
        country_rules=country_rules,
        window_size=200,
        stride=150,
        method='jax',
        constraint_method='penalty',
        verbose=False
    )
    
    W_raw = results['batch_results']['W_avg']
    print(f"Raw coefficients shape: {W_raw.shape}")
    print(f"Raw non-zeros: {jnp.sum(jnp.abs(W_raw) > 1e-10)}")
    
    # Test different post-processing configurations
    print("\n3. APPLYING DIFFERENT POST-PROCESSING CONFIGURATIONS")
    print("-"*60)
    
    r2_comparison = []
    
    for config_name, config in post_processing_configs.items():
        print(f"\n{config_name} configuration:")
        
        r2_results = compute_true_r_squared_with_post_processing(
            X, Y, W_raw,
            window_size=200,
            stride=150,
            post_processing_config=config,
            verbose=True
        )
        
        r2_comparison.append({
            'Config': config_name,
            'R² Overall': r2_results['r2_overall'],
            'R² Mean': r2_results['r2_mean'],
            'RMSE': r2_results['rmse'],
            'Non-zeros': r2_results['n_nonzero'],
            'Sparsity': r2_results['sparsity']
        })
    
    # Create comparison DataFrame
    df_comparison = pd.DataFrame(r2_comparison)
    
    print("\n4. POST-PROCESSING COMPARISON SUMMARY")
    print("-"*60)
    print(df_comparison.to_string(index=False))
    
    # Test CVXPY with different post-processing
    print("\n5. CVXPY WITH POST-PROCESSING OPTIONS")
    print("-"*60)
    
    try:
        import cvxpy
        
        cvxpy_results = []
        
        for config_name, config in cvxpy_configs.items():
            print(f"\nTesting CVXPY with {config_name}...")
            
            results_cvxpy = compute_all_countries_hedge_ratios_batch(
                X=X, Y=Y,
                country_rules=country_rules,
                window_size=200,
                stride=150,
                method='cvxpy',
                cvxpy_config=config,
                verbose=False
            )
            
            W_cvxpy = results_cvxpy['batch_results']['W_avg']
            
            # CVXPY already applies post_zero_threshold internally
            # But we can apply additional post-processing
            additional_config = {'zero_threshold': 0}  # Already done by CVXPY
            
            r2_results = compute_true_r_squared_with_post_processing(
                X, Y, W_cvxpy,
                window_size=200,
                stride=150,
                post_processing_config=additional_config,
                verbose=False
            )
            
            cvxpy_results.append({
                'Config': f'CVXPY_{config_name}',
                'R² Overall': r2_results['r2_overall'],
                'Non-zeros': r2_results['n_nonzero'],
                'Sparsity': r2_results['sparsity']
            })
            
            print(f"  R²: {r2_results['r2_overall']:.4f}, "
                  f"Sparsity: {r2_results['sparsity']:.1%}")
        
        df_cvxpy = pd.DataFrame(cvxpy_results)
        print("\nCVXPY Results:")
        print(df_cvxpy.to_string(index=False))
        
    except ImportError:
        print("CVXPY not available")
    
    # Visualize the effect of post-processing
    print("\n6. CREATING VISUALIZATION")
    print("-"*60)
    
    fig = visualize_post_processing_effects(r2_comparison, W_raw)
    
    if fig:
        plt.savefig('post_processing_r2.png', dpi=150, bbox_inches='tight')
        print("Saved to post_processing_r2.png")
    
    return df_comparison


def visualize_post_processing_effects(
    r2_comparison: List[Dict],
    W_raw: jnp.ndarray
) -> plt.Figure:
    """
    Visualize the effects of different post-processing configurations.
    """
    
    df = pd.DataFrame(r2_comparison)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: R² vs Sparsity trade-off
    ax1 = axes[0, 0]
    scatter = ax1.scatter(df['Sparsity'], df['R² Overall'], 
                         s=100, alpha=0.7, c=range(len(df)), cmap='viridis')
    
    for i, row in df.iterrows():
        ax1.annotate(row['Config'], 
                    (row['Sparsity'], row['R² Overall']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8)
    
    ax1.set_xlabel('Sparsity')
    ax1.set_ylabel('R² Overall')
    ax1.set_title('R² vs Sparsity Trade-off')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Number of non-zeros
    ax2 = axes[0, 1]
    bars = ax2.bar(df['Config'], df['Non-zeros'], alpha=0.7)
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('Number of Non-zero Coefficients')
    ax2.set_title('Sparsity by Configuration')
    ax2.tick_params(axis='x', rotation=45)
    
    # Color bars by sparsity
    for bar, sparsity in zip(bars, df['Sparsity']):
        if sparsity > 0.8:
            bar.set_color('green')
        elif sparsity > 0.6:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    # Plot 3: RMSE comparison
    ax3 = axes[1, 0]
    ax3.bar(df['Config'], df['RMSE'], alpha=0.7)
    ax3.set_xlabel('Configuration')
    ax3.set_ylabel('RMSE')
    ax3.set_title('Prediction Error by Configuration')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Summary text
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = "Post-Processing Options:\n" + "="*35 + "\n\n"
    summary_text += "zero_threshold: Basic threshold\n"
    summary_text += "  Set coefficients < threshold to 0\n\n"
    summary_text += "relative_threshold: % of max\n"
    summary_text += "  Zero if < x% of max in output\n\n"
    summary_text += "keep_top_k: Keep k largest\n"
    summary_text += "  Keep only top k per output\n\n"
    summary_text += "sparsity_target: Target sparsity\n"
    summary_text += "  Achieve specific sparsity %\n\n"
    summary_text += "Key Insight:\n"
    summary_text += "Post-processing creates sparser\n"
    summary_text += "solutions with small R² loss"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
            verticalalignment='top', fontsize=10, family='monospace')
    
    # Add best configuration highlight
    best_idx = df['R² Overall'].idxmax()
    best_config = df.loc[best_idx, 'Config']
    best_r2 = df.loc[best_idx, 'R² Overall']
    best_sparsity = df.loc[best_idx, 'Sparsity']
    
    ax4.text(0.1, 0.2, f"Best R²: {best_config}\n  R²={best_r2:.4f}\n  Sparsity={best_sparsity:.1%}",
            transform=ax4.transAxes, fontsize=11, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))
    
    plt.suptitle('Post-Processing Effects on Hedge Ratios', fontsize=14)
    plt.tight_layout()
    
    return fig


if __name__ == "__main__":
    df_comparison = example_post_processing_and_r_squared()
    
    print("\n" + "="*80)
    print("POST-PROCESSING EXAMPLE COMPLETED")
    print("="*80)
    print("\nKey findings:")
    print("1. Post-processing can significantly increase sparsity")
    print("2. R² is computed on final coefficients (no penalties)")
    print("3. Different thresholds create different sparsity/accuracy trade-offs")
    print("4. CVXPY's post_zero_threshold is applied automatically")
    print("5. Additional post-processing can be applied after any method")
    print("\nGenerated files:")
    print("  - post_processing_r2.png")


# Different CVXPY configs for different market conditions
configs = {
    'normal_market': {
        'loss': 'squared',
        'tc_lambda': 0.01  # Low transaction cost penalty
    },
    'volatile_market': {
        'loss': 'huber',
        'huber_delta': 0.5,  # Strong outlier resistance
        'tc_lambda': 0.1     # Higher penalty to reduce trading
    },
    'stressed_market': {
        'loss': 'huber',
        'huber_delta': 0.3,
        'dv01_neutral': True,  # Full hedging required
        'tc_lambda': 0.2       # Minimize rebalancing
    }
}


post_processing_config = {
    'zero_threshold': 1e-6,          # Absolute threshold
    'relative_threshold': 0.05,      # Zero if < 5% of max in same output
    'keep_top_k': 3,                 # Keep only top 3 coefficients per output
    'sparsity_target': 0.8,          # Achieve 80% sparsity
    'enforce_sign_consistency': True  # Zero if sign changes across windows
}

# Define post-processing for different needs
configs = {
    # Minimal cleaning
    'minimal': {
        'zero_threshold': 1e-10
    },
    
    # Standard cleaning
    'standard': {
        'zero_threshold': 1e-6
    },
    
    # Aggressive sparsification
    'sparse': {
        'zero_threshold': 1e-4,
        'relative_threshold': 0.1,  # Zero if < 10% of max
        'keep_top_k': 3              # Max 3 hedges per tenor
    },
    
    # Target specific sparsity
    'target_80': {
        'sparsity_target': 0.8  # Force 80% zeros
    }
}

# Run with post-processing
results = compute_all_countries_hedge_ratios_batch(...)
W_raw = results['batch_results']['W_avg']

# Apply post-processing
W_final = apply_post_processing(
    W_raw,
    post_processing_config=configs['sparse']
)

# Compute TRUE R² (on final coefficients, no penalties)
r2_results = compute_true_r_squared_with_post_processing(
    X, Y, W_raw,
    window_size=200,
    stride=150,
    post_processing_config=configs['sparse']
)

print(f"True R²: {r2_results['r2_overall']:.4f}")
print(f"Sparsity: {r2_results['sparsity']:.1%}")


# Compare methods with same post-processing
methods = ['jax_penalty', 'jax_exact', 'cvxpy']
post_config = {'zero_threshold': 1e-6}

for method in methods:
    results = run_method(method)
    W_raw = results['W_avg']
    
    # Apply SAME post-processing to all
    W_final = apply_post_processing(W_raw, post_config)
    
    # Compute R² on FINAL coefficients
    r2 = compute_r2(X, Y, W_final)
    
    print(f"{method}: R²={r2:.4f}, Sparsity={sparsity:.1%}")
/====

import jax.numpy as jnp
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt


def add_post_processing_to_results(
    results: Dict,
    X: jnp.ndarray,
    Y: jnp.ndarray,
    post_processing_config: Dict = None,
    verbose: bool = True
) -> Dict:
    """
    Add post-processing and true R² to existing results from batch computation.
    
    Args:
        results: Results from compute_all_countries_hedge_ratios_batch
        X: Original input data
        Y: Original output data  
        post_processing_config: Post-processing configuration
        verbose: Print details
        
    Returns:
        Updated results with post-processed coefficients and true R²
    """
    
    if 'batch_results' not in results:
        print("No batch_results found")
        return results
    
    batch_results = results['batch_results']
    
    # Get configuration
    config = batch_results.get('config', {})
    window_size = config.get('window_size', 200)
    stride = config.get('stride', 150)
    
    # Get raw coefficients
    if 'W_all' in batch_results:
        W_raw_all = batch_results['W_all']  # (n_windows, n_features, n_outputs)
        W_raw_avg = batch_results.get('W_avg', jnp.mean(W_raw_all, axis=0))
    elif 'W_avg' in batch_results:
        W_raw_avg = batch_results['W_avg']  # (n_features, n_outputs)
        W_raw_all = None
    else:
        print("No coefficients found")
        return results
    
    # Default post-processing if not specified
    if post_processing_config is None:
        post_processing_config = {'zero_threshold': 1e-6}
    
    if verbose:
        print("="*60)
        print("POST-PROCESSING AND TRUE R² COMPUTATION")
        print("="*60)
        print(f"Post-processing config: {post_processing_config}")
    
    # Apply post-processing to averaged coefficients
    W_processed_avg = apply_post_processing(
        W_raw_avg,
        post_processing_config=post_processing_config,
        verbose=verbose
    )
    
    # Store post-processed coefficients
    results['W_processed'] = W_processed_avg
    batch_results['W_processed'] = W_processed_avg
    
    # Apply post-processing to all windows if available
    if W_raw_all is not None:
        W_processed_all = apply_post_processing(
            W_raw_all,
            post_processing_config=post_processing_config,
            verbose=False
        )
        results['W_processed_all'] = W_processed_all
        batch_results['W_processed_all'] = W_processed_all
    
    # Compute true R² using post-processed coefficients
    r2_results = compute_true_r_squared_with_post_processing(
        X, Y, W_raw_avg,
        window_size=window_size,
        stride=stride,
        post_processing_config=post_processing_config,
        verbose=verbose
    )
    
    # Add R² metrics to results
    results['r2_true'] = r2_results
    batch_results['r2_true'] = r2_results
    
    # Add post-processing config for reference
    results['post_processing_config'] = post_processing_config
    
    # Update country results if present
    if 'country_results' in results:
        builder = HedgeConstraintBuilder()
        
        for country, country_data in results['country_results'].items():
            country_idx = builder.countries.index(country)
            n_tenors = len(builder.tenors)
            
            # Extract post-processed coefficients for this country
            start_idx = country_idx * n_tenors
            end_idx = (country_idx + 1) * n_tenors
            W_country_processed = W_processed_avg[:, start_idx:end_idx]
            
            # Store in country results
            country_data['W_processed'] = W_country_processed
            country_data['n_nonzero_processed'] = int(jnp.sum(jnp.abs(W_country_processed) > 1e-10))
            country_data['sparsity_processed'] = float(1 - country_data['n_nonzero_processed'] / W_country_processed.size)
    
    if verbose:
        print(f"\nPost-processing complete:")
        print(f"  Original non-zeros: {jnp.sum(jnp.abs(W_raw_avg) > 1e-10)}")
        print(f"  Post-processed non-zeros: {r2_results['n_nonzero']}")
        print(f"  Sparsity achieved: {r2_results['sparsity']:.1%}")
    
    return results


def compare_post_processing_on_methods(
    comparison_results: Dict,
    X: jnp.ndarray,
    Y: jnp.ndarray,
    post_processing_configs: Dict[str, Dict],
    verbose: bool = True
) -> pd.DataFrame:
    """
    Apply different post-processing configs to method comparison results.
    
    Args:
        comparison_results: Output from compare_regression_methods
        X: Input data
        Y: Output data
        post_processing_configs: Dict of config_name -> config
        verbose: Print details
        
    Returns:
        DataFrame comparing R² and sparsity across methods and configs
    """
    
    comparison_data = []
    
    for method_name, method_data in comparison_results.items():
        if 'results' not in method_data:
            continue
        
        # Get raw coefficients from the method
        results = method_data['results']
        if 'batch_results' in results and 'W_avg' in results['batch_results']:
            W_raw = results['batch_results']['W_avg']
        else:
            continue
        
        # Get window configuration
        config = results['batch_results'].get('config', {})
        window_size = config.get('window_size', 200)
        stride = config.get('stride', 150)
        
        # Apply each post-processing configuration
        for config_name, post_config in post_processing_configs.items():
            
            # Compute R² with this post-processing
            r2_results = compute_true_r_squared_with_post_processing(
                X, Y, W_raw,
                window_size=window_size,
                stride=stride,
                post_processing_config=post_config,
                verbose=False
            )
            
            comparison_data.append({
                'Method': method_name,
                'Post-Processing': config_name,
                'R² Overall': r2_results['r2_overall'],
                'R² Mean': r2_results['r2_mean'],
                'RMSE': r2_results['rmse'],
                'Non-zeros': r2_results['n_nonzero'],
                'Sparsity': r2_results['sparsity'],
                'Time (s)': method_data.get('computation_time', np.nan)
            })
    
    df = pd.DataFrame(comparison_data)
    
    if verbose:
        print("="*60)
        print("POST-PROCESSING COMPARISON ACROSS METHODS")
        print("="*60)
        
        # Create pivot table for better visualization
        pivot_r2 = df.pivot_table(
            values='R² Overall',
            index='Method',
            columns='Post-Processing'
        )
        
        print("\nR² by Method and Post-Processing:")
        print(pivot_r2.round(4))
        
        pivot_sparsity = df.pivot_table(
            values='Sparsity',
            index='Method',
            columns='Post-Processing'
        )
        
        print("\nSparsity by Method and Post-Processing:")
        print(pivot_sparsity.round(3))
        
        # Find best configurations
        best_r2 = df.loc[df['R² Overall'].idxmax()]
        print(f"\nBest R²: {best_r2['Method']} with {best_r2['Post-Processing']}")
        print(f"  R² = {best_r2['R² Overall']:.4f}, Sparsity = {best_r2['Sparsity']:.1%}")
        
        # Find sparsest with good R²
        good_r2 = df[df['R² Overall'] > df['R² Overall'].max() * 0.95]
        sparsest_good = good_r2.loc[good_r2['Sparsity'].idxmax()]
        print(f"\nSparsest with R² > 95% of best:")
        print(f"  {sparsest_good['Method']} with {sparsest_good['Post-Processing']}")
        print(f"  R² = {sparsest_good['R² Overall']:.4f}, Sparsity = {sparsest_good['Sparsity']:.1%}")
    
    return df


def example_correct_usage():
    """
    Example showing the correct way to access and use post-processed results.
    """
    
    print("="*80)
    print("CORRECT WAY TO USE POST-PROCESSING WITH RESULTS")
    print("="*80)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1256
    X = jnp.array(np.random.randn(n_samples, 7))
    Y = jnp.array(np.random.randn(n_samples, 168))
    
    # Country rules
    country_rules = {
        'BEL': {'allowed_countries': ['DEU', 'FRA'], 'use_adjacent_only': False},
        'NLD': {'allowed_countries': ['DEU'], 'use_adjacent_only': True}
    }
    
    print("\n1. SINGLE BATCH COMPUTATION (ONE VECTORIZED CALL)")
    print("-"*60)
    
    # ONE vectorized call that computes everything
    results = compute_all_countries_hedge_ratios_batch(
        X=X, Y=Y,
        country_rules=country_rules,
        window_size=200,
        stride=150,
        method='jax',
        constraint_method='penalty',
        verbose=False
    )
    
    print("Results structure after batch computation:")
    print(f"  results['batch_results']['W_avg'] shape: {results['batch_results']['W_avg'].shape}")
    print(f"  results['batch_results']['W_all'] shape: {results['batch_results']['W_all'].shape}")
    
    # Access raw coefficients (before post-processing)
    W_raw = results['batch_results']['W_avg']
    print(f"\nRaw coefficients:")
    print(f"  Non-zeros: {jnp.sum(jnp.abs(W_raw) > 1e-10)}")
    print(f"  Sparsity: {1 - jnp.sum(jnp.abs(W_raw) > 1e-10) / W_raw.size:.1%}")
    
    print("\n2. APPLYING POST-PROCESSING TO EXISTING RESULTS")
    print("-"*60)
    
    # Define post-processing configurations
    post_configs = {
        'standard': {'zero_threshold': 1e-6},
        'aggressive': {'zero_threshold': 1e-4, 'relative_threshold': 0.05},
        'sparse': {'zero_threshold': 1e-6, 'keep_top_k': 3}
    }
    
    # Apply each post-processing to the SAME results
    for config_name, config in post_configs.items():
        print(f"\n{config_name} post-processing:")
        
        # Add post-processing to existing results
        results_updated = add_post_processing_to_results(
            results.copy(),  # Work on a copy
            X, Y,
            post_processing_config=config,
            verbose=False
        )
        
        # Access post-processed coefficients
        W_processed = results_updated['W_processed']
        r2_true = results_updated['r2_true']['r2_overall']
        sparsity = results_updated['r2_true']['sparsity']
        
        print(f"  True R²: {r2_true:.4f}")
        print(f"  Sparsity: {sparsity:.1%}")
        print(f"  Non-zeros: {jnp.sum(jnp.abs(W_processed) > 1e-10)}")
    
    print("\n3. COMPARING METHODS (STILL ONE CALL PER METHOD)")
    print("-"*60)
    
    # Run comparison - each method is ONE vectorized call
    comparison = compare_regression_methods(
        X=X, Y=Y,
        country_rules=country_rules,
        window_size=200,
        stride=150,
        methods_to_test=['jax', 'vectorized'],
        verbose=False
    )
    
    print("Comparison structure:")
    for method_name in comparison.keys():
        W = comparison[method_name]['results']['batch_results']['W_avg']
        print(f"  {method_name}: W shape = {W.shape}")
    
    # Apply post-processing to all methods
    print("\n4. POST-PROCESSING ALL METHODS")
    print("-"*60)
    
    df_comparison = compare_post_processing_on_methods(
        comparison,
        X, Y,
        post_configs,
        verbose=True
    )
    
    print("\n5. ACCESSING SPECIFIC RESULTS")
    print("-"*60)
    
    # Example: Get Belgium 10yr RX coefficient after standard post-processing
    results_with_post = add_post_processing_to_results(
        results.copy(),
        X, Y,
        post_processing_config={'zero_threshold': 1e-6},
        verbose=False
    )
    
    builder = HedgeConstraintBuilder()
    bel_idx = builder.countries.index('BEL')
    tenor_10yr_idx = builder.tenors.index('10yr')
    rx_idx = builder.hedges.index('RX')
    
    # From post-processed coefficients
    W_processed = results_with_post['W_processed']
    bel_10yr_rx = W_processed[rx_idx, bel_idx * 14 + tenor_10yr_idx]
    
    print(f"Belgium 10yr RX coefficient (post-processed): {bel_10yr_rx:.6f}")
    
    # True R² for Belgium
    bel_r2 = results_with_post['country_results']['BEL'].get('r2_metrics', {})
    print(f"Belgium sparsity: {results_with_post['country_results']['BEL']['sparsity_processed']:.1%}")
    
    return results_with_post, df_comparison


def visualize_post_processing_from_results(
    results: Dict,
    X: jnp.ndarray,
    Y: jnp.ndarray
) -> plt.Figure:
    """
    Visualize post-processing effects directly from results.
    """
    
    # Apply different post-processing configs to same results
    configs = {
        'none': {'zero_threshold': 0},
        'minimal': {'zero_threshold': 1e-10},
        'standard': {'zero_threshold': 1e-6},
        'aggressive': {'zero_threshold': 1e-4},
        'sparse': {'zero_threshold': 1e-6, 'keep_top_k': 3}
    }
    
    metrics = []
    
    for name, config in configs.items():
        results_copy = results.copy()
        results_updated = add_post_processing_to_results(
            results_copy, X, Y, config, verbose=False
        )
        
        metrics.append({
            'Config': name,
            'R²': results_updated['r2_true']['r2_overall'],
            'Sparsity': results_updated['r2_true']['sparsity'],
            'Non-zeros': results_updated['r2_true']['n_nonzero']
        })
    
    df = pd.DataFrame(metrics)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # R² vs Sparsity
    ax1.scatter(df['Sparsity'], df['R²'], s=100, alpha=0.7)
    for _, row in df.iterrows():
        ax1.annotate(row['Config'], 
                    (row['Sparsity'], row['R²']),
                    xytext=(5, 5), textcoords='offset points')
    ax1.set_xlabel('Sparsity')
    ax1.set_ylabel('True R² (no penalties)')
    ax1.set_title('Post-Processing Trade-off')
    ax1.grid(True, alpha=0.3)
    
    # Bar chart of non-zeros
    ax2.bar(df['Config'], df['Non-zeros'], alpha=0.7)
    ax2.set_ylabel('Non-zero Coefficients')
    ax2.set_title('Sparsity by Configuration')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.suptitle('Post-Processing Effects on Single Batch Results')
    plt.tight_layout()
    
    return fig


if __name__ == "__main__":
    results_with_post, df_comparison = example_correct_usage()
    
    print("\n" + "="*80)
    print("KEY POINTS")
    print("="*80)
    print("\n1. Everything is computed in ONE vectorized call per method")
    print("2. Results contain both raw and can contain post-processed coefficients")
    print("3. R² is computed on post-processed coefficients (no penalties)")
    print("4. You can apply different post-processing to the SAME results")
    print("5. Access pattern:")
    print("   - Raw: results['batch_results']['W_avg']")
    print("   - Post-processed: results['W_processed']")
    print("   - True R²: results['r2_true']['r2_overall']")


# This ONE call computes everything for all countries/tenors/windows
results = compute_all_countries_hedge_ratios_batch(
    X=X, Y=Y,
    country_rules=country_rules,
    window_size=200,
    stride=150,
    method='jax'
)

# Raw coefficients (before post-processing)
W_raw = results['batch_results']['W_avg']  # Shape: (7, 168)
W_all_windows = results['batch_results']['W_all']  # Shape: (n_windows, 7, 168)

# Add post-processing to existing results
results = add_post_processing_to_results(
    results,
    X, Y,
    post_processing_config={'zero_threshold': 1e-6}
)

# Now access post-processed coefficients
W_processed = results['W_processed']  # Post-processed version
r2_true = results['r2_true']['r2_overall']  # R² on clean coefficients


# Compare methods - each method is ONE vectorized call
comparison = compare_regression_methods(
    X, Y, country_rules,
    window_size=200,
    stride=150,
    methods_to_test=['jax', 'vectorized', 'cvxpy']
)

# Each method's results are stored
jax_results = comparison['jax_penalty']['results']
vec_results = comparison['vectorized']['results']
cvx_results = comparison['cvxpy']['results']

# Apply post-processing to each
for method_name, method_data in comparison.items():
    W_raw = method_data['results']['batch_results']['W_avg']
    # Apply same post-processing to compare fairly
    r2_results = compute_true_r_squared_with_post_processing(
        X, Y, W_raw,
        window_size=200,
        stride=150,
        post_processing_config={'zero_threshold': 1e-6}
    )
    print(f"{method_name}: R²={r2_results['r2_overall']:.4f}")
