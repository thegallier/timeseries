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
