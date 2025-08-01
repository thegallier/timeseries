import jax
import jax.numpy as jnp
import numpy as np
import time

# ============= SIMPLE FAST VERSION =============

def fast_sliding_discovery(X, Y, window_size, stride, n_countries, n_tenors,
                          forced_group_mask=None,
                          discovery_config=None,
                          combination_mode='union',
                          penalty_strength=1e10):
    """
    Fast sliding window discovery with minimal complexity.
    Avoids JIT compilation issues while still being fast.
    """
    if discovery_config is None:
        discovery_config = {
            'consistency_threshold': 0.9,
            'magnitude_threshold': 0.05,
            'relative_threshold': 0.1,
            'check_relative': True
        }
    
    n_samples, n_features = X.shape
    n_outputs = Y.shape[1]
    n_windows = (n_samples - window_size) // stride + 1
    
    print(f"Fast discovery: {n_windows} windows, {n_features} features, {n_outputs} outputs")
    
    # Convert to numpy for window extraction (faster for this operation)
    X_np = np.array(X)
    Y_np = np.array(Y)
    
    # Phase 1: Extract and solve windows
    print("Phase 1: Solving windows...")
    start_time = time.time()
    
    # Pre-allocate arrays
    W_all = np.zeros((n_windows, n_features, n_outputs))
    masks_all = np.zeros((n_windows, n_features, n_outputs), dtype=bool)
    
    # Solve each window (vectorized operations within each window)
    for i in range(n_windows):
        start_idx = i * stride
        end_idx = start_idx + window_size
        
        X_win = jnp.array(X_np[start_idx:end_idx])
        Y_win = jnp.array(Y_np[start_idx:end_idx])
        
        # Fast Cholesky solve
        XtX = X_win.T @ X_win
        XtY = X_win.T @ Y_win
        L = jnp.linalg.cholesky(XtX + 1e-6 * jnp.eye(n_features))
        z = jax.scipy.linalg.solve_triangular(L, XtY, lower=True)
        W = jax.scipy.linalg.solve_triangular(L.T, z, lower=False)
        
        W_all[i] = W
        
        # Compute mask for this window
        W_abs = jnp.abs(W)
        mask_abs = W_abs < discovery_config['magnitude_threshold']
        
        if discovery_config['check_relative']:
            W_max = jnp.max(W_abs, axis=0, keepdims=True)
            mask_rel = W_abs < (discovery_config['relative_threshold'] * W_max)
            masks_all[i] = mask_abs & mask_rel
        else:
            masks_all[i] = mask_abs
    
    W_all = jnp.array(W_all)
    masks_all = jnp.array(masks_all)
    
    solve_time = time.time() - start_time
    print(f"  Solved in {solve_time:.3f}s ({n_windows/solve_time:.1f} windows/sec)")
    
    # Phase 2: Compute discovery mask
    print("Phase 2: Discovery analysis...")
    start_time = time.time()
    
    # Statistics
    consistency = jnp.mean(masks_all, axis=0)
    mean_mags = jnp.mean(jnp.abs(W_all), axis=0)
    max_mags = jnp.max(jnp.abs(W_all), axis=0)
    
    # Discovery criteria
    discovered = (
        (consistency >= discovery_config['consistency_threshold']) &
        (mean_mags < discovery_config['magnitude_threshold']) &
        (max_mags < 2 * discovery_config['magnitude_threshold'])
    )
    
    # Reshape to group structure
    discovery_mask = discovered.T.reshape(n_countries, n_tenors, n_features)
    
    discovery_time = time.time() - start_time
    print(f"  Discovery in {discovery_time:.3f}s")
    
    # Phase 3: Combine masks
    if forced_group_mask is None:
        combined_mask = discovery_mask
    else:
        if combination_mode == 'union':
            combined_mask = forced_group_mask | discovery_mask
        elif combination_mode == 'intersection':
            combined_mask = forced_group_mask & discovery_mask
        else:
            combined_mask = forced_group_mask | discovery_mask
    
    # Phase 4: Apply constraints
    print("Phase 3: Applying constraints...")
    start_time = time.time()
    
    mask_flat = combined_mask.transpose(2, 0, 1).reshape(n_features, n_outputs)
    W_constrained = np.zeros_like(W_all)
    
    # Apply constraints to each window
    for i in range(n_windows):
        start_idx = i * stride
        end_idx = start_idx + window_size
        
        X_win = jnp.array(X_np[start_idx:end_idx])
        Y_win = jnp.array(Y_np[start_idx:end_idx])
        
        XtX = X_win.T @ X_win
        XtY = X_win.T @ Y_win
        
        # Solve with penalties
        W_con = jnp.zeros((n_features, n_outputs))
        for j in range(n_outputs):
            penalty_diag = jnp.where(mask_flat[:, j], penalty_strength, 0.0)
            XtX_pen = XtX + jnp.diag(penalty_diag) + 1e-6 * jnp.eye(n_features)
            W_con = W_con.at[:, j].set(jnp.linalg.solve(XtX_pen, XtY[:, j]))
        
        W_constrained[i] = W_con
    
    W_constrained = jnp.array(W_constrained)
    constraint_time = time.time() - start_time
    print(f"  Constraints in {constraint_time:.3f}s")
    
    # Summary
    n_forced = jnp.sum(forced_group_mask) if forced_group_mask is not None else 0
    n_discovered = jnp.sum(discovery_mask)
    n_combined = jnp.sum(combined_mask)
    
    print(f"\nSummary:")
    print(f"  Forced constraints: {n_forced} ({100*n_forced/(n_features*n_outputs):.1f}%)")
    print(f"  Discovered: {n_discovered} ({100*n_discovered/(n_features*n_outputs):.1f}%)")
    print(f"  Combined: {n_combined} ({100*n_combined/(n_features*n_outputs):.1f}%)")
    print(f"  Total time: {solve_time + discovery_time + constraint_time:.3f}s")
    
    return {
        'combined_mask': combined_mask,
        'discovery_mask': discovery_mask,
        'W_all': W_all,
        'W_constrained': W_constrained,
        'consistency': consistency,
        'mean_magnitudes': mean_mags,
        'max_magnitudes': max_mags,
        'timing': {
            'solve': solve_time,
            'discovery': discovery_time,
            'constraints': constraint_time,
            'total': solve_time + discovery_time + constraint_time
        }
    }

# ============= BATCH VERSION FOR LARGE DATA =============

@jax.jit
def solve_window_batch(X_batch, Y_batch):
    """Solve a batch of windows in parallel"""
    def solve_single(X_win, Y_win):
        XtX = jnp.einsum('ni,nj->ij', X_win, X_win)
        XtY = jnp.einsum('ni,nj->ij', X_win, Y_win)
        L = jnp.linalg.cholesky(XtX + 1e-6 * jnp.eye(XtX.shape[0]))
        z = jax.scipy.linalg.solve_triangular(L, XtY, lower=True)
        return jax.scipy.linalg.solve_triangular(L.T, z, lower=False)
    
    return jax.vmap(solve_single)(X_batch, Y_batch)

def fast_batched_discovery(X, Y, window_size, stride, n_countries, n_tenors,
                          batch_size=50, **kwargs):
    """
    Batched version for better GPU utilization.
    Process windows in batches to leverage parallelism.
    """
    n_samples, n_features = X.shape
    n_outputs = Y.shape[1]
    n_windows = (n_samples - window_size) // stride + 1
    
    print(f"Batched discovery: {n_windows} windows in batches of {batch_size}")
    
    # Pre-allocate
    W_all = []
    
    # Process in batches
    start_time = time.time()
    for batch_start in range(0, n_windows, batch_size):
        batch_end = min(batch_start + batch_size, n_windows)
        batch_indices = jnp.arange(batch_start, batch_end)
        
        # Extract batch of windows
        X_windows = []
        Y_windows = []
        
        for i in batch_indices:
            start_idx = i * stride
            end_idx = start_idx + window_size
            X_windows.append(X[start_idx:end_idx])
            Y_windows.append(Y[start_idx:end_idx])
        
        X_batch = jnp.stack(X_windows)
        Y_batch = jnp.stack(Y_windows)
        
        # Solve batch
        W_batch = solve_window_batch(X_batch, Y_batch)
        W_all.append(W_batch)
        
        if batch_start % (batch_size * 5) == 0:
            print(f"  Processed {batch_end}/{n_windows} windows...")
    
    W_all = jnp.concatenate(W_all, axis=0)
    solve_time = time.time() - start_time
    print(f"Batch solving completed in {solve_time:.3f}s")
    
    # Continue with discovery...
    # (rest of the logic similar to fast_sliding_discovery)
    
    return W_all

# ============= COMPREHENSIVE VISUALIZATION =============

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# Set matplotlib to display inline in Jupyter
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except:
    pass

def visualize_discovery_results(result, X, Y, window_size, stride, 
                              n_countries, n_tenors, true_W=None,
                              save_path=None):
    """
    Comprehensive visualization of discovery results.
    Includes all the graphs from before plus performance metrics.
    
    Args:
        save_path: If provided, save figure to this path instead of showing
    """
    
    # Set style for better appearance
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Extract results
    W_all = result['W_all']
    W_constrained = result['W_constrained']
    discovery_mask = result['discovery_mask']
    combined_mask = result['combined_mask']
    consistency = result['consistency']
    mean_mags = result['mean_magnitudes']
    max_mags = result['max_magnitudes']
    
    n_windows, n_features, n_outputs = W_all.shape
    
    # Create large figure with subplots
    fig = plt.figure(figsize=(24, 20), dpi=100)
    gs = fig.add_gridspec(5, 4, hspace=0.3, wspace=0.3)
    
    # === Row 1: Coefficient Analysis ===
    
    # 1.1 Average coefficients (unconstrained)
    ax = fig.add_subplot(gs[0, 0])
    W_avg = jnp.mean(W_all, axis=0)
    im = ax.imshow(W_avg, cmap='RdBu_r', vmin=-2, vmax=2, aspect='auto')
    ax.set_title('Average Coefficients\n(Unconstrained)', fontsize=12)
    ax.set_xlabel('Output')
    ax.set_ylabel('Feature')
    plt.colorbar(im, ax=ax)
    
    # 1.2 Average coefficients (constrained)
    ax = fig.add_subplot(gs[0, 1])
    W_con_avg = jnp.mean(W_constrained, axis=0)
    im = ax.imshow(W_con_avg, cmap='RdBu_r', vmin=-2, vmax=2, aspect='auto')
    ax.set_title('Average Coefficients\n(With Constraints)', fontsize=12)
    ax.set_xlabel('Output')
    ax.set_ylabel('Feature')
    plt.colorbar(im, ax=ax)
    
    # 1.3 Coefficient stability (CV)
    ax = fig.add_subplot(gs[0, 2])
    cv = jnp.std(W_all, axis=0) / (jnp.abs(jnp.mean(W_all, axis=0)) + 1e-8)
    im = ax.imshow(jnp.log10(cv + 1e-8), cmap='viridis', aspect='auto')
    ax.set_title('Coefficient Stability\n(log10 CV)', fontsize=12)
    ax.set_xlabel('Output')
    ax.set_ylabel('Feature')
    plt.colorbar(im, ax=ax)
    
    # 1.4 True coefficients (if provided)
    ax = fig.add_subplot(gs[0, 3])
    if true_W is not None:
        im = ax.imshow(true_W, cmap='RdBu_r', vmin=-2, vmax=2, aspect='auto')
        ax.set_title('True Coefficients', fontsize=12)
        ax.set_xlabel('Output')
        ax.set_ylabel('Feature')
        plt.colorbar(im, ax=ax)
    else:
        ax.text(0.5, 0.5, 'True coefficients\nnot provided', 
                ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
    
    # === Row 2: Discovery Analysis ===
    
    # 2.1 Consistency heatmap
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(consistency, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='auto')
    ax.set_title('Consistency Across Windows\n(Fraction small)', fontsize=12)
    ax.set_xlabel('Output')
    ax.set_ylabel('Feature')
    plt.colorbar(im, ax=ax)
    
    # 2.2 Mean magnitudes
    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(mean_mags, cmap='viridis', aspect='auto')
    ax.set_title('Mean |Coefficient|', fontsize=12)
    ax.set_xlabel('Output')
    ax.set_ylabel('Feature')
    plt.colorbar(im, ax=ax)
    
    # 2.3 Discovery mask
    ax = fig.add_subplot(gs[1, 2])
    discovery_flat = discovery_mask.transpose(2, 0, 1).reshape(n_features, n_outputs)
    im = ax.imshow(discovery_flat, cmap='Reds', aspect='auto')
    ax.set_title('Discovered Zeros\n(Adaptive)', fontsize=12)
    ax.set_xlabel('Output')
    ax.set_ylabel('Feature')
    plt.colorbar(im, ax=ax)
    
    # 2.4 Combined mask
    ax = fig.add_subplot(gs[1, 3])
    combined_flat = combined_mask.transpose(2, 0, 1).reshape(n_features, n_outputs)
    im = ax.imshow(combined_flat, cmap='Reds', aspect='auto')
    ax.set_title('Combined Mask\n(Forced + Discovered)', fontsize=12)
    ax.set_xlabel('Output')
    ax.set_ylabel('Feature')
    plt.colorbar(im, ax=ax)
    
    # === Row 3: Constraint Violations ===
    
    # 3.1 Violation heatmap
    ax = fig.add_subplot(gs[2, 0:2])
    violations = jnp.abs(W_con_avg) * combined_flat
    im = ax.imshow(violations, cmap='Reds', vmin=0, vmax=0.1, aspect='auto')
    ax.set_title('Constraint Violations', fontsize=12)
    ax.set_xlabel('Output')
    ax.set_ylabel('Feature')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('|Coefficient| where should be zero')
    
    # 3.2 Violation evolution
    ax = fig.add_subplot(gs[2, 2:])
    max_violations = []
    mean_violations = []
    for i in range(n_windows):
        viol_i = jnp.abs(W_constrained[i]) * combined_flat
        max_violations.append(jnp.max(viol_i))
        mean_violations.append(jnp.mean(viol_i[combined_flat]))
    
    ax.semilogy(max_violations, 'r-', label='Max violation', linewidth=2)
    ax.semilogy(mean_violations, 'b--', label='Mean violation', linewidth=2)
    ax.set_xlabel('Window Index')
    ax.set_ylabel('Violation Magnitude (log scale)')
    ax.set_title('Constraint Violations Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # === Row 4: Coefficient Evolution ===
    
    # 4.1 Selected coefficient trajectories
    ax = fig.add_subplot(gs[3, 0:2])
    # Plot evolution of first 5 features for first output
    for f in range(min(5, n_features)):
        coef_evolution = W_constrained[:, f, 0]
        ax.plot(coef_evolution, label=f'Feature {f}', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Window Index')
    ax.set_ylabel('Coefficient Value')
    ax.set_title('Coefficient Evolution (Output 0)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 4.2 Sparsity evolution
    ax = fig.add_subplot(gs[3, 2])
    sparsity_per_window = []
    for i in range(n_windows):
        sparse_i = jnp.sum(jnp.abs(W_constrained[i]) < 1e-6) / (n_features * n_outputs)
        sparsity_per_window.append(sparse_i)
    
    ax.plot(sparsity_per_window, 'g-', linewidth=2)
    ax.axhline(y=jnp.mean(combined_flat), color='r', linestyle='--', 
               label=f'Target: {jnp.mean(combined_flat):.2f}')
    ax.set_xlabel('Window Index')
    ax.set_ylabel('Sparsity')
    ax.set_title('Sparsity Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4.3 R² evolution
    ax = fig.add_subplot(gs[3, 3])
    r2_values = []
    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        Y_pred = X[start:end] @ W_constrained[i]
        Y_true = Y[start:end]
        ss_res = jnp.sum((Y_true - Y_pred)**2)
        ss_tot = jnp.sum((Y_true - jnp.mean(Y_true))**2)
        r2 = 1 - ss_res / ss_tot
        r2_values.append(r2)
    
    ax.plot(r2_values, 'b-', linewidth=2)
    ax.set_xlabel('Window Index')
    ax.set_ylabel('R²')
    ax.set_title('Model Performance Over Time')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # === Row 5: Summary Statistics ===
    
    # 5.1 Performance comparison
    ax = fig.add_subplot(gs[4, 0])
    # Calculate overall R²
    Y_pred_all = X @ W_con_avg
    r2_overall = 1 - jnp.sum((Y - Y_pred_all)**2) / jnp.sum((Y - jnp.mean(Y))**2)
    
    # If true W provided, calculate oracle R²
    if true_W is not None:
        Y_pred_true = X @ true_W
        r2_oracle = 1 - jnp.sum((Y - Y_pred_true)**2) / jnp.sum((Y - jnp.mean(Y))**2)
        
        methods = ['Constrained', 'Oracle']
        r2_vals = [r2_overall, r2_oracle]
    else:
        methods = ['Constrained']
        r2_vals = [r2_overall]
    
    bars = ax.bar(methods, r2_vals, color=['blue', 'green'][:len(methods)])
    ax.set_ylabel('R²')
    ax.set_title('Overall Performance')
    ax.set_ylim([0, 1])
    for bar, val in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.3f}', ha='center', va='bottom')
    
    # 5.2 Sparsity breakdown
    ax = fig.add_subplot(gs[4, 1])
    n_forced = jnp.sum(combined_flat) - jnp.sum(discovery_flat)
    n_discovered_only = jnp.sum(discovery_flat & ~(combined_flat ^ discovery_flat))
    n_overlap = jnp.sum(discovery_flat & (combined_flat ^ discovery_flat))
    
    labels = ['Forced Only', 'Discovered Only', 'Overlap']
    sizes = [n_forced, n_discovered_only, n_overlap]
    colors = ['red', 'blue', 'purple']
    
    # Filter out zero values
    non_zero = [(l, s, c) for l, s, c in zip(labels, sizes, colors) if s > 0]
    if non_zero:
        labels, sizes, colors = zip(*non_zero)
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%')
    ax.set_title('Constraint Sources')
    
    # 5.3 Timing breakdown
    ax = fig.add_subplot(gs[4, 2])
    if 'timing' in result:
        times = [result['timing']['solve'], 
                result['timing']['discovery'], 
                result['timing']['constraints']]
        labels = ['Solve', 'Discovery', 'Constraints']
        ax.pie(times, labels=labels, autopct='%1.1f%%')
        ax.set_title(f'Time Breakdown\n(Total: {sum(times):.2f}s)')
    
    # 5.4 Summary text
    ax = fig.add_subplot(gs[4, 3])
    ax.axis('off')
    
    summary_text = f"""Summary Statistics
    
Windows: {n_windows}
Features: {n_features}
Outputs: {n_outputs}

R² Overall: {r2_overall:.4f}
Avg R² per window: {jnp.mean(jnp.array(r2_values)):.4f}

Sparsity: {100*jnp.mean(combined_flat):.1f}%
Discovered: {100*jnp.mean(discovery_flat):.1f}%

Max violation: {jnp.max(violations):.2e}
Mean violation: {jnp.mean(violations[combined_flat]):.2e}

Processing rate: {n_windows/result['timing']['total']:.1f} windows/sec
"""
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
            fontsize=11, va='top', family='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
    
    plt.suptitle('Comprehensive Discovery Analysis', fontsize=16)
    
    # Handle display
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    # Force display in Jupyter
    plt.show()
    plt.close('all')  # Clean up
    
    return fig

def plot_method_comparison(results_dict, X, Y, true_W=None, save_path=None):
    """
    Compare multiple methods/configurations.
    
    Args:
        results_dict: Dictionary of {method_name: result}
        save_path: If provided, save figure to this path
    """
    n_methods = len(results_dict)
    
    fig, axes = plt.subplots(3, n_methods, figsize=(5*n_methods, 12), dpi=100)
    if n_methods == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, (method_name, result) in enumerate(results_dict.items()):
        # Average coefficients
        ax = axes[0, idx]
        W_avg = jnp.mean(result['W_constrained'], axis=0)
        im = ax.imshow(W_avg, cmap='RdBu_r', vmin=-2, vmax=2, aspect='auto')
        ax.set_title(f'{method_name}\nCoefficients')
        ax.set_xlabel('Output')
        ax.set_ylabel('Feature')
        plt.colorbar(im, ax=ax)
        
        # Combined mask
        ax = axes[1, idx]
        mask_flat = result['combined_mask'].transpose(2, 0, 1).reshape(
            result['combined_mask'].shape[2], -1)
        im = ax.imshow(mask_flat, cmap='Reds', aspect='auto')
        ax.set_title('Zero Mask')
        ax.set_xlabel('Output')
        ax.set_ylabel('Feature')
        plt.colorbar(im, ax=ax)
        
        # Error from truth (if available)
        ax = axes[2, idx]
        if true_W is not None:
            error = W_avg - true_W
            im = ax.imshow(error, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
            ax.set_title('Error from Truth')
            ax.set_xlabel('Output')
            ax.set_ylabel('Feature')
            plt.colorbar(im, ax=ax)
        else:
            # Show performance metrics instead
            Y_pred = X @ W_avg
            r2 = 1 - jnp.sum((Y - Y_pred)**2) / jnp.sum((Y - jnp.mean(Y))**2)
            sparsity = jnp.mean(mask_flat)
            
            text = f"R²: {r2:.4f}\nSparsity: {100*sparsity:.1f}%"
            if 'timing' in result:
                text += f"\nTime: {result['timing']['total']:.2f}s"
            
            ax.text(0.5, 0.5, text, ha='center', va='center',
                   transform=ax.transAxes, fontsize=14,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
            ax.axis('off')
    
    plt.suptitle('Method Comparison', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    # Force display in Jupyter
    plt.show()
    plt.close('all')
    
    return fig

# Helper function to create multiple plots without overlap
def show_plots_separately(result, X, Y, window_size, stride, n_countries, n_tenors, true_W=None):
    """
    Show plots in separate cells to avoid Jupyter display issues
    """
    # Plot 1: Coefficients
    fig1, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    W_avg = jnp.mean(result['W_all'], axis=0)
    W_con_avg = jnp.mean(result['W_constrained'], axis=0)
    
    # Unconstrained
    ax = axes[0, 0]
    im = ax.imshow(W_avg, cmap='RdBu_r', vmin=-2, vmax=2)
    ax.set_title('Average Coefficients (Unconstrained)')
    plt.colorbar(im, ax=ax)
    
    # Constrained
    ax = axes[0, 1]
    im = ax.imshow(W_con_avg, cmap='RdBu_r', vmin=-2, vmax=2)
    ax.set_title('Average Coefficients (Constrained)')
    plt.colorbar(im, ax=ax)
    
    # Masks
    ax = axes[1, 0]
    discovery_flat = result['discovery_mask'].transpose(2, 0, 1).reshape(
        result['discovery_mask'].shape[2], -1)
    im = ax.imshow(discovery_flat, cmap='Reds')
    ax.set_title('Discovered Zeros')
    plt.colorbar(im, ax=ax)
    
    ax = axes[1, 1]
    combined_flat = result['combined_mask'].transpose(2, 0, 1).reshape(
        result['combined_mask'].shape[2], -1)
    im = ax.imshow(combined_flat, cmap='Reds')
    ax.set_title('Combined Mask')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.show()
    
    # Plot 2: Performance
    fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # R² evolution
    ax = axes[0]
    r2_values = []
    n_windows = len(result['W_constrained'])
    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        Y_pred = X[start:end] @ result['W_constrained'][i]
        Y_true = Y[start:end]
        r2 = 1 - jnp.sum((Y_true - Y_pred)**2) / jnp.sum((Y_true - jnp.mean(Y_true))**2)
        r2_values.append(r2)
    
    ax.plot(r2_values, 'b-', linewidth=2)
    ax.set_xlabel('Window Index')
    ax.set_ylabel('R²')
    ax.set_title('Model Performance Over Time')
    ax.grid(True, alpha=0.3)
    
    # Sparsity
    ax = axes[1]
    sparsity_per_window = []
    for i in range(n_windows):
        sparse_i = jnp.sum(jnp.abs(result['W_constrained'][i]) < 1e-6) / (
            result['W_constrained'].shape[1] * result['W_constrained'].shape[2])
        sparsity_per_window.append(sparse_i)
    
    ax.plot(sparsity_per_window, 'g-', linewidth=2)
    ax.set_xlabel('Window Index')
    ax.set_ylabel('Sparsity')
    ax.set_title('Sparsity Over Time')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig1, fig2

# ============= ENHANCED BENCHMARK WITH VISUALIZATION =============

def benchmark_with_visualization():
    """Enhanced benchmark with comprehensive visualization"""
    
    # Generate test data
    key = jax.random.PRNGKey(42)
    n_samples = 2000
    n_features = 15
    n_countries = 4
    n_tenors = 6
    n_outputs = n_countries * n_tenors
    
    print(f"Benchmark data: {n_samples} samples, {n_features} features, {n_outputs} outputs")
    
    # Create data with known structure
    X = jax.random.normal(key, (n_samples, n_features))
    W_true = jnp.zeros((n_features, n_outputs))
    
    # Country 0: Features 0,1,2 active
    for t in range(n_tenors):
        W_true = W_true.at[0, 0*n_tenors + t].set(2.0)
        W_true = W_true.at[1, 0*n_tenors + t].set(-1.5)
        W_true = W_true.at[2, 0*n_tenors + t].set(0.8)
    
    # Country 1: Features 3,4 active for some tenors
    for t in range(n_tenors//2):
        W_true = W_true.at[3, 1*n_tenors + t].set(1.2)
        W_true = W_true.at[4, 1*n_tenors + t].set(-0.9)
    
    # Country 2: Feature 5 active
    for t in range(n_tenors):
        W_true = W_true.at[5, 2*n_tenors + t].set(1.0)
    
    # Add some small random coefficients
    key, subkey = jax.random.split(key)
    W_true = W_true.at[7:9, :].set(0.1 * jax.random.normal(subkey, (2, n_outputs)))
    
    # Generate data
    Y = X @ W_true + 0.15 * jax.random.normal(key, (n_samples, n_outputs))
    
    # Create forced mask (prior knowledge)
    forced_mask = jnp.zeros((n_countries, n_tenors, n_features), dtype=bool)
    # We know features 10-14 are always zero
    forced_mask = forced_mask.at[:, :, 10:].set(True)
    # We know feature 6 is zero everywhere
    forced_mask = forced_mask.at[:, :, 6].set(True)
    
    window_size = 200
    stride = 100
    
    # Test different configurations
    results = {}
    
    # Configuration 1: Conservative discovery
    print("\n" + "="*60)
    print("Configuration 1: Conservative Discovery")
    print("="*60)
    
    results['Conservative'] = fast_sliding_discovery(
        X, Y, window_size, stride, n_countries, n_tenors,
        forced_group_mask=forced_mask,
        discovery_config={
            'consistency_threshold': 0.9,
            'magnitude_threshold': 0.05,
            'relative_threshold': 0.1,
            'check_relative': True
        },
        combination_mode='union'
    )
    
    # Configuration 2: Aggressive discovery
    print("\n" + "="*60)
    print("Configuration 2: Aggressive Discovery")
    print("="*60)
    
    results['Aggressive'] = fast_sliding_discovery(
        X, Y, window_size, stride, n_countries, n_tenors,
        forced_group_mask=forced_mask,
        discovery_config={
            'consistency_threshold': 0.7,
            'magnitude_threshold': 0.1,
            'relative_threshold': 0.15,
            'check_relative': True
        },
        combination_mode='union'
    )
    
    # Configuration 3: Forced only (no discovery)
    print("\n" + "="*60)
    print("Configuration 3: Forced Only")
    print("="*60)
    
    results['Forced Only'] = fast_sliding_discovery(
        X, Y, window_size, stride, n_countries, n_tenors,
        forced_group_mask=forced_mask,
        discovery_config={
            'consistency_threshold': 1.1,  # Impossible threshold
            'magnitude_threshold': 0.0,
            'relative_threshold': 0.0,
            'check_relative': False
        },
        combination_mode='union'
    )
    
    # Visualize individual results
    print("\n" + "="*60)
    print("VISUALIZATION: Conservative Discovery")
    print("="*60)
    
    fig1 = visualize_discovery_results(
        results['Conservative'], X, Y, window_size, stride,
        n_countries, n_tenors, true_W=W_true
    )
    
    # Compare methods
    print("\n" + "="*60)
    print("METHOD COMPARISON")
    print("="*60)
    
    fig2 = plot_method_comparison(results, X, Y, true_W=W_true)
    
    # Performance summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    for method, result in results.items():
        W_avg = jnp.mean(result['W_constrained'], axis=0)
        Y_pred = X @ W_avg
        r2 = 1 - jnp.sum((Y - Y_pred)**2) / jnp.sum((Y - jnp.mean(Y))**2)
        
        sparsity = jnp.mean(result['combined_mask'])
        discovered = jnp.sum(result['discovery_mask'])
        
        print(f"\n{method}:")
        print(f"  R²: {r2:.4f}")
        print(f"  Sparsity: {100*sparsity:.1f}%")
        print(f"  Discovered zeros: {discovered}")
        print(f"  Processing time: {result['timing']['total']:.2f}s")
        print(f"  Windows/sec: {len(result['W_all'])/result['timing']['total']:.1f}")
    
    return results, (fig1, fig2)
    """Benchmark the fast implementation"""
    
    # Generate test data
    key = jax.random.PRNGKey(42)
    n_samples = 2000
    n_features = 15
    n_countries = 4
    n_tenors = 6
    n_outputs = n_countries * n_tenors
    
    print(f"Benchmark data: {n_samples} samples, {n_features} features, {n_outputs} outputs")
    
    # Create data
    X = jax.random.normal(key, (n_samples, n_features))
    W_true = jnp.zeros((n_features, n_outputs))
    
    # Some non-zero coefficients
    key, subkey = jax.random.split(key)
    W_true = W_true.at[:5, :10].set(jax.random.normal(subkey, (5, 10)) * 2)
    W_true = W_true.at[7:10, 15:].set(jax.random.normal(subkey, (3, n_outputs-15)) * 1.5)
    
    Y = X @ W_true + 0.1 * jax.random.normal(key, (n_samples, n_outputs))
    
    # Forced mask
    forced_mask = jnp.zeros((n_countries, n_tenors, n_features), dtype=bool)
    forced_mask = forced_mask.at[:, :, 10:].set(True)  # Last 5 features forced to zero
    
    window_size = 200
    stride = 100
    
    # Run fast version
    print("\n" + "="*60)
    print("FAST IMPLEMENTATION")
    print("="*60)
    
    result = fast_sliding_discovery(
        X, Y, window_size, stride, n_countries, n_tenors,
        forced_group_mask=forced_mask,
        discovery_config={
            'consistency_threshold': 0.85,
            'magnitude_threshold': 0.08,
            'relative_threshold': 0.1,
            'check_relative': True
        }
    )
    
    # Evaluate performance
    W_avg = jnp.mean(result['W_constrained'], axis=0)
    Y_pred = X @ W_avg
    r2 = 1 - jnp.sum((Y - Y_pred)**2) / jnp.sum((Y - jnp.mean(Y))**2)
    
    print(f"\nPerformance:")
    print(f"  R²: {r2:.4f}")
    print(f"  Windows/sec: {len(result['W_all'])/result['timing']['total']:.1f}")
    
    # Test batched version
    print("\n" + "="*60)
    print("BATCHED VERSION TEST")
    print("="*60)
    
    W_batched = fast_batched_discovery(
        X, Y, window_size, stride, n_countries, n_tenors,
        batch_size=20
    )
    
    print(f"Shape check: {W_batched.shape}")
    print(f"Results match: {jnp.allclose(W_batched, result['W_all'], atol=1e-5)}")
    
    return result

# ============= PRACTICAL TIPS =============

def print_practical_tips():
    """Print practical optimization tips"""
    
    print("\n" + "="*60)
    print("PRACTICAL OPTIMIZATION TIPS")
    print("="*60)
    print("""
1. CHOOSE THE RIGHT APPROACH:
   - Small data (<1000 windows): Use simple fast version
   - Large data: Use batched version
   - Memory constraints: Process in chunks
   
2. PARAMETER TUNING:
   - window_size: Larger = more stable, slower
   - stride: Smaller = more windows, slower
   - batch_size: Tune for GPU memory (typically 20-100)
   
3. SPEED OPTIMIZATIONS:
   - Use float32 instead of float64
   - Pre-allocate arrays
   - Minimize data copies
   - Use numpy for indexing, JAX for math
   
4. QUALITY vs SPEED TRADEOFFS:
   - check_relative=False: 20% faster, may miss some patterns
   - Larger magnitude_threshold: Faster but less accurate
   - Lower consistency_threshold: More discoveries but more false positives
   
5. MONITORING:
   - Track windows/second
   - Monitor GPU utilization (nvidia-smi)
   - Check memory usage
   - Validate discoveries on test data
""")

if __name__ == "__main__":
    results, figures = benchmark_with_visualization()
    print_practical_tips()

import jax
import jax.numpy as jnp
import numpy as np
import time

# ============= SIMPLE FAST VERSION =============

def fast_sliding_discovery(X, Y, window_size, stride, n_countries, n_tenors,
                          forced_group_mask=None,
                          discovery_config=None,
                          combination_mode='union',
                          penalty_strength=1e10):
    """
    Fast sliding window discovery with minimal complexity.
    Avoids JIT compilation issues while still being fast.
    """
    if discovery_config is None:
        discovery_config = {
            'consistency_threshold': 0.9,
            'magnitude_threshold': 0.05,
            'relative_threshold': 0.1,
            'check_relative': True
        }
    
    n_samples, n_features = X.shape
    n_outputs = Y.shape[1]
    n_windows = (n_samples - window_size) // stride + 1
    
    print(f"Fast discovery: {n_windows} windows, {n_features} features, {n_outputs} outputs")
    
    # Convert to numpy for window extraction (faster for this operation)
    X_np = np.array(X)
    Y_np = np.array(Y)
    
    # Phase 1: Extract and solve windows
    print("Phase 1: Solving windows...")
    start_time = time.time()
    
    # Pre-allocate arrays
    W_all = np.zeros((n_windows, n_features, n_outputs))
    masks_all = np.zeros((n_windows, n_features, n_outputs), dtype=bool)
    
    # Solve each window (vectorized operations within each window)
    for i in range(n_windows):
        start_idx = i * stride
        end_idx = start_idx + window_size
        
        X_win = jnp.array(X_np[start_idx:end_idx])
        Y_win = jnp.array(Y_np[start_idx:end_idx])
        
        # Fast Cholesky solve
        XtX = X_win.T @ X_win
        XtY = X_win.T @ Y_win
        L = jnp.linalg.cholesky(XtX + 1e-6 * jnp.eye(n_features))
        z = jax.scipy.linalg.solve_triangular(L, XtY, lower=True)
        W = jax.scipy.linalg.solve_triangular(L.T, z, lower=False)
        
        W_all[i] = W
        
        # Compute mask for this window
        W_abs = jnp.abs(W)
        mask_abs = W_abs < discovery_config['magnitude_threshold']
        
        if discovery_config['check_relative']:
            W_max = jnp.max(W_abs, axis=0, keepdims=True)
            mask_rel = W_abs < (discovery_config['relative_threshold'] * W_max)
            masks_all[i] = mask_abs & mask_rel
        else:
            masks_all[i] = mask_abs
    
    W_all = jnp.array(W_all)
    masks_all = jnp.array(masks_all)
    
    solve_time = time.time() - start_time
    print(f"  Solved in {solve_time:.3f}s ({n_windows/solve_time:.1f} windows/sec)")
    
    # Phase 2: Compute discovery mask
    print("Phase 2: Discovery analysis...")
    start_time = time.time()
    
    # Statistics
    consistency = jnp.mean(masks_all, axis=0)
    mean_mags = jnp.mean(jnp.abs(W_all), axis=0)
    max_mags = jnp.max(jnp.abs(W_all), axis=0)
    
    # Discovery criteria
    discovered = (
        (consistency >= discovery_config['consistency_threshold']) &
        (mean_mags < discovery_config['magnitude_threshold']) &
        (max_mags < 2 * discovery_config['magnitude_threshold'])
    )
    
    # Reshape to group structure
    discovery_mask = discovered.T.reshape(n_countries, n_tenors, n_features)
    
    discovery_time = time.time() - start_time
    print(f"  Discovery in {discovery_time:.3f}s")
    
    # Phase 3: Combine masks
    if forced_group_mask is None:
        combined_mask = discovery_mask
    else:
        if combination_mode == 'union':
            combined_mask = forced_group_mask | discovery_mask
        elif combination_mode == 'intersection':
            combined_mask = forced_group_mask & discovery_mask
        else:
            combined_mask = forced_group_mask | discovery_mask
    
    # Phase 4: Apply constraints
    print("Phase 3: Applying constraints...")
    start_time = time.time()
    
    mask_flat = combined_mask.transpose(2, 0, 1).reshape(n_features, n_outputs)
    W_constrained = np.zeros_like(W_all)
    
    # Apply constraints to each window
    for i in range(n_windows):
        start_idx = i * stride
        end_idx = start_idx + window_size
        
        X_win = jnp.array(X_np[start_idx:end_idx])
        Y_win = jnp.array(Y_np[start_idx:end_idx])
        
        XtX = X_win.T @ X_win
        XtY = X_win.T @ Y_win
        
        # Solve with penalties
        W_con = jnp.zeros((n_features, n_outputs))
        for j in range(n_outputs):
            penalty_diag = jnp.where(mask_flat[:, j], penalty_strength, 0.0)
            XtX_pen = XtX + jnp.diag(penalty_diag) + 1e-6 * jnp.eye(n_features)
            W_con = W_con.at[:, j].set(jnp.linalg.solve(XtX_pen, XtY[:, j]))
        
        W_constrained[i] = W_con
    
    W_constrained = jnp.array(W_constrained)
    constraint_time = time.time() - start_time
    print(f"  Constraints in {constraint_time:.3f}s")
    
    # Summary
    n_forced = jnp.sum(forced_group_mask) if forced_group_mask is not None else 0
    n_discovered = jnp.sum(discovery_mask)
    n_combined = jnp.sum(combined_mask)
    
    print(f"\nSummary:")
    print(f"  Forced constraints: {n_forced} ({100*n_forced/(n_features*n_outputs):.1f}%)")
    print(f"  Discovered: {n_discovered} ({100*n_discovered/(n_features*n_outputs):.1f}%)")
    print(f"  Combined: {n_combined} ({100*n_combined/(n_features*n_outputs):.1f}%)")
    print(f"  Total time: {solve_time + discovery_time + constraint_time:.3f}s")
    
    return {
        'combined_mask': combined_mask,
        'discovery_mask': discovery_mask,
        'W_all': W_all,
        'W_constrained': W_constrained,
        'consistency': consistency,
        'mean_magnitudes': mean_mags,
        'max_magnitudes': max_mags,
        'timing': {
            'solve': solve_time,
            'discovery': discovery_time,
            'constraints': constraint_time,
            'total': solve_time + discovery_time + constraint_time
        }
    }

# ============= BATCH VERSION FOR LARGE DATA =============

@jax.jit
def solve_window_batch(X_batch, Y_batch):
    """Solve a batch of windows in parallel"""
    def solve_single(X_win, Y_win):
        XtX = jnp.einsum('ni,nj->ij', X_win, X_win)
        XtY = jnp.einsum('ni,nj->ij', X_win, Y_win)
        L = jnp.linalg.cholesky(XtX + 1e-6 * jnp.eye(XtX.shape[0]))
        z = jax.scipy.linalg.solve_triangular(L, XtY, lower=True)
        return jax.scipy.linalg.solve_triangular(L.T, z, lower=False)
    
    return jax.vmap(solve_single)(X_batch, Y_batch)

def fast_batched_discovery(X, Y, window_size, stride, n_countries, n_tenors,
                          batch_size=50, **kwargs):
    """
    Batched version for better GPU utilization.
    Process windows in batches to leverage parallelism.
    """
    n_samples, n_features = X.shape
    n_outputs = Y.shape[1]
    n_windows = (n_samples - window_size) // stride + 1
    
    print(f"Batched discovery: {n_windows} windows in batches of {batch_size}")
    
    # Pre-allocate
    W_all = []
    
    # Process in batches
    start_time = time.time()
    for batch_start in range(0, n_windows, batch_size):
        batch_end = min(batch_start + batch_size, n_windows)
        batch_indices = jnp.arange(batch_start, batch_end)
        
        # Extract batch of windows
        X_windows = []
        Y_windows = []
        
        for i in batch_indices:
            start_idx = i * stride
            end_idx = start_idx + window_size
            X_windows.append(X[start_idx:end_idx])
            Y_windows.append(Y[start_idx:end_idx])
        
        X_batch = jnp.stack(X_windows)
        Y_batch = jnp.stack(Y_windows)
        
        # Solve batch
        W_batch = solve_window_batch(X_batch, Y_batch)
        W_all.append(W_batch)
        
        if batch_start % (batch_size * 5) == 0:
            print(f"  Processed {batch_end}/{n_windows} windows...")
    
    W_all = jnp.concatenate(W_all, axis=0)
    solve_time = time.time() - start_time
    print(f"Batch solving completed in {solve_time:.3f}s")
    
    # Continue with discovery...
    # (rest of the logic similar to fast_sliding_discovery)
    
    return W_all

# ============= COMPREHENSIVE VISUALIZATION =============

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# Set matplotlib to display inline in Jupyter
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except:
    pass

def visualize_discovery_results(result, X, Y, window_size, stride, 
                              n_countries, n_tenors, true_W=None,
                              save_path=None):
    """
    Comprehensive visualization of discovery results.
    Includes all the graphs from before plus performance metrics.
    
    Args:
        save_path: If provided, save figure to this path instead of showing
    """
    
    # Set style for better appearance
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Extract results
    W_all = result['W_all']
    W_constrained = result['W_constrained']
    discovery_mask = result['discovery_mask']
    combined_mask = result['combined_mask']
    consistency = result['consistency']
    mean_mags = result['mean_magnitudes']
    max_mags = result['max_magnitudes']
    
    n_windows, n_features, n_outputs = W_all.shape
    
    # Create large figure with subplots
    fig = plt.figure(figsize=(24, 20), dpi=100)
    gs = fig.add_gridspec(5, 4, hspace=0.3, wspace=0.3)
    
    # === Row 1: Coefficient Analysis ===
    
    # 1.1 Average coefficients (unconstrained)
    ax = fig.add_subplot(gs[0, 0])
    W_avg = jnp.mean(W_all, axis=0)
    im = ax.imshow(W_avg, cmap='RdBu_r', vmin=-2, vmax=2, aspect='auto')
    ax.set_title('Average Coefficients\n(Unconstrained)', fontsize=12)
    ax.set_xlabel('Output')
    ax.set_ylabel('Feature')
    plt.colorbar(im, ax=ax)
    
    # 1.2 Average coefficients (constrained)
    ax = fig.add_subplot(gs[0, 1])
    W_con_avg = jnp.mean(W_constrained, axis=0)
    im = ax.imshow(W_con_avg, cmap='RdBu_r', vmin=-2, vmax=2, aspect='auto')
    ax.set_title('Average Coefficients\n(With Constraints)', fontsize=12)
    ax.set_xlabel('Output')
    ax.set_ylabel('Feature')
    plt.colorbar(im, ax=ax)
    
    # 1.3 Coefficient stability (CV)
    ax = fig.add_subplot(gs[0, 2])
    cv = jnp.std(W_all, axis=0) / (jnp.abs(jnp.mean(W_all, axis=0)) + 1e-8)
    im = ax.imshow(jnp.log10(cv + 1e-8), cmap='viridis', aspect='auto')
    ax.set_title('Coefficient Stability\n(log10 CV)', fontsize=12)
    ax.set_xlabel('Output')
    ax.set_ylabel('Feature')
    plt.colorbar(im, ax=ax)
    
    # 1.4 True coefficients (if provided)
    ax = fig.add_subplot(gs[0, 3])
    if true_W is not None:
        im = ax.imshow(true_W, cmap='RdBu_r', vmin=-2, vmax=2, aspect='auto')
        ax.set_title('True Coefficients', fontsize=12)
        ax.set_xlabel('Output')
        ax.set_ylabel('Feature')
        plt.colorbar(im, ax=ax)
    else:
        ax.text(0.5, 0.5, 'True coefficients\nnot provided', 
                ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
    
    # === Row 2: Discovery Analysis ===
    
    # 2.1 Consistency heatmap
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(consistency, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='auto')
    ax.set_title('Consistency Across Windows\n(Fraction small)', fontsize=12)
    ax.set_xlabel('Output')
    ax.set_ylabel('Feature')
    plt.colorbar(im, ax=ax)
    
    # 2.2 Mean magnitudes
    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(mean_mags, cmap='viridis', aspect='auto')
    ax.set_title('Mean |Coefficient|', fontsize=12)
    ax.set_xlabel('Output')
    ax.set_ylabel('Feature')
    plt.colorbar(im, ax=ax)
    
    # 2.3 Discovery mask
    ax = fig.add_subplot(gs[1, 2])
    discovery_flat = discovery_mask.transpose(2, 0, 1).reshape(n_features, n_outputs)
    im = ax.imshow(discovery_flat, cmap='Reds', aspect='auto')
    ax.set_title('Discovered Zeros\n(Adaptive)', fontsize=12)
    ax.set_xlabel('Output')
    ax.set_ylabel('Feature')
    plt.colorbar(im, ax=ax)
    
    # 2.4 Combined mask
    ax = fig.add_subplot(gs[1, 3])
    combined_flat = combined_mask.transpose(2, 0, 1).reshape(n_features, n_outputs)
    im = ax.imshow(combined_flat, cmap='Reds', aspect='auto')
    ax.set_title('Combined Mask\n(Forced + Discovered)', fontsize=12)
    ax.set_xlabel('Output')
    ax.set_ylabel('Feature')
    plt.colorbar(im, ax=ax)
    
    # === Row 3: Constraint Violations ===
    
    # 3.1 Violation heatmap
    ax = fig.add_subplot(gs[2, 0:2])
    violations = jnp.abs(W_con_avg) * combined_flat
    im = ax.imshow(violations, cmap='Reds', vmin=0, vmax=0.1, aspect='auto')
    ax.set_title('Constraint Violations', fontsize=12)
    ax.set_xlabel('Output')
    ax.set_ylabel('Feature')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('|Coefficient| where should be zero')
    
    # 3.2 Violation evolution
    ax = fig.add_subplot(gs[2, 2:])
    max_violations = []
    mean_violations = []
    for i in range(n_windows):
        viol_i = jnp.abs(W_constrained[i]) * combined_flat
        max_violations.append(jnp.max(viol_i))
        mean_violations.append(jnp.mean(viol_i[combined_flat]))
    
    ax.semilogy(max_violations, 'r-', label='Max violation', linewidth=2)
    ax.semilogy(mean_violations, 'b--', label='Mean violation', linewidth=2)
    ax.set_xlabel('Window Index')
    ax.set_ylabel('Violation Magnitude (log scale)')
    ax.set_title('Constraint Violations Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # === Row 4: Coefficient Evolution ===
    
    # 4.1 Selected coefficient trajectories
    ax = fig.add_subplot(gs[3, 0:2])
    # Plot evolution of first 5 features for first output
    for f in range(min(5, n_features)):
        coef_evolution = W_constrained[:, f, 0]
        ax.plot(coef_evolution, label=f'Feature {f}', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Window Index')
    ax.set_ylabel('Coefficient Value')
    ax.set_title('Coefficient Evolution (Output 0)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 4.2 Sparsity evolution
    ax = fig.add_subplot(gs[3, 2])
    sparsity_per_window = []
    for i in range(n_windows):
        sparse_i = jnp.sum(jnp.abs(W_constrained[i]) < 1e-6) / (n_features * n_outputs)
        sparsity_per_window.append(sparse_i)
    
    ax.plot(sparsity_per_window, 'g-', linewidth=2)
    ax.axhline(y=jnp.mean(combined_flat), color='r', linestyle='--', 
               label=f'Target: {jnp.mean(combined_flat):.2f}')
    ax.set_xlabel('Window Index')
    ax.set_ylabel('Sparsity')
    ax.set_title('Sparsity Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4.3 R² evolution
    ax = fig.add_subplot(gs[3, 3])
    r2_values = []
    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        Y_pred = X[start:end] @ W_constrained[i]
        Y_true = Y[start:end]
        ss_res = jnp.sum((Y_true - Y_pred)**2)
        ss_tot = jnp.sum((Y_true - jnp.mean(Y_true))**2)
        r2 = 1 - ss_res / ss_tot
        r2_values.append(r2)
    
    ax.plot(r2_values, 'b-', linewidth=2)
    ax.set_xlabel('Window Index')
    ax.set_ylabel('R²')
    ax.set_title('Model Performance Over Time')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # === Row 5: Summary Statistics ===
    
    # 5.1 Performance comparison
    ax = fig.add_subplot(gs[4, 0])
    # Calculate overall R²
    Y_pred_all = X @ W_con_avg
    r2_overall = 1 - jnp.sum((Y - Y_pred_all)**2) / jnp.sum((Y - jnp.mean(Y))**2)
    
    # If true W provided, calculate oracle R²
    if true_W is not None:
        Y_pred_true = X @ true_W
        r2_oracle = 1 - jnp.sum((Y - Y_pred_true)**2) / jnp.sum((Y - jnp.mean(Y))**2)
        
        methods = ['Constrained', 'Oracle']
        r2_vals = [r2_overall, r2_oracle]
    else:
        methods = ['Constrained']
        r2_vals = [r2_overall]
    
    bars = ax.bar(methods, r2_vals, color=['blue', 'green'][:len(methods)])
    ax.set_ylabel('R²')
    ax.set_title('Overall Performance')
    ax.set_ylim([0, 1])
    for bar, val in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.3f}', ha='center', va='bottom')
    
    # 5.2 Sparsity breakdown
    ax = fig.add_subplot(gs[4, 1])
    n_forced = jnp.sum(combined_flat) - jnp.sum(discovery_flat)
    n_discovered_only = jnp.sum(discovery_flat & ~(combined_flat ^ discovery_flat))
    n_overlap = jnp.sum(discovery_flat & (combined_flat ^ discovery_flat))
    
    labels = ['Forced Only', 'Discovered Only', 'Overlap']
    sizes = [n_forced, n_discovered_only, n_overlap]
    colors = ['red', 'blue', 'purple']
    
    # Filter out zero values
    non_zero = [(l, s, c) for l, s, c in zip(labels, sizes, colors) if s > 0]
    if non_zero:
        labels, sizes, colors = zip(*non_zero)
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%')
    ax.set_title('Constraint Sources')
    
    # 5.3 Timing breakdown
    ax = fig.add_subplot(gs[4, 2])
    if 'timing' in result:
        times = [result['timing']['solve'], 
                result['timing']['discovery'], 
                result['timing']['constraints']]
        labels = ['Solve', 'Discovery', 'Constraints']
        ax.pie(times, labels=labels, autopct='%1.1f%%')
        ax.set_title(f'Time Breakdown\n(Total: {sum(times):.2f}s)')
    
    # 5.4 Summary text
    ax = fig.add_subplot(gs[4, 3])
    ax.axis('off')
    
    summary_text = f"""Summary Statistics
    
Windows: {n_windows}
Features: {n_features}
Outputs: {n_outputs}

R² Overall: {r2_overall:.4f}
Avg R² per window: {jnp.mean(jnp.array(r2_values)):.4f}

Sparsity: {100*jnp.mean(combined_flat):.1f}%
Discovered: {100*jnp.mean(discovery_flat):.1f}%

Max violation: {jnp.max(violations):.2e}
Mean violation: {jnp.mean(violations[combined_flat]):.2e}

Processing rate: {n_windows/result['timing']['total']:.1f} windows/sec
"""
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
            fontsize=11, va='top', family='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
    
    plt.suptitle('Comprehensive Discovery Analysis', fontsize=16)
    
    # Handle display
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    # Force display in Jupyter
    plt.show()
    plt.close('all')  # Clean up
    
    return fig

def plot_method_comparison(results_dict, X, Y, true_W=None, save_path=None):
    """
    Compare multiple methods/configurations.
    
    Args:
        results_dict: Dictionary of {method_name: result}
        save_path: If provided, save figure to this path
    """
    n_methods = len(results_dict)
    
    fig, axes = plt.subplots(3, n_methods, figsize=(5*n_methods, 12), dpi=100)
    if n_methods == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, (method_name, result) in enumerate(results_dict.items()):
        # Average coefficients
        ax = axes[0, idx]
        W_avg = jnp.mean(result['W_constrained'], axis=0)
        im = ax.imshow(W_avg, cmap='RdBu_r', vmin=-2, vmax=2, aspect='auto')
        ax.set_title(f'{method_name}\nCoefficients')
        ax.set_xlabel('Output')
        ax.set_ylabel('Feature')
        plt.colorbar(im, ax=ax)
        
        # Combined mask
        ax = axes[1, idx]
        mask_flat = result['combined_mask'].transpose(2, 0, 1).reshape(
            result['combined_mask'].shape[2], -1)
        im = ax.imshow(mask_flat, cmap='Reds', aspect='auto')
        ax.set_title('Zero Mask')
        ax.set_xlabel('Output')
        ax.set_ylabel('Feature')
        plt.colorbar(im, ax=ax)
        
        # Error from truth (if available)
        ax = axes[2, idx]
        if true_W is not None:
            error = W_avg - true_W
            im = ax.imshow(error, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
            ax.set_title('Error from Truth')
            ax.set_xlabel('Output')
            ax.set_ylabel('Feature')
            plt.colorbar(im, ax=ax)
        else:
            # Show performance metrics instead
            Y_pred = X @ W_avg
            r2 = 1 - jnp.sum((Y - Y_pred)**2) / jnp.sum((Y - jnp.mean(Y))**2)
            sparsity = jnp.mean(mask_flat)
            
            text = f"R²: {r2:.4f}\nSparsity: {100*sparsity:.1f}%"
            if 'timing' in result:
                text += f"\nTime: {result['timing']['total']:.2f}s"
            
            ax.text(0.5, 0.5, text, ha='center', va='center',
                   transform=ax.transAxes, fontsize=14,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
            ax.axis('off')
    
    plt.suptitle('Method Comparison', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    # Force display in Jupyter
    plt.show()
    plt.close('all')
    
    return fig

# Helper function to create multiple plots without overlap
def show_plots_separately(result, X, Y, window_size, stride, n_countries, n_tenors, true_W=None):
    """
    Show plots in separate cells to avoid Jupyter display issues
    """
    # Plot 1: Coefficients
    fig1, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    W_avg = jnp.mean(result['W_all'], axis=0)
    W_con_avg = jnp.mean(result['W_constrained'], axis=0)
    
    # Unconstrained
    ax = axes[0, 0]
    im = ax.imshow(W_avg, cmap='RdBu_r', vmin=-2, vmax=2)
    ax.set_title('Average Coefficients (Unconstrained)')
    plt.colorbar(im, ax=ax)
    
    # Constrained
    ax = axes[0, 1]
    im = ax.imshow(W_con_avg, cmap='RdBu_r', vmin=-2, vmax=2)
    ax.set_title('Average Coefficients (Constrained)')
    plt.colorbar(im, ax=ax)
    
    # Masks
    ax = axes[1, 0]
    discovery_flat = result['discovery_mask'].transpose(2, 0, 1).reshape(
        result['discovery_mask'].shape[2], -1)
    im = ax.imshow(discovery_flat, cmap='Reds')
    ax.set_title('Discovered Zeros')
    plt.colorbar(im, ax=ax)
    
    ax = axes[1, 1]
    combined_flat = result['combined_mask'].transpose(2, 0, 1).reshape(
        result['combined_mask'].shape[2], -1)
    im = ax.imshow(combined_flat, cmap='Reds')
    ax.set_title('Combined Mask')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.show()
    
    # Plot 2: Performance
    fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # R² evolution
    ax = axes[0]
    r2_values = []
    n_windows = len(result['W_constrained'])
    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        Y_pred = X[start:end] @ result['W_constrained'][i]
        Y_true = Y[start:end]
        r2 = 1 - jnp.sum((Y_true - Y_pred)**2) / jnp.sum((Y_true - jnp.mean(Y_true))**2)
        r2_values.append(r2)
    
    ax.plot(r2_values, 'b-', linewidth=2)
    ax.set_xlabel('Window Index')
    ax.set_ylabel('R²')
    ax.set_title('Model Performance Over Time')
    ax.grid(True, alpha=0.3)
    
    # Sparsity
    ax = axes[1]
    sparsity_per_window = []
    for i in range(n_windows):
        sparse_i = jnp.sum(jnp.abs(result['W_constrained'][i]) < 1e-6) / (
            result['W_constrained'].shape[1] * result['W_constrained'].shape[2])
        sparsity_per_window.append(sparse_i)
    
    ax.plot(sparsity_per_window, 'g-', linewidth=2)
    ax.set_xlabel('Window Index')
    ax.set_ylabel('Sparsity')
    ax.set_title('Sparsity Over Time')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig1, fig2

# ============= ENHANCED BENCHMARK WITH VISUALIZATION =============

def benchmark_with_visualization():
    """Enhanced benchmark with comprehensive visualization"""
    
    # Generate test data
    key = jax.random.PRNGKey(42)
    n_samples = 2000
    n_features = 15
    n_countries = 4
    n_tenors = 6
    n_outputs = n_countries * n_tenors
    
    print(f"Benchmark data: {n_samples} samples, {n_features} features, {n_outputs} outputs")
    
    # Create data with known structure
    X = jax.random.normal(key, (n_samples, n_features))
    W_true = jnp.zeros((n_features, n_outputs))
    
    # Country 0: Features 0,1,2 active
    for t in range(n_tenors):
        W_true = W_true.at[0, 0*n_tenors + t].set(2.0)
        W_true = W_true.at[1, 0*n_tenors + t].set(-1.5)
        W_true = W_true.at[2, 0*n_tenors + t].set(0.8)
    
    # Country 1: Features 3,4 active for some tenors
    for t in range(n_tenors//2):
        W_true = W_true.at[3, 1*n_tenors + t].set(1.2)
        W_true = W_true.at[4, 1*n_tenors + t].set(-0.9)
    
    # Country 2: Feature 5 active
    for t in range(n_tenors):
        W_true = W_true.at[5, 2*n_tenors + t].set(1.0)
    
    # Add some small random coefficients
    key, subkey = jax.random.split(key)
    W_true = W_true.at[7:9, :].set(0.1 * jax.random.normal(subkey, (2, n_outputs)))
    
    # Generate data
    Y = X @ W_true + 0.15 * jax.random.normal(key, (n_samples, n_outputs))
    
    # Create forced mask (prior knowledge)
    forced_mask = jnp.zeros((n_countries, n_tenors, n_features), dtype=bool)
    # We know features 10-14 are always zero
    forced_mask = forced_mask.at[:, :, 10:].set(True)
    # We know feature 6 is zero everywhere
    forced_mask = forced_mask.at[:, :, 6].set(True)
    
    window_size = 200
    stride = 100
    
    # Test different configurations
    results = {}
    
    # Configuration 1: Conservative discovery
    print("\n" + "="*60)
    print("Configuration 1: Conservative Discovery")
    print("="*60)
    
    results['Conservative'] = fast_sliding_discovery(
        X, Y, window_size, stride, n_countries, n_tenors,
        forced_group_mask=forced_mask,
        discovery_config={
            'consistency_threshold': 0.9,
            'magnitude_threshold': 0.05,
            'relative_threshold': 0.1,
            'check_relative': True
        },
        combination_mode='union'
    )
    
    # Configuration 2: Aggressive discovery
    print("\n" + "="*60)
    print("Configuration 2: Aggressive Discovery")
    print("="*60)
    
    results['Aggressive'] = fast_sliding_discovery(
        X, Y, window_size, stride, n_countries, n_tenors,
        forced_group_mask=forced_mask,
        discovery_config={
            'consistency_threshold': 0.7,
            'magnitude_threshold': 0.1,
            'relative_threshold': 0.15,
            'check_relative': True
        },
        combination_mode='union'
    )
    
    # Configuration 3: Forced only (no discovery)
    print("\n" + "="*60)
    print("Configuration 3: Forced Only")
    print("="*60)
    
    results['Forced Only'] = fast_sliding_discovery(
        X, Y, window_size, stride, n_countries, n_tenors,
        forced_group_mask=forced_mask,
        discovery_config={
            'consistency_threshold': 1.1,  # Impossible threshold
            'magnitude_threshold': 0.0,
            'relative_threshold': 0.0,
            'check_relative': False
        },
        combination_mode='union'
    )
    
    # Visualize individual results
    print("\n" + "="*60)
    print("VISUALIZATION: Conservative Discovery")
    print("="*60)
    
    fig1 = visualize_discovery_results(
        results['Conservative'], X, Y, window_size, stride,
        n_countries, n_tenors, true_W=W_true
    )
    
    # Compare methods
    print("\n" + "="*60)
    print("METHOD COMPARISON")
    print("="*60)
    
    fig2 = plot_method_comparison(results, X, Y, true_W=W_true)
    
    # Performance summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    for method, result in results.items():
        W_avg = jnp.mean(result['W_constrained'], axis=0)
        Y_pred = X @ W_avg
        r2 = 1 - jnp.sum((Y - Y_pred)**2) / jnp.sum((Y - jnp.mean(Y))**2)
        
        sparsity = jnp.mean(result['combined_mask'])
        discovered = jnp.sum(result['discovery_mask'])
        
        print(f"\n{method}:")
        print(f"  R²: {r2:.4f}")
        print(f"  Sparsity: {100*sparsity:.1f}%")
        print(f"  Discovered zeros: {discovered}")
        print(f"  Processing time: {result['timing']['total']:.2f}s")
        print(f"  Windows/sec: {len(result['W_all'])/result['timing']['total']:.1f}")
    
    return results, (fig1, fig2)
    """Benchmark the fast implementation"""
    
    # Generate test data
    key = jax.random.PRNGKey(42)
    n_samples = 2000
    n_features = 15
    n_countries = 4
    n_tenors = 6
    n_outputs = n_countries * n_tenors
    
    print(f"Benchmark data: {n_samples} samples, {n_features} features, {n_outputs} outputs")
    
    # Create data
    X = jax.random.normal(key, (n_samples, n_features))
    W_true = jnp.zeros((n_features, n_outputs))
    
    # Some non-zero coefficients
    key, subkey = jax.random.split(key)
    W_true = W_true.at[:5, :10].set(jax.random.normal(subkey, (5, 10)) * 2)
    W_true = W_true.at[7:10, 15:].set(jax.random.normal(subkey, (3, n_outputs-15)) * 1.5)
    
    Y = X @ W_true + 0.1 * jax.random.normal(key, (n_samples, n_outputs))
    
    # Forced mask
    forced_mask = jnp.zeros((n_countries, n_tenors, n_features), dtype=bool)
    forced_mask = forced_mask.at[:, :, 10:].set(True)  # Last 5 features forced to zero
    
    window_size = 200
    stride = 100
    
    # Run fast version
    print("\n" + "="*60)
    print("FAST IMPLEMENTATION")
    print("="*60)
    
    result = fast_sliding_discovery(
        X, Y, window_size, stride, n_countries, n_tenors,
        forced_group_mask=forced_mask,
        discovery_config={
            'consistency_threshold': 0.85,
            'magnitude_threshold': 0.08,
            'relative_threshold': 0.1,
            'check_relative': True
        }
    )
    
    # Evaluate performance
    W_avg = jnp.mean(result['W_constrained'], axis=0)
    Y_pred = X @ W_avg
    r2 = 1 - jnp.sum((Y - Y_pred)**2) / jnp.sum((Y - jnp.mean(Y))**2)
    
    print(f"\nPerformance:")
    print(f"  R²: {r2:.4f}")
    print(f"  Windows/sec: {len(result['W_all'])/result['timing']['total']:.1f}")
    
    # Test batched version
    print("\n" + "="*60)
    print("BATCHED VERSION TEST")
    print("="*60)
    
    W_batched = fast_batched_discovery(
        X, Y, window_size, stride, n_countries, n_tenors,
        batch_size=20
    )
    
    print(f"Shape check: {W_batched.shape}")
    print(f"Results match: {jnp.allclose(W_batched, result['W_all'], atol=1e-5)}")
    
    return result

# ============= PRACTICAL TIPS =============

def print_practical_tips():
    """Print practical optimization tips"""
    
    print("\n" + "="*60)
    print("PRACTICAL OPTIMIZATION TIPS")
    print("="*60)
    print("""
1. CHOOSE THE RIGHT APPROACH:
   - Small data (<1000 windows): Use simple fast version
   - Large data: Use batched version
   - Memory constraints: Process in chunks
   
2. PARAMETER TUNING:
   - window_size: Larger = more stable, slower
   - stride: Smaller = more windows, slower
   - batch_size: Tune for GPU memory (typically 20-100)
   
3. SPEED OPTIMIZATIONS:
   - Use float32 instead of float64
   - Pre-allocate arrays
   - Minimize data copies
   - Use numpy for indexing, JAX for math
   
4. QUALITY vs SPEED TRADEOFFS:
   - check_relative=False: 20% faster, may miss some patterns
   - Larger magnitude_threshold: Faster but less accurate
   - Lower consistency_threshold: More discoveries but more false positives
   
5. MONITORING:
   - Track windows/second
   - Monitor GPU utilization (nvidia-smi)
   - Check memory usage
   - Validate discoveries on test data
""")

if __name__ == "__main__":
    results, figures = benchmark_with_visualization()
    print_practical_tips()


