import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import time
from functools import partial

# ============= CORE SOLVERS =============

@jax.jit
def ols_kernel_cholesky(X_win, Y_win):
    """OLS using Cholesky decomposition for numerical stability"""
    XtX = jnp.einsum('wf,wg->fg', X_win, X_win)
    XtY = jnp.einsum('wf,wo->fo', X_win, Y_win)
    
    # Adaptive regularization
    reg = 1e-6 * jnp.maximum(1.0, jnp.trace(XtX) / XtX.shape[0])
    XtX_reg = XtX + reg * jnp.eye(XtX.shape[0])
    
    L = jnp.linalg.cholesky(XtX_reg)
    z = jax.scipy.linalg.solve_triangular(L, XtY, lower=True)
    return jax.scipy.linalg.solve_triangular(L.T, z, lower=False)

@jax.jit
def ols_with_penalty(X_win, Y_win, mask=None, penalty_strength=1e8):
    """Solve OLS with penalty on masked coefficients"""
    n_features = X_win.shape[1]
    n_outputs = Y_win.shape[1]
    
    XtX = jnp.einsum('wf,wg->fg', X_win, X_win)
    XtY = jnp.einsum('wf,wo->fo', X_win, Y_win)
    
    reg = 1e-6 * jnp.eye(XtX.shape[0])
    
    if mask is None:
        return jnp.linalg.solve(XtX + reg, XtY)
    
    # Solve for each output with its own penalty
    def solve_single_output(j):
        mask_j = mask[:, j]
        penalty_diag = jnp.where(mask_j, penalty_strength, 0.0)
        XtX_penalized = XtX + reg + jnp.diag(penalty_diag)
        
        L = jnp.linalg.cholesky(XtX_penalized)
        z = jax.scipy.linalg.solve_triangular(L, XtY[:, j], lower=True)
        return jax.scipy.linalg.solve_triangular(L.T, z, lower=False)
    
    W = jax.vmap(solve_single_output, in_axes=0, out_axes=1)(jnp.arange(n_outputs))
    return W

# ============= KKT SOLVER =============

def build_constraint_matrices(n_features, n_outputs, constraints, max_constraints):
    """
    Build constraint matrices for KKT system.
    
    Args:
        n_features: Number of features
        n_outputs: Number of outputs
        constraints: List of constraint specifications
        max_constraints: Maximum number of constraints
    
    Returns:
        A: Constraint matrix (max_constraints, n_features * n_outputs)
        b: Constraint RHS (max_constraints,)
        n_active: Number of active constraints
    """
    n_vars = n_features * n_outputs
    A = jnp.zeros((max_constraints, n_vars))
    b = jnp.zeros(max_constraints)
    
    constraint_idx = 0
    
    for constraint in constraints:
        if constraint_idx >= max_constraints:
            break
            
        ctype = constraint['type']
        
        if ctype == 'zero':
            # w[i,j] = 0
            i, j = constraint['feature'], constraint['output']
            idx = i * n_outputs + j
            A = A.at[constraint_idx, idx].set(1.0)
            b = b.at[constraint_idx].set(0.0)
            constraint_idx += 1
            
        elif ctype == 'value':
            # w[i,j] = v
            i, j, v = constraint['feature'], constraint['output'], constraint['value']
            idx = i * n_outputs + j
            A = A.at[constraint_idx, idx].set(1.0)
            b = b.at[constraint_idx].set(v)
            constraint_idx += 1
            
        elif ctype == 'opposite':
            # w[i1,j1] = -w[i2,j2]
            i1, j1 = constraint['feature1'], constraint['output1']
            i2, j2 = constraint['feature2'], constraint['output2']
            idx1 = i1 * n_outputs + j1
            idx2 = i2 * n_outputs + j2
            A = A.at[constraint_idx, idx1].set(1.0)
            A = A.at[constraint_idx, idx2].set(1.0)
            b = b.at[constraint_idx].set(0.0)
            constraint_idx += 1
            
        elif ctype == 'equal':
            # w[i1,j1] = w[i2,j2]
            i1, j1 = constraint['feature1'], constraint['output1']
            i2, j2 = constraint['feature2'], constraint['output2']
            idx1 = i1 * n_outputs + j1
            idx2 = i2 * n_outputs + j2
            A = A.at[constraint_idx, idx1].set(1.0)
            A = A.at[constraint_idx, idx2].set(-1.0)
            b = b.at[constraint_idx].set(0.0)
            constraint_idx += 1
            
        elif ctype == 'sum':
            # sum of specified coefficients = v
            indices = constraint['indices']  # List of (i,j) tuples
            value = constraint['value']
            for i, j in indices:
                idx = i * n_outputs + j
                A = A.at[constraint_idx, idx].set(1.0)
            b = b.at[constraint_idx].set(value)
            constraint_idx += 1
    
    return A, b, constraint_idx

def make_kkt_solver(n_features, n_outputs, max_constraints):
    """Create KKT solver for exact constraints"""
    
    @jax.jit
    def solve_with_kkt(XtX, XtY, constraints_matrix, constraints_rhs, n_active_constraints):
        n_vars = n_features * n_outputs
        
        XtY_flat = XtY.ravel()
        XtX_block = jnp.kron(jnp.eye(n_outputs), XtX)
        
        # Build KKT system
        KKT_top = jnp.hstack([XtX_block, constraints_matrix.T])
        KKT_bottom = jnp.hstack([constraints_matrix, jnp.zeros((max_constraints, max_constraints))])
        KKT = jnp.vstack([KKT_top, KKT_bottom])
        
        rhs = jnp.concatenate([XtY_flat, constraints_rhs])
        
        # Mask inactive constraints
        mask_constraints = jnp.arange(max_constraints) >= n_active_constraints
        mask_full = jnp.concatenate([jnp.zeros(n_vars, dtype=bool), mask_constraints])
        KKT = KKT + jnp.diag(mask_full.astype(jnp.float32) * 1e10)
        
        solution = jnp.linalg.solve(KKT, rhs)
        W_flat = solution[:n_vars]
        W = W_flat.reshape(n_features, n_outputs)
        
        return W
    
    return solve_with_kkt

# ============= GROUP CONSTRAINTS =============

def process_forced_group_mask(forced_group_mask, n_windows, n_countries, n_tenors):
    """Convert forced_group_mask to window-wise masks"""
    if forced_group_mask is None:
        return None
    
    # Broadcast to all windows
    mask = jnp.broadcast_to(
        forced_group_mask[None, :, :, :], 
        (n_windows, *forced_group_mask.shape)
    )
    # Reshape: (n_windows, n_countries, n_tenors, n_features) -> (n_windows, n_features, n_countries*n_tenors)
    mask = mask.transpose(0, 3, 1, 2).reshape(n_windows, forced_group_mask.shape[2], n_countries * n_tenors)
    return mask

def build_group_constraints(forced_group_mask, n_countries, n_tenors, constraint_type='zero'):
    """Build constraints from forced_group_mask"""
    constraints = []
    n_features = forced_group_mask.shape[2]
    
    for country in range(n_countries):
        for tenor in range(n_tenors):
            for feature in range(n_features):
                if forced_group_mask[country, tenor, feature]:
                    output_idx = country * n_tenors + tenor
                    if constraint_type == 'zero':
                        constraints.append({
                            'type': 'zero',
                            'feature': feature,
                            'output': output_idx
                        })
    
    return constraints

# ============= MAIN SLIDING WINDOW FUNCTION =============

def sliding_window_regression_enhanced(
    X, Y, 
    window_size, 
    stride,
    method='penalty',  # 'penalty' or 'kkt'
    penalty_strength=1e8,
    # Group constraints
    forced_group_mask=None,
    n_countries=None,
    n_tenors=None,
    # KKT constraints
    constraints=None,
    max_constraints=None,
    # Other options
    use_cholesky=True,
    adaptive_threshold=False,
    threshold_k=2.0
):
    """
    Enhanced sliding window regression with group constraints support.
    
    Args:
        X: (n_samples, n_features)
        Y: (n_samples, n_outputs) 
        window_size: Size of sliding window
        stride: Step between windows
        method: 'penalty' or 'kkt'
        penalty_strength: Penalty for masked coefficients
        forced_group_mask: (n_countries, n_tenors, n_features) boolean mask
        n_countries, n_tenors: Dimensions for group constraints
        constraints: List of constraint dicts for KKT
        max_constraints: Maximum constraints for KKT
        use_cholesky: Use Cholesky decomposition
        adaptive_threshold: Use adaptive thresholding
        threshold_k: Threshold factor for adaptive method
    """
    n_samples, n_features = X.shape
    n_outputs = Y.shape[1]
    n_windows = (n_samples - window_size) // stride + 1
    
    # Extract windows
    def get_window(start_idx):
        X_win = jax.lax.dynamic_slice(X, (start_idx, 0), (window_size, n_features))
        Y_win = jax.lax.dynamic_slice(Y, (start_idx, 0), (window_size, n_outputs))
        return X_win, Y_win
    
    start_indices = jnp.arange(n_windows) * stride
    X_wins, Y_wins = jax.vmap(get_window)(start_indices)
    
    # Process group mask if provided
    if forced_group_mask is not None:
        group_masks = process_forced_group_mask(forced_group_mask, n_windows, n_countries, n_tenors)
    else:
        group_masks = None
    
    if method == 'penalty':
        # Handle masks
        if group_masks is not None:
            masks = group_masks
        elif adaptive_threshold:
            # Adaptive thresholding
            def compute_adaptive_mask(X_win, Y_win):
                W_init = ols_kernel_cholesky(X_win, Y_win) if use_cholesky else ols_with_penalty(X_win, Y_win)
                W_flat = jnp.abs(W_init).ravel()
                median = jnp.median(W_flat)
                mad = jnp.median(jnp.abs(W_flat - median))
                threshold = threshold_k * mad * 1.4826
                return jnp.abs(W_init) < threshold
            
            masks = jax.vmap(compute_adaptive_mask)(X_wins, Y_wins)
        else:
            masks = jnp.zeros((n_windows, n_features, n_outputs), dtype=bool)
        
        # Solve with penalties
        solver = ols_with_penalty if not use_cholesky else partial(ols_with_penalty)
        W_all = jax.vmap(solver, in_axes=(0, 0, 0, None))(X_wins, Y_wins, masks, penalty_strength)
        
        return W_all, masks
    
    elif method == 'kkt':
        # Build constraints
        if constraints is None and forced_group_mask is not None:
            # Convert group mask to constraints
            constraints = build_group_constraints(forced_group_mask, n_countries, n_tenors)
        elif constraints is None:
            constraints = []
        
        if max_constraints is None:
            max_constraints = len(constraints) + 10
        
        # Build constraint matrices
        A, b, n_active = build_constraint_matrices(n_features, n_outputs, constraints, max_constraints)
        
        # Create KKT solver
        kkt_solver = make_kkt_solver(n_features, n_outputs, max_constraints)
        
        # Solve with KKT
        def solve_window_kkt(X_win, Y_win):
            XtX = jnp.einsum('wf,wg->fg', X_win, X_win)
            XtY = jnp.einsum('wf,wo->fo', X_win, Y_win)
            XtX = XtX + 1e-6 * jnp.eye(XtX.shape[0])
            return kkt_solver(XtX, XtY, A, b, n_active)
        
        W_all = jax.vmap(solve_window_kkt)(X_wins, Y_wins)
        
        # Create mask showing constrained coefficients
        masks = jnp.zeros((n_windows, n_features, n_outputs), dtype=bool)
        for constraint in constraints:
            if constraint['type'] == 'zero':
                i, j = constraint['feature'], constraint['output']
                masks = masks.at[:, i, j].set(True)
        
        return W_all, masks

# ============= COMPARISON WITH OLD METHOD =============

def compare_methods(X, Y, window_size, stride, forced_group_mask, n_countries, n_tenors):
    """Compare different methods including old implementation style"""
    
    results = {}
    
    # 1. Baseline - no constraints
    print("1. Running baseline (no constraints)...")
    start = time.time()
    W_baseline, _ = sliding_window_regression_enhanced(
        X, Y, window_size, stride, method='penalty', forced_group_mask=None
    )
    results['baseline'] = {
        'W': W_baseline,
        'time': time.time() - start,
        'name': 'Baseline'
    }
    
    # 2. Penalty with forced group mask
    print("2. Running penalty with forced group mask...")
    start = time.time()
    W_penalty_group, mask_penalty_group = sliding_window_regression_enhanced(
        X, Y, window_size, stride,
        method='penalty',
        forced_group_mask=forced_group_mask,
        n_countries=n_countries,
        n_tenors=n_tenors,
        penalty_strength=1e10
    )
    results['penalty_group'] = {
        'W': W_penalty_group,
        'mask': mask_penalty_group,
        'time': time.time() - start,
        'name': 'Penalty (Group Mask)'
    }
    
    # 3. KKT with forced group mask
    print("3. Running KKT with forced group mask...")
    start = time.time()
    W_kkt_group, mask_kkt_group = sliding_window_regression_enhanced(
        X, Y, window_size, stride,
        method='kkt',
        forced_group_mask=forced_group_mask,
        n_countries=n_countries,
        n_tenors=n_tenors
    )
    results['kkt_group'] = {
        'W': W_kkt_group,
        'mask': mask_kkt_group,
        'time': time.time() - start,
        'name': 'KKT (Group Mask)'
    }
    
    # 4. Adaptive threshold
    print("4. Running adaptive threshold...")
    start = time.time()
    W_adaptive, mask_adaptive = sliding_window_regression_enhanced(
        X, Y, window_size, stride,
        method='penalty',
        adaptive_threshold=True,
        threshold_k=2.0,
        penalty_strength=1e10
    )
    results['adaptive'] = {
        'W': W_adaptive,
        'mask': mask_adaptive,
        'time': time.time() - start,
        'name': 'Adaptive Threshold'
    }
    
    return results

# ============= VISUALIZATION =============

def visualize_comprehensive_comparison(results, X, Y, window_size, stride, true_W, forced_group_mask):
    """Create comprehensive visualization comparing all methods"""
    
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # Calculate R² for all methods
    r2_results = {}
    for key, result in results.items():
        r2_list = []
        W = result['W']
        for i in range(W.shape[0]):
            start = i * stride
            end = start + window_size
            Y_pred = X[start:end] @ W[i]
            r2 = 1 - jnp.sum((Y[start:end] - Y_pred)**2) / jnp.sum((Y[start:end] - jnp.mean(Y[start:end]))**2)
            r2_list.append(r2)
        r2_results[key] = jnp.array(r2_list)
    
    # Row 1: Coefficient heatmaps
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(true_W, cmap='RdBu_r', vmin=-2, vmax=2)
    ax.set_title('True Coefficients')
    ax.set_xlabel('Output')
    ax.set_ylabel('Feature')
    plt.colorbar(im, ax=ax)
    
    col = 1
    for key in ['baseline', 'penalty_group', 'kkt_group']:
        if key in results:
            ax = fig.add_subplot(gs[0, col])
            W_avg = jnp.mean(results[key]['W'], axis=0)
            im = ax.imshow(W_avg, cmap='RdBu_r', vmin=-2, vmax=2)
            ax.set_title(results[key]['name'])
            ax.set_xlabel('Output')
            ax.set_ylabel('Feature')
            plt.colorbar(im, ax=ax)
            col += 1
    
    # Row 2: Constraint violation heatmaps
    true_zero_mask = (true_W == 0.0)
    
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(true_zero_mask.astype(float), cmap='Greys', vmin=0, vmax=1)
    ax.set_title('True Zero Locations')
    ax.set_xlabel('Output')
    ax.set_ylabel('Feature')
    
    col = 1
    for key in ['penalty_group', 'kkt_group', 'adaptive']:
        if key in results and 'mask' in results[key]:
            ax = fig.add_subplot(gs[1, col])
            W_avg = jnp.mean(results[key]['W'], axis=0)
            mask = results[key]['mask'][0]  # First window mask
            violations = jnp.abs(W_avg) * mask
            im = ax.imshow(violations, cmap='Reds', vmin=0, vmax=0.1)
            ax.set_title(f'{results[key]["name"]}\nViolations')
            ax.set_xlabel('Output')
            ax.set_ylabel('Feature')
            plt.colorbar(im, ax=ax, format='%.3f')
            col += 1
    
    # Row 3: Performance metrics
    # R² over time
    ax = fig.add_subplot(gs[2, 0:2])
    for key, r2 in r2_results.items():
        ax.plot(r2, label=results[key]['name'], linewidth=2)
    ax.set_xlabel('Window Index')
    ax.set_ylabel('R²')
    ax.set_title('R² Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.5, 1.0])
    
    # Average R² bar chart
    ax = fig.add_subplot(gs[2, 2])
    names = [results[k]['name'] for k in r2_results.keys()]
    avg_r2 = [jnp.mean(r2) for r2 in r2_results.values()]
    bars = ax.bar(range(len(names)), avg_r2)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Average R²')
    ax.set_title('Average Performance')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, avg_r2):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom')
    
    # Computation time
    ax = fig.add_subplot(gs[2, 3])
    times = [results[k]['time'] for k in r2_results.keys()]
    bars = ax.bar(range(len(names)), times)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Computation Time')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Row 4: Detailed violation analysis
    # Max violations over time
    ax = fig.add_subplot(gs[3, 0:2])
    for key in ['penalty_group', 'kkt_group']:
        if key in results and 'mask' in results[key]:
            W = results[key]['W']
            mask = results[key]['mask']
            max_violations = []
            for i in range(W.shape[0]):
                violations = jnp.abs(W[i]) * mask[i]
                max_violations.append(jnp.max(violations))
            ax.semilogy(max_violations, label=results[key]['name'], linewidth=2)
    
    ax.set_xlabel('Window Index')
    ax.set_ylabel('Max Violation (log scale)')
    ax.set_title('Maximum Constraint Violations Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Sparsity analysis
    ax = fig.add_subplot(gs[3, 2:])
    sparsity_data = []
    perf_data = []
    names_sparse = []
    
    for key in results:
        if 'mask' in results[key]:
            sparsity = jnp.mean(results[key]['mask'])
            perf = jnp.mean(r2_results[key])
            sparsity_data.append(sparsity)
            perf_data.append(perf)
            names_sparse.append(results[key]['name'])
    
    ax.scatter(sparsity_data, perf_data, s=200)
    for i, name in enumerate(names_sparse):
        ax.annotate(name, (sparsity_data[i], perf_data[i]), 
                   xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Sparsity (Fraction of Zeros)')
    ax.set_ylabel('Average R²')
    ax.set_title('Sparsity vs Performance Trade-off')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("COMPREHENSIVE METHOD COMPARISON")
    print("="*80)
    
    print("\n1. PERFORMANCE (Average R²):")
    for key in results:
        print(f"   {results[key]['name']:25s}: {jnp.mean(r2_results[key]):>8.4f}")
    
    print("\n2. COMPUTATION TIME:")
    for key in results:
        print(f"   {results[key]['name']:25s}: {results[key]['time']:>8.4f} seconds")
    
    print("\n3. CONSTRAINT SATISFACTION:")
    for key in ['penalty_group', 'kkt_group']:
        if key in results and 'mask' in results[key]:
            W = results[key]['W']
            mask = results[key]['mask']
            violations = jnp.abs(W) * mask
            max_viol = jnp.max(violations)
            mean_viol = jnp.mean(violations[mask])
            print(f"   {results[key]['name']:25s}: max={max_viol:.2e}, mean={mean_viol:.2e}")
    
    print("\n4. SPEEDUP vs KKT:")
    if 'kkt_group' in results:
        kkt_time = results['kkt_group']['time']
        for key in results:
            if key != 'kkt_group':
                speedup = kkt_time / results[key]['time']
                print(f"   {results[key]['name']:25s}: {speedup:>6.2f}x faster")

# ============= DEMO =============

if __name__ == "__main__":
    # Generate test data
    key = jax.random.PRNGKey(42)
    n_samples, n_features = 1000, 7
    n_countries, n_tenors = 5, 10
    n_outputs = n_countries * n_tenors
    
    # True coefficients with structure
    true_W = jnp.ones((n_features, n_outputs)) * 0.5
    # Create some true zeros
    true_W = true_W.at[0, :10].set(0.0)  # First feature zero for first country
    true_W = true_W.at[2, 10:20].set(0.0)  # Third feature zero for second country
    true_W = true_W.at[4:6, :].set(0.0)  # Features 4,5 zero everywhere
    
    X = jax.random.normal(key, (n_samples, n_features))
    noise = 0.1 * jax.random.normal(key, (n_samples, n_outputs))
    Y = X @ true_W + noise
    
    # Create forced group mask
    forced_group_mask = jnp.zeros((n_countries, n_tenors, n_features), dtype=bool)
    # First country: zero out feature 0
    forced_group_mask = forced_group_mask.at[0, :, 0].set(True)
    # Second country: zero out feature 2
    forced_group_mask = forced_group_mask.at[1, :, 2].set(True)
    # All countries: zero out features 4,5
    forced_group_mask = forced_group_mask.at[:, :, 4:6].set(True)
    
    # Run comparison
    window_size = 100
    stride = 50
    
    print("Running comprehensive comparison...")
    results = compare_methods(X, Y, window_size, stride, forced_group_mask, n_countries, n_tenors)
    
    # Visualize
    visualize_comprehensive_comparison(results, X, Y, window_size, stride, true_W, forced_group_mask)
