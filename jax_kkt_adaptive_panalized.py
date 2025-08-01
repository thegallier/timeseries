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


#====================== v6

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
import time

# ============= HUBER LOSS IMPLEMENTATION =============

@jax.jit
def huber_loss(residuals, delta=1.0):
    """
    Huber loss: quadratic for small errors, linear for large errors.
    More robust to outliers than squared loss.
    
    L(r) = 0.5 * r^2           if |r| <= delta
           delta * |r| - 0.5 * delta^2   if |r| > delta
    """
    abs_res = jnp.abs(residuals)
    return jnp.where(
        abs_res <= delta,
        0.5 * residuals**2,
        delta * abs_res - 0.5 * delta**2
    )

@jax.jit
def huber_gradient(residuals, delta=1.0):
    """Gradient of Huber loss"""
    return jnp.where(
        jnp.abs(residuals) <= delta,
        residuals,
        delta * jnp.sign(residuals)
    )

# ============= IRLS SOLVER WITH HUBER LOSS =============

@jax.jit
def solve_huber_regression(X, Y, delta=1.0, max_iter=20, tol=1e-4):
    """
    Solve regression with Huber loss using Iteratively Reweighted Least Squares (IRLS).
    """
    n_samples, n_features = X.shape
    n_outputs = Y.shape[1]
    
    # Initialize with OLS solution
    XtX = X.T @ X + 1e-6 * jnp.eye(n_features)
    XtY = X.T @ Y
    W = jnp.linalg.solve(XtX, XtY)
    
    def irls_step(carry, _):
        W_prev = carry
        
        # Compute residuals
        residuals = Y - X @ W_prev
        
        # Compute weights for Huber loss
        # w_i = 1 if |r_i| <= delta, delta/|r_i| otherwise
        abs_res = jnp.abs(residuals)
        weights = jnp.where(abs_res <= delta, 1.0, delta / (abs_res + 1e-8))
        
        # Weighted least squares update - vectorized version
        W_new = jnp.zeros_like(W_prev)
        
        # Vectorize over outputs
        def solve_single_output(j):
            w_j = weights[:, j]
            # Create weighted design matrix
            X_weighted = X * w_j[:, None]
            XtWX = X_weighted.T @ X
            XtWY = X_weighted.T @ Y[:, j]
            return jnp.linalg.solve(XtWX + 1e-6 * jnp.eye(n_features), XtWY)
        
        W_new = jax.vmap(solve_single_output, in_axes=0, out_axes=1)(jnp.arange(n_outputs))
        
        # Check convergence
        converged = jnp.max(jnp.abs(W_new - W_prev)) < tol
        
        return W_new, converged
    
    # Use lax.scan for the iteration
    W_final, _ = jax.lax.scan(irls_step, W, None, length=max_iter)
    
    return W_final

# ============= ADAPTIVE THRESHOLD DISCOVERY ACROSS WINDOWS =============

def discover_group_sparsity_pattern(X, Y, window_size, stride, n_countries, n_tenors, 
                                   threshold_k=2.0, consistency_threshold=0.8,
                                   use_huber=False, huber_delta=1.0):
    """
    Analyze coefficient patterns across windows to identify which should be consistently zero.
    
    Args:
        X, Y: Full dataset
        window_size, stride: Sliding window parameters
        n_countries, n_tenors: Group structure
        threshold_k: Adaptive threshold factor
        consistency_threshold: Fraction of windows where coef must be small to be zeroed
        use_huber: Whether to use Huber loss
        huber_delta: Huber loss parameter
    
    Returns:
        forced_group_mask: (n_countries, n_tenors, n_features) boolean mask
        analysis_results: Dictionary with detailed analysis
    """
    n_samples, n_features = X.shape
    n_outputs = Y.shape[1]
    n_windows = (n_samples - window_size) // stride + 1
    
    # Storage for coefficient estimates across windows
    all_coefficients = []
    all_masks = []
    
    print(f"Analyzing {n_windows} windows...")
    
    # Extract windows and compute coefficients
    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        X_win = X[start:end]
        Y_win = Y[start:end]
        
        # Solve regression
        if use_huber:
            W = solve_huber_regression(X_win, Y_win, delta=huber_delta)
        else:
            XtX = X_win.T @ X_win + 1e-6 * jnp.eye(n_features)
            XtY = X_win.T @ Y_win
            W = jnp.linalg.solve(XtX, XtY)
        
        all_coefficients.append(W)
        
        # Apply adaptive threshold
        W_flat = jnp.abs(W).ravel()
        median = jnp.median(W_flat)
        mad = jnp.median(jnp.abs(W_flat - median))
        threshold = threshold_k * mad * 1.4826
        mask = jnp.abs(W) < threshold
        all_masks.append(mask)
    
    all_coefficients = jnp.stack(all_coefficients)  # (n_windows, n_features, n_outputs)
    all_masks = jnp.stack(all_masks)
    
    # Analyze consistency of small coefficients
    # For each coefficient, calculate fraction of windows where it's small
    consistency = jnp.mean(all_masks, axis=0)  # (n_features, n_outputs)
    
    # Create forced group mask based on consistency
    forced_group_mask = jnp.zeros((n_countries, n_tenors, n_features), dtype=bool)
    
    for feature in range(n_features):
        for country in range(n_countries):
            for tenor in range(n_tenors):
                output_idx = country * n_tenors + tenor
                # Mark for zeroing if consistently small across windows
                if consistency[feature, output_idx] >= consistency_threshold:
                    forced_group_mask = forced_group_mask.at[country, tenor, feature].set(True)
    
    # Additional group-level analysis
    # Check if entire feature should be zeroed for a country (all tenors)
    for feature in range(n_features):
        for country in range(n_countries):
            tenor_consistency = []
            for tenor in range(n_tenors):
                output_idx = country * n_tenors + tenor
                tenor_consistency.append(consistency[feature, output_idx])
            
            # If all tenors show high consistency, enforce group constraint
            if jnp.min(jnp.array(tenor_consistency)) >= consistency_threshold:
                forced_group_mask = forced_group_mask.at[country, :, feature].set(True)
    
    analysis_results = {
        'all_coefficients': all_coefficients,
        'all_masks': all_masks,
        'consistency': consistency,
        'mean_coefficients': jnp.mean(all_coefficients, axis=0),
        'std_coefficients': jnp.std(all_coefficients, axis=0),
        'cv_coefficients': jnp.std(all_coefficients, axis=0) / (jnp.abs(jnp.mean(all_coefficients, axis=0)) + 1e-8)
    }
    
    return forced_group_mask, analysis_results

# ============= COMPREHENSIVE SLIDING WINDOW WITH DISCOVERED CONSTRAINTS =============

def sliding_window_with_discovered_constraints(
    X, Y, window_size, stride, n_countries, n_tenors,
    discovery_params=None, method='penalty', use_huber=False, huber_delta=1.0,
    penalty_strength=1e10):
    """
    Two-stage approach:
    1. Discover sparsity pattern using adaptive thresholding across windows
    2. Apply discovered constraints to final regression
    """
    
    if discovery_params is None:
        discovery_params = {
            'threshold_k': 2.0,
            'consistency_threshold': 0.8,
            'use_huber': use_huber,
            'huber_delta': huber_delta
        }
    
    # Stage 1: Discover sparsity pattern
    print("Stage 1: Discovering sparsity pattern...")
    forced_group_mask, analysis = discover_group_sparsity_pattern(
        X, Y, window_size, stride, n_countries, n_tenors, **discovery_params
    )
    
    # Print discovered pattern
    n_constrained = jnp.sum(forced_group_mask)
    n_total = forced_group_mask.size
    print(f"Discovered {n_constrained}/{n_total} ({100*n_constrained/n_total:.1f}%) coefficients to zero")
    
    # Stage 2: Apply constraints
    print("\nStage 2: Applying discovered constraints...")
    
    n_samples, n_features = X.shape
    n_outputs = Y.shape[1]
    n_windows = (n_samples - window_size) // stride + 1
    
    # Process forced_group_mask for window-wise application
    mask_reshaped = forced_group_mask.transpose(2, 0, 1).reshape(n_features, n_outputs)
    masks = jnp.broadcast_to(mask_reshaped[None, :, :], (n_windows, n_features, n_outputs))
    
    # Extract windows
    def get_window(start_idx):
        X_win = jax.lax.dynamic_slice(X, (start_idx, 0), (window_size, n_features))
        Y_win = jax.lax.dynamic_slice(Y, (start_idx, 0), (window_size, n_outputs))
        return X_win, Y_win
    
    start_indices = jnp.arange(n_windows) * stride
    X_wins, Y_wins = jax.vmap(get_window)(start_indices)
    
    if method == 'penalty':
        if use_huber:
            # Huber regression with penalty constraints
            def solve_window(X_win, Y_win, mask):
                # First solve unconstrained Huber regression
                W = solve_huber_regression(X_win, Y_win, delta=huber_delta)
                
                # Then apply penalties for masked coefficients
                # Vectorized approach
                def apply_penalty_to_output(j):
                    mask_j = mask[:, j]
                    penalty_diag = jnp.where(mask_j, penalty_strength, 0.0)
                    
                    XtX = X_win.T @ X_win
                    XtY = X_win.T @ Y_win[:, j]
                    XtX_pen = XtX + jnp.diag(penalty_diag) + 1e-6 * jnp.eye(n_features)
                    
                    # Only re-solve if there are penalties
                    return jax.lax.cond(
                        jnp.any(mask_j),
                        lambda: jnp.linalg.solve(XtX_pen, XtY),
                        lambda: W[:, j]
                    )
                
                W_penalized = jax.vmap(apply_penalty_to_output)(jnp.arange(n_outputs))
                return W_penalized.T
            
            W_all = jax.vmap(solve_window)(X_wins, Y_wins, masks)
        else:
            # Standard OLS with penalties
            def solve_window_ols(X_win, Y_win, mask):
                XtX = X_win.T @ X_win + 1e-6 * jnp.eye(n_features)
                XtY = X_win.T @ Y_win
                
                def solve_output(j):
                    mask_j = mask[:, j]
                    penalty_diag = jnp.where(mask_j, penalty_strength, 0.0)
                    XtX_pen = XtX + jnp.diag(penalty_diag)
                    return jnp.linalg.solve(XtX_pen, XtY[:, j])
                
                W = jax.vmap(solve_output)(jnp.arange(n_outputs))
                return W.T
            
            W_all = jax.vmap(solve_window_ols)(X_wins, Y_wins, masks)
    
    else:  # KKT method
        # Convert mask to constraints and use KKT (implementation from previous code)
        # ... (omitted for brevity, use KKT implementation from before)
        W_all = None  # Placeholder
    
    return W_all, masks, forced_group_mask, analysis

# ============= VISUALIZATION =============

def visualize_discovery_and_results(analysis, forced_group_mask, W_final, 
                                   n_countries, n_tenors, true_W=None):
    """Comprehensive visualization of the discovery process and results"""
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Consistency heatmap
    ax = plt.subplot(3, 4, 1)
    consistency_reshaped = analysis['consistency'].T.reshape(n_countries, n_tenors, -1)
    # Average over tenors to show by country
    consistency_country = jnp.mean(consistency_reshaped, axis=1)
    im = ax.imshow(consistency_country, cmap='RdYlBu_r', vmin=0, vmax=1)
    ax.set_title('Consistency by Country\n(Fraction of windows with small coef)')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Country')
    plt.colorbar(im, ax=ax)
    
    # 2. Coefficient variation (CV)
    ax = plt.subplot(3, 4, 2)
    cv_reshaped = analysis['cv_coefficients'].T.reshape(n_countries, n_tenors, -1)
    cv_country = jnp.mean(cv_reshaped, axis=1)
    im = ax.imshow(jnp.log10(cv_country + 1e-8), cmap='viridis')
    ax.set_title('log10(CV) by Country\n(Coefficient stability)')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Country')
    plt.colorbar(im, ax=ax)
    
    # 3. Discovered mask
    ax = plt.subplot(3, 4, 3)
    mask_visual = forced_group_mask.reshape(n_countries * n_tenors, -1).T
    im = ax.imshow(mask_visual, cmap='RdBu_r', aspect='auto')
    ax.set_title('Discovered Constraints\n(Red = Zero)')
    ax.set_xlabel('Country-Tenor')
    ax.set_ylabel('Feature')
    plt.colorbar(im, ax=ax)
    
    # 4. Mean coefficients
    ax = plt.subplot(3, 4, 4)
    mean_coef = analysis['mean_coefficients']
    im = ax.imshow(mean_coef, cmap='RdBu_r', vmin=-2, vmax=2)
    ax.set_title('Mean Coefficients\n(Across windows)')
    ax.set_xlabel('Output')
    ax.set_ylabel('Feature')
    plt.colorbar(im, ax=ax)
    
    # 5. Coefficient evolution for selected features
    ax = plt.subplot(3, 4, 5)
    n_windows = analysis['all_coefficients'].shape[0]
    # Plot evolution for first 3 features, first output
    for f in range(min(3, analysis['all_coefficients'].shape[1])):
        ax.plot(analysis['all_coefficients'][:, f, 0], label=f'Feature {f}')
    ax.set_xlabel('Window')
    ax.set_ylabel('Coefficient Value')
    ax.set_title('Coefficient Evolution\n(First output)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Sparsity over windows
    ax = plt.subplot(3, 4, 6)
    sparsity_per_window = jnp.mean(analysis['all_masks'], axis=(1, 2))
    ax.plot(sparsity_per_window, 'b-', linewidth=2)
    ax.axhline(y=jnp.mean(forced_group_mask), color='r', linestyle='--', 
               label=f'Final sparsity: {jnp.mean(forced_group_mask):.2f}')
    ax.set_xlabel('Window')
    ax.set_ylabel('Sparsity')
    ax.set_title('Sparsity Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 7. Final coefficients
    ax = plt.subplot(3, 4, 7)
    W_final_avg = jnp.mean(W_final, axis=0)
    im = ax.imshow(W_final_avg, cmap='RdBu_r', vmin=-2, vmax=2)
    ax.set_title('Final Coefficients\n(With discovered constraints)')
    ax.set_xlabel('Output')
    ax.set_ylabel('Feature')
    plt.colorbar(im, ax=ax)
    
    # 8. Violations
    ax = plt.subplot(3, 4, 8)
    mask_reshaped = forced_group_mask.transpose(2, 0, 1).reshape(W_final_avg.shape[0], -1)
    violations = jnp.abs(W_final_avg) * mask_reshaped
    im = ax.imshow(violations, cmap='Reds', vmin=0, vmax=0.1)
    ax.set_title('Constraint Violations')
    ax.set_xlabel('Output')
    ax.set_ylabel('Feature')
    plt.colorbar(im, ax=ax)
    
    # 9. Group structure visualization
    ax = plt.subplot(3, 4, 9)
    # Show which entire country-feature combinations are zeroed
    country_feature_zeros = jnp.zeros((n_countries, W_final_avg.shape[0]))
    for c in range(n_countries):
        for f in range(W_final_avg.shape[0]):
            # Check if all tenors are zeroed for this country-feature
            if jnp.all(forced_group_mask[c, :, f]):
                country_feature_zeros = country_feature_zeros.at[c, f].set(1)
    
    im = ax.imshow(country_feature_zeros, cmap='Reds', aspect='auto')
    ax.set_title('Country-Feature Zeros\n(Entire row zeroed)')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Country')
    plt.colorbar(im, ax=ax)
    
    # 10-12: Comparison with truth if available
    if true_W is not None:
        # True coefficients
        ax = plt.subplot(3, 4, 10)
        im = ax.imshow(true_W, cmap='RdBu_r', vmin=-2, vmax=2)
        ax.set_title('True Coefficients')
        ax.set_xlabel('Output')
        ax.set_ylabel('Feature')
        plt.colorbar(im, ax=ax)
        
        # Error
        ax = plt.subplot(3, 4, 11)
        error = W_final_avg - true_W
        im = ax.imshow(error, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title('Estimation Error')
        ax.set_xlabel('Output')
        ax.set_ylabel('Feature')
        plt.colorbar(im, ax=ax)
        
        # Performance metrics
        ax = plt.subplot(3, 4, 12)
        true_zeros = true_W == 0
        discovered_zeros = mask_reshaped
        
        tp = jnp.sum(discovered_zeros & true_zeros)
        fp = jnp.sum(discovered_zeros & ~true_zeros)
        fn = jnp.sum(~discovered_zeros & true_zeros)
        tn = jnp.sum(~discovered_zeros & ~true_zeros)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = ['Precision', 'Recall', 'F1']
        values = [precision, recall, f1]
        bars = ax.bar(metrics, values)
        ax.set_ylim([0, 1])
        ax.set_title('Zero Detection Performance')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

# ============= DEMO WITH HUBER LOSS =============

def demo_adaptive_discovery_with_huber():
    """Demonstrate the complete framework with Huber loss"""
    
    # Generate data with outliers
    key = jax.random.PRNGKey(42)
    n_samples, n_features = 1500, 8
    n_countries, n_tenors = 4, 6
    n_outputs = n_countries * n_tenors
    
    # True coefficients with group structure
    true_W = jnp.zeros((n_features, n_outputs))
    
    # Country 0: Features 0,1 active for all tenors
    for t in range(n_tenors):
        true_W = true_W.at[0, 0*n_tenors + t].set(1.5)
        true_W = true_W.at[1, 0*n_tenors + t].set(-1.0)
    
    # Country 1: Features 2,3 active for first half of tenors
    for t in range(n_tenors // 2):
        true_W = true_W.at[2, 1*n_tenors + t].set(2.0)
        true_W = true_W.at[3, 1*n_tenors + t].set(0.5)
    
    # Country 2: Feature 4 active for all tenors
    for t in range(n_tenors):
        true_W = true_W.at[4, 2*n_tenors + t].set(1.0)
    
    # Generate data
    X = jax.random.normal(key, (n_samples, n_features))
    Y_clean = X @ true_W
    
    # Add noise with outliers
    noise = 0.1 * jax.random.normal(key, (n_samples, n_outputs))
    # Add 5% outliers
    outlier_mask = jax.random.uniform(key, (n_samples,)) < 0.05
    outlier_noise = 5.0 * jax.random.normal(key, (n_samples, n_outputs))
    noise = jnp.where(outlier_mask[:, None], outlier_noise, noise)
    Y = Y_clean + noise
    
    window_size = 150
    stride = 75
    
    print("="*80)
    print("DEMO: Adaptive Discovery with Huber Loss")
    print("="*80)
    
    # Compare methods
    methods = {
        'OLS': {
            'use_huber': False,
            'discovery_params': {'threshold_k': 2.0, 'consistency_threshold': 0.8}
        },
        'Huber': {
            'use_huber': True,
            'huber_delta': 1.0,
            'discovery_params': {'threshold_k': 2.0, 'consistency_threshold': 0.8, 
                               'use_huber': True, 'huber_delta': 1.0}
        }
    }
    
    results = {}
    
    for method_name, params in methods.items():
        print(f"\n{method_name} Method:")
        print("-"*40)
        
        start_time = time.time()
        
        W_all, masks, forced_group_mask, analysis = sliding_window_with_discovered_constraints(
            X, Y, window_size, stride, n_countries, n_tenors,
            method='penalty',
            **params
        )
        
        elapsed = time.time() - start_time
        
        # Calculate R²
        r2_list = []
        for i in range(W_all.shape[0]):
            start = i * stride
            end = start + window_size
            Y_pred = X[start:end] @ W_all[i]
            r2 = 1 - jnp.sum((Y[start:end] - Y_pred)**2) / jnp.sum((Y[start:end] - jnp.mean(Y[start:end]))**2)
            r2_list.append(r2)
        
        results[method_name] = {
            'W': W_all,
            'mask': forced_group_mask,
            'analysis': analysis,
            'r2': jnp.array(r2_list),
            'time': elapsed
        }
        
        print(f"Time: {elapsed:.2f}s")
        print(f"Average R²: {jnp.mean(jnp.array(r2_list)):.4f}")
        print(f"Discovered sparsity: {100*jnp.mean(forced_group_mask):.1f}%")
    
    # Visualize comparison
    print("\nVisualizing results...")
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx, (method_name, result) in enumerate(results.items()):
        # Discovered mask
        ax = axes[0, idx]
        mask_visual = result['mask'].reshape(n_countries * n_tenors, -1).T
        im = ax.imshow(mask_visual, cmap='Reds', aspect='auto')
        ax.set_title(f'{method_name}: Discovered Zeros')
        ax.set_xlabel('Country-Tenor')
        ax.set_ylabel('Feature')
        
        # R² over time
        ax = axes[1, idx]
        ax.plot(result['r2'], linewidth=2)
        ax.set_xlabel('Window')
        ax.set_ylabel('R²')
        ax.set_title(f'{method_name}: R² = {jnp.mean(result["r2"]):.3f}')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
    
    # Comparison
    ax = axes[0, 2]
    ax.bar(['OLS', 'Huber'], 
           [jnp.mean(results['OLS']['r2']), jnp.mean(results['Huber']['r2'])])
    ax.set_ylabel('Average R²')
    ax.set_title('Performance Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[1, 2]
    ax.bar(['OLS', 'Huber'], 
           [jnp.mean(results['OLS']['mask']), jnp.mean(results['Huber']['mask'])])
    ax.set_ylabel('Sparsity')
    ax.set_title('Discovered Sparsity')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # Detailed visualization for Huber method
    print("\nDetailed analysis for Huber method:")
    visualize_discovery_and_results(
        results['Huber']['analysis'],
        results['Huber']['mask'],
        results['Huber']['W'],
        n_countries, n_tenors,
        true_W
    )

# Run the demo
if __name__ == "__main__":
    demo_adaptive_discovery_with_huber()
#===v9 speed

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
    ax.set_title('Combinimport jax
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

def visualize_discovery_results(result, X, Y, window_size, stride, 
                              n_countries, n_tenors, true_W=None):
    """
    Comprehensive visualization of discovery results.
    Includes all the graphs from before plus performance metrics.
    """
    
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
    fig = plt.figure(figsize=(24, 20))
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
    plt.tight_layout()
    plt.show()
    
    return fig

def plot_method_comparison(results_dict, X, Y, true_W=None):
    """
    Compare multiple methods/configurations.
    
    Args:
        results_dict: Dictionary of {method_name: result}
    """
    n_methods = len(results_dict)
    
    fig, axes = plt.subplots(3, n_methods, figsize=(5*n_methods, 12))
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
    plt.show()
    
    return fig

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
