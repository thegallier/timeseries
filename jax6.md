Here's the complete reference with text and examples together for easy copying:

```python
# ============================================================================
# COMPLETE REFERENCE: ALL OPTIONS AND METHODS FOR UNIFIED SLIDING REGRESSION
# ============================================================================

# ============================================================================
# 1. METHODS - Core algorithms for regression
# ============================================================================

# VECTORIZED - Fastest method
method='vectorized'
# Speed: 0.3-0.5s
# Pros: Fully parallel GPU computation, fastest
# Cons: Limited constraint types
# Use when: Maximum speed needed, simple constraints

# JAX - Balanced speed and flexibility
method='jax'
constraints_config={'method': 'penalty'}  # Soft constraints (approximate)
# OR
constraints_config={'method': 'exact'}    # Hard constraints via KKT
# Speed: 0.5-1.5s
# Pros: GPU accelerated, good constraint handling
# Cons: Limited vs CVXPY
# Use when: Need balance of speed and constraints

# CVXPY - Most flexible
method='cvxpy'
cvxpy_config={'loss': 'huber', 'dv01_neutral': True}
# Speed: 10-20s (slow)
# Pros: Most constraint types, robust regression
# Cons: Slow, processes windows sequentially
# Use when: Complex constraints, outlier resistance

# HYBRID - JAX discovery + CVXPY regression
method='hybrid'
cvxpy_config={'loss': 'squared'}
# Speed: 10-20s
# Pros: Fast discovery + flexible regression
# Use when: Need CVXPY features with faster discovery

# ============================================================================
# 2. DISCOVERY CONFIG - Automatic zero pattern detection
# ============================================================================

discovery_config = {
    'enabled': True,                    # Whether to use discovery
    'consistency_threshold': 0.9,       # Fraction of windows that must agree (0-1)
    'magnitude_threshold': 0.05,        # Absolute value threshold
    'relative_threshold': 0.1,          # Fraction of max coefficient (0-1)
    'forced_mask': None                 # Optional pre-specified zero mask (n_features, n_outputs)
}

# Example with custom mask
import jax.numpy as jnp
mask = jnp.ones((7, 168), dtype=bool)
mask[0, :] = False  # Allow DU for all outputs
discovery_config = {
    'enabled': True,
    'forced_mask': mask
}

# ============================================================================
# 3. CONSTRAINTS CONFIG - Control coefficient values
# ============================================================================

constraints_config = {
    # Method for JAX only
    'method': 'penalty',                # 'penalty' (soft) or 'exact' (hard via KKT)
    
    # Constraint specifications
    'offset_indices': (2, 4),           # w[2] + w[4] = 0
    # OR multiple: [(2, 4), (5, 6)]
    
    'fixed_constraints': [(0, 0.5)],    # w[0] = 0.5
    # OR multiple: [(0, 0.5), (2, -0.3)]
    
    'positive_constraints': [1, 3],     # w[1] >= 0, w[3] >= 0
    'negative_constraints': [2],        # w[2] <= 0
    
    # Penalty strengths (for penalty method)
    'zero_penalty': 1e12,               # For zero constraints
    'offset_penalty': 1e10,             # For offset constraints
    'fixed_penalty': 1e10,              # For fixed value constraints
    'positive_penalty': 1e10,           # For positive constraints
    'negative_penalty': 1e10            # For negative constraints
}

# ============================================================================
# 4. CVXPY CONFIG - Advanced optimization options (cvxpy method only)
# ============================================================================

# Standard least squares
cvxpy_config = {
    'loss': 'squared',
    'post_zero_threshold': 1e-6
}

# Robust regression (outlier resistant)
cvxpy_config = {
    'loss': 'huber',
    'huber_delta': 1.0,  # Threshold for outlier detection (smaller = more robust)
    'post_zero_threshold': 1e-6
}

# Sparse solution via transaction costs
cvxpy_config = {
    'loss': 'squared',
    'transaction_costs': np.ones(7) * 0.01,  # Cost per unit for each coefficient
    'tc_lambda': 0.1,                        # Weight for transaction cost penalty
    'post_zero_threshold': 1e-6
}

# Market neutral constraint
cvxpy_config = {
    'loss': 'squared',
    'dv01_neutral': True,  # Enforce sum of coefficients = 1
    'post_zero_threshold': 1e-6
}

# Combined configuration
cvxpy_config = {
    'loss': 'huber',
    'huber_delta': 0.5,
    'transaction_costs': np.array([0.01, 0.01, 0.02, 0.02, 0.015, 0.03, 0.03]),
    'tc_lambda': 0.05,
    'dv01_neutral': True,
    'post_zero_threshold': 1e-4
}

# ============================================================================
# 5. LAYERS - Hierarchical regression
# ============================================================================

# Single layer (default)
layers = None

# Three layers with default configs
layers = [{}, {}, {}]

# How layered regression works:
# Layer 1: Fit Y with constraints → W₁ (main patterns)
# Layer 2: Fit residuals (Y - XW₁) → W₂ (secondary patterns)  
# Layer 3: Fit remaining residuals → W₃ (fine details)
# Final: W = W₁ + W₂ + W₃

# ============================================================================
# 6. POST-PROCESSING CONFIG - Clean up coefficients after optimization
# ============================================================================

post_processing_config = {
    'zero_threshold': 1e-6,          # Zero out below this absolute value
    'relative_threshold': 0.05,      # Zero if < 5% of max in same output
    'keep_top_k': 3,                 # Keep only top 3 coefficients per output
    'sparsity_target': 0.8,          # Target 80% sparsity
    'enforce_sign_consistency': True # Zero if sign changes across windows
}

# Examples of different post-processing strategies
# Minimal
post_config_minimal = {'zero_threshold': 1e-10}

# Standard
post_config_standard = {'zero_threshold': 1e-6}

# Aggressive
post_config_aggressive = {
    'zero_threshold': 1e-4,
    'relative_threshold': 0.1
}

# Sparse
post_config_sparse = {
    'zero_threshold': 1e-6,
    'keep_top_k': 3
}

# ============================================================================
# 7. COUNTRY RULES - For batch processing multiple countries
# ============================================================================

country_rules = {
    'BEL': {
        'allowed_countries': ['DEU', 'FRA'],     # Which countries' hedges can be used
        'use_adjacent_only': True,               # Use only 1-2 adjacent maturity futures
        'sign_constraints': {'RX': 'negative'},  # RX must be <= 0
        'fixed_coefficients': {                  # Fixed coefficient values
            'all': {'OAT': 0.2},                 # OAT = 0.2 for all tenors
            '2yr': {'DU': 0.8},                  # DU = 0.8 for 2yr only
            '10yr': {'DU': 0.5, 'RX': -0.3}     # Multiple for 10yr
        }
    },
    'NLD': {
        'allowed_countries': ['DEU'],
        'use_adjacent_only': False,
        'sign_constraints': {},
        'fixed_coefficients': {'DU': 0.4}  # Country-wide
    }
}

# ============================================================================
# 8. COMPLETE WORKING EXAMPLES
# ============================================================================

import jax.numpy as jnp
import numpy as np

# Generate sample data
X = jnp.array(np.random.randn(1256, 7))    # 7 hedge instruments
Y = jnp.array(np.random.randn(1256, 168))  # 12 countries × 14 tenors

# -----------------------------
# EXAMPLE 1: Simple and fast
# -----------------------------
results_simple = unified_sliding_regression_extended(
    X=X,
    Y=Y,
    window_size=200,
    stride=150,
    n_countries=12,
    n_tenors=14,
    method='vectorized',
    discovery_config={'enabled': True},
    constraints_config={'method': 'penalty'}
)

# -----------------------------
# EXAMPLE 2: With exact constraints
# -----------------------------
results_constrained = unified_sliding_regression_extended(
    X=X,
    Y=Y,
    window_size=200,
    stride=150,
    n_countries=12,
    n_tenors=14,
    method='jax',
    discovery_config={
        'enabled': True,
        'consistency_threshold': 0.9,
        'magnitude_threshold': 0.05
    },
    constraints_config={
        'method': 'exact',
        'offset_indices': (2, 4),        # w[2] + w[4] = 0
        'fixed_constraints': [(0, 0.5)], # w[0] = 0.5
        'positive_constraints': [1],     # w[1] >= 0
        'negative_constraints': [2]      # w[2] <= 0
    }
)

# -----------------------------
# EXAMPLE 3: Robust regression with CVXPY
# -----------------------------
results_robust = unified_sliding_regression_extended(
    X=X,
    Y=Y,
    window_size=200,
    stride=150,
    n_countries=12,
    n_tenors=14,
    method='cvxpy',
    discovery_config={'enabled': True},
    constraints_config={
        'positive_constraints': [0, 1, 2],
        'negative_constraints': [3, 4]
    },
    cvxpy_config={
        'loss': 'huber',
        'huber_delta': 1.0,
        'dv01_neutral': True,
        'post_zero_threshold': 1e-6
    }
)

# -----------------------------
# EXAMPLE 4: Layered regression
# -----------------------------
results_layered = unified_sliding_regression_extended(
    X=X,
    Y=Y,
    window_size=200,
    stride=150,
    n_countries=12,
    n_tenors=14,
    method='jax',
    layers=[{}, {}, {}],  # 3 layers
    discovery_config={'enabled': True},
    constraints_config={
        'method': 'penalty',
        'zero_penalty': 1e12
    }
)

# -----------------------------
# EXAMPLE 5: Batch processing with country rules
# -----------------------------
country_rules = {
    'BEL': {
        'allowed_countries': ['DEU', 'FRA'],
        'use_adjacent_only': True,
        'sign_constraints': {'RX': 'negative'},
        'fixed_coefficients': {'DU': 0.5}
    },
    'NLD': {
        'allowed_countries': ['DEU'],
        'use_adjacent_only': False,
        'sign_constraints': {}
    }
}

results_batch = compute_all_countries_hedge_ratios_batch(
    X=X,
    Y=Y,
    country_rules=country_rules,
    window_size=200,
    stride=150,
    method='jax',
    constraint_method='penalty'
)

# -----------------------------
# EXAMPLE 6: Post-processing
# -----------------------------
# Apply post-processing to results
W_processed = apply_post_processing(
    results_simple['W_avg'],
    post_processing_config={
        'zero_threshold': 1e-6,
        'relative_threshold': 0.05,
        'keep_top_k': 3
    }
)

# Compute true R² on post-processed coefficients
r2_results = compute_true_r_squared(
    X, Y, W_processed,
    window_size=200,
    stride=150
)
print(f"True R²: {r2_results['r2_overall']:.4f}")
print(f"Sparsity: {r2_results['sparsity']:.1%}")

# ============================================================================
# 9. COMMON USE CASES
# ============================================================================

# CASE 1: Maximum speed, basic hedging
results = unified_sliding_regression_extended(
    X, Y, 200, 150, 12, 14,
    method='vectorized',
    discovery_config={'enabled': True}
)

# CASE 2: Outlier-resistant hedging
results = unified_sliding_regression_extended(
    X, Y, 200, 150, 12, 14,
    method='cvxpy',
    cvxpy_config={'loss': 'huber', 'huber_delta': 1.0}
)

# CASE 3: Sparse hedging (minimize positions)
results = unified_sliding_regression_extended(
    X, Y, 200, 150, 12, 14,
    method='cvxpy',
    cvxpy_config={
        'transaction_costs': np.ones(7) * 0.01,
        'tc_lambda': 0.1
    }
)

# CASE 4: Fixed hedge ratios
results = unified_sliding_regression_extended(
    X, Y, 200, 150, 12, 14,
    method='jax',
    constraints_config={
        'method': 'exact',
        'fixed_constraints': [(0, 0.5), (2, -0.3)]
    }
)

# CASE 5: Market neutral (sum = 1)
results = unified_sliding_regression_extended(
    X, Y, 200, 150, 12, 14,
    method='cvxpy',
    cvxpy_config={'dv01_neutral': True}
)

# ============================================================================
# 10. METHOD SELECTION GUIDE
# ============================================================================

# PERFORMANCE COMPARISON:
# Method          | Speed | Constraints | Robustness | Typical Time
# ----------------|-------|-------------|------------|---------------
# vectorized      | ★★★★★ | ★          | ★          | 0.3-0.5s
# jax (penalty)   | ★★★★  | ★★★        | ★★         | 0.5-1.0s
# jax (exact)     | ★★★★  | ★★★★       | ★★         | 0.8-1.5s
# cvxpy           | ★     | ★★★★★      | ★★★★★      | 10-20s
# hybrid          | ★★    | ★★★★★      | ★★★★       | 10-20s

# DECISION TREE:
# Need maximum speed? → vectorized
# Need exact offset/fixed constraints? → jax with exact
# Need robust regression (outliers)? → cvxpy with huber
# Need sparsity control? → cvxpy with transaction costs
# Need market neutral? → cvxpy with dv01_neutral
# Need balance? → jax with penalty
```

This is the complete reference with all text and examples together, ready to copy and use!
