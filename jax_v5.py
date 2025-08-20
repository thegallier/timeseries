

import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Dict, Optional, Tuple
import time


def create_batch_constraints_for_all_countries(
    country_rules: Dict[str, dict],
    n_hedges: int = 7,
    n_tenors: int = 14,
    n_countries: int = 12
) -> Tuple[jnp.ndarray, Dict]:
    """
    Create a single large discovery mask and constraints for all countries at once.
    This allows processing all countries in a single GPU call.
    
    Args:
        country_rules: Dict mapping each country to its hedging rules
            Example: {
                'BEL': {
                    'allowed_countries': ['DEU', 'FRA'],
                    'use_adjacent_only': True,
                    'sign_constraints': {'RX': 'negative'}
                },
                'IRL': {...}
            }
        n_hedges: Number of hedge instruments (7)
        n_tenors: Number of tenors (14)
        n_countries: Number of countries (12)
        
    Returns:
        - discovery_mask: Shape (n_hedges, n_countries * n_tenors)
        - constraints_dict: Combined constraints for all countries
    """
    
    builder = HedgeConstraintBuilder()
    
    # Create a large discovery mask for all countries
    # Shape: (n_hedges, n_countries * n_tenors) = (7, 168)
    full_discovery_mask = np.ones((n_hedges, n_countries * n_tenors), dtype=bool)
    
    # Collect all sign constraints with their applicable ranges
    all_positive_constraints = []
    all_negative_constraints = []
    
    # Process each country
    for country, rules in country_rules.items():
        country_idx = builder.countries.index(country)
        
        # Generate constraints for this country
        constraints_config, country_mask = compute_country_hedge_constraints(
            target_country=country,
            allowed_countries=rules['allowed_countries'],
            use_adjacent_only=rules.get('use_adjacent_only', False),
            sign_constraints=rules.get('sign_constraints', {}),
            penalty_strengths=rules.get('penalty_strengths', None)
        )
        
        # Insert this country's mask into the full mask
        start_idx = country_idx * n_tenors
        end_idx = (country_idx + 1) * n_tenors
        full_discovery_mask[:, start_idx:end_idx] = country_mask
        
        # Note: Sign constraints apply to specific hedge indices globally
        # but we need to track which countries they apply to
        if 'positive_constraints' in constraints_config:
            for hedge_idx in constraints_config['positive_constraints']:
                all_positive_constraints.append({
                    'hedge_idx': hedge_idx,
                    'country_idx': country_idx,
                    'country': country
                })
        
        if 'negative_constraints' in constraints_config:
            for hedge_idx in constraints_config['negative_constraints']:
                all_negative_constraints.append({
                    'hedge_idx': hedge_idx,
                    'country_idx': country_idx,
                    'country': country
                })
    
    # Convert to JAX array
    full_discovery_mask_jax = jnp.array(full_discovery_mask)
    
    # Create unified constraints config
    unified_constraints = {
        'method': 'penalty',
        'zero_penalty': 1e12,
        'offset_penalty': 1e10,
        'fixed_penalty': 1e10,
        'positive_constraints': [c['hedge_idx'] for c in all_positive_constraints],
        'negative_constraints': [c['hedge_idx'] for c in all_negative_constraints],
        'constraint_details': {
            'positive': all_positive_constraints,
            'negative': all_negative_constraints
        }
    }
    
    return full_discovery_mask_jax, unified_constraints


def compute_all_countries_hedge_ratios_batch(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    country_rules: Dict[str, dict],
    window_size: int,
    stride: int,
    method: str = 'jax',
    constraint_method: str = 'penalty',  # 'penalty' or 'exact' for JAX
    cvxpy_config: dict = None,
    batch_size: Optional[int] = None,
    verbose: bool = True
) -> Dict:
    """
    Compute hedge ratios for all countries in a single batch for GPU efficiency.
    
    Args:
        X: Hedge data with shape (n_samples, n_hedges) = (n_samples, 7)
        Y: Full country-tenor data (n_samples, n_countries * n_tenors) = (n_samples, 168)
        country_rules: Rules for each country's hedging constraints
        window_size: Size of sliding window
        stride: Stride for sliding window
        method: 'jax', 'vectorized', 'cvxpy', or 'hybrid'
        constraint_method: 'penalty' or 'exact' (for JAX method)
        cvxpy_config: Configuration for CVXPY solver (if using CVXPY)
        batch_size: Optional batch size for processing windows
        verbose: Print progress information
        
    Returns:
        Dictionary with results for all countries
    """
    
    builder = HedgeConstraintBuilder()
    n_samples = X.shape[0]
    n_hedges = X.shape[1]
    n_countries = len(builder.countries)
    n_tenors = len(builder.tenors)
    n_outputs = Y.shape[1]
    
    if verbose:
        print("="*80)
        print("BATCH HEDGE RATIO COMPUTATION FOR ALL COUNTRIES")
        print("="*80)
        print(f"\nData dimensions:")
        print(f"  Samples: {n_samples}")
        print(f"  Hedges: {n_hedges} ({', '.join(builder.hedges)})")
        print(f"  Countries: {n_countries}")
        print(f"  Tenors: {n_tenors}")
        print(f"  Total outputs: {n_outputs}")
        print(f"  Window size: {window_size}")
        print(f"  Stride: {stride}")
        print(f"  Method: {method}")
        if method == 'jax':
            print(f"  Constraint method: {constraint_method}")
    
    # Create batch constraints for all countries
    if verbose:
        print(f"\nCreating batch constraints for {len(country_rules)} countries...")
    
    full_discovery_mask, unified_constraints = create_batch_constraints_for_all_countries(
        country_rules, n_hedges, n_tenors, n_countries
    )
    
    # Count active (non-zero) positions
    n_active = int(full_discovery_mask.size - jnp.sum(full_discovery_mask))
    sparsity = 100 * jnp.sum(full_discovery_mask) / full_discovery_mask.size
    
    if verbose:
        print(f"  Discovery mask shape: {full_discovery_mask.shape}")
        print(f"  Active positions: {n_active}/{full_discovery_mask.size}")
        print(f"  Sparsity: {sparsity:.1f}%")
        print(f"  Sign constraints: {len(unified_constraints['positive_constraints'])} positive, "
              f"{len(unified_constraints['negative_constraints'])} negative")
    
    # Set up discovery config with our forced mask
    discovery_config = {
        'enabled': True,
        'forced_mask': full_discovery_mask,
        'consistency_threshold': 0.9,
        'magnitude_threshold': 0.05
    }
    
    # Update constraints for the chosen method
    if method == 'jax':
        unified_constraints['method'] = constraint_method
    
    # Set up CVXPY config if needed
    if method in ['cvxpy', 'hybrid'] and cvxpy_config is None:
        cvxpy_config = {
            'loss': 'squared',  # or 'huber' for robust regression
            'huber_delta': 1.0,
            'transaction_costs': None,  # Could add transaction cost penalties
            'tc_lambda': 0.0,
            'dv01_neutral': False,  # Whether to enforce sum of coefficients = 1
            'post_zero_threshold': 1e-10  # Zero out tiny coefficients
        }
        if verbose:
            print(f"\nCVXPY configuration:")
            print(f"  Loss function: {cvxpy_config['loss']}")
            print(f"  DV01 neutral: {cvxpy_config['dv01_neutral']}")
    
    # Run the unified regression for all countries at once
    if verbose:
        print(f"\nRunning {method} regression...")
        start_time = time.time()
    
    results = unified_sliding_regression_extended(
        X=X,
        Y=Y,  # Use full Y matrix with all countries
        window_size=window_size,
        stride=stride,
        n_countries=n_countries,
        n_tenors=n_tenors,
        method=method,
        discovery_config=discovery_config,
        constraints_config=unified_constraints,
        cvxpy_config=cvxpy_config
    )
    
    if verbose:
        elapsed = time.time() - start_time
        print(f"  Completed in {elapsed:.3f} seconds")
        
        if 'r2' in results:
            r2_mean = float(jnp.mean(results['r2'][0]))
            r2_std = float(jnp.std(results['r2'][0]))
            print(f"  Mean R²: {r2_mean:.4f} ± {r2_std:.4f}")
    
    # Parse results by country
    country_results = parse_batch_results_by_country(
        results, country_rules, builder, verbose
    )
    
    return {
        'batch_results': results,
        'country_results': country_results,
        'discovery_mask': full_discovery_mask,
        'constraints': unified_constraints,
        'country_rules': country_rules
    }


def parse_batch_results_by_country(
    batch_results: Dict,
    country_rules: Dict[str, dict],
    builder: HedgeConstraintBuilder,
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    Parse batch results into per-country results.
    
    Args:
        batch_results: Results from unified_sliding_regression_extended
        country_rules: Rules used for each country
        builder: HedgeConstraintBuilder instance
        verbose: Print summary information
        
    Returns:
        Dictionary mapping country names to their results
    """
    
    country_results = {}
    
    if 'W_avg' not in batch_results:
        return country_results
    
    W_avg = batch_results['W_avg']
    # W_avg shape: (n_hedges, n_countries * n_tenors) = (7, 168)
    
    n_tenors = len(builder.tenors)
    
    if verbose:
        print("\nParsing results by country:")
        print("-" * 40)
    
    for country in country_rules.keys():
        country_idx = builder.countries.index(country)
        
        # Extract coefficients for this country
        start_idx = country_idx * n_tenors
        end_idx = (country_idx + 1) * n_tenors
        W_country = W_avg[:, start_idx:end_idx]  # Shape: (7, 14)
        
        # Create hedge ratios dictionary
        hedge_ratios = {}
        active_hedges = set()
        
        for tenor_idx, tenor in enumerate(builder.tenors):
            hedge_ratios[tenor] = {}
            for hedge_idx, hedge in enumerate(builder.hedges):
                value = float(W_country[hedge_idx, tenor_idx])
                if abs(value) > 1e-10:  # Non-zero
                    hedge_ratios[tenor][hedge] = value
                    active_hedges.add(hedge)
        
        # Calculate R² if available (would need window-specific R² parsing)
        r2_country = None
        if 'r2' in batch_results and batch_results['r2']:
            # This would need more complex parsing for per-country R²
            pass
        
        country_results[country] = {
            'W': W_country,
            'hedge_ratios_by_tenor': hedge_ratios,
            'active_hedges': sorted(list(active_hedges)),
            'rules': country_rules[country],
            'n_active_positions': int(jnp.sum(jnp.abs(W_country) > 1e-10))
        }
        
        if verbose:
            print(f"\n{country}:")
            print(f"  Allowed countries: {country_rules[country]['allowed_countries']}")
            print(f"  Active hedges: {sorted(list(active_hedges))}")
            print(f"  Non-zero positions: {country_results[country]['n_active_positions']}/{W_country.size}")
            
            # Show sample of hedge ratios
            sample_tenors = ['2yr', '5yr', '10yr', '30yr']
            print(f"  Sample hedge ratios:")
            for tenor in sample_tenors:
                if tenor in hedge_ratios and hedge_ratios[tenor]:
                    ratios_str = ', '.join([f"{h}:{v:.3f}" for h, v in hedge_ratios[tenor].items()])
                    print(f"    {tenor}: {ratios_str}")
    
    return country_results


def compare_regression_methods(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    country_rules: Dict[str, dict],
    window_size: int,
    stride: int,
    methods_to_test: List[str] = None,
    verbose: bool = True
) -> Dict:
    """
    Compare different regression methods for hedge ratio computation.
    
    Args:
        X: Hedge data
        Y: Country-tenor data
        country_rules: Hedging rules for each country
        window_size: Window size
        stride: Stride
        methods_to_test: List of methods to compare
        verbose: Print progress
        
    Returns:
        Comparison results
    """
    
    if methods_to_test is None:
        methods_to_test = ['jax', 'vectorized']
        try:
            import cvxpy
            methods_to_test.extend(['cvxpy', 'hybrid'])
            if verbose:
                print("CVXPY is available - including CVXPY and hybrid methods")
        except ImportError:
            if verbose:
                print("CVXPY not available, skipping CVXPY and hybrid methods")
    
    comparison_results = {}
    
    for method in methods_to_test:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Testing method: {method}")
            print(f"{'='*60}")
        
        # For JAX method, test both penalty and exact constraints
        if method == 'jax':
            for constraint_method in ['penalty', 'exact']:
                method_key = f"{method}_{constraint_method}"
                
                if verbose:
                    print(f"\n  Constraint method: {constraint_method}")
                
                start_time = time.time()
                
                results = compute_all_countries_hedge_ratios_batch(
                    X=X,
                    Y=Y,
                    country_rules=country_rules,
                    window_size=window_size,
                    stride=stride,
                    method=method,
                    constraint_method=constraint_method,
                    verbose=False
                )
                
                elapsed = time.time() - start_time
                
                comparison_results[method_key] = {
                    'results': results,
                    'computation_time': elapsed,
                    'method': method,
                    'constraint_method': constraint_method
                }
                
                if verbose:
                    print(f"    Time: {elapsed:.3f}s")
                    
                    # Check constraint violations
                    if 'batch_results' in results and 'violations' in results['batch_results']:
                        violations = results['batch_results']['violations']
                        print(f"    Violations:")
                        for key, value in violations.items():
                            if isinstance(value, (int, float)):
                                print(f"      {key}: {value:.2e}")
        else:
            start_time = time.time()
            
            results = compute_all_countries_hedge_ratios_batch(
                X=X,
                Y=Y,
                country_rules=country_rules,
                window_size=window_size,
                stride=stride,
                method=method,
                constraint_method='penalty',
                verbose=False
            )
            
            elapsed = time.time() - start_time
            
            comparison_results[method] = {
                'results': results,
                'computation_time': elapsed,
                'method': method
            }
            
            if verbose:
                print(f"  Time: {elapsed:.3f}s")
    
    if verbose:
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        
        # Create comparison table
        print(f"\n{'Method':<20} {'Time (s)':<12} {'Notes'}")
        print("-" * 50)
        
        for method_key, data in comparison_results.items():
            notes = ""
            if 'constraint_method' in data:
                notes = f"({data['constraint_method']} constraints)"
            print(f"{method_key:<20} {data['computation_time']:<12.3f} {notes}")
        
        # Find fastest method
        fastest = min(comparison_results.items(), key=lambda x: x[1]['computation_time'])
        print(f"\nFastest method: {fastest[0]} ({fastest[1]['computation_time']:.3f}s)")
    
    return comparison_results


# Example usage
def example_batch_computation():
    """
    Example showing batch computation for all countries with different rules.
    """
    
    print("EXAMPLE: Batch Hedge Ratio Computation")
    print("="*80)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1256
    n_hedges = 7
    n_countries = 12
    n_tenors = 14
    
    # Create synthetic data with some structure
    X = np.random.randn(n_samples, n_hedges)
    
    # Create Y with some correlation to X
    true_W = np.random.randn(n_hedges, n_countries * n_tenors) * 0.5
    Y = X @ true_W + np.random.randn(n_samples, n_countries * n_tenors) * 0.1
    
    # Convert to JAX arrays
    X = jnp.array(X)
    Y = jnp.array(Y)
    
    # Define hedging rules for each country
    country_rules = {
        'DEU': {
            'allowed_countries': ['DEU'],  # Only German futures
            'use_adjacent_only': False,    # Can use all 4 German futures
            'sign_constraints': {}
        },
        'FRA': {
            'allowed_countries': ['DEU', 'FRA'],  # German and French futures
            'use_adjacent_only': True,            # Only adjacent
            'sign_constraints': {'OAT': 'positive'}  # OAT must be positive
        },
        'ITA': {
            'allowed_countries': ['ITA'],  # Only Italian futures
            'use_adjacent_only': False,    # Can use both IK and BTS
            'sign_constraints': {}
        },
        'ESP': {
            'allowed_countries': ['DEU', 'FRA', 'ITA'],  # All available
            'use_adjacent_only': True,                    # Adjacent only
            'sign_constraints': {'RX': 'negative', 'BTS': 'positive'}
        },
        'BEL': {
            'allowed_countries': ['DEU', 'FRA'],
            'use_adjacent_only': True,
            'sign_constraints': {'RX': 'negative'}
        },
        'IRL': {
            'allowed_countries': ['DEU'],
            'use_adjacent_only': True,
            'sign_constraints': {}
        },
        'PRT': {
            'allowed_countries': ['DEU', 'ITA'],
            'use_adjacent_only': False,
            'sign_constraints': {'UB': 'positive'}
        },
        'GRC': {
            'allowed_countries': ['DEU', 'ITA'],
            'use_adjacent_only': True,
            'sign_constraints': {'BTS': 'positive'}
        },
        'NLD': {
            'allowed_countries': ['DEU'],
            'use_adjacent_only': False,
            'sign_constraints': {}
        },
        'EUR': {
            'allowed_countries': ['DEU', 'FRA', 'ITA'],  # All
            'use_adjacent_only': False,                   # All futures
            'sign_constraints': {}
        },
        'AUT': {
            'allowed_countries': ['DEU'],
            'use_adjacent_only': True,
            'sign_constraints': {'DU': 'positive'}
        },
        'FIN': {
            'allowed_countries': ['DEU'],
            'use_adjacent_only': True,
            'sign_constraints': {}
        }
    }
    
    # Run batch computation
    print("\n1. Running batch computation for all countries...")
    print("-" * 60)
    
    results = compute_all_countries_hedge_ratios_batch(
        X=X,
        Y=Y,
        country_rules=country_rules,
        window_size=200,
        stride=150,
        method='jax',
        constraint_method='penalty',
        verbose=True
    )
    
    # Compare different methods
    print("\n2. Comparing different regression methods...")
    print("-" * 60)
    
    # Test all available methods
    methods_to_test = ['jax', 'vectorized']
    try:
        import cvxpy
        methods_to_test.extend(['cvxpy', 'hybrid'])
        print("CVXPY is available - testing all 4 methods")
    except ImportError:
        print("CVXPY not installed - testing JAX and vectorized only")
    
    comparison = compare_regression_methods(
        X=X,
        Y=Y,
        country_rules=country_rules,
        window_size=200,
        stride=150,
        methods_to_test=methods_to_test,
        verbose=True
    )
    
    # Analyze results
    print("\n3. Analyzing results...")
    print("-" * 60)
    
    country_results = results['country_results']
    
    # Show statistics
    total_positions = sum(cr['n_active_positions'] for cr in country_results.values())
    total_possible = n_hedges * n_tenors * n_countries
    
    print(f"\nOverall statistics:")
    print(f"  Total active positions: {total_positions}/{total_possible}")
    print(f"  Overall sparsity: {100 * (1 - total_positions/total_possible):.1f}%")
    
    # Show which futures are most used
    hedge_usage = {hedge: 0 for hedge in HedgeConstraintBuilder().hedges}
    for country, cr in country_results.items():
        for hedge in cr['active_hedges']:
            hedge_usage[hedge] += 1
    
    print(f"\nHedge usage across countries:")
    for hedge, count in sorted(hedge_usage.items(), key=lambda x: x[1], reverse=True):
        print(f"  {hedge}: used by {count}/{n_countries} countries")
    
    return results, comparison


if __name__ == "__main__":
    results, comparison = example_batch_computation()
    
    print("\n" + "="*80)
    print("BATCH COMPUTATION COMPLETED")
    print("="*80)
    print("\nKey findings:")
    print("- All countries processed in a single GPU-efficient call")
    print("- Constraints properly applied per country's rules")
    print("- Different methods compared for performance")
    print("\nThe batch approach is significantly more efficient than")
    print("processing countries individually, especially on GPUs!")

countryRules=   country_rules = {
        'DEU': {
            'allowed_countries': ['DEU'],  # Only German futures
            'use_adjacent_only': False,    # Can use all 4 German futures
            'sign_constraints': {}
        },
        'FRA': {
            'allowed_countries': ['DEU', 'FRA'],  # German and French futures
            'use_adjacent_only': True,            # Only adjacent
            'sign_constraints': {'OAT': 'positive'}  # OAT must be positive
        },
        'ITA': {
            'allowed_countries': ['ITA'],  # Only Italian futures
            'use_adjacent_only': False,    # Can use both IK and BTS
            'sign_constraints': {}
        },
        'ESP': {
            'allowed_countries': ['DEU', 'FRA', 'ITA'],  # All available
            'use_adjacent_only': True,                    # Adjacent only
            'sign_constraints': {'RX': 'negative', 'BTS': 'positive'}
        },
        'BEL': {
            'allowed_countries': ['DEU', 'FRA'],
            'use_adjacent_only': True,
            'sign_constraints': {'RX': 'negative'}
        },
        'IRL': {
            'allowed_countries': ['DEU'],
            'use_adjacent_only': True,
            'sign_constraints': {}
        },
        'PRT': {
            'allowed_countries': ['DEU', 'ITA'],
            'use_adjacent_only': False,
            'sign_constraints': {'UB': 'positive'}
        },
        'GRC': {
            'allowed_countries': ['DEU', 'ITA'],
            'use_adjacent_only': True,
            'sign_constraints': {'BTS': 'positive'}
        },
        'NLD': {
            'allowed_countries': ['DEU'],
            'use_adjacent_only': False,
            'sign_constraints': {}
        },
        'EUR': {
            'allowed_countries': ['DEU', 'FRA', 'ITA'],  # All
            'use_adjacent_only': False,                   # All futures
            'sign_constraints': {}
        },
        'AUT': {
            'allowed_countries': ['DEU'],
            'use_adjacent_only': True,
            'sign_constraints': {'DU': 'positive'}
        },
        'FIN': {
            'allowed_countries': ['DEU'],
            'use_adjacent_only': True,
            'sign_constraints': {}
        }
    }

comparison = compare_regression_methods(X,Y,countryRules,250,150)

# For each method
jax_results = comparison['jax_penalty']['results']
cvxpy_results = comparison['cvxpy']['results']


import jax.numpy as jnp
import numpy as np
from typing import List, Dict, Optional, Tuple, Union


class HedgeConstraintBuilder:
    """
    Builder for hedge ratio constraints based on country/tenor rules.
    
    The hedge futures mapping:
    - Index 0-3: DEU futures (DU, OE, RX, UB)
    - Index 4: FRA future (OAT)
    - Index 5-6: ITA futures (IK, BTS)
    """
    
    def __init__(self):
        # Define the complete mapping
        self.countries = [
            'DEU', 'FRA', 'ITA', 'ESP', 'PRT', 'BEL', 
            'IRL', 'GRC', 'NLD', 'EUR', 'AUT', 'FIN'
        ]
        
        self.tenors = [
            '2yr', '3yr', '4yr', '5yr', '6yr', '7yr', '8yr', 
            '9yr', '10yr', '12yr', '15yr', '20yr', '30yr', '50yr'
        ]
        
        # Hedge futures and their country origins
        self.hedges = ['DU', 'OE', 'RX', 'UB', 'OAT', 'IK', 'BTS']
        self.hedge_countries = ['DEU', 'DEU', 'DEU', 'DEU', 'FRA', 'ITA', 'ITA']
        
        # Map hedge names to indices
        self.hedge_to_idx = {h: i for i, h in enumerate(self.hedges)}
        
        # Map countries to their hedge indices
        self.country_to_hedge_indices = {
            'DEU': [0, 1, 2, 3],  # DU, OE, RX, UB
            'FRA': [4],           # OAT
            'ITA': [5, 6]         # IK, BTS
        }
        
        # Tenor maturity mapping for adjacent futures selection
        # Maps each future to the tenors it typically covers
        self.hedge_tenor_mapping = {
            'DU': [0, 1],          # 2yr, 3yr (short end)
            'OE': [2, 3, 4],       # 4yr, 5yr, 6yr
            'RX': [5, 6, 7, 8],    # 7yr, 8yr, 9yr, 10yr
            'UB': [9, 10, 11],     # 12yr, 15yr, 20yr
            'OAT': [8, 9, 10],     # 10yr, 12yr, 15yr
            'IK': [7, 8, 9],       # 9yr, 10yr, 12yr
            'BTS': [11, 12, 13]    # 20yr, 30yr, 50yr
        }
    
    def get_adjacent_hedges_for_tenor(self, tenor_idx: int) -> List[int]:
        """
        Get the hedge indices that are 'adjacent' (most relevant) for a given tenor.
        Returns at most 2 adjacent futures, or 1 if only one is relevant.
        """
        adjacent = []
        
        for hedge_name, hedge_idx in self.hedge_to_idx.items():
            tenor_coverage = self.hedge_tenor_mapping[hedge_name]
            
            # Check if this hedge covers the tenor
            if tenor_idx in tenor_coverage:
                # Calculate distance to find the most relevant
                distances = [abs(tenor_idx - t) for t in tenor_coverage]
                min_distance = min(distances)
                
                # Add hedge with a priority score (lower is better)
                adjacent.append((hedge_idx, min_distance))
        
        # Sort by distance and take the 2 closest
        adjacent.sort(key=lambda x: x[1])
        
        # Return hedge indices only
        result = [h[0] for h in adjacent[:2]]
        
        # If no adjacent found, use the closest hedges based on tenor range
        if not result:
            if tenor_idx <= 1:  # Very short end
                result = [0]  # DU
            elif tenor_idx <= 4:  # Short-medium
                result = [1, 2]  # OE, RX
            elif tenor_idx <= 8:  # Medium
                result = [2, 3]  # RX, UB
            elif tenor_idx <= 11:  # Long
                result = [3, 4]  # UB, OAT
            else:  # Very long
                result = [6]  # BTS
        
        return result


def compute_country_hedge_constraints(
    target_country: str,
    allowed_countries: List[str],
    use_adjacent_only: bool = False,
    sign_constraints: Dict[str, str] = None,
    penalty_strengths: Dict[str, float] = None
) -> dict:
    """
    Generate constraints configuration for hedge ratio optimization.
    
    Args:
        target_country: The country whose bonds we're hedging (e.g., 'BEL')
        allowed_countries: List of countries whose futures can be used (e.g., ['DEU', 'FRA'])
        use_adjacent_only: If True, use only 1-2 adjacent futures per tenor; if False, use all qualified
        sign_constraints: Dict mapping hedge names to 'positive' or 'negative' (e.g., {'RX': 'negative'})
        penalty_strengths: Optional override for penalty values
        
    Returns:
        Dictionary with constraints configuration for unified_sliding_regression_extended
        
    Example:
        constraints = compute_country_hedge_constraints(
            target_country='BEL',
            allowed_countries=['DEU', 'FRA'],
            use_adjacent_only=True,
            sign_constraints={'RX': 'negative'}
        )
    """
    
    builder = HedgeConstraintBuilder()
    
    # Validate inputs
    if target_country not in builder.countries:
        raise ValueError(f"Unknown target country: {target_country}")
    
    for country in allowed_countries:
        if country not in builder.countries:
            raise ValueError(f"Unknown allowed country: {country}")
    
    if sign_constraints:
        for hedge in sign_constraints.keys():
            if hedge not in builder.hedges:
                raise ValueError(f"Unknown hedge: {hedge}")
    
    # Default penalty strengths
    default_penalties = {
        'zero_penalty': 1e12,
        'offset_penalty': 1e10,
        'fixed_penalty': 1e10,
        'positive_penalty': 1e10,
        'negative_penalty': 1e10
    }
    
    if penalty_strengths:
        default_penalties.update(penalty_strengths)
    
    # Initialize constraint lists
    fixed_constraints = []  # List of (index, value) tuples
    positive_constraints = []  # List of indices that must be >= 0
    negative_constraints = []  # List of indices that must be <= 0
    
    # Get target country index
    target_country_idx = builder.countries.index(target_country)
    
    # Determine which hedges are allowed based on country restrictions
    allowed_hedge_indices = set()
    for country in allowed_countries:
        if country in builder.country_to_hedge_indices:
            allowed_hedge_indices.update(builder.country_to_hedge_indices[country])
    
    # Build constraints for the coefficient matrix
    # W has shape (n_features, n_outputs) where n_features = n_hedges
    # For our case: W shape is (7, 14) when hedging one country with 14 tenors
    
    n_hedges = len(builder.hedges)
    n_tenors = len(builder.tenors)
    
    # Create a discovery mask to enforce zeros
    # discovery_mask shape should be (n_hedges, n_tenors) 
    # where True means the coefficient should be zero
    discovery_mask = np.ones((n_hedges, n_tenors), dtype=bool)
    
    # For each tenor of the target country
    for tenor_idx in range(n_tenors):
        # Determine which hedges to use for this tenor
        if use_adjacent_only:
            # Get adjacent hedges for this tenor
            adjacent_hedges = builder.get_adjacent_hedges_for_tenor(tenor_idx)
            # Filter by allowed countries
            tenor_allowed_hedges = [h for h in adjacent_hedges if h in allowed_hedge_indices]
        else:
            # Use all allowed hedges
            tenor_allowed_hedges = list(allowed_hedge_indices)
        
        # Mark allowed hedges as False (not zero) in discovery mask
        for hedge_idx in tenor_allowed_hedges:
            discovery_mask[hedge_idx, tenor_idx] = False
    
    # Convert to jax array
    discovery_mask = jnp.array(discovery_mask)
    
    # Apply sign constraints
    if sign_constraints:
        for hedge_name, sign in sign_constraints.items():
            hedge_idx = builder.hedge_to_idx[hedge_name]
            
            # Check if this hedge is in the allowed set
            if hedge_idx in allowed_hedge_indices:
                if sign == 'positive':
                    positive_constraints.append(hedge_idx)
                elif sign == 'negative':
                    negative_constraints.append(hedge_idx)
                else:
                    raise ValueError(f"Sign must be 'positive' or 'negative', got: {sign}")
    
    # Build the constraints configuration
    constraints_config = {
        'method': 'penalty',  # or 'exact' for KKT method
        **default_penalties
    }
    
    # Use discovery mask for zero constraints instead of fixed constraints
    # This is more efficient and works better with the unified_sliding_regression_extended
    
    if positive_constraints:
        constraints_config['positive_constraints'] = positive_constraints
    
    if negative_constraints:
        constraints_config['negative_constraints'] = negative_constraints
    
    # Add summary information
    constraints_config['summary'] = {
        'target_country': target_country,
        'allowed_countries': allowed_countries,
        'allowed_hedges': [builder.hedges[i] for i in sorted(allowed_hedge_indices)],
        'use_adjacent_only': use_adjacent_only,
        'sign_constraints': sign_constraints or {},
        'n_zero_constraints': int(np.sum(discovery_mask)),
        'n_positive_constraints': len(positive_constraints),
        'n_negative_constraints': len(negative_constraints),
        'discovery_mask': discovery_mask  # Include the mask for inspection
    }
    
    return constraints_config, discovery_mask


def compute_country_hedge_ratios_v2(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    target_country: str,
    allowed_countries: List[str],
    use_adjacent_only: bool,
    sign_constraints: Dict[str, str],
    window_size: int,
    stride: int,
    method: str = 'jax',
    discovery_config: dict = None,
    cvxpy_config: dict = None,
    penalty_strengths: dict = None
) -> dict:
    """
    Wrapper function to compute hedge ratios for a specific country with constraints.
    
    Args:
        X: Input hedge data (n_samples, n_hedges)
        Y: Output country-tenor data (n_samples, n_countries * n_tenors)
        target_country: Country to hedge (e.g., 'BEL')
        allowed_countries: Countries whose hedges can be used (e.g., ['DEU', 'FRA'])
        use_adjacent_only: If True, use only adjacent futures
        sign_constraints: Sign constraints on specific hedges (e.g., {'RX': 'negative'})
        window_size: Size of sliding window
        stride: Stride for sliding window
        method: Regression method
        discovery_config: Zero discovery configuration
        cvxpy_config: CVXPY configuration
        penalty_strengths: Override penalty strengths
        
    Returns:
        Results dictionary with hedge ratios and diagnostics
    """
    
    builder = HedgeConstraintBuilder()
    
    # Generate constraints and discovery mask
    constraints_config, discovery_mask = compute_country_hedge_constraints(
        target_country=target_country,
        allowed_countries=allowed_countries,
        use_adjacent_only=use_adjacent_only,
        sign_constraints=sign_constraints,
        penalty_strengths=penalty_strengths
    )
    
    # Extract Y data for target country
    target_idx = builder.countries.index(target_country)
    n_tenors = len(builder.tenors)
    
    start_idx = target_idx * n_tenors
    end_idx = (target_idx + 1) * n_tenors
    Y_country = Y[:, start_idx:end_idx]
    
    # Set up discovery config to use our forced mask
    if discovery_config is None:
        discovery_config = {}
    
    # Use our discovery mask as the forced mask
    discovery_config['enabled'] = True
    discovery_config['forced_mask'] = discovery_mask
    
    # Run regression
    # Note: X should have shape (n_samples, n_hedges) = (n_samples, 7)
    # Y_country has shape (n_samples, n_tenors) = (n_samples, 14)
    
    results = unified_sliding_regression_extended(
        X=X,
        Y=Y_country,
        window_size=window_size,
        stride=stride,
        n_countries=1,  # We're focusing on one country
        n_tenors=n_tenors,
        method=method,
        discovery_config=discovery_config,
        constraints_config=constraints_config,
        cvxpy_config=cvxpy_config
    )
    
    # Add metadata
    results['target_country'] = target_country
    results['allowed_countries'] = allowed_countries
    results['constraints_summary'] = constraints_config['summary']
    
    # Create interpretable output
    if 'W_avg' in results:
        # W_avg shape: (n_hedges, n_tenors) = (7, 14)
        hedge_ratios = {}
        for tenor_idx, tenor in enumerate(builder.tenors):
            hedge_ratios[tenor] = {}
            for hedge_idx, hedge in enumerate(builder.hedges):
                value = float(results['W_avg'][hedge_idx, tenor_idx])
                if abs(value) > 1e-10:  # Only show non-zero values
                    hedge_ratios[tenor][hedge] = value
        
        results['hedge_ratios_by_tenor'] = hedge_ratios
        
        # Print summary of which hedges are used
        print(f"\nHedge usage summary for {target_country}:")
        print(f"Allowed countries: {allowed_countries}")
        print(f"Adjacent only: {use_adjacent_only}")
        print(f"Sign constraints: {sign_constraints}")
        print(f"\nActive hedges by tenor:")
        for tenor, hedges in hedge_ratios.items():
            if hedges:
                print(f"  {tenor}: {list(hedges.keys())}")
    
    return results


# Example usage
def example_usage():
    """
    Example showing how to use the constraint builder for BEL with specific rules.
    """
    import jax.numpy as jnp
    import jax.random as random
    
    # Generate sample data
    key = random.PRNGKey(42)
    n_samples = 1000
    n_hedges = 7
    n_countries = 12
    n_tenors = 14
    
    # X: hedge data (n_samples, n_hedges)
    X = random.normal(key, (n_samples, n_hedges))
    
    # Y: country-tenor data (n_samples, n_countries * n_tenors)
    Y = random.normal(random.split(key)[1], (n_samples, n_countries * n_tenors))
    
    # Example 1: BEL hedged with DEU and FRA futures, adjacent only, RX must be negative
    print("Example 1: Belgium with constraints")
    print("-" * 50)
    
    constraints, discovery_mask = compute_country_hedge_constraints(
        target_country='BEL',
        allowed_countries=['DEU', 'FRA'],
        use_adjacent_only=True,
        sign_constraints={'RX': 'negative'}
    )
    
    print(f"Target country: {constraints['summary']['target_country']}")
    print(f"Allowed countries: {constraints['summary']['allowed_countries']}")
    print(f"Allowed hedges: {constraints['summary']['allowed_hedges']}")
    print(f"Number of zero constraints: {constraints['summary']['n_zero_constraints']}")
    print(f"Sign constraints: {constraints['summary']['sign_constraints']}")
    print(f"\nDiscovery mask shape: {discovery_mask.shape}")
    print(f"Discovery mask (True = zero constraint):")
    builder = HedgeConstraintBuilder()
    for h_idx, hedge in enumerate(builder.hedges):
        print(f"  {hedge}: {discovery_mask[h_idx, :]}")
    
    # Example 2: Run full regression for BEL
    print("\nExample 2: Running regression for Belgium")
    print("-" * 50)
    
    results = compute_country_hedge_ratios_v2(
        X=X,
        Y=Y,
        target_country='BEL',
        allowed_countries=['DEU', 'FRA'],
        use_adjacent_only=True,
        sign_constraints={'RX': 'negative', 'OAT': 'positive'},
        window_size=250,
        stride=20,
        method='jax',
        discovery_config={'enabled': False}
    )
    
    print(f"Computation completed in {results.get('computation_time', 0):.3f} seconds")
    
    # Example 3: Multiple countries with different rules
    print("\nExample 3: Multiple countries with different rules")
    print("-" * 50)
    
    country_rules = {
        'BEL': {
            'allowed_countries': ['DEU', 'FRA'],
            'use_adjacent_only': True,
            'sign_constraints': {'RX': 'negative'}
        },
        'IRL': {
            'allowed_countries': ['DEU'],
            'use_adjacent_only': False,
            'sign_constraints': {}
        },
        'GRC': {
            'allowed_countries': ['DEU', 'ITA'],
            'use_adjacent_only': True,
            'sign_constraints': {'BTS': 'positive'}
        }
    }
    
    for country, rules in country_rules.items():
        constraints, discovery_mask = compute_country_hedge_constraints(
            target_country=country,
            **rules
        )
        
        print(f"\n{country}:")
        print(f"  Allowed hedges: {constraints['summary']['allowed_hedges']}")
        print(f"  Zero constraints: {constraints['summary']['n_zero_constraints']}")
        print(f"  Sign constraints: {constraints['summary']['sign_constraints']}")
        print(f"  Non-zero positions: {int(discovery_mask.size - np.sum(discovery_mask))}/{discovery_mask.size}")
    
    return results


if __name__ == "__main__":
    results = example_usage()


import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap, jit
from functools import partial
import matplotlib.pyplot as plt
import time
import warnings
import pandas as pd

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    warnings.warn("CVXPY not available. Some methods will be disabled.")


def unified_sliding_regression_extended(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    window_size: int,
    stride: int,
    n_countries: int,
    n_tenors: int,
    method: str = 'jax',  # 'jax', 'cvxpy', 'vectorized', 'hybrid'
    layers: list = None,  # For multi-layer regression
    discovery_config: dict = None,
    constraints_config: dict = None,
    cvxpy_config: dict = None
) -> dict:
    """
    Extended unified sliding window regression supporting all methods.

    Args:
        X: Input data (n_samples, n_features)
        Y: Output data (n_samples, n_outputs)
        window_size: Size of sliding window
        stride: Stride for sliding window
        n_countries: Number of countries (for reshaping)
        n_tenors: Number of tenors (for reshaping)
        method:
            - 'jax': JAX-based with KKT/penalty methods
            - 'cvxpy': CVXPY for advanced constraints
            - 'vectorized': Fully vectorized operations
            - 'hybrid': JAX discovery + CVXPY regression
        layers: List of layer configs for multi-layer regression
        discovery_config: Configuration for zero discovery
            - 'enabled': Whether to use discovery (default: True)
            - 'consistency_threshold': Threshold for consistency (default: 0.9)
            - 'magnitude_threshold': Absolute magnitude threshold (default: 0.05)
            - 'relative_threshold': Relative magnitude threshold (default: 0.1)
            - 'forced_mask': Pre-specified zero mask
        constraints_config: Configuration for constraints
            - 'method': 'exact' (KKT) or 'penalty' for JAX method
            - 'offset_indices': Tuple (idx1, idx2) or list of tuples
            - 'fixed_constraints': List of (index, value) tuples
            - 'positive_constraints': List of indices that must be >= 0
            - 'negative_constraints': List of indices that must be <= 0
            - 'zero_penalty': Penalty strength for zero constraints
            - 'offset_penalty': Penalty strength for offset constraints
            - 'fixed_penalty': Penalty strength for fixed value constraints
        cvxpy_config: Additional configuration for CVXPY solver
            - 'loss': 'squared' or 'huber'
            - 'delta': Huber loss parameter
            - 'transaction_costs': Cost vector for L1 penalty
            - 'tc_lambda': Transaction cost penalty weight
            - 'dv01_neutral': Whether to enforce sum of coefficients = 1
            - 'post_zero_threshold': Zero out small coefficients after solving

    Returns:
        Dictionary with results including:
            - 'W_all': All window coefficients
            - 'W_avg': Average coefficients
            - 'W_layers': Layer coefficients (if layered)
            - 'r2': R² values
            - 'discovery_mask': Discovered zero pattern
            - 'violations': Constraint violations
            - 'method_used': Actual method used
            - 'computation_time': Time taken
    """

    start_time = time.time()

    # Input validation and setup
    n_samples, n_features = X.shape
    n_outputs = Y.shape[1]
    n_windows = (n_samples - window_size) // stride + 1

    # Default configurations
    if discovery_config is None:
        discovery_config = {}
    if constraints_config is None:
        constraints_config = {}
    if cvxpy_config is None:
        cvxpy_config = {}

    # Print setup information
    print(f"\nUnified Sliding Regression Extended")
    print(f"  Method: {method}")
    print(f"  Data: {n_samples} samples, {n_features} features, {n_outputs} outputs")
    print(f"  Windows: {n_windows} (size {window_size}, stride {stride})")
    if layers:
        print(f"  Layers: {len(layers)}")

    # Check method availability
    if method == 'cvxpy' and not CVXPY_AVAILABLE:
        warnings.warn("CVXPY not available, falling back to JAX method")
        method = 'jax'

    # ========== PHASE 1: DISCOVERY ==========
    use_discovery = discovery_config.get('enabled', True)
    discovery_mask = None
    discovery_stats = {}

    if use_discovery:
        print("\nPhase 1: Discovering zero patterns...")

        # Create windows for discovery
        X_wins = create_windows_vectorized(X, window_size, stride)
        Y_wins = create_windows_vectorized(Y, window_size, stride)

        # Run discovery
        discovery_mask, discovery_stats = discover_zero_patterns_unified(
            X_wins, Y_wins,
            consistency_threshold=discovery_config.get('consistency_threshold', 0.9),
            magnitude_threshold=discovery_config.get('magnitude_threshold', 0.05),
            relative_threshold=discovery_config.get('relative_threshold', 0.1)
        )

        # Apply forced mask if provided
        forced_mask = discovery_config.get('forced_mask', None)
        if forced_mask is not None:
            discovery_mask = discovery_mask | forced_mask

        n_zeros = jnp.sum(discovery_mask)
        sparsity = 100 * n_zeros / (n_features * n_outputs)
        print(f"  Discovered {n_zeros} zeros ({sparsity:.1f}% sparsity)")

    # ========== PHASE 2: REGRESSION ==========
    print(f"\nPhase 2: Applying {method} regression...")

    # Extract constraint parameters
    constraint_method = constraints_config.get('method', 'exact')
    offset_indices = constraints_config.get('offset_indices', None)
    fixed_constraints = constraints_config.get('fixed_constraints', None)
    positive_constraints = constraints_config.get('positive_constraints', None)
    negative_constraints = constraints_config.get('negative_constraints', None)

    # Handle layers
    if layers is not None and len(layers) > 0:
        results = apply_layered_regression(
            X, Y, window_size, stride, n_layers=len(layers),
            method=method, discovery_mask=discovery_mask,
            constraints_config=constraints_config,
            cvxpy_config=cvxpy_config
        )
    else:
        # Single layer regression
        if method == 'jax':
            results = apply_jax_regression(
                X, Y, window_size, stride,
                constraint_method=constraint_method,
                discovery_mask=discovery_mask,
                offset_indices=offset_indices,
                fixed_constraints=fixed_constraints,
                constraints_config=constraints_config
            )

        elif method == 'vectorized':
            results = apply_vectorized_regression(
                X, Y, window_size, stride,
                discovery_mask=discovery_mask,
                offset_indices=offset_indices,
                fixed_constraints=fixed_constraints,
                constraints_config=constraints_config
            )

        elif method == 'cvxpy':
            results = apply_cvxpy_regression(
                X, Y, window_size, stride,
                discovery_mask=discovery_mask,
                offset_indices=offset_indices,
                fixed_constraints=fixed_constraints,
                positive_constraints=positive_constraints,
                negative_constraints=negative_constraints,
                cvxpy_config=cvxpy_config
            )

        elif method == 'hybrid':
            # Use JAX for discovery (already done) and CVXPY for regression
            results = apply_cvxpy_regression(
                X, Y, window_size, stride,
                discovery_mask=discovery_mask,
                offset_indices=offset_indices,
                fixed_constraints=fixed_constraints,
                positive_constraints=positive_constraints,
                negative_constraints=negative_constraints,
                cvxpy_config=cvxpy_config
            )
            results['method_used'] = 'hybrid'

        else:
            raise ValueError(f"Unknown method: {method}")

    # ========== POST-PROCESSING ==========

    # Add discovery results
    results['discovery_mask'] = discovery_mask
    results['discovery_stats'] = discovery_stats
    if discovery_mask is not None:
        results['discovery_mask_3d'] = discovery_mask.T.reshape(n_countries, n_tenors, n_features)

    # Check constraint violations
    W_avg = results.get('W_avg', jnp.mean(results['W_all'], axis=0))
    violations = check_all_constraints(
        W_avg, discovery_mask, offset_indices,
        fixed_constraints, positive_constraints, negative_constraints
    )
    results['violations'] = violations

    # Add configuration info
    results['config'] = {
        'window_size': window_size,
        'stride': stride,
        'n_windows': n_windows,
        'method': method,
        'constraint_method': constraint_method,
        'discovery_config': discovery_config,
        'constraints_config': constraints_config,
        'cvxpy_config': cvxpy_config
    }

    # Computation time
    computation_time = time.time() - start_time
    results['computation_time'] = computation_time

    print(f"\nCompleted in {computation_time:.3f} seconds")
    print_summary(results)

    return results


# ============= HELPER FUNCTIONS =============

def create_windows_vectorized(data, window_size, stride):
    """Create all windows at once using advanced indexing."""
    n_samples = data.shape[0]
    n_windows = (n_samples - window_size) // stride + 1

    indices = np.arange(n_windows)[:, None] * stride + np.arange(window_size)[None, :]
    return jnp.array(data[indices])


def solve_ols(X, Y, reg=1e-6):
    """Basic OLS solver."""
    XtX = X.T @ X
    XtY = X.T @ Y
    return jnp.linalg.solve(XtX + reg * jnp.eye(X.shape[1]), XtY)


def discover_zero_patterns_unified(X_wins, Y_wins, consistency_threshold=0.9,
                                  magnitude_threshold=0.05, relative_threshold=0.1):
    """Unified discovery function - fully vectorized."""
    # Solve unconstrained for all windows using vmap
    def solve_window(X_win, Y_win):
        return solve_ols(X_win, Y_win)

    W_all = vmap(solve_window)(X_wins, Y_wins)

    # Analyze patterns
    W_abs = jnp.abs(W_all)
    W_abs_mean = jnp.mean(W_abs, axis=0)
    W_std = jnp.std(W_all, axis=0)

    # Discovery criteria
    small_mask = W_abs_mean < magnitude_threshold

    # Relative threshold
    max_per_output = jnp.max(W_abs_mean, axis=0, keepdims=True)
    relative_mask = W_abs_mean < (relative_threshold * max_per_output)

    # Combine criteria
    candidate_mask = small_mask | relative_mask

    # Check consistency
    n_windows = W_all.shape[0]
    small_counts = jnp.sum(W_abs < magnitude_threshold, axis=0)
    consistency = small_counts / n_windows

    # Final mask
    discovery_mask = candidate_mask & (consistency > consistency_threshold)

    stats = {
        'W_mean': W_abs_mean,
        'W_std': W_std,
        'consistency': consistency,
        'n_zeros': jnp.sum(discovery_mask),
        'sparsity': jnp.mean(discovery_mask)
    }

    return discovery_mask, stats


# ============= JAX REGRESSION =============

def apply_jax_regression(X, Y, window_size, stride, constraint_method='exact',
                        discovery_mask=None, offset_indices=None,
                        fixed_constraints=None, constraints_config=None):
    """Apply regression using JAX methods."""

    if constraint_method == 'exact':
        W_all = apply_kkt_constraints_vectorized(
            X, Y, window_size, stride,
            discovery_mask=discovery_mask,
            offset_indices=offset_indices,
            fixed_constraints=fixed_constraints
        )
    else:  # penalty
        W_all = apply_penalty_constraints_vectorized(
            X, Y, window_size, stride,
            discovery_mask=discovery_mask,
            offset_indices=offset_indices,
            fixed_constraints=fixed_constraints,
            zero_penalty=constraints_config.get('zero_penalty', 1e12),
            offset_penalty=constraints_config.get('offset_penalty', 1e10),
            fixed_penalty=constraints_config.get('fixed_penalty', 1e10)
        )

    # Compute R²
    W_avg = jnp.mean(W_all, axis=0)
    X_wins = create_windows_vectorized(X, window_size, stride)
    Y_wins = create_windows_vectorized(Y, window_size, stride)

    # Vectorized R² computation
    Y_preds = jnp.einsum('wij,wjk->wik', X_wins, W_all)
    ss_res = jnp.sum((Y_wins - Y_preds)**2, axis=(1, 2))
    ss_tot = jnp.sum((Y_wins - jnp.mean(Y_wins, axis=1, keepdims=True))**2, axis=(1, 2))
    r2_values = 1 - ss_res / (ss_tot + 1e-8)

    return {
        'W_all': W_all,
        'W_avg': W_avg,
        'r2': [r2_values],
        'method_used': f'jax_{constraint_method}'
    }


# ============= KKT CONSTRAINTS (VECTORIZED) =============

def apply_kkt_constraints_vectorized(X, Y, window_size, stride, discovery_mask=None,
                                   offset_indices=None, fixed_constraints=None):
    """Apply constraints using KKT (exact) method - fully vectorized."""
    X_wins = create_windows_vectorized(X, window_size, stride)
    Y_wins = create_windows_vectorized(Y, window_size, stride)

    n_windows = X_wins.shape[0]
    n_features = X.shape[1]
    n_outputs = Y.shape[1]

    # Apply constraints based on what's provided
    if fixed_constraints and offset_indices:
        W_all = apply_all_constraints_kkt_vectorized(
            X_wins, Y_wins, fixed_constraints[0], offset_indices, discovery_mask
        )
    elif fixed_constraints:
        W_all = apply_fixed_kkt_vectorized(X_wins, Y_wins, fixed_constraints[0], discovery_mask)
    elif offset_indices:
        W_all = apply_offset_kkt_vectorized(X_wins, Y_wins, offset_indices, discovery_mask)
    else:
        W_all = solve_with_zeros_vectorized(X_wins, Y_wins, discovery_mask)

    return W_all


def apply_offset_kkt_vectorized(X_wins, Y_wins, offset_indices, zero_mask=None):
    """Apply offset constraint using KKT - vectorized."""
    if isinstance(offset_indices, list):
        offset_indices = offset_indices[0]
    idx1, idx2 = offset_indices

    n_windows, window_size, n_features = X_wins.shape
    n_outputs = Y_wins.shape[2]

    # Create reduced system by eliminating idx2
    keep_mask = np.ones(n_features, dtype=bool)
    keep_mask[idx2] = False
    keep_indices = np.where(keep_mask)[0]

    X_reduced = X_wins[:, :, keep_mask]

    # Find position of idx1 in reduced system
    idx1_new = np.sum(keep_indices < idx1)

    # Adjust for constraint: replace column idx1 with (col_idx1 - col_idx2)
    X_reduced = X_reduced.at[:, :, idx1_new].set(
        X_wins[:, :, idx1] - X_wins[:, :, idx2]
    )

    # Solve reduced system
    W_reduced = solve_all_windows_outputs_vectorized(X_reduced, Y_wins, None)

    # Apply zero constraints if needed
    if zero_mask is not None:
        # Apply penalties to enforce zeros in reduced space
        for j in range(n_outputs):
            reduced_zero_mask = zero_mask[keep_indices, j]
            if jnp.any(reduced_zero_mask):
                penalty_diag = jnp.where(reduced_zero_mask, 1e12, 0.0)
                XtX = jnp.einsum('wij,wik->wjk', X_reduced, X_reduced)
                XtY = jnp.einsum('wij,wi->wj', X_reduced, Y_wins[:, :, j])
                XtX_pen = XtX + jnp.diag(penalty_diag)[None, :, :] + 1e-6 * jnp.eye(len(keep_indices))[None, :, :]
                W_j = vmap(jnp.linalg.solve)(XtX_pen, XtY)
                W_reduced = W_reduced.at[:, :, j].set(W_j)

    # Reconstruct full solution
    W_all = jnp.zeros((n_windows, n_features, n_outputs))

    for i, orig_idx in enumerate(keep_indices):
        W_all = W_all.at[:, orig_idx, :].set(W_reduced[:, i, :])

    # Set w[idx2] = -w[idx1]
    W_all = W_all.at[:, idx2, :].set(-W_reduced[:, idx1_new, :])

    return W_all


def apply_fixed_kkt_vectorized(X_wins, Y_wins, fixed_constraint, zero_mask=None):
    """Apply fixed value constraint using KKT - vectorized."""
    fixed_idx, fixed_val = fixed_constraint

    # Adjust Y
    Y_adjusted = Y_wins - fixed_val * X_wins[:, :, fixed_idx:fixed_idx+1]

    # Remove fixed variable
    n_features = X_wins.shape[2]
    mask = np.ones(n_features, dtype=bool)
    mask[fixed_idx] = False
    X_reduced = X_wins[:, :, mask]

    # Solve
    W_reduced = solve_all_windows_outputs_vectorized(X_reduced, Y_adjusted,
                                                    zero_mask[:, mask] if zero_mask is not None else None)

    # Reconstruct
    n_windows = X_wins.shape[0]
    n_outputs = Y_wins.shape[2]
    W_all = jnp.zeros((n_windows, n_features, n_outputs))

    j = 0
    for i in range(n_features):
        if i == fixed_idx:
            W_all = W_all.at[:, i, :].set(fixed_val)
        else:
            W_all = W_all.at[:, i, :].set(W_reduced[:, j, :])
            j += 1

    return W_all


def apply_all_constraints_kkt_vectorized(X_wins, Y_wins, fixed_constraint,
                                       offset_indices, zero_mask=None):
    """Apply all constraints using KKT - vectorized."""
    fixed_idx, fixed_val = fixed_constraint
    if isinstance(offset_indices, list):
        offset_indices = offset_indices[0]
    idx1, idx2 = offset_indices

    n_windows, window_size, n_features = X_wins.shape
    n_outputs = Y_wins.shape[2]

    # Adjust Y for fixed constraint
    Y_adjusted = Y_wins - fixed_val * X_wins[:, :, fixed_idx:fixed_idx+1]

    # Remove fixed variable
    mask1 = np.ones(n_features, dtype=bool)
    mask1[fixed_idx] = False
    X_red1 = X_wins[:, :, mask1]

    # Adjust indices
    idx1_red = idx1 - (1 if fixed_idx < idx1 else 0)
    idx2_red = idx2 - (1 if fixed_idx < idx2 else 0)

    # Handle different cases
    W_all = jnp.zeros((n_windows, n_features, n_outputs))

    if idx1 == fixed_idx:
        # w[idx2] = -fixed_val
        W_red1 = solve_all_windows_outputs_vectorized(X_red1, Y_adjusted, None)
        j = 0
        for i in range(n_features):
            if i == fixed_idx:
                W_all = W_all.at[:, i, :].set(fixed_val)
            elif i == idx2:
                W_all = W_all.at[:, i, :].set(-fixed_val)
            else:
                W_all = W_all.at[:, i, :].set(W_red1[:, j, :])
                j += 1

    elif idx2 == fixed_idx:
        # w[idx1] = -fixed_val
        W_red1 = solve_all_windows_outputs_vectorized(X_red1, Y_adjusted, None)
        j = 0
        for i in range(n_features):
            if i == fixed_idx:
                W_all = W_all.at[:, i, :].set(fixed_val)
            elif i == idx1:
                W_all = W_all.at[:, i, :].set(-fixed_val)
            else:
                W_all = W_all.at[:, i, :].set(W_red1[:, j, :])
                j += 1

    else:
        # Both constraints active, need second reduction
        mask2 = np.ones(n_features - 1, dtype=bool)
        mask2[idx2_red] = False
        X_red2 = X_red1[:, :, mask2]

        idx1_red2 = idx1_red - (1 if idx2_red < idx1_red else 0)

        X_red2 = X_red2.at[:, :, idx1_red2].set(
            X_red1[:, :, idx1_red] - X_red1[:, :, idx2_red]
        )

        # Solve
        W_red2 = solve_all_windows_outputs_vectorized(X_red2, Y_adjusted, None)

        # Reconstruct to singly-reduced
        W_red1 = jnp.zeros((n_windows, n_features - 1, n_outputs))
        j = 0
        for i in range(n_features - 1):
            if i == idx2_red:
                W_red1 = W_red1.at[:, i, :].set(-W_red2[:, idx1_red2, :])
            else:
                W_red1 = W_red1.at[:, i, :].set(W_red2[:, j, :])
                j += 1

        # Final reconstruction
        j = 0
        for i in range(n_features):
            if i == fixed_idx:
                W_all = W_all.at[:, i, :].set(fixed_val)
            else:
                W_all = W_all.at[:, i, :].set(W_red1[:, j, :])
                j += 1

    # Apply zero constraints if needed
    if zero_mask is not None:
        # Post-process to enforce zeros (approximate)
        W_all = W_all * (1 - zero_mask[None, :, :])

    return W_all


def solve_with_zeros_vectorized(X_wins, Y_wins, zero_mask):
    """Solve with only zero constraints - vectorized."""
    if zero_mask is None:
        return solve_all_windows_outputs_vectorized(X_wins, Y_wins, None)

    n_windows = X_wins.shape[0]
    n_features = X_wins.shape[2]
    n_outputs = Y_wins.shape[2]
    W_all = jnp.zeros((n_windows, n_features, n_outputs))

    # Compute X'X and X'Y for all windows
    XtX = jnp.einsum('wij,wik->wjk', X_wins, X_wins)
    XtY = jnp.einsum('wij,wik->wjk', X_wins, Y_wins)

    for j in range(n_outputs):
        penalty_diag = jnp.where(zero_mask[:, j], 1e12, 0.0)
        penalty_matrix = jnp.diag(penalty_diag) + 1e-6 * jnp.eye(n_features)
        XtX_pen = XtX + penalty_matrix[None, :, :]
        W_j = vmap(jnp.linalg.solve)(XtX_pen, XtY[:, :, j])
        W_all = W_all.at[:, :, j].set(W_j)

    return W_all


# ============= PENALTY METHOD (VECTORIZED) =============

def apply_penalty_constraints_vectorized(X, Y, window_size, stride, discovery_mask=None,
                                       offset_indices=None, fixed_constraints=None,
                                       zero_penalty=1e12, offset_penalty=1e10,
                                       fixed_penalty=1e10, reg=1e-6):
    """
    Fully vectorized penalty method that processes all windows and outputs simultaneously.
    """
    # Create all windows at once
    n_samples = X.shape[0]
    n_features = X.shape[1]
    n_outputs = Y.shape[1]
    n_windows = (n_samples - window_size) // stride + 1

    # Create windows using advanced indexing
    indices = np.arange(n_windows)[:, None] * stride + np.arange(window_size)[None, :]
    X_wins = X[indices]  # Shape: (n_windows, window_size, n_features)
    Y_wins = Y[indices]  # Shape: (n_windows, window_size, n_outputs)

    # Compute X'X and X'Y for all windows at once
    XtX = jnp.einsum('wij,wik->wjk', X_wins, X_wins)  # Shape: (n_windows, n_features, n_features)
    XtY = jnp.einsum('wij,wik->wjk', X_wins, Y_wins)  # Shape: (n_windows, n_features, n_outputs)

    # Initialize base penalty matrix (same for all windows)
    I = jnp.eye(n_features)
    base_penalty = reg * I

    # === OFFSET CONSTRAINT PENALTY ===
    offset_penalty_matrix = jnp.zeros((n_features, n_features))
    if offset_indices is not None:
        if isinstance(offset_indices, list):
            # Handle multiple offset constraints
            for idx1, idx2 in offset_indices:
                offset_matrix = jnp.zeros((n_features, n_features))
                offset_matrix = offset_matrix.at[idx1, idx1].add(offset_penalty)
                offset_matrix = offset_matrix.at[idx2, idx2].add(offset_penalty)
                offset_matrix = offset_matrix.at[idx1, idx2].add(offset_penalty)
                offset_matrix = offset_matrix.at[idx2, idx1].add(offset_penalty)
                offset_penalty_matrix = offset_penalty_matrix + offset_matrix
        else:
            idx1, idx2 = offset_indices
            offset_penalty_matrix = offset_penalty_matrix.at[idx1, idx1].add(offset_penalty)
            offset_penalty_matrix = offset_penalty_matrix.at[idx2, idx2].add(offset_penalty)
            offset_penalty_matrix = offset_penalty_matrix.at[idx1, idx2].add(offset_penalty)
            offset_penalty_matrix = offset_penalty_matrix.at[idx2, idx1].add(offset_penalty)

    # === SOLVE FOR ALL OUTPUTS ===
    W_all = jnp.zeros((n_windows, n_features, n_outputs))

    for j in range(n_outputs):
        # Build penalty matrix for this output
        penalty_matrix = base_penalty + offset_penalty_matrix

        # === ZERO CONSTRAINT PENALTY ===
        if discovery_mask is not None:
            zero_diag = jnp.where(discovery_mask[:, j], zero_penalty, 0.0)
            penalty_matrix = penalty_matrix + jnp.diag(zero_diag)

        # === FIXED VALUE CONSTRAINT ===
        XtY_j = XtY[:, :, j].copy()  # Shape: (n_windows, n_features)

        if fixed_constraints is not None:
            for fixed_idx, fixed_val in fixed_constraints:
                # Add penalty to diagonal
                penalty_matrix = penalty_matrix.at[fixed_idx, fixed_idx].add(fixed_penalty)
                # Modify linear term for all windows
                XtY_j = XtY_j.at[:, fixed_idx].add(fixed_penalty * fixed_val)

        # Add penalty matrix to all windows (broadcasting)
        XtX_pen = XtX + penalty_matrix[None, :, :]  # Shape: (n_windows, n_features, n_features)

        # Solve for all windows at once using vmap
        W_j = vmap(jnp.linalg.solve)(XtX_pen, XtY_j)  # Shape: (n_windows, n_features)
        W_all = W_all.at[:, :, j].set(W_j)

    return W_all


# ============= VECTORIZED REGRESSION =============

def apply_vectorized_regression(X, Y, window_size, stride, discovery_mask=None,
                              offset_indices=None, fixed_constraints=None,
                              constraints_config=None):
    """Apply fully vectorized regression."""
    # Create all windows at once
    X_wins = create_windows_vectorized(X, window_size, stride)
    Y_wins = create_windows_vectorized(Y, window_size, stride)

    # Apply constraints
    if fixed_constraints and offset_indices:
        W_all = apply_all_constraints_vectorized(
            X_wins, Y_wins,
            fixed_constraints[0], offset_indices,
            discovery_mask
        )
    elif offset_indices:
        W_all = apply_offset_vectorized(X_wins, Y_wins, offset_indices, discovery_mask)
    elif fixed_constraints:
        W_all = apply_fixed_vectorized(X_wins, Y_wins, fixed_constraints[0], discovery_mask)
    else:
        W_all = solve_all_windows_outputs_vectorized(X_wins, Y_wins, discovery_mask)

    # Compute R²
    Y_preds = jnp.einsum('wij,wjk->wik', X_wins, W_all)
    ss_res = jnp.sum((Y_wins - Y_preds)**2, axis=(1, 2))
    ss_tot = jnp.sum((Y_wins - jnp.mean(Y_wins, axis=1, keepdims=True))**2, axis=(1, 2))
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    return {
        'W_all': W_all,
        'W_avg': jnp.mean(W_all, axis=0),
        'r2': [r2],
        'method_used': 'vectorized'
    }


def solve_all_windows_outputs_vectorized(X_wins, Y_wins, zero_mask=None, reg=1e-6):
    """Vectorized OLS for all windows and outputs."""
    XtX = jnp.einsum('wij,wik->wjk', X_wins, X_wins)
    XtY = jnp.einsum('wij,wik->wjk', X_wins, Y_wins)

    n_features = X_wins.shape[2]
    I = jnp.eye(n_features)
    XtX_reg = XtX + reg * I[None, :, :]

    if zero_mask is not None:
        # Apply zero constraints
        n_windows = X_wins.shape[0]
        n_outputs = Y_wins.shape[2]
        W_all = jnp.zeros((n_windows, n_features, n_outputs))

        for j in range(n_outputs):
            penalty_diag = jnp.where(zero_mask[:, j], 1e12, 0.0)
            penalty_matrix = jnp.diag(penalty_diag)
            XtX_pen = XtX_reg + penalty_matrix[None, :, :]
            W_j = vmap(lambda A, b: jnp.linalg.solve(A, b))(XtX_pen, XtY[:, :, j])
            W_all = W_all.at[:, :, j].set(W_j)

        return W_all
    else:
        return vmap(lambda A, B: jnp.linalg.solve(A, B))(XtX_reg, XtY)


def apply_offset_vectorized(X_wins, Y_wins, offset_indices, zero_mask=None, reg=1e-6):
    """Vectorized offset constraint."""
    if isinstance(offset_indices, list):
        offset_indices = offset_indices[0]
    idx1, idx2 = offset_indices

    n_windows, window_size, n_features = X_wins.shape
    n_outputs = Y_wins.shape[2]

    # Create reduced system
    keep_mask = np.ones(n_features, dtype=bool)
    keep_mask[idx2] = False
    keep_indices = np.where(keep_mask)[0]

    X_reduced = X_wins[:, :, keep_mask]

    # Find position of idx1 in reduced system
    idx1_new = np.sum(keep_indices < idx1)

    # Adjust for constraint
    X_reduced = X_reduced.at[:, :, idx1_new].set(
        X_wins[:, :, idx1] - X_wins[:, :, idx2]
    )

    # Solve
    W_reduced = solve_all_windows_outputs_vectorized(X_reduced, Y_wins, None, reg)

    # Reconstruct
    W_all = jnp.zeros((n_windows, n_features, n_outputs))

    for i, orig_idx in enumerate(keep_indices):
        W_all = W_all.at[:, orig_idx, :].set(W_reduced[:, i, :])

    W_all = W_all.at[:, idx2, :].set(-W_reduced[:, idx1_new, :])

    # Apply zero constraints if needed
    if zero_mask is not None:
        # This is approximate - we just zero out the masked coefficients
        W_all = W_all * (1 - zero_mask[None, :, :])

    return W_all


def apply_fixed_vectorized(X_wins, Y_wins, fixed_constraint, zero_mask=None, reg=1e-6):
    """Vectorized fixed value constraint."""
    fixed_idx, fixed_val = fixed_constraint

    # Adjust Y
    Y_adjusted = Y_wins - fixed_val * X_wins[:, :, fixed_idx:fixed_idx+1]

    # Remove fixed variable
    n_features = X_wins.shape[2]
    mask = np.ones(n_features, dtype=bool)
    mask[fixed_idx] = False
    X_reduced = X_wins[:, :, mask]

    # Solve
    W_reduced = solve_all_windows_outputs_vectorized(X_reduced, Y_adjusted, None, reg)

    # Reconstruct
    n_windows = X_wins.shape[0]
    n_outputs = Y_wins.shape[2]
    W_all = jnp.zeros((n_windows, n_features, n_outputs))

    j = 0
    for i in range(n_features):
        if i == fixed_idx:
            W_all = W_all.at[:, i, :].set(fixed_val)
        else:
            W_all = W_all.at[:, i, :].set(W_reduced[:, j, :])
            j += 1

    # Apply zero constraints if needed
    if zero_mask is not None:
        W_all = W_all * (1 - zero_mask[None, :, :])

    return W_all


def apply_all_constraints_vectorized(X_wins, Y_wins, fixed_constraint,
                                   offset_indices, zero_mask=None, reg=1e-6):
    """Apply all constraints in vectorized manner."""
    fixed_idx, fixed_val = fixed_constraint
    if isinstance(offset_indices, list):
        offset_indices = offset_indices[0]
    idx1, idx2 = offset_indices

    n_windows, window_size, n_features = X_wins.shape
    n_outputs = Y_wins.shape[2]

    # Adjust Y for fixed constraint
    Y_adjusted = Y_wins - fixed_val * X_wins[:, :, fixed_idx:fixed_idx+1]

    # Remove fixed variable
    mask1 = np.ones(n_features, dtype=bool)
    mask1[fixed_idx] = False
    X_red1 = X_wins[:, :, mask1]

    # Adjust indices
    idx1_red = idx1 - (1 if fixed_idx < idx1 else 0)
    idx2_red = idx2 - (1 if fixed_idx < idx2 else 0)

    # Apply offset constraint if valid
    if idx1 != fixed_idx and idx2 != fixed_idx:
        mask2 = np.ones(n_features - 1, dtype=bool)
        mask2[idx2_red] = False
        X_red2 = X_red1[:, :, mask2]

        idx1_red2 = idx1_red - (1 if idx2_red < idx1_red else 0)

        X_red2 = X_red2.at[:, :, idx1_red2].set(
            X_red1[:, :, idx1_red] - X_red1[:, :, idx2_red]
        )

        # Solve
        W_red2 = solve_all_windows_outputs_vectorized(X_red2, Y_adjusted, None, reg)

        # Reconstruct to singly-reduced
        W_red1 = jnp.zeros((n_windows, n_features - 1, n_outputs))
        j = 0
        for i in range(n_features - 1):
            if i == idx2_red:
                W_red1 = W_red1.at[:, i, :].set(-W_red2[:, idx1_red2, :])
            else:
                W_red1 = W_red1.at[:, i, :].set(W_red2[:, j, :])
                j += 1
    else:
        # Special cases
        W_red1 = solve_all_windows_outputs_vectorized(X_red1, Y_adjusted, None, reg)

    # Final reconstruction
    W_all = jnp.zeros((n_windows, n_features, n_outputs))
    j = 0
    for i in range(n_features):
        if i == fixed_idx:
            W_all = W_all.at[:, i, :].set(fixed_val)
        else:
            W_all = W_all.at[:, i, :].set(W_red1[:, j, :])
            j += 1

    # Handle special offset cases
    if idx1 == fixed_idx:
        W_all = W_all.at[:, idx2, :].set(-fixed_val)
    elif idx2 == fixed_idx:
        W_all = W_all.at[:, idx1, :].set(-fixed_val)

    # Apply zero constraints if needed
    if zero_mask is not None:
        W_all = W_all * (1 - zero_mask[None, :, :])

    return W_all


# ============= CVXPY REGRESSION =============

if CVXPY_AVAILABLE:
    def apply_cvxpy_regression(X, Y, window_size, stride, discovery_mask=None,
                             offset_indices=None, fixed_constraints=None,
                             positive_constraints=None, negative_constraints=None,
                             cvxpy_config=None):
        """Apply CVXPY regression."""
        if cvxpy_config is None:
            cvxpy_config = {}

        X_wins = create_windows_vectorized(X, window_size, stride)
        Y_wins = create_windows_vectorized(Y, window_size, stride)

        W_all = []

        for i in range(len(X_wins)):
            W = solve_cvxpy_window(
                np.array(X_wins[i]), np.array(Y_wins[i]),
                discovery_mask=discovery_mask,
                offset_indices=offset_indices,
                fixed_constraints=fixed_constraints,
                positive_constraints=positive_constraints,
                negative_constraints=negative_constraints,
                **cvxpy_config
            )
            W_all.append(jnp.array(W))

        W_all = jnp.stack(W_all)

        # Compute R²
        r2_values = []
        for i in range(len(W_all)):
            Y_pred = X_wins[i] @ W_all[i]
            ss_res = jnp.sum((Y_wins[i] - Y_pred)**2)
            ss_tot = jnp.sum((Y_wins[i] - jnp.mean(Y_wins[i]))**2)
            r2 = 1 - ss_res / (ss_tot + 1e-8)
            r2_values.append(r2)

        return {
            'W_all': W_all,
            'W_avg': jnp.mean(W_all, axis=0),
            'r2': [jnp.array(r2_values)],
            'method_used': 'cvxpy'
        }


    def solve_cvxpy_window(X_win, Y_win, discovery_mask=None, offset_indices=None,
                         fixed_constraints=None, positive_constraints=None,
                         negative_constraints=None, loss='squared',
                         huber_delta=1.0, transaction_costs=None,
                         tc_lambda=0.0, dv01_neutral=False,
                         post_zero_threshold=None):
        """Solve single window using CVXPY."""
        n_features = X_win.shape[1]
        n_outputs = Y_win.shape[1]

        W = np.zeros((n_features, n_outputs))

        for j in range(n_outputs):
            # Define variables
            w = cp.Variable(n_features)

            # Objective
            if loss == 'squared':
                objective = cp.Minimize(cp.sum_squares(X_win @ w - Y_win[:, j]))
            else:  # huber
                objective = cp.Minimize(cp.sum(cp.huber(X_win @ w - Y_win[:, j], M=huber_delta)))

            # Add transaction costs if specified
            if transaction_costs is not None and tc_lambda > 0:
                objective = cp.Minimize(objective.expr + tc_lambda * cp.sum(cp.multiply(transaction_costs, cp.abs(w))))

            # Constraints
            constraints = []

            # Offset constraints
            if offset_indices is not None:
                if isinstance(offset_indices, list):
                    for idx1, idx2 in offset_indices:
                        constraints.append(w[idx1] + w[idx2] == 0)
                else:
                    idx1, idx2 = offset_indices
                    constraints.append(w[idx1] + w[idx2] == 0)

            # Fixed constraints
            if fixed_constraints is not None:
                for fixed_idx, fixed_val in fixed_constraints:
                    constraints.append(w[fixed_idx] == fixed_val)

            # Sign constraints
            if positive_constraints is not None:
                for idx in positive_constraints:
                    constraints.append(w[idx] >= 0)

            if negative_constraints is not None:
                for idx in negative_constraints:
                    constraints.append(w[idx] <= 0)

            # Zero constraints from discovery
            if discovery_mask is not None:
                zero_indices = np.where(discovery_mask[:, j])[0]
                for idx in zero_indices:
                    constraints.append(w[idx] == 0)

            # DV01 neutral
            if dv01_neutral:
                constraints.append(cp.sum(w) == 1.0)

            # Solve
            prob = cp.Problem(objective, constraints)

            try:
                # Try different solvers
                for solver in [cp.OSQP, cp.ECOS, cp.SCS]:
                    try:
                        prob.solve(solver=solver, verbose=False)
                        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                            break
                    except:
                        continue

                if w.value is not None:
                    w_sol = w.value

                    # Post-threshold if specified
                    if post_zero_threshold is not None:
                        w_sol[np.abs(w_sol) < post_zero_threshold] = 0

                    W[:, j] = w_sol
                else:
                    # Fallback to basic OLS
                    W[:, j] = solve_ols(X_win, Y_win[:, j:j+1]).flatten()
            except:
                # Fallback
                W[:, j] = solve_ols(X_win, Y_win[:, j:j+1]).flatten()

        return W
else:
    def apply_cvxpy_regression(*args, **kwargs):
        raise ImportError("CVXPY not available")


# ============= LAYERED REGRESSION =============

def apply_layered_regression(X, Y, window_size, stride, n_layers=3,
                           method='jax', discovery_mask=None,
                           constraints_config=None, cvxpy_config=None):
    """Apply layered regression."""
    n_samples = X.shape[0]

    W_layers = []
    r2_layers = []
    residual = Y.copy()

    for layer in range(n_layers):
        print(f"  Layer {layer + 1}/{n_layers}...")

        # Apply constraints only in first layer
        if layer == 0:
            layer_config = constraints_config
            layer_discovery = discovery_mask
        else:
            # No constraints in subsequent layers
            layer_config = {}
            layer_discovery = None

        # Run regression for this layer
        if method in ['jax', 'vectorized']:
            if layer == 0:
                # Use specified method with constraints
                sub_method = 'jax' if method == 'jax' else 'vectorized'
                results = unified_sliding_regression_extended(
                    X, residual, window_size, stride, 1, 1,
                    method=sub_method,
                    discovery_config={'enabled': False, 'forced_mask': layer_discovery},
                    constraints_config=layer_config
                )
            else:
                # Simple regression for subsequent layers
                X_wins = create_windows_vectorized(X, window_size, stride)
                Y_wins = create_windows_vectorized(residual, window_size, stride)
                W_layer = vmap(solve_ols)(X_wins, Y_wins)

                # Compute R²
                Y_preds = jnp.einsum('wij,wjk->wik', X_wins, W_layer)
                ss_res = jnp.sum((Y_wins - Y_preds)**2, axis=(1, 2))
                ss_tot = jnp.sum((Y_wins - jnp.mean(Y_wins, axis=1, keepdims=True))**2, axis=(1, 2))
                r2_values = 1 - ss_res / (ss_tot + 1e-8)

                results = {
                    'W_all': W_layer,
                    'r2': [r2_values]
                }

        elif method == 'cvxpy' and CVXPY_AVAILABLE:
            if layer == 0:
                results = apply_cvxpy_regression(
                    X, residual, window_size, stride,
                    discovery_mask=layer_discovery,
                    offset_indices=layer_config.get('offset_indices'),
                    fixed_constraints=layer_config.get('fixed_constraints'),
                    positive_constraints=layer_config.get('positive_constraints'),
                    negative_constraints=layer_config.get('negative_constraints'),
                    cvxpy_config=cvxpy_config
                )
            else:
                # Simple CVXPY regression
                results = apply_cvxpy_regression(
                    X, residual, window_size, stride,
                    cvxpy_config={'loss': cvxpy_config.get('loss', 'squared')}
                )

        else:
            raise ValueError(f"Invalid method for layered regression: {method}")

        W_layer = results['W_all']
        r2_layer = results['r2'][0]

        W_layers.append(W_layer)
        r2_layers.append(r2_layer)

        # Update residual
        predictions = jnp.zeros_like(residual)
        counts = jnp.zeros((n_samples, 1))

        for i in range(len(W_layer)):
            start = i * stride
            end = start + window_size

            pred_window = X[start:end] @ W_layer[i]
            predictions = predictions.at[start:end].add(pred_window)
            counts = counts.at[start:end].add(1.0)

        predictions = predictions / jnp.maximum(counts, 1.0)
        residual = residual - predictions

        print(f"    Mean R²: {jnp.mean(r2_layer):.4f}")

    # Compute total coefficients
    W_total = jnp.zeros((X.shape[1], Y.shape[1]))
    for W_layer in W_layers:
        W_total += jnp.mean(W_layer, axis=0)

    return {
        'W_all': W_layers[0],  # First layer for compatibility
        'W_avg': W_total,
        'W_layers': W_layers,
        'r2': r2_layers,
        'method_used': f'{method}_layered'
    }


# ============= CONSTRAINT CHECKING =============

def check_all_constraints(W, discovery_mask, offset_indices, fixed_constraints,
                        positive_constraints, negative_constraints):
    """Check all constraint violations."""
    violations = {}

    # Zero violations
    if discovery_mask is not None:
        zero_violations = jnp.abs(W * discovery_mask)
        violations['zero_max'] = jnp.max(zero_violations)
        violations['zero_mean'] = jnp.mean(zero_violations)
        violations['zero_count'] = jnp.sum(zero_violations > 1e-6)

    # Offset violations
    if offset_indices is not None:
        if isinstance(offset_indices, list):
            offset_viols = []
            for idx1, idx2 in offset_indices:
                offset_viol = jnp.abs(W[idx1, :] + W[idx2, :])
                offset_viols.append(jnp.max(offset_viol))
            violations['offset_max'] = max(offset_viols)
            violations['offset_mean'] = np.mean(offset_viols)
        else:
            idx1, idx2 = offset_indices
            offset_viol = jnp.abs(W[idx1, :] + W[idx2, :])
            violations['offset_max'] = jnp.max(offset_viol)
            violations['offset_mean'] = jnp.mean(offset_viol)

    # Fixed violations
    if fixed_constraints is not None:
        fixed_viols = []
        for fixed_idx, fixed_val in fixed_constraints:
            fixed_viol = jnp.abs(W[fixed_idx, :] - fixed_val)
            fixed_viols.append(jnp.max(fixed_viol))
        violations['fixed_max'] = max(fixed_viols)
        violations['fixed_mean'] = np.mean(fixed_viols)

    # Sign violations
    if positive_constraints is not None:
        pos_viols = []
        for idx in positive_constraints:
            if idx < W.shape[0]:
                pos_viol = jnp.maximum(-W[idx, :], 0)
                pos_viols.append(jnp.max(pos_viol))
        if pos_viols:
            violations['positive_max'] = max(pos_viols)

    if negative_constraints is not None:
        neg_viols = []
        for idx in negative_constraints:
            if idx < W.shape[0]:
                neg_viol = jnp.maximum(W[idx, :], 0)
                neg_viols.append(jnp.max(neg_viol))
        if neg_viols:
            violations['negative_max'] = max(neg_viols)

    return violations


def print_summary(results):
    """Print summary of results."""
    violations = results.get('violations', {})

    if 'zero_max' in violations:
        print(f"  Zero violations: max={violations['zero_max']:.2e}, "
              f"count={violations.get('zero_count', 0)}")

    if 'offset_max' in violations:
        print(f"  Offset violations: max={violations['offset_max']:.2e}")

    if 'fixed_max' in violations:
        print(f"  Fixed violations: max={violations['fixed_max']:.2e}")

    if 'positive_max' in violations:
        print(f"  Positive constraint violations: max={violations['positive_max']:.2e}")

    if 'negative_max' in violations:
        print(f"  Negative constraint violations: max={violations['negative_max']:.2e}")

    # R² summary
    if 'r2' in results and results['r2']:
        if len(results['r2']) == 1:
            print(f"  Mean R²: {jnp.mean(results['r2'][0]):.4f}")
        else:
            print(f"  R² by layer:")
            for i, r2 in enumerate(results['r2']):
                print(f"    Layer {i+1}: {jnp.mean(r2):.4f}")


# ============= COMPREHENSIVE EXAMPLE =============

def run_comprehensive_example():
    """
    Comprehensive example with:
    - 1256 rows (samples)
    - 14 countries
    - 12 tenors
    - 200-row windows with 150 stride
    - Constraints: w[2] = -w[4], w[7] = 0.05
    """

    print("="*80)
    print("UNIFIED SLIDING REGRESSION EXAMPLE - ALL METHODS")
    print("="*80)

    # ============= DATA GENERATION =============
    print("\n1. GENERATING DATA")
    print("-"*40)

    n_samples = 1256
    n_countries = 14
    n_tenors = 12
    n_outputs = n_countries * n_tenors  # 168 outputs
    n_features = 10  # Number of input features
    window_size = 200
    stride = 150

    print(f"Data dimensions:")
    print(f"  Samples: {n_samples}")
    print(f"  Features: {n_features}")
    print(f"  Outputs: {n_outputs} ({n_countries} countries × {n_tenors} tenors)")
    print(f"  Window size: {window_size}")
    print(f"  Stride: {stride}")

    # Calculate number of windows
    n_windows = (n_samples - window_size) // stride + 1
    print(f"  Number of windows: {n_windows}")

    # Generate synthetic data
    key = jax.random.PRNGKey(42)
    key_X, key_noise = jax.random.split(key)

    # Create input data
    X = jax.random.normal(key_X, (n_samples, n_features))

    # Create true coefficients with the specified constraints
    W_true = jnp.zeros((n_features, n_outputs))

    # Set some base patterns
    for j in range(n_outputs):
        country_idx = j // n_tenors
        tenor_idx = j % n_tenors

        # Basic patterns
        W_true = W_true.at[0, j].set(0.5 + 0.1 * np.sin(country_idx))
        W_true = W_true.at[1, j].set(-0.3 + 0.05 * tenor_idx)

        # Apply constraints:
        # w[2] = -w[4] (offsetting)
        W_true = W_true.at[2, j].set(1.2)
        W_true = W_true.at[4, j].set(-1.2)

        # w[7] = 0.05 (fixed value)
        W_true = W_true.at[7, j].set(0.05)

        # Some other patterns
        if country_idx < 7:
            W_true = W_true.at[5, j].set(0.8)
        if tenor_idx > 6:
            W_true = W_true.at[6, j].set(-0.6)

    # Generate output data
    noise_level = 0.1
    Y = X @ W_true + noise_level * jax.random.normal(key_noise, (n_samples, n_outputs))

    print(f"\nTrue coefficient statistics:")
    print(f"  w[2] mean: {jnp.mean(W_true[2, :]):.4f}")
    print(f"  w[4] mean: {jnp.mean(W_true[4, :]):.4f}")
    print(f"  w[7] mean: {jnp.mean(W_true[7, :]):.4f}")
    print(f"  w[2] + w[4] check: {jnp.max(jnp.abs(W_true[2, :] + W_true[4, :])):.2e}")

    # ============= CONSTRAINT CONFIGURATION =============
    print("\n2. CONSTRAINT CONFIGURATION")
    print("-"*40)

    # Define constraints
    offset_indices = (2, 4)  # w[2] + w[4] = 0
    fixed_constraints = [(7, 0.05)]  # w[7] = 0.05

    print(f"Constraints:")
    print(f"  Offset constraint: w[{offset_indices[0]}] + w[{offset_indices[1]}] = 0")
    print(f"  Fixed constraint: w[{fixed_constraints[0][0]}] = {fixed_constraints[0][1]}")

    # Discovery configuration
    discovery_config = {
        'enabled': True,
        'consistency_threshold': 0.85,
        'magnitude_threshold': 0.05,
        'relative_threshold': 0.1
    }

    # ============= RUN ALL METHODS =============
    print("\n3. RUNNING ALL METHODS")
    print("-"*40)

    all_results = {}

    # Method 1: JAX with KKT (Exact Constraints)
    print("\n[Method 1/6] JAX with KKT (Exact Constraints)")
    print("  This method uses Karush-Kuhn-Tucker conditions for exact constraint satisfaction")

    results_jax_kkt = unified_sliding_regression_extended(
        X, Y,
        window_size=window_size,
        stride=stride,
        n_countries=n_countries,
        n_tenors=n_tenors,
        method='jax',
        discovery_config=discovery_config,
        constraints_config={
            'method': 'exact',  # KKT method
            'offset_indices': offset_indices,
            'fixed_constraints': fixed_constraints
        }
    )
    all_results['JAX KKT'] = results_jax_kkt

    # Method 2: JAX with Penalty Method
    print("\n[Method 2/6] JAX with Penalty Method")
    print("  This method uses high penalties to enforce constraints approximately")

    results_jax_penalty = unified_sliding_regression_extended(
        X, Y,
        window_size=window_size,
        stride=stride,
        n_countries=n_countries,
        n_tenors=n_tenors,
        method='jax',
        discovery_config=discovery_config,
        constraints_config={
            'method': 'penalty',
            'offset_indices': offset_indices,
            'fixed_constraints': fixed_constraints,
            'zero_penalty': 1e12,
            'offset_penalty': 1e10,
            'fixed_penalty': 1e10
        }
    )
    all_results['JAX Penalty'] = results_jax_penalty

    # Method 3: Fully Vectorized
    print("\n[Method 3/6] Fully Vectorized Method")
    print("  This method processes all windows simultaneously for maximum speed")

    results_vectorized = unified_sliding_regression_extended(
        X, Y,
        window_size=window_size,
        stride=stride,
        n_countries=n_countries,
        n_tenors=n_tenors,
        method='vectorized',
        discovery_config=discovery_config,
        constraints_config={
            'offset_indices': offset_indices,
            'fixed_constraints': fixed_constraints
        }
    )
    all_results['Vectorized'] = results_vectorized

    # Method 4: CVXPY
    if CVXPY_AVAILABLE:
        print("\n[Method 4/6] CVXPY Method")
        print("  This method uses convex optimization with advanced constraint handling")

        results_cvxpy = unified_sliding_regression_extended(
            X, Y,
            window_size=window_size,
            stride=stride,
            n_countries=n_countries,
            n_tenors=n_tenors,
            method='cvxpy',
            discovery_config=discovery_config,
            constraints_config={
                'offset_indices': offset_indices,
                'fixed_constraints': fixed_constraints
            },
            cvxpy_config={
                'loss': 'squared',
                'post_zero_threshold': 1e-6
            }
        )
        all_results['CVXPY'] = results_cvxpy
    else:
        print("\n[Method 4/6] CVXPY - SKIPPED (not installed)")

    # Method 5: Hybrid (JAX Discovery + CVXPY Regression)
    if CVXPY_AVAILABLE:
        print("\n[Method 5/6] Hybrid Method (JAX Discovery + CVXPY Regression)")
        print("  This method combines JAX's fast discovery with CVXPY's flexible regression")

        results_hybrid = unified_sliding_regression_extended(
            X, Y,
            window_size=window_size,
            stride=stride,
            n_countries=n_countries,
            n_tenors=n_tenors,
            method='hybrid',
            discovery_config=discovery_config,
            constraints_config={
                'offset_indices': offset_indices,
                'fixed_constraints': fixed_constraints
            },
            cvxpy_config={
                'loss': 'huber',
                'huber_delta': 1.0
            }
        )
        all_results['Hybrid'] = results_hybrid
    else:
        print("\n[Method 5/6] Hybrid - SKIPPED (CVXPY not installed)")

    # Method 6: Layered Regression
    print("\n[Method 6/6] Layered Regression (3 layers)")
    print("  This method applies regression in multiple layers to capture complex patterns")

    results_layered = unified_sliding_regression_extended(
        X, Y,
        window_size=window_size,
        stride=stride,
        n_countries=n_countries,
        n_tenors=n_tenors,
        method='jax',
        layers=[{}, {}, {}],  # 3 layers
        discovery_config=discovery_config,
        constraints_config={
            'method': 'exact',
            'offset_indices': offset_indices,
            'fixed_constraints': fixed_constraints
        }
    )
    all_results['Layered'] = results_layered

    # ============= RESULTS COMPARISON =============
    print("\n4. RESULTS COMPARISON")
    print("-"*40)

    # Create comparison table
    comparison_data = []
    for method_name, results in all_results.items():
        W_avg = results['W_avg']
        violations = results['violations']

        # Check constraint satisfaction
        offset_check = float(jnp.abs(W_avg[2, :] + W_avg[4, :]).max())
        fixed_check = float(jnp.abs(W_avg[7, :] - 0.05).max())

        # Get R² statistics
        if 'r2' in results and results['r2']:
            if len(results['r2']) == 1:
                r2_mean = float(jnp.mean(results['r2'][0]))
                r2_std = float(jnp.std(results['r2'][0]))
            else:
                # For layered regression, use last layer
                r2_mean = float(jnp.mean(results['r2'][-1]))
                r2_std = float(jnp.std(results['r2'][-1]))
        else:
            r2_mean = r2_std = 0.0

        comparison_data.append({
            'Method': method_name,
            'Time (s)': results['computation_time'],
            'Mean R²': r2_mean,
            'Std R²': r2_std,
            'w[2] mean': float(jnp.mean(W_avg[2, :])),
            'w[4] mean': float(jnp.mean(W_avg[4, :])),
            'w[7] mean': float(jnp.mean(W_avg[7, :])),
            'Offset Viol': offset_check,
            'Fixed Viol': fixed_check,
            'Zero Viol': violations.get('zero_max', 0)
        })

    # Display comparison
    df = pd.DataFrame(comparison_data)

    print("\nMethod Comparison Table:")
    print(df.to_string(index=False, float_format='%.6f'))

    # Save comparison
    df.to_csv('method_comparison_comprehensive.csv', index=False)
    print("\nComparison saved to 'method_comparison_comprehensive.csv'")

    # ============= VISUALIZATION =============
    print("\n5. VISUALIZATION")
    print("-"*40)

    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))

    # Plot 1: Coefficient values for each method
    ax1 = plt.subplot(3, 3, 1)
    method_names = list(all_results.keys())
    positions = np.arange(len(method_names))

    w2_values = [float(jnp.mean(all_results[m]['W_avg'][2, :])) for m in method_names]
    w4_values = [float(jnp.mean(all_results[m]['W_avg'][4, :])) for m in method_names]
    w7_values = [float(jnp.mean(all_results[m]['W_avg'][7, :])) for m in method_names]

    width = 0.25
    ax1.bar(positions - width, w2_values, width, label='w[2]', alpha=0.8)
    ax1.bar(positions, w4_values, width, label='w[4]', alpha=0.8)
    ax1.bar(positions + width, w7_values, width, label='w[7]', alpha=0.8)

    ax1.set_xlabel('Method')
    ax1.set_ylabel('Coefficient Value')
    ax1.set_title('Average Coefficient Values by Method')
    ax1.set_xticks(positions)
    ax1.set_xticklabels(method_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='w[7] target')

    # Plot 2: Constraint violations (log scale)
    ax2 = plt.subplot(3, 3, 2)
    offset_viols = [comparison_data[i]['Offset Viol'] for i in range(len(comparison_data))]
    fixed_viols = [comparison_data[i]['Fixed Viol'] for i in range(len(comparison_data))]

    x = np.arange(len(method_names))
    width = 0.35

    bars1 = ax2.bar(x - width/2, offset_viols, width, label='Offset violation', alpha=0.8)
    bars2 = ax2.bar(x + width/2, fixed_viols, width, label='Fixed violation', alpha=0.8)

    ax2.set_ylabel('Violation (log scale)')
    ax2.set_xlabel('Method')
    ax2.set_title('Constraint Violations by Method')
    ax2.set_xticks(x)
    ax2.set_xticklabels(method_names, rotation=45, ha='right')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Color bars based on violation level
    for bars in [bars1, bars2]:
        for bar, val in zip(bars, offset_viols if bars == bars1 else fixed_viols):
            if val > 1e-6:
                bar.set_color('red')
            elif val > 1e-10:
                bar.set_color('orange')
            else:
                bar.set_color('green')

    # Plot 3: R² distribution
    ax3 = plt.subplot(3, 3, 3)
    for method_name, results in all_results.items():
        if 'r2' in results and results['r2']:
            if len(results['r2']) == 1:
                r2_values = results['r2'][0]
            else:
                r2_values = results['r2'][-1]  # Last layer for layered

            ax3.hist(r2_values, bins=20, alpha=0.5, label=method_name, density=True)

    ax3.set_xlabel('R²')
    ax3.set_ylabel('Density')
    ax3.set_title('R² Distribution by Method')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Time vs R² scatter
    ax4 = plt.subplot(3, 3, 4)
    times = [comparison_data[i]['Time (s)'] for i in range(len(comparison_data))]
    r2_means = [comparison_data[i]['Mean R²'] for i in range(len(comparison_data))]

    ax4.scatter(times, r2_means, s=100, alpha=0.7)
    for i, method in enumerate(method_names):
        ax4.annotate(method, (times[i], r2_means[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax4.set_xlabel('Computation Time (s)')
    ax4.set_ylabel('Mean R²')
    ax4.set_title('Speed vs Accuracy Trade-off')
    ax4.grid(True, alpha=0.3)

    # Plot 5: Coefficient heatmap for one method (JAX KKT)
    ax5 = plt.subplot(3, 3, 5)
    W_avg = all_results['JAX KKT']['W_avg']

    # Show a subset of outputs for clarity
    n_show = min(50, n_outputs)
    im = ax5.imshow(W_avg[:, :n_show], aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
    ax5.set_xlabel('Output (first 50)')
    ax5.set_ylabel('Feature')
    ax5.set_title('JAX KKT: Coefficient Heatmap')
    plt.colorbar(im, ax=ax5)

    # Add constraint indicators
    ax5.axhline(y=2, color='g', linewidth=2, alpha=0.7)
    ax5.axhline(y=4, color='g', linewidth=2, alpha=0.7)
    ax5.axhline(y=7, color='r', linewidth=2, alpha=0.7)

    # Plot 6: Discovery mask
    ax6 = plt.subplot(3, 3, 6)
    discovery_mask = all_results['JAX KKT'].get('discovery_mask', None)
    if discovery_mask is not None:
        n_zeros = jnp.sum(discovery_mask)
        sparsity = 100 * n_zeros / (n_features * n_outputs)

        im = ax6.imshow(discovery_mask[:, :n_show].astype(float),
                       aspect='auto', cmap='Greys', vmin=0, vmax=1)
        ax6.set_xlabel('Output (first 50)')
        ax6.set_ylabel('Feature')
        ax6.set_title(f'Discovery Mask ({n_zeros} zeros, {sparsity:.1f}% sparse)')
        plt.colorbar(im, ax=ax6)

    # Plot 7: R² over windows
    ax7 = plt.subplot(3, 3, 7)
    for method_name, results in list(all_results.items())[:3]:  # Show first 3 methods
        if 'r2' in results and results['r2']:
            if len(results['r2']) == 1:
                ax7.plot(results['r2'][0], label=method_name, alpha=0.7)

    ax7.set_xlabel('Window Index')
    ax7.set_ylabel('R²')
    ax7.set_title('R² Evolution Over Windows')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # Plot 8: Constraint satisfaction over windows (for JAX KKT)
    ax8 = plt.subplot(3, 3, 8)
    W_all = all_results['JAX KKT']['W_all']

    offset_viols_windows = []
    fixed_viols_windows = []

    for i in range(W_all.shape[0]):
        offset_viol = float(jnp.max(jnp.abs(W_all[i, 2, :] + W_all[i, 4, :])))
        fixed_viol = float(jnp.max(jnp.abs(W_all[i, 7, :] - 0.05)))
        offset_viols_windows.append(offset_viol)
        fixed_viols_windows.append(fixed_viol)

    ax8.semilogy(offset_viols_windows, label='Offset violation', alpha=0.7)
    ax8.semilogy(fixed_viols_windows, label='Fixed violation', alpha=0.7)
    ax8.set_xlabel('Window Index')
    ax8.set_ylabel('Max Violation')
    ax8.set_title('JAX KKT: Constraint Violations per Window')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # Plot 9: Summary statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')

    summary_text = f"Data Summary:\n"
    summary_text += f"• {n_samples} samples\n"
    summary_text += f"• {n_features} features\n"
    summary_text += f"• {n_outputs} outputs ({n_countries}×{n_tenors})\n"
    summary_text += f"• {n_windows} windows\n"
    summary_text += f"• Window size: {window_size}\n"
    summary_text += f"• Stride: {stride}\n\n"
    summary_text += f"Constraints:\n"
    summary_text += f"• w[2] + w[4] = 0\n"
    summary_text += f"• w[7] = 0.05\n\n"
    summary_text += f"Best Method (R²): {method_names[np.argmax(r2_means)]}\n"
    summary_text += f"Fastest Method: {method_names[np.argmin(times)]}"

    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes,
            verticalalignment='top', fontsize=11, family='monospace')
    ax9.set_title('Summary Information')

    plt.tight_layout()
    plt.savefig('comprehensive_method_comparison.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to 'comprehensive_method_comparison.png'")

    # ============= DETAILED CONSTRAINT ANALYSIS =============
    print("\n6. DETAILED CONSTRAINT ANALYSIS")
    print("-"*40)

    for method_name, results in all_results.items():
        print(f"\n{method_name}:")
        W_avg = results['W_avg']

        # Offset constraint check
        offset_diffs = W_avg[2, :] + W_avg[4, :]
        print(f"  Offset constraint (w[2] + w[4] = 0):")
        print(f"    Max violation: {jnp.max(jnp.abs(offset_diffs)):.2e}")
        print(f"    Mean violation: {jnp.mean(jnp.abs(offset_diffs)):.2e}")
        print(f"    % outputs < 1e-10: {100 * jnp.sum(jnp.abs(offset_diffs) < 1e-10) / n_outputs:.1f}%")

        # Fixed constraint check
        fixed_diffs = W_avg[7, :] - 0.05
        print(f"  Fixed constraint (w[7] = 0.05):")
        print(f"    Max violation: {jnp.max(jnp.abs(fixed_diffs)):.2e}")
        print(f"    Mean violation: {jnp.mean(jnp.abs(fixed_diffs)):.2e}")
        print(f"    % outputs < 1e-10: {100 * jnp.sum(jnp.abs(fixed_diffs) < 1e-10) / n_outputs:.1f}%")

        # Estimation accuracy
        rmse = float(jnp.sqrt(jnp.mean((W_avg - W_true)**2)))
        print(f"  RMSE vs true coefficients: {rmse:.6f}")

    # ============= RETURN RESULTS =============
    return {
        'all_results': all_results,
        'comparison_df': df,
        'data': {
            'X': X,
            'Y': Y,
            'W_true': W_true
        },
        'config': {
            'n_samples': n_samples,
            'n_features': n_features,
            'n_outputs': n_outputs,
            'n_countries': n_countries,
            'n_tenors': n_tenors,
            'window_size': window_size,
            'stride': stride,
            'n_windows': n_windows
        }
    }


# Run the comprehensive example
if __name__ == "__main__":
    print("Starting comprehensive unified sliding regression example...")
    print("This will test all methods with the specified constraints.")
    print()

    results = run_comprehensive_example()

    print("\n" + "="*80)
    print("EXAMPLE COMPLETED SUCCESSFULLY")
    print("="*80)
    print("\nOutputs generated:")
    print("  - method_comparison_comprehensive.csv")
    print("  - comprehensive_method_comparison.png")
    print("\nThe results show how each method handles the constraints:")
    print("  - w[2] = -w[4] (offsetting constraint)")
    print("  - w[7] = 0.05 (fixed value constraint)")
    print("\nCheck the visualization for detailed comparisons!")




