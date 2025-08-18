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
    
    # Build constraints for each output (country-tenor combination)
    # The output Y has shape (n_samples, n_countries * n_tenors) = (n_samples, 168)
    # We're focusing on one country, so we have 14 outputs (one per tenor)
    
    n_hedges = len(builder.hedges)
    n_tenors = len(builder.tenors)
    
    # For each tenor of the target country
    for tenor_idx in range(n_tenors):
        # Output index in the full Y matrix
        output_idx = target_country_idx * n_tenors + tenor_idx
        
        # Determine which hedges to use for this tenor
        if use_adjacent_only:
            # Get adjacent hedges for this tenor
            adjacent_hedges = builder.get_adjacent_hedges_for_tenor(tenor_idx)
            # Filter by allowed countries
            tenor_allowed_hedges = [h for h in adjacent_hedges if h in allowed_hedge_indices]
        else:
            # Use all allowed hedges
            tenor_allowed_hedges = list(allowed_hedge_indices)
        
        # Create constraints for hedges not allowed for this tenor
        for hedge_idx in range(n_hedges):
            # The coefficient index in W matrix would be hedge_idx
            # W has shape (n_hedges, n_outputs) = (7, 14) for single country
            
            if hedge_idx not in tenor_allowed_hedges:
                # This hedge should be fixed at 0 for this tenor
                # Using fixed_constraints: (coefficient_index, value)
                fixed_constraints.append((hedge_idx, 0.0))
    
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
    
    if fixed_constraints:
        constraints_config['fixed_constraints'] = fixed_constraints
    
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
        'n_zero_constraints': len([c for c in fixed_constraints if c[1] == 0.0]),
        'n_positive_constraints': len(positive_constraints),
        'n_negative_constraints': len(negative_constraints)
    }
    
    return constraints_config


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
    
    # Generate constraints
    constraints_config = compute_country_hedge_constraints(
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
    
    constraints = compute_country_hedge_constraints(
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
        constraints = compute_country_hedge_constraints(
            target_country=country,
            **rules
        )
        
        print(f"\n{country}:")
        print(f"  Allowed hedges: {constraints['summary']['allowed_hedges']}")
        print(f"  Zero constraints: {constraints['summary']['n_zero_constraints']}")
        print(f"  Sign constraints: {constraints['summary']['sign_constraints']}")
    
    return results


if __name__ == "__main__":
    results = example_usage()
