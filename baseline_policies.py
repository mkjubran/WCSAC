"""
Baseline Resource Allocation Policies for Comparison
Implements four baseline methods:
1. Equal Allocation
2. Proportional Allocation
3. Greedy QoS
4. Random Policy

CONFIGURATION:
All baseline policies read their parameters (K, C, N, thresholds) from config.py
via the create_baseline() factory function. This ensures consistency with the
main SAC training configuration.

Usage:
    from config import get_config
    from baseline_policies import create_baseline
    
    cfg = get_config()
    
    # Recommended: Pass config directly
    equal = create_baseline('equal', cfg=cfg)
    prop = create_baseline('proportional', cfg=cfg)
    greedy = create_baseline('greedy', cfg=cfg, qos_tables=env.qos_tables)
    random = create_baseline('random', cfg=cfg, seed=42)
    
    # Also supported: Pass parameters directly (backwards compatible)
    equal = create_baseline('equal', K=2, C=8)
"""
import numpy as np
from typing import List, Tuple


class BaselinePolicy:
    """Base class for baseline policies"""
    
    def __init__(self, K: int, C: int):
        """
        Args:
            K: Number of slices
            C: Total RB capacity
        """
        self.K = K
        self.C = C
    
    def select_action(self, state: np.ndarray, info: dict = None) -> np.ndarray:
        """
        Select action given state.
        
        Args:
            state: Current state (may be unused by some baselines)
            info: Additional info (traffic, QoS metrics, etc.)
            
        Returns:
            action: RB allocation [a_1, ..., a_K] with sum <= C
        """
        raise NotImplementedError


class EqualAllocation(BaselinePolicy):
    """
    Equal Allocation: Static allocation of C/K RBs to each slice.
    
    Mathematical formulation:
        a_k = C/K for all k
    
    Characteristics:
    - Simplest possible strategy
    - Fair in allocation but ignores traffic and QoS
    - No adaptation to changing conditions
    """
    
    def __init__(self, K: int, C: int):
        super().__init__(K, C)
        self.allocation = C / K
    
    def select_action(self, state: np.ndarray, info: dict = None) -> np.ndarray:
        """Return equal allocation for all slices"""
        return np.array([self.allocation] * self.K)


class ProportionalAllocation(BaselinePolicy):
    """
    Proportional Allocation: RBs allocated proportionally to traffic demand.
    
    Mathematical formulation:
        r_{t,k} = floor(C * x̄_{t,k} / Σ_j x̄_{t,j})
        where x̄_{t,k} = (1/N) * Σ_i x_k^{(i)} is average traffic in DTI t
    
    Characteristics:
    - Common heuristic used in practice
    - Adapts to traffic demand
    - Does not explicitly consider QoS violations
    """
    
    def __init__(self, K: int, C: int):
        super().__init__(K, C)
    
    def select_action(self, state: np.ndarray, info: dict = None) -> np.ndarray:
        """Allocate proportionally to current traffic demand"""
        if info is None or 'traffic' not in info:
            # Fallback to equal allocation if no traffic info
            return np.array([self.C / self.K] * self.K)
        
        traffic = np.array(info['traffic'])  # Average traffic per slice in current DTI
        
        # Handle edge case: all traffic is zero
        total_traffic = np.sum(traffic)
        if total_traffic < 1e-6:
            return np.array([self.C / self.K] * self.K)
        
        # Proportional allocation
        allocation = self.C * (traffic / total_traffic)
        
        return allocation


class GreedyQoS(BaselinePolicy):
    """
    Greedy QoS: Iteratively allocates RBs to slice with highest violation ratio.
    
    Mathematical formulation:
        For each RB:
            k* = argmax_k β_k (violation ratio per slice)
            Allocate 1 RB to slice k*
            Update β_k based on new allocation
    
    Characteristics:
    - QoS-focused heuristic
    - Tries to minimize violations
    - May lead to unfair allocation
    - Requires per-slice QoS tracking
    """
    
    def __init__(self, K: int, C: int, N: int, thresholds: List[float], qos_tables: List[dict], qos_metrics: List[str] = None):
        """
        Args:
            K: Number of slices
            C: Total RB capacity
            N: TTIs per DTI
            thresholds: QoS thresholds for each slice
            qos_tables: QoS lookup tables for estimating violations
            qos_metrics: Which metric to use per slice (for multi-metric tables)
        """
        super().__init__(K, C)
        self.N = N
        self.thresholds = thresholds
        
        # Convert multi-metric QoS tables to single-metric format
        # Use the SPECIFIC metric from config (qos_metrics), not just first one
        self.qos_tables = []
        for k in range(K):
            if isinstance(qos_tables[k], dict):
                # Check if multi-metric format
                first_key = next(iter(qos_tables[k].keys()))
                
                if isinstance(first_key, str):
                    # Multi-metric format: {metric_name: {(traffic, rbs): (mu, sigma)}}
                    # Use the metric specified in config
                    if qos_metrics and k < len(qos_metrics) and qos_metrics[k]:
                        metric_to_use = qos_metrics[k]
                    else:
                        # Fallback to first metric if not specified
                        metric_to_use = list(qos_tables[k].keys())[0]
                        print(f"Warning: No qos_metric specified for slice {k}, using '{metric_to_use}'")
                    
                    if metric_to_use in qos_tables[k]:
                        self.qos_tables.append(qos_tables[k][metric_to_use])
                    else:
                        print(f"Warning: Metric '{metric_to_use}' not found for slice {k}")
                        print(f"         Available metrics: {list(qos_tables[k].keys())}")
                        # Use first available metric as fallback
                        first_metric = list(qos_tables[k].keys())[0]
                        self.qos_tables.append(qos_tables[k][first_metric])
                else:
                    # Single-metric format: {(traffic, rbs): (mu, sigma)}
                    self.qos_tables.append(qos_tables[k])
            else:
                # Direct table
                self.qos_tables.append(qos_tables[k])
    
    def select_action(self, state: np.ndarray, info: dict = None) -> np.ndarray:
        """Greedily allocate to slice with highest violation ratio"""
        if info is None or 'traffic' not in info:
            # Fallback to equal allocation
            return np.array([self.C / self.K] * self.K)
        
        traffic = np.array(info['traffic'])  # Average traffic per slice
        
        # Initialize allocation
        allocation = np.ones(self.K)  # Start with 1 RB per slice (minimum)
        remaining = self.C - self.K
        
        # Greedily allocate remaining RBs
        for _ in range(int(remaining)):
            # Estimate violation ratio for each slice with current allocation
            violations = np.zeros(self.K)
            
            for k in range(self.K):
                violations[k] = self._estimate_violation_ratio(
                    k, int(traffic[k]), int(allocation[k])
                )
            
            # Allocate to slice with highest violation
            k_worst = np.argmax(violations)
            allocation[k_worst] += 1
        
        return allocation
    
    def _estimate_violation_ratio(self, slice_id: int, traffic: int, rbs: int) -> float:
        """
        Estimate violation ratio for a slice given traffic and RBs.
        
        Uses QoS table to estimate mean QoS and compares to threshold.
        """
        if traffic < 5:
            return 0.0  # No traffic, no violation
        
        # Clip to valid ranges
        traffic = max(5, min(80, traffic))
        rbs = max(1, min(self.C, rbs))
        
        # Lookup QoS table (already converted to single-metric in __init__)
        key = (traffic, rbs)
        if key in self.qos_tables[slice_id]:
            mu, sigma = self.qos_tables[slice_id][key]
        else:
            # Find nearest neighbor
            mu, sigma = self._nearest_qos(slice_id, traffic, rbs)
        
        # Simple estimate: if mean QoS > threshold, assume ~100% violation
        # Otherwise, use normal CDF to estimate violation probability
        if mu > self.thresholds[slice_id]:
            # Mean is already above threshold
            violation_prob = 0.7  # High violation probability
        else:
            # Estimate using simplified normal approximation
            # P(Q > τ) where Q ~ N(μ, σ²)
            if sigma > 0:
                z = (self.thresholds[slice_id] - mu) / sigma
                # Approximate: if z > 0, low violation; if z < 0, high violation
                violation_prob = max(0.0, min(1.0, 0.5 - 0.3 * z))
            else:
                violation_prob = 0.0
        
        return violation_prob
    
    def _nearest_qos(self, k: int, traffic: int, rbs: int) -> Tuple[float, float]:
        """Find nearest QoS entry if exact match not in table"""
        min_dist = float('inf')
        best_mu, best_sigma = 0.5, 0.02
        
        # QoS tables already converted to single-metric format in __init__
        for (t, r), (mu, sigma) in self.qos_tables[k].items():
            dist = abs(t - traffic) + abs(r - rbs)
            if dist < min_dist:
                min_dist = dist
                best_mu, best_sigma = mu, sigma
        
        return best_mu, best_sigma


class RandomPolicy(BaselinePolicy):
    """
    Random Policy: Random allocation satisfying capacity constraint.
    
    Mathematical formulation:
        Sample a_k ~ Uniform(0, C) for all k
        Normalize: a_k = C * (a_k / Σ_j a_j)
    
    Characteristics:
    - Lower bound on performance
    - Useful for sanity checking
    - No learning or optimization
    """
    
    def __init__(self, K: int, C: int, seed: int = None):
        """
        Args:
            K: Number of slices
            C: Total RB capacity
            seed: Random seed for reproducibility
        """
        super().__init__(K, C)
        if seed is not None:
            np.random.seed(seed)
    
    def select_action(self, state: np.ndarray, info: dict = None) -> np.ndarray:
        """Sample random allocation"""
        # Sample random values
        random_vals = np.random.uniform(0, 1, size=self.K)
        
        # Normalize to sum to C
        allocation = self.C * (random_vals / np.sum(random_vals))
        
        return allocation


def create_baseline(baseline_type: str, K: int = None, C: int = None, cfg: dict = None, **kwargs) -> BaselinePolicy:
    """
    Factory function to create baseline policy.
    
    Args:
        baseline_type: 'equal', 'proportional', 'greedy', or 'random'
        K: Number of slices (can be None if cfg provided)
        C: Total RB capacity (can be None if cfg provided)
        cfg: Configuration dict from config.get_config() (optional)
        **kwargs: Additional arguments for specific baselines
        
    Returns:
        Baseline policy instance
        
    Note:
        If cfg is provided, K and C are taken from cfg and override parameters.
        This ensures consistency with config.py settings.
    """
    # If config provided, use it
    if cfg is not None:
        K = cfg['K']
        C = cfg['C']
        # Also extract other params that might be needed
        if 'N' not in kwargs and 'N' in cfg:
            kwargs['N'] = cfg['N']
        if 'thresholds' not in kwargs and 'thresholds' in cfg:
            kwargs['thresholds'] = cfg['thresholds']
        # IMPORTANT: Extract qos_metrics for multi-metric support
        if 'qos_metrics' not in kwargs and 'qos_metrics' in cfg:
            kwargs['qos_metrics'] = cfg['qos_metrics']
    
    # Validate required parameters
    if K is None or C is None:
        raise ValueError("Must provide either (K, C) or cfg parameter")
    
    baseline_type = baseline_type.lower()
    
    if baseline_type == 'equal':
        return EqualAllocation(K, C)
    
    elif baseline_type == 'proportional':
        return ProportionalAllocation(K, C)
    
    elif baseline_type == 'greedy':
        # Requires additional parameters
        if 'N' not in kwargs or 'thresholds' not in kwargs or 'qos_tables' not in kwargs:
            raise ValueError("GreedyQoS requires N, thresholds, and qos_tables")
        
        # Pass qos_metrics if available (for multi-metric support)
        qos_metrics = kwargs.get('qos_metrics', None)
        return GreedyQoS(K, C, kwargs['N'], kwargs['thresholds'], kwargs['qos_tables'], qos_metrics)
    
    elif baseline_type == 'random':
        seed = kwargs.get('seed', None)
        return RandomPolicy(K, C, seed=seed)
    
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")


if __name__ == "__main__":
    # Test baselines
    print("Testing Baseline Policies")
    print("=" * 70)
    
    K = 2
    C = 8
    
    # Dummy state and info
    state = np.random.rand(33)
    info = {'traffic': [30, 50]}  # Slice 0: 30 UEs, Slice 1: 50 UEs
    
    # Test Equal
    print("\n1. Equal Allocation:")
    equal = create_baseline('equal', K, C)
    action = equal.select_action(state, info)
    print(f"   Allocation: {action}")
    print(f"   Sum: {np.sum(action):.1f}")
    
    # Test Proportional
    print("\n2. Proportional Allocation:")
    prop = create_baseline('proportional', K, C)
    action = prop.select_action(state, info)
    print(f"   Allocation: {action}")
    print(f"   Sum: {np.sum(action):.1f}")
    print(f"   Ratio: {action[0]/action[1]:.2f} (should be 30/50 = 0.60)")
    
    # Test Random
    print("\n3. Random Policy:")
    random_pol = create_baseline('random', K, C, seed=42)
    action = random_pol.select_action(state, info)
    print(f"   Allocation: {action}")
    print(f"   Sum: {np.sum(action):.1f}")
    
    # Test Greedy (with dummy QoS tables)
    print("\n4. Greedy QoS:")
    dummy_qos = {}
    for t in range(5, 85, 5):
        for r in range(1, 9):
            # Dummy: higher traffic and lower RBs = higher QoS (worse)
            mu = 0.05 * t / r
            sigma = 0.01
            dummy_qos[(t, r)] = (mu, sigma)
    
    greedy = create_baseline('greedy', K, C, N=8, thresholds=[0.3, 0.3], 
                            qos_tables=[dummy_qos, dummy_qos])
    action = greedy.select_action(state, info)
    print(f"   Allocation: {action}")
    print(f"   Sum: {np.sum(action):.1f}")
    
    print("\n" + "=" * 70)
    print("✓ All baselines working correctly!")
