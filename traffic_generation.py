"""
Traffic Generation Module - Implements Algorithm 4 from LaTeX
Supports multiple traffic profiles for network slice simulation
"""

import numpy as np
from typing import List, Union, Optional
from scipy.stats import beta


class TrafficGenerator:
    """
    Traffic generation for network slices following Algorithm 4.
    
    Profiles:
    - uniform: Uniform distribution over T
    - low: Beta(2,5) - light traffic (5-30 UEs typical)
    - medium: Beta(2,2) - moderate traffic (25-55 UEs typical)  
    - high: Beta(5,2) - heavy traffic (50-80 UEs typical)
    - external: Load from file/array
    """
    
    def __init__(self, traffic_values: List[int] = None):
        """
        Args:
            traffic_values: Discrete traffic levels, e.g., [5, 10, 15, ..., 80]
        """
        if traffic_values is None:
            self.traffic_values = list(range(5, 85, 5))  # [5, 10, ..., 80]
        else:
            self.traffic_values = sorted(traffic_values)
        
        self.T = np.array(self.traffic_values)
        self.external_data = {}
        self.external_index = {}
    
    def generate_traffic(self, profile: str, N: int, slice_id: int = 0) -> np.ndarray:
        """
        Generate traffic vector following Algorithm 4.
        
        Args:
            profile: 'uniform', 'low', 'medium', 'high', or 'external'
            N: Number of TTIs to generate
            slice_id: Slice identifier (for external data)
            
        Returns:
            x: Traffic vector [x_1, ..., x_N] where x_i ∈ T
        """
        if profile == 'uniform':
            return self._generate_uniform(N)
        elif profile == 'low':
            return self._generate_beta(N, alpha=2, beta=5)
        elif profile == 'medium':
            return self._generate_beta(N, alpha=2, beta=2)
        elif profile == 'high':
            return self._generate_beta(N, alpha=5, beta=2)
        elif profile == 'external':
            return self._load_external(N, slice_id)
        else:
            raise ValueError(f"Unknown profile: {profile}")
    
    def _generate_uniform(self, N: int) -> np.ndarray:
        """Uniform profile: x_i ~ Uniform(T)"""
        return np.random.choice(self.T, size=N)
    
    def _generate_beta(self, N: int, alpha: float, beta_param: float) -> np.ndarray:
        """
        Beta-based profile: x_i ~ Quantize(Beta(alpha, beta), T)
        
        Args:
            N: Number of samples
            alpha, beta_param: Beta distribution parameters
        """
        # Sample from Beta(alpha, beta) -> [0, 1]
        u = beta.rvs(alpha, beta_param, size=N)
        
        # Quantize to discrete traffic values
        x = self._quantize(u)
        return x
    
    def _quantize(self, u: np.ndarray) -> np.ndarray:
        """
        Quantize continuous u ∈ [0,1] to discrete T.
        
        Algorithm 4: Quantize function
        """
        # Scale u to index range [0, |T|-1]
        indices = (u * len(self.T)).astype(int)
        
        # Clamp to valid range
        indices = np.clip(indices, 0, len(self.T) - 1)
        
        # Map to traffic values
        return self.T[indices]
    
    def load_external_data(self, slice_id: int, data: Union[str, np.ndarray]):
        """
        Load external traffic data for a slice.
        
        Args:
            slice_id: Slice identifier
            data: File path (CSV) or numpy array
        """
        if isinstance(data, str):
            # Load from CSV file
            self.external_data[slice_id] = np.loadtxt(data, delimiter=',')
        else:
            # Use provided array
            self.external_data[slice_id] = np.array(data)
        
        self.external_index[slice_id] = 0
    
    def _load_external(self, N: int, slice_id: int) -> np.ndarray:
        """
        Load N values from external source.
        
        Algorithm 4: LoadFromSource function
        """
        if slice_id not in self.external_data:
            raise ValueError(f"No external data loaded for slice {slice_id}")
        
        data = self.external_data[slice_id]
        idx = self.external_index[slice_id]
        
        # Extract N values (wrap around if needed)
        if idx + N <= len(data):
            x = data[idx:idx + N]
            self.external_index[slice_id] = idx + N
        else:
            # Wrap around
            x = np.concatenate([data[idx:], data[:N - (len(data) - idx)]])
            self.external_index[slice_id] = N - (len(data) - idx)
        
        return x
    
    def reset_external_index(self, slice_id: int):
        """Reset external data index for a slice (for new episode)"""
        if slice_id in self.external_index:
            self.external_index[slice_id] = 0
    
    def get_profile_stats(self, profile: str, num_samples: int = 10000) -> dict:
        """
        Get statistics for a traffic profile.
        
        Returns:
            dict with 'mean', 'std', 'min', 'max', 'percentiles'
        """
        samples = self.generate_traffic(profile, num_samples)
        
        return {
            'mean': np.mean(samples),
            'std': np.std(samples),
            'min': np.min(samples),
            'max': np.max(samples),
            'p25': np.percentile(samples, 25),
            'p50': np.percentile(samples, 50),
            'p75': np.percentile(samples, 75),
        }


if __name__ == "__main__":
    # Test traffic generation
    generator = TrafficGenerator()
    
    print("Traffic Generation Profiles (N=20 TTIs)")
    print("=" * 60)
    
    for profile in ['uniform', 'low', 'medium', 'high']:
        traffic = generator.generate_traffic(profile, N=20)
        stats = generator.get_profile_stats(profile)
        
        print(f"\n{profile.upper()} Profile:")
        print(f"  Sample: {traffic[:10]}...")
        print(f"  Mean: {stats['mean']:.1f} UEs")
        print(f"  Std:  {stats['std']:.1f} UEs")
        print(f"  Range: [{stats['min']}, {stats['max']}] UEs")
        print(f"  Quartiles: [{stats['p25']:.0f}, {stats['p50']:.0f}, {stats['p75']:.0f}]")
