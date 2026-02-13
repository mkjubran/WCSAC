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
    - extremely_low: Beta(1,5) - very light traffic (5-20 UEs typical)
    - low: Beta(2,5) - light traffic (5-30 UEs typical)
    - medium: Beta(2,2) - moderate traffic (25-55 UEs typical)  
    - high: Beta(5,2) - heavy traffic (50-80 UEs typical)
    - extremely_high: Beta(5,1) - very heavy traffic (60-80 UEs typical)
    - dynamic: Switches between profiles periodically
    - external: Load from file/array
    """
    
    def __init__(self, traffic_values: List[int] = None, dynamic_config: dict = None, seed: int = None):
        """
        Args:
            traffic_values: Discrete traffic levels, e.g., [5, 10, 15, ..., 80]
            dynamic_config: Configuration for dynamic profile switching
                           {'profile_set': [...], 'change_period': int}
            seed: Random seed for reproducibility
        """
        # Set up seeded RNG FIRST
        if seed is not None:
            self.rng = np.random.RandomState(seed)
            self.seed = seed
        else:
            self.rng = np.random.RandomState()
            self.seed = None
        
        if traffic_values is None:
            self.traffic_values = list(range(5, 85, 5))  # [5, 10, ..., 80]
        else:
            self.traffic_values = sorted(traffic_values)
        
        self.T = np.array(self.traffic_values)
        self.external_data = {}
        self.external_index = {}
        
        # Dynamic profile configuration
        if dynamic_config is None:
            self.dynamic_config = {
                'profile_set': ['low', 'medium', 'high'],
                'change_period': 100
            }
        else:
            self.dynamic_config = dynamic_config
            # Validate that profile_set doesn't contain 'dynamic'
            if 'profile_set' in self.dynamic_config:
                invalid = [p for p in self.dynamic_config['profile_set'] if p == 'dynamic']
                if invalid:
                    raise ValueError(
                        "Dynamic profile_set cannot contain 'dynamic' itself. "
                        "Use static profiles like 'low', 'medium', 'high', etc."
                    )
        
        # Track current profile for each slice (for dynamic)
        self.current_profiles = {}
        self.dti_counter = 0
        self.last_change_dti = 0
    
    def reset_dynamic(self):
        """Reset dynamic profile state at episode start"""
        self.current_profiles = {}
        self.dti_counter = 0
        self.last_change_dti = 0
    
    def update_dti(self):
        """Update DTI counter for dynamic profile switching"""
        self.dti_counter += 1
        if (self.dti_counter - self.last_change_dti) >= self.dynamic_config['change_period']:
            self.last_change_dti = self.dti_counter
            self.current_profiles = {}
    
    def _select_dynamic_profile(self, slice_id: int) -> str:
        """
        Select a profile for dynamic mode.
        
        Args:
            slice_id: Slice identifier
            
        Returns:
            profile: Selected profile name
        """
        if slice_id not in self.current_profiles:
            # Randomly select from profile set
            profile_set = self.dynamic_config['profile_set']
            if not profile_set:
                profile_set = ['low', 'medium', 'high']
            
            # Filter out 'dynamic' to prevent recursion
            valid_profiles = [p for p in profile_set if p != 'dynamic']
            if not valid_profiles:
                raise ValueError("Dynamic profile_set must contain at least one non-dynamic profile")
            
            selected_profile = self.rng.choice(valid_profiles)
            self.current_profiles[slice_id] = selected_profile
        
        return self.current_profiles[slice_id]
    
    def generate_traffic(self, profile: str, N: int, slice_id: int = 0) -> np.ndarray:
        """
        Generate traffic vector following Algorithm 4.
        
        Args:
            profile: 'uniform', 'extremely_low', 'low', 'medium', 'high',
                    'extremely_high', 'dynamic', or 'external'
            N: Number of TTIs to generate
            slice_id: Slice identifier (for external data and dynamic)
            
        Returns:
            x: Traffic vector [x_1, ..., x_N] where x_i ∈ T
        """
        # Handle dynamic profile
        if profile == 'dynamic':
            actual_profile = self._select_dynamic_profile(slice_id)
            return self.generate_traffic(actual_profile, N, slice_id)
        
        if profile == 'uniform':
            return self._generate_uniform(N)
        elif profile == 'extremely_low':
            return self._generate_beta(N, alpha=1, beta_param=5)
        elif profile == 'low':
            return self._generate_beta(N, alpha=2, beta_param=5)
        elif profile == 'medium':
            return self._generate_beta(N, alpha=2, beta_param=2)
        elif profile == 'high':
            return self._generate_beta(N, alpha=5, beta_param=2)
        elif profile == 'extremely_high':
            return self._generate_beta(N, alpha=5, beta_param=1)
        elif profile == 'external':
            return self._load_external(N, slice_id)
        else:
            raise ValueError(f"Unknown profile: {profile}")
    
    def get_active_profile(self, profile: str, slice_id: int = 0) -> str:
        """Get the currently active profile for a slice"""
        if profile == 'dynamic':
            return self._select_dynamic_profile(slice_id)
        else:
            return profile
    
    def _generate_uniform(self, N: int) -> np.ndarray:
        """Uniform profile: x_i ~ Uniform(T)"""
        return self.rng.choice(self.T, size=N)
    
    def _generate_beta(self, N: int, alpha: float, beta_param: float) -> np.ndarray:
        """
        Beta-based profile: x_i ~ Quantize(Beta(alpha, beta), T)
        
        Args:
            N: Number of samples
            alpha, beta_param: Beta distribution parameters
        """
        # Sample from Beta(alpha, beta) -> [0, 1] using seeded RNG
        u = self.rng.beta(alpha, beta_param, size=N)
        
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
    
    for profile in ['uniform', 'extremely_low', 'low', 'medium', 'high', 'extremely_high']:
        traffic = generator.generate_traffic(profile, N=20)
        stats = generator.get_profile_stats(profile)
        
        print(f"\n{profile.upper()} Profile:")
        print(f"  Sample: {traffic[:10]}...")
        print(f"  Mean: {stats['mean']:.1f} UEs")
        print(f"  Std:  {stats['std']:.1f} UEs")
        print(f"  Range: [{stats['min']}, {stats['max']}] UEs")
        print(f"  Quartiles: [{stats['p25']:.0f}, {stats['p50']:.0f}, {stats['p75']:.0f}]")
