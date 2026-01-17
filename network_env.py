"""
Network Resource Allocation Environment - Implements Algorithm 1 from LaTeX
Multi-slice network with sliding window for beta and CDF computation
"""

import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple, Optional
from traffic_generation import TrafficGenerator
import pdb

class NetworkEnvironment:
    """
    Multi-Slice Network Resource Allocation Environment (Algorithm 1).
    
    State: s = (β, CDF_1, ..., CDF_K)
    Action: a = [a_1, ..., a_K] where Σa_k ≤ C
    Reward: R = -β + λ·(C - Σa_k)/C
    """
    
    def __init__(
        self,
        K: int = 2,
        C: int = 8,
        N: int = 20,
        thresholds: List[float] = None,
        lambda_reward: float = 0.5,
        window_size: int = None,  # W parameter, None = infinity
        traffic_profiles: List[str] = None,
        qos_tables: List[pd.DataFrame] = None,
        qos_table_files: List[str] = None,  # JSON file paths
        qos_metrics: List[str] = None,  # NEW: Which metric to use per slice
    ):
        """
        Args:
            K: Number of slices
            C: Total RB capacity
            N: TTIs per DTI
            thresholds: QoS thresholds [τ_1, ..., τ_K]
            lambda_reward: Weight for resource efficiency
            window_size: W for sliding window (None = ∞)
            traffic_profiles: ['uniform', 'low', etc.] for each slice
            qos_tables: QoS lookup tables for each slice
            qos_table_files: JSON file paths for QoS tables (if provided)
            qos_metrics: Which metric to use from each JSON file (if multi-metric)
        """
        # Parameters
        self.K = K
        self.C = C
        self.N = N
        self.lambda_reward = lambda_reward
        self.W = window_size  # Sliding window size
        
        # Thresholds
        if thresholds is None:
            self.thresholds = [0.2] * K  # Default 20% for all slices
        else:
            self.thresholds = thresholds
        
        # Traffic generation
        self.traffic_gen = TrafficGenerator()
        if traffic_profiles is None:
            self.traffic_profiles = ['uniform'] * K
        else:
            self.traffic_profiles = traffic_profiles
        
        # QoS metrics to use
        self.qos_metrics = qos_metrics
        if self.qos_metrics is None:
            self.qos_metrics = [None] * K
        
        # QoS tables (traffic, RBs) -> (μ, σ)
        self.qos_tables = qos_tables
        if self.qos_tables is None:
            if qos_table_files is not None:
                # Load from JSON files
                self._load_qos_tables_from_files(qos_table_files)
            else:
                # Use default model
                self._create_default_qos_tables()
        
        # State variables (Algorithm 1: Initialize)
        self.X = [[] for _ in range(K)]  # Cumulative traffic per slice
        self.Q = [[] for _ in range(K)]  # Cumulative QoS per slice
        self.S = [[] for _ in range(K)]  # Cumulative satisfaction per slice
        
        # Traffic values for CDF
        self.traffic_values = list(range(5, 85, 5))  # [5, 10, ..., 80]
        
        # Current DTI
        self.current_dti = 0
        self.max_dtis = 200  # T_max
    
    def _create_default_qos_tables(self):
        """Create default QoS tables: (traffic, RBs) -> (μ, σ)"""
        self.qos_tables = []
        
        for k in range(self.K):
            # Simple model: QoS degrades with traffic, improves with RBs
            # For loss-based QoS: higher is worse
            table = {}
            for traffic in range(5, 85, 5):
                for rbs in range(1, self.C + 1):
                    # Mean: increases with traffic, decreases with RBs
                    mu = max(0.01, min(0.95, (traffic / 100) * (self.C / (rbs + 1))))
                    # Std: some variability
                    sigma = 0.02
                    table[(traffic, rbs)] = (mu, sigma)
            self.qos_tables.append(table)
    
    def _load_qos_tables_from_files(self, qos_table_files: List[str]):
        """
        Load QoS tables from JSON files.
        
        Supports two formats:
        
        1. Single-metric format:
        {
          "5": {
            "1": {"mu": 0.15, "sigma": 0.02}
          }
        }
        
        2. Multi-metric format:
        {
          "5": {
            "1": {
              "metric1": {"mu": 0.15, "sigma": 0.02},
              "metric2": {"mu": 5.6, "sigma": 3.1}
            }
          }
        }
        """
        self.qos_tables = []
        
        for k in range(self.K):
            if qos_table_files[k] is None:
                # Use default for this slice
                table = {}
                for traffic in range(5, 85, 5):
                    for rbs in range(1, self.C + 1):
                        mu = max(0.01, min(0.95, (traffic / 100) * (self.C / (rbs + 1))))
                        sigma = 0.02
                        table[(traffic, rbs)] = (mu, sigma)
                self.qos_tables.append(table)
            else:
                # Load from JSON file
                try:
                    with open(qos_table_files[k], 'r') as f:
                        json_data = json.load(f)
                    
                    table = {}
                    for traffic_str, rbs_dict in json_data.items():
                        traffic = int(traffic_str)
                        for rbs_str, qos_data in rbs_dict.items():
                            rbs = int(rbs_str)
                            
                            # Check if multi-metric format
                            if isinstance(qos_data, dict) and 'mu' in qos_data:
                                # Single-metric format
                                mu = qos_data['mu']
                                sigma = qos_data['sigma']
                            else:
                                # Multi-metric format - select specific metric
                                metric_to_use = self.qos_metrics[k]
                                
                                if metric_to_use is None:
                                    # Use first available metric
                                    metric_to_use = list(qos_data.keys())[0]
                                    if k == 0:  # Only print once
                                        print(f"  No metric specified for slice {k}, using: {metric_to_use}")
                                
                                if metric_to_use not in qos_data:
                                    raise KeyError(f"Metric '{metric_to_use}' not found in QoS file for slice {k}")
                                
                                mu = qos_data[metric_to_use]['mu']
                                sigma = qos_data[metric_to_use]['sigma']
                            
                            table[(traffic, rbs)] = (mu, sigma)
                    
                    self.qos_tables.append(table)
                    
                    metric_info = f" (using metric: {self.qos_metrics[k]})" if self.qos_metrics[k] else ""
                    print(f"Loaded QoS table for slice {k} from {qos_table_files[k]}{metric_info}")
                    
                except FileNotFoundError:
                    print(f"Warning: QoS file {qos_table_files[k]} not found for slice {k}")
                    print(f"Using default QoS model for slice {k}")
                    # Fallback to default
                    table = {}
                    for traffic in range(5, 85, 5):
                        for rbs in range(1, self.C + 1):
                            mu = max(0.01, min(0.95, (traffic / 100) * (self.C / (rbs + 1))))
                            sigma = 0.02
                            table[(traffic, rbs)] = (mu, sigma)
                    self.qos_tables.append(table)
                
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    print(f"Error parsing QoS file {qos_table_files[k]}: {e}")
                    print(f"Using default QoS model for slice {k}")
                    # Fallback to default
                    table = {}
                    for traffic in range(5, 85, 5):
                        for rbs in range(1, self.C + 1):
                            mu = max(0.01, min(0.95, (traffic / 100) * (self.C / (rbs + 1))))
                            sigma = 0.02
                            table[(traffic, rbs)] = (mu, sigma)
                    self.qos_tables.append(table)
    
    def reset(self) -> np.ndarray:
        """
        Algorithm 1: Reset() function
        Clears all cumulative data for episodic learning.
        
        Returns:
            s_0: Initial state (β, CDF_1, ..., CDF_K)
        """
        # Clear cumulative data
        self.X = [[] for _ in range(self.K)]
        self.Q = [[] for _ in range(self.K)]
        self.S = [[] for _ in range(self.K)]
        
        # Reset DTI counter
        self.current_dti = 0
        
        # Reset external traffic indices if used
        for k in range(self.K):
            if self.traffic_profiles[k] == 'external':
                self.traffic_gen.reset_external_index(k)
        
        # Initial state: β=0, uniform CDFs
        beta = 0.0
        cdfs = [np.ones(len(self.traffic_values)) / len(self.traffic_values)] * self.K
        
        state = self._build_state(beta, cdfs)
        return state
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Algorithm 1: Step(a) function
        
        Args:
            action: Continuous action [a_1, ..., a_K] from actor
            
        Returns:
            next_state, reward, done, info
        """
        # Step 0: Validate action (before rounding)
        if np.sum(action) > self.C + 1e-6:  # Small tolerance
            # Constraint violated
            beta = 1.0
            cdfs = [np.ones(len(self.traffic_values))] * self.K
            state = self._build_state(beta, cdfs)
            reward = -100.0
            done = True
            info = {'constraint_violated': True}
            return state, reward, done, info


        # Step 1: Generate traffic for K slices
        for k in range(self.K):
            x_k = self.traffic_gen.generate_traffic(
                self.traffic_profiles[k], self.N, slice_id=k
            )
            self.X[k].extend(x_k)
        
        # Step 2: Compute QoS using rounded actions
        r_used = np.round(action).astype(int)
        
        for k in range(self.K):
            rbs = int(np.clip(r_used[k], 1, self.C))
            q_k = []
            
            # Get last N traffic values for this DTI
            traffic_values = self.X[k][-self.N:]
            
            for traffic in traffic_values:
                # Lookup (μ, σ) from QoS table
                key = (int(traffic), rbs)
                if key in self.qos_tables[k]:
                    mu, sigma = self.qos_tables[k][key]
                else:
                    # Nearest neighbor if exact match not found
                    mu, sigma = self._nearest_qos(k, int(traffic), rbs)
                
                # Sample from Gaussian
                qos_value = np.random.normal(mu, sigma)
                # Note: No clipping - QoS values can be any positive number
                # (e.g., delay in ms can be > 1, loss % from QoS files is raw value)
                qos_value = max(0, qos_value)  # Only ensure non-negative
                q_k.append(qos_value)
            
            self.Q[k].extend(q_k)
        
        # Step 3: Compute satisfaction vectors
        for k in range(self.K):
            for qos_val in self.Q[k][-self.N:]:
                satisfied = 0 if qos_val <= self.thresholds[k] else 1
                self.S[k].append(satisfied)
        
        # Step 4 & 5: Compute CDF and beta over sliding window
        n_start, n_end = self._get_window_range()
        cdfs = self._compute_cdfs(n_start, n_end)
        beta = self._compute_beta(n_start, n_end)
        
        # Step 6: Compute reward
        reward = -beta + self.lambda_reward * (self.C - np.sum(action)) / self.C
        
        # Step 7: Build state and check if done
        state = self._build_state(beta, cdfs)
        
        self.current_dti += 1
        done = (self.current_dti >= self.max_dtis)
        
        info = {
            'beta': beta,
            'dti': self.current_dti,
            'r_used': r_used,
            'constraint_violated': False,
        }
        
        return state, reward, done, info
    
    def _get_window_range(self) -> Tuple[int, int]:
        """
        Get window range for CDF and beta computation.
        
        Returns:
            (n_start, n_end): Indices for window
        """
        total_ttis = len(self.X[0])  # |X_k|
        
        if self.W is None:  # W = ∞
            n_start = 0
            n_end = total_ttis
        else:
            window_ttis = self.W * self.N
            n_start = max(0, total_ttis - window_ttis)
            n_end = total_ttis
        
        return n_start, n_end
    
    def _compute_cdfs(self, n_start: int, n_end: int) -> List[np.ndarray]:
        """
        Algorithm 1, Step 4: Compute CDF for each slice over window.
        
        Returns:
            List of CDF arrays, one per slice
        """
        cdfs = []
        
        for k in range(self.K):
            if len(self.X[k]) == 0:
                # No data yet, uniform CDF
                cdf = np.ones(len(self.traffic_values)) / len(self.traffic_values)
            else:
                # Get traffic in window
                traffic_window = self.X[k][n_start:n_end]
                window_size = len(traffic_window)
                
                cdf = []
                for t_val in self.traffic_values:
                    # CDF[j] = P(X ≤ T[j])
                    prob = np.sum(np.array(traffic_window) <= t_val) / window_size
                    cdf.append(prob)
                
                cdf = np.array(cdf)
            
            cdfs.append(cdf)
        
        return cdfs
    
    def _compute_beta(self, n_start: int, n_end: int) -> float:
        """
        Algorithm 1, Step 5: Compute global beta over window.
        
        Returns:
            β: Violation ratio [0, 1]
        """
        if len(self.X[0]) == 0:
            return 0.0
        
        violated_traffic = 0.0
        total_traffic = 0.0
        
        for k in range(self.K):
            # Traffic and satisfaction in window
            traffic_window = self.X[k][n_start:n_end]
            satisfaction_window = self.S[k][n_start:n_end]
            
            for i, traffic in enumerate(traffic_window):
                total_traffic += traffic
                if satisfaction_window[i] == 1:  # Violated
                    violated_traffic += traffic
        
        if total_traffic == 0:
            return 0.0
        
        beta = violated_traffic / total_traffic
        return beta
    
    def _build_state(self, beta: float, cdfs: List[np.ndarray]) -> np.ndarray:
        """
        Build state vector: s = (β, CDF_1, ..., CDF_K)
        
        Note: β ∈ [0,1], no scaling by 100
        """
        state = [beta]
        for cdf in cdfs:
            state.extend(cdf)
        return np.array(state, dtype=np.float32)
    
    def _nearest_qos(self, k: int, traffic: int, rbs: int) -> Tuple[float, float]:
        """Find nearest QoS entry if exact match not in table"""
        # Simple nearest neighbor in table
        min_dist = float('inf')
        best_mu, best_sigma = 0.5, 0.02
        
        for (t, r), (mu, sigma) in self.qos_tables[k].items():
            dist = abs(t - traffic) + abs(r - rbs)
            if dist < min_dist:
                min_dist = dist
                best_mu, best_sigma = mu, sigma
        
        return best_mu, best_sigma
    
    @property
    def state_dim(self) -> int:
        """State dimension: 1 (beta) + K * |T| (CDFs)"""
        return 1 + self.K * len(self.traffic_values)
    
    @property
    def action_dim(self) -> int:
        """Action dimension: K (one per slice)"""
        return self.K


if __name__ == "__main__":
    # Test environment
    print("Testing Network Environment (Algorithm 1)")
    print("=" * 60)
    
    # Create environment
    env = NetworkEnvironment(
        K=2,
        C=8,
        N=20,
        thresholds=[0.2, 0.15],
        lambda_reward=0.5,
        window_size=5,  # Last 5 DTIs
        traffic_profiles=['low', 'high']
    )
    
    print(f"State dimension: {env.state_dim}")
    print(f"Action dimension: {env.action_dim}")
    print(f"Sliding window: {env.W} DTIs")
    
    # Reset and test
    state = env.reset()
    print(f"\nInitial state shape: {state.shape}")
    print(f"Initial beta: {state[0]:.3f}")
    
    # Test step with valid action
    action = np.array([5.0, 3.0])  # Sums to 8
    next_state, reward, done, info = env.step(action)
    
    print(f"\nAfter DTI 1:")
    print(f"  Action: {action}")
    print(f"  RBs used: {info['r_used']}")
    print(f"  Beta: {info['beta']:.3f}")
    print(f"  Reward: {reward:.3f}")
    print(f"  Done: {done}")
    
    # Test constraint violation
    action_invalid = np.array([6.0, 4.0])  # Sums to 10 > 8
    next_state, reward, done, info = env.step(action_invalid)
    
    print(f"\nConstraint violation test:")
    print(f"  Action: {action_invalid} (sum={action_invalid.sum()})")
    print(f"  Violated: {info['constraint_violated']}")
    print(f"  Reward: {reward:.3f}")
    print(f"  Done: {done}")
