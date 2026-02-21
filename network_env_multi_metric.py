"""
Network Environment for Multi-Slice RAN Resource Allocation
Enhanced with Multi-Metric QoS Support

Supports both single-metric (backward compatible) and multi-metric QoS evaluation.
"""

import numpy as np
import pandas as pd
import json
from typing import List, Tuple, Dict
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
        qos_metrics: List[str] = None,  # Single metric per slice (backward compatible)
        qos_metrics_multi: List[List[str]] = None,  # NEW: Multiple metrics per slice
        thresholds_multi: List[List[float]] = None,  # NEW: Multiple thresholds per slice
        qos_metric_directions: List[List[str]] = None,  # NEW: 'lower' or 'higher' for each metric
        dynamic_profile_config: dict = None,  # Config for dynamic profiles
        max_dtis: int = 200,  # T_max: Maximum DTIs per episode
        traffic_seed: int = None,  # Seed for traffic generation
        profile_seed: int = None,  # Seed for dynamic profile selection
        use_efficient_allocation: bool = False,  # NEW: Enable efficient resource allocation
        unused_capacity_reward_weight: float = 0.0,  # NEW: Reward weight for unused capacity
        use_transport_layer: bool = False,  # NEW: Enable transport layer modeling
        transport_link_capacity: float = 50_000_000,  # NEW: Transport link capacity (bits/sec)
        slice_packet_sizes: List[int] = None,  # NEW: Packet sizes per slice (bits)
        slice_bit_rates: List[int] = None,  # NEW: Bit rates per slice (bits/sec per user)
        slice_priorities: List[int] = None,  # NEW: Priority ordering per slice
        max_transport_delay_per_slice: List[float] = None,  # NEW: Max delay caps (seconds)
        transport_delay_weights: List[float] = None,  # NEW: Reward weights for delays
        service_time_distribution: str = "deterministic",  # NEW: Service time distribution
        mg1_stability_threshold: float = 0.999,  # NEW: Stability threshold
    ):
        """
        Args:
            K: Number of slices
            C: Total RB capacity
            N: TTIs per DTI
            thresholds: QoS thresholds [τ_1, ..., τ_K]
            lambda_reward: Weight for resource efficiency
            window_size: W for sliding window (None = ∞)
            traffic_profiles: ['uniform', 'low', 'dynamic', etc.] for each slice
            qos_tables: QoS lookup tables for each slice
            qos_table_files: JSON file paths for QoS tables (if provided)
            qos_metrics: Which metric to use from each JSON file (if multi-metric)
            dynamic_profile_config: {'profile_set': [...], 'change_period': int}
            max_dtis: Maximum DTIs per episode (T_max)
        """
        # Parameters
        self.K = K
        self.C = C
        self.N = N
        self.lambda_reward = lambda_reward
        self.W = window_size  # Sliding window size
        
        # Multi-metric QoS configuration
        if qos_metrics_multi is not None and thresholds_multi is not None:
            # Multi-metric mode
            self.use_multi_metric = True
            self.qos_metrics_multi = qos_metrics_multi
            self.thresholds_multi = thresholds_multi
            self.qos_metric_directions = qos_metric_directions if qos_metric_directions else \
                [['lower'] * len(qos_metrics_multi[k]) for k in range(K)]
            
            # For backward compatibility
            self.thresholds = [thresholds_multi[k][0] for k in range(K)]
            if qos_metrics is None:
                self.qos_metrics = [qos_metrics_multi[k][0] for k in range(K)]
            else:
                self.qos_metrics = qos_metrics
        else:
            # Single-metric mode (backward compatible)
            self.use_multi_metric = False
            if thresholds is None:
                self.thresholds = [0.2] * K
            else:
                self.thresholds = thresholds
            
            if qos_metrics is None:
                self.qos_metrics = [None] * K
            else:
                self.qos_metrics = qos_metrics
            
            # Convert to multi-metric format internally
            self.qos_metrics_multi = [[self.qos_metrics[k]] for k in range(K)]
            self.thresholds_multi = [[self.thresholds[k]] for k in range(K)]
            self.qos_metric_directions = [['lower'] for k in range(K)]
        
        
        # Seeds
        self.traffic_seed = traffic_seed
        self.profile_seed = profile_seed if profile_seed is not None else traffic_seed
        
        # Efficient allocation mode
        self.use_efficient_allocation = use_efficient_allocation
        self.unused_reward_weight = unused_capacity_reward_weight
        
        # Transport layer configuration
        self.use_transport_layer = use_transport_layer
        
        if self.use_transport_layer:
            self.transport_link_capacity = transport_link_capacity
            self.slice_packet_sizes = slice_packet_sizes if slice_packet_sizes else [12_000] * K
            self.slice_bit_rates = slice_bit_rates if slice_bit_rates else [1_000_000] * K
            self.slice_priorities = slice_priorities if slice_priorities else list(range(K))
            self.max_transport_delay_per_slice = max_transport_delay_per_slice if max_transport_delay_per_slice else [1.0] * K
            self.transport_delay_weights = transport_delay_weights if transport_delay_weights else [1.0] * K
            self.service_time_distribution = service_time_distribution
            self.mg1_stability_threshold = mg1_stability_threshold
            
            # Validate transport configuration
            self._validate_transport_config()
        
        # Thresholds
        if thresholds is None:
            self.thresholds = [0.2] * K  # Default 20% for all slices
        else:
            self.thresholds = thresholds
        
        # Traffic generation with seed
        self.traffic_gen = TrafficGenerator(dynamic_config=dynamic_profile_config, seed=self.traffic_seed)
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
        self.max_dtis = max_dtis  # T_max from config
    
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
        Load QoS tables from JSON files with FULL multi-metric support.
        
        In multi-metric mode:
        - Loads ALL metrics specified in qos_metrics_multi[k]
        - Stores as dict: {metric_name: {(traffic, rbs): (mu, sigma)}}
        
        In single-metric mode (backward compatible):
        - Loads single metric
        - Stores as dict: {(traffic, rbs): (mu, sigma)}
        
        JSON Format:
        {
          "5": {                          # UE count (traffic)
            "1": {                        # RB allocation
              "metric1": {"mu": 0.15, "sigma": 0.02},
              "metric2": {"mu": 5.6, "variance": 3.1}
            }
          }
        }
        """
        self.qos_tables = []
        
        for k in range(self.K):
            if qos_table_files[k] is None:
                # Use default tables for this slice
                if self.use_multi_metric:
                    # Create default table for each metric
                    tables_dict = {}
                    for metric_name in self.qos_metrics_multi[k]:
                        table = {}
                        for traffic in range(5, 85, 5):
                            for rbs in range(1, self.C + 1):
                                mu = max(0.01, min(0.95, (traffic / 100) * (self.C / (rbs + 1))))
                                sigma = 0.02
                                table[(traffic, rbs)] = (mu, sigma)
                        tables_dict[metric_name] = table
                    self.qos_tables.append(tables_dict)
                else:
                    # Single-metric: single table
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
                    
                    if self.use_multi_metric:
                        # Multi-metric mode: Load ALL metrics for this slice
                        tables_dict = {}
                        
                        for metric_name in self.qos_metrics_multi[k]:
                            table = {}
                            metrics_found = 0
                            
                            for traffic_str, rbs_dict in json_data.items():
                                traffic = int(traffic_str)
                                for rbs_str, metrics_data in rbs_dict.items():
                                    rbs = int(rbs_str)
                                    
                                    # Check if this is multi-metric JSON format
                                    if metric_name in metrics_data:
                                        # Found the metric
                                        metric_data = metrics_data[metric_name]
                                        mu = metric_data['mu']
                                        
                                        # Handle both 'sigma' and 'variance'
                                        if 'sigma' in metric_data:
                                            sigma = metric_data['sigma']
                                        elif 'variance' in metric_data:
                                            import math
                                            sigma = math.sqrt(metric_data['variance'])
                                        else:
                                            sigma = 0.02  # Default
                                        
                                        table[(traffic, rbs)] = (mu, sigma)
                                        metrics_found += 1
                                    elif 'mu' in metrics_data and len(self.qos_metrics_multi[k]) == 1:
                                        # Single-metric JSON format with only one metric needed
                                        mu = metrics_data['mu']
                                        sigma = metrics_data.get('sigma', 0.02)
                                        table[(traffic, rbs)] = (mu, sigma)
                                        metrics_found += 1
                            
                            if metrics_found == 0:
                                print(f"Warning: Metric '{metric_name}' not found in {qos_table_files[k]}")
                                print(f"         Using default QoS model for this metric")
                                # Create default for this metric
                                for traffic in range(5, 85, 5):
                                    for rbs in range(1, self.C + 1):
                                        mu = max(0.01, min(0.95, (traffic / 100) * (self.C / (rbs + 1))))
                                        sigma = 0.02
                                        table[(traffic, rbs)] = (mu, sigma)
                            
                            tables_dict[metric_name] = table
                        
                        self.qos_tables.append(tables_dict)
                        print(f"✓ Loaded {len(tables_dict)} metrics for slice {k}: {list(tables_dict.keys())}")
                    
                    else:
                        # Single-metric mode (backward compatible)
                        table = {}
                        metric_to_use = self.qos_metrics[k]
                        
                        for traffic_str, rbs_dict in json_data.items():
                            traffic = int(traffic_str)
                            for rbs_str, qos_data in rbs_dict.items():
                                rbs = int(rbs_str)
                                
                                # Check format
                                if isinstance(qos_data, dict) and 'mu' in qos_data:
                                    # Single-metric JSON format
                                    mu = qos_data['mu']
                                    sigma = qos_data.get('sigma', 0.02)
                                elif metric_to_use and metric_to_use in qos_data:
                                    # Multi-metric JSON, extract one metric
                                    metric_data = qos_data[metric_to_use]
                                    mu = metric_data['mu']
                                    sigma = metric_data.get('sigma', 0.02)
                                else:
                                    # Unknown format, use default
                                    continue
                                
                                table[(traffic, rbs)] = (mu, sigma)
                        
                        self.qos_tables.append(table)
                        print(f"✓ Loaded single metric for slice {k}: {metric_to_use}")
                
                except FileNotFoundError:
                    print(f"Warning: QoS file {qos_table_files[k]} not found for slice {k}")
                    print(f"Using default QoS model for slice {k}")
                    # Fallback to default
                    if self.use_multi_metric:
                        tables_dict = {}
                        for metric_name in self.qos_metrics_multi[k]:
                            table = {}
                            for traffic in range(5, 85, 5):
                                for rbs in range(1, self.C + 1):
                                    mu = max(0.01, min(0.95, (traffic / 100) * (self.C / (rbs + 1))))
                                    sigma = 0.02
                                    table[(traffic, rbs)] = (mu, sigma)
                            tables_dict[metric_name] = table
                        self.qos_tables.append(tables_dict)
                    else:
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
                    # Fallback to default (same as FileNotFoundError)
                    if self.use_multi_metric:
                        tables_dict = {}
                        for metric_name in self.qos_metrics_multi[k]:
                            table = {}
                            for traffic in range(5, 85, 5):
                                for rbs in range(1, self.C + 1):
                                    mu = max(0.01, min(0.95, (traffic / 100) * (self.C / (rbs + 1))))
                                    sigma = 0.02
                                    table[(traffic, rbs)] = (mu, sigma)
                            tables_dict[metric_name] = table
                        self.qos_tables.append(tables_dict)
                    else:
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
            s_0: Initial state (β, CDF_1, ..., CDF_K, [ρ, W_vector])
        """
        # Clear cumulative data
        self.X = [[] for _ in range(self.K)]
        self.Q = [[] for _ in range(self.K)]
        self.S = [[] for _ in range(self.K)]
        
        # Reset DTI counter
        self.current_dti = 0
        
        # Reset dynamic profile state
        self.traffic_gen.reset_dynamic()
        
        # Reset external traffic indices if used
        for k in range(self.K):
            if self.traffic_profiles[k] == 'external':
                self.traffic_gen.reset_external_index(k)
        
        # Initial state: β=0, uniform CDFs
        beta = 0.0
        cdfs = [np.ones(len(self.traffic_values)) / len(self.traffic_values)] * self.K
        
        # Initial transport metrics (if enabled)
        if self.use_transport_layer:
            # At reset, no traffic yet, so zero utilization and minimal delays
            transport_utilization = 0.0
            transport_delays = np.array([0.001] * self.K, dtype=np.float32)  # Small baseline delay
        else:
            transport_utilization = None
            transport_delays = None
        
        state = self._build_state(beta, cdfs, transport_utilization, transport_delays)
        return state
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Algorithm 1: Step(a) function
        
        Args:
            action: Continuous action [a_1, ..., a_K] from actor
            
        Returns:
            next_state, reward, done, info
        """
        # Update DTI counter for dynamic profile switching
        self.traffic_gen.update_dti()
        
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
        
        # Step 2: Compute QoS using rounded actions (MULTI-METRIC SUPPORT)
        r_used = np.round(action).astype(int)
        
        for k in range(self.K):
            rbs = int(np.clip(r_used[k], 1, self.C))
            q_k_samples = []  # List of dicts, one per TTI
            
            # Get last N traffic values for this DTI
            traffic_values = self.X[k][-self.N:]
            
            for traffic in traffic_values:
                qos_sample_dict = {}  # Store all metrics for this TTI
                
                # Sample each metric
                for metric_name in self.qos_metrics_multi[k]:
                    # Lookup (μ, σ) from QoS table
                    key = (int(traffic), rbs)
                    
                    if isinstance(self.qos_tables[k], dict):
                        # Multi-metric mode
                        if metric_name in self.qos_tables[k] and key in self.qos_tables[k][metric_name]:
                            mu, sigma = self.qos_tables[k][metric_name][key]
                        else:
                            mu, sigma = self._nearest_qos_multi(k, metric_name, int(traffic), rbs)
                    else:
                        # Backward compatible
                        if key in self.qos_tables[k]:
                            mu, sigma = self.qos_tables[k][key]
                        else:
                            mu, sigma = self._nearest_qos(k, int(traffic), rbs)
                    
                    # Sample from Gaussian
                    qos_value = np.random.normal(mu, sigma)
                    qos_value = max(0, qos_value)
                    qos_sample_dict[metric_name] = qos_value
                
                q_k_samples.append(qos_sample_dict)
            
            self.Q[k].extend(q_k_samples)
        
        # Step 3: Compute satisfaction vectors (MULTI-METRIC SUPPORT)
        for k in range(self.K):
            # Get last N QoS samples for this slice
            recent_qos_samples = self.Q[k][-self.N:]
            
            for qos_dict in recent_qos_samples:
                # Check ALL metrics for this TTI
                all_metrics_satisfied = True
                
                for metric_idx, metric_name in enumerate(self.qos_metrics_multi[k]):
                    qos_val = qos_dict.get(metric_name, 0)
                    threshold = self.thresholds_multi[k][metric_idx]
                    direction = self.qos_metric_directions[k][metric_idx]
                    
                    # Check based on direction
                    if direction == 'lower':
                        metric_satisfied = (qos_val <= threshold)
                    else:  # 'higher'
                        metric_satisfied = (qos_val >= threshold)
                    
                    if not metric_satisfied:
                        all_metrics_satisfied = False
                        break
                
                # 0 = satisfied, 1 = violated
                satisfied = 0 if all_metrics_satisfied else 1
                self.S[k].append(satisfied)
        
        # Step 4 & 5: Compute CDF and beta over sliding window
        n_start, n_end = self._get_window_range()
        cdfs = self._compute_cdfs(n_start, n_end)
        beta = self._compute_beta(n_start, n_end)
        
        # Step 5.5: Compute transport layer metrics (if enabled)
        if self.use_transport_layer:
            # Compute success rates per slice (binary: 1.0 if all metrics satisfied, 0.0 otherwise)
            success_rates = []
            for k in range(self.K):
                # Get recent samples for this slice
                recent_qos_samples = self.Q[k][-self.N:]
                
                if len(recent_qos_samples) == 0:
                    success_rates.append(0.0)
                    continue
                
                # Check last TTI (most recent)
                last_qos = recent_qos_samples[-1]
                all_metrics_satisfied = True
                
                for metric_idx, metric_name in enumerate(self.qos_metrics_multi[k]):
                    qos_val = last_qos.get(metric_name, 0)
                    threshold = self.thresholds_multi[k][metric_idx]
                    direction = self.qos_metric_directions[k][metric_idx]
                    
                    if direction == 'lower':
                        metric_satisfied = (qos_val <= threshold)
                    else:
                        metric_satisfied = (qos_val >= threshold)
                    
                    if not metric_satisfied:
                        all_metrics_satisfied = False
                        break
                
                # Binary success rate
                success_rates.append(1.0 if all_metrics_satisfied else 0.0)
            
            # Get current traffic per slice
            current_traffic = [int(self.X[k][-1]) if len(self.X[k]) > 0 else 0 for k in range(self.K)]
            
            # Compute transport metrics
            transport_utilization, transport_delays = self._compute_transport_metrics(
                current_traffic, success_rates
            )
        else:
            transport_utilization = None
            transport_delays = None
            success_rates = [0.0] * self.K
        
        # Step 6: Compute reward
        used_capacity = np.sum(action)
        unused_capacity = self.C - used_capacity
        
        # Base reward: -beta + lambda * (unused / C)
        reward = -beta #+ self.lambda_reward * (unused_capacity / self.C)
        
        # In efficient allocation mode, add bonus for saving resources
        if self.use_efficient_allocation: #and self.unused_reward_weight > 0:
            #pdb.set_trace()
            efficiency_bonus = self.unused_reward_weight * (unused_capacity / self.C)
            reward += efficiency_bonus
            #print(f"beta={beta}, reward={reward}, unused_capacity={unused_capacity}, efficiency_bonus={efficiency_bonus}")

        # Transport layer penalty (if enabled)
        if self.use_transport_layer:
            # Weighted delay penalty
            transport_penalty = sum(
                self.transport_delay_weights[k] * transport_delays[k]
                for k in range(self.K)
            )
            reward -= transport_penalty
       
        # Step 7: Build state and check if done
        state = self._build_state(beta, cdfs, transport_utilization, transport_delays)
        
        self.current_dti += 1
        done = (self.current_dti >= self.max_dtis)
        
        # Collect current DTI traffic for each slice (average over N TTIs)
        traffic_per_slice = []
        active_profiles = []
        for k in range(self.K):
            current_traffic = self.X[k][-self.N:]  # Last N TTIs
            avg_traffic = np.mean(current_traffic)
            traffic_per_slice.append(avg_traffic)
            
            # Get active profile (for logging dynamic profiles)
            active_profile = self.traffic_gen.get_active_profile(
                self.traffic_profiles[k], slice_id=k
            )
            active_profiles.append(active_profile)
        
        info = {
            'beta': beta,
            'dti': self.current_dti,
            'r_used': r_used,
            'constraint_violated': False,
            'traffic': traffic_per_slice,
            'active_profiles': active_profiles,
            'used_capacity': used_capacity,
            'unused_capacity': unused_capacity,
            'capacity_utilization': used_capacity / self.C,
        }
        
        # Add transport metrics to info if enabled
        if self.use_transport_layer:
            info.update({
                'transport_utilization': transport_utilization,
                'transport_delays': transport_delays.tolist(),  # Convert numpy to list for JSON
                'transport_stable': transport_utilization < self.mg1_stability_threshold,
                'success_rate_per_slice': success_rates,
                'transport_penalty': transport_penalty if self.use_transport_layer else 0.0,
            })
        
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
    
    def _compute_transport_metrics(self, traffic: List[int], success_rates: List[float]) -> Tuple[float, np.ndarray]:
        """
        Compute M/G/1 priority queueing transport layer metrics.
        
        Args:
            traffic: Array of traffic per slice [traffic_0, ..., traffic_{K-1}]
            success_rates: Array of RAN success rates [sr_0, ..., sr_{K-1}] (binary: 0 or 1)
        
        Returns:
            rho_total: Total transport utilization (scalar)
            delays: Transport delays as numpy array of shape (K,)
        """
        K = len(traffic)
        
        # === Step 1: Compute loads and arrival rates ===
        loads = []
        arrival_rates = []
        service_rates = []
        utilizations = []
        
        for k in range(K):
            # Successful traffic (users that passed RAN QoS check)
            successful_traffic_k = traffic[k] * success_rates[k]
            
            # Load (bits/sec)
            load_k = successful_traffic_k * self.slice_bit_rates[k]
            loads.append(load_k)
            
            # Arrival rate (packets/sec)
            lambda_k = load_k / self.slice_packet_sizes[k]
            arrival_rates.append(lambda_k)
            
            # Service rate (packets/sec)
            mu_k = self.transport_link_capacity / self.slice_packet_sizes[k]
            service_rates.append(mu_k)
            
            # Utilization
            rho_k = lambda_k / mu_k if mu_k > 0 else 0.0
            utilizations.append(rho_k)
        
        # Total utilization
        rho_total = sum(utilizations)
        
        # === Step 2: Check stability ===
        if rho_total >= self.mg1_stability_threshold:
            # System unstable - cap all delays
            delays = np.array(self.max_transport_delay_per_slice, dtype=np.float32)
            return rho_total, delays
        
        # === Step 3: Compute E[R] (residual service time) ===
        E_R = 0.0
        for k in range(K):
            E_S_k = 1.0 / service_rates[k] if service_rates[k] > 0 else 0.0
            
            if self.service_time_distribution == "deterministic":
                E_S_k_squared = E_S_k ** 2
            elif self.service_time_distribution == "exponential":
                E_S_k_squared = 2.0 * (E_S_k ** 2)
            else:
                E_S_k_squared = E_S_k ** 2  # Default to deterministic
            
            E_R += arrival_rates[k] * E_S_k_squared
        
        E_R = E_R / 2.0
        
        # === Step 4: Sort slices by priority ===
        # Create list of (priority, slice_index) pairs
        priority_order = sorted(
            [(self.slice_priorities[k], k) for k in range(K)],
            key=lambda x: x[0]  # Sort by priority value (lower = higher priority)
        )
        
        # Extract sorted slice indices
        sorted_indices = [idx for _, idx in priority_order]
        
        # === Step 5: Compute cumulative utilizations in priority order ===
        cumulative_rho = [0.0] * (K + 1)
        for i, k in enumerate(sorted_indices):
            cumulative_rho[i + 1] = cumulative_rho[i] + utilizations[k]
        
        # === Step 6: Compute delays in priority order ===
        delays_list = [0.0] * K
        
        for i, k in enumerate(sorted_indices):
            # Denominator for this priority level
            denominator = (1.0 - cumulative_rho[i]) * (1.0 - cumulative_rho[i + 1])
            
            if denominator > 0.001:  # Numerical stability threshold
                # Queue delay
                W_q_k = E_R / denominator
                
                # Service time
                E_S_k = 1.0 / service_rates[k] if service_rates[k] > 0 else 0.0
                
                # Total delay
                W_k = W_q_k + E_S_k
                
                # Cap at maximum
                W_k = min(W_k, self.max_transport_delay_per_slice[k])
            else:
                # Near-unstable for this class
                W_k = self.max_transport_delay_per_slice[k]
            
            delays_list[k] = W_k
        
        # Convert to numpy array
        delays = np.array(delays_list, dtype=np.float32)
        
        return rho_total, delays
    
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
    
    def _build_state(self, beta: float, cdfs: List[np.ndarray], 
                     transport_utilization: float = None, 
                     transport_delays: np.ndarray = None) -> np.ndarray:
        """
        Build state vector: s = (β, CDF_1, ..., CDF_K, [ρ_total, W_0, ..., W_{K-1}])
        
        Transport metrics are optional (only if transport layer enabled).
        
        Args:
            beta: RAN QoS violation ratio
            cdfs: List of K CDF arrays, each of length |T|
            transport_utilization: Total transport utilization (optional)
            transport_delays: Numpy array of K delays (optional)
        
        Returns:
            state: Numpy array
        """
        state = [beta]
        
        # Add all CDFs
        for cdf in cdfs:
            state.extend(cdf)
        
        # Add transport metrics if enabled
        if self.use_transport_layer and transport_utilization is not None:
            state.append(transport_utilization)
            state.extend(transport_delays)  # Add delay vector
        
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
    
    def _nearest_qos_multi(self, k: int, metric_name: str, traffic: int, rbs: int) -> Tuple[float, float]:
        """Find nearest QoS entry for a specific metric"""
        min_dist = float('inf')
        best_mu, best_sigma = 0.5, 0.02
        
        if metric_name in self.qos_tables[k]:
            for (t, r), (mu, sigma) in self.qos_tables[k][metric_name].items():
                dist = abs(t - traffic) + abs(r - rbs)
                if dist < min_dist:
                    min_dist = dist
                    best_mu, best_sigma = mu, sigma
        
        return best_mu, best_sigma
    
    @property
    def state_dim(self) -> int:
        """State dimension: 1 (beta) + K * |T| (CDFs) + transport metrics"""
        base_dim = 1 + self.K * len(self.traffic_values)
        if self.use_transport_layer:
            # Add: 1 (rho_total) + K (delays vector)
            return base_dim + 1 + self.K
        return base_dim
    
    @property
    def action_dim(self) -> int:
        """Action dimension: K (one per slice)"""
        return self.K
    
    def _validate_transport_config(self):
        """Validate transport layer configuration arrays"""
        arrays_to_check = {
            'slice_packet_sizes': self.slice_packet_sizes,
            'slice_bit_rates': self.slice_bit_rates,
            'slice_priorities': self.slice_priorities,
            'max_transport_delay_per_slice': self.max_transport_delay_per_slice,
            'transport_delay_weights': self.transport_delay_weights,
        }
        
        for name, array in arrays_to_check.items():
            if len(array) != self.K:
                raise ValueError(
                    f"Transport config error: {name} has length {len(array)} "
                    f"but K={self.K}. All transport parameters must have K entries."
                )
        
        # Check positive values
        if any(p <= 0 for p in self.slice_packet_sizes):
            raise ValueError("slice_packet_sizes must be positive")
        
        if any(b <= 0 for b in self.slice_bit_rates):
            raise ValueError("slice_bit_rates must be positive")
        
        if any(d <= 0 for d in self.max_transport_delay_per_slice):
            raise ValueError("max_transport_delay_per_slice must be positive")
        
        if any(w < 0 for w in self.transport_delay_weights):
            raise ValueError("transport_delay_weights must be non-negative")
        
        print(f"✓ Transport layer configuration validated for K={self.K} slices")


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
