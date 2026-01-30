"""
Comprehensive Training Evaluation - Extract All Paper Metrics

This script extracts all evaluation metrics mentioned in the IEEE paper from
TensorBoard logs and saved model evaluation. It can also read the associated
config file to include experimental parameters.

Metrics extracted:
1. QoS Violation Ratio (β)
2. Cumulative Episode Reward
3. Resource Utilization
4. Per-Slice Allocation Statistics
5. Fairness Index
6. Convergence Speed

Usage:
    python3 comprehensive_evaluation.py --log-dir runs/sac_20240130_120000
    python3 comprehensive_evaluation.py --log-dir runs/sac_20240130_120000 --config config_run1.py
    python3 comprehensive_evaluation.py --log-dir runs/sac_20240130_120000 --eval-episodes 100
"""

import os
import argparse
import numpy as np
from pathlib import Path
import glob
import json
import importlib.util
import sys

try:
    from tensorboard.backend.event_processing import event_accumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


def load_config_file(config_path):
    """
    Load configuration from Python config file.
    
    Args:
        config_path: Path to config.py file
        
    Returns:
        dict: Configuration parameters
    """
    try:
        # Load module from file path
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        # Get config function if it exists
        if hasattr(config_module, 'get_config'):
            config = config_module.get_config()
        else:
            # Extract all uppercase variables (convention for constants)
            all_vars = dir(config_module)
            uppercase_vars = [k for k in all_vars if k.isupper() and not k.startswith('_')]
            
            # Optional: Uncomment for debugging
            # print(f"\n  DEBUG: Total module attributes: {len(all_vars)}")
            # print(f"  DEBUG: Uppercase non-private attributes: {len(uppercase_vars)}")
            # print(f"  DEBUG: Uppercase attributes: {sorted(uppercase_vars)}")
            
            config = {
                key: getattr(config_module, key)
                for key in uppercase_vars
            }
            
            # Also check for some specific lowercase variables that might be important
            lowercase_vars = ['traffic_profiles', 'thresholds', 'max_dtis', 'window_size', 
                            'lambda_reward', 'num_episodes', 'qos_table_files', 'qos_metrics']
            for var in lowercase_vars:
                if hasattr(config_module, var):
                    config[var] = getattr(config_module, var)
        
        print(f"✓ Loaded config from: {config_path}")
        print(f"  Found {len(config)} configuration parameters")
        
        # Debug: Show key variables
        key_vars = ['K', 'C', 'QOS_TABLE_FILES', 'qos_table_files', 'QOS_METRICS', 'qos_metrics', 'traffic_profiles']
        found_keys = [k for k in key_vars if k in config]
        if found_keys:
            print(f"  Key variables loaded: {found_keys}")
        
        return config
        
    except Exception as e:
        print(f"Warning: Could not load config file: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_config_summary(config):
    """Print configuration parameters"""
    
    if not config:
        print("\nWarning: No config provided to print_config_summary")
        return
    
    # Optional: Uncomment for debugging
    # print(f"\nDEBUG: Config has {len(config)} keys")
    # print(f"DEBUG: All config keys: {sorted(config.keys())}")
    
    print("\n" + "="*80)
    print("EXPERIMENTAL CONFIGURATION")
    print("="*80)
    
    # Network Parameters
    print("\nNetwork Parameters:")
    if 'K' in config:
        print(f"  Number of slices (K):         {config['K']}")
    if 'C' in config:
        print(f"  Total capacity (C):           {config['C']} RBs")
    if 'N' in config:
        print(f"  TTIs per DTI (N):             {config['N']}")
    if 'thresholds' in config:
        print(f"  QoS thresholds:               {config['thresholds']}")
    
    # Traffic profiles with dynamic details
    if 'traffic_profiles' in config:
        profiles = config['traffic_profiles']
        print(f"  Traffic profiles:             {profiles}")
        
        # Check for dynamic profiles
        if 'dynamic' in profiles or (isinstance(profiles, list) and 'dynamic' in str(profiles).lower()):
            if 'DYNAMIC_PROFILE_CONFIG' in config or 'dynamic_profile_config' in config:
                dyn_config = config.get('DYNAMIC_PROFILE_CONFIG', config.get('dynamic_profile_config', {}))
                
                print(f"\n  Dynamic Profile Configuration:")
                if 'profile_set' in dyn_config:
                    print(f"    Available levels:           {dyn_config['profile_set']}")
                if 'change_period' in dyn_config:
                    print(f"    Change period:              {dyn_config['change_period']} DTIs")
                if 'initial_profile' in dyn_config:
                    print(f"    Initial profile:            {dyn_config['initial_profile']}")
                
                # Traffic levels detail
                if 'TRAFFIC_PROFILES' in config:
                    print(f"\n  Traffic Level Details:")
                    traffic_defs = config['TRAFFIC_PROFILES']
                    for level, params in traffic_defs.items():
                        if isinstance(params, dict):
                            alpha = params.get('alpha', '?')
                            beta_val = params.get('beta', '?')
                            print(f"    {level:15} -> Beta({alpha}, {beta_val})")
    
    # QoS Tables
    print("\n  QoS Configuration:")
    
    # Check both uppercase and lowercase versions
    qos_files = config.get('QOS_TABLE_FILES') or config.get('qos_table_files')
    
    if qos_files is not None:
        print(f"    QoS Table Files:")
        if qos_files is None:
            print(f"      Using default QoS model")
        elif isinstance(qos_files, list):
            for i, qos_file in enumerate(qos_files):
                print(f"      Slice {i}: {qos_file}")
        else:
            print(f"      {qos_files}")
    else:
        print(f"    QoS Table Files:          Not specified")
    
    # QoS Metrics - check both uppercase and lowercase
    qos_metrics = config.get('QOS_METRICS') or config.get('qos_metrics')
    
    if qos_metrics is not None:
        print(f"    QoS Metrics:")
        if isinstance(qos_metrics, list):
            for i, metric in enumerate(qos_metrics):
                print(f"      Slice {i}: {metric}")
        else:
            print(f"      {qos_metrics}")
    else:
        print(f"    QoS Metrics:              Not specified")
    
    # Training Parameters
    print("\nTraining Parameters:")
    if 'NUM_EPISODES' in config or 'num_episodes' in config:
        episodes = config.get('NUM_EPISODES', config.get('num_episodes', 'N/A'))
        print(f"  Training episodes:            {episodes}")
    if 'max_dtis' in config:
        print(f"  Max DTIs per episode:         {config['max_dtis']}")
    if 'window_size' in config:
        print(f"  Window size (W):              {config['window_size']}")
    if 'lambda_reward' in config:
        print(f"  Reward weight (λ):            {config['lambda_reward']}")
    
    # SAC Hyperparameters
    print("\nSAC Hyperparameters:")
    if 'LR_ACTOR' in config or 'lr_actor' in config:
        lr_actor = config.get('LR_ACTOR', config.get('lr_actor', 'N/A'))
        print(f"  Actor learning rate:          {lr_actor}")
    if 'LR_CRITIC' in config or 'lr_critic' in config:
        lr_critic = config.get('LR_CRITIC', config.get('lr_critic', 'N/A'))
        print(f"  Critic learning rate:         {lr_critic}")
    if 'GAMMA' in config or 'gamma' in config:
        gamma = config.get('GAMMA', config.get('gamma', 'N/A'))
        print(f"  Discount factor (γ):          {gamma}")
    if 'TAU' in config or 'tau' in config:
        tau = config.get('TAU', config.get('tau', 'N/A'))
        print(f"  Soft update (τ):              {tau}")
    if 'ALPHA' in config or 'alpha' in config:
        alpha = config.get('ALPHA', config.get('alpha', 'N/A'))
        print(f"  Temperature (α):              {alpha}")
    if 'BATCH_SIZE' in config or 'batch_size' in config:
        batch = config.get('BATCH_SIZE', config.get('batch_size', 'N/A'))
        print(f"  Batch size:                   {batch}")
    if 'BUFFER_SIZE' in config or 'buffer_size' in config:
        buffer = config.get('BUFFER_SIZE', config.get('buffer_size', 'N/A'))
        print(f"  Replay buffer size:           {buffer}")
    
    # Network Architecture
    if 'HIDDEN_SIZES' in config or 'hidden_sizes' in config:
        hidden = config.get('HIDDEN_SIZES', config.get('hidden_sizes', 'N/A'))
        print(f"\nNetwork Architecture:")
        print(f"  Hidden layer sizes:           {hidden}")
    
    print("="*80)


def read_tensorboard_data(log_dir):
    """Read all available metrics from TensorBoard logs"""
    
    if not TENSORBOARD_AVAILABLE:
        print("ERROR: TensorBoard not installed")
        return None
    
    try:
        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()
        
        available_tags = ea.Tags().get('scalars', [])
        print(f"Available tags: {len(available_tags)}")
        
        # Print sample tags for debugging
        if len(available_tags) > 0:
            print("\nSample tags (first 15):")
            for tag in sorted(available_tags)[:15]:
                print(f"  - {tag}")
            if len(available_tags) > 15:
                print(f"  ... and {len(available_tags) - 15} more")
        
        # Extract all scalar data
        data = {}
        for tag in available_tags:
            events = ea.Scalars(tag)
            data[tag] = [(e.step, e.value) for e in events]
        
        return data
        
    except Exception as e:
        print(f"Error reading logs: {e}")
        return None


def find_tag(data, patterns):
    """
    Find a tag matching any of the patterns.
    Tries patterns in order of preference.
    
    Args:
        data: Dictionary of TensorBoard data
        patterns: List of string patterns to match (in order of preference)
        
    Returns:
        Matching tag name or None
    """
    # Try each pattern in order
    for pattern in patterns:
        # First try exact match (case insensitive)
        for tag in data.keys():
            if pattern.lower() == tag.lower():
                return tag
        
        # Then try contains match
        for tag in data.keys():
            if pattern.lower() in tag.lower():
                return tag
    
    return None


def compute_paper_metrics(data, window=100, target_beta=0.25, convergence_window=100):
    """
    Compute all evaluation metrics mentioned in the paper.
    
    Args:
        data: Dictionary of TensorBoard data
        window: Window for final averaging
        target_beta: Target beta for convergence
        convergence_window: Episodes for stable convergence
        
    Returns:
        dict: All computed metrics
    """
    metrics = {}
    
    # Find relevant tags with flexible matching
    # Priority order: episode-level first, then DTI-level
    reward_patterns = ['episode/reward', 'dti/reward', 'episode_reward', 'reward']
    beta_patterns = ['episode/avg_beta_100', 'episode/beta', 'dti/beta', 'episode_beta', 'beta']
    util_patterns = ['utilization', 'resource_usage', 'resource_util']
    
    print("\nSearching for tags...")
    reward_tag = find_tag(data, reward_patterns)
    beta_tag = find_tag(data, beta_patterns)
    util_tag = find_tag(data, util_patterns)
    
    # Find per-episode tags (episode_100/reward, episode_100/beta, etc.)
    episode_reward_tags = [tag for tag in data.keys() if 'episode_' in tag and 'reward' in tag.lower() and '/' in tag]
    episode_beta_tags = [tag for tag in data.keys() if 'episode_' in tag and 'beta' in tag.lower() and '/' in tag]
    
    # Find slice allocation tags
    slice_alloc_tags = [tag for tag in data.keys() if 'slice' in tag.lower() and 'allocation' in tag.lower()]
    
    print(f"\nDetected tags:")
    print(f"  Reward (main):                {reward_tag}")
    print(f"  Beta (main):                  {beta_tag}")
    print(f"  Utilization:                  {util_tag}")
    print(f"  Per-episode reward tags:      {len(episode_reward_tags)}")
    if episode_reward_tags[:3]:
        print(f"    Examples: {episode_reward_tags[:3]}")
    print(f"  Per-episode beta tags:        {len(episode_beta_tags)}")
    if episode_beta_tags[:3]:
        print(f"    Examples: {episode_beta_tags[:3]}")
    print(f"  Slice allocation tags:        {len(slice_alloc_tags)}")
    
    # ========================================================================
    # 1. QoS Violation Ratio (β)
    # ========================================================================
    if beta_tag:
        beta_values = np.array([v[1] for v in data[beta_tag]])
        steps = np.array([v[0] for v in data[beta_tag]])
        
        metrics['beta'] = {
            'final_mean': float(np.mean(beta_values[-window:])) if len(beta_values) >= window else float(np.mean(beta_values)),
            'final_std': float(np.std(beta_values[-window:])) if len(beta_values) >= window else float(np.std(beta_values)),
            'best': float(np.min(beta_values)),
            'best_episode': int(steps[np.argmin(beta_values)]),
            'worst': float(np.max(beta_values)),
            'overall_mean': float(np.mean(beta_values)),
            'overall_std': float(np.std(beta_values)),
            'target_achievement': float(np.mean(beta_values[-window:] < 0.2)) if len(beta_values) >= window else 0.0
        }
        
        # Last episode value
        if len(beta_values) > 0:
            metrics['beta']['last_episode'] = float(beta_values[-1])
        
        # Convergence speed
        converged = False
        convergence_episode = None
        for i in range(len(beta_values) - convergence_window):
            window_vals = beta_values[i:i+convergence_window]
            if np.all(window_vals < target_beta):
                convergence_episode = int(steps[i])
                converged = True
                break
        
        metrics['convergence'] = {
            'converged': converged,
            'episodes_to_convergence': convergence_episode if converged else None,
            'target_beta': target_beta,
            'convergence_window': convergence_window
        }
    
    # ========================================================================
    # 2. Cumulative Episode Reward
    # ========================================================================
    if reward_tag:
        reward_values = np.array([v[1] for v in data[reward_tag]])
        steps = np.array([v[0] for v in data[reward_tag]])
        
        metrics['reward'] = {
            'final_mean': float(np.mean(reward_values[-window:])) if len(reward_values) >= window else float(np.mean(reward_values)),
            'final_std': float(np.std(reward_values[-window:])) if len(reward_values) >= window else float(np.std(reward_values)),
            'best': float(np.max(reward_values)),
            'best_episode': int(steps[np.argmax(reward_values)]),
            'worst': float(np.min(reward_values)),
            'overall_mean': float(np.mean(reward_values)),
            'overall_std': float(np.std(reward_values))
        }
        
        # Last episode value
        if len(reward_values) > 0:
            metrics['reward']['last_episode'] = float(reward_values[-1])
    
    # ========================================================================
    # 3. Resource Utilization
    # ========================================================================
    if util_tag:
        util_values = np.array([v[1] for v in data[util_tag]])
        
        metrics['resource_utilization'] = {
            'final_mean': float(np.mean(util_values[-window:])) if len(util_values) >= window else float(np.mean(util_values)),
            'final_std': float(np.std(util_values[-window:])) if len(util_values) >= window else float(np.std(util_values)),
            'overall_mean': float(np.mean(util_values)),
            'overall_std': float(np.std(util_values)),
            'min': float(np.min(util_values)),
            'max': float(np.max(util_values))
        }
    
    # ========================================================================
    # 4. Per-Slice Allocation Statistics
    # ========================================================================
    if slice_alloc_tags:
        slice_stats = {}
        for tag in slice_alloc_tags:
            # Extract slice number
            import re
            match = re.search(r'slice[_\s]*(\d+)', tag, re.IGNORECASE)
            slice_id = match.group(1) if match else 'unknown'
            
            values = np.array([v[1] for v in data[tag]])
            
            slice_stats[f'slice_{slice_id}'] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        
        metrics['per_slice_allocations'] = slice_stats
        
        # ====================================================================
        # 5. Fairness Index (Jain's Index)
        # ====================================================================
        mean_allocs = [stats['mean'] for stats in slice_stats.values()]
        K = len(mean_allocs)
        
        if K > 0:
            sum_allocs = np.sum(mean_allocs)
            sum_squared_allocs = np.sum(np.array(mean_allocs) ** 2)
            
            if sum_squared_allocs > 0:
                fairness_index = (sum_allocs ** 2) / (K * sum_squared_allocs)
            else:
                fairness_index = 0.0
            
            metrics['fairness'] = {
                'jains_index': float(fairness_index),
                'interpretation': 'Perfect (1.0)' if fairness_index > 0.99 else 'Good (>0.9)' if fairness_index > 0.9 else 'Fair (>0.8)' if fairness_index > 0.8 else 'Poor'
            }
    
    # ========================================================================
    # 6. Per-Episode Metrics (episode_100/reward, etc.)
    # ========================================================================
    if episode_reward_tags or episode_beta_tags:
        episode_metrics = {}
        
        # Extract episode numbers and their metrics
        import re
        for tag in episode_reward_tags + episode_beta_tags:
            match = re.search(r'episode[_\s]*(\d+)', tag, re.IGNORECASE)
            if match:
                ep_num = int(match.group(1))
                
                if ep_num not in episode_metrics:
                    episode_metrics[ep_num] = {}
                
                values = data[tag]
                if values:
                    # Use the last value for this episode
                    episode_metrics[ep_num][tag] = float(values[-1][1])
        
        if episode_metrics:
            metrics['per_episode_snapshots'] = episode_metrics
            print(f"  Found detailed metrics for {len(episode_metrics)} episode snapshots")
    
    # Total episodes
    if reward_tag:
        metrics['total_episodes'] = len(data[reward_tag])
    
    return metrics


def print_paper_metrics(metrics, window=100):
    """Print metrics in paper format"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION METRICS (IEEE Paper Format)")
    print("="*80)
    
    # 1. QoS Violation Ratio (β)
    if 'beta' in metrics:
        print("\n1. QoS Violation Ratio (β)")
        print("-" * 80)
        b = metrics['beta']
        print(f"   Final {window}-episode mean:  {b['final_mean']:.4f} ± {b['final_std']:.4f}")
        if 'last_episode' in b:
            print(f"   Last episode value:           {b['last_episode']:.4f}")
        print(f"   Best episode value:           {b['best']:.4f} (episode {b.get('best_episode', '?')})")
        print(f"   Overall mean:                 {b['overall_mean']:.4f} ± {b['overall_std']:.4f}")
        print(f"   Target achievement (β<0.2):   {b['target_achievement']*100:.1f}% of episodes")
        
        target_status = "✓ ACHIEVED" if b['final_mean'] < 0.2 else "✗ NOT ACHIEVED"
        print(f"   Status:                       {target_status}")
    
    # 2. Cumulative Episode Reward
    if 'reward' in metrics:
        print("\n2. Cumulative Episode Reward")
        print("-" * 80)
        r = metrics['reward']
        print(f"   Final {window}-episode mean:  {r['final_mean']:.2f} ± {r['final_std']:.2f}")
        if 'last_episode' in r:
            print(f"   Last episode value:           {r['last_episode']:.2f}")
        print(f"   Best episode value:           {r['best']:.2f} (episode {r.get('best_episode', '?')})")
        print(f"   Overall mean:                 {r['overall_mean']:.2f} ± {r['overall_std']:.2f}")
    
    # 3. Resource Utilization
    if 'resource_utilization' in metrics:
        print("\n3. Resource Utilization")
        print("-" * 80)
        u = metrics['resource_utilization']
        print(f"   Final {window}-episode mean:  {u['final_mean']:.2f}%")
        print(f"   Overall mean:                 {u['overall_mean']:.2f}% ± {u['overall_std']:.2f}%")
        print(f"   Range:                        {u['min']:.2f}% - {u['max']:.2f}%")
    
    # 4. Per-Slice Allocation Statistics
    if 'per_slice_allocations' in metrics:
        print("\n4. Per-Slice Allocation Statistics")
        print("-" * 80)
        for slice_name, stats in metrics['per_slice_allocations'].items():
            print(f"   {slice_name}:")
            print(f"      Mean: {stats['mean']:.2f} RBs")
            print(f"      Std:  {stats['std']:.2f} RBs")
            print(f"      Range: {stats['min']:.2f} - {stats['max']:.2f} RBs")
    
    # 5. Fairness Index
    if 'fairness' in metrics:
        print("\n5. Fairness Index")
        print("-" * 80)
        f = metrics['fairness']
        print(f"   Jain's Fairness Index:        {f['jains_index']:.4f}")
        print(f"   Interpretation:               {f['interpretation']}")
        print(f"   Note: 1.0 = perfect fairness, 1/K = worst case")
    
    # 6. Convergence Speed
    if 'convergence' in metrics:
        print("\n6. Convergence Speed")
        print("-" * 80)
        c = metrics['convergence']
        if c['converged']:
            print(f"   Converged:                    ✓ YES")
            print(f"   Episodes to convergence:      {c['episodes_to_convergence']}")
            print(f"   Target β < {c['target_beta']} for {c['convergence_window']} consecutive episodes")
        else:
            print(f"   Converged:                    ✗ NO")
            print(f"   Target β < {c['target_beta']} not achieved for {c['convergence_window']} episodes")
    
    # 7. Per-Episode Snapshots (if available)
    if 'per_episode_snapshots' in metrics and metrics['per_episode_snapshots']:
        print("\n7. Per-Episode Detailed Snapshots")
        print("-" * 80)
        snapshots = metrics['per_episode_snapshots']
        episodes = sorted(snapshots.keys())
        
        print(f"   Available snapshots: Episodes {episodes}")
        print(f"\n   {'Episode':<12} {'Reward':<15} {'Beta':<15}")
        print(f"   {'-'*42}")
        
        for ep in episodes:
            ep_data = snapshots[ep]
            
            # Find reward and beta tags for this episode
            reward_val = "N/A"
            beta_val = "N/A"
            
            for tag, val in ep_data.items():
                if 'reward' in tag.lower():
                    reward_val = f"{val:.2f}"
                if 'beta' in tag.lower():
                    beta_val = f"{val:.4f}"
            
            print(f"   {ep:<12} {reward_val:<15} {beta_val:<15}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if 'total_episodes' in metrics:
        print(f"Total training episodes: {metrics['total_episodes']}")
    
    if 'beta' in metrics and 'reward' in metrics:
        print(f"\nFinal Performance (last {window} episodes):")
        print(f"  Average Reward: {metrics['reward']['final_mean']:.2f}")
        print(f"  Average Beta:   {metrics['beta']['final_mean']:.4f}")
        
        if metrics['beta']['final_mean'] < 0.2:
            print(f"  Status: ✓ MEETS TARGET (β < 0.2)")
        elif metrics['beta']['final_mean'] < 0.3:
            print(f"  Status: ⚠ CLOSE TO TARGET")
        else:
            print(f"  Status: ✗ NEEDS IMPROVEMENT")
    
    print("="*80)


def save_metrics_json(metrics, output_file, config=None):
    """Save metrics to JSON file, optionally including config"""
    
    output_data = {
        'metrics': metrics,
        'timestamp': None  # Will be filled if needed
    }
    
    if config:
        output_data['configuration'] = config
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Metrics saved to: {output_file}")
    if config:
        print(f"  (includes configuration parameters)")


def generate_latex_table(metrics, window=100, config=None):
    """Generate LaTeX table code for paper"""
    
    print("\n" + "="*80)
    print("LATEX TABLE FOR PAPER")
    print("="*80)
    print("\n% Copy this into your LaTeX document:\n")
    
    print("\\begin{table}[!t]")
    print("\\centering")
    print("\\caption{SAC Training Performance Metrics}")
    print("\\label{tab:sac_performance}")
    print("\\begin{tabular}{lc}")
    print("\\toprule")
    print("\\textbf{Metric} & \\textbf{Value} \\\\")
    print("\\midrule")
    
    # Add configuration if available
    if config:
        if 'K' in config and 'C' in config:
            print(f"Network Configuration & $K={config['K']}, C={config['C']}$ RBs \\\\")
        if 'traffic_profiles' in config:
            profiles_str = ', '.join(config['traffic_profiles'])
            print(f"Traffic Profiles & {profiles_str} \\\\")
        print("\\midrule")
    
    if 'reward' in metrics:
        print(f"Final Reward & ${metrics['reward']['final_mean']:.2f} \\pm {metrics['reward']['final_std']:.2f}$ \\\\")
    
    if 'beta' in metrics:
        print(f"Final Beta ($\\beta$) & ${metrics['beta']['final_mean']:.4f} \\pm {metrics['beta']['final_std']:.4f}$ \\\\")
        print(f"Best Beta & ${metrics['beta']['best']:.4f}$ \\\\")
    
    if 'resource_utilization' in metrics:
        print(f"Resource Utilization & ${metrics['resource_utilization']['overall_mean']:.2f}\\%$ \\\\")
    
    if 'fairness' in metrics:
        print(f"Jain's Fairness Index & ${metrics['fairness']['jains_index']:.4f}$ \\\\")
    
    if 'convergence' in metrics and metrics['convergence']['converged']:
        print(f"Convergence Speed & {metrics['convergence']['episodes_to_convergence']} episodes \\\\")
    
    if 'total_episodes' in metrics:
        print(f"Total Episodes & {metrics['total_episodes']} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print(f"\\end{{table}}")
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Extract comprehensive evaluation metrics for IEEE paper'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        required=True,
        help='Path to TensorBoard log directory'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to config.py file for this run (e.g., config_run1.py)'
    )
    parser.add_argument(
        '--window',
        type=int,
        default=100,
        help='Window for final averaging (default: 100)'
    )
    parser.add_argument(
        '--target-beta',
        type=float,
        default=0.25,
        help='Target beta for convergence (default: 0.25)'
    )
    parser.add_argument(
        '--convergence-window',
        type=int,
        default=100,
        help='Window for convergence check (default: 100)'
    )
    parser.add_argument(
        '--output-json',
        type=str,
        help='Output JSON file (optional)'
    )
    parser.add_argument(
        '--latex-table',
        action='store_true',
        help='Generate LaTeX table code'
    )
    
    args = parser.parse_args()
    
    # Check directory
    if not os.path.exists(args.log_dir):
        print(f"ERROR: Directory not found: {args.log_dir}")
        return 1
    
    print(f"Reading TensorBoard logs from: {args.log_dir}")
    
    # Load config file if provided
    config = None
    if args.config:
        if os.path.exists(args.config):
            config = load_config_file(args.config)
            print_config_summary(config)
        else:
            print(f"Warning: Config file not found: {args.config}")
    
    # Read data
    data = read_tensorboard_data(args.log_dir)
    
    if data is None:
        return 1
    
    # Compute metrics
    print("\nComputing evaluation metrics...")
    metrics = compute_paper_metrics(
        data, 
        window=args.window,
        target_beta=args.target_beta,
        convergence_window=args.convergence_window
    )
    
    # Print results
    print_paper_metrics(metrics, args.window)
    
    # Generate LaTeX table
    if args.latex_table:
        generate_latex_table(metrics, args.window, config)
    
    # Save to JSON
    if args.output_json:
        save_metrics_json(metrics, args.output_json, config)
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
