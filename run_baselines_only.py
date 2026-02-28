"""
Standalone Baseline Evaluation Script
Run this SEPARATELY from SAC training to generate baseline results.

Usage:
    python3 run_baselines_clean.py --config ./configs/config_multi_metric_sac_K2_W5_20260225_165710.py
    
Output:
    ./baseline_results/baselines_sac_K2_W5_20260225_165710.json
"""
import numpy as np
import json
import os
import sys
import argparse
import importlib.util
from pathlib import Path
from datetime import datetime
from network_env_multi_metric import NetworkEnvironment
from baseline_policies import create_baseline


def load_config(config_path):
    """Load configuration from Python file."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def extract_run_name(config_path):
    """Extract run name from config filename."""
    filename = Path(config_path).stem
    if filename.startswith('config_multi_metric_'):
        return filename.replace('config_multi_metric_', '')
    return filename


def evaluate_baseline(baseline_policy, env, num_episodes=100, policy_name="Policy"):
    """
    Evaluate a baseline policy over multiple episodes.
    
    Returns:
        dict with detailed metrics
    """
    print(f"\n  [{policy_name}]")
    
    episode_rewards = []
    episode_betas = []
    episode_violations = []
    episode_utilization = []
    
    for episode in range(num_episodes):
        if (episode + 1) % 10 == 0:
            print(f"    Episode {episode + 1}/{num_episodes}...", end='\r')
        
        state = env.reset()
        episode_reward = 0
        episode_betas_list = []
        episode_violations_count = 0
        episode_rb_used = []
        last_info = None
        
        for dti in range(env.max_dtis):
            # Select action
            if dti == 0:
                action = baseline_policy.select_action(state, info=None)
            else:
                action = baseline_policy.select_action(state, info=last_info)
            
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_betas_list.append(info['beta'])
            episode_rb_used.append(np.sum(info['r_used']))
            
            if info['constraint_violated']:
                episode_violations_count += 1
            
            state = next_state
            last_info = info
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_betas.append(np.mean(episode_betas_list))
        episode_violations.append(episode_violations_count)
        episode_utilization.append(np.mean(episode_rb_used) / env.C * 100)
    
    print(f"    Episode {num_episodes}/{num_episodes}... Done!")
    
    # Compute statistics (same format as SAC)
    rewards_arr = np.array(episode_rewards)
    betas_arr = np.array(episode_betas)
    
    # Last 100 episodes
    last_100_rewards = rewards_arr[-100:] if len(rewards_arr) >= 100 else rewards_arr
    last_100_betas = betas_arr[-100:] if len(betas_arr) >= 100 else betas_arr
    
    return {
        'policy_name': policy_name,
        'num_episodes': num_episodes,
        'statistics': {
            'reward': {
                'mean': float(np.mean(rewards_arr)),
                'std': float(np.std(rewards_arr)),
                'min': float(np.min(rewards_arr)),
                'max': float(np.max(rewards_arr)),
                'last_100_mean': float(np.mean(last_100_rewards)),
                'last_100_std': float(np.std(last_100_rewards))
            },
            'beta': {
                'mean': float(np.mean(betas_arr)),
                'std': float(np.std(betas_arr)),
                'min': float(np.min(betas_arr)),
                'max': float(np.max(betas_arr)),
                'last_100_mean': float(np.mean(last_100_betas)),
                'last_100_std': float(np.std(last_100_betas))
            },
            'violations': {
                'mean': float(np.mean(episode_violations)),
                'total': int(np.sum(episode_violations))
            },
            'utilization': {
                'mean': float(np.mean(episode_utilization)),
                'std': float(np.std(episode_utilization))
            }
        },
        'episode_data': {
            'rewards': [float(x) for x in episode_rewards],
            'betas': [float(x) for x in episode_betas]
        }
    }


def print_summary(all_results):
    """Print summary table"""
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    print(f"\n{'Policy':<15} {'Reward (last 100)':<25} {'Beta (last 100)':<25}")
    print("-"*65)
    
    for name, results in all_results.items():
        r_mean = results['statistics']['reward']['last_100_mean']
        r_std = results['statistics']['reward']['last_100_std']
        b_mean = results['statistics']['beta']['last_100_mean']
        b_std = results['statistics']['beta']['last_100_std']
        
        print(f"{name:<15} {r_mean:>8.2f} ± {r_std:<8.2f}   {b_mean:>8.4f} ± {b_std:<8.4f}")


def main():
    parser = argparse.ArgumentParser(description='Run baseline comparisons')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--output-dir', type=str, default='./baseline_results',
                       help='Output directory')
    parser.add_argument('--num-episodes', type=int, default=None,
                       help='Number of episodes (default: from config)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("BASELINE EVALUATION")
    print("="*80)
    
    # Load config
    print(f"\n[1/5] Loading config: {args.config}")
    if not os.path.exists(args.config):
        print(f"ERROR: Config file not found: {args.config}")
        return
    
    config_module = load_config(args.config)
    cfg = config_module.get_config()
    run_name = extract_run_name(args.config)
    
    print(f"  Run name: {run_name}")
    print(f"  K={cfg['K']}, C={cfg['C']}, Profiles={cfg['traffic_profiles']}")
    
    # Get parameters
    num_episodes = args.num_episodes if args.num_episodes else cfg.get('eval_episodes', 100)
    
    # Extract seeds
    env_seed = getattr(config_module, 'ENV_SEED', None)
    traffic_seed = getattr(config_module, 'TRAFFIC_SEED', None)
    agent_seed = getattr(config_module, 'AGENT_SEED', None)
    
    print(f"  Episodes: {num_episodes}")
    print(f"  Seeds: env={env_seed}, traffic={traffic_seed}, agent={agent_seed}")
    
    # Create environment
    print(f"\n[2/5] Creating environment...")
    
    env = NetworkEnvironment(
        K=cfg['K'],
        C=cfg['C'],
        N=cfg['N'],
        thresholds=cfg['thresholds'],
        lambda_reward=cfg['lambda_reward'],
        window_size=cfg['window_size'],
        traffic_profiles=cfg['traffic_profiles'],
        qos_table_files=cfg['qos_table_files'],
        qos_metrics=cfg.get('qos_metrics'),
        qos_metrics_multi=cfg.get('qos_metrics_multi'),
        thresholds_multi=cfg.get('thresholds_multi'),
        qos_metric_directions=cfg.get('qos_metric_directions'),
        dynamic_profile_config=cfg['dynamic_profile_config'],
        max_dtis=cfg['max_dtis'],
        traffic_seed=traffic_seed,
        profile_seed=traffic_seed,
        use_efficient_allocation=cfg.get('use_efficient_allocation', False),
        unused_capacity_reward_weight=cfg.get('unused_capacity_reward_weight', 0.0)
    )
    
    print(f"  ✓ Environment created: K={env.K} slices, C={env.C} RBs")
    
    # Set numpy seed for baselines
    if agent_seed is not None:
        np.random.seed(agent_seed)
        print(f"  ✓ NumPy seed set: {agent_seed}")
    
    # Create baselines
    print(f"\n[3/5] Creating baseline policies...")
    
    baselines = {
        'equal': create_baseline('equal', cfg=cfg),
        'proportional': create_baseline('proportional', cfg=cfg),
        'greedy': create_baseline('greedy', cfg=cfg, qos_tables=env.qos_tables),
        'random': create_baseline('random', cfg=cfg, seed=agent_seed)
    }
    
    print(f"  ✓ Created {len(baselines)} policies: {list(baselines.keys())}")
    
    # Show which metrics greedy is using
    if cfg.get('qos_metrics'):
        print(f"  ✓ Greedy using metrics: {cfg['qos_metrics']}")
    
    # Evaluate baselines
    print(f"\n[4/5] Evaluating baselines ({num_episodes} episodes each)...")
    
    all_results = {}
    
    for name, baseline in baselines.items():
        results = evaluate_baseline(baseline, env, num_episodes, name)
        all_results[name] = results
    
    # Print summary
    print_summary(all_results)
    
    # Save results
    print(f"\n[5/5] Saving results...")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    output_data = {
        'metadata': {
            'run_name': run_name,
            'config_file': args.config,
            'timestamp': datetime.now().isoformat(),
            'num_episodes': num_episodes,
            'num_slices': cfg['K'],
            'capacity': cfg['C'],
            'traffic_profiles': cfg['traffic_profiles'],
            'seeds': {
                'env_seed': env_seed,
                'traffic_seed': traffic_seed,
                'agent_seed': agent_seed
            }
        },
        'baselines': all_results
    }
    
    output_filename = f"baselines_{run_name}.json"
    output_path = os.path.join(args.output_dir, output_filename)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    file_size = os.path.getsize(output_path)
    print(f"  ✓ Saved to: {output_path}")
    print(f"  File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
    
    # Final summary
    print("\n" + "="*80)
    print("✓ BASELINE EVALUATION COMPLETE")
    print("="*80)
    
    best = min(all_results.keys(), key=lambda k: all_results[k]['statistics']['beta']['last_100_mean'])
    best_beta = all_results[best]['statistics']['beta']['last_100_mean']
    
    print(f"\n  Best baseline: {best} (β = {best_beta:.4f})")
    print(f"  Results saved to: {output_path}")
    print(f"\n  Next: Use step2_generate_figures_v2.py to compare with SAC")


if __name__ == "__main__":
    main()
