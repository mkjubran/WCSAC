"""
Standalone Baseline Evaluation Script

Run this SEPARATELY from SAC training to generate baseline results.
Fair comparison guaranteed as long as config.py matches SAC training config.

Usage:
    python3 run_baselines_only.py
    
Output:
    - Baseline performance metrics
    - Results saved to results/baseline_results.json
    - Comparison plots
"""

import numpy as np
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from network_env import NetworkEnvironment
from baseline_policies import create_baseline
import config


def evaluate_baseline(baseline_policy, env, num_episodes=100, policy_name="Policy"):
    """
    Evaluate a baseline policy over multiple episodes.
    
    Args:
        baseline_policy: Baseline policy instance
        env: Environment instance
        num_episodes: Number of episodes to evaluate
        policy_name: Name for logging
        
    Returns:
        dict with detailed metrics
    """
    print(f"\n  Evaluating {policy_name} over {num_episodes} episodes...")
    
    episode_rewards = []
    episode_betas = []
    episode_violations = []
    episode_utilization = []
    slice_allocations = [[] for _ in range(env.K)]
    
    for episode in range(num_episodes):
        if episode % 20 == 0:
            print(f"    Episode {episode}/{num_episodes}...", end='\r')
        
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
            
            # Track allocations
            for k in range(env.K):
                slice_allocations[k].append(info['r_used'][k])
            
            state = next_state
            last_info = info
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_betas.append(np.mean(episode_betas_list))
        episode_violations.append(episode_violations_count)
        episode_utilization.append(np.mean(episode_rb_used) / env.C * 100)
    
    print(f"    Episode {num_episodes}/{num_episodes}... Done!")
    
    return {
        'policy_name': policy_name,
        'num_episodes': num_episodes,
        'rewards': episode_rewards,
        'betas': episode_betas,
        'violations': episode_violations,
        'utilization': episode_utilization,
        'slice_allocations': slice_allocations,
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'mean_beta': float(np.mean(episode_betas)),
        'std_beta': float(np.std(episode_betas)),
        'mean_violations': float(np.mean(episode_violations)),
        'mean_utilization': float(np.mean(episode_utilization)),
    }


def save_results(all_results, cfg, save_dir):
    """Save baseline results to JSON file"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare data for JSON (remove non-serializable parts)
    json_results = {}
    for name, results in all_results.items():
        json_results[name] = {
            'policy_name': results['policy_name'],
            'num_episodes': results['num_episodes'],
            'mean_reward': results['mean_reward'],
            'std_reward': results['std_reward'],
            'mean_beta': results['mean_beta'],
            'std_beta': results['std_beta'],
            'mean_violations': results['mean_violations'],
            'mean_utilization': results['mean_utilization'],
        }
    
    # Add config info
    output = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'K': cfg['K'],
            'C': cfg['C'],
            'N': cfg['N'],
            'thresholds': cfg['thresholds'],
            'traffic_profiles': cfg['traffic_profiles'],
            'window_size': cfg['window_size'],
            'max_dtis': cfg['max_dtis'],
            'lambda_reward': cfg['lambda_reward'],
        },
        'results': json_results
    }
    
    # Save to JSON
    json_path = os.path.join(save_dir, 'baseline_results.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to: {json_path}")
    
    return json_path


def plot_baseline_comparison(all_results, cfg, save_dir):
    """Create comparison plots"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    methods = list(all_results.keys())
    n_methods = len(methods)
    colors = plt.cm.Set3(range(n_methods))
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Baseline Comparison (K={cfg["K"]}, C={cfg["C"]}, Profiles={cfg["traffic_profiles"]})', 
                 fontsize=14, fontweight='bold')
    
    # 1. Rewards
    rewards_mean = [all_results[m]['mean_reward'] for m in methods]
    rewards_std = [all_results[m]['std_reward'] for m in methods]
    axes[0, 0].bar(methods, rewards_mean, yerr=rewards_std, capsize=5,
                   color=colors, edgecolor='black', alpha=0.7)
    axes[0, 0].set_ylabel('Average Episode Reward', fontsize=11)
    axes[0, 0].set_title('Total Reward Performance', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # 2. Beta (QoS)
    betas_mean = [all_results[m]['mean_beta'] for m in methods]
    betas_std = [all_results[m]['std_beta'] for m in methods]
    bar_colors = ['green' if b < 0.2 else 'orange' if b < 0.3 else 'red' 
                  for b in betas_mean]
    axes[0, 1].bar(methods, betas_mean, yerr=betas_std, capsize=5,
                   color=bar_colors, edgecolor='black', alpha=0.7)
    axes[0, 1].axhline(y=0.2, color='r', linestyle='--', linewidth=2, label='Target (0.2)')
    axes[0, 1].set_ylabel('Beta (QoS Violation Ratio)', fontsize=11)
    axes[0, 1].set_title('QoS Performance', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Violations
    viols_mean = [all_results[m]['mean_violations'] for m in methods]
    axes[1, 0].bar(methods, viols_mean, color=colors, edgecolor='black', alpha=0.7)
    axes[1, 0].set_ylabel('Constraint Violations per Episode', fontsize=11)
    axes[1, 0].set_title('Capacity Constraint Violations', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Resource Utilization
    util_mean = [all_results[m]['mean_utilization'] for m in methods]
    axes[1, 1].bar(methods, util_mean, color=colors, edgecolor='black', alpha=0.7)
    axes[1, 1].axhline(y=100, color='r', linestyle='--', linewidth=2, label='Full capacity')
    axes[1, 1].set_ylabel('Resource Utilization (%)', fontsize=11)
    axes[1, 1].set_title('RB Usage Efficiency', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].set_ylim([0, 105])
    
    plt.tight_layout()
    
    # Save
    plot_path = os.path.join(save_dir, 'baseline_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Comparison plot saved to: {plot_path}")
    plt.close()
    
    # Distribution plots
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle('Performance Distributions', fontsize=14, fontweight='bold')
    
    # Reward distributions
    reward_data = [all_results[m]['rewards'] for m in methods]
    bp1 = axes2[0].boxplot(reward_data, labels=methods, patch_artist=True)
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
    axes2[0].set_ylabel('Total Reward', fontsize=11)
    axes2[0].set_title('Reward Distribution', fontsize=12, fontweight='bold')
    axes2[0].grid(True, alpha=0.3, axis='y')
    
    # Beta distributions
    beta_data = [all_results[m]['betas'] for m in methods]
    bp2 = axes2[1].boxplot(beta_data, labels=methods, patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
    axes2[1].axhline(y=0.2, color='r', linestyle='--', linewidth=2, label='Target')
    axes2[1].set_ylabel('Beta (Violation Ratio)', fontsize=11)
    axes2[1].set_title('QoS Distribution', fontsize=12, fontweight='bold')
    axes2[1].legend()
    axes2[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    dist_path = os.path.join(save_dir, 'baseline_distributions.png')
    plt.savefig(dist_path, dpi=150, bbox_inches='tight')
    print(f"✓ Distribution plot saved to: {dist_path}")
    plt.close()


def print_summary_table(all_results):
    """Print formatted summary table"""
    
    print("\n" + "="*80)
    print("BASELINE PERFORMANCE SUMMARY")
    print("="*80)
    
    # Header
    print(f"\n{'Method':<15} {'Reward':<20} {'Beta':<20} {'Violations':<12} {'Utilization':<12}")
    print("-"*80)
    
    # Sort by beta (best first)
    sorted_methods = sorted(all_results.keys(), key=lambda m: all_results[m]['mean_beta'])
    
    for method in sorted_methods:
        r = all_results[method]
        reward_str = f"{r['mean_reward']:>7.2f} ± {r['std_reward']:<6.2f}"
        beta_str = f"{r['mean_beta']:>6.4f} ± {r['std_beta']:<6.4f}"
        viol_str = f"{r['mean_violations']:>6.2f}"
        util_str = f"{r['mean_utilization']:>6.2f}%"
        
        # Highlight best
        if method == sorted_methods[0]:
            print(f"**{method:<13} {reward_str:<20} {beta_str:<20} {viol_str:<12} {util_str:<12}**")
        else:
            print(f"{method:<15} {reward_str:<20} {beta_str:<20} {viol_str:<12} {util_str:<12}")
    
    print("\n** = Best performance (lowest beta)")
    print("="*80)


def main():
    """Main function to run baseline evaluation"""
    
    print("="*80)
    print("STANDALONE BASELINE EVALUATION")
    print("="*80)
    
    # Load config
    cfg = config.get_config()
    
    print("\nConfiguration:")
    print(f"  K (slices):          {cfg['K']}")
    print(f"  C (capacity):        {cfg['C']} RBs")
    print(f"  N (TTIs/DTI):        {cfg['N']}")
    print(f"  Thresholds:          {cfg['thresholds']}")
    print(f"  Traffic profiles:    {cfg['traffic_profiles']}")
    print(f"  Window size (W):     {cfg['window_size']}")
    print(f"  Max DTIs/episode:    {cfg['max_dtis']}")
    print(f"  Lambda (reward):     {cfg['lambda_reward']}")
    print(f"  Evaluation episodes: {cfg['eval_episodes']}")
    
    # Create environment
    print("\n" + "-"*80)
    print("Creating environment...")
    print("-"*80)
    
    env = NetworkEnvironment(
        K=cfg['K'],
        C=cfg['C'],
        N=cfg['N'],
        thresholds=cfg['thresholds'],
        lambda_reward=cfg['lambda_reward'],
        window_size=cfg['window_size'],
        traffic_profiles=cfg['traffic_profiles'],
        qos_table_files=cfg['qos_table_files'],
        qos_metrics=cfg['qos_metrics'],
        dynamic_profile_config=cfg['dynamic_profile_config'],
        max_dtis=cfg['max_dtis']
    )
    
    print(f"✓ Environment created (state_dim={env.state_dim}, action_dim={env.action_dim})")
    
    # Create baselines
    print("\n" + "-"*80)
    print("Creating baseline policies...")
    print("-"*80)
    
    baselines = {
        'Equal': create_baseline('equal', cfg=cfg),
        'Proportional': create_baseline('proportional', cfg=cfg),
        'Greedy': create_baseline('greedy', cfg=cfg, qos_tables=env.qos_tables),
        'Random': create_baseline('random', cfg=cfg, seed=42)
    }
    
    print(f"✓ Created {len(baselines)} baseline policies: {list(baselines.keys())}")
    
    # Evaluate each baseline
    print("\n" + "="*80)
    print(f"EVALUATING BASELINES ({cfg['eval_episodes']} episodes each)")
    print("="*80)
    
    all_results = {}
    
    for name, baseline in baselines.items():
        results = evaluate_baseline(baseline, env, cfg['eval_episodes'], name)
        all_results[name] = results
        
        print(f"\n  {name} Results:")
        print(f"    Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"    Beta:   {results['mean_beta']:.4f} ± {results['std_beta']:.4f}")
        print(f"    Violations: {results['mean_violations']:.2f}")
        print(f"    Utilization: {results['mean_utilization']:.2f}%")
    
    # Print summary
    print_summary_table(all_results)
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    save_results(all_results, cfg, cfg['results_dir'])
    
    # Create plots
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    
    plot_baseline_comparison(all_results, cfg, cfg['results_dir'])
    
    # Final summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    
    best_method = min(all_results.keys(), key=lambda m: all_results[m]['mean_beta'])
    worst_method = max(all_results.keys(), key=lambda m: all_results[m]['mean_beta'])
    
    print(f"\n✓ Best baseline: {best_method} (β = {all_results[best_method]['mean_beta']:.4f})")
    print(f"✓ Worst baseline: {worst_method} (β = {all_results[worst_method]['mean_beta']:.4f})")
    print(f"\nResults saved in: {cfg['results_dir']}/")
    print("  - baseline_results.json (metrics)")
    print("  - baseline_comparison.png (bar charts)")
    print("  - baseline_distributions.png (box plots)")
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("\n1. Train SAC separately:")
    print("   python3 sac_training.py")
    print("\n2. Compare SAC with these baselines:")
    print("   python3 compare_sac_with_baselines.py")
    print("\n3. Make sure both use the SAME config.py for fair comparison!")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
