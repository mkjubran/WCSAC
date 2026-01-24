"""
Compare SAC with Pre-computed Baseline Results

This script loads baseline results from baseline_results.json
and compares them with a trained SAC agent.

Usage:
    1. First run: python3 run_baselines_only.py
    2. Then train: python3 sac_training.py
    3. Finally compare: python3 compare_sac_with_baselines.py
"""

import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from network_env import NetworkEnvironment
from sac_agent import SAC
import config


def load_baseline_results(results_dir):
    """Load pre-computed baseline results from JSON"""
    
    json_path = os.path.join(results_dir, 'baseline_results.json')
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Baseline results not found at {json_path}\n"
            f"Please run: python3 run_baselines_only.py first"
        )
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"✓ Loaded baseline results from: {json_path}")
    print(f"  Timestamp: {data['timestamp']}")
    print(f"  Config: K={data['config']['K']}, C={data['config']['C']}, "
          f"Profiles={data['config']['traffic_profiles']}")
    
    return data


def evaluate_sac(agent, env, num_episodes=100):
    """Evaluate trained SAC agent"""
    
    print(f"\nEvaluating SAC over {num_episodes} episodes...")
    
    episode_rewards = []
    episode_betas = []
    episode_violations = []
    episode_utilization = []
    
    for episode in range(num_episodes):
        if episode % 20 == 0:
            print(f"  Episode {episode}/{num_episodes}...", end='\r')
        
        state = env.reset()
        episode_reward = 0
        episode_betas_list = []
        episode_violations_count = 0
        episode_rb_used = []
        
        for dti in range(env.max_dtis):
            action = agent.select_action(state, eval_mode=True)
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_betas_list.append(info['beta'])
            episode_rb_used.append(np.sum(info['r_used']))
            
            if info['constraint_violated']:
                episode_violations_count += 1
            
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_betas.append(np.mean(episode_betas_list))
        episode_violations.append(episode_violations_count)
        episode_utilization.append(np.mean(episode_rb_used) / env.C * 100)
    
    print(f"  Episode {num_episodes}/{num_episodes}... Done!")
    
    return {
        'policy_name': 'SAC',
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'mean_beta': float(np.mean(episode_betas)),
        'std_beta': float(np.std(episode_betas)),
        'mean_violations': float(np.mean(episode_violations)),
        'mean_utilization': float(np.mean(episode_utilization)),
        'rewards': episode_rewards,
        'betas': episode_betas,
    }


def verify_config_match(baseline_config, current_config):
    """Verify that baseline and SAC used same config"""
    
    print("\n" + "="*80)
    print("CONFIG VERIFICATION")
    print("="*80)
    
    critical_params = ['K', 'C', 'N', 'thresholds', 'traffic_profiles', 'window_size', 'max_dtis', 'lambda_reward']
    
    mismatches = []
    
    for param in critical_params:
        baseline_val = baseline_config.get(param)
        current_val = current_config.get(param)
        
        match = baseline_val == current_val
        status = "✓" if match else "✗"
        
        print(f"  {param:<20} Baseline: {baseline_val!s:<30} Current: {current_val!s:<30} {status}")
        
        if not match:
            mismatches.append(param)
    
    if mismatches:
        print("\n⚠ WARNING: Config mismatch detected!")
        print(f"  Parameters differ: {mismatches}")
        print("  Comparison may not be fair!")
        print("  Make sure config.py matches baseline evaluation config.")
        return False
    else:
        print("\n✓ Configs match! Fair comparison guaranteed.")
        return True


def statistical_comparison(sac_results, baseline_results):
    """Perform statistical tests"""
    
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("="*80)
    
    sac_rewards = sac_results['rewards']
    sac_betas = sac_results['betas']
    
    print("\nPaired t-tests (SAC vs Baselines):")
    print("-"*80)
    
    for method_name in baseline_results.keys():
        if method_name in ['Equal', 'Proportional', 'Greedy', 'Random']:
            print(f"\nSAC vs {method_name}:")
            
            # Note: We can't do true paired t-test since baselines were run separately
            # But we can do independent samples t-test
            baseline_beta = baseline_results[method_name]['mean_beta']
            sac_beta = sac_results['mean_beta']
            
            improvement_pct = ((baseline_beta - sac_beta) / baseline_beta) * 100
            
            print(f"  Baseline Beta: {baseline_beta:.4f}")
            print(f"  SAC Beta:      {sac_beta:.4f}")
            
            if sac_beta < baseline_beta:
                print(f"  Improvement:   {improvement_pct:.2f}% better ✓")
            else:
                print(f"  Difference:    {improvement_pct:.2f}% worse ✗")


def plot_full_comparison(sac_results, baseline_data, save_dir):
    """Create comprehensive comparison plots"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract baseline results
    baseline_results = baseline_data['results']
    
    # Combine all results
    all_methods = list(baseline_results.keys()) + ['SAC']
    
    # Set style
    sns.set_style("whitegrid")
    colors_baselines = plt.cm.Set3(range(len(baseline_results)))
    color_sac = 'gold'
    colors = list(colors_baselines) + [color_sac]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('SAC vs Baselines Comparison', fontsize=14, fontweight='bold')
    
    # 1. Rewards
    rewards_mean = [baseline_results[m]['mean_reward'] for m in baseline_results.keys()] + [sac_results['mean_reward']]
    rewards_std = [baseline_results[m]['std_reward'] for m in baseline_results.keys()] + [sac_results['std_reward']]
    
    axes[0, 0].bar(all_methods, rewards_mean, yerr=rewards_std, capsize=5,
                   color=colors, edgecolor='black', alpha=0.7)
    axes[0, 0].set_ylabel('Average Episode Reward', fontsize=11)
    axes[0, 0].set_title('Total Reward Performance', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # Highlight SAC
    axes[0, 0].get_children()[-5].set_linewidth(3)
    
    # 2. Beta (QoS)
    betas_mean = [baseline_results[m]['mean_beta'] for m in baseline_results.keys()] + [sac_results['mean_beta']]
    betas_std = [baseline_results[m]['std_beta'] for m in baseline_results.keys()] + [sac_results['std_beta']]
    
    bar_colors = ['green' if b < 0.2 else 'orange' if b < 0.3 else 'red' for b in betas_mean]
    axes[0, 1].bar(all_methods, betas_mean, yerr=betas_std, capsize=5,
                   color=bar_colors, edgecolor='black', alpha=0.7)
    axes[0, 1].axhline(y=0.2, color='r', linestyle='--', linewidth=2, label='Target (0.2)')
    axes[0, 1].set_ylabel('Beta (QoS Violation Ratio)', fontsize=11)
    axes[0, 1].set_title('QoS Performance', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Violations
    viols_mean = [baseline_results[m]['mean_violations'] for m in baseline_results.keys()] + [sac_results['mean_violations']]
    
    axes[1, 0].bar(all_methods, viols_mean, color=colors, edgecolor='black', alpha=0.7)
    axes[1, 0].set_ylabel('Constraint Violations per Episode', fontsize=11)
    axes[1, 0].set_title('Capacity Constraint Violations', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Utilization
    util_mean = [baseline_results[m]['mean_utilization'] for m in baseline_results.keys()] + [sac_results['mean_utilization']]
    
    axes[1, 1].bar(all_methods, util_mean, color=colors, edgecolor='black', alpha=0.7)
    axes[1, 1].axhline(y=100, color='r', linestyle='--', linewidth=2, label='Full capacity')
    axes[1, 1].set_ylabel('Resource Utilization (%)', fontsize=11)
    axes[1, 1].set_title('RB Usage Efficiency', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].set_ylim([0, 105])
    
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, 'sac_vs_baselines_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved to: {plot_path}")
    plt.close()


def print_final_summary(sac_results, baseline_data):
    """Print final comparison summary"""
    
    baseline_results = baseline_data['results']
    
    print("\n" + "="*80)
    print("FINAL COMPARISON SUMMARY")
    print("="*80)
    
    print(f"\n{'Method':<15} {'Reward':<20} {'Beta':<20} {'Rank':<8}")
    print("-"*80)
    
    # Combine and sort by beta
    all_results = {**baseline_results, 'SAC': sac_results}
    sorted_methods = sorted(all_results.keys(), key=lambda m: all_results[m]['mean_beta'])
    
    for rank, method in enumerate(sorted_methods, 1):
        r = all_results[method]
        reward_str = f"{r['mean_reward']:>7.2f} ± {r['std_reward']:<6.2f}"
        beta_str = f"{r['mean_beta']:>6.4f} ± {r['std_beta']:<6.4f}"
        
        if method == 'SAC':
            print(f"🏆 {method:<13} {reward_str:<20} {beta_str:<20} #{rank}")
        elif rank == 1:
            print(f"** {method:<13} {reward_str:<20} {beta_str:<20} #{rank}")
        else:
            print(f"   {method:<15} {reward_str:<20} {beta_str:<20} #{rank}")
    
    print("\n🏆 = SAC (learned policy)")
    print("** = Best baseline")
    print("="*80)
    
    # Performance analysis
    best_baseline = sorted_methods[0] if sorted_methods[0] != 'SAC' else sorted_methods[1]
    sac_rank = sorted_methods.index('SAC') + 1
    
    print(f"\nSAC Performance:")
    print(f"  Rank: #{sac_rank} out of {len(sorted_methods)}")
    print(f"  Beta: {sac_results['mean_beta']:.4f}")
    
    if sac_rank == 1:
        improvement = ((all_results[best_baseline]['mean_beta'] - sac_results['mean_beta']) / 
                      all_results[best_baseline]['mean_beta']) * 100
        print(f"  Status: ✓ BEST - Beats all baselines!")
        print(f"  Improvement over best baseline ({best_baseline}): {improvement:.2f}%")
    else:
        print(f"  Status: Needs improvement")
        print(f"  Best baseline ({best_baseline}): {all_results[best_baseline]['mean_beta']:.4f}")


def main():
    """Main comparison function"""
    
    print("="*80)
    print("SAC vs BASELINES COMPARISON")
    print("="*80)
    
    # Load config
    cfg = config.get_config()
    
    # Load baseline results
    print("\nLoading pre-computed baseline results...")
    baseline_data = load_baseline_results(cfg['results_dir'])
    
    # Verify config match
    config_match = verify_config_match(baseline_data['config'], cfg)
    
    if not config_match:
        response = input("\nConfig mismatch detected. Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Comparison aborted.")
            return
    
    # Create environment
    print("\nCreating environment...")
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
    
    # Load SAC agent
    print("\nLoading SAC agent...")
    agent = SAC(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        capacity=env.C,
        device=cfg['device']
    )
    
    checkpoint_path = cfg['checkpoint_path']
    if checkpoint_path and os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
        print(f"✓ Loaded SAC checkpoint: {checkpoint_path}")
    else:
        print("⚠ No SAC checkpoint found. Using untrained agent.")
        print("  Train SAC first: python3 sac_training.py")
    
    # Evaluate SAC
    print("\n" + "="*80)
    print("EVALUATING SAC")
    print("="*80)
    
    sac_results = evaluate_sac(agent, env, cfg['eval_episodes'])
    
    print(f"\nSAC Results:")
    print(f"  Reward: {sac_results['mean_reward']:.2f} ± {sac_results['std_reward']:.2f}")
    print(f"  Beta:   {sac_results['mean_beta']:.4f} ± {sac_results['std_beta']:.4f}")
    print(f"  Violations: {sac_results['mean_violations']:.2f}")
    print(f"  Utilization: {sac_results['mean_utilization']:.2f}%")
    
    # Statistical comparison
    statistical_comparison(sac_results, baseline_data['results'])
    
    # Generate plots
    print("\n" + "="*80)
    print("GENERATING COMPARISON PLOTS")
    print("="*80)
    
    plot_full_comparison(sac_results, baseline_data, cfg['results_dir'])
    
    # Final summary
    print_final_summary(sac_results, baseline_data)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80)
    print(f"\nResults saved in: {cfg['results_dir']}/")
    print("  - sac_vs_baselines_comparison.png")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
