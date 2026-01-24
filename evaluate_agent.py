"""
Evaluation and Visualization for Trained SAC Agent
Generates detailed plots and metrics
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os
import torch

from network_env import NetworkEnvironment
from sac_agent import SAC
import config


def evaluate_agent(agent, env, num_episodes=100):
    """
    Evaluate trained agent over multiple episodes.
    
    Returns:
        dict with detailed metrics
    """
    episode_rewards = []
    episode_betas = []
    episode_violations = []
    
    # Per-slice metrics
    slice_allocations = [[] for _ in range(env.K)]
    slice_betas_individual = [[] for _ in range(env.K)]
    
    # DTI-level tracking
    beta_progression = []
    reward_progression = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_betas_list = []
        episode_violations_count = 0
        
        for dti in range(env.max_dtis):
            action = agent.select_action(state, eval_mode=True)
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_betas_list.append(info['beta'])
            
            if info['constraint_violated']:
                episode_violations_count += 1
            
            # Track allocations
            for k in range(env.K):
                slice_allocations[k].append(info['r_used'][k])
            
            # Track progression (first episode only for visualization)
            if episode == 0:
                beta_progression.append(info['beta'])
                reward_progression.append(reward)
            
            state = next_state
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_betas.append(np.mean(episode_betas_list))
        episode_violations.append(episode_violations_count)
    
    return {
        'rewards': episode_rewards,
        'betas': episode_betas,
        'violations': episode_violations,
        'slice_allocations': slice_allocations,
        'beta_progression': beta_progression,
        'reward_progression': reward_progression,
    }


def plot_comprehensive_results(results, env, save_path='results'):
    """
    Create comprehensive visualization of results.
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 12)
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Episode Rewards Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(results['rewards'], bins=30, edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(results['rewards']), color='r', linestyle='--', 
                label=f'Mean: {np.mean(results["rewards"]):.2f}')
    ax1.set_xlabel('Total Reward')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Episode Reward Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Beta Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(results['betas'], bins=30, edgecolor='black', alpha=0.7, color='orange')
    ax2.axvline(np.mean(results['betas']), color='r', linestyle='--',
                label=f'Mean: {np.mean(results["betas"]):.4f}')
    ax2.axvline(0.2, color='g', linestyle='--', label='Target: 0.20')
    ax2.set_xlabel('Beta (Violation Ratio)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Beta Distribution (Lower is Better)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Constraint Violations
    ax3 = fig.add_subplot(gs[0, 2])
    violation_counts = np.bincount(results['violations'])
    ax3.bar(range(len(violation_counts)), violation_counts, edgecolor='black', alpha=0.7)
    ax3.set_xlabel('Number of Violations per Episode')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Constraint Violations')
    ax3.grid(True, alpha=0.3)
    
    # 4. Beta Progression (first episode)
    ax4 = fig.add_subplot(gs[1, :2])
    dtis = range(len(results['beta_progression']))
    ax4.plot(dtis, results['beta_progression'], linewidth=2)
    ax4.axhline(y=0.2, color='r', linestyle='--', alpha=0.5, label='Threshold')
    ax4.fill_between(dtis, 0, results['beta_progression'], alpha=0.3)
    ax4.set_xlabel('DTI')
    ax4.set_ylabel('Beta')
    ax4.set_title('Beta Progression Over Episode')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Reward Progression (first episode)
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(results['reward_progression'], linewidth=2, color='green')
    ax5.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax5.set_xlabel('DTI')
    ax5.set_ylabel('Reward')
    ax5.set_title('Reward per DTI')
    ax5.grid(True, alpha=0.3)
    
    # 6. Resource Allocation per Slice (Boxplot)
    ax6 = fig.add_subplot(gs[2, 0])
    slice_data = [results['slice_allocations'][k] for k in range(env.K)]
    bp = ax6.boxplot(slice_data, labels=[f'Slice {k+1}' for k in range(env.K)],
                     patch_artist=True)
    for patch, color in zip(bp['boxes'], plt.cm.Set3.colors):
        patch.set_facecolor(color)
    ax6.axhline(y=env.C/env.K, color='r', linestyle='--', alpha=0.5,
                label=f'Equal split: {env.C/env.K:.1f}')
    ax6.set_ylabel('RBs Allocated')
    ax6.set_title('Resource Allocation Distribution')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Allocation Heatmap (sample from first episode)
    ax7 = fig.add_subplot(gs[2, 1])
    sample_length = min(50, len(results['slice_allocations'][0])//env.max_dtis)
    alloc_matrix = np.array([results['slice_allocations'][k][:sample_length] 
                              for k in range(env.K)])
    im = ax7.imshow(alloc_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax7.set_ylabel('Slice')
    ax7.set_xlabel('DTI')
    ax7.set_title(f'Resource Allocation Heatmap (First {sample_length} DTIs)')
    ax7.set_yticks(range(env.K))
    ax7.set_yticklabels([f'Slice {k+1}' for k in range(env.K)])
    plt.colorbar(im, ax=ax7, label='RBs')
    
    # 8. Summary Statistics
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    summary_text = f"""
    EVALUATION SUMMARY
    {'='*30}
    
    Episodes: {len(results['rewards'])}
    
    Reward:
      Mean: {np.mean(results['rewards']):.2f}
      Std:  {np.std(results['rewards']):.2f}
      Max:  {np.max(results['rewards']):.2f}
      Min:  {np.min(results['rewards']):.2f}
    
    Beta (Violation Ratio):
      Mean: {np.mean(results['betas']):.4f}
      Std:  {np.std(results['betas']):.4f}
      Min:  {np.min(results['betas']):.4f}
      Max:  {np.max(results['betas']):.4f}
      % < 0.2: {100*np.mean(np.array(results['betas']) < 0.2):.1f}%
    
    Violations:
      Total: {np.sum(results['violations'])}
      Avg/Episode: {np.mean(results['violations']):.2f}
    
    Resource Usage:
      Slice 1 Avg: {np.mean(results['slice_allocations'][0]):.2f} RBs
      Slice 2 Avg: {np.mean(results['slice_allocations'][1]):.2f} RBs
    """
    
    ax8.text(0.1, 0.5, summary_text, transform=ax8.transAxes,
             fontsize=10, verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('SAC Agent Evaluation Results', fontsize=16, fontweight='bold')
    
    # Save
    plt.savefig(os.path.join(save_path, 'comprehensive_results.png'), 
                dpi=150, bbox_inches='tight')
    print(f"Comprehensive results saved to {save_path}/comprehensive_results.png")
    plt.close()


def compare_traffic_profiles(agent, save_path='results'):
    """
    Compare agent performance across different traffic profiles.
    """
    profiles = ['uniform', 'extremely_low', 'low', 'medium', 'high', 'extremely_high']
    results_by_profile = {}
    
    for profile in profiles:
        env = NetworkEnvironment(
            K=2, C=8, N=20,
            thresholds=[0.2, 0.15],
            lambda_reward=0.5,
            window_size=5,
            traffic_profiles=[profile, profile]
        )
        
        results = evaluate_agent(agent, env, num_episodes=50)
        results_by_profile[profile] = results
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Rewards
    reward_data = [results_by_profile[p]['rewards'] for p in profiles]
    bp1 = axes[0].boxplot(reward_data, labels=profiles, patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('lightblue')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Reward by Traffic Profile')
    axes[0].grid(True, alpha=0.3)
    
    # Betas
    beta_data = [results_by_profile[p]['betas'] for p in profiles]
    bp2 = axes[1].boxplot(beta_data, labels=profiles, patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('lightcoral')
    axes[1].axhline(y=0.2, color='g', linestyle='--', label='Target')
    axes[1].set_ylabel('Beta (Violation Ratio)')
    axes[1].set_title('QoS Performance by Traffic Profile')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Violations
    viol_data = [results_by_profile[p]['violations'] for p in profiles]
    mean_viols = [np.mean(v) for v in viol_data]
    axes[2].bar(profiles, mean_viols, edgecolor='black', alpha=0.7)
    axes[2].set_ylabel('Avg Violations per Episode')
    axes[2].set_title('Constraint Violations by Traffic Profile')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'traffic_profile_comparison.png'),
                dpi=150, bbox_inches='tight')
    print(f"Traffic profile comparison saved to {save_path}/traffic_profile_comparison.png")
    plt.close()


if __name__ == "__main__":
    
    print("SAC Agent Evaluation and Visualization")
    print("=" * 70)
    
    # Load configuration from config.py
    cfg = config.get_config()
    config.print_config()
    
    # Find checkpoint
    checkpoint_path = cfg['checkpoint_path']
    if checkpoint_path is None:
        # Try to find latest checkpoint
        if os.path.exists(cfg['checkpoint_dir']):
            checkpoints = [f for f in os.listdir(cfg['checkpoint_dir']) if f.endswith('_final.pt')]
            if checkpoints:
                checkpoint_path = os.path.join(cfg['checkpoint_dir'], checkpoints[-1])
    
    # Create environment
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
    
    # Create agent
    agent = SAC(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        capacity=env.C,
        device=cfg['device']
    )
    
    # Load if checkpoint exists
    if checkpoint_path and os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
        print(f"\nLoaded checkpoint: {checkpoint_path}")
    else:
        print("\nNo checkpoint found, using random policy")
    
    # Evaluate
    print("\nEvaluating agent...")
    results = evaluate_agent(agent, env, num_episodes=cfg['eval_episodes'])
    
    print(f"\nEvaluation complete:")
    print(f"  Mean reward: {np.mean(results['rewards']):.2f}")
    print(f"  Mean beta: {np.mean(results['betas']):.4f}")
    print(f"  Total violations: {np.sum(results['violations'])}")
    
    # Generate plots
    print("\nGenerating visualizations...")
    plot_comprehensive_results(results, env, save_path=cfg['results_dir'])
    
    # Compare traffic profiles
    if checkpoint_path:
        print("\nComparing traffic profiles...")
        compare_traffic_profiles(agent, save_path=cfg['results_dir'])
    
    print("\nVisualization complete!")
