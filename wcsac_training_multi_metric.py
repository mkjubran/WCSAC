"""
WCSAC Training for Multi-Slice Resource Allocation

Worst-Case SAC training with adversarial scenario generation.
Handles traffic uncertainty and QoS variations for robust policies.
"""

import numpy as np
import torch
import random
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from datetime import datetime

from network_env_multi_metric import NetworkEnvironment
from wcsac_agent import WCSAC
import config_multi_metric as config


def set_seeds(seed):
    """Set all random seeds for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def generate_worst_case_scenarios(state, next_state, reward, info, 
                                  uncertainty_radius=0.1):
    """
    Generate worst-case scenarios by perturbing state and reward.
    
    Perturbations model:
    - Traffic variations (± uncertainty_radius of current traffic)
    - QoS metric variations
    - Worst-case reward (penalized)
    
    Args:
        state: Current state
        next_state: Next state from environment
        reward: Nominal reward
        info: Environment info dict
        uncertainty_radius: Size of uncertainty set
        
    Returns:
        worst_case_reward: Pessimistic reward
        worst_case_next_state: Perturbed next state
    """
    # Worst-case reward: reduce by uncertainty factor
    # This makes the agent conservative
    worst_case_reward = reward - abs(reward) * uncertainty_radius
    
    # Worst-case next state: perturb traffic and QoS components
    # Identify which parts of state are traffic/QoS related
    wc_next_state = next_state.copy()
    
    # Perturb state components (add noise to make it harder)
    noise = np.random.randn(len(next_state)) * uncertainty_radius * np.abs(next_state + 1e-8)
    wc_next_state = wc_next_state + noise
    
    # Ensure state remains valid (clip to reasonable ranges)
    wc_next_state = np.clip(wc_next_state, -10, 10)
    
    return worst_case_reward, wc_next_state


def train_wcsac():
    """
    Train WCSAC agent with worst-case robustness.
    """
    
    # Load configuration
    cfg = config.get_config()
    
    # Set all random seeds FIRST
    if hasattr(config, 'RANDOM_SEED'):
        set_seeds(config.RANDOM_SEED)
        print(f"✓ Set global random seed: {config.RANDOM_SEED}")
    
    # Create directories
    os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
    os.makedirs(cfg['tensorboard_dir'], exist_ok=True)
    os.makedirs(cfg['results_dir'], exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f'wcsac_K{cfg["K"]}_W{cfg["window_size"]}_{timestamp}'
    
    writer = SummaryWriter(os.path.join(cfg['tensorboard_dir'], run_name))
    
    # Print configuration
    print("\n" + "="*70)
    print("WORST-CASE SAC (WCSAC) TRAINING")
    print("="*70)
    config.print_config()
    
    # WCSAC-specific parameters
    kappa = getattr(config, 'WCSAC_KAPPA', 0.5)
    uncertainty_radius = getattr(config, 'WCSAC_UNCERTAINTY_RADIUS', 0.1)
    pessimism_penalty = getattr(config, 'WCSAC_PESSIMISM_PENALTY', 0.1)
    
    print(f"\nWCSAC Parameters:")
    print(f"  Robustness (κ):         {kappa} (0=SAC, 1=worst-case)")
    print(f"  Uncertainty radius:     {uncertainty_radius}")
    print(f"  Pessimism penalty:      {pessimism_penalty}")
    print("="*70)
    
    # Create environment with multi-metric QoS support
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
        traffic_seed=config.TRAFFIC_SEED if hasattr(config, 'TRAFFIC_SEED') else None,
        profile_seed=config.PROFILE_SEED if hasattr(config, 'PROFILE_SEED') else None,
        use_efficient_allocation=cfg.get('use_efficient_allocation', False),
        unused_capacity_reward_weight=cfg.get('unused_capacity_reward_weight', 0.0),
        use_transport_layer=cfg.get('use_transport_layer', False),
        transport_link_capacity=cfg.get('transport_link_capacity', 50_000_000),
        slice_packet_sizes=cfg.get('slice_packet_sizes'),
        slice_bit_rates=cfg.get('slice_bit_rates'),
        slice_priorities=cfg.get('slice_priorities'),
        max_transport_delay_per_slice=cfg.get('max_transport_delay_per_slice'),
        transport_delay_weights=cfg.get('transport_delay_weights'),
        service_time_distribution=cfg.get('service_time_distribution', 'deterministic'),
        mg1_stability_threshold=cfg.get('mg1_stability_threshold', 0.999),
    )
    
    # Print QoS mode
    if cfg.get('use_multi_metric_qos'):
        print(f"✓ Using MULTI-METRIC QoS mode")
        for k in range(cfg['K']):
            metrics = cfg['qos_metrics_multi'][k]
            print(f"  Slice {k}: {len(metrics)} metrics - {', '.join(metrics)}")
    else:
        print(f"✓ Using SINGLE-METRIC QoS mode (backward compatible)")
    
    # Print allocation mode
    if cfg.get('use_efficient_allocation'):
        print(f"✓ Using EFFICIENT RESOURCE ALLOCATION mode (K+1 actions)")
        print(f"  Actor output dimension: {cfg['K']+1} (K slices + 1 null)")
        print(f"  Unused capacity reward weight: {cfg.get('unused_capacity_reward_weight', 0.0)}")
    else:
        print(f"✓ Using STANDARD ALLOCATION mode (K actions, sum=C)")
    
    if hasattr(config, 'TRAFFIC_SEED'):
        print(f"✓ Traffic generator seed: {config.TRAFFIC_SEED}")
    if hasattr(config, 'PROFILE_SEED'):
        print(f"✓ Profile manager seed: {config.PROFILE_SEED}")
    
    # Get dimensions (call the property methods)
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    print(f"\nState dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Create WCSAC agent
    agent = WCSAC(
        state_dim=state_dim,
        action_dim=action_dim,
        capacity=cfg['C'],
        lr_actor=cfg['lr_actor'],
        lr_critic=cfg['lr_critic'],
        gamma=cfg['gamma'],
        tau=cfg['tau'],
        kappa=kappa,
        uncertainty_radius=uncertainty_radius,
        pessimism_penalty=pessimism_penalty,
        device=cfg['device'],
        seed=config.NETWORK_SEED if hasattr(config, 'NETWORK_SEED') else None,
        use_efficient_allocation=cfg.get('use_efficient_allocation', False),  # NEW
    )
    
    if hasattr(config, 'NETWORK_SEED'):
        print(f"✓ Agent network seed: {config.NETWORK_SEED}")
    
    # Training statistics
    episode_rewards = []
    episode_betas = []
    episode_violations = []
    episode_worst_case_rewards = []
    
    print(f"\nStarting WCSAC Training: {cfg['num_episodes']} episodes")
    print("=" * 70)
    
    # Training loop
    for episode in tqdm(range(1, cfg['num_episodes'] + 1), desc="WCSAC Training"):
        
        # Reset environment
        state = env.reset()
        
        episode_reward = 0
        episode_beta_sum = 0
        num_violations = 0
        episode_wc_reward = 0
        
        # Transport layer tracking (if enabled)
        if cfg.get('use_transport_layer', False):
            episode_transport_util_sum = 0.0
            episode_transport_delay_sum = [0.0] * cfg['K']
            episode_transport_penalty_sum = 0.0
            transport_steps = 0
        
        # Episode loop
        for dti in range(cfg['max_dtis']):
            
            # Select action
            action = agent.select_action(state)
            
            # Environment step
            next_state, reward, done, info = env.step(action)
            
            # Generate worst-case scenarios
            wc_reward, wc_next_state = generate_worst_case_scenarios(
                state, next_state, reward, info, uncertainty_radius
            )
            
            # Store transition with worst-case variants
            agent.replay_buffer.push(
                state, action, reward, next_state, done,
                worst_case_reward=wc_reward,
                worst_case_next_state=wc_next_state
            )
            
            # Log per-DTI metrics
            global_step = episode * cfg['max_dtis'] + dti
            writer.add_scalar('dti/reward', reward, global_step)
            writer.add_scalar('dti/worst_case_reward', wc_reward, global_step)
            writer.add_scalar('dti/beta', info['beta'], global_step)
            
            # Log transport layer metrics (if enabled)
            if cfg.get('use_transport_layer', False) and 'transport_utilization' in info:
                writer.add_scalar('dti/transport_utilization', info['transport_utilization'], global_step)
                writer.add_scalar('dti/transport_stable', int(info['transport_stable']), global_step)
                writer.add_scalar('dti/transport_penalty', info.get('transport_penalty', 0.0), global_step)
                
                # Per-slice transport delays (in milliseconds)
                for k in range(cfg['K']):
                    delay_ms = info['transport_delays'][k] * 1000
                    writer.add_scalar(f'dti/transport_delay_slice{k}_ms', delay_ms, global_step)
                
                # Per-slice success rates
                for k in range(cfg['K']):
                    writer.add_scalar(f'dti/success_rate_slice{k}', info['success_rate_per_slice'][k], global_step)
            
            # Log actions per slice
            for k in range(cfg['K']):
                writer.add_scalar(f'dti/action_slice{k}', action[k], global_step)
            
            # Log traffic per slice
            for k in range(cfg['K']):
                writer.add_scalar(f'dti/traffic_slice{k}', info['traffic'][k], global_step)
            
            # Aggregate metrics: total allocation and total traffic
            total_allocation = sum(action[:cfg['K']])  # Exclude null allocation if present
            total_traffic = sum(info['traffic'])
            writer.add_scalar('dti/total_allocation', total_allocation, global_step)
            writer.add_scalar('dti/total_traffic', total_traffic, global_step)
            writer.add_scalar('dti/allocation_utilization', total_allocation / cfg['C'], global_step)
            
            # Update statistics
            episode_reward += reward
            episode_wc_reward += wc_reward
            episode_beta_sum += info['beta']
            if info['constraint_violated']:
                num_violations += 1
            
            # Accumulate transport metrics (if enabled)
            if cfg.get('use_transport_layer', False) and 'transport_utilization' in info:
                episode_transport_util_sum += info['transport_utilization']
                for k in range(cfg['K']):
                    episode_transport_delay_sum[k] += info['transport_delays'][k]
                episode_transport_penalty_sum += info.get('transport_penalty', 0.0)
                transport_steps += 1
            
            # Train agent
            if len(agent.replay_buffer) >= cfg['min_buffer_size']:
                metrics = agent.update(cfg['batch_size'])
                
                # Log training metrics
                if metrics and dti % 10 == 0:
                    step = episode * cfg['max_dtis'] + dti
                    for key, value in metrics.items():
                        writer.add_scalar(f'train/{key}', value, step)
            
            state = next_state
            
            if done:
                break
        
        # Episode statistics
        avg_beta = episode_beta_sum / (dti + 1)
        avg_wc_reward = episode_wc_reward / (dti + 1)
        
        episode_rewards.append(episode_reward)
        episode_betas.append(avg_beta)
        episode_violations.append(num_violations)
        episode_worst_case_rewards.append(episode_wc_reward)
        
        # Logging
        if episode % cfg['log_interval'] == 0:
            writer.add_scalar('episode/reward', episode_reward, episode)
            writer.add_scalar('episode/worst_case_reward', episode_wc_reward, episode)
            writer.add_scalar('episode/avg_beta', avg_beta, episode)
            writer.add_scalar('episode/violations', num_violations, episode)
            writer.add_scalar('episode/buffer_size', len(agent.replay_buffer), episode)
            
            # Transport layer episode averages (if enabled)
            if cfg.get('use_transport_layer', False) and transport_steps > 0:
                avg_transport_util = episode_transport_util_sum / transport_steps
                avg_transport_penalty = episode_transport_penalty_sum / transport_steps
                
                writer.add_scalar('episode/avg_transport_utilization', avg_transport_util, episode)
                writer.add_scalar('episode/avg_transport_penalty', avg_transport_penalty, episode)
                
                # Average delay per slice
                for k in range(cfg['K']):
                    avg_delay_ms = (episode_transport_delay_sum[k] / transport_steps) * 1000
                    writer.add_scalar(f'episode/avg_transport_delay_slice{k}_ms', avg_delay_ms, episode)
            
            # Moving averages
            if len(episode_rewards) >= 100:
                avg_reward_100 = np.mean(episode_rewards[-100:])
                avg_beta_100 = np.mean(episode_betas[-100:])
                avg_wc_reward_100 = np.mean(episode_worst_case_rewards[-100:])
                
                writer.add_scalar('episode/avg_reward_100', avg_reward_100, episode)
                writer.add_scalar('episode/avg_beta_100', avg_beta_100, episode)
                writer.add_scalar('episode/avg_worst_case_reward_100', avg_wc_reward_100, episode)
            
            print(f"\nEpisode {episode}/{cfg['num_episodes']}:")
            print(f"  Reward: {episode_reward:.2f} (WC: {episode_wc_reward:.2f})")
            print(f"  Avg Beta: {avg_beta:.4f}")
            print(f"  Violations: {num_violations}")
            print(f"  Buffer: {len(agent.replay_buffer)}")
        
        # Save checkpoint
        if episode % cfg['save_interval'] == 0:
            save_path = os.path.join(cfg['checkpoint_dir'], f'{run_name}_ep{episode}.pt')
            agent.save(save_path)
            print(f"  Saved checkpoint: {save_path}")
    
    # Final save
    final_path = os.path.join(cfg['checkpoint_dir'], f'{run_name}_final.pt')
    agent.save(final_path)
    print(f"\nWCSAC Training complete! Final model saved: {final_path}")
    
    # Close writer
    writer.close()
    
    # Plot training curves
    plot_wcsac_training_curves(
        episode_rewards, episode_worst_case_rewards, episode_betas, 
        episode_violations, cfg['checkpoint_dir'], run_name
    )
    
    return agent, env, episode_rewards, episode_betas


def plot_wcsac_training_curves(rewards, wc_rewards, betas, violations, 
                               save_dir, run_name):
    """Plot WCSAC training curves"""
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))
    
    # Reward
    axes[0].plot(rewards, alpha=0.3, label='Nominal Reward')
    axes[0].plot(wc_rewards, alpha=0.3, label='Worst-Case Reward', color='red')
    if len(rewards) >= 10:
        rewards_smooth = np.convolve(rewards, np.ones(10)/10, mode='valid')
        wc_smooth = np.convolve(wc_rewards, np.ones(10)/10, mode='valid')
        axes[0].plot(range(9, len(rewards)), rewards_smooth, 
                    label='Smoothed Nominal', linewidth=2)
        axes[0].plot(range(9, len(wc_rewards)), wc_smooth, 
                    label='Smoothed WC', linewidth=2, color='darkred')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('WCSAC Training Reward (Nominal vs Worst-Case)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Reward gap
    gap = np.array(rewards) - np.array(wc_rewards)
    axes[1].plot(gap, alpha=0.6, color='purple')
    if len(gap) >= 10:
        gap_smooth = np.convolve(gap, np.ones(10)/10, mode='valid')
        axes[1].plot(range(9, len(gap)), gap_smooth, 
                    label='Smoothed', linewidth=2, color='darkviolet')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Reward Gap (Nominal - WC)')
    axes[1].set_title('Robustness Gap')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Beta
    axes[2].plot(betas, alpha=0.3, label='Avg Beta')
    if len(betas) >= 10:
        betas_smooth = np.convolve(betas, np.ones(10)/10, mode='valid')
        axes[2].plot(range(9, len(betas)), betas_smooth, 
                    label='Smoothed (10 ep)', linewidth=2)
    axes[2].axhline(y=0.2, color='r', linestyle='--', label='Threshold (0.2)')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Beta (Violation Ratio)')
    axes[2].set_title('QoS Performance')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Violations
    axes[3].plot(violations, alpha=0.6, color='orange')
    axes[3].set_xlabel('Episode')
    axes[3].set_ylabel('Constraint Violations')
    axes[3].set_title('Constraint Violations per Episode')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    plot_path = os.path.join(save_dir, f'{run_name}_training_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"WCSAC training curves saved: {plot_path}")
    plt.close()


if __name__ == "__main__":
    
    print("Worst-Case SAC Training for Multi-Slice Resource Allocation")
    print("Robust RL with Adversarial Scenarios")
    print("=" * 70)
    
    # Train
    agent, env, rewards, betas = train_wcsac()
    
    print("\nWCSAC Training Summary:")
    print(f"  Final 100-episode avg reward: {np.mean(rewards[-100:]):.2f}")
    print(f"  Final 100-episode avg beta: {np.mean(betas[-100:]):.4f}")
    print(f"  Best episode reward: {np.max(rewards):.2f}")
    print(f"  Best episode beta: {np.min(betas):.4f}")
