"""
SAC Training for Multi-Slice Resource Allocation - Implements Algorithm 2
Main training loop with TensorBoard logging and visualization
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
from sac_agent import SAC
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

def train_sac():
    """
    Train SAC agent following Algorithm 2.
    Uses parameters from config.py
    
    Algorithm 2: SAC Training for Multi-Slice Resource Allocation
    - Outer loop: E_max episodes
    - Inner loop: T_max DTIs per episode
    - Episodic learning: Reset() at start of each episode
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
    run_name = f'sac_K{cfg["K"]}_W{cfg["window_size"]}_{timestamp}'
    
    writer = SummaryWriter(os.path.join(cfg['tensorboard_dir'], run_name))
    
    # Print configuration
    config.print_config()
    
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
        qos_metrics_multi=cfg.get('qos_metrics_multi'),  # NEW: Multi-metric support
        thresholds_multi=cfg.get('thresholds_multi'),  # NEW
        qos_metric_directions=cfg.get('qos_metric_directions'),  # NEW
        dynamic_profile_config=cfg['dynamic_profile_config'],
        max_dtis=cfg['max_dtis'],
        traffic_seed=config.TRAFFIC_SEED if hasattr(config, 'TRAFFIC_SEED') else None,
        profile_seed=config.PROFILE_SEED if hasattr(config, 'PROFILE_SEED') else None
    )
    
    # Print QoS mode
    if cfg.get('use_multi_metric_qos'):
        print(f"✓ Using MULTI-METRIC QoS mode")
        for k in range(cfg['K']):
            metrics = cfg['qos_metrics_multi'][k]
            print(f"  Slice {k}: {len(metrics)} metrics - {', '.join(metrics)}")
    else:
        print(f"✓ Using SINGLE-METRIC QoS mode (backward compatible)")
    
    if hasattr(config, 'TRAFFIC_SEED'):
        print(f"✓ Traffic generator seed: {config.TRAFFIC_SEED}")
    if hasattr(config, 'PROFILE_SEED'):
        print(f"✓ Profile manager seed: {config.PROFILE_SEED}")
    
    # Get dimensions (call the property methods)
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    print(f"\nState dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Create SAC agent
    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        capacity=cfg['C'],
        lr_actor=cfg['lr_actor'],
        lr_critic=cfg['lr_critic'],
        gamma=cfg['gamma'],
        tau=cfg['tau'],
        device=cfg['device'],
        seed=config.NETWORK_SEED if hasattr(config, 'NETWORK_SEED') else None
    )
    
    if hasattr(config, 'NETWORK_SEED'):
        print(f"✓ Agent network seed: {config.NETWORK_SEED}")
    
    # Training statistics
    episode_rewards = []
    episode_betas = []
    episode_violations = []
    
    print(f"\nStarting Training: {cfg['num_episodes']} episodes")
    print("=" * 70)
    
    # Algorithm 2: Outer loop (episodes)
    for episode in tqdm(range(1, cfg['num_episodes'] + 1), desc="Training"):
        
        # Reset environment (Algorithm 1: Reset())
        state = env.reset()
        
        episode_reward = 0
        episode_beta_sum = 0
        num_violations = 0
        
        # Algorithm 2: Inner loop (DTIs)
        for dti in range(cfg['max_dtis']):
            
            # Select action from policy
            action = agent.select_action(state)
            
            # Environment step (Algorithm 1: Step(a))
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # Log per-DTI metrics (for detailed analysis)
            global_step = episode * cfg['max_dtis'] + dti
            writer.add_scalar('dti/reward', reward, global_step)
            writer.add_scalar('dti/beta', info['beta'], global_step)
            
            # Log actions per slice
            for k in range(cfg['K']):
                writer.add_scalar(f'dti/action_slice{k}', action[k], global_step)
            
            # Log traffic per slice
            for k in range(cfg['K']):
                writer.add_scalar(f'dti/traffic_slice{k}', info['traffic'][k], global_step)
            
            # Log active profiles per slice (for dynamic profiles)
            for k in range(cfg['K']):
                profile_map = {
                    'uniform': 0, 'extremely_low': 1, 'low': 2,
                    'medium': 3, 'high': 4, 'extremely_high': 5, 'external': 6
                }
                profile_name = str(info['active_profiles'][k])
                profile_id = profile_map.get(profile_name, -1)
                writer.add_scalar(f'dti/active_profile_slice{k}', profile_id, global_step)
            
            # Log specific episodes for detailed per-DTI analysis
            # Log every 10th episode, first/last episodes, and episodes around milestones
            log_this_episode = (
                episode in [1, 2, 5, cfg['num_episodes']] or
                episode % 10 == 0
            )
            
            if log_this_episode:
                writer.add_scalar(f'episode_{episode}/reward', reward, dti)
                writer.add_scalar(f'episode_{episode}/beta', info['beta'], dti)
                for k in range(cfg['K']):
                    writer.add_scalar(f'episode_{episode}/action_slice{k}', action[k], dti)
                    writer.add_scalar(f'episode_{episode}/traffic_slice{k}', info['traffic'][k], dti)
                    profile_name = str(info['active_profiles'][k])
                    profile_id = profile_map.get(profile_name, -1)
                    writer.add_scalar(f'episode_{episode}/active_profile_slice{k}', profile_id, dti)
            
            # Update statistics
            episode_reward += reward
            episode_beta_sum += info['beta']
            if info['constraint_violated']:
                num_violations += 1
            
            # Train agent (if enough data)
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
        episode_rewards.append(episode_reward)
        episode_betas.append(avg_beta)
        episode_violations.append(num_violations)
        
        # Logging
        if episode % cfg['log_interval'] == 0:
            writer.add_scalar('episode/reward', episode_reward, episode)
            writer.add_scalar('episode/avg_beta', avg_beta, episode)
            writer.add_scalar('episode/violations', num_violations, episode)
            writer.add_scalar('episode/buffer_size', len(agent.replay_buffer), episode)
            
            # Moving averages (last 100 episodes)
            if len(episode_rewards) >= 100:
                avg_reward_100 = np.mean(episode_rewards[-100:])
                avg_beta_100 = np.mean(episode_betas[-100:])
                writer.add_scalar('episode/avg_reward_100', avg_reward_100, episode)
                writer.add_scalar('episode/avg_beta_100', avg_beta_100, episode)
            
            print(f"\nEpisode {episode}/{cfg['num_episodes']}:")
            print(f"  Reward: {episode_reward:.2f}")
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
    print(f"\nTraining complete! Final model saved: {final_path}")
    
    # Close writer
    writer.close()
    
    # Plot training curves
    plot_training_curves(episode_rewards, episode_betas, episode_violations, 
                         cfg['checkpoint_dir'], run_name)
    
    return agent, env, episode_rewards, episode_betas


def plot_training_curves(rewards, betas, violations, save_dir, run_name):
    """Plot and save training curves"""
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Reward
    axes[0].plot(rewards, alpha=0.3, label='Episode Reward')
    if len(rewards) >= 10:
        rewards_smooth = np.convolve(rewards, np.ones(10)/10, mode='valid')
        axes[0].plot(range(9, len(rewards)), rewards_smooth, label='Smoothed (10 ep)', linewidth=2)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Training Reward')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Beta
    axes[1].plot(betas, alpha=0.3, label='Avg Beta')
    if len(betas) >= 10:
        betas_smooth = np.convolve(betas, np.ones(10)/10, mode='valid')
        axes[1].plot(range(9, len(betas)), betas_smooth, label='Smoothed (10 ep)', linewidth=2)
    axes[1].axhline(y=0.2, color='r', linestyle='--', label='Threshold (0.2)')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Beta (Violation Ratio)')
    axes[1].set_title('QoS Performance (Lower is Better)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Violations
    axes[2].plot(violations, alpha=0.6)
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Constraint Violations')
    axes[2].set_title('Constraint Violations per Episode')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    plot_path = os.path.join(save_dir, f'{run_name}_training_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved: {plot_path}")
    plt.close()


if __name__ == "__main__":
    
    print("SAC Training for Multi-Slice Resource Allocation")
    print("Implementing Algorithm 2 from LaTeX")
    print("=" * 70)
    
    # Configuration loaded from config.py
    # Edit config.py to change parameters
    
    # Train
    agent, env, rewards, betas = train_sac()
    
    print("\nTraining Summary:")
    print(f"  Final 100-episode avg reward: {np.mean(rewards[-100:]):.2f}")
    print(f"  Final 100-episode avg beta: {np.mean(betas[-100:]):.4f}")
    print(f"  Best episode reward: {np.max(rewards):.2f}")
    print(f"  Best episode beta: {np.min(betas):.4f}")
