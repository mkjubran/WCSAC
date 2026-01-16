"""
SAC Training for Multi-Slice Resource Allocation - Implements Algorithm 2
Main training loop with TensorBoard logging and visualization
"""

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from datetime import datetime

from network_env import NetworkEnvironment
from sac_agent import SAC
import config


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
    
    # Create directories
    os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
    os.makedirs(cfg['tensorboard_dir'], exist_ok=True)
    os.makedirs(cfg['results_dir'], exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f'sac_K{cfg["K"]}_W{cfg["window_size"]}_{timestamp}'
    
    writer = SummaryWriter(os.path.join(cfg['tensorboard_dir'], run_name))
    
    # Print configuration
    config.print_config()
    
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
        qos_metrics=cfg['qos_metrics']
    )
    
    print(f"\nState dimension: {env.state_dim}")
    print(f"Action dimension: {env.action_dim}")
    
    # Create SAC agent
    agent = SAC(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        capacity=cfg['C'],
        lr_actor=cfg['lr_actor'],
        lr_critic=cfg['lr_critic'],
        gamma=cfg['gamma'],
        tau=cfg['tau'],
        device=cfg['device']
    )
    
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
    """
    Train SAC agent following Algorithm 2.
    
    Args:
        config: Configuration object with all parameters
    
    Algorithm 2: SAC Training for Multi-Slice Resource Allocation
    - Outer loop: E_max episodes
    - Inner loop: T_max DTIs per episode
    - Episodic learning: Reset() at start of each episode
    """
    
    # Create directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if config.training.experiment_name:
        run_name = config.training.experiment_name
    else:
        run_name = f'sac_K{config.env.K}_W{config.env.window_size}_{timestamp}'
    
    writer = SummaryWriter(os.path.join(config.training.log_dir, run_name))
    
    # Create environment
    env = NetworkEnvironment(
        K=config.env.K,
        C=config.env.C,
        N=config.env.N,
        thresholds=config.env.thresholds,
        lambda_reward=config.env.lambda_reward,
        window_size=config.env.window_size,
        traffic_profiles=config.env.traffic_profiles
    )
    
    print(config.summary())
    print(f"\nState dimension: {env.state_dim}")
    print(f"Action dimension: {env.action_dim}")
    
    # Create SAC agent
    agent = SAC(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        capacity=config.env.C,
        lr_actor=config.sac.lr_actor,
        lr_critic=config.sac.lr_critic,
        gamma=config.sac.gamma,
        tau=config.sac.tau,
        alpha=config.sac.alpha,
        device=config.sac.device
    )
    
    # Training statistics
    episode_rewards = []
    episode_betas = []
    episode_violations = []
    
    print(f"\nStarting Training: {config.training.num_episodes} episodes")
    print("=" * 70)
    
    # Algorithm 2: Outer loop (episodes)
    for episode in tqdm(range(1, config.training.num_episodes + 1), desc="Training"):
        
        # Reset environment (Algorithm 1: Reset())
        state = env.reset()
        
        episode_reward = 0
        episode_beta_sum = 0
        num_violations = 0
        
        # Algorithm 2: Inner loop (DTIs)
        for dti in range(config.env.max_dtis):
            
            # Select action from policy
            action = agent.select_action(state)
            
            # Environment step (Algorithm 1: Step(a))
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # Update statistics
            episode_reward += reward
            episode_beta_sum += info['beta']
            if info['constraint_violated']:
                num_violations += 1
            
            # Train agent (if enough data)
            if len(agent.replay_buffer) >= config.sac.min_buffer_size:
                metrics = agent.update(config.sac.batch_size)
                
                # Log training metrics
                if metrics and dti % 10 == 0:
                    step = episode * config.env.max_dtis + dti
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
        if episode % config.training.log_interval == 0:
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
            
            print(f"\nEpisode {episode}/{config.training.num_episodes}:")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Avg Beta: {avg_beta:.4f}")
            print(f"  Violations: {num_violations}")
            print(f"  Buffer: {len(agent.replay_buffer)}")
        
        # Save checkpoint
        if episode % config.training.save_interval == 0:
            save_path = os.path.join(config.training.save_dir, f'{run_name}_ep{episode}.pt')
            agent.save(save_path)
            print(f"  Saved checkpoint: {save_path}")
    
    # Final save
    final_path = os.path.join(config.training.save_dir, f'{run_name}_final.pt')
    agent.save(final_path)
    print(f"\nTraining complete! Final model saved: {final_path}")
    
    # Close writer
    writer.close()
    
    # Plot training curves
    plot_training_curves(episode_rewards, episode_betas, episode_violations, 
                         config.training.save_dir, run_name)
    
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

    """
    Train SAC agent following Algorithm 2.
    
    Algorithm 2: SAC Training for Multi-Slice Resource Allocation
    - Outer loop: E_max episodes
    - Inner loop: T_max DTIs per episode
    - Episodic learning: Reset() at start of each episode
    """
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f'sac_K{K}_W{window_size}_{timestamp}'
    writer = SummaryWriter(os.path.join(log_dir, run_name))
    
    # Create environment
    if thresholds is None:
        thresholds = [0.2] * K
    if traffic_profiles is None:
        traffic_profiles = ['uniform'] * K
    
    env = NetworkEnvironment(
        K=K,
        C=C,
        N=N,
        thresholds=thresholds,
        lambda_reward=lambda_reward,
        window_size=window_size,
        traffic_profiles=traffic_profiles
    )
    
    print(f"\nEnvironment Configuration:")
    print(f"  Slices (K): {K}")
    print(f"  Capacity (C): {C} RBs")
    print(f"  TTIs per DTI (N): {N}")
    print(f"  Thresholds: {thresholds}")
    print(f"  Lambda: {lambda_reward}")
    print(f"  Window size (W): {window_size} DTIs")
    print(f"  Traffic profiles: {traffic_profiles}")
    print(f"  State dim: {env.state_dim}")
    print(f"  Action dim: {env.action_dim}")
    
    # Create SAC agent
    agent = SAC(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        capacity=C,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        tau=tau,
        device=device
    )
    
    print(f"\nSAC Agent Configuration:")
    print(f"  Learning rates: π={lr_actor}, Q={lr_critic}")
    print(f"  Gamma: {gamma}")
    print(f"  Tau (soft update): {tau}")
    print(f"  Device: {device}")
    
    # Training statistics
    episode_rewards = []
    episode_betas = []
    episode_violations = []
    
    print(f"\nStarting Training: {num_episodes} episodes")
    print("=" * 70)
    
    # Algorithm 2: Outer loop (episodes)
    for episode in tqdm(range(1, num_episodes + 1), desc="Training"):
        
        # Reset environment (Algorithm 1: Reset())
        state = env.reset()
        
        episode_reward = 0
        episode_beta_sum = 0
        num_violations = 0
        
        # Algorithm 2: Inner loop (DTIs)
        for dti in range(max_dtis):
            
            # Select action from policy
            action = agent.select_action(state)
            
            # Environment step (Algorithm 1: Step(a))
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # Update statistics
            episode_reward += reward
            episode_beta_sum += info['beta']
            if info['constraint_violated']:
                num_violations += 1
            
            # Train agent (if enough data)
            if len(agent.replay_buffer) >= min_buffer_size:
                metrics = agent.update(batch_size)
                
                # Log training metrics
                if metrics and dti % 10 == 0:
                    step = episode * max_dtis + dti
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
        if episode % log_interval == 0:
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
            
            print(f"\nEpisode {episode}/{num_episodes}:")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Avg Beta: {avg_beta:.4f}")
            print(f"  Violations: {num_violations}")
            print(f"  Buffer: {len(agent.replay_buffer)}")
        
        # Save checkpoint
        if episode % save_interval == 0:
            save_path = os.path.join(save_dir, f'{run_name}_ep{episode}.pt')
            agent.save(save_path)
            print(f"  Saved checkpoint: {save_path}")
    
    # Final save
    final_path = os.path.join(save_dir, f'{run_name}_final.pt')
    agent.save(final_path)
    print(f"\nTraining complete! Final model saved: {final_path}")
    
    # Close writer
    writer.close()
    
    # Plot training curves
    plot_training_curves(episode_rewards, episode_betas, episode_violations, save_dir, run_name)
    
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
    # Example: Train SAC agent
    
    print("SAC Training for Multi-Slice Resource Allocation")
    print("Implementing Algorithm 2 from LaTeX")
    print("=" * 70)
    
    # Configuration
    config = {
        'K': 2,
        'C': 8,
        'N': 20,
        'thresholds': [0.2, 0.15],
        'lambda_reward': 0.5,
        'window_size': 5,
        'traffic_profiles': ['low', 'high'],
        
        'num_episodes': 1000,
        'max_dtis': 200,
        'batch_size': 256,
        'min_buffer_size': 1000,
        
        'lr_actor': 3e-4,
        'lr_critic': 3e-4,
        'gamma': 0.99,
        'tau': 0.005,
        
        'log_interval': 10,
        'save_interval': 100,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Train
    agent, env, rewards, betas = train_sac(**config)
    
    print("\nTraining Summary:")
    print(f"  Final 100-episode avg reward: {np.mean(rewards[-100:]):.2f}")
    print(f"  Final 100-episode avg beta: {np.mean(betas[-100:]):.4f}")
    print(f"  Best episode reward: {np.max(rewards):.2f}")
    print(f"  Best episode beta: {np.min(betas):.4f}")
