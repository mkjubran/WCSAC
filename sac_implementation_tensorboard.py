# =============================================================================
# SAC (Soft Actor-Critic) Implementation - Detailed Explanations
# =============================================================================
"""
WHAT IS SAC?
------------
SAC (Soft Actor-Critic) is a state-of-the-art off-policy reinforcement learning
algorithm that maximizes both expected return and entropy.

ANALOGY: 
Imagine learning to play basketball. Regular RL learns one specific way to shoot.
SAC learns multiple ways to make a basket, making it more adaptable and robust.

KEY INNOVATION:
Maximum Entropy RL - Encourages the agent to be as random as possible while
still achieving high rewards. This leads to more robust and exploratory policies.

MATHEMATICAL FORMULATION:
-------------------------
Objective: max E[Σ γ^t (r_t + α H(π))]

Where:
- r_t: Reward at time t
- γ: Discount factor
- α: Temperature parameter (balances exploration vs exploitation)
- H(π): Entropy of policy (measures randomness)

WHY IS THIS USEFUL?
-------------------
1. Sample Efficiency: Off-policy learning reuses past experiences
2. Robustness: Maximum entropy leads to policies that work in varied conditions
3. Exploration: Automatic exploration through entropy maximization
4. Stability: More stable than on-policy methods like PPO
5. Continuous Control: Excellent for robotics and control tasks
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import random
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import clear_output
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# Create directories
FIGURE_DIR = 'sac_figures'
TENSORBOARD_DIR = 'sac_tensorboard'
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)

print(f"📁 Figures will be saved to: {FIGURE_DIR}/")
print(f"📊 TensorBoard logs: {TENSORBOARD_DIR}/")
print(f"   To view: tensorboard --logdir={TENSORBOARD_DIR}")

# Device setup
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("⚠️  Using CPU")

# =============================================================================
# Replay Buffer
# =============================================================================
class ReplayBuffer:
    """
    Experience Replay Buffer for Off-Policy Learning
    
    Stores: (state, action, reward, next_state, done)
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Store a transition"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample random batch"""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward),
                np.array(next_state), np.array(done))
    
    def __len__(self):
        return len(self.buffer)

# =============================================================================
# Q-Network (Critic)
# =============================================================================
class QNetwork(nn.Module):
    """
    Twin Q-Networks for value estimation
    
    Uses two Q-networks (Q1, Q2) to reduce overestimation bias.
    Takes minimum of both estimates for conservative value estimation.
    """
    def __init__(self, num_inputs, num_actions, hidden_size=256):
        super(QNetwork, self).__init__()
        
        # Q1 Network
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        
        # Q2 Network
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear5 = nn.Linear(hidden_size, hidden_size)
        self.linear6 = nn.Linear(hidden_size, 1)
        
    def forward(self, state, action):
        xu = torch.cat([state, action], dim=1)
        
        # Q1
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)
        
        # Q2
        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)
        
        return x1, x2

# =============================================================================
# Gaussian Policy (Actor)
# =============================================================================
class GaussianPolicy(nn.Module):
    """
    Stochastic policy that outputs Gaussian distribution
    
    Outputs mean and log_std for action distribution.
    Uses reparameterization trick for backpropagation through sampling.
    """
    def __init__(self, num_inputs, num_outputs, hidden_size=256, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        
        self.mean_linear = nn.Linear(hidden_size, num_outputs)
        self.log_std_linear = nn.Linear(hidden_size, num_outputs)
        
        # Action scaling
        if action_space is None:
            self.action_scale = torch.tensor(1.0).to(device)
            self.action_bias = torch.tensor(0.0).to(device)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.0).to(device)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.0).to(device)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std
    
    def sample(self, state):
        """
        Sample action using reparameterization trick
        Returns: action, log_prob, mean_action
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        
        # Reparameterization trick
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        
        # Log probability with tanh correction
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        
        return action, log_prob, mean_action

# =============================================================================
# SAC Agent
# =============================================================================
class SAC:
    """
    Soft Actor-Critic Agent
    
    Components:
    - Policy (Actor): Gaussian policy network
    - Q-Networks (Critics): Twin Q-networks for value estimation
    - Target Q-Networks: Slowly updated targets for stability
    - Automatic Entropy Tuning: Adjusts exploration dynamically
    """
    def __init__(self, num_inputs, action_space, args):
        self.gamma = args['gamma']
        self.tau = args['tau']
        self.alpha = args['alpha']
        self.target_update_interval = args['target_update_interval']
        self.automatic_entropy_tuning = args['automatic_entropy_tuning']
        
        num_actions = action_space.shape[0]
        
        # Critic networks
        self.critic = QNetwork(num_inputs, num_actions, 
                               args['hidden_size']).to(device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=args['lr'])
        
        self.critic_target = QNetwork(num_inputs, num_actions,
                                      args['hidden_size']).to(device)
        self.hard_update(self.critic_target, self.critic)
        
        # Policy network
        self.policy = GaussianPolicy(num_inputs, num_actions,
                                     args['hidden_size'], action_space).to(device)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=args['lr'])
        
        # Automatic entropy tuning
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(
                torch.Tensor(action_space.shape).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=args['lr'])
        
        self.updates = 0
    
    def select_action(self, state, evaluate=False):
        """Select action from policy"""
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        if evaluate:
            _, _, action = self.policy.sample(state)
        else:
            action, _, _ = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]
    
    def update_parameters(self, memory, batch_size):
        """Update all networks"""
        # Sample batch
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = \
            memory.sample(batch_size)
        
        state_batch = torch.FloatTensor(state_batch).to(device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(device)
        action_batch = torch.FloatTensor(action_batch).to(device)
        reward_batch = torch.FloatTensor(reward_batch).to(device).unsqueeze(1)
        done_batch = torch.FloatTensor(done_batch).to(device).unsqueeze(1)
        
        # Update Critic
        with torch.no_grad():
            next_action, next_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(
                next_state_batch, next_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - \
                                 self.alpha * next_log_pi
            next_q_value = reward_batch + (1 - done_batch) * self.gamma * min_qf_next_target
        
        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss
        
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()
        
        # Update Policy
        pi, log_pi, _ = self.policy.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        
        # Update temperature
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * 
                          (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()
        
        # Soft update target networks
        if self.updates % self.target_update_interval == 0:
            self.soft_update(self.critic_target, self.critic, self.tau)
        
        self.updates += 1
        
        return qf1_loss.item(), qf2_loss.item(), policy_loss.item()
    
    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau)
    
    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

# =============================================================================
# Training Functions
# =============================================================================
def plot_results(episode, rewards, losses, save_path=None):
    """Plot training progress"""
    fig = plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.title('Episode Rewards', fontsize=14, fontweight='bold')
    plt.plot(rewards, linewidth=2, color='blue', alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)
    
    if losses:
        plt.subplot(132)
        plt.title('Critic Loss', fontsize=14, fontweight='bold')
        plt.plot([l[0] for l in losses], linewidth=1.5, color='red', alpha=0.7)
        plt.xlabel('Update')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(133)
        plt.title('Policy Loss', fontsize=14, fontweight='bold')
        plt.plot([l[2] for l in losses], linewidth=1.5, color='green', alpha=0.7)
        plt.xlabel('Update')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   💾 Saved: {save_path}")
        plt.close(fig)
    else:
        plt.show()

def train_sac(env_name, args):
    """Main training loop for SAC"""
    # TensorBoard setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(TENSORBOARD_DIR, f'run_{timestamp}')
    writer = SummaryWriter(log_dir=log_dir)
    
    print(f"\n📊 TensorBoard: {log_dir}")
    print(f"   tensorboard --logdir={TENSORBOARD_DIR}\n")
    
    writer.add_text('Hyperparameters',
                    '\n'.join([f'{k}: {v}' for k, v in args.items()]))
    
    # Environment
    env = gym.make(env_name)
    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])
    random.seed(args['seed'])
    
    # Agent
    agent = SAC(env.observation_space.shape[0], env.action_space, args)
    memory = ReplayBuffer(args['replay_size'])
    
    # Training metrics
    episode_rewards = []
    losses = []
    total_numsteps = 0
    update_count = 0
    
    print(f"Training on {env_name}")
    print(f"State dim: {env.observation_space.shape[0]}")
    print(f"Action dim: {env.action_space.shape[0]}")
    print("=" * 60)
    
    # Training loop
    for i_episode in range(args['max_episodes']):
        episode_reward = 0
        episode_steps = 0
        done = False
        state, _ = env.reset()
        
        while not done:
            # Select action
            if total_numsteps < args['start_steps']:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, evaluate=False)
            
            # Environment step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward
            
            # Store transition
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)
            memory.push(state, action, reward, next_state, mask)
            
            state = next_state
            
            # Update networks
            if len(memory) > args['batch_size']:
                for _ in range(args['updates_per_step']):
                    loss = agent.update_parameters(memory, args['batch_size'])
                    losses.append(loss)
                    
                    # Log to TensorBoard
                    if update_count % 10 == 0:
                        writer.add_scalar('Loss/Critic_Q1', loss[0], update_count)
                        writer.add_scalar('Loss/Critic_Q2', loss[1], update_count)
                        writer.add_scalar('Loss/Policy', loss[2], update_count)
                        
                        if agent.automatic_entropy_tuning:
                            writer.add_scalar('Temperature/Alpha',
                                            agent.alpha.item(), update_count)
                    
                    update_count += 1
        
        # Episode complete
        episode_rewards.append(episode_reward)
        
        # Log episode metrics
        writer.add_scalar('Episode/Reward', episode_reward, i_episode)
        writer.add_scalar('Episode/Steps', episode_steps, i_episode)
        writer.add_scalar('Episode/Total_Steps', total_numsteps, i_episode)
        
        if len(episode_rewards) >= 10:
            avg_10 = np.mean(episode_rewards[-10:])
            writer.add_scalar('Episode/Avg_Reward_10', avg_10, i_episode)
        if len(episode_rewards) >= 100:
            avg_100 = np.mean(episode_rewards[-100:])
            writer.add_scalar('Episode/Avg_Reward_100', avg_100, i_episode)
        
        writer.add_scalar('Training/Replay_Buffer_Size', len(memory), i_episode)
        
        # Print progress
        if i_episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {i_episode:4d} | "
                  f"Steps: {total_numsteps:6d} | "
                  f"Reward: {episode_reward:7.2f} | "
                  f"Avg(10): {avg_reward:7.2f}")
            
            if i_episode % 50 == 0 and losses:
                save_path = os.path.join(FIGURE_DIR, f'training_ep_{i_episode:04d}.png')
                plot_results(i_episode, episode_rewards, losses, save_path)
        
        # Early stopping
        if len(episode_rewards) >= 100 and \
           np.mean(episode_rewards[-100:]) > args['target_reward']:
            print(f"\n🎉 Solved in {i_episode} episodes!")
            writer.add_text('Training', f'Solved in {i_episode} episodes!')
            break
    
    writer.close()
    env.close()
    
    # Save final plot
    if episode_rewards:
        final_path = os.path.join(FIGURE_DIR, 'training_final.png')
        plot_results(len(episode_rewards), episode_rewards, losses, final_path)
    
    return agent, episode_rewards

# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    args = {
        'env_name': 'Pendulum-v1',
        'gamma': 0.99,
        'tau': 0.005,
        'lr': 0.0003,
        'alpha': 0.2,
        'automatic_entropy_tuning': True,
        'seed': 42,
        'batch_size': 256,
        'max_episodes': 1000,
        'hidden_size': 256,
        'start_steps': 1000,
        'updates_per_step': 1,
        'target_update_interval': 1,
        'replay_size': 1000000,
        'target_reward': -200,
    }
    
    # CPU optimization
    if device.type == 'cpu':
        print("\n⚡ CPU optimization")
        args['batch_size'] = 128
        args['hidden_size'] = 128
        args['max_episodes'] = 500
    
    print("=" * 60)
    print("SAC: Soft Actor-Critic")
    print("=" * 60)
    print(f"Environment: {args['env_name']}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Train
    agent, rewards = train_sac(args['env_name'], args)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print("=" * 60)
    print(f"Episodes: {len(rewards)}")
    print(f"Final avg (last 100): {np.mean(rewards[-100:]):.2f}")
    print("=" * 60)
    
    # Test
    print("\n" + "=" * 60)
    print("TESTING AGENT")
    print("=" * 60)
    
    env = gym.make(args['env_name'], render_mode='human')
    test_rewards = []
    
    for test_ep in range(5):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.select_action(state, evaluate=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        test_rewards.append(total_reward)
        print(f"Test {test_ep + 1}: {total_reward:.2f}")
    
    print(f"\n✅ Avg test reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    env.close()
    
    # Summary figure
    fig = plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.plot(rewards, alpha=0.5, color='blue')
    if len(rewards) >= 10:
        moving_avg = np.convolve(rewards, np.ones(10)/10, mode='valid')
        plt.plot(range(9, len(rewards)), moving_avg, 'r-', linewidth=2)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(132)
    plt.hist(rewards, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2)
    plt.title('Reward Distribution')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(133)
    stats_text = f"""
    STATISTICS
    {'='*25}
    Episodes: {len(rewards)}
    Mean: {np.mean(rewards):.2f}
    Std: {np.std(rewards):.2f}
    Min: {np.min(rewards):.2f}
    Max: {np.max(rewards):.2f}
    
    Final 100: {np.mean(rewards[-100:]):.2f}
    
    Test: {np.mean(test_rewards):.2f}
    """
    plt.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.axis('off')
    
    plt.tight_layout()
    summary_path = os.path.join(FIGURE_DIR, 'summary.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Saved summary: {summary_path}")
    plt.close()
    
    print("\n✅ Complete!")
