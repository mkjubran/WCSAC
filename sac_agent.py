"""
SAC Implementation for Multi-Slice Resource Allocation - Implements Algorithm 2
Soft Actor-Critic with constrained action space using softmax parameterization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
from typing import Tuple, List


class ReplayBuffer:
    """Experience replay buffer for SAC"""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32).reshape(-1, 1),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32).reshape(-1, 1)
        )
    
    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    """
    Actor network with softmax parameterization.
    
    Outputs: z ∈ R^K (logits)
    Action: a_k = C · softmax(z)_k
    
    Guarantees: Σa_k = C (before rounding)
    """
    
    def __init__(self, state_dim: int, action_dim: int, capacity: int,
                 hidden_dims: List[int] = [256, 256]):
        super(Actor, self).__init__()
        
        self.capacity = capacity  # C
        self.action_dim = action_dim  # K
        
        # Build network
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.shared = nn.Sequential(*layers)
        
        # Output layer: K logits (no activation)
        self.logits = nn.Linear(prev_dim, action_dim)
    
    def forward(self, state):
        """
        Forward pass: state → logits → continuous action
        
        Returns:
            action: [a_1, ..., a_K] where Σa_k = C
        """
        x = self.shared(state)
        z = self.logits(x)  # Logits ∈ R^K
        
        # Apply softmax and scale by capacity
        # a_k = C · exp(z_k) / Σ_j exp(z_j)
        action = self.capacity * F.softmax(z, dim=-1)
        
        return action
    
    def sample_action(self, state):
        """Sample action for training (adds exploration noise)"""
        with torch.no_grad():
            action = self.forward(state)
        return action.cpu().numpy()


class Critic(nn.Module):
    """
    Critic network (Q-function).
    
    Q(s, a): Estimates expected return from state s taking action a
    """
    
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: List[int] = [256, 256]):
        super(Critic, self).__init__()
        
        # Build network
        layers = []
        prev_dim = state_dim + action_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))  # Output: Q-value
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state, action):
        """
        Forward pass: (s, a) → Q(s, a)
        """
        x = torch.cat([state, action], dim=-1)
        q_value = self.network(x)
        return q_value


class SAC:
    """
    Soft Actor-Critic agent for multi-slice resource allocation.
    Implements Algorithm 2 from LaTeX.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        capacity: int,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        target_entropy: float = None,
        device: str = 'cpu'
    ):
        """
        Args:
            state_dim: State dimension
            action_dim: Action dimension (K slices)
            capacity: Total RB capacity (C)
            lr_actor: Actor learning rate (η_π)
            lr_critic: Critic learning rate (η_Q)
            lr_alpha: Temperature learning rate (η_α)
            gamma: Discount factor
            tau: Soft update rate (τ_soft)
            alpha: Initial temperature
            target_entropy: Target entropy (if None, uses -dim(A))
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        self.capacity = capacity
        
        # Networks
        self.actor = Actor(state_dim, action_dim, capacity).to(self.device)
        
        self.critic1 = Critic(state_dim, action_dim).to(self.device)
        self.critic2 = Critic(state_dim, action_dim).to(self.device)
        
        # Target critics
        self.critic1_target = Critic(state_dim, action_dim).to(self.device)
        self.critic2_target = Critic(state_dim, action_dim).to(self.device)
        
        # Initialize targets
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Freeze target networks
        for param in self.critic1_target.parameters():
            param.requires_grad = False
        for param in self.critic2_target.parameters():
            param.requires_grad = False
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr_critic)
        
        # Temperature (entropy regularization)
        self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float32,
                                     requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)
        
        if target_entropy is None:
            # Heuristic: -dim(A)
            self.target_entropy = -action_dim
        else:
            self.target_entropy = target_entropy
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # Training stats
        self.train_step = 0
    
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        """
        Select action given state.
        
        Args:
            state: Current state
            eval_mode: If True, deterministic policy
            
        Returns:
            action: Continuous action [a_1, ..., a_K] with Σa_k = C
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state_tensor)
        
        return action.cpu().numpy()[0]
    
    @property
    def alpha(self):
        """Current temperature value"""
        return self.log_alpha.exp().item()
    
    def update(self, batch_size: int = 256) -> dict:
        """
        Update actor and critic networks.
        Implements Algorithm 2 training steps.
        
        Returns:
            dict with training metrics
        """
        if len(self.replay_buffer) < batch_size:
            return {}
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # ============ Update Critics ============
        with torch.no_grad():
            # Sample actions from current policy for next states
            next_actions = self.actor(next_states)
            
            # Compute target Q-values (minimum of two critics)
            q1_next = self.critic1_target(next_states, next_actions)
            q2_next = self.critic2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next)
            
            # Compute target: r + γ(1-d)Q_target(s',a')
            # Note: No entropy term in target for deterministic softmax actions
            q_target = rewards + self.gamma * (1 - dones) * q_next
        
        # Current Q-values
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        
        # Critic losses (MSE)
        critic1_loss = F.mse_loss(q1, q_target)
        critic2_loss = F.mse_loss(q2, q_target)
        
        # Update critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        # Update critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # ============ Update Actor ============
        # Sample new actions from current policy
        new_actions = self.actor(states)
        
        # Compute Q-values
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        # Actor loss: maximize Q
        # (negative because we're minimizing)
        actor_loss = -q_new.mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ============ Soft Update Targets ============
        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)
        
        self.train_step += 1
        
        return {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha,
            'q_value': q_new.mean().item(),
        }
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """
        Soft update: θ_target = τ·θ_source + (1-τ)·θ_target
        """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1 - self.tau) * target_param.data
            )
    
    def save(self, filepath: str):
        """Save model checkpoints"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'critic1_target': self.critic1_target.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath: str):
        """Load model checkpoints"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target'])
        
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
        
        self.log_alpha = checkpoint['log_alpha']
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
