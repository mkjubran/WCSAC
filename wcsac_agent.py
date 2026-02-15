"""
Worst-Case SAC (WCSAC) Implementation for Multi-Slice Resource Allocation

Extends SAC with worst-case robustness by considering uncertainty sets
around the transition dynamics and rewards.

Key differences from SAC:
1. Adversarial value function that considers worst-case transitions
2. Robust policy optimization under uncertainty
3. Uncertainty set parameterization for traffic and QoS variations
4. Conservative Q-learning with pessimistic value estimates

Reference:
- "Worst-Case Soft Actor Critic for Safety-Constrained Reinforcement Learning"
- Applies robust RL to handle traffic uncertainty and QoS variations
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
from typing import Tuple, List, Dict


class ReplayBuffer:
    """Experience replay buffer for WCSAC"""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, 
             worst_case_reward=None, worst_case_next_state=None):
        """
        Store transition with optional worst-case scenarios.
        
        Args:
            worst_case_reward: Worst-case reward variant
            worst_case_next_state: Worst-case next state variant
        """
        self.buffer.append((
            state, action, reward, next_state, done,
            worst_case_reward, worst_case_next_state
        ))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones, wc_rewards, wc_next_states = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32).reshape(-1, 1),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32).reshape(-1, 1),
            np.array([r if r is not None else rewards[i] for i, r in enumerate(wc_rewards)], 
                    dtype=np.float32).reshape(-1, 1),
            np.array([s if s is not None else next_states[i] for i, s in enumerate(wc_next_states)],
                    dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    """
    Actor network with softmax parameterization.
    Same as SAC actor - outputs allocation that sums to C.
    """
    
    def __init__(self, state_dim: int, action_dim: int, capacity: int,
                 hidden_dims: List[int] = [256, 256]):
        super(Actor, self).__init__()
        
        self.capacity = capacity
        self.action_dim = action_dim
        
        # Build network
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.shared = nn.Sequential(*layers)
        self.logits = nn.Linear(prev_dim, action_dim)
    
    def forward(self, state):
        """Forward pass: state → logits → continuous action"""
        x = self.shared(state)
        z = self.logits(x)
        action = self.capacity * F.softmax(z, dim=-1)
        return action


class RobustCritic(nn.Module):
    """
    Robust Critic network for worst-case value estimation.
    
    Outputs both nominal Q-value and worst-case Q-value.
    """
    
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: List[int] = [256, 256]):
        super(RobustCritic, self).__init__()
        
        # Nominal Q-network
        layers_nominal = []
        prev_dim = state_dim + action_dim
        
        for hidden_dim in hidden_dims:
            layers_nominal.append(nn.Linear(prev_dim, hidden_dim))
            layers_nominal.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers_nominal.append(nn.Linear(prev_dim, 1))
        self.q_nominal = nn.Sequential(*layers_nominal)
        
        # Worst-case Q-network (shares features but separate head)
        self.shared = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )
        
        self.q_worst = nn.Linear(hidden_dims[1], 1)
        self.q_nominal_head = nn.Linear(hidden_dims[1], 1)
    
    def forward(self, state, action, return_worst_case=True):
        """
        Forward pass: (s, a) → Q_nominal(s,a), Q_worst(s,a)
        
        Args:
            return_worst_case: If True, return both nominal and worst-case Q
        """
        x = torch.cat([state, action], dim=-1)
        features = self.shared(x)
        
        q_nominal = self.q_nominal_head(features)
        
        if return_worst_case:
            q_worst = self.q_worst(features)
            return q_nominal, q_worst
        else:
            return q_nominal


class WCSAC:
    """
    Worst-Case Soft Actor-Critic agent.
    
    Extends SAC with robustness to worst-case scenarios in:
    - Traffic variations
    - QoS metric variations
    - Transition dynamics uncertainty
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        capacity: int,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        # WCSAC-specific parameters
        kappa: float = 0.5,  # Robustness parameter (0=SAC, 1=fully worst-case)
        uncertainty_radius: float = 0.1,  # Size of uncertainty set
        pessimism_penalty: float = 0.1,  # Penalty for pessimistic Q-learning
        device: str = 'cpu',
        seed: int = None,
        use_efficient_allocation: bool = False  # NEW: Enable K+1 allocation mode
    ):
        """
        Args:
            kappa: Robustness trade-off (0=nominal SAC, 1=worst-case)
            uncertainty_radius: Radius of uncertainty set for perturbations
            pessimism_penalty: Conservative penalty for Q-values
            use_efficient_allocation: If True, actor outputs K+1 actions (K slices + 1 null)
        """
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.capacity = capacity
        
        # Efficient allocation mode
        self.use_efficient_allocation = use_efficient_allocation
        self.num_slices = action_dim  # K (actual number of slices)
        
        if use_efficient_allocation:
            # Actor outputs K+1 (K slices + 1 "null slice" for unused capacity)
            self.actor_action_dim = action_dim + 1
            self.action_dim = action_dim  # Critics still see K actions
        else:
            # Standard mode: Actor outputs K (all capacity must be allocated)
            self.actor_action_dim = action_dim
            self.action_dim = action_dim
        
        # WCSAC parameters
        self.kappa = kappa
        self.uncertainty_radius = uncertainty_radius
        self.pessimism_penalty = pessimism_penalty
        
        # Set seed
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        # Networks
        # Actor outputs actor_action_dim (K or K+1), critics see action_dim (K)
        self.actor = Actor(state_dim, self.actor_action_dim, capacity).to(self.device)
        
        # Robust critics (2 for double Q-learning)
        self.critic1 = RobustCritic(state_dim, self.action_dim).to(self.device)
        self.critic2 = RobustCritic(state_dim, self.action_dim).to(self.device)
        
        # Target critics
        self.critic1_target = RobustCritic(state_dim, self.action_dim).to(self.device)
        self.critic2_target = RobustCritic(state_dim, self.action_dim).to(self.device)
        
        # Initialize targets
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Freeze targets
        for param in self.critic1_target.parameters():
            param.requires_grad = False
        for param in self.critic2_target.parameters():
            param.requires_grad = False
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr_critic)
        
        # Temperature
        self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float32,
                                     requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
        self.target_entropy = -action_dim
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # Training stats
        self.train_step = 0
    
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        """
        Select action given state.
        
        Returns:
            action: K actions for slices (discards "null slice" if efficient mode)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            actor_output = self.actor(state_tensor)  # Shape: (1, K) or (1, K+1)
            
            if self.use_efficient_allocation:
                # Extract first K actions (discard last "null slice" action)
                action = actor_output[:, :self.num_slices]
            else:
                # Standard mode: return all K actions
                action = actor_output
        
        return action.cpu().numpy()[0]
    
    @property
    def alpha(self):
        """Current temperature value"""
        return self.log_alpha.exp().item()
    
    def _compute_worst_case_target(self, next_states, wc_next_states, 
                                   next_actions, rewards, wc_rewards, dones):
        """
        Compute worst-case target Q-value.
        
        Uses min over uncertainty set:
        Q_target = r_worst + γ * min(Q(s_worst, a'), Q(s_nominal, a'))
        """
        with torch.no_grad():
            # Get Q-values for nominal next states
            q1_next_nominal, q1_next_worst = self.critic1_target(next_states, next_actions)
            q2_next_nominal, q2_next_worst = self.critic2_target(next_states, next_actions)
            
            # Get Q-values for worst-case next states
            q1_wc_nominal, q1_wc_worst = self.critic1_target(wc_next_states, next_actions)
            q2_wc_nominal, q2_wc_worst = self.critic2_target(wc_next_states, next_actions)
            
            # Take minimum Q (double Q-learning)
            q_next_nominal = torch.min(q1_next_nominal, q2_next_nominal)
            q_next_worst_case = torch.min(q1_next_worst, q2_next_worst)
            q_wc_nominal = torch.min(q1_wc_nominal, q2_wc_nominal)
            q_wc_worst = torch.min(q1_wc_worst, q2_wc_worst)
            
            # Robust aggregation: interpolate between nominal and worst-case
            q_next_robust = (1 - self.kappa) * q_next_nominal + self.kappa * q_next_worst_case
            q_wc_robust = (1 - self.kappa) * q_wc_nominal + self.kappa * q_wc_worst
            
            # Take minimum over uncertainty set
            q_next = torch.min(q_next_robust, q_wc_robust)
            
            # Robust reward: weighted combination
            r_robust = (1 - self.kappa) * rewards + self.kappa * wc_rewards
            
            # Conservative penalty
            q_next = q_next - self.pessimism_penalty
            
            # Compute target
            q_target = r_robust + self.gamma * (1 - dones) * q_next
        
        return q_target
    
    def update(self, batch_size: int = 256) -> dict:
        """
        Update actor and critic networks with worst-case robustness.
        """
        if len(self.replay_buffer) < batch_size:
            return {}
        
        # Sample batch
        states, actions, rewards, next_states, dones, wc_rewards, wc_next_states = \
            self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        wc_rewards = torch.FloatTensor(wc_rewards).to(self.device)
        wc_next_states = torch.FloatTensor(wc_next_states).to(self.device)
        
        # ============ Update Critics (Worst-Case) ============
        # Sample actions for next states
        next_actor_output = self.actor(next_states)  # Shape: (batch, K) or (batch, K+1)
        
        if self.use_efficient_allocation:
            # Extract first K actions (critics only see K actions)
            next_actions = next_actor_output[:, :self.num_slices]
        else:
            # Standard mode
            next_actions = next_actor_output
        
        # Compute worst-case target
        q_target = self._compute_worst_case_target(
            next_states, wc_next_states, next_actions, 
            rewards, wc_rewards, dones
        )
        
        # Current Q-values (both nominal and worst-case)
        q1_nominal, q1_worst = self.critic1(states, actions)
        q2_nominal, q2_worst = self.critic2(states, actions)
        
        # Robust Q-value: weighted combination
        q1 = (1 - self.kappa) * q1_nominal + self.kappa * q1_worst
        q2 = (1 - self.kappa) * q2_nominal + self.kappa * q2_worst
        
        # Critic losses
        critic1_loss = F.mse_loss(q1, q_target)
        critic2_loss = F.mse_loss(q2, q_target)
        
        # Update critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_optimizer.step()
        
        # Update critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_optimizer.step()
        
        # ============ Update Actor (Robust Policy) ============
        new_actor_output = self.actor(states)  # Shape: (batch, K) or (batch, K+1)
        
        if self.use_efficient_allocation:
            # Extract first K actions for critics
            new_actions = new_actor_output[:, :self.num_slices]
        else:
            # Standard mode
            new_actions = new_actor_output
        
        # Get Q-values
        q1_new_nominal, q1_new_worst = self.critic1(states, new_actions)
        q2_new_nominal, q2_new_worst = self.critic2(states, new_actions)
        
        # Robust Q: weighted combination
        q1_new = (1 - self.kappa) * q1_new_nominal + self.kappa * q1_new_worst
        q2_new = (1 - self.kappa) * q2_new_nominal + self.kappa * q2_new_worst
        
        # Take minimum (conservative)
        q_new = torch.min(q1_new, q2_new)
        
        # Actor loss: maximize robust Q
        actor_loss = -q_new.mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
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
            'q_nominal': q1_nominal.mean().item(),
            'q_worst_case': q1_worst.mean().item(),
            'q_robust': q1_new.mean().item(),
            'kappa': self.kappa,
        }
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network"""
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
            'kappa': self.kappa,
            'uncertainty_radius': self.uncertainty_radius,
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
        self.kappa = checkpoint['kappa']
        self.uncertainty_radius = checkpoint['uncertainty_radius']
