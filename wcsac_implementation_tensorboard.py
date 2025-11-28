# =============================================================================
# WCSAC (Worst-Case Soft Actor-Critic) Implementation - Detailed Explanations
# =============================================================================
"""
WHAT IS WCSAC?
--------------
WCSAC is a robust reinforcement learning algorithm that trains an agent (protagonist)
to perform well even in the worst-case scenarios. It does this by simultaneously
training an adversary that tries to make the agent fail.

ANALOGY: 
Imagine learning to drive in various weather conditions. Regular RL is like
practicing only on sunny days. WCSAC is like having an instructor who throws
random challenges at you (rain, snow, strong winds) so you become a more
robust driver.

KEY INNOVATION:
Instead of optimizing for average performance, WCSAC optimizes for worst-case
performance by training against an intelligent adversary.

MATHEMATICAL FORMULATION:
-------------------------
Standard RL: max E[Σ rewards]
WCSAC: max min E[Σ rewards under disturbances]
       protagonist adversary

This is called a minimax optimization or two-player zero-sum game.

WHY IS THIS USEFUL?
-------------------
1. Safety-Critical Systems: Autonomous vehicles, medical robots
2. Real-World Deployment: Handles model uncertainty and environmental changes
3. Adversarial Robustness: Resistant to attacks and perturbations
4. Generalization: Performs better on unseen scenarios
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
import matplotlib.pyplot as plt
from IPython.display import clear_output
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# Create directories for saving figures and tensorboard logs
FIGURE_DIR = 'wcsac_figures'
TENSORBOARD_DIR = 'wcsac_tensorboard'
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)

print(f"📁 Figures will be saved to: {FIGURE_DIR}/")
print(f"📊 TensorBoard logs will be saved to: {TENSORBOARD_DIR}/")
print(f"   To view TensorBoard, run: tensorboard --logdir={TENSORBOARD_DIR}")

# ===== GPU/CPU Device Setup =====
# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    device = torch.device("cpu")
    print("⚠️  GPU not available, using CPU")
    print("   For faster training, install CUDA-enabled PyTorch:")
    print("   pip install torch --index-url https://download.pytorch.org/whl/cu118")

# Optional: Force GPU usage (will crash if GPU not available)
# device = torch.device("cuda")  # Uncomment to force GPU

# =============================================================================
# Replay Buffer
# =============================================================================
class ReplayBuffer:
    """
    Experience Replay Buffer
    ========================
    
    PURPOSE:
    --------
    Stores past experiences (transitions) for training. This is crucial for
    off-policy learning where we can learn from any past experience, not just
    recent ones.
    
    WHY REPLAY BUFFER?
    ------------------
    1. Sample Efficiency: Reuse past experiences multiple times
    2. Break Correlations: Random sampling breaks temporal correlations
    3. Stabilize Learning: Smooths out the training process
    
    WHAT WE STORE:
    --------------
    Each transition contains:
    - state: Where we were (e.g., robot position)
    - action: What we decided to do (e.g., move forward)
    - disturbance: What adversary did (e.g., push robot sideways)
    - reward: What we got (e.g., +1 for progress, -10 for falling)
    - next_state: Where we ended up
    - done: Whether episode ended
    
    CAPACITY:
    ---------
    Using deque with maxlen automatically removes oldest experiences when full.
    This keeps memory usage bounded and ensures we learn from recent experiences.
    """
    def __init__(self, capacity):
        """
        Initialize replay buffer
        
        Args:
            capacity: Maximum number of transitions to store (e.g., 1,000,000)
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, disturbance, reward, next_state, done):
        """
        Store a transition in the buffer
        
        IMPORTANT: We store the disturbance separately because we need to
        learn from it. The adversary needs to know what disturbances led
        to what outcomes.
        """
        self.buffer.append((state, action, disturbance, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Sample a random batch of transitions
        
        WHY RANDOM?
        -----------
        Random sampling breaks correlations between consecutive experiences.
        If we always trained on sequential experiences, the network would
        overfit to specific sequences rather than learning general patterns.
        
        Returns:
            Tuple of numpy arrays: (states, actions, disturbances, rewards, 
                                   next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, disturbance, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), np.array(disturbance),
                np.array(reward), np.array(next_state), np.array(done))
    
    def __len__(self):
        """Return current number of transitions in buffer"""
        return len(self.buffer)

# =============================================================================
# Neural Network Architectures
# =============================================================================

class QNetwork(nn.Module):
    """
    Q-Network (Critic) - Value Estimator
    =====================================
    
    WHAT IS A Q-NETWORK?
    --------------------
    A Q-network estimates Q(s,a): "How good is action 'a' in state 's'?"
    
    ANALOGY:
    Think of Q-network as an experienced coach watching you play. Given your
    current position (state) and a potential move (action), the coach estimates
    how many points you'll likely score if you make that move.
    
    WHY TWO Q-NETWORKS (Q1 and Q2)?
    --------------------------------
    This is called "Double Q-Learning" or "Clipped Double Q-Learning"
    
    PROBLEM: Single Q-networks tend to overestimate values because they always
    pick the maximum. This is like a student who always guesses the highest
    possible score - they'll be wrong and overly optimistic.
    
    SOLUTION: Use two Q-networks and take the minimum of their estimates.
    This reduces overestimation bias:
    - Q1 estimates: "I think this action is worth 10 points"
    - Q2 estimates: "I think this action is worth 8 points"
    - We use: min(10, 8) = 8 (conservative estimate)
    
    ARCHITECTURE:
    -------------
    Input: Concatenated [state, action]
    Example: [robot_x, robot_y, robot_angle, motor_speed] 
             → Hidden layers with ReLU activation
             → Output: Single value (Q-value)
    
    WHY CONCATENATE STATE AND ACTION?
    ----------------------------------
    We need to evaluate specific (state, action) pairs. Different actions
    in the same state have different values.
    Example: In chess, moving your queen vs moving a pawn from the same
    board position have very different values.
    """
    
    def __init__(self, num_inputs, num_actions, hidden_size=256):
        """
        Initialize Q-Network with dual architecture
        
        Args:
            num_inputs: Dimension of state space (e.g., 3 for Pendulum)
            num_actions: Dimension of action space (e.g., 1 for Pendulum)
            hidden_size: Number of neurons in hidden layers (256 is typical)
        """
        super(QNetwork, self).__init__()
        
        # ===== Q1 Network =====
        # Input dimension: num_inputs + num_actions
        # Example: 3 (state) + 1 (action) = 4
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)  # Output: single Q-value
        
        # ===== Q2 Network (independent architecture) =====
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear5 = nn.Linear(hidden_size, hidden_size)
        self.linear6 = nn.Linear(hidden_size, 1)  # Output: single Q-value
        
    def forward(self, state, action):
        """
        Forward pass through both Q-networks
        
        PROCESS:
        --------
        1. Concatenate state and action into single vector
        2. Pass through two independent networks
        3. Return both Q-values
        
        Args:
            state: Current state tensor [batch_size, num_inputs]
            action: Action tensor [batch_size, num_actions]
            
        Returns:
            (Q1_value, Q2_value): Two independent estimates
        """
        # Concatenate state and action along dimension 1 (features)
        # Shape: [batch_size, num_inputs + num_actions]
        xu = torch.cat([state, action], dim=1)
        
        # ===== Q1 Forward Pass =====
        x1 = F.relu(self.linear1(xu))  # First hidden layer with ReLU
        x1 = F.relu(self.linear2(x1))  # Second hidden layer with ReLU
        x1 = self.linear3(x1)          # Output layer (no activation)
        
        # ===== Q2 Forward Pass =====
        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)
        
        return x1, x2  # Return both estimates

class GaussianPolicy(nn.Module):
    """
    Gaussian Policy Network - Action/Disturbance Generator
    =======================================================
    
    WHAT IS A POLICY?
    -----------------
    A policy π(a|s) tells us what action to take in a given state.
    In continuous control, we output a probability distribution over actions.
    
    WHY GAUSSIAN (NORMAL) DISTRIBUTION?
    ------------------------------------
    1. Continuous Actions: Can represent any real-valued action
    2. Exploration: Natural randomness for trying new actions
    3. Differentiable: Can use gradient descent for optimization
    
    STOCHASTIC vs DETERMINISTIC:
    ----------------------------
    - Deterministic: Always output same action for same state
      Example: "Always turn 30 degrees left"
    - Stochastic: Sample from a distribution
      Example: "Turn left with mean 30° and std 5°" → might turn 28°, 32°, etc.
    
    WHY STOCHASTIC IS BETTER:
    -------------------------
    1. Exploration: Automatically tries variations
    2. Maximum Entropy: Learns multiple ways to solve task (more robust)
    3. Better Generalization: Doesn't overfit to specific trajectories
    
    NETWORK OUTPUT:
    ---------------
    Instead of outputting action directly, we output:
    1. Mean (μ): Center of the distribution
    2. Log Std (log σ): Spread of the distribution
    
    Then we sample: action ~ N(μ, σ²)
    
    REPARAMETERIZATION TRICK:
    --------------------------
    To make sampling differentiable:
    1. Sample ε ~ N(0, 1) (standard normal)
    2. Transform: x = μ + σ * ε
    3. Apply tanh to bound: y = tanh(x)
    4. Scale to action space: action = y * scale + bias
    
    This allows gradients to flow through the sampling operation!
    
    PROTAGONIST vs ADVERSARY:
    --------------------------
    Same architecture is used for both:
    - Protagonist: Outputs actions (what to do)
    - Adversary: Outputs disturbances (how to perturb actions)
    """
    
    def __init__(self, num_inputs, num_outputs, hidden_size=256, 
                 action_space=None, is_adversary=False):
        """
        Initialize Gaussian Policy Network
        
        Args:
            num_inputs: State dimension
            num_outputs: Action/disturbance dimension
            hidden_size: Hidden layer size
            action_space: Gym action space (for scaling)
            is_adversary: If True, this is adversary policy (centered at 0)
        """
        super(GaussianPolicy, self).__init__()
        
        # ===== Shared Feature Extractor =====
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        
        # ===== Separate Heads for Mean and Std =====
        # Why separate? Mean and std serve different purposes:
        # - Mean: Where to center the distribution
        # - Std: How much to explore around the mean
        self.mean_linear = nn.Linear(hidden_size, num_outputs)
        self.log_std_linear = nn.Linear(hidden_size, num_outputs)
        
        self.is_adversary = is_adversary
        
        # ===== Action Space Scaling =====
        # Real environments have bounded action spaces
        # Example: Motor speed can be [-2, 2], not unbounded
        if action_space is None:
            # Default: no scaling
            self.action_scale = torch.tensor(1.0).to(device)
            self.action_bias = torch.tensor(0.0).to(device)
        else:
            if is_adversary:
                # ADVERSARY: Disturbances are centered at 0
                # Budget determines maximum disturbance magnitude
                # Example: budget=0.5 → disturbances in [-0.5, 0.5]
                self.action_scale = torch.FloatTensor(
                    (action_space.high - action_space.low) / 2.0).to(device)
                self.action_bias = torch.tensor(0.0).to(device)
            else:
                # PROTAGONIST: Actions use full environment range
                # Example: action space [-2, 2]
                # Scale: (2 - (-2))/2 = 2
                # Bias: (2 + (-2))/2 = 0
                self.action_scale = torch.FloatTensor(
                    (action_space.high - action_space.low) / 2.0).to(device)
                self.action_bias = torch.FloatTensor(
                    (action_space.high + action_space.low) / 2.0).to(device)
        
    def forward(self, state):
        """
        Forward pass through policy network
        
        Args:
            state: Current state [batch_size, num_inputs]
            
        Returns:
            (mean, log_std): Parameters of action distribution
        """
        # Extract features from state
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        
        # Compute mean and log_std
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        
        # Clamp log_std to prevent numerical instability
        # Why clamp? Extremely high/low std causes:
        # - High std: Actions become too random (no learning)
        # - Low std: No exploration (gets stuck in local optima)
        # Range [-20, 2] means std ∈ [exp(-20), exp(2)] ≈ [2e-9, 7.4]
        log_std = torch.clamp(log_std, min=-20, max=2)
        
        return mean, log_std
    
    def sample(self, state):
        """
        Sample an action/disturbance using reparameterization trick
        
        DETAILED PROCESS:
        -----------------
        1. Get mean (μ) and log_std from network
        2. Compute std: σ = exp(log_std)
        3. Create Normal distribution: N(μ, σ²)
        4. Sample using reparameterization: x = μ + σ * ε, where ε ~ N(0,1)
        5. Apply tanh squashing: y = tanh(x) ∈ [-1, 1]
        6. Scale to action space: action = y * scale + bias
        7. Compute log probability (for policy gradient)
        
        WHY TANH SQUASHING?
        -------------------
        - Bounds output to [-1, 1] (then scaled to action bounds)
        - Smooth function (differentiable everywhere)
        - Natural exploration around mean
        
        LOG PROBABILITY WITH TANH CORRECTION:
        -------------------------------------
        When we apply tanh transformation, we need to correct the log probability.
        This is because we're changing variables in the probability distribution.
        
        Formula: log π(action|state) = log π(x|state) - log|dy/dx|
        Where dy/dx is the Jacobian of the tanh transformation
        
        Args:
            state: Current state tensor
            
        Returns:
            action: Sampled action [batch_size, num_actions]
            log_prob: Log probability of action [batch_size, 1]
            mean: Mean action (for evaluation) [batch_size, num_actions]
        """
        # Get distribution parameters
        mean, log_std = self.forward(state)
        std = log_std.exp()  # Convert log_std to std
        
        # Create normal distribution
        normal = Normal(mean, std)
        
        # ===== Reparameterization Trick =====
        # rsample() uses reparameterization (gradients flow through)
        # sample() doesn't (no gradients)
        x_t = normal.rsample()
        
        # ===== Tanh Squashing to [-1, 1] =====
        y_t = torch.tanh(x_t)
        
        # ===== Scale to Action Space =====
        # Example: y_t=0.5, scale=2, bias=0 → action=1.0
        action = y_t * self.action_scale + self.action_bias
        
        # ===== Compute Log Probability =====
        # Log probability of sampling x_t from normal distribution
        log_prob = normal.log_prob(x_t)
        
        # ===== Tanh Correction =====
        # Jacobian of tanh: d(tanh(x))/dx = 1 - tanh²(x)
        # We subtract log of Jacobian from log probability
        # Adding 1e-6 prevents log(0) when tanh²(x) = 1
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        
        # Sum log probabilities across action dimensions
        # Why sum? For multivariate actions, assuming independence
        log_prob = log_prob.sum(1, keepdim=True)
        
        # ===== Mean Action (for evaluation/testing) =====
        # When evaluating, we use deterministic policy (mean action)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        
        return action, log_prob, mean_action

# =============================================================================
# WCSAC Agent - The Core Algorithm
# =============================================================================
class WCSAC:
    """
    Worst-Case Soft Actor-Critic Agent
    ===================================
    
    CONCEPTUAL OVERVIEW:
    --------------------
    Imagine training a robot to walk. Standard RL trains it on flat ground.
    WCSAC trains it while someone is actively trying to push it over!
    
    The robot (protagonist) learns to walk robustly, and the adversary learns
    the best ways to make it fall. The result? A robot that can handle
    real-world disturbances like uneven terrain, wind, or unexpected bumps.
    
    TWO-PLAYER GAME:
    ----------------
    Player 1 (Protagonist): 
        - Goal: Maximize reward
        - Outputs: Actions
        - Learns: Robust policy that works under worst-case disturbances
    
    Player 2 (Adversary):
        - Goal: Minimize protagonist's reward (maximize negative reward)
        - Outputs: Disturbances within a budget
        - Learns: Worst-case disturbances to test protagonist
    
    MATHEMATICAL FORMULATION:
    -------------------------
    Protagonist objective: max E[Σ γ^t (r_t + α H(π_protagonist))]
    Adversary objective:   max E[Σ γ^t (-r_t + β H(π_adversary))]
    
    Where:
    - γ: Discount factor (how much we care about future rewards)
    - r_t: Reward at time t
    - α, β: Temperature parameters (balance exploitation vs exploration)
    - H(π): Entropy (measures randomness/exploration)
    
    KEY COMPONENTS:
    ---------------
    1. Protagonist Policy: π(a|s) - selects actions
    2. Adversary Policy: π_adv(δ|s) - selects disturbances
    3. Protagonist Critics: Q(s,a) - evaluates actions
    4. Adversary Critics: Q_adv(s,δ) - evaluates disturbances
    5. Target Networks: Slowly updated copies for stability
    
    TRAINING PROCESS (one iteration):
    ---------------------------------
    1. Protagonist selects action: a ~ π(·|s)
    2. Adversary adds disturbance: δ ~ π_adv(·|s)
    3. Environment receives: a' = clip(a + δ)
    4. Observe reward r and next state s'
    5. Store (s, a, δ, r, s') in replay buffer
    6. Sample batch from buffer
    7. Update protagonist critic using Bellman equation
    8. Update adversary critic using negative rewards
    9. Update protagonist policy to maximize Q - α*entropy
    10. Update adversary policy to maximize Q_adv - β*entropy
    11. Soft update target networks
    
    WHY THIS WORKS:
    ---------------
    - Protagonist learns to be robust because it's trained against intelligent
      disturbances, not just random noise
    - Adversary provides a curriculum: it starts simple and gets harder as
      the protagonist improves
    - The disturbance budget prevents adversary from being impossible to beat
    - Maximum entropy ensures both agents explore, leading to robust solutions
    """
    
    def __init__(self, num_inputs, action_space, args):
        """
        Initialize WCSAC agent with all networks and optimizers
        
        Args:
            num_inputs: State space dimension
            action_space: Gym action space object
            args: Dictionary of hyperparameters
        """
        # ===== Hyperparameters =====
        self.gamma = args['gamma']  # Discount factor (typically 0.99)
        self.tau = args['tau']      # Soft update rate (typically 0.005)
        self.alpha = args['alpha']  # Protagonist temperature
        self.beta = args['beta']    # Adversary temperature
        
        self.target_update_interval = args['target_update_interval']
        self.automatic_entropy_tuning = args['automatic_entropy_tuning']
        self.disturbance_budget = args['disturbance_budget']
        
        num_actions = action_space.shape[0]
        
        # =====================================================================
        # PROTAGONIST NETWORKS (learns optimal action policy)
        # =====================================================================
        
        # ----- Protagonist Critic (Q-Network) -----
        # Purpose: Estimate value of state-action pairs
        # Input: (state, action) → Output: Q-value
        self.critic = QNetwork(num_inputs, num_actions, 
                               args['hidden_size']).to(device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=args['lr'])
        
        # ----- Protagonist Target Critic -----
        # Purpose: Provide stable targets for training
        # Why? If we use same network for targets and predictions, training
        # becomes unstable (chasing a moving target)
        self.critic_target = QNetwork(num_inputs, num_actions, 
                                      args['hidden_size']).to(device)
        self.hard_update(self.critic_target, self.critic)  # Copy initial weights
        
        # ----- Protagonist Policy -----
        # Purpose: Decide what actions to take
        # Input: state → Output: Action distribution N(μ, σ²)
        self.policy = GaussianPolicy(num_inputs, num_actions, 
                                     args['hidden_size'], action_space,
                                     is_adversary=False).to(device)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=args['lr'])
        
        # =====================================================================
        # ADVERSARY NETWORKS (learns worst-case disturbances)
        # =====================================================================
        
        # ----- Adversary Critic (Q-Network) -----
        # Purpose: Estimate value of applying disturbances
        # Note: Adversary tries to MINIMIZE protagonist's reward
        # So adversary Q-value represents negative reward from protagonist's view
        self.adversary_critic = QNetwork(num_inputs, num_actions,
                                        args['hidden_size']).to(device)
        self.adversary_critic_optim = optim.Adam(
            self.adversary_critic.parameters(), lr=args['lr'])
        
        # ----- Adversary Target Critic -----
        self.adversary_critic_target = QNetwork(num_inputs, num_actions,
                                               args['hidden_size']).to(device)
        self.hard_update(self.adversary_critic_target, self.adversary_critic)
        
        # ----- Adversary Policy -----
        # Purpose: Decide what disturbances to apply
        # Input: state → Output: Disturbance distribution N(μ, σ²)
        # Constraint: Disturbances bounded by disturbance_budget
        
        # Create a custom action space for adversary with disturbance budget
        class DisturbanceSpace:
            """
            Defines the space of possible disturbances
            Symmetric around 0 with bounds determined by budget
            """
            def __init__(self, shape, budget):
                self.shape = shape
                self.high = np.ones(shape) * budget   # Maximum disturbance
                self.low = -np.ones(shape) * budget   # Minimum disturbance
        
        disturbance_space = DisturbanceSpace(
            action_space.shape, self.disturbance_budget)
        
        self.adversary_policy = GaussianPolicy(
            num_inputs, num_actions, args['hidden_size'], 
            disturbance_space, is_adversary=True).to(device)
        self.adversary_policy_optim = optim.Adam(
            self.adversary_policy.parameters(), lr=args['lr'])
        
        # =====================================================================
        # AUTOMATIC ENTROPY TUNING
        # =====================================================================
        # Purpose: Automatically adjust exploration vs exploitation balance
        # 
        # ENTROPY INTUITION:
        # ------------------
        # High entropy = More random = More exploration
        # Low entropy = More deterministic = More exploitation
        # 
        # We want to maximize reward AND entropy:
        # J = E[Σ rewards + α * entropy]
        # 
        # But what should α be? Instead of manually tuning, we learn it!
        # We set a target entropy and adjust α to achieve it.
        
        if self.automatic_entropy_tuning:
            # Target entropy heuristic: -dim(action_space)
            # Intuition: For 1D action, target entropy = -1
            # This encourages reasonable exploration without being too random
            self.target_entropy = -torch.prod(
                torch.Tensor(action_space.shape).to(device)).item()
            
            # Log of temperature (we optimize log(α) for numerical stability)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=args['lr'])
            
            # Same for adversary
            self.target_beta = -torch.prod(
                torch.Tensor(action_space.shape).to(device)).item()
            self.log_beta = torch.zeros(1, requires_grad=True, device=device)
            self.beta_optim = optim.Adam([self.log_beta], lr=args['lr'])
        
        self.updates = 0  # Track number of updates
    
    def select_action(self, state, evaluate=False, use_adversary=False):
        """
        Select action from protagonist or disturbance from adversary
        
        EVALUATION MODE vs TRAINING MODE:
        ----------------------------------
        Training: Sample from distribution (stochastic, explores)
        Evaluation: Use mean of distribution (deterministic, exploits)
        
        Args:
            state: Current state observation
            evaluate: If True, use deterministic policy (mean action)
            use_adversary: If True, sample from adversary instead of protagonist
            
        Returns:
            Action or disturbance as numpy array
        """
        # Convert state to tensor and add batch dimension
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        
        if use_adversary:
            # Select disturbance from adversary policy
            if evaluate:
                # Deterministic: use mean disturbance
                _, _, disturbance = self.adversary_policy.sample(state)
            else:
                # Stochastic: sample from distribution
                disturbance, _, _ = self.adversary_policy.sample(state)
            return disturbance.detach().cpu().numpy()[0]
        else:
            # Select action from protagonist policy
            if evaluate:
                # Deterministic: use mean action
                _, _, action = self.policy.sample(state)
            else:
                # Stochastic: sample from distribution
                action, _, _ = self.policy.sample(state)
            return action.detach().cpu().numpy()[0]
    
    def update_parameters(self, memory, batch_size):
        """
        Update both protagonist and adversary networks
        This is the core learning algorithm - updates happen here!
        
        OVERVIEW OF UPDATE PROCESS:
        ---------------------------
        1. Sample batch of experiences from replay buffer
        2. Update protagonist critic (Q-function)
        3. Update adversary critic (Q-function for disturbances)
        4. Update protagonist policy (action selection)
        5. Update adversary policy (disturbance selection)
        6. Update temperature parameters (if automatic tuning enabled)
        7. Soft update target networks
        
        Args:
            memory: Replay buffer containing past experiences
            batch_size: Number of experiences to sample for this update
            
        Returns:
            Tuple of losses for monitoring training progress
        """
        
        # =====================================================================
        # STEP 1: SAMPLE BATCH FROM REPLAY BUFFER
        # =====================================================================
        state_batch, action_batch, disturbance_batch, reward_batch, \
            next_state_batch, done_batch = memory.sample(batch_size)
        
        # Convert numpy arrays to PyTorch tensors
        state_batch = torch.FloatTensor(state_batch).to(device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(device)
        action_batch = torch.FloatTensor(action_batch).to(device)
        disturbance_batch = torch.FloatTensor(disturbance_batch).to(device)
        reward_batch = torch.FloatTensor(reward_batch).to(device).unsqueeze(1)
        done_batch = torch.FloatTensor(done_batch).to(device).unsqueeze(1)
        
        # =====================================================================
        # STEP 2: UPDATE PROTAGONIST CRITIC (Q-FUNCTION)
        # =====================================================================
        """
        GOAL: Learn to estimate Q(s,a) = expected return from taking action a in state s
        
        BELLMAN EQUATION:
        Q(s,a) = r + γ * E[Q(s',a') - α*log π(a'|s')]
        
        INTUITION:
        Current Q-value = immediate reward + discounted future value
        
        TARGET NETWORK:
        We use target network for stability. Otherwise, we're chasing a moving target.
        Think of it like: "I evaluate my current actions based on my past self's
        knowledge, not my constantly changing current knowledge"
        """
        
        with torch.no_grad():  # Don't compute gradients for target computation
            # Sample next action from protagonist policy
            # This is what protagonist would do in next state
            next_action, next_log_pi, _ = self.policy.sample(next_state_batch)
            
            # Sample worst-case disturbance for next state
            # This is what adversary would do to make life difficult
            next_disturbance, _, _ = self.adversary_policy.sample(next_state_batch)
            
            # Compute target Q-value using TARGET network (for stability)
            # Get both Q1 and Q2 estimates
            qf1_next_target, qf2_next_target = self.critic_target(
                next_state_batch, next_action)
            
            # Take minimum to reduce overestimation bias (clipped double Q-learning)
            # Why minimum? Conservative estimate prevents overoptimistic predictions
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            
            # Subtract entropy bonus: encourages exploration
            # If policy is very certain (low entropy), subtract less
            # If policy is very random (high entropy), subtract more
            min_qf_next_target = min_qf_next_target - self.alpha * next_log_pi
            
            # BELLMAN TARGET: r + γ * (1 - done) * Q_target(s', a')
            # (1 - done) zeroes out future value if episode ended
            next_q_value = reward_batch + (1 - done_batch) * self.gamma * min_qf_next_target
        
        # Get current Q-value estimates from CURRENT network
        qf1, qf2 = self.critic(state_batch, action_batch)
        
        # LOSS: Mean Squared Error between prediction and target
        # Goal: Make our predictions match the Bellman targets
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss
        
        # Backpropagation and optimization step
        self.critic_optim.zero_grad()  # Clear previous gradients
        qf_loss.backward()              # Compute gradients
        self.critic_optim.step()        # Update weights
        
        # =====================================================================
        # STEP 3: UPDATE ADVERSARY CRITIC (Q-FUNCTION FOR DISTURBANCES)
        # =====================================================================
        """
        GOAL: Learn to estimate Q_adv(s,δ) = expected negative return from 
              applying disturbance δ in state s
        
        KEY DIFFERENCE FROM PROTAGONIST:
        Adversary wants to MINIMIZE protagonist's reward, so it maximizes
        NEGATIVE reward. We train adversary critic to estimate negative rewards.
        
        INTUITION:
        Adversary learns: "If I apply this disturbance, how much can I reduce
        the protagonist's reward?"
        """
        
        with torch.no_grad():
            # Sample next disturbance from adversary policy
            next_disturbance_adv, next_log_pi_adv, _ = \
                self.adversary_policy.sample(next_state_batch)
            
            # Compute target Q-value for adversary using TARGET network
            adv_qf1_next_target, adv_qf2_next_target = \
                self.adversary_critic_target(next_state_batch, next_disturbance_adv)
            
            # Take minimum (conservative estimate)
            min_adv_qf_next_target = torch.min(
                adv_qf1_next_target, adv_qf2_next_target)
            
            # Subtract entropy bonus for adversary exploration
            min_adv_qf_next_target = min_adv_qf_next_target - self.beta * next_log_pi_adv
            
            # ADVERSARY BELLMAN TARGET: -r + γ * (1 - done) * Q_adv_target(s', δ')
            # Note the NEGATIVE reward! Adversary maximizes negative reward.
            next_adv_q_value = -reward_batch + (1 - done_batch) * \
                               self.gamma * min_adv_qf_next_target
        
        # Get current adversary Q-value estimates
        adv_qf1, adv_qf2 = self.adversary_critic(state_batch, disturbance_batch)
        
        # Adversary critic loss
        adv_qf1_loss = F.mse_loss(adv_qf1, next_adv_q_value)
        adv_qf2_loss = F.mse_loss(adv_qf2, next_adv_q_value)
        adv_qf_loss = adv_qf1_loss + adv_qf2_loss
        
        # Update adversary critic
        self.adversary_critic_optim.zero_grad()
        adv_qf_loss.backward()
        self.adversary_critic_optim.step()
        
        # =====================================================================
        # STEP 4: UPDATE PROTAGONIST POLICY
        # =====================================================================
        """
        GOAL: Improve policy to maximize Q-value (expected return)
        
        POLICY GRADIENT OBJECTIVE:
        J = E[Q(s,a) - α*log π(a|s)]
        
        We want to:
        1. Maximize Q-value (get high rewards)
        2. Maximize entropy (explore, be robust)
        
        INTUITION:
        "Choose actions that give high Q-values, but also maintain some
        randomness for exploration and robustness"
        
        WHY SAMPLE NEW ACTIONS?
        We need to evaluate how good our CURRENT policy is, not past actions.
        So we sample fresh actions from current policy and evaluate them.
        """
        
        # Sample actions from CURRENT policy
        pi, log_pi, _ = self.policy.sample(state_batch)
        
        # Evaluate these actions using Q-network
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        
        # POLICY LOSS: -E[Q(s,a) - α*log π(a|s)]
        # We want to maximize this, so we minimize the negative
        # Breaking it down:
        # - min_qf_pi: reward we expect from this action (maximize this)
        # - α * log_pi: entropy penalty (higher entropy = more exploration)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        
        # Update protagonist policy
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        
        # =====================================================================
        # STEP 5: UPDATE ADVERSARY POLICY
        # =====================================================================
        """
        GOAL: Improve adversary to apply worst-case disturbances
        
        ADVERSARY OBJECTIVE:
        J_adv = E[Q_adv(s,δ) - β*log π_adv(δ|s)]
        
        Remember: Q_adv represents negative reward from protagonist's view
        So maximizing Q_adv = minimizing protagonist's reward
        
        INTUITION:
        "Apply disturbances that minimize protagonist's reward, but maintain
        some randomness to test different attack strategies"
        
        CONSTRAINT:
        Disturbances are bounded by disturbance_budget (enforced in policy network)
        """
        
        # Sample disturbances from CURRENT adversary policy
        pi_adv, log_pi_adv, _ = self.adversary_policy.sample(state_batch)
        
        # Evaluate these disturbances using adversary Q-network
        adv_qf1_pi, adv_qf2_pi = self.adversary_critic(state_batch, pi_adv)
        min_adv_qf_pi = torch.min(adv_qf1_pi, adv_qf2_pi)
        
        # ADVERSARY POLICY LOSS: maximize Q_adv (which represents negative reward)
        adversary_policy_loss = ((self.beta * log_pi_adv) - min_adv_qf_pi).mean()
        
        # Update adversary policy
        self.adversary_policy_optim.zero_grad()
        adversary_policy_loss.backward()
        self.adversary_policy_optim.step()
        
        # =====================================================================
        # STEP 6: UPDATE TEMPERATURE PARAMETERS (AUTOMATIC ENTROPY TUNING)
        # =====================================================================
        """
        GOAL: Automatically adjust exploration vs exploitation balance
        
        TEMPERATURE PARAMETER (α, β):
        Controls how much we care about entropy (exploration)
        - High α: prioritize exploration (more random actions)
        - Low α: prioritize exploitation (more deterministic actions)
        
        AUTOMATIC TUNING:
        Instead of manually setting α, we learn it to maintain target entropy
        
        OBJECTIVE: Match current entropy to target entropy
        Loss = -E[α * (log π + H_target)]
        
        If current entropy < target: increase α (encourage more exploration)
        If current entropy > target: decrease α (allow more exploitation)
        
        WHY LOGARITHM?
        We optimize log(α) instead of α for numerical stability and to ensure α > 0
        """
        
        if self.automatic_entropy_tuning:
            # ----- Update Protagonist Temperature (α) -----
            # Compute loss: want entropy to match target
            # If log_pi is very negative (high entropy), loss pushes α down
            # If log_pi is less negative (low entropy), loss pushes α up
            alpha_loss = -(self.log_alpha * 
                          (log_pi + self.target_entropy).detach()).mean()
            
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            
            # Update α from log_α
            self.alpha = self.log_alpha.exp()
            
            # ----- Update Adversary Temperature (β) -----
            beta_loss = -(self.log_beta * 
                         (log_pi_adv + self.target_beta).detach()).mean()
            
            self.beta_optim.zero_grad()
            beta_loss.backward()
            self.beta_optim.step()
            
            # Update β from log_β
            self.beta = self.log_beta.exp()
        
        # =====================================================================
        # STEP 7: SOFT UPDATE TARGET NETWORKS
        # =====================================================================
        """
        GOAL: Slowly update target networks for stability
        
        SOFT UPDATE:
        θ_target = τ * θ_current + (1 - τ) * θ_target
        
        Where τ is typically very small (e.g., 0.005)
        
        INTUITION:
        Target network is like a "slow-moving average" of current network
        This provides stability because targets don't change too rapidly
        
        ANALOGY:
        Imagine learning to shoot basketballs. If the hoop moves every time
        you shoot (hard update), you'll never learn. If the hoop slowly drifts
        based on where you're aiming (soft update), you can gradually improve.
        
        WHY NOT JUST USE CURRENT NETWORK?
        Using current network for both predictions and targets causes instability:
        - Network predicts Q-value
        - We use same network to compute target
        - Network updates based on its own predictions
        - This creates a feedback loop → divergence!
        
        Target networks break this loop by providing stable targets
        """
        
        if self.updates % self.target_update_interval == 0:
            # Update protagonist critic target
            self.soft_update(self.critic_target, self.critic, self.tau)
            
            # Update adversary critic target
            self.soft_update(self.adversary_critic_target, 
                           self.adversary_critic, self.tau)
        
        self.updates += 1  # Increment update counter
        
        # Return losses for monitoring/logging
        return (qf1_loss.item(),           # Protagonist critic loss
                qf2_loss.item(),           # (not used, but available)
                policy_loss.item(),        # Protagonist policy loss
                adv_qf1_loss.item(),       # Adversary critic loss
                adversary_policy_loss.item())  # Adversary policy loss
    
    def soft_update(self, target, source, tau):
        """
        Soft update target network parameters
        
        Formula: θ_target ← τ*θ_source + (1-τ)*θ_target
        
        Args:
            target: Target network to update
            source: Source network to copy from
            tau: Update rate (typically 0.005)
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau)
    
    def hard_update(self, target, source):
        """
        Hard copy all parameters from source to target
        Used for initial synchronization
        
        Args:
            target: Target network
            source: Source network
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

# =============================================================================
# Training Functions
# =============================================================================

def plot_results(episode, rewards, losses, adv_losses, save_path=None):
    """
    Plot training progress for both protagonist and adversary
    
    WHAT TO LOOK FOR IN PLOTS:
    --------------------------
    1. Rewards: Should increase over time (protagonist improving)
    2. Protagonist Losses: Should decrease and stabilize
    3. Adversary Losses: Should also stabilize (adversary finding optimal disturbances)
    4. If rewards plateau but losses keep changing: might be oscillating between
       protagonist and adversary improvements (this is normal!)
    
    Args:
        episode: Current episode number
        rewards: List of episode rewards
        losses: List of loss tuples
        adv_losses: List of adversary losses (same as losses in this case)
        save_path: Path to save the figure (if None, displays interactively)
    """
    fig = plt.figure(figsize=(20, 10))
    
    # Episode Rewards
    plt.subplot(231)
    plt.title('Episode Rewards (Higher is Better)', fontsize=14, fontweight='bold')
    plt.plot(rewards, linewidth=2, color='blue', alpha=0.7)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    if losses:
        # Protagonist Critic Loss
        plt.subplot(232)
        plt.title('Protagonist Critic Loss', fontsize=14, fontweight='bold')
        plt.plot([l[0] for l in losses], linewidth=1.5, color='red', alpha=0.7)
        plt.xlabel('Update Step', fontsize=12)
        plt.ylabel('MSE Loss', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Protagonist Policy Loss
        plt.subplot(233)
        plt.title('Protagonist Policy Loss', fontsize=14, fontweight='bold')
        plt.plot([l[2] for l in losses], linewidth=1.5, color='green', alpha=0.7)
        plt.xlabel('Update Step', fontsize=12)
        plt.ylabel('Policy Loss', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Adversary Critic Loss
        plt.subplot(235)
        plt.title('Adversary Critic Loss', fontsize=14, fontweight='bold')
        plt.plot([l[3] for l in losses], linewidth=1.5, color='orange', alpha=0.7)
        plt.xlabel('Update Step', fontsize=12)
        plt.ylabel('MSE Loss', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Adversary Policy Loss
        plt.subplot(236)
        plt.title('Adversary Policy Loss', fontsize=14, fontweight='bold')
        plt.plot([l[4] for l in losses], linewidth=1.5, color='purple', alpha=0.7)
        plt.xlabel('Update Step', fontsize=12)
        plt.ylabel('Policy Loss', fontsize=12)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save or show the figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   💾 Saved figure: {save_path}")
        plt.close(fig)  # Close to free memory
    else:
        plt.show()

def train_wcsac(env_name, args):
    """
    Main training loop for WCSAC
    
    TRAINING PROCESS:
    -----------------
    1. Initialize environment and agent
    2. For each episode:
       a. Reset environment
       b. For each timestep:
          - Protagonist selects action
          - Adversary adds disturbance
          - Environment executes perturbed action
          - Store experience in replay buffer
          - Update networks (if enough data collected)
       c. Track episode reward
    3. Continue until solved or max episodes reached
    
    EARLY TRAINING (Random Exploration):
    ------------------------------------
    For first 'start_steps' timesteps, we use random actions.
    Why? Neural networks start with random weights, so initial policy is garbage.
    Random exploration ensures we collect diverse experiences before learning.
    
    Args:
        env_name: Gym environment name (e.g., 'Pendulum-v1')
        args: Dictionary of hyperparameters
        
    Returns:
        Trained agent and list of episode rewards
    """
    # =========================================================================
    # TENSORBOARD SETUP
    # =========================================================================
    # Create unique run directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(TENSORBOARD_DIR, f'run_{timestamp}')
    writer = SummaryWriter(log_dir=log_dir)
    
    print(f"\n📊 TensorBoard logging to: {log_dir}")
    print(f"   To visualize: tensorboard --logdir={TENSORBOARD_DIR}\n")
    
    # Log hyperparameters to TensorBoard
    writer.add_text('Hyperparameters', 
                    '\n'.join([f'{k}: {v}' for k, v in args.items()]))
    
    # Create environment
    env = gym.make(env_name)
    
    # Set random seeds for reproducibility
    # This ensures you get same results when running with same seed
    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])
    random.seed(args['seed'])
    
    # Initialize WCSAC agent
    agent = WCSAC(env.observation_space.shape[0], env.action_space, args)
    
    # Initialize replay buffer
    memory = ReplayBuffer(args['replay_size'])
    
    # Training metrics
    episode_rewards = []  # Track reward for each episode
    losses = []           # Track losses for plotting
    total_numsteps = 0    # Total environment steps taken
    update_count = 0      # Track number of network updates
    
    print(f"Starting training on {env_name}")
    print(f"State dimension: {env.observation_space.shape[0]}")
    print(f"Action dimension: {env.action_space.shape[0]}")
    print(f"Disturbance budget: {args['disturbance_budget']}")
    print("=" * 60)
    
    # ===== MAIN TRAINING LOOP =====
    for i_episode in range(args['max_episodes']):
        episode_reward = 0   # Accumulate reward for this episode
        episode_steps = 0    # Count steps in this episode
        episode_protagonist_actions = []  # Track actions for logging
        episode_adversary_disturbances = []  # Track disturbances for logging
        done = False
        state, _ = env.reset()  # Get initial state
        
        # ===== EPISODE LOOP =====
        while not done:
            # ----- Action Selection -----
            if total_numsteps < args['start_steps']:
                # Random exploration phase
                action = env.action_space.sample()
            else:
                # Use learned protagonist policy
                action = agent.select_action(state, evaluate=False)
            
            episode_protagonist_actions.append(action)
            
            # ----- Disturbance Selection -----
            if total_numsteps < args['start_steps']:
                # No disturbance during random exploration
                disturbance = np.zeros(env.action_space.shape[0])
            else:
                # Use learned adversary policy
                disturbance = agent.select_action(state, evaluate=False, 
                                                 use_adversary=True)
            
            episode_adversary_disturbances.append(disturbance)
            
            # ----- Apply Action with Disturbance -----
            # Clip to ensure we stay within valid action bounds
            perturbed_action = np.clip(
                action + disturbance,
                env.action_space.low,
                env.action_space.high
            )
            
            # ----- Environment Step -----
            next_state, reward, terminated, truncated, _ = env.step(perturbed_action)
            done = terminated or truncated
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward
            
            # ----- Store Transition -----
            # Mask: 0 if episode ended naturally, 1 if truncated by time limit
            # This prevents learning that time limit is a "bad" state
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)
            memory.push(state, action, disturbance, reward, next_state, mask)
            
            state = next_state
            
            # ----- Update Networks -----
            if len(memory) > args['batch_size']:
                # Perform gradient updates
                for _ in range(args['updates_per_step']):
                    loss = agent.update_parameters(memory, args['batch_size'])
                    losses.append(loss)
                    
                    # ===== LOG TO TENSORBOARD (per update) =====
                    if update_count % 10 == 0:  # Log every 10 updates to reduce overhead
                        writer.add_scalar('Loss/Protagonist_Critic_Q1', loss[0], update_count)
                        writer.add_scalar('Loss/Protagonist_Critic_Q2', loss[1], update_count)
                        writer.add_scalar('Loss/Protagonist_Policy', loss[2], update_count)
                        writer.add_scalar('Loss/Adversary_Critic_Q1', loss[3], update_count)
                        writer.add_scalar('Loss/Adversary_Policy', loss[4], update_count)
                        
                        # Log temperature parameters if auto-tuning is enabled
                        if agent.automatic_entropy_tuning:
                            writer.add_scalar('Temperature/Alpha_Protagonist', 
                                            agent.alpha.item(), update_count)
                            writer.add_scalar('Temperature/Beta_Adversary', 
                                            agent.beta.item(), update_count)
                    
                    update_count += 1
        
        # ----- Episode Complete -----
        episode_rewards.append(episode_reward)
        
        # ===== LOG TO TENSORBOARD (per episode) =====
        writer.add_scalar('Episode/Reward', episode_reward, i_episode)
        writer.add_scalar('Episode/Steps', episode_steps, i_episode)
        writer.add_scalar('Episode/Total_Steps', total_numsteps, i_episode)
        
        # Log action and disturbance statistics
        if episode_protagonist_actions:
            actions_array = np.array(episode_protagonist_actions)
            disturbances_array = np.array(episode_adversary_disturbances)
            
            writer.add_scalar('Episode/Mean_Action', 
                            np.mean(np.abs(actions_array)), i_episode)
            writer.add_scalar('Episode/Mean_Disturbance', 
                            np.mean(np.abs(disturbances_array)), i_episode)
            writer.add_scalar('Episode/Max_Disturbance', 
                            np.max(np.abs(disturbances_array)), i_episode)
            
            # Log action and disturbance distributions as histograms
            writer.add_histogram('Actions/Protagonist', actions_array, i_episode)
            writer.add_histogram('Disturbances/Adversary', disturbances_array, i_episode)
        
        # Log moving average rewards
        if len(episode_rewards) >= 10:
            avg_10 = np.mean(episode_rewards[-10:])
            writer.add_scalar('Episode/Avg_Reward_10', avg_10, i_episode)
        if len(episode_rewards) >= 100:
            avg_100 = np.mean(episode_rewards[-100:])
            writer.add_scalar('Episode/Avg_Reward_100', avg_100, i_episode)
        
        # Log replay buffer size
        writer.add_scalar('Training/Replay_Buffer_Size', len(memory), i_episode)
        
        # ----- Logging and Visualization -----
        if i_episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode: {i_episode:4d} | "
                  f"Steps: {total_numsteps:6d} | "
                  f"Reward: {episode_reward:7.2f} | "
                  f"Avg(10): {avg_reward:7.2f}")
            
            # Plot every 50 episodes
            if i_episode % 50 == 0 and losses:
                # Save figure with episode number in filename
                save_path = os.path.join(FIGURE_DIR, f'training_episode_{i_episode:04d}.png')
                plot_results(i_episode, episode_rewards, losses, losses, save_path=save_path)
                
                # Also add the figure to TensorBoard
                fig = plt.figure(figsize=(20, 10))
                # Recreate the plot for TensorBoard
                plt.subplot(231)
                plt.plot(episode_rewards)
                plt.title('Episode Rewards')
                writer.add_figure('Training_Progress', fig, i_episode)
                plt.close(fig)
        
        # ----- Early Stopping -----
        # If average reward over last 100 episodes exceeds threshold, we're done!
        if len(episode_rewards) >= 100 and \
           np.mean(episode_rewards[-100:]) > args['target_reward']:
            print(f"\n🎉 Environment solved in {i_episode} episodes!")
            print(f"Average reward: {np.mean(episode_rewards[-100:]):.2f}")
            writer.add_text('Training', f'Solved in {i_episode} episodes!')
            break
    
    # ===== CLOSE TENSORBOARD WRITER =====
    writer.close()
    print(f"\n📊 TensorBoard logs saved to: {log_dir}")
    
    env.close()
    
    # Save final training plot
    if episode_rewards:
        final_plot_path = os.path.join(FIGURE_DIR, 'training_final.png')
        plot_results(len(episode_rewards), episode_rewards, losses, losses, save_path=final_plot_path)
    
    return agent, episode_rewards

# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    """
    HYPERPARAMETER GUIDE:
    ---------------------
    
    gamma (0.99): Discount factor
        - Higher = care more about future rewards
        - Lower = care more about immediate rewards
        - 0.99 is standard for most tasks
    
    tau (0.005): Target network update rate
        - Lower = more stable but slower learning
        - Higher = faster learning but less stable
        - 0.005 is standard
    
    lr (0.0003): Learning rate
        - Too high = unstable training
        - Too low = slow training
        - 0.0003 is good default
    
    alpha (0.2): Protagonist temperature
        - Higher = more exploration
        - Lower = more exploitation
        - Usually auto-tuned
    
    beta (0.2): Adversary temperature
        - Higher = more diverse disturbances
        - Lower = more focused attacks
        - Usually auto-tuned
    
    batch_size (256): Number of experiences per update
        - Larger = more stable gradients but slower
        - Smaller = faster but noisier
        - 256 is good balance
    
    disturbance_budget (0.5): Maximum disturbance magnitude
        - Lower = easier for protagonist (less robust)
        - Higher = harder for protagonist (more robust)
        - Should be tuned based on environment
        - Too high = impossible to learn
        - Too low = not enough robustness training
    """
    
    # Hyperparameters
    args = {
        'env_name': 'Pendulum-v1',
        'gamma': 0.99,                    # Discount factor
        'tau': 0.005,                     # Soft update rate
        'lr': 0.0003,                     # Learning rate
        'alpha': 0.2,                     # Protagonist temperature (initial)
        'beta': 0.02,                      # Adversary temperature (initial)
        'automatic_entropy_tuning': True, # Auto-adjust α and β
        'seed': 42,                       # Random seed
        'batch_size': 256,                # Batch size for updates
        'max_episodes': 1000,             # Maximum episodes
        'hidden_size': 256,               # Neural network hidden size
        'start_steps': 1000,              # Random exploration steps
        'updates_per_step': 1,            # Gradient updates per env step
        'target_update_interval': 1,      # Target network update frequency
        'replay_size': 1000000,           # Replay buffer capacity
        'target_reward': -200,            # Reward threshold for early stopping
        'disturbance_budget': 0.05,        # Maximum disturbance magnitude
    }
    
    # ===== CPU Optimization =====
    # If using CPU, reduce these for faster training:
    if device.type == 'cpu':
        print("\n⚡ Optimizing for CPU training...")
        args['batch_size'] = 128          # Smaller batches
        args['hidden_size'] = 128         # Smaller networks
        args['max_episodes'] = 500        # Fewer episodes
        print(f"   Adjusted batch_size: {args['batch_size']}")
        print(f"   Adjusted hidden_size: {args['hidden_size']}")
        print(f"   Adjusted max_episodes: {args['max_episodes']}\n")
    
    print("=" * 60)
    print("WCSAC: Worst-Case Soft Actor-Critic")
    print("=" * 60)
    print(f"Environment: {args['env_name']}")
    print(f"Disturbance Budget: {args['disturbance_budget']}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Train the agent
    agent, rewards = train_wcsac(args['env_name'], args)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print("=" * 60)
    print(f"Total episodes: {len(rewards)}")
    print(f"Final average reward (last 100): {np.mean(rewards[-100:]):.2f}")
    print("=" * 60)
    
    # =========================================================================
    # TESTING PHASE: Evaluate trained agent
    # =========================================================================
    print("\n" + "=" * 60)
    print("TESTING TRAINED AGENT")
    print("=" * 60)
    
    # Test 1: Without adversarial disturbances
    print("\n📊 Test 1: Performance without disturbances")
    print("-" * 60)
    env = gym.make(args['env_name'], render_mode='human')
    state, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done:
        action = agent.select_action(state, evaluate=True)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
    
    print(f"✅ Clean environment reward: {total_reward:.2f}")
    print(f"   Steps: {steps}")
    
    # Test 2: With adversarial disturbances
    print("\n📊 Test 2: Performance with adversarial disturbances")
    print("-" * 60)
    state, _ = env.reset()
    done = False
    total_reward_adv = 0
    steps_adv = 0
    total_disturbance = 0
    
    while not done:
        action = agent.select_action(state, evaluate=True)
        disturbance = agent.select_action(state, evaluate=True, use_adversary=True)
        perturbed_action = np.clip(
            action + disturbance,
            env.action_space.low,
            env.action_space.high
        )
        state, reward, terminated, truncated, _ = env.step(perturbed_action)
        done = terminated or truncated
        total_reward_adv += reward
        steps_adv += 1
        total_disturbance += np.abs(disturbance).sum()
    
    print(f"⚠️  Adversarial environment reward: {total_reward_adv:.2f}")
    print(f"   Steps: {steps_adv}")
    print(f"   Average disturbance magnitude: {total_disturbance/steps_adv:.4f}")
    
    # Calculate robustness metrics
    print("\n" + "=" * 60)
    print("ROBUSTNESS ANALYSIS")
    print("=" * 60)
    robustness_gap = total_reward - total_reward_adv
    robustness_ratio = (total_reward_adv / total_reward * 100) if total_reward != 0 else 0
    
    print(f"🔍 Robustness Gap: {robustness_gap:.2f}")
    print(f"   (How much performance degrades under attack)")
    print(f"\n🔍 Robustness Ratio: {robustness_ratio:.1f}%")
    print(f"   (Percentage of performance retained under attack)")
    print(f"\n💡 Interpretation:")
    if robustness_ratio > 80:
        print("   Excellent! Agent is very robust to disturbances.")
    elif robustness_ratio > 60:
        print("   Good! Agent maintains decent performance under attack.")
    elif robustness_ratio > 40:
        print("   Fair. Agent is somewhat robust but could improve.")
    else:
        print("   Poor. Agent struggles significantly with disturbances.")
    print(f"\n   Consider {'decreasing' if robustness_ratio < 60 else 'increasing'} "
          f"disturbance_budget for {'easier' if robustness_ratio < 60 else 'harder'} training.")
    
    print("=" * 60)
    env.close()
    print("\n✅ All tests complete!")
    
    # =========================================================================
    # SAVE FINAL SUMMARY FIGURE
    # =========================================================================
    print("\n" + "=" * 60)
    print("SAVING SUMMARY VISUALIZATIONS")
    print("=" * 60)
    
    # Create comprehensive summary figure
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Episode Rewards with Moving Average
    plt.subplot(2, 3, 1)
    plt.plot(rewards, alpha=0.5, label='Episode Reward', color='blue')
    if len(rewards) >= 10:
        moving_avg = np.convolve(rewards, np.ones(10)/10, mode='valid')
        plt.plot(range(9, len(rewards)), moving_avg, 'r-', linewidth=2, label='Moving Avg (10)')
    plt.title('Training Rewards Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Reward Distribution
    plt.subplot(2, 3, 2)
    plt.hist(rewards, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rewards):.2f}')
    plt.title('Reward Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Performance Comparison
    plt.subplot(2, 3, 3)
    performance = [total_reward, total_reward_adv]
    labels = ['Clean\nEnvironment', 'Adversarial\nEnvironment']
    colors = ['green', 'orange']
    bars = plt.bar(labels, performance, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    plt.title('Performance Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Total Reward')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, performance):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 4. Robustness Metrics
    plt.subplot(2, 3, 4)
    metrics = ['Robustness\nRatio (%)', 'Robustness\nGap']
    values = [robustness_ratio, abs(robustness_gap)]
    colors_met = ['purple', 'coral']
    bars = plt.bar(metrics, values, color=colors_met, alpha=0.7, edgecolor='black', linewidth=2)
    plt.title('Robustness Metrics', fontsize=14, fontweight='bold')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 5. Training Progress Statistics
    plt.subplot(2, 3, 5)
    stats_text = f"""
    TRAINING STATISTICS
    {'='*35}
    
    Total Episodes: {len(rewards)}
    
    Mean Reward: {np.mean(rewards):.2f}
    Std Reward: {np.std(rewards):.2f}
    Min Reward: {np.min(rewards):.2f}
    Max Reward: {np.max(rewards):.2f}
    
    Final 100 Episodes Avg: {np.mean(rewards[-100:]):.2f}
    
    {'='*35}
    ROBUSTNESS ANALYSIS
    {'='*35}
    
    Clean Performance: {total_reward:.2f}
    Adversarial Performance: {total_reward_adv:.2f}
    
    Robustness Ratio: {robustness_ratio:.1f}%
    Robustness Gap: {robustness_gap:.2f}
    
    Avg Disturbance: {total_disturbance/steps_adv:.4f}
    """
    plt.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.axis('off')
    
    # 6. Interpretation
    plt.subplot(2, 3, 6)
    if robustness_ratio > 80:
        interpretation = "EXCELLENT!\n\nAgent is very robust\nto disturbances."
        color = 'green'
    elif robustness_ratio > 60:
        interpretation = "GOOD!\n\nAgent maintains decent\nperformance under attack."
        color = 'yellowgreen'
    elif robustness_ratio > 40:
        interpretation = "FAIR\n\nAgent is somewhat robust\nbut could improve."
        color = 'orange'
    else:
        interpretation = "NEEDS IMPROVEMENT\n\nAgent struggles with\ndisturbances."
        color = 'red'
    
    plt.text(0.5, 0.5, interpretation, fontsize=16, fontweight='bold',
             ha='center', va='center', color=color,
             bbox=dict(boxstyle='round', facecolor='white', edgecolor=color, linewidth=3))
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save summary figure
    summary_path = os.path.join(FIGURE_DIR, 'training_summary.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved summary figure: {summary_path}")
    plt.close()
    
    # Create a simple reward curve figure
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, linewidth=2, alpha=0.7, color='blue', label='Episode Rewards')
    if len(rewards) >= 10:
        moving_avg = np.convolve(rewards, np.ones(10)/10, mode='valid')
        plt.plot(range(9, len(rewards)), moving_avg, 'r-', linewidth=3, label='Moving Average (10 episodes)')
    plt.title('WCSAC Training Progress', fontsize=16, fontweight='bold')
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Total Reward', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    rewards_curve_path = os.path.join(FIGURE_DIR, 'reward_curve.png')
    plt.savefig(rewards_curve_path, dpi=300, bbox_inches='tight')
    print(f"📈 Saved reward curve: {rewards_curve_path}")
    plt.close()
    
    print(f"\n✅ All figures saved to: {FIGURE_DIR}/")
    print("\nGenerated files:")
    print(f"  - training_episode_XXXX.png (progress plots every 50 episodes)")
    print(f"  - training_final.png (final training plot)")
    print(f"  - training_summary.png (comprehensive summary)")
    print(f"  - reward_curve.png (reward progression)")
    print("=" * 60)
