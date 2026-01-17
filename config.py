"""
Simple Configuration File
Contains only the main parameters from LaTeX algorithms
"""

# ============================================================================
# ALGORITHM 1: ENVIRONMENT PARAMETERS
# ============================================================================

# Network Topology
K = 2           # Number of slices
C = 8           # Total RB capacity
N = 8          # TTIs per DTI

# QoS Parameters
THRESHOLDS = [40, 40]  # τ_k: QoS threshold for each slice k
BETA_THRESH = 0.2         # β_thresh: Beta threshold (not used in current implementation)

# Reward Parameter
LAMBDA = 0.5    # λ: Weight for resource efficiency bonus

# Sliding Window
W = 5           # Window size for β and CDF computation (None = ∞, 5 = last 5 DTIs)

# Traffic Generation (Algorithm 4)
TRAFFIC_PROFILES = ['uniform', 'uniform']  # Profile for each slice
# Options: 'uniform', 'low', 'medium', 'high', 'external'

# QoS Tables
QOS_TABLE_FILES = ['qos_voip_all_metrics.json', 'qos_voip_all_metrics.json']  # JSON file path for each slice's QoS table
# Example: ['qos_voip_all_metrics.json', 'qos_cbr_all_metrics.json']
# Set to None to use default QoS model

# QoS Metrics to Use
QOS_METRICS = ['voIPFrameDelay', 'voIPFrameDelay']  # Which metric to use from each QoS file
# Example: ['voIPFrameLoss', 'cbrFrameDelay']
# Set to None to use the first available metric
# 
# Available metrics (check your JSON files):
#   VoIP: voIPFrameLoss, voIPFrameDelay, voIPJitter, voIPReceivedThroughput, etc.
#   CBR: cbrFrameDelay, cbrReceivedThroughput
#   Video: rtVideoStreamingEnd2endDelaySegment, rtVideoStreamingSegmentLoss
#
# JSON format after conversion contains ALL metrics:
# {
#   "5": {
#     "1": {
#       "voIPFrameLoss": {"mu": 0.149, "sigma": 0.525},
#       "voIPFrameDelay": {"mu": 5.614, "sigma": 3.180},
#       ...
#     }
#   }
# }

# Traffic Values
TRAFFIC_VALUES = list(range(5, 85, 5))  # T = {5, 10, 15, ..., 80}

# Episode Length
T_MAX = 200     # Maximum DTIs per episode


# ============================================================================
# ALGORITHM 2: SAC TRAINING PARAMETERS
# ============================================================================

# Training Duration
NUM_EPISODES = 1000  # E_max: Total episodes
MAX_DTIS = 200      # T_max: DTIs per episode (same as T_MAX above)

# Learning Rates
LR_ACTOR = 3e-4   # η_π: Actor learning rate
LR_CRITIC = 3e-4  # η_Q: Critic learning rate

# RL Parameters
GAMMA = 0.99      # Discount factor
TAU = 0.005       # τ_soft: Soft update rate

# Training Parameters
BATCH_SIZE = 8
BUFFER_CAPACITY = 100000
MIN_BUFFER_SIZE = 1000

# Logging
LOG_INTERVAL = 10
SAVE_INTERVAL = 100

# Directories
CHECKPOINT_DIR = 'checkpoints'
TENSORBOARD_DIR = 'runs'
RESULTS_DIR = 'results'

# Device
DEVICE = 'cpu'  # 'cpu' or 'cuda'


# ============================================================================
# EVALUATION PARAMETERS
# ============================================================================

EVAL_EPISODES = 100
CHECKPOINT_PATH = None  # Path to checkpoint (None = auto-find latest)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_config():
    """Return all parameters as a dictionary"""
    return {
        # Environment
        'K': K,
        'C': C,
        'N': N,
        'thresholds': THRESHOLDS,
        'lambda_reward': LAMBDA,
        'window_size': W,
        'traffic_profiles': TRAFFIC_PROFILES,
        'qos_table_files': QOS_TABLE_FILES,
        'qos_metrics': QOS_METRICS,
        'max_dtis': T_MAX,
        
        # SAC
        'lr_actor': LR_ACTOR,
        'lr_critic': LR_CRITIC,
        'gamma': GAMMA,
        'tau': TAU,
        'batch_size': BATCH_SIZE,
        'buffer_capacity': BUFFER_CAPACITY,
        'min_buffer_size': MIN_BUFFER_SIZE,
        'device': DEVICE,
        
        # Training
        'num_episodes': NUM_EPISODES,
        'log_interval': LOG_INTERVAL,
        'save_interval': SAVE_INTERVAL,
        'checkpoint_dir': CHECKPOINT_DIR,
        'tensorboard_dir': TENSORBOARD_DIR,
        'results_dir': RESULTS_DIR,
        
        # Evaluation
        'eval_episodes': EVAL_EPISODES,
        'checkpoint_path': CHECKPOINT_PATH,
    }


def print_config():
    """Print configuration summary"""
    print("=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    print("\n[ENVIRONMENT - Algorithm 1]")
    print(f"  K (slices):           {K}")
    print(f"  C (capacity):         {C} RBs")
    print(f"  N (TTIs/DTI):         {N}")
    print(f"  τ (thresholds):       {THRESHOLDS}")
    print(f"  λ (lambda):           {LAMBDA}")
    print(f"  W (window size):      {W if W else '∞'} DTIs")
    print(f"  Traffic profiles:     {TRAFFIC_PROFILES}")
    print(f"  QoS table files:      {QOS_TABLE_FILES}")
    print(f"  QoS metrics:          {QOS_METRICS}")
    print(f"  T_max (max DTIs):     {T_MAX}")
    
    print("\n[SAC - Algorithm 2]")
    print(f"  E_max (episodes):     {NUM_EPISODES}")
    print(f"  η_π (lr_actor):       {LR_ACTOR}")
    print(f"  η_Q (lr_critic):      {LR_CRITIC}")
    print(f"  γ (gamma):            {GAMMA}")
    print(f"  τ_soft (tau):         {TAU}")
    print(f"  Batch size:           {BATCH_SIZE}")
    print(f"  Device:               {DEVICE}")
    
    print("\n[TRAINING]")
    print(f"  Log interval:         {LOG_INTERVAL}")
    print(f"  Save interval:        {SAVE_INTERVAL}")
    print(f"  Checkpoint dir:       {CHECKPOINT_DIR}")
    
    print("\n[EVALUATION]")
    print(f"  Eval episodes:        {EVAL_EPISODES}")
    print("=" * 70)


if __name__ == "__main__":
    print_config()
