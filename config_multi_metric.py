"""
Enhanced Configuration File with Multi-Metric QoS Support

This configuration supports:
1. Single metric per slice (backward compatible)
2. Multiple metrics per slice (new feature)

Each slice can have:
- One or more QoS metrics
- Corresponding thresholds for each metric
- All metrics must be satisfied for slice QoS success
"""

# ============================================================================
# ALGORITHM 1: ENVIRONMENT PARAMETERS
# ============================================================================

# Network Topology
K = 3           # Number of slices
C = 8           # Total RB capacity
N = 8           # TTIs per DTI

# Random Seeds
RANDOM_SEED = 42
TRAFFIC_SEED = 42
PROFILE_SEED = 42
NETWORK_SEED = 42
DETERMINISTIC = False

# ============================================================================
# MULTI-METRIC QOS CONFIGURATION
# ============================================================================

# Option 1: Single metric per slice (Backward Compatible)
# -------------------------------------------------------
# Uncomment this for single-metric mode:

# QOS_TABLE_FILES = [
#     'qos_voip_all_metrics.json',
#     'qos_cbr_all_metrics.json',
#     'qos_video_all_metrics.json'
# ]
# 
# QOS_METRICS = [
#     'voIPFrameDelay',        # Single metric for slice 0
#     'cbrFrameDelay',         # Single metric for slice 1
#     'rtVideoStreamingSegmentLoss'  # Single metric for slice 2
# ]
# 
# THRESHOLDS = [40, 1600, 15]  # One threshold per slice


# Option 2: Multiple metrics per slice (New Feature)
# ---------------------------------------------------
# Each slice can have multiple metrics that ALL must be satisfied

QOS_TABLE_FILES = [
    'qos_voip_all_metrics.json',
    'qos_cbr_all_metrics.json',
    'qos_video_all_metrics.json'
]

# NEW: List of lists - each inner list contains metrics for that slice
# Available metrics (check your JSON files):
#   VoIP: voIPFrameLoss, voIPFrameDelay, voIPJitter, voIPReceivedThroughput, etc.
#   CBR: cbrFrameDelay, cbrReceivedThroughput
#   Video: rtVideoStreamingEnd2endDelaySegment, rtVideoStreamingSegmentLoss

QOS_METRICS_MULTI = [
    # Slice 0 (VoIP): Must satisfy BOTH delay AND loss
    ['voIPFrameDelay', 'voIPFrameLoss', 'voIPJitter'],
    
    # Slice 1 (CBR): Must satisfy BOTH delay AND throughput
    ['cbrFrameDelay'],
    
    # Slice 2 (Video): Must satisfy BOTH segment loss AND delay
    ['rtVideoStreamingSegmentLoss']
]

# NEW: List of lists - thresholds corresponding to each metric
# Format: [[thresholds for slice 0], [thresholds for slice 1], ...]
THRESHOLDS_MULTI = [
    # Slice 0: [delay threshold, loss threshold]
    [40, 0.5, 75],      # delay <= 40ms AND loss <= 5%
    
    # Slice 1: [delay threshold, throughput threshold (minimum)]
    [1600],  # delay <= 1600ms AND throughput >= 500 kbps
    
    # Slice 2: [loss threshold, delay threshold]
    [25]        # loss <= 15% AND delay <= 500ms
]


# NEW: Specify which metrics are "lower is better" vs "higher is better"
# Format: [[directions for slice 0], [directions for slice 1], ...]
# 'lower' = metric should be below threshold (delay, loss, jitter)
# 'higher' = metric should be above threshold (throughput, rate)
QOS_METRIC_DIRECTIONS = [
    # Slice 0
    ['lower', 'lower', 'lower'],     # delay: lower is better, loss: lower is better
    
    # Slice 1
    ['lower'],    # delay: lower is better, throughput: higher is better
    
    # Slice 2
    ['lower']      # loss: lower is better, delay: lower is better
]

# Beta threshold (overall QoS violation ratio target)
BETA_THRESH = 0.2

# ============================================================================
# BACKWARD COMPATIBILITY HELPER
# ============================================================================

# Auto-detect mode and set backward-compatible variables
if 'QOS_METRICS_MULTI' in locals():
    # Multi-metric mode
    USE_MULTI_METRIC_QOS = True
    
    # For backward compatibility, create single-metric versions
    # (using first metric of each slice)
    QOS_METRICS = [metrics[0] for metrics in QOS_METRICS_MULTI]
    THRESHOLDS = [thresholds[0] for thresholds in THRESHOLDS_MULTI]
else:
    # Single-metric mode (backward compatible)
    USE_MULTI_METRIC_QOS = False
    
    # Convert to multi-metric format internally
    QOS_METRICS_MULTI = [[metric] for metric in QOS_METRICS]
    THRESHOLDS_MULTI = [[threshold] for threshold in THRESHOLDS]
    QOS_METRIC_DIRECTIONS = [['lower'] for _ in range(K)]

# ============================================================================
# REWARD PARAMETER
# ============================================================================

LAMBDA = 0.5    # λ: Weight for resource efficiency bonus

# Sliding Window
W = 5           # Window size for β and CDF computation

# ============================================================================
# EFFICIENT RESOURCE ALLOCATION MODE
# ============================================================================

# Enable efficient resource allocation (K+1 actions with "null slice")
# When True: Agent can choose to NOT allocate all capacity (save resources)
# When False: Agent must allocate all capacity (standard SAC, sum = C)
USE_EFFICIENT_ALLOCATION = False #True

# Reward weight for unused capacity (only used if USE_EFFICIENT_ALLOCATION=True)
# Positive value: Rewards saving resources (encourages efficiency)
# Zero: Neutral (no reward/penalty for unused capacity)
# Negative: Penalizes unused capacity (encourages full allocation)
UNUSED_CAPACITY_REWARD_WEIGHT = 0.1

# How efficient allocation works:
# - Actor output dimension: K+1 (K slices + 1 "null slice")
# - After softmax: allocations sum to C
# - First K allocations go to actual slices
# - Last allocation represents "saved/unused" capacity
# - Environment receives only K actions (sum may be < C)
# - Constraint: sum(allocations) ≤ C (enforced, but not required to equal C)

# ============================================================================
# TRAFFIC GENERATION
# ============================================================================

TRAFFIC_PROFILES = ['dynamic', 'dynamic', 'dynamic']

# Dynamic Profile Configuration
DYNAMIC_PROFILE_CONFIG = {
    'profile_set': ['low', 'medium', 'high'],
    'change_period': 200
}

# Traffic Values
TRAFFIC_VALUES = list(range(5, 85, 5))

# Episode Length
T_MAX = 2000

# ============================================================================
# ALGORITHM 2: SAC TRAINING PARAMETERS
# ============================================================================

NUM_EPISODES = 100
MAX_DTIS = 2000

# Learning Rates
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4

# RL Parameters
GAMMA = 0.99
TAU = 0.005

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
DEVICE = 'cpu'

# ============================================================================
# WCSAC PARAMETERS (if using Worst-Case SAC)
# ============================================================================

WCSAC_KAPPA = 0.5
WCSAC_UNCERTAINTY_RADIUS = 0.1
WCSAC_PESSIMISM_PENALTY = 0.1

# ============================================================================
# EVALUATION PARAMETERS
# ============================================================================

EVAL_EPISODES = 100
CHECKPOINT_PATH = None

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
        'thresholds': THRESHOLDS,  # Backward compatible
        'thresholds_multi': THRESHOLDS_MULTI,  # New multi-metric
        'qos_metric_directions': QOS_METRIC_DIRECTIONS,  # New
        'use_multi_metric_qos': USE_MULTI_METRIC_QOS,  # New flag
        'lambda_reward': LAMBDA,
        'window_size': W,
        'traffic_profiles': TRAFFIC_PROFILES,
        'qos_table_files': QOS_TABLE_FILES,
        'qos_metrics': QOS_METRICS,  # Backward compatible
        'qos_metrics_multi': QOS_METRICS_MULTI,  # New multi-metric
        'dynamic_profile_config': DYNAMIC_PROFILE_CONFIG,
        'max_dtis': T_MAX,
        
        # Efficient allocation
        'use_efficient_allocation': USE_EFFICIENT_ALLOCATION,
        'unused_capacity_reward_weight': UNUSED_CAPACITY_REWARD_WEIGHT,
        
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
        
        # WCSAC
        'wcsac_kappa': WCSAC_KAPPA,
        'wcsac_uncertainty_radius': WCSAC_UNCERTAINTY_RADIUS,
        'wcsac_pessimism_penalty': WCSAC_PESSIMISM_PENALTY,
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
    
    # Print QoS configuration
    if USE_MULTI_METRIC_QOS:
        print(f"\n  QoS Mode:             MULTI-METRIC")
        print(f"  Metrics per slice:")
        for k in range(K):
            metrics_str = ', '.join(QOS_METRICS_MULTI[k])
            thresholds_str = ', '.join(map(str, THRESHOLDS_MULTI[k]))
            directions_str = ', '.join(QOS_METRIC_DIRECTIONS[k])
            print(f"    Slice {k}: {metrics_str}")
            print(f"              Thresholds: {thresholds_str}")
            print(f"              Directions: {directions_str}")
    else:
        print(f"\n  QoS Mode:             SINGLE-METRIC")
        print(f"  τ (thresholds):       {THRESHOLDS}")
        print(f"  QoS metrics:          {QOS_METRICS}")
    
    print(f"\n  λ (lambda):           {LAMBDA}")
    print(f"  W (window size):      {W if W else '∞'} DTIs")
    print(f"  Traffic profiles:     {TRAFFIC_PROFILES}")
    print(f"  Dynamic config:")
    print(f"    Profile set:        {DYNAMIC_PROFILE_CONFIG['profile_set']}")
    print(f"    Change period:      {DYNAMIC_PROFILE_CONFIG['change_period']} DTIs")
    print(f"  QoS table files:      {QOS_TABLE_FILES}")
    print(f"  T_max (max DTIs):     {T_MAX}")
    
    print("\n[RESOURCE ALLOCATION MODE]")
    if USE_EFFICIENT_ALLOCATION:
        print(f"  Mode:                 EFFICIENT (K+1 actions)")
        print(f"  Actor output dim:     {K+1} (K slices + 1 null)")
        print(f"  Unused reward weight: {UNUSED_CAPACITY_REWARD_WEIGHT}")
        print(f"  Constraint:           sum(actions) ≤ C")
        print(f"  Benefit:              Agent can save resources when not needed")
    else:
        print(f"  Mode:                 STANDARD (K actions)")
        print(f"  Actor output dim:     {K}")
        print(f"  Constraint:           sum(actions) = C (softmax guaranteed)")
        print(f"  Behavior:             All capacity always allocated")
    
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


def validate_multi_metric_config():
    """Validate multi-metric QoS configuration"""
    if not USE_MULTI_METRIC_QOS:
        return True
    
    errors = []
    
    # Check lengths match
    if len(QOS_METRICS_MULTI) != K:
        errors.append(f"QOS_METRICS_MULTI length ({len(QOS_METRICS_MULTI)}) != K ({K})")
    
    if len(THRESHOLDS_MULTI) != K:
        errors.append(f"THRESHOLDS_MULTI length ({len(THRESHOLDS_MULTI)}) != K ({K})")
    
    if len(QOS_METRIC_DIRECTIONS) != K:
        errors.append(f"QOS_METRIC_DIRECTIONS length ({len(QOS_METRIC_DIRECTIONS)}) != K ({K})")
    
    # Check each slice has matching lengths
    for k in range(K):
        n_metrics = len(QOS_METRICS_MULTI[k])
        n_thresholds = len(THRESHOLDS_MULTI[k])
        n_directions = len(QOS_METRIC_DIRECTIONS[k])
        
        if n_metrics != n_thresholds:
            errors.append(f"Slice {k}: {n_metrics} metrics but {n_thresholds} thresholds")
        
        if n_metrics != n_directions:
            errors.append(f"Slice {k}: {n_metrics} metrics but {n_directions} directions")
        
        # Check directions are valid
        for direction in QOS_METRIC_DIRECTIONS[k]:
            if direction not in ['lower', 'higher']:
                errors.append(f"Slice {k}: Invalid direction '{direction}' (must be 'lower' or 'higher')")
    
    if errors:
        print("ERROR: Multi-metric QoS configuration invalid:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True


if __name__ == "__main__":
    # Validate configuration
    if not validate_multi_metric_config():
        print("\n⚠️  Please fix configuration errors above")
        exit(1)
    
    print_config()
    print("\n✓ Configuration valid!")
