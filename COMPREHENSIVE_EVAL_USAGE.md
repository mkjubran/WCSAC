# Comprehensive Evaluation Script - Usage Guide

## ✅ Updates Applied

### 1. **Better Tag Detection**
- Now finds `dti/reward`, `dti/beta` tags
- Finds `episode_100/reward`, `episode_100/beta` snapshot tags
- Prioritizes episode-level over DTI-level metrics
- Shows examples of found tags

### 2. **Last & Best Episode Reporting**
- Shows **last episode value** for reward and beta
- Shows **best episode value** with episode number
- Format: `Best episode value: 0.2754 (episode 450)`

### 3. **Dynamic Profile Details**
- Shows traffic profiles: `['dynamic', 'dynamic', 'dynamic']`
- **Extracts traffic levels** from config:
  ```
  Dynamic Profile Configuration:
    Available levels:           ['low', 'medium', 'high']
    Change period:              100 DTIs
    Initial profile:            low
  
  Traffic Level Details:
    low             -> Beta(2, 5)
    medium          -> Beta(3, 4)
    high            -> Beta(5, 3)
  ```

### 4. **Per-Episode Snapshots**
- Reports detailed metrics for episode checkpoints (100, 200, etc.)
- Shows reward and beta for specific episodes
- Example output:
  ```
  7. Per-Episode Detailed Snapshots
  --------------------------------------------------
     Available snapshots: Episodes [100, 200, 300, 400, 500]
  
     Episode      Reward          Beta           
     ------------------------------------------
     100          -720.34         0.3567        
     200          -680.12         0.3234        
     300          -650.45         0.3012        
     400          -625.78         0.2856        
     500          -610.23         0.2734        
  ```

---

## 📊 Example Output

### Config with Dynamic Profiles:

```
================================================================================
EXPERIMENTAL CONFIGURATION
================================================================================

Network Parameters:
  Number of slices (K):         3
  Total capacity (C):           12 RBs
  TTIs per DTI (N):             8
  QoS thresholds:               [40, 40, 40]
  Traffic profiles:             ['dynamic', 'dynamic', 'dynamic']

  Dynamic Profile Configuration:
    Available levels:           ['low', 'medium', 'high']
    Change period:              100 DTIs
    Initial profile:            medium

  Traffic Level Details:
    low             -> Beta(2, 5)
    medium          -> Beta(3, 4)
    high            -> Beta(5, 3)

  QoS Table Files:
    Slice 0: qos_voip_all_metrics.json
    Slice 1: qos_cbr_all_metrics.json
    Slice 2: qos_video_all_metrics.json

  QoS Metrics:
    Slice 0: voIPFrameDelay
    Slice 1: cbrFrameDelay
    Slice 2: rtVideoStreamingSegmentLoss

Training Parameters:
  Training episodes:            1000
  Max DTIs per episode:         2000
  Window size (W):              5
  Reward weight (λ):            0.5
...
```

### Metrics with Last/Best Episode:

```
1. QoS Violation Ratio (β)
--------------------------------------------------------------------------------
   Final 100-episode mean:  0.3339 ± 0.0234
   Last episode value:      0.3156
   Best episode value:      0.2754 (episode 450)
   Overall mean:            0.3445 ± 0.0312
   Target achievement (β<0.2):   15.0% of episodes
   Status:                  ✗ NOT ACHIEVED

2. Cumulative Episode Reward
--------------------------------------------------------------------------------
   Final 100-episode mean:  -667.78 ± 45.23
   Last episode value:      -650.34
   Best episode value:      -550.73 (episode 450)
   Overall mean:            -682.34 ± 58.91
```

---

## 🚀 Usage

### Basic Usage:
```bash
python3 comprehensive_evaluation.py \
    --log-dir runs/sac_run1 \
    --config config_run1.py
```

### With All Options:
```bash
python3 comprehensive_evaluation.py \
    --log-dir runs/sac_dynamic_profiles \
    --config config_dynamic.py \
    --window 100 \
    --target-beta 0.25 \
    --output-json metrics.json \
    --latex-table
```

---

## 📋 Expected Config File Format

Your config file should include:

```python
# Network parameters
K = 3
C = 12
N = 8
thresholds = [40, 40, 40]
traffic_profiles = ['dynamic', 'dynamic', 'dynamic']

# Dynamic profile configuration
DYNAMIC_PROFILE_CONFIG = {
    'profile_set': ['low', 'medium', 'high'],
    'change_period': 100,  # DTIs
    'initial_profile': 'medium'
}

# Traffic level definitions
TRAFFIC_PROFILES = {
    'low': {'alpha': 2, 'beta': 5},
    'medium': {'alpha': 3, 'beta': 4},
    'high': {'alpha': 5, 'beta': 3}
}

# QoS Tables
QOS_TABLE_FILES = ['qos_voip_all_metrics.json', 'qos_cbr_all_metrics.json', 'qos_video_all_metrics.json']
# Set to None to use default QoS model

# QoS Metrics to Use
QOS_METRICS = ['voIPFrameDelay', 'cbrFrameDelay', 'rtVideoStreamingSegmentLoss']

# Training parameters
NUM_EPISODES = 1000
max_dtis = 2000
window_size = 5
lambda_reward = 0.5

# SAC hyperparameters
LR_ACTOR = 0.0003
LR_CRITIC = 0.0003
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2
BATCH_SIZE = 256
BUFFER_SIZE = 100000
HIDDEN_SIZES = [256, 256]
```

---

## 🔍 Tag Detection Examples

### What the Script Finds:

**Your tags:**
- `dti/reward` → Used for main reward metric
- `dti/beta` → Used for main beta metric
- `episode_100/reward` → Snapshot at episode 100
- `episode_100/beta` → Snapshot at episode 100
- `episode/avg_beta_100` → 100-episode average beta

**Priority order:**
1. `episode/reward` (highest priority)
2. `dti/reward`
3. `episode_reward`
4. `reward` (fallback)

---

## 📊 Complete Example Output

```
Available tags: 45

Sample tags (first 15):
  - dti/beta
  - dti/reward
  - episode/avg_beta_100
  - episode_100/beta
  - episode_100/reward
  - episode_200/beta
  - episode_200/reward
  ... and 30 more

Searching for tags...

Detected tags:
  Reward (main):                dti/reward
  Beta (main):                  episode/avg_beta_100
  Utilization:                  None
  Per-episode reward tags:      5
    Examples: ['episode_100/reward', 'episode_200/reward', 'episode_300/reward']
  Per-episode beta tags:        5
    Examples: ['episode_100/beta', 'episode_200/beta', 'episode_300/beta']
  Slice allocation tags:        0
  Found detailed metrics for 5 episode snapshots

================================================================================
COMPREHENSIVE EVALUATION METRICS (IEEE Paper Format)
================================================================================

1. QoS Violation Ratio (β)
--------------------------------------------------------------------------------
   Final 100-episode mean:  0.3339 ± 0.0234
   Last episode value:      0.3156
   Best episode value:      0.2754 (episode 450)
   Overall mean:            0.3445 ± 0.0312
   Target achievement (β<0.2):   15.0% of episodes
   Status:                  ✗ NOT ACHIEVED

2. Cumulative Episode Reward
--------------------------------------------------------------------------------
   Final 100-episode mean:  -667.78 ± 45.23
   Last episode value:      -650.34
   Best episode value:      -550.73 (episode 450)
   Overall mean:            -682.34 ± 58.91

7. Per-Episode Detailed Snapshots
--------------------------------------------------------------------------------
   Available snapshots: Episodes [100, 200, 300, 400, 500]

   Episode      Reward          Beta           
   ------------------------------------------
   100          -720.34         0.3567        
   200          -680.12         0.3234        
   300          -650.45         0.3012        
   400          -625.78         0.2856        
   500          -610.23         0.2734        

================================================================================
SUMMARY
================================================================================
Total training episodes: 500

Final Performance (last 100 episodes):
  Average Reward: -667.78
  Average Beta:   0.3339
  Status: ✗ NEEDS IMPROVEMENT
================================================================================
```

---

## ✅ All Requested Features Implemented

1. ✅ Finds `dti/reward` and `dti/beta` tags
2. ✅ Uses `episode_100/reward` and `episode_100/beta` snapshots
3. ✅ Reports **last episode** value
4. ✅ Reports **best episode** value with episode number
5. ✅ Shows dynamic profile details with traffic levels
6. ✅ Shows Beta(alpha, beta) parameters for each level
7. ✅ Per-episode snapshots table

Perfect for your IEEE paper! 📄✨
