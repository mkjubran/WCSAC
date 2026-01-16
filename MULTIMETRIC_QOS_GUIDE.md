# Multi-Metric QoS Guide

## Why Preserve All Metrics?

Your JSON files contain multiple QoS metrics:
- **VoIP:** 7 metrics (loss, delay, jitter, throughput, etc.)
- **CBR Video:** 2 metrics (delay, throughput)
- **Video Conferencing:** 2 metrics (delay, segment loss)

Now **ALL metrics are preserved** and you can choose which one to use!

---

## Step 1: Convert with ALL Metrics Preserved

```bash
python convert_qos_multimetric.py
```

**Output files:**
- `qos_voip_all_metrics.json` - All 7 VoIP metrics
- `qos_cbr_all_metrics.json` - All 2 CBR metrics
- `qos_video_all_metrics.json` - All 2 Video metrics

**Example structure:**
```json
{
  "5": {
    "1": {
      "voIPFrameLoss": {"mu": 0.149, "sigma": 0.525},
      "voIPFrameDelay": {"mu": 5.614, "sigma": 3.180},
      "voIPJitter": {"mu": 6.283, "sigma": 12.191},
      "voIPReceivedThroughput": {"mu": 1408.872, "sigma": 261.141},
      ...
    }
  }
}
```

---

## Step 2: Configure - Choose Which Metrics to Use

### Edit `config.py`:

```python
K = 3

# Set thresholds (must match the metric you choose!)
THRESHOLDS = [
    0.5,   # VoIP: 0.5% loss
    20.0,  # CBR: 20ms delay
    40.0   # Video: 40% segment loss
]

# Point to multi-metric JSON files
QOS_TABLE_FILES = [
    'qos_voip_all_metrics.json',
    'qos_cbr_all_metrics.json',
    'qos_video_all_metrics.json'
]

# Choose which metric to use for each slice
QOS_METRICS = [
    'voIPFrameLoss',                      # VoIP: use loss
    'cbrFrameDelay',                       # CBR: use delay
    'rtVideoStreamingSegmentLoss'          # Video: use segment loss
]
```

---

## Available Metrics

### VoIP (7 metrics):
```python
QOS_METRICS = [
    'voIPFrameLoss',              # Packet loss (%)
    'voIPFrameDelay',             # Frame delay (ms)
    'voIPJitter',                 # Jitter (ms)
    'voIPReceivedThroughput',     # Received throughput (bps)
    'voIPGeneratedThroughput',    # Generated throughput (bps)
    'voIPPlayoutDelay',           # Playout delay (ms)
    'voIPPlayoutLoss'             # Playout loss (%)
]
```

### CBR Video (2 metrics):
```python
QOS_METRICS = [
    'cbrFrameDelay',              # Frame delay (ms)
    'cbrReceivedThroughput'       # Received throughput (bps)
]
```

### Video Conferencing (2 metrics):
```python
QOS_METRICS = [
    'rtVideoStreamingEnd2endDelaySegment',  # End-to-end delay (ms)
    'rtVideoStreamingSegmentLoss'           # Segment loss (%)
]
```

---

## Configuration Examples

### Example 1: Use Loss for All Slices
```python
K = 3

QOS_TABLE_FILES = [
    'qos_voip_all_metrics.json',
    'qos_cbr_all_metrics.json',  # CBR doesn't have loss, will fail!
    'qos_video_all_metrics.json'
]

QOS_METRICS = [
    'voIPFrameLoss',                      # ✓ Available
    'cbrFrameLoss',                        # ✗ Not available!
    'rtVideoStreamingSegmentLoss'          # ✓ Available
]

THRESHOLDS = [0.5, 0.5, 40.0]  # Loss thresholds
```

### Example 2: Use Delay for All Slices
```python
K = 3

QOS_TABLE_FILES = [
    'qos_voip_all_metrics.json',
    'qos_cbr_all_metrics.json',
    'qos_video_all_metrics.json'
]

QOS_METRICS = [
    'voIPFrameDelay',                           # ✓ Available
    'cbrFrameDelay',                             # ✓ Available
    'rtVideoStreamingEnd2endDelaySegment'        # ✓ Available
]

THRESHOLDS = [10.0, 20.0, 30.0]  # Delay thresholds in ms
```

### Example 3: Mixed Metrics (Recommended)
```python
K = 3

QOS_TABLE_FILES = [
    'qos_voip_all_metrics.json',
    'qos_cbr_all_metrics.json',
    'qos_video_all_metrics.json'
]

QOS_METRICS = [
    'voIPFrameLoss',                      # VoIP: loss (%)
    'cbrFrameDelay',                       # CBR: delay (ms)
    'rtVideoStreamingSegmentLoss'          # Video: loss (%)
]

THRESHOLDS = [
    0.5,   # VoIP: 0.5% loss acceptable
    20.0,  # CBR: 20ms delay acceptable
    40.0   # Video: 40% segment loss acceptable
]
```

### Example 4: Auto-Select First Metric
```python
K = 3

QOS_TABLE_FILES = [
    'qos_voip_all_metrics.json',
    'qos_cbr_all_metrics.json',
    'qos_video_all_metrics.json'
]

QOS_METRICS = [None, None, None]  # Auto-select first available metric

THRESHOLDS = [0.5, 20.0, 40.0]  # Adjust based on auto-selected metrics
```

---

## Step 3: Run Training

```bash
python sac_training.py
```

**Output:**
```
Loaded QoS table for slice 0 from qos_voip_all_metrics.json (using metric: voIPFrameLoss)
Loaded QoS table for slice 1 from qos_cbr_all_metrics.json (using metric: cbrFrameDelay)
Loaded QoS table for slice 2 from qos_video_all_metrics.json (using metric: rtVideoStreamingSegmentLoss)

Training starts...
```

---

## Advantages of Multi-Metric Approach

### 1. **Flexibility**
Try different metrics without re-converting files:
```python
# Experiment 1: Loss-based
QOS_METRICS = ['voIPFrameLoss', 'cbrFrameDelay', 'rtVideoStreamingSegmentLoss']

# Experiment 2: Delay-based
QOS_METRICS = ['voIPFrameDelay', 'cbrFrameDelay', 'rtVideoStreamingEnd2endDelaySegment']

# Experiment 3: Jitter-based
QOS_METRICS = ['voIPJitter', 'cbrFrameDelay', 'rtVideoStreamingEnd2endDelaySegment']
```

### 2. **Single Conversion**
Convert once, use many times:
```bash
python convert_qos_multimetric.py  # Run once
# Now experiment with different metrics in config.py
```

### 3. **Comprehensive Analysis**
Compare agent performance across different QoS metrics:
- Train with loss → Evaluate with delay
- Train with delay → Evaluate with throughput

### 4. **Metric Correlation**
Study relationships between metrics:
- Does optimizing for loss improve delay?
- Does improving throughput reduce jitter?

---

## Matching Thresholds to Metrics

⚠️ **CRITICAL:** Thresholds must match your chosen metric's scale!

| Metric Type | Typical Range | Example Threshold |
|-------------|---------------|-------------------|
| Loss (%) | 0-100% | 0.5%, 1.0%, 5.0% |
| Delay (ms) | 1-1000ms | 10ms, 20ms, 50ms |
| Jitter (ms) | 1-100ms | 5ms, 10ms, 20ms |
| Throughput (bps) | 100-5000 | 1000, 2000, 3000 |

**Example:**
```python
# Using loss metrics
QOS_METRICS = ['voIPFrameLoss', 'cbrFrameDelay', 'rtVideoStreamingSegmentLoss']
THRESHOLDS = [0.5, 20.0, 40.0]  # 0.5% loss, 20ms delay, 40% loss

# Using delay metrics
QOS_METRICS = ['voIPFrameDelay', 'cbrFrameDelay', 'rtVideoStreamingEnd2endDelaySegment']
THRESHOLDS = [10.0, 20.0, 30.0]  # All in ms
```

---

## Verification

Check which metrics are available:
```bash
python convert_qos_multimetric.py
```

Output shows:
```
Available metrics in voip_qos.json:
  1. voIPFrameLoss (%)
  2. voIPFrameDelay (ms)
  3. voIPJitter (ms)
  4. voIPReceivedThroughput (bps)
  5. voIPGeneratedThroughput (bps)
  6. voIPPlayoutDelay (ms)
  7. voIPPlayoutLoss (%)

Available metrics in cbr_qos.json:
  1. cbrFrameDelay (ms)
  2. cbrReceivedThroughput (bps)

Available metrics in video_qos.json:
  1. rtVideoStreamingEnd2endDelaySegment (ms)
  2. rtVideoStreamingSegmentLoss (%)
```

---

## Summary

**Old approach (single metric):**
```python
# Only preserved one metric per file
convert_qos_file('voip_qos.json', 'output.json', metric_name='voIPFrameLoss')
# Had to re-convert to try different metric
```

**New approach (all metrics):**
```python
# Preserve ALL metrics in one conversion
convert_qos_file_all_metrics('voip_qos.json', 'qos_voip_all_metrics.json')

# Choose metric in config.py
QOS_METRICS = ['voIPFrameLoss']  # or 'voIPFrameDelay', 'voIPJitter', etc.
# No need to re-convert!
```

**Benefits:**
✅ Convert once, use many times  
✅ Easy to experiment with different metrics  
✅ Compare agent performance across metrics  
✅ All data preserved in one place  

🚀 Much more flexible!
