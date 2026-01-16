# SAC for Multi-Slice Network Resource Allocation

Complete implementation of Soft Actor-Critic (SAC) algorithm for multi-slice 5G/6G network resource allocation with QoS constraints.

**Implements Algorithms from LaTeX:**
- Algorithm 1: Multi-Slice Network Environment
- Algorithm 2: SAC Training
- Algorithm 4: Traffic Generation

---

## 📦 **Files to Download**

### **Essential Files (Required to Run)**

| File | Size | Description |
|------|------|-------------|
| `config.py` | 4.6 KB | **Configuration file** - Set all parameters here |
| `traffic_generation.py` | 6.0 KB | Traffic generation (Algorithm 4) |
| `network_env.py` | 13 KB | Network environment (Algorithm 1) |
| `sac_agent.py` | 13 KB | SAC agent (Algorithm 2) |
| `sac_training.py` | 21 KB | **Main training script** |
| `evaluate_agent.py` | 12 KB | Evaluation and visualization |

### **QoS Conversion (If Using Your Own QoS Data)**

| File | Description |
|------|-------------|
| `convert_qos_multimetric.py` | **Converter for your QoS JSON files** |

### **Documentation**

| File | Description |
|------|-------------|
| `README.md` | This file |
| `MULTIMETRIC_QOS_GUIDE.md` | Guide for using your QoS files |
| `algorithm_multislice_final.tex` | Complete LaTeX algorithms |

**Total: 6 Python files minimum to run**

---

## 🚀 **Quick Start (3 Steps)**

### **Step 1: Install Dependencies**

```bash
pip install numpy scipy torch matplotlib seaborn tensorboard tqdm pandas
```

### **Step 2: Configure Parameters**

Edit `config.py`:

```python
# Network topology
K = 3              # Number of slices
C = 8              # Total RB capacity
N = 20             # TTIs per DTI

# QoS thresholds
THRESHOLDS = [0.5, 20.0, 40.0]  # One per slice

# Traffic profiles
TRAFFIC_PROFILES = ['low', 'medium', 'high']

# Training
NUM_EPISODES = 1000
DEVICE = 'cpu'  # or 'cuda' for GPU
```

### **Step 3: Train**

```bash
python sac_training.py
```

**That's it!** Training starts immediately.

---

## 📊 **Using Your Own QoS Data**

If you have existing QoS JSON files (like VoIP, CBR, Video data):

### **Step 1: Convert Your Files**

Place your JSON files in the directory:
```
voip_qos.json
cbr_qos.json
video_qos.json
```

Run converter:
```bash
python convert_qos_multimetric.py
```

**Output:**
```
qos_voip_all_metrics.json
qos_cbr_all_metrics.json
qos_video_all_metrics.json
```

### **Step 2: Configure QoS**

Edit `config.py`:

```python
K = 3

# Point to your converted QoS files
QOS_TABLE_FILES = [
    'qos_voip_all_metrics.json',
    'qos_cbr_all_metrics.json',
    'qos_video_all_metrics.json'
]

# Choose which metric to use from each file
QOS_METRICS = [
    'voIPFrameLoss',                      # VoIP: use loss
    'cbrFrameDelay',                       # CBR: use delay
    'rtVideoStreamingSegmentLoss'          # Video: use loss
]

# Set thresholds matching your metrics (IMPORTANT!)
THRESHOLDS = [
    0.5,   # VoIP loss: 0.5%
    20.0,  # CBR delay: 20ms
    40.0   # Video loss: 40%
]
```

### **Step 3: Train**

```bash
python sac_training.py
```

**See `MULTIMETRIC_QOS_GUIDE.md` for detailed QoS instructions.**

---

## 📈 **Monitoring Training**

### **TensorBoard**

```bash
tensorboard --logdir runs/
```

Open: http://localhost:6006

**Metrics tracked:**
- Episode rewards
- Beta (QoS violations)
- Constraint violations
- Q-values, losses

### **Training Curves**

Automatically saved to `checkpoints/` directory.

---

## 🧪 **Evaluation**

```bash
python evaluate_agent.py
```

Generates comprehensive analysis plots in `results/` directory.

---

## ⚙️ **Configuration Reference**

All parameters in `config.py`:

```python
# Environment (Algorithm 1)
K = 3                              # Number of slices
C = 8                              # Total RB capacity
N = 20                             # TTIs per DTI
THRESHOLDS = [0.2, 0.15, 0.25]    # QoS threshold per slice
LAMBDA = 0.5                       # Reward weight
W = 5                              # Window size (None = ∞)

# Traffic (Algorithm 4)
TRAFFIC_PROFILES = ['uniform', 'low', 'medium']

# QoS Tables
QOS_TABLE_FILES = [None, None, None]  # Or your converted files
QOS_METRICS = [None, None, None]      # Which metric to use

# SAC (Algorithm 2)
NUM_EPISODES = 1000                # Training episodes
LR_ACTOR = 3e-4                    # Learning rates
LR_CRITIC = 3e-4
GAMMA = 0.99                       # Discount factor
BATCH_SIZE = 256
DEVICE = 'cpu'                     # or 'cuda'
```

---

## 🎮 **Usage Examples**

### **Example 1: Default Configuration**

```bash
python sac_training.py
```

### **Example 2: Large Network**

Edit `config.py`:
```python
K = 4
C = 16
THRESHOLDS = [0.2, 0.15, 0.25, 0.18]
NUM_EPISODES = 2000
```

Run:
```bash
python sac_training.py
```

### **Example 3: GPU Training**

Edit `config.py`:
```python
DEVICE = 'cuda'
BATCH_SIZE = 512
```

### **Example 4: With Your QoS Data**

1. Convert: `python convert_qos_multimetric.py`
2. Configure `config.py` with QoS files
3. Train: `python sac_training.py`

---

## 🔬 **Testing Components**

```bash
# Test traffic generation
python traffic_generation.py

# Test environment
python network_env.py

# View configuration
python config.py
```

---

## 📊 **Key Algorithms**

### **Algorithm 1: Network Environment** (`network_env.py`)

**State:** `s = (β, CDF_1, ..., CDF_K)`
- β: Violation ratio [0,1]
- CDF_k: Traffic CDF for slice k

**Action:** `a = [a_1, ..., a_K]` (continuous, Σa_k ≤ C)

**Reward:** `R = -β + λ·(C - Σa_k)/C`

### **Algorithm 2: SAC Training** (`sac_training.py`)

- Softmax action parameterization
- Twin critics
- Replay buffer
- Episodic learning with reset

### **Algorithm 4: Traffic Generation** (`traffic_generation.py`)

**Profiles:**
- Uniform: All levels equally likely
- Low: Beta(2,5) - 5-30 UEs
- Medium: Beta(2,2) - 25-55 UEs
- High: Beta(5,2) - 50-80 UEs
- External: Load from file

---

## 🎯 **Troubleshooting**

| Issue | Solution |
|-------|----------|
| Module not found | `pip install numpy scipy torch matplotlib seaborn tensorboard tqdm pandas` |
| QoS file not found | Check paths in `config.py`, run converter |
| CUDA out of memory | Reduce `BATCH_SIZE` in `config.py` |
| High beta | Adjust `THRESHOLDS` (make less strict) |
| Training slow | Use `DEVICE = 'cuda'`, reduce episodes |
| Thresholds mismatch | Ensure `len(THRESHOLDS) == K` |

---

## 📚 **Key Concepts**

### **β (Beta): Violation Ratio**
```
β = (violated_traffic) / (total_traffic)
```
- 0.0 = Perfect
- 1.0 = All violated
- **Lower is better**

### **Sliding Window (W)**
- `W = None`: Full history
- `W = 5`: Last 5 DTIs
- `W = 1`: Current DTI only

### **Reward Lambda (λ)**
- λ = 0.0: Only QoS
- λ = 0.5: Balanced ✓
- λ = 1.0: Equal weight

---

## 📁 **Output Structure**

```
project/
├── config.py              # Your configuration
├── sac_training.py        # Run this to train
│
├── checkpoints/           # Generated
│   └── *.pt              # Model checkpoints
│
├── runs/                  # TensorBoard logs
│   └── sac_*/
│
└── results/               # Evaluation plots
    └── *.png
```

---

## ✅ **Checklist**

Before training:
- [ ] Downloaded 6 Python files
- [ ] Installed: `pip install numpy scipy torch matplotlib seaborn tensorboard tqdm pandas`
- [ ] Configured `config.py`
- [ ] (Optional) Converted QoS files
- [ ] Set thresholds matching your metrics

Ready:
```bash
python sac_training.py
```

---

## 🚀 **Summary**

**Basic Usage:**
1. Download 6 files
2. Edit `config.py`
3. `python sac_training.py`

**With Your QoS:**
1. `python convert_qos_multimetric.py`
2. Configure in `config.py`
3. `python sac_training.py`

**Monitor:**
```bash
tensorboard --logdir runs/
```

**Evaluate:**
```bash
python evaluate_agent.py
```

---

## 📖 **Documentation Files**

- **`MULTIMETRIC_QOS_GUIDE.md`** - Complete QoS guide
- **`algorithm_multislice_final.tex`** - LaTeX algorithms
- **`README.md`** - This file

---

**Version:** 1.0  
**Last Updated:** January 2025  
**Python:** 3.8+  
**PyTorch:** 1.10+

---

## 🎓 **Quick Reference**

**Train:** `python sac_training.py`  
**Evaluate:** `python evaluate_agent.py`  
**Monitor:** `tensorboard --logdir runs/`  
**Configure:** Edit `config.py`  
**QoS Convert:** `python convert_qos_multimetric.py`

**That's it!** 🎯
