# SAC Training with Baseline Comparison

This directory contains two sets of training scripts:

## 1. Standard SAC Training (Original)
- **sac_training.py** - Train SAC agent without baseline comparison
- **evaluate_agent.py** - Evaluate trained SAC agent
- Faster training (no baseline evaluation overhead)
- Use when you just want to train and test SAC

## 2. SAC Training with Baseline Comparison (NEW)
- **sac_training_with_baselines.py** - Train SAC with periodic baseline evaluation
- **evaluate_with_baselines.py** - Comprehensive comparison of SAC vs all baselines
- **baseline_policies.py** - Implementation of 4 baseline methods
- Generates comparison plots and statistical tests
- Use for research papers and comprehensive analysis

---

## Baseline Methods Implemented

### 1. Equal Allocation
- **Formula**: `a_k = C/K` for all slices
- **Characteristics**: Simplest strategy, fair but ignores traffic/QoS
- **Use case**: Baseline for fairness

### 2. Proportional Allocation
- **Formula**: `a_k = C × (x̄_k / Σ_j x̄_j)`
- **Characteristics**: Common heuristic, adapts to traffic
- **Use case**: Industry standard comparison

### 3. Greedy QoS
- **Formula**: Iteratively allocate to slice with highest β_k
- **Characteristics**: QoS-focused, may be unfair
- **Use case**: Shows importance of balance

### 4. Random Policy
- **Formula**: Random allocation normalized to C
- **Characteristics**: Lower bound on performance
- **Use case**: Sanity check

---

## Quick Start

### Option A: Standard Training (No Baselines)
```bash
# Train SAC
python3 sac_training.py

# Evaluate
python3 evaluate_agent.py

# View results
tensorboard --logdir runs/
```

### Option B: Training with Baseline Comparison
```bash
# Train SAC with baseline comparison
python3 sac_training_with_baselines.py

# This will:
# - Evaluate baselines before training (initial comparison)
# - Train SAC for 1000 episodes
# - Evaluate baselines every 100 episodes (periodic comparison)
# - Evaluate baselines after training (final comparison)
# - Log everything to TensorBoard

# View results
tensorboard --logdir runs/
```

### Option C: Evaluate Existing Model with Baselines
```bash
# Comprehensive evaluation and comparison
python3 evaluate_with_baselines.py

# This generates:
# - comparison_metrics.png (bar charts)
# - comparison_distributions.png (box plots)
# - comparison_per_slice.png (allocation patterns)
# - comparison_table.tex (LaTeX table for paper)
# - Statistical significance tests (printed to console)
```

---

## TensorBoard Metrics

### SAC Metrics (both modes)
- `sac/dti/reward` - Per-DTI reward
- `sac/dti/beta` - Per-DTI QoS violation ratio
- `sac/dti/action_slice{k}` - Per-slice allocation
- `sac/episode/reward` - Episode total reward
- `sac/episode/avg_beta` - Episode average beta

### Baseline Metrics (comparison mode only)
- `baseline_initial/{Method}/reward` - Before SAC training
- `baseline_initial/{Method}/beta`
- `baseline_periodic/{Method}/reward` - Every 100 episodes
- `baseline_periodic/{Method}/beta`
- `baseline_final/{Method}/reward` - After SAC training
- `baseline_final/{Method}/beta`

Where `{Method}` ∈ {Equal, Proportional, Greedy, Random, SAC}

---

## File Structure

```
├── config.py                           # Configuration (shared)
├── network_env.py                      # Environment (shared)
├── sac_agent.py                        # SAC agent (shared)
├── traffic_generation.py               # Traffic generator (shared)
│
├── sac_training.py                     # Standard training
├── evaluate_agent.py                   # Standard evaluation
│
├── baseline_policies.py                # NEW: Baseline implementations
├── sac_training_with_baselines.py      # NEW: Training with comparison
├── evaluate_with_baselines.py          # NEW: Comprehensive evaluation
│
├── checkpoints/                        # Saved models
├── runs/                               # TensorBoard logs
└── results/                            # Plots and tables
```

---

## Configuration

Edit `config.py` to adjust:
- Network parameters (K, C, N, W)
- Traffic profiles (static or dynamic)
- QoS thresholds
- SAC hyperparameters
- Number of episodes

Both training modes use the same configuration.

---

## Example Workflow for Research Paper

1. **Initial Training** (get baseline reference):
   ```bash
   python3 sac_training_with_baselines.py
   ```

2. **View Training Progress**:
   ```bash
   tensorboard --logdir runs/
   ```
   - Monitor SAC convergence
   - Compare with baseline performance over time

3. **Comprehensive Evaluation** (after training):
   ```bash
   python3 evaluate_with_baselines.py
   ```

4. **Use Generated Outputs**:
   - Copy `results/comparison_metrics.png` to paper
   - Copy `results/comparison_distributions.png` to paper
   - Copy `results/comparison_table.tex` to paper LaTeX
   - Use statistical significance results in discussion

---

## Performance Expectations

Based on typical scenarios:

| Method        | Expected Beta | Expected Reward | Notes |
|---------------|---------------|-----------------|-------|
| **SAC**       | **0.05-0.15** | **Highest**     | Best overall |
| Equal         | 0.20-0.40     | Medium          | Simple baseline |
| Proportional  | 0.15-0.25     | Medium-High     | Good heuristic |
| Greedy        | 0.10-0.20     | Medium          | QoS-focused |
| Random        | 0.40-0.60     | Lowest          | Lower bound |

*Actual values depend on traffic profiles, thresholds, and network configuration*

---

## Troubleshooting

### Issue: Baselines perform better than SAC
**Solution**: 
- Check if SAC has converged (1000 episodes may not be enough)
- Verify hyperparameters (learning rates, batch size)
- Ensure replay buffer has sufficient data

### Issue: All methods have high beta
**Solution**:
- QoS thresholds may be too strict
- Run `python3 debug_beta.py` to analyze threshold appropriateness
- Consider adjusting thresholds in `config.py`

### Issue: Greedy baseline crashes
**Solution**:
- Ensure QoS tables are loaded correctly
- Check that QoS file paths in config.py are correct
- Greedy needs QoS tables to estimate violations

---

## Citation

If you use the baseline comparison framework in your research, please cite:

```bibtex
@article{your_paper,
  title={Deep Reinforcement Learning for Dynamic Resource Allocation 
         in Multi-Slice 5G/6G Radio Access Networks},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

---

## Contact

For questions or issues, please contact [your email].
