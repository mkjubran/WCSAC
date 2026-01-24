# Quick Comparison: Standard vs Baseline Mode

## Two Complete Sets of Code

### SET 1: Standard SAC Training (Original)
**Files**: `sac_training.py`, `evaluate_agent.py`

**Use when**:
- Quick experiments
- Hyperparameter tuning
- Initial development
- Just want SAC results

**Outputs**:
- SAC training curves
- SAC performance metrics
- Single method analysis

**Command**:
```bash
python3 sac_training.py
```

---

### SET 2: SAC with Baseline Comparison (NEW - For Papers)
**Files**: `sac_training_with_baselines.py`, `evaluate_with_baselines.py`, `baseline_policies.py`

**Use when**:
- Writing research papers
- Need comprehensive comparison
- Want to show SAC superiority
- Statistical significance required

**Outputs**:
- SAC training curves
- Baseline performance at 3 stages:
  - Initial (before SAC training)
  - Periodic (every 100 episodes)
  - Final (after SAC training)
- Comparison plots
- Statistical tests
- LaTeX table

**Command**:
```bash
python3 sac_training_with_baselines.py  # Train with comparison
python3 evaluate_with_baselines.py      # Detailed evaluation
```

---

## Key Differences

| Feature | Standard Mode | Baseline Mode |
|---------|---------------|---------------|
| Training speed | ✓ Faster | Slower (baseline eval) |
| Memory usage | ✓ Lower | Higher (multiple policies) |
| TensorBoard metrics | SAC only | SAC + 4 baselines |
| Generated plots | Basic SAC | Comprehensive comparison |
| Statistical tests | No | ✓ Yes (t-tests, p-values) |
| LaTeX table | No | ✓ Yes (auto-generated) |
| Paper-ready figures | No | ✓ Yes |

---

## What You Get from Baseline Mode

### TensorBoard Metrics
```
runs/
└── sac_with_baselines_K2_W5_20240124_123456/
    ├── sac/
    │   ├── dti/reward
    │   ├── dti/beta
    │   └── episode/...
    ├── baseline_initial/
    │   ├── Equal/{reward, beta, violations}
    │   ├── Proportional/{reward, beta, violations}
    │   ├── Greedy/{reward, beta, violations}
    │   └── Random/{reward, beta, violations}
    ├── baseline_periodic/
    │   └── (same structure, logged every 100 episodes)
    └── baseline_final/
        └── (same structure, final evaluation)
```

### Generated Files (from evaluate_with_baselines.py)
```
results/
├── comparison_metrics.png          # 4 bar charts (reward, beta, violations, utilization)
├── comparison_distributions.png    # 2 box plots (reward, beta distributions)
├── comparison_per_slice.png        # Per-slice allocation patterns
└── comparison_table.tex            # Ready for paper
```

### Console Output Example
```
STATISTICAL SIGNIFICANCE TESTS
======================================================================

Paired t-tests (SAC vs Baselines):
----------------------------------------------------------------------

SAC vs Equal:
  Reward: t=12.453, p=0.0001 ***
  Beta:   t=-8.234, p=0.0001 ***

SAC vs Proportional:
  Reward: t=5.678, p=0.0023 **
  Beta:   t=-4.321, p=0.0089 **

SAC vs Greedy:
  Reward: t=3.456, p=0.0234 *
  Beta:   t=-2.987, p=0.0345 *

SAC vs Random:
  Reward: t=18.901, p=0.0000 ***
  Beta:   t=-15.234, p=0.0000 ***

Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant
```

---

## Recommended Workflow

### For Development
1. Use **Standard Mode** during development
2. Quick iterations with `sac_training.py`
3. Test different configurations rapidly

### For Paper Writing
1. Once you have good SAC hyperparameters, switch to **Baseline Mode**
2. Run `sac_training_with_baselines.py` (takes longer but worth it)
3. Run `evaluate_with_baselines.py` for comprehensive analysis
4. Copy generated figures and table directly to paper
5. Use statistical test results in discussion

---

## Example Paper Sections

### Using Baseline Mode Outputs:

**Figure Caption**:
> "Performance comparison of SAC against baseline methods. (a) Average episode reward, (b) QoS violation ratio (β), (c) Constraint violations, (d) Resource utilization. SAC achieves significantly lower β while maintaining high reward. Error bars show standard deviation over 100 evaluation episodes. *** indicates p < 0.001."

**Table Caption**:
> "Quantitative comparison of SAC and baseline methods. Win Rate indicates percentage of episodes achieving β < 0.2. SAC outperforms all baselines with statistical significance (p < 0.001)."

**Discussion**:
> "Statistical analysis reveals that SAC significantly outperforms all baseline methods. Paired t-tests show SAC achieves 45% lower β compared to Equal Allocation (p < 0.001), 28% lower than Proportional Allocation (p < 0.01), and 15% lower than Greedy QoS (p < 0.05)..."

---

## Storage Requirements

### Standard Mode
- TensorBoard logs: ~50 MB
- Checkpoints: ~10 MB
- Total: ~60 MB

### Baseline Mode
- TensorBoard logs: ~200 MB (5× baselines × periodic eval)
- Checkpoints: ~10 MB
- Plots: ~5 MB
- Total: ~215 MB

**Recommendation**: Use baseline mode for final experiments only.

---

## Which Mode to Use?

✅ **Use Standard Mode** if:
- Developing and testing
- Hyperparameter search
- Quick experiments
- Limited compute time
- Don't need comparison

✅ **Use Baseline Mode** if:
- Writing a paper
- Need to show superiority
- Want statistical tests
- Comprehensive analysis
- Have compute time (~3-4× longer)

---

## Both Modes Share

All core files are identical:
- `config.py` - Same configuration
- `network_env.py` - Same environment
- `sac_agent.py` - Same SAC implementation
- `traffic_generation.py` - Same traffic profiles

Only difference is **evaluation and logging**.

---

## Summary

You now have **TWO COMPLETE SYSTEMS**:

1. **Fast SAC Training**: For development
2. **SAC with Baselines**: For publication

Both use the same core code. Switch between them based on your needs. No modifications required to existing files!
