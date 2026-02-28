"""
STEP 2: Generate Figures and Tables from Extracted Data
Fast iteration - no TensorBoard loading!

Usage:
    python3 step2_generate_figures.py --data extracted_data.json --output-dir ./paper_figures

Modify figure aesthetics, labels, colors, etc. and re-run instantly!
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================================
# FIGURE STYLE CONFIGURATION
# You can easily modify these settings and re-run
# ============================================================================

# Publication-quality settings
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'  # or 'pdf' for LaTeX

# Style
sns.set_style("whitegrid")
plt.rcParams.update({
    'figure.dpi': FIGURE_DPI,
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'lines.linewidth': 1.5,
    'grid.alpha': 0.3,
})

# Colors (easy to customize)
COLOR_REWARD = 'blue'
COLOR_BETA = 'red'
COLOR_TARGET = 'green'
COLOR_HOMOGENEOUS = 'steelblue'
COLOR_HETEROGENEOUS = 'coral'
COLOR_DYNAMIC = 'purple'

# Labels (easy to customize)
LABEL_BETA = 'QoS Violation Ratio (β)'
LABEL_REWARD = 'Episode Reward'
LABEL_EPISODE = 'Training Episode'
LABEL_DTI = 'Decision Time Interval (DTI)'
LABEL_RBS = 'Resource Blocks Allocated'

# Target threshold
BETA_TARGET = 0.2


def load_extracted_data(json_path):
    """Load extracted data from JSON."""
    print(f"Loading data from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"  ✓ Loaded {data['metadata']['num_experiments']} experiments")
    print(f"  Extraction date: {data['metadata']['extraction_date']}")
    
    return data


def categorize_experiments(experiments):
    """Organize experiments by category."""
    categories = {
        'static_homogeneous': [],
        'static_heterogeneous': [],
        'dynamic': []
    }
    
    for exp in experiments:
        cat = exp['category']
        categories[cat].append(exp)
    
    print(f"\nExperiments by category:")
    print(f"  Homogeneous: {len(categories['static_homogeneous'])}")
    print(f"  Heterogeneous: {len(categories['static_heterogeneous'])}")
    print(f"  Dynamic: {len(categories['dynamic'])}")
    
    return categories


# ============================================================================
# FIGURE 1: Training Convergence
# ============================================================================

def fig1_training_convergence(experiments, output_dir):
    """
    Figure 1: Training convergence (reward and beta over episodes)
    Choose Medium-Medium scenario if available
    """
    print("\n[Figure 1] Training Convergence...")
    
    # Find Medium-Medium experiment
    exp = None
    for e in experiments:
        scenario_lower = e['scenario_str'].lower()
        if 'medium' in scenario_lower and e['category'] == 'static_homogeneous':
            exp = e
            break
    
    if not exp:
        exp = experiments[0] if experiments else None
    
    if not exp:
        print("  ⚠️  No experiments available")
        return
    
    print(f"  Using: {exp['scenario_str']}")
    
    data = exp['data']
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Top: Reward
    if 'episode_reward' in data:
        steps = np.array(data['episode_reward']['steps'])
        values = np.array(data['episode_reward']['values'])
        
        axes[0].plot(steps, values, alpha=0.3, color=COLOR_REWARD, linewidth=0.5, label='Raw')
        
        # Moving average
        if len(values) >= 100:
            ma = np.convolve(values, np.ones(100)/100, mode='valid')
            axes[0].plot(steps[99:], ma, color=COLOR_REWARD, linewidth=2, label='100-ep MA')
        
        axes[0].set_ylabel(LABEL_REWARD)
        axes[0].set_title('Training Convergence: Episode Reward')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Bottom: Beta
    if 'episode_beta' in data:
        steps = np.array(data['episode_beta']['steps'])
        values = np.array(data['episode_beta']['values'])
        
        axes[1].plot(steps, values, alpha=0.3, color=COLOR_BETA, linewidth=0.5, label='Raw')
        
        # Moving average
        if len(values) >= 100:
            ma = np.convolve(values, np.ones(100)/100, mode='valid')
            axes[1].plot(steps[99:], ma, color=COLOR_BETA, linewidth=2, label='100-ep MA')
        
        axes[1].set_xlabel(LABEL_EPISODE)
        axes[1].set_ylabel(LABEL_BETA)
        axes[1].set_title('Training Convergence: QoS Performance')
        axes[1].axhline(y=BETA_TARGET, color=COLOR_TARGET, linestyle='--', 
                       label=f'Target (β < {BETA_TARGET})', alpha=0.5)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'fig1_training_convergence.{FIGURE_FORMAT}')
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")


# ============================================================================
# FIGURE 2: Static Homogeneous Performance
# ============================================================================

def fig2_static_homogeneous(exps_by_cat, output_dir):
    """
    Figure 2: Bar chart of beta across homogeneous traffic profiles
    """
    print("\n[Figure 2] Static Homogeneous Performance...")
    
    exps = exps_by_cat['static_homogeneous']
    if not exps:
        print("  ⚠️  No homogeneous experiments")
        return
    
    # Sort by load level
    load_order = {'extremely_low': 0, 'low': 1, 'medium': 2, 'high': 3, 'extremely_high': 4}
    
    def get_load_order(exp):
        profile = list(exp['scenario'].values())[0]
        return load_order.get(profile, 999)
    
    exps = sorted(exps, key=get_load_order)
    
    # Extract data
    profiles = []
    betas = []
    stds = []
    
    for exp in exps:
        profile = list(exp['scenario'].values())[0].replace('_', ' ').title()
        stats = exp['statistics'].get('beta')
        
        if stats:
            # Try last_100_mean first, fall back to mean
            beta_val = stats.get('last_100_mean', stats.get('mean'))
            std_val = stats.get('last_100_std', stats.get('std'))
            
            if beta_val is not None and std_val is not None:
                profiles.append(profile)
                betas.append(beta_val)
                stds.append(std_val)
    
    if not profiles:
        print("  ⚠️  No valid data")
        print(f"  Debug: Found {len(exps)} homogeneous experiments")
        for exp in exps:
            print(f"    - {exp['scenario_str']}: stats = {exp['statistics'].get('beta')}")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(profiles))
    ax.bar(x_pos, betas, yerr=stds, capsize=5, 
           color=COLOR_HOMOGENEOUS, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax.set_xlabel('Traffic Profile')
    ax.set_ylabel(LABEL_BETA)
    ax.set_title('SAC Performance Across Static Homogeneous Traffic')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(profiles, rotation=45, ha='right')
    ax.axhline(y=BETA_TARGET, color=COLOR_TARGET, linestyle='--', 
               label=f'Target', alpha=0.7)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'fig2_static_homogeneous.{FIGURE_FORMAT}')
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")


# ============================================================================
# FIGURE 3: Heterogeneous Performance
# ============================================================================

def fig3_heterogeneous(exps_by_cat, output_dir):
    """
    Figure 3: Bar chart of beta across heterogeneous scenarios
    """
    print("\n[Figure 3] Heterogeneous Performance...")
    
    exps = exps_by_cat['static_heterogeneous']
    if not exps:
        print("  ⚠️  No heterogeneous experiments")
        return
    
    scenarios = []
    betas = []
    stds = []
    
    for exp in exps:
        stats = exp['statistics'].get('beta')
        
        if stats:
            # Try last_100_mean first, fall back to mean
            beta_val = stats.get('last_100_mean', stats.get('mean'))
            std_val = stats.get('last_100_std', stats.get('std'))
            
            if beta_val is not None and std_val is not None:
                scenarios.append(exp['scenario_str'])
                betas.append(beta_val)
                stds.append(std_val)
    
    if not scenarios:
        print("  ⚠️  No valid data")
        print(f"  Debug: Found {len(exps)} heterogeneous experiments")
        for exp in exps:
            print(f"    - {exp['scenario_str']}: stats = {exp['statistics'].get('beta')}")
        return
    
    # Ensure all lists are same length
    assert len(scenarios) == len(betas) == len(stds), f"List length mismatch: {len(scenarios)}, {len(betas)}, {len(stds)}"
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(scenarios))
    ax.bar(x_pos, betas, yerr=stds, capsize=5,
           color=COLOR_HETEROGENEOUS, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax.set_xlabel('Traffic Scenario')
    ax.set_ylabel(LABEL_BETA)
    ax.set_title('SAC Performance Under Heterogeneous Traffic')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.axhline(y=BETA_TARGET, color=COLOR_TARGET, linestyle='--', 
               label='Target', alpha=0.7)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'fig3_heterogeneous.{FIGURE_FORMAT}')
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")


# ============================================================================
# FIGURE 4: Allocation Patterns (Episode 80)
# ============================================================================

def fig4_allocation_patterns(experiments, output_dir):
    """
    Figure 4: Box plots of resource allocation per slice
    """
    print("\n[Figure 4] Allocation Patterns (Episode 80)...")
    
    # Use heterogeneous experiment if available
    exp = None
    for e in experiments:
        if e['category'] == 'static_heterogeneous':
            exp = e
            break
    
    if not exp:
        exp = experiments[0] if experiments else None
    
    if not exp:
        print("  ⚠️  No experiments available")
        return
    
    print(f"  Using: {exp['scenario_str']}")
    
    data = exp['data']
    
    # Extract allocations
    allocations = {}
    for key in ['ep80_action_slice0', 'ep80_action_slice1']:
        if key in data:
            slice_num = key.split('slice')[-1]
            allocations[f'Slice {slice_num}'] = data[key]['values']
    
    if not allocations:
        print("  ⚠️  No episode 80 allocation data")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.boxplot(allocations.values(), labels=allocations.keys(),
               patch_artist=True,
               boxprops=dict(facecolor='lightblue', alpha=0.7),
               medianprops=dict(color='red', linewidth=2))
    
    ax.set_ylabel(LABEL_RBS)
    ax.set_title(f'Allocation Distribution - {exp["scenario_str"]} (Episode 80)')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'fig4_allocation_patterns.{FIGURE_FORMAT}')
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")


# ============================================================================
# FIGURE 5: Dynamic Adaptation
# ============================================================================

def fig5_dynamic_adaptation(exps_by_cat, output_dir):
    """
    Figure 5: Time series showing adaptation to dynamic traffic
    """
    print("\n[Figure 5] Dynamic Adaptation...")
    
    exps = exps_by_cat['dynamic']
    if not exps:
        print("  ⚠️  No dynamic experiments")
        return
    
    exp = exps[0]
    data = exp['data']
    
    if 'dti_beta' not in data:
        print("  ⚠️  No DTI-level beta data")
        return
    
    steps = np.array(data['dti_beta']['steps'])
    betas = np.array(data['dti_beta']['values'])
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Top: Beta
    axes[0].plot(steps, betas, color=COLOR_BETA, alpha=0.7, linewidth=1)
    axes[0].set_ylabel(LABEL_BETA)
    axes[0].set_title('Dynamic Traffic Adaptation (Switches Every 100 DTIs)')
    axes[0].axhline(y=BETA_TARGET, color=COLOR_TARGET, linestyle='--', 
                   alpha=0.5, label='Target')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Mark switches
    for switch in range(100, int(max(steps)), 100):
        axes[0].axvline(x=switch, color='gray', linestyle=':', alpha=0.5)
        axes[1].axvline(x=switch, color='gray', linestyle=':', alpha=0.5)
    
    # Bottom: Allocations
    for key in ['dti_action_slice0', 'dti_action_slice1']:
        if key in data:
            slice_num = key.split('slice')[-1]
            alloc_steps = np.array(data[key]['steps'])
            alloc_values = np.array(data[key]['values'])
            axes[1].plot(alloc_steps, alloc_values, 
                        label=f'Slice {slice_num}', alpha=0.7, linewidth=1)
    
    axes[1].set_xlabel(LABEL_DTI)
    axes[1].set_ylabel(LABEL_RBS)
    axes[1].set_title('Per-Slice Allocation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'fig5_dynamic_adaptation.{FIGURE_FORMAT}')
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")


# ============================================================================
# TABLES
# ============================================================================

def generate_summary_table(experiments, output_dir):
    """Generate LaTeX summary table."""
    print("\n[Table] Summary Table...")
    
    rows = []
    for exp in experiments:
        beta_stats = exp['statistics'].get('beta')
        reward_stats = exp['statistics'].get('reward')
        
        # Handle beta
        if beta_stats:
            beta_mean = beta_stats.get('last_100_mean', beta_stats.get('mean'))
            beta_std = beta_stats.get('last_100_std', beta_stats.get('std'))
            if beta_mean is not None and beta_std is not None:
                beta_str = f"{beta_mean:.3f} $\\pm$ {beta_std:.3f}"
            else:
                beta_str = "N/A"
        else:
            beta_str = "N/A"
        
        # Handle reward
        if reward_stats:
            reward_mean = reward_stats.get('last_100_mean', reward_stats.get('mean'))
            reward_std = reward_stats.get('last_100_std', reward_stats.get('std'))
            if reward_mean is not None and reward_std is not None:
                reward_str = f"{reward_mean:.2f} $\\pm$ {reward_std:.2f}"
            else:
                reward_str = "N/A"
        else:
            reward_str = "N/A"
        
        rows.append((exp['scenario_str'], beta_str, reward_str))
    
    latex = r"""\begin{table*}[!t]
\caption{Performance Summary Across Traffic Scenarios (Last 100 Episodes)}
\label{tab:performance_summary}
\centering
\begin{tabular}{@{}lcc@{}}
\toprule
\textbf{Scenario} & \textbf{QoS Violation (β)} & \textbf{Episode Reward} \\
\midrule
"""
    for scenario, beta, reward in rows:
        latex += f"{scenario} & {beta} & {reward} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table*}
"""
    
    output_path = os.path.join(output_dir, 'summary_table.tex')
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f"  ✓ Saved: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate figures from extracted data')
    parser.add_argument('--data', type=str, default='extracted_data.json', 
                       help='Extracted data JSON file')
    parser.add_argument('--output-dir', type=str, default='./paper_figures', 
                       help='Output directory for figures')
    parser.add_argument('--debug', action='store_true',
                       help='Print debug information about data')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("STEP 2: GENERATING FIGURES AND TABLES")
    print("="*80)
    
    # Load data
    data = load_extracted_data(args.data)
    experiments = data['experiments']
    
    # Debug mode
    if args.debug:
        print("\n" + "="*80)
        print("DEBUG: DATA INSPECTION")
        print("="*80)
        for i, exp in enumerate(experiments, 1):
            print(f"\n[{i}] {exp['run_name']}")
            print(f"  Scenario: {exp['scenario_str']}")
            print(f"  Category: {exp['category']}")
            print(f"  Data keys: {list(exp['data'].keys())}")
            print(f"  Statistics:")
            for stat_key, stat_val in exp['statistics'].items():
                print(f"    {stat_key}: {stat_val}")
        print("\n" + "="*80)
        return
    
    # Categorize
    exps_by_cat = categorize_experiments(experiments)
    
    # Generate all figures
    print("\n" + "="*80)
    print("GENERATING FIGURES")
    print("="*80)
    
    fig1_training_convergence(experiments, args.output_dir)
    fig2_static_homogeneous(exps_by_cat, args.output_dir)
    fig3_heterogeneous(exps_by_cat, args.output_dir)
    fig4_allocation_patterns(experiments, args.output_dir)
    fig5_dynamic_adaptation(exps_by_cat, args.output_dir)
    
    # Generate tables
    print("\n" + "="*80)
    print("GENERATING TABLES")
    print("="*80)
    
    generate_summary_table(experiments, args.output_dir)
    
    print("\n" + "="*80)
    print("✓ ALL FIGURES AND TABLES GENERATED")
    print("="*80)
    print(f"\nOutput directory: {args.output_dir}")
    print(f"\nGenerated files:")
    print(f"  - fig1_training_convergence.{FIGURE_FORMAT}")
    print(f"  - fig2_static_homogeneous.{FIGURE_FORMAT}")
    print(f"  - fig3_heterogeneous.{FIGURE_FORMAT}")
    print(f"  - fig4_allocation_patterns.{FIGURE_FORMAT}")
    print(f"  - fig5_dynamic_adaptation.{FIGURE_FORMAT}")
    print(f"  - summary_table.tex")
    print(f"\nTo modify figures:")
    print(f"  1. Edit settings at top of this script (colors, labels, etc.)")
    print(f"  2. Re-run: python3 step2_generate_figures.py --data {args.data}")
    print(f"  3. Figures regenerate in seconds!")


if __name__ == "__main__":
    main()
