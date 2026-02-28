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
    Figure 1a: Training convergence - Episode Reward (multiple scenarios)
    Figure 1b: Training convergence - Beta (multiple scenarios)
    Figure 1c: Training convergence - Dynamic scenario
    Generate as separate PNG files
    """
    print("\n[Figure 1] Training Convergence...")
    
    # Categorize experiments
    homogeneous_exps = [e for e in experiments if e['category'] == 'static_homogeneous']
    heterogeneous_exps = [e for e in experiments if e['category'] == 'static_heterogeneous']
    dynamic_exps = [e for e in experiments if e['category'] == 'dynamic']
    
    # ========================================================================
    # Figure 1a: Episode Reward - Multiple Static Scenarios
    # ========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define colors for different scenarios
    colors = {
        'extremely_low': '#1f77b4',  # Blue
        'low': '#2ca02c',             # Green
        'medium': '#ff7f0e',          # Orange
        'high': '#d62728',            # Red
        'extremely_high': '#9467bd'   # Purple
    }
    
    scenarios_plotted = []
    
    # Plot homogeneous scenarios
    for exp in homogeneous_exps:
        profile = list(exp['scenario'].values())[0]
        
        if 'episode_reward' in exp['data']:
            steps = np.array(exp['data']['episode_reward']['steps'])
            values = np.array(exp['data']['episode_reward']['values'])
            
            # Moving average
            if len(values) >= 100:
                ma = np.convolve(values, np.ones(100)/100, mode='valid')
                color = colors.get(profile, 'gray')
                label = profile.replace('_', ' ').title()
                
                ax.plot(steps[99:], ma, color=color, linewidth=2, label=label, alpha=0.8)
                scenarios_plotted.append(profile)
    
    # Plot one heterogeneous as example (if available)
    if heterogeneous_exps and 'episode_reward' in heterogeneous_exps[0]['data']:
        exp = heterogeneous_exps[0]
        steps = np.array(exp['data']['episode_reward']['steps'])
        values = np.array(exp['data']['episode_reward']['values'])
        
        if len(values) >= 100:
            ma = np.convolve(values, np.ones(100)/100, mode='valid')
            ax.plot(steps[99:], ma, color='black', linewidth=2, 
                   label=exp['scenario_str'], alpha=0.8, linestyle='--')
    
    ax.set_xlabel(LABEL_EPISODE)
    ax.set_ylabel(LABEL_REWARD)
    ax.set_title('Training Convergence: Episode Reward (Multiple Scenarios)')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'fig1a_reward_multi.{FIGURE_FORMAT}')
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: fig1a_reward_multi.{FIGURE_FORMAT} ({len(scenarios_plotted)} scenarios)")
    
    # ========================================================================
    # Figure 1b: QoS Violation (Beta) - Multiple Static Scenarios
    # ========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scenarios_plotted = []
    
    # Plot homogeneous scenarios
    for exp in homogeneous_exps:
        profile = list(exp['scenario'].values())[0]
        
        if 'episode_beta' in exp['data']:
            steps = np.array(exp['data']['episode_beta']['steps'])
            values = np.array(exp['data']['episode_beta']['values'])
            
            # Moving average
            if len(values) >= 100:
                ma = np.convolve(values, np.ones(100)/100, mode='valid')
                color = colors.get(profile, 'gray')
                label = profile.replace('_', ' ').title()
                
                ax.plot(steps[99:], ma, color=color, linewidth=2, label=label, alpha=0.8)
                scenarios_plotted.append(profile)
    
    # Plot one heterogeneous as example (if available)
    if heterogeneous_exps and 'episode_beta' in heterogeneous_exps[0]['data']:
        exp = heterogeneous_exps[0]
        steps = np.array(exp['data']['episode_beta']['steps'])
        values = np.array(exp['data']['episode_beta']['values'])
        
        if len(values) >= 100:
            ma = np.convolve(values, np.ones(100)/100, mode='valid')
            ax.plot(steps[99:], ma, color='black', linewidth=2, 
                   label=exp['scenario_str'], alpha=0.8, linestyle='--')
    
    ax.set_xlabel(LABEL_EPISODE)
    ax.set_ylabel(LABEL_BETA)
    ax.set_title('Training Convergence: QoS Performance (Multiple Scenarios)')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'fig1b_beta_multi.{FIGURE_FORMAT}')
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: fig1b_beta_multi.{FIGURE_FORMAT} ({len(scenarios_plotted)} scenarios)")
    
    # ========================================================================
    # Figure 1c: Dynamic Scenario Training Convergence (Both Metrics)
    # ========================================================================
    if dynamic_exps:
        exp = dynamic_exps[0]
        
        # Create figure with 2 subplots for dynamic case only
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Top: Reward
        if 'episode_reward' in exp['data']:
            steps = np.array(exp['data']['episode_reward']['steps'])
            values = np.array(exp['data']['episode_reward']['values'])
            
            axes[0].plot(steps, values, alpha=0.3, color=COLOR_REWARD, linewidth=0.5, label='Raw')
            
            if len(values) >= 100:
                ma = np.convolve(values, np.ones(100)/100, mode='valid')
                axes[0].plot(steps[99:], ma, color=COLOR_REWARD, linewidth=2, label='100-ep MA')
            
            axes[0].set_ylabel(LABEL_REWARD)
            axes[0].set_title(f'Training Convergence (Dynamic): Episode Reward')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Bottom: Beta
        if 'episode_beta' in exp['data']:
            steps = np.array(exp['data']['episode_beta']['steps'])
            values = np.array(exp['data']['episode_beta']['values'])
            
            axes[1].plot(steps, values, alpha=0.3, color=COLOR_BETA, linewidth=0.5, label='Raw')
            
            if len(values) >= 100:
                ma = np.convolve(values, np.ones(100)/100, mode='valid')
                axes[1].plot(steps[99:], ma, color=COLOR_BETA, linewidth=2, label='100-ep MA')
            
            axes[1].set_xlabel(LABEL_EPISODE)
            axes[1].set_ylabel(LABEL_BETA)
            axes[1].set_title(f'Training Convergence (Dynamic): QoS Performance')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'fig1c_dynamic_training.{FIGURE_FORMAT}')
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: fig1c_dynamic_training.{FIGURE_FORMAT}")
    else:
        print(f"  ⚠️  No dynamic experiments for fig1c")


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
    # NO THRESHOLD LINE
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
    # NO THRESHOLD LINE
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
    Figure 5a: Dynamic adaptation - Beta over time
    Figure 5b: Dynamic adaptation - Allocations over time
    Figure 5c: Dynamic adaptation - Active traffic profiles over time
    Figure 5d: Dynamic allocation patterns (box plots, like fig4 but for dynamic)
    Generate as separate PNG files
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
    
    # ========================================================================
    # Figure 5a: Beta over time with switches
    # ========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(steps, betas, color=COLOR_BETA, alpha=0.7, linewidth=1)
    ax.set_xlabel(LABEL_DTI)
    ax.set_ylabel(LABEL_BETA)
    ax.set_title('Dynamic Traffic Adaptation: QoS Performance')
    # NO THRESHOLD LINE
    ax.grid(True, alpha=0.3)
    
    # Mark switches every 200 DTIs
    for switch in range(200, int(max(steps)), 200):
        ax.axvline(x=switch, color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'fig5a_dynamic_beta.{FIGURE_FORMAT}')
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: fig5a_dynamic_beta.{FIGURE_FORMAT}")
    
    # ========================================================================
    # Figure 5b: Per-slice allocations over time
    # ========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for key in ['dti_action_slice0', 'dti_action_slice1']:
        if key in data:
            slice_num = key.split('slice')[-1]
            alloc_steps = np.array(data[key]['steps'])
            alloc_values = np.array(data[key]['values'])
            ax.plot(alloc_steps, alloc_values, 
                   label=f'Slice {slice_num}', alpha=0.7, linewidth=1)
    
    ax.set_xlabel(LABEL_DTI)
    ax.set_ylabel(LABEL_RBS)
    ax.set_title('Dynamic Traffic Adaptation: Resource Allocation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Mark switches every 200 DTIs
    for switch in range(200, int(max(steps)), 200):
        ax.axvline(x=switch, color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'fig5b_dynamic_allocation.{FIGURE_FORMAT}')
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: fig5b_dynamic_allocation.{FIGURE_FORMAT}")
    
    # ========================================================================
    # Figure 5c: Active traffic profiles over time
    # ========================================================================
    if 'dti_active_profile_slice0' in data and 'dti_active_profile_slice1' in data:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Profile mapping
        profile_names = {
            0: 'Uniform',
            1: 'Extremely Low',
            2: 'Low',
            3: 'Medium',
            4: 'High',
            5: 'Extremely High',
            6: 'External'
        }
        
        for key in ['dti_active_profile_slice0', 'dti_active_profile_slice1']:
            slice_num = key.split('slice')[-1]
            prof_steps = np.array(data[key]['steps'])
            prof_values = np.array(data[key]['values'])
            ax.plot(prof_steps, prof_values, 
                   label=f'Slice {slice_num}', alpha=0.7, linewidth=2, marker='o', markersize=3)
        
        ax.set_xlabel(LABEL_DTI)
        ax.set_ylabel('Active Traffic Profile')
        ax.set_title('Dynamic Traffic Adaptation: Traffic Profile Changes')
        ax.set_yticks(list(profile_names.keys()))
        ax.set_yticklabels(list(profile_names.values()))
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Mark switches every 200 DTIs
        for switch in range(200, int(max(steps)), 200):
            ax.axvline(x=switch, color='gray', linestyle=':', alpha=0.5)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'fig5c_dynamic_profiles.{FIGURE_FORMAT}')
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: fig5c_dynamic_profiles.{FIGURE_FORMAT}")
    
    # ========================================================================
    # Figure 5d: Allocation patterns (box plots) for dynamic scenario
    # Similar to fig4 but for dynamic experiment
    # ========================================================================
    # Find episode 80 data for dynamic experiment
    allocations = {}
    for key in ['ep80_action_slice0', 'ep80_action_slice1']:
        if key in data:
            slice_num = key.split('slice')[-1]
            allocations[f'Slice {slice_num}'] = data[key]['values']
    
    if allocations:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.boxplot(allocations.values(), labels=allocations.keys(),
                   patch_artist=True,
                   boxprops=dict(facecolor='lightgreen', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
        
        ax.set_ylabel(LABEL_RBS)
        ax.set_title(f'Allocation Distribution - {exp["scenario_str"]} (Episode 80)')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'fig5d_dynamic_allocation_patterns.{FIGURE_FORMAT}')
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: fig5d_dynamic_allocation_patterns.{FIGURE_FORMAT}")


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
    print(f"\n  Training Convergence:")
    print(f"    - fig1a_reward_multi.{FIGURE_FORMAT} (Reward: Multiple Scenarios)")
    print(f"    - fig1b_beta_multi.{FIGURE_FORMAT} (Beta: Multiple Scenarios)")
    print(f"    - fig1c_dynamic_training.{FIGURE_FORMAT} (Dynamic: Both Metrics)")
    print(f"\n  Static Scenarios:")
    print(f"    - fig2_static_homogeneous.{FIGURE_FORMAT} (Homogeneous Performance)")
    print(f"    - fig3_heterogeneous.{FIGURE_FORMAT} (Heterogeneous Performance)")
    print(f"    - fig4_allocation_patterns.{FIGURE_FORMAT} (Static: Allocation Box Plots)")
    print(f"\n  Dynamic Scenarios:")
    print(f"    - fig5a_dynamic_beta.{FIGURE_FORMAT} (QoS Time Series)")
    print(f"    - fig5b_dynamic_allocation.{FIGURE_FORMAT} (Allocation Time Series)")
    print(f"    - fig5c_dynamic_profiles.{FIGURE_FORMAT} (Traffic Profile Changes)")
    print(f"    - fig5d_dynamic_allocation_patterns.{FIGURE_FORMAT} (Allocation Box Plots)")
    print(f"\n  Table:")
    print(f"    - summary_table.tex (LaTeX Table)")
    print(f"\nTotal: 10 figures + 1 table")
    print(f"\nNote: Profile switches occur every 200 DTIs (not 100)")
    print(f"\nTo modify figures:")
    print(f"  1. Edit settings at top of this script (colors, labels, etc.)")
    print(f"  2. Re-run: python3 step2_generate_figures.py --data {args.data}")
    print(f"  3. Figures regenerate in seconds!")


if __name__ == "__main__":
    main()
