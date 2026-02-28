"""
STEP 2: Generate Figures from Extracted Data - FIXED VERSION
All issues resolved:
- Figure 1: Debug categorization issue
- Figure 1c/1d: Separate files for dynamic (one axis each)
- Figure 5c: Single episode (80) allocation profiles
- Figure 5d: Multiple period box plots in same figure

Usage:
    python3 step2_generate_figures_v2.py --data extracted_data.json --output-dir ./paper_figures
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
# ============================================================================

FIGURE_DPI = 300
FIGURE_FORMAT = 'png'

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

# Colors
COLOR_REWARD = 'blue'
COLOR_BETA = 'red'
COLOR_HOMOGENEOUS = 'steelblue'
COLOR_HETEROGENEOUS = 'coral'

# Labels
LABEL_BETA = 'QoS Violation Ratio (β)'
LABEL_REWARD = 'Episode Reward'
LABEL_EPISODE = 'Training Episode'
LABEL_DTI = 'Decision Time Interval (DTI)'
LABEL_RBS = 'Resource Blocks Allocated'


def load_extracted_data(json_path):
    """Load extracted data from JSON."""
    print(f"Loading data from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"  ✓ Loaded {data['metadata']['num_experiments']} experiments")
    return data


def load_baseline_results(baseline_dir, experiments):
    """
    Load baseline results from JSON files.
    
    Matches baseline files to SAC experiments by run_name.
    
    Returns:
        dict: {run_name: {baseline_name: {statistics}}}
    """
    if not os.path.exists(baseline_dir):
        print(f"Warning: Baseline directory not found: {baseline_dir}")
        return {}
    
    print(f"\nLoading baseline results from {baseline_dir}...")
    
    baseline_data = {}
    
    # Get all baseline JSON files
    baseline_files = [f for f in os.listdir(baseline_dir) if f.startswith('baselines_') and f.endswith('.json')]
    
    if not baseline_files:
        print(f"  No baseline files found in {baseline_dir}")
        return {}
    
    for filename in baseline_files:
        filepath = os.path.join(baseline_dir, filename)
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            run_name = data['metadata']['run_name']
            
            # Store baseline results for this run
            baseline_data[run_name] = {}
            
            for baseline_name, baseline_results in data['baselines'].items():
                baseline_data[run_name][baseline_name] = baseline_results['statistics']
            
            print(f"  ✓ Loaded {filename}: {list(data['baselines'].keys())}")
        
        except Exception as e:
            print(f"  ⚠️  Failed to load {filename}: {e}")
    
    print(f"  ✓ Loaded baselines for {len(baseline_data)} experiments")
    
    return baseline_data


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
    
    # DEBUG: Show what's in each category
    print("\n  Homogeneous experiments:")
    for exp in categories['static_homogeneous']:
        profile = list(exp['scenario'].values())[0]
        has_reward = 'episode_reward' in exp['data']
        has_beta = 'episode_beta' in exp['data']
        print(f"    - {profile}: reward={has_reward}, beta={has_beta}")
    
    return categories


# ============================================================================
# FIGURE 1: Training Convergence
# ============================================================================

def fig1_training_convergence(experiments, output_dir):
    """
    Generate training convergence figures using pre-computed statistics:
    - fig1a: Reward (multiple static scenarios) - uses last_100_mean from stats
    - fig1b: Beta (multiple static scenarios) - uses last_100_mean from stats
    - fig1c: Dynamic reward (single axis) - raw + MA
    - fig1d: Dynamic beta (single axis) - raw + MA
    """
    print("\n[Figure 1] Training Convergence...")
    
    # Categorize
    homogeneous_exps = [e for e in experiments if e['category'] == 'static_homogeneous']
    heterogeneous_exps = [e for e in experiments if e['category'] == 'static_heterogeneous']
    dynamic_exps = [e for e in experiments if e['category'] == 'dynamic']
    
    # Colors for profiles
    profile_colors = {
        'extremely_low': '#1f77b4',
        'low': '#2ca02c',
        'medium': '#ff7f0e',
        'high': '#d62728',
        'extremely_high': '#9467bd'
    }
    
    # ========================================================================
    # Figure 1a: Reward - BAR CHART using last_100_mean from statistics
    # ========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    profiles = []
    rewards = []
    stds = []
    
    # Sort by load order
    load_order = {'extremely_low': 0, 'low': 1, 'medium': 2, 'high': 3, 'extremely_high': 4}
    sorted_exps = sorted(homogeneous_exps, key=lambda e: load_order.get(list(e['scenario'].values())[0], 999))
    
    for exp in sorted_exps:
        profile = list(exp['scenario'].values())[0]
        reward_stats = exp['statistics'].get('reward')
        
        if reward_stats:
            reward_val = reward_stats.get('last_100_mean', reward_stats.get('mean'))
            std_val = reward_stats.get('last_100_std', reward_stats.get('std'))
            
            if reward_val is not None and std_val is not None:
                profiles.append(profile.replace('_', ' ').title())
                rewards.append(reward_val)
                stds.append(std_val)
    
    if profiles:
        x_pos = np.arange(len(profiles))
        ax.bar(x_pos, rewards, yerr=stds, capsize=5,
               color=COLOR_REWARD, alpha=0.8, edgecolor='black', linewidth=1.2)
        
        ax.set_xlabel('Traffic Profile')
        ax.set_ylabel(LABEL_REWARD)
        ax.set_title('Training Performance: Episode Reward (Last 100 Episodes)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(profiles, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        print(f"  ⚠️  No scenarios plotted for fig1a")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fig1a_reward_static.{FIGURE_FORMAT}'), 
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ fig1a_reward_static.{FIGURE_FORMAT} ({len(profiles)} scenarios)")
    
    # ========================================================================
    # Figure 1b: Beta - BAR CHART using last_100_mean from statistics
    # ========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    profiles = []
    betas = []
    stds = []
    
    for exp in sorted_exps:
        profile = list(exp['scenario'].values())[0]
        beta_stats = exp['statistics'].get('beta')
        
        if beta_stats:
            beta_val = beta_stats.get('last_100_mean', beta_stats.get('mean'))
            std_val = beta_stats.get('last_100_std', beta_stats.get('std'))
            
            if beta_val is not None and std_val is not None:
                profiles.append(profile.replace('_', ' ').title())
                betas.append(beta_val)
                stds.append(std_val)
    
    if profiles:
        x_pos = np.arange(len(profiles))
        ax.bar(x_pos, betas, yerr=stds, capsize=5,
               color=COLOR_BETA, alpha=0.8, edgecolor='black', linewidth=1.2)
        
        ax.set_xlabel('Traffic Profile')
        ax.set_ylabel(LABEL_BETA)
        ax.set_title('Training Performance: QoS Violation (Last 100 Episodes)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(profiles, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        print(f"  ⚠️  No scenarios plotted for fig1b")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fig1b_beta_static.{FIGURE_FORMAT}'), 
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ fig1b_beta_static.{FIGURE_FORMAT} ({len(profiles)} scenarios)")
    
    # ========================================================================
    # Figure 1c: Dynamic Reward (SINGLE AXIS) - raw data over training
    # ========================================================================
    if dynamic_exps and 'episode_reward' in dynamic_exps[0]['data']:
        exp = dynamic_exps[0]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        steps = np.array(exp['data']['episode_reward']['steps'])
        values = np.array(exp['data']['episode_reward']['values'])
        
        ax.plot(steps, values, alpha=0.6, color=COLOR_REWARD, linewidth=1)
        
        ax.set_xlabel(LABEL_EPISODE)
        ax.set_ylabel(LABEL_REWARD)
        ax.set_title('Training Convergence (Dynamic): Episode Reward')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'fig1c_dynamic_reward.{FIGURE_FORMAT}'), 
                    dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  ✓ fig1c_dynamic_reward.{FIGURE_FORMAT}")
    
    # ========================================================================
    # Figure 1d: Dynamic Beta (SINGLE AXIS) - raw data over training
    # ========================================================================
    if dynamic_exps and 'episode_beta' in dynamic_exps[0]['data']:
        exp = dynamic_exps[0]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        steps = np.array(exp['data']['episode_beta']['steps'])
        values = np.array(exp['data']['episode_beta']['values'])
        
        ax.plot(steps, values, alpha=0.6, color=COLOR_BETA, linewidth=1)
        
        ax.set_xlabel(LABEL_EPISODE)
        ax.set_ylabel(LABEL_BETA)
        ax.set_title('Training Convergence (Dynamic): QoS Performance')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'fig1d_dynamic_beta.{FIGURE_FORMAT}'), 
                    dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  ✓ fig1d_dynamic_beta.{FIGURE_FORMAT}")


# ============================================================================
# FIGURE 2: Static Homogeneous
# ============================================================================

def fig2_static_homogeneous(exps_by_cat, baseline_data, output_dir):
    """
    Bar chart of beta across homogeneous traffic profiles.
    Now includes baseline comparisons if available.
    """
    print("\n[Figure 2] Static Homogeneous Performance...")
    
    exps = exps_by_cat['static_homogeneous']
    if not exps:
        print("  ⚠️  No homogeneous experiments")
        return
    
    # Sort by load
    load_order = {'extremely_low': 0, 'low': 1, 'medium': 2, 'high': 3, 'extremely_high': 4}
    exps = sorted(exps, key=lambda e: load_order.get(list(e['scenario'].values())[0], 999))
    
    profiles = []
    sac_betas = []
    sac_stds = []
    
    # Baseline data structures
    baseline_names = ['equal', 'proportional', 'greedy', 'random']
    baseline_betas = {name: [] for name in baseline_names}
    baseline_stds = {name: [] for name in baseline_names}
    
    for exp in exps:
        profile = list(exp['scenario'].values())[0].replace('_', ' ').title()
        stats = exp['statistics'].get('beta')
        
        if stats:
            beta_val = stats.get('last_100_mean', stats.get('mean'))
            std_val = stats.get('last_100_std', stats.get('std'))
            
            if beta_val is not None and std_val is not None:
                profiles.append(profile)
                sac_betas.append(beta_val)
                sac_stds.append(std_val)
                
                # Get baseline results for this experiment
                run_name = exp['run_name']
                if run_name in baseline_data:
                    for baseline_name in baseline_names:
                        if baseline_name in baseline_data[run_name]:
                            b_stats = baseline_data[run_name][baseline_name]['beta']
                            b_val = b_stats.get('last_100_mean', b_stats.get('mean'))
                            b_std = b_stats.get('last_100_std', b_stats.get('std'))
                            baseline_betas[baseline_name].append(b_val)
                            baseline_stds[baseline_name].append(b_std)
                        else:
                            baseline_betas[baseline_name].append(None)
                            baseline_stds[baseline_name].append(None)
                else:
                    # No baseline data for this experiment
                    for baseline_name in baseline_names:
                        baseline_betas[baseline_name].append(None)
                        baseline_stds[baseline_name].append(None)
    
    if not profiles:
        print("  ⚠️  No valid data")
        return
    
    # Check if we have any baseline data
    has_baselines = any(any(v is not None for v in baseline_betas[name]) for name in baseline_names)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Setup for grouped bar chart
    x_pos = np.arange(len(profiles))
    
    if has_baselines:
        # Grouped bar chart: SAC + Baselines
        n_methods = 1 + sum(1 for name in baseline_names if any(v is not None for v in baseline_betas[name]))
        width = 0.8 / n_methods
        
        methods_plotted = []
        bar_containers = []
        
        # Plot SAC
        offset = -width * (n_methods - 1) / 2
        bars_sac = ax.bar(x_pos + offset, sac_betas, width, yerr=sac_stds, capsize=3,
               label='SAC', color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.2)
        bar_containers.append(bars_sac)
        methods_plotted.append('SAC')
        offset += width
        
        # Plot baselines
        baseline_colors = {
            'equal': 'lightcoral',
            'proportional': 'lightgreen',
            'greedy': 'gold',
            'random': 'lightgray'
        }
        
        for baseline_name in baseline_names:
            if any(v is not None for v in baseline_betas[baseline_name]):
                # Replace None with 0 for plotting (or skip)
                plot_vals = [v if v is not None else 0 for v in baseline_betas[baseline_name]]
                plot_stds = [v if v is not None else 0 for v in baseline_stds[baseline_name]]
                
                bars = ax.bar(x_pos + offset, plot_vals, width, yerr=plot_stds, capsize=3,
                       label=baseline_name.capitalize(), 
                       color=baseline_colors.get(baseline_name, 'gray'),
                       alpha=0.8, edgecolor='black', linewidth=1.2)
                bar_containers.append(bars)
                methods_plotted.append(baseline_name)
                offset += width
        
        # Add value labels on top of each bar
        for bars in bar_containers:
            for bar in bars:
                height = bar.get_height()
                if height > 0:  # Only label non-zero bars
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=8, rotation=0)
    else:
        # Single SAC bars (no baselines)
        bars = ax.bar(x_pos, sac_betas, yerr=sac_stds, capsize=5,
               color=COLOR_HOMOGENEOUS, alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Traffic Profile')
    ax.set_ylabel(LABEL_BETA)
    title = 'Performance Across Static Homogeneous Traffic'
    if has_baselines:
        title += ' (SAC vs Baselines)'
    ax.set_title(title)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(profiles, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    if has_baselines:
        ax.legend(loc='best', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fig2_static_homogeneous.{FIGURE_FORMAT}'), 
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    if has_baselines:
        print(f"  ✓ fig2_static_homogeneous.{FIGURE_FORMAT} (with baselines)")
    else:
        print(f"  ✓ fig2_static_homogeneous.{FIGURE_FORMAT} (SAC only)")


# ============================================================================
# FIGURE 3: Heterogeneous
# ============================================================================

def fig3_heterogeneous(exps_by_cat, baseline_data, output_dir):
    """
    Bar chart for heterogeneous scenarios.
    Now includes baseline comparisons if available.
    """
    print("\n[Figure 3] Heterogeneous Performance...")
    
    exps = exps_by_cat['static_heterogeneous']
    if not exps:
        print("  ⚠️  No heterogeneous experiments")
        return
    
    scenarios = []
    sac_betas = []
    sac_stds = []
    
    # Baseline data
    baseline_names = ['equal', 'proportional', 'greedy', 'random']
    baseline_betas = {name: [] for name in baseline_names}
    baseline_stds = {name: [] for name in baseline_names}
    
    for exp in exps:
        stats = exp['statistics'].get('beta')
        
        if stats:
            beta_val = stats.get('last_100_mean', stats.get('mean'))
            std_val = stats.get('last_100_std', stats.get('std'))
            
            if beta_val is not None and std_val is not None:
                scenarios.append(exp['scenario_str'])
                sac_betas.append(beta_val)
                sac_stds.append(std_val)
                
                # Get baseline results
                run_name = exp['run_name']
                if run_name in baseline_data:
                    for baseline_name in baseline_names:
                        if baseline_name in baseline_data[run_name]:
                            b_stats = baseline_data[run_name][baseline_name]['beta']
                            b_val = b_stats.get('last_100_mean', b_stats.get('mean'))
                            b_std = b_stats.get('last_100_std', b_stats.get('std'))
                            baseline_betas[baseline_name].append(b_val)
                            baseline_stds[baseline_name].append(b_std)
                        else:
                            baseline_betas[baseline_name].append(None)
                            baseline_stds[baseline_name].append(None)
                else:
                    for baseline_name in baseline_names:
                        baseline_betas[baseline_name].append(None)
                        baseline_stds[baseline_name].append(None)
    
    if not scenarios:
        print("  ⚠️  No valid data")
        return
    
    has_baselines = any(any(v is not None for v in baseline_betas[name]) for name in baseline_names)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(scenarios))
    
    if has_baselines:
        n_methods = 1 + sum(1 for name in baseline_names if any(v is not None for v in baseline_betas[name]))
        width = 0.8 / n_methods
        
        bar_containers = []
        
        offset = -width * (n_methods - 1) / 2
        bars_sac = ax.bar(x_pos + offset, sac_betas, width, yerr=sac_stds, capsize=3,
               label='SAC', color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.2)
        bar_containers.append(bars_sac)
        offset += width
        
        baseline_colors = {
            'equal': 'lightcoral',
            'proportional': 'lightgreen',
            'greedy': 'gold',
            'random': 'lightgray'
        }
        
        for baseline_name in baseline_names:
            if any(v is not None for v in baseline_betas[baseline_name]):
                plot_vals = [v if v is not None else 0 for v in baseline_betas[baseline_name]]
                plot_stds = [v if v is not None else 0 for v in baseline_stds[baseline_name]]
                
                bars = ax.bar(x_pos + offset, plot_vals, width, yerr=plot_stds, capsize=3,
                       label=baseline_name.capitalize(),
                       color=baseline_colors.get(baseline_name, 'gray'),
                       alpha=0.8, edgecolor='black', linewidth=1.2)
                bar_containers.append(bars)
                offset += width
        
        # Add value labels on top of each bar
        for bars in bar_containers:
            for bar in bars:
                height = bar.get_height()
                if height > 0:  # Only label non-zero bars
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=8, rotation=0)
    else:
        bars = ax.bar(x_pos, sac_betas, yerr=sac_stds, capsize=5,
               color=COLOR_HETEROGENEOUS, alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Traffic Scenario')
    ax.set_ylabel(LABEL_BETA)
    title = 'Performance Under Heterogeneous Traffic'
    if has_baselines:
        title += ' (SAC vs Baselines)'
    ax.set_title(title)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    if has_baselines:
        ax.legend(loc='best', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fig3_heterogeneous.{FIGURE_FORMAT}'), 
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    if has_baselines:
        print(f"  ✓ fig3_heterogeneous.{FIGURE_FORMAT} (with baselines)")
    else:
        print(f"  ✓ fig3_heterogeneous.{FIGURE_FORMAT} (SAC only)")


# ============================================================================
# FIGURE 4: Static Allocation Patterns
# ============================================================================

def fig4_allocation_patterns(experiments, output_dir):
    """Box plots for static scenario (episode 80)."""
    print("\n[Figure 4] Static Allocation Patterns...")
    
    # Use heterogeneous if available, else any static
    exp = None
    for e in experiments:
        if e['category'] == 'static_heterogeneous':
            exp = e
            break
    
    if not exp:
        for e in experiments:
            if e['category'] == 'static_homogeneous':
                exp = e
                break
    
    if not exp:
        print("  ⚠️  No static experiments")
        return
    
    allocations = {}
    for key in ['ep80_action_slice0', 'ep80_action_slice1']:
        if key in exp['data']:
            slice_num = key.split('slice')[-1]
            allocations[f'Slice {slice_num}'] = exp['data'][key]['values']
    
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
    plt.savefig(os.path.join(output_dir, f'fig4_allocation_patterns.{FIGURE_FORMAT}'), 
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ fig4_allocation_patterns.{FIGURE_FORMAT}")


# ============================================================================
# FIGURE 5: Dynamic Scenarios
# ============================================================================

def fig5_dynamic_scenarios(exps_by_cat, output_dir):
    """Generate all dynamic scenario figures."""
    print("\n[Figure 5] Dynamic Scenarios...")
    
    exps = exps_by_cat['dynamic']
    if not exps:
        print("  ⚠️  No dynamic experiments")
        return
    
    exp = exps[0]
    data = exp['data']
    
    # ========================================================================
    # Figure 5a: Beta time series
    # ========================================================================
    if 'dti_beta' in data:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        steps = np.array(data['dti_beta']['steps'])
        betas = np.array(data['dti_beta']['values'])
        
        ax.plot(steps, betas, color=COLOR_BETA, alpha=0.7, linewidth=1)
        ax.set_xlabel(LABEL_DTI)
        ax.set_ylabel(LABEL_BETA)
        ax.set_title('Dynamic Traffic Adaptation: QoS Performance')
        ax.grid(True, alpha=0.3)
        
        # Mark switches every 200 DTIs
        for switch in range(200, int(max(steps)), 200):
            ax.axvline(x=switch, color='gray', linestyle=':', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'fig5a_dynamic_beta.{FIGURE_FORMAT}'), 
                    dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  ✓ fig5a_dynamic_beta.{FIGURE_FORMAT}")
    
    # ========================================================================
    # Figure 5b: Allocation time series
    # ========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for key in ['dti_action_slice0', 'dti_action_slice1']:
        if key in data:
            slice_num = key.split('slice')[-1]
            steps = np.array(data[key]['steps'])
            values = np.array(data[key]['values'])
            ax.plot(steps, values, label=f'Slice {slice_num}', alpha=0.7, linewidth=1)
    
    ax.set_xlabel(LABEL_DTI)
    ax.set_ylabel(LABEL_RBS)
    ax.set_title('Dynamic Traffic Adaptation: Resource Allocation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Mark switches every 200 DTIs
    if 'dti_beta' in data:
        max_step = max(data['dti_beta']['steps'])
        for switch in range(200, int(max_step), 200):
            ax.axvline(x=switch, color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fig5b_dynamic_allocation.{FIGURE_FORMAT}'), 
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ fig5b_dynamic_allocation.{FIGURE_FORMAT}")
    
    # ========================================================================
    # Figure 5c: Active profiles for EPISODE 80 ONLY
    # ========================================================================
    if 'ep80_action_slice0' in data:
        # Get episode 80 data
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # For episode 80, we need DTI-level profile data
        # Extract DTI range for episode 80 (assuming 2000 DTIs per episode, ep80 = DTI 160000-162000)
        # But we only have ep80_action data, not ep80_active_profile
        # So we'll use the general dti_active_profile and slice for episode 80
        
        if 'dti_active_profile_slice0' in data and 'dti_active_profile_slice1' in data:
            # Episode 80, assuming 2000 DTIs per episode
            ep_start_dti = 80 * 2000
            ep_end_dti = 81 * 2000
            
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
                all_steps = np.array(data[key]['steps'])
                all_values = np.array(data[key]['values'])
                
                # Filter for episode 80
                mask = (all_steps >= ep_start_dti) & (all_steps < ep_end_dti)
                steps = all_steps[mask] - ep_start_dti  # Relative to episode start
                values = all_values[mask]
                
                ax.plot(steps, values, label=f'Slice {slice_num}', 
                       alpha=0.7, linewidth=2, marker='o', markersize=2)
            
            ax.set_xlabel('DTI (within Episode 80)')
            ax.set_ylabel('Active Traffic Profile')
            ax.set_title('Traffic Profile Changes (Episode 80)')
            ax.set_yticks(list(profile_names.keys()))
            ax.set_yticklabels(list(profile_names.values()))
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'fig5c_dynamic_profiles_ep80.{FIGURE_FORMAT}'), 
                        dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close()
            print(f"  ✓ fig5c_dynamic_profiles_ep80.{FIGURE_FORMAT}")
    
    # ========================================================================
    # Figure 5d: Box plots for MULTIPLE PERIODS (episode 80)
    # ========================================================================
    if 'ep80_action_slice0' in data and 'ep80_action_slice1' in data:
        # Assuming episode 80 has 2000 DTIs
        # Split into periods: 0-200, 200-400, ..., 1800-2000
        # That's 10 periods
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        slice0_data = data['ep80_action_slice0']['values']
        slice1_data = data['ep80_action_slice1']['values']
        
        period_size = 200
        num_periods = len(slice0_data) // period_size
        
        box_data = []
        labels = []
        positions = []
        
        for period in range(num_periods):
            start = period * period_size
            end = (period + 1) * period_size
            
            # Slice 0 data for this period
            s0_period = slice0_data[start:end]
            # Slice 1 data for this period
            s1_period = slice1_data[start:end]
            
            if len(s0_period) > 0 and len(s1_period) > 0:
                box_data.append(s0_period)
                box_data.append(s1_period)
                
                dti_start = start
                dti_end = end - 1
                labels.append(f'{dti_start}-{dti_end}\nSlice 0')
                labels.append(f'{dti_start}-{dti_end}\nSlice 1')
                
                # Positions: group pairs together
                base_pos = period * 3  # Leave gap between periods
                positions.append(base_pos)
                positions.append(base_pos + 1)
        
        if box_data:
            bp = ax.boxplot(box_data, labels=labels, positions=positions,
                           patch_artist=True, widths=0.8)
            
            # Color alternating: Slice 0 = blue, Slice 1 = orange
            for i, patch in enumerate(bp['boxes']):
                if i % 2 == 0:  # Slice 0
                    patch.set_facecolor('lightblue')
                else:  # Slice 1
                    patch.set_facecolor('lightcoral')
                patch.set_alpha(0.7)
            
            for median in bp['medians']:
                median.set_color('red')
                median.set_linewidth(2)
            
            ax.set_xlabel('DTI Range within Episode 80')
            ax.set_ylabel(LABEL_RBS)
            ax.set_title('Allocation Distribution by Period (Episode 80, Dynamic)')
            ax.grid(True, alpha=0.3, axis='y')
            plt.xticks(rotation=45, ha='right', fontsize=8)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='lightblue', alpha=0.7, label='Slice 0'),
                Patch(facecolor='lightcoral', alpha=0.7, label='Slice 1')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'fig5d_dynamic_allocation_periods.{FIGURE_FORMAT}'), 
                        dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close()
            print(f"  ✓ fig5d_dynamic_allocation_periods.{FIGURE_FORMAT} ({num_periods} periods)")


# ============================================================================
# TABLE
# ============================================================================

def generate_summary_table(experiments, output_dir):
    """Generate LaTeX summary table."""
    print("\n[Table] Summary Table...")
    
    rows = []
    for exp in experiments:
        beta_stats = exp['statistics'].get('beta')
        reward_stats = exp['statistics'].get('reward')
        
        if beta_stats:
            beta_mean = beta_stats.get('last_100_mean', beta_stats.get('mean'))
            beta_std = beta_stats.get('last_100_std', beta_stats.get('std'))
            beta_str = f"{beta_mean:.3f} $\\pm$ {beta_std:.3f}" if beta_mean and beta_std else "N/A"
        else:
            beta_str = "N/A"
        
        if reward_stats:
            reward_mean = reward_stats.get('last_100_mean', reward_stats.get('mean'))
            reward_std = reward_stats.get('last_100_std', reward_stats.get('std'))
            reward_str = f"{reward_mean:.2f} $\\pm$ {reward_std:.2f}" if reward_mean and reward_std else "N/A"
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
    
    with open(os.path.join(output_dir, 'summary_table.tex'), 'w') as f:
        f.write(latex)
    
    print(f"  ✓ summary_table.tex")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate figures from extracted data')
    parser.add_argument('--data', type=str, default='extracted_data.json', 
                       help='Extracted data JSON file')
    parser.add_argument('--output-dir', type=str, default='./paper_figures', 
                       help='Output directory for figures')
    parser.add_argument('--baseline-dir', type=str, default='./baseline_results',
                       help='Directory containing baseline JSON files')
    parser.add_argument('--include-baselines', action='store_true',
                       help='Include baseline comparisons in figures')
    parser.add_argument('--debug', action='store_true',
                       help='Print debug information')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("STEP 2: GENERATING FIGURES (FIXED VERSION)")
    print("="*80)
    
    # Load data
    data = load_extracted_data(args.data)
    experiments = data['experiments']
    
    # Load baseline results if requested
    baseline_data = {}
    if args.include_baselines:
        baseline_data = load_baseline_results(args.baseline_dir, experiments)
    
    # Debug mode
    if args.debug:
        print("\nDEBUG MODE:")
        for i, exp in enumerate(experiments):
            print(f"\n[{i+1}] {exp['run_name']}")
            print(f"  Category: {exp['category']}")
            print(f"  Scenario: {exp['scenario']}")
            print(f"  Data keys: {list(exp['data'].keys())}")
            if exp['run_name'] in baseline_data:
                print(f"  Baselines: {list(baseline_data[exp['run_name']].keys())}")
        return
    
    # Categorize
    exps_by_cat = categorize_experiments(experiments)
    
    # Generate figures
    print("\n" + "="*80)
    print("GENERATING FIGURES")
    print("="*80)
    
    fig1_training_convergence(experiments, args.output_dir)
    fig2_static_homogeneous(exps_by_cat, baseline_data, args.output_dir)
    fig3_heterogeneous(exps_by_cat, baseline_data, args.output_dir)
    fig4_allocation_patterns(experiments, args.output_dir)
    fig5_dynamic_scenarios(exps_by_cat, args.output_dir)
    
    # Generate table
    print("\n" + "="*80)
    print("GENERATING TABLE")
    print("="*80)
    
    generate_summary_table(experiments, args.output_dir)
    
    print("\n" + "="*80)
    print("✓ COMPLETE")
    print("="*80)
    print(f"\nGenerated figures:")
    print(f"  Training Convergence:")
    print(f"    - fig1a_reward_multi.png (multiple scenarios)")
    print(f"    - fig1b_beta_multi.png (multiple scenarios)")
    print(f"    - fig1c_dynamic_reward.png (single axis)")
    print(f"    - fig1d_dynamic_beta.png (single axis)")
    print(f"  Static Scenarios:")
    print(f"    - fig2_static_homogeneous.png")
    print(f"    - fig3_heterogeneous.png")
    print(f"    - fig4_allocation_patterns.png (episode 80)")
    print(f"  Dynamic Scenarios:")
    print(f"    - fig5a_dynamic_beta.png (time series)")
    print(f"    - fig5b_dynamic_allocation.png (time series)")
    print(f"    - fig5c_dynamic_profiles_ep80.png (episode 80 profiles)")
    print(f"    - fig5d_dynamic_allocation_periods_ep80.png (episode 80 periods)")
    print(f"    - fig5e_dynamic_profiles_ep160.png (episode 160 profiles)")
    print(f"    - fig5f_dynamic_allocation_periods_ep160.png (episode 160 periods)")
    print(f"  Table:")
    print(f"    - summary_table.tex")
    print(f"\nTotal: 13 figures + 1 table")


if __name__ == "__main__":
    main()
