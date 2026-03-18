"""
figures_static.py — Figures 1–4: training convergence and static scenarios.

    fig1_training_convergence  — fig1a/b/c/d
    fig2_static_homogeneous    — fig2
    fig3_heterogeneous         — fig3
    fig4_allocation_patterns   — fig4
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from plot_style import (
    FIGURE_DPI, FIGURE_FORMAT,
    COLOR_REWARD, COLOR_BETA, COLOR_HOMOGENEOUS, COLOR_HETEROGENEOUS,
    BASELINE_COLORS, BASELINE_NAMES,
    LABEL_BETA, LABEL_REWARD, LABEL_EPISODE, LABEL_RBS,
    LOAD_ORDER,
)
from data_utils import collect_baseline_vectors


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _plot_grouped_bars(ax, x_pos, profiles, sac_vals, sac_stds,
                       baseline_betas, baseline_stds):
    """Render grouped bars for SAC + baselines (or SAC-only if no baselines)."""
    active_baselines = [
        name for name in BASELINE_NAMES
        if any(v is not None for v in baseline_betas[name])
    ]
    has_baselines = bool(active_baselines)

    if has_baselines:
        n_methods = 1 + len(active_baselines)
        width = 0.8 / n_methods
        offset = -width * (n_methods - 1) / 2

        ax.bar(x_pos + offset, sac_vals, width, yerr=sac_stds, capsize=3,
               label='SAC', color='steelblue', alpha=0.8,
               edgecolor='black', linewidth=1.2)
        offset += width

        for name in active_baselines:
            vals = [v if v is not None else 0 for v in baseline_betas[name]]
            errs = [v if v is not None else 0 for v in baseline_stds[name]]
            ax.bar(x_pos + offset, vals, width, yerr=errs, capsize=3,
                   label=name.capitalize(),
                   color=BASELINE_COLORS.get(name, 'gray'),
                   alpha=0.8, edgecolor='black', linewidth=1.2)
            offset += width
    else:
        color = COLOR_HOMOGENEOUS if len(profiles) > 1 else COLOR_HETEROGENEOUS
        ax.bar(x_pos, sac_vals, yerr=sac_stds, capsize=5,
               color=color, alpha=0.8, edgecolor='black', linewidth=1.2)

    return has_baselines


# ---------------------------------------------------------------------------
# Figure 1: Training Convergence
# ---------------------------------------------------------------------------

def fig1_training_convergence(experiments, output_dir):
    """
    fig1a — Reward bar chart (static homogeneous, last-100 mean).
    fig1b — Beta bar chart  (static homogeneous, last-100 mean).
    fig1c — Dynamic reward time-series (single axis).
    fig1d — Dynamic beta time-series  (single axis).
    """
    print("\n[Figure 1] Training Convergence...")

    homogeneous_exps = [e for e in experiments if e['category'] == 'static_homogeneous']
    dynamic_exps = [e for e in experiments if e['category'] == 'dynamic']

    sorted_exps = sorted(
        homogeneous_exps,
        key=lambda e: LOAD_ORDER.get(list(e['scenario'].values())[0], 999),
    )

    # ---- 1a: Reward bar chart ----
    _bar_chart_metric(
        sorted_exps, metric='reward',
        ylabel=LABEL_REWARD,
        title='Training Performance: Episode Reward (Last 100 Episodes)',
        color=COLOR_REWARD,
        out_path=os.path.join(output_dir, f'fig1a_reward_static.{FIGURE_FORMAT}'),
        label_key='fig1a',
    )

    # ---- 1b: Beta bar chart ----
    _bar_chart_metric(
        sorted_exps, metric='beta',
        ylabel=LABEL_BETA,
        title='Training Performance: QoS Violation (Last 100 Episodes)',
        color=COLOR_BETA,
        out_path=os.path.join(output_dir, f'fig1b_beta_static.{FIGURE_FORMAT}'),
        label_key='fig1b',
    )

    # ---- 1c: Dynamic reward time-series ----
    if dynamic_exps and 'episode_reward' in dynamic_exps[0]['data']:
        _single_timeseries(
            data=dynamic_exps[0]['data']['episode_reward'],
            xlabel=LABEL_EPISODE, ylabel=LABEL_REWARD,
            title='Training Convergence (Dynamic): Episode Reward',
            color=COLOR_REWARD,
            out_path=os.path.join(output_dir, f'fig1c_dynamic_reward.{FIGURE_FORMAT}'),
            label='fig1c',
        )

    # ---- 1d: Dynamic beta time-series ----
    if dynamic_exps and 'episode_beta' in dynamic_exps[0]['data']:
        _single_timeseries(
            data=dynamic_exps[0]['data']['episode_beta'],
            xlabel=LABEL_EPISODE, ylabel=LABEL_BETA,
            title='Training Convergence (Dynamic): QoS Performance',
            color=COLOR_BETA,
            out_path=os.path.join(output_dir, f'fig1d_dynamic_beta.{FIGURE_FORMAT}'),
            label='fig1d',
        )


def _bar_chart_metric(sorted_exps, metric, ylabel, title, color, out_path, label_key):
    profiles, vals, stds = [], [], []
    for exp in sorted_exps:
        stats = exp['statistics'].get(metric, {})
        val = stats.get('last_100_mean', stats.get('mean'))
        std = stats.get('last_100_std', stats.get('std'))
        if val is not None and std is not None:
            profiles.append(list(exp['scenario'].values())[0].replace('_', ' ').title())
            vals.append(val)
            stds.append(std)

    if not profiles:
        print(f"  ⚠️  No data for {label_key}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(profiles))
    ax.bar(x, vals, yerr=stds, capsize=5,
           color=color, alpha=0.8, edgecolor='black', linewidth=1.2)
    ax.set_xlabel('Traffic Profile')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(profiles, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {os.path.basename(out_path)} ({len(profiles)} scenarios)")


def _single_timeseries(data, xlabel, ylabel, title, color, out_path, label):
    steps = np.array(data['steps'])
    values = np.array(data['values'])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, values, alpha=0.6, color=color, linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {os.path.basename(out_path)}")


# ---------------------------------------------------------------------------
# Figure 2: Static Homogeneous
# ---------------------------------------------------------------------------

def fig2_static_homogeneous(exps_by_cat, baseline_data, output_dir):
    """Bar chart of beta across homogeneous traffic profiles (+ baselines)."""
    print("\n[Figure 2] Static Homogeneous Performance...")

    exps = sorted(
        exps_by_cat['static_homogeneous'],
        key=lambda e: LOAD_ORDER.get(list(e['scenario'].values())[0], 999),
    )
    if not exps:
        print("  ⚠️  No homogeneous experiments")
        return

    profiles, sac_betas, sac_stds = [], [], []
    bl_betas = {name: [] for name in BASELINE_NAMES}
    bl_stds = {name: [] for name in BASELINE_NAMES}

    for exp in exps:
        stats = exp['statistics'].get('beta', {})
        val = stats.get('last_100_mean', stats.get('mean'))
        std = stats.get('last_100_std', stats.get('std'))
        if val is None or std is None:
            continue

        profiles.append(list(exp['scenario'].values())[0].replace('_', ' ').title())
        sac_betas.append(val)
        sac_stds.append(std)

        betas, stds = collect_baseline_vectors(exp, baseline_data, BASELINE_NAMES)
        for name in BASELINE_NAMES:
            bl_betas[name].append(betas[name])
            bl_stds[name].append(stds[name])

    if not profiles:
        print("  ⚠️  No valid data")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(profiles))
    has_bl = _plot_grouped_bars(ax, x_pos, profiles, sac_betas, sac_stds,
                                bl_betas, bl_stds)

    ax.set_xlabel('Traffic Profile')
    ax.set_ylabel(LABEL_BETA)
    suffix = ' (SAC vs Baselines)' if has_bl else ''
    ax.set_title(f'Performance Across Static Homogeneous Traffic{suffix}')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(profiles, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    if has_bl:
        ax.legend(loc='best', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fig2_static_homogeneous.{FIGURE_FORMAT}'),
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    tag = 'with baselines' if has_bl else 'SAC only'
    print(f"  ✓ fig2_static_homogeneous.{FIGURE_FORMAT} ({tag})")


# ---------------------------------------------------------------------------
# Figure 3: Heterogeneous
# ---------------------------------------------------------------------------

def fig3_heterogeneous(exps_by_cat, baseline_data, output_dir):
    """Bar chart for heterogeneous scenarios (+ baselines)."""
    print("\n[Figure 3] Heterogeneous Performance...")

    exps = exps_by_cat['static_heterogeneous']
    if not exps:
        print("  ⚠️  No heterogeneous experiments")
        return

    scenarios, sac_betas, sac_stds = [], [], []
    bl_betas = {name: [] for name in BASELINE_NAMES}
    bl_stds = {name: [] for name in BASELINE_NAMES}

    for exp in exps:
        stats = exp['statistics'].get('beta', {})
        val = stats.get('last_100_mean', stats.get('mean'))
        std = stats.get('last_100_std', stats.get('std'))
        if val is None or std is None:
            continue

        scenarios.append(exp['scenario_str'])
        sac_betas.append(val)
        sac_stds.append(std)

        betas, stds = collect_baseline_vectors(exp, baseline_data, BASELINE_NAMES)
        for name in BASELINE_NAMES:
            bl_betas[name].append(betas[name])
            bl_stds[name].append(stds[name])

    if not scenarios:
        print("  ⚠️  No valid data")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(scenarios))
    has_bl = _plot_grouped_bars(ax, x_pos, scenarios, sac_betas, sac_stds,
                                bl_betas, bl_stds)

    ax.set_xlabel('Traffic Scenario')
    ax.set_ylabel(LABEL_BETA)
    suffix = ' (SAC vs Baselines)' if has_bl else ''
    ax.set_title(f'Performance Under Heterogeneous Traffic{suffix}')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    if has_bl:
        ax.legend(loc='best', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fig3_heterogeneous.{FIGURE_FORMAT}'),
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    tag = 'with baselines' if has_bl else 'SAC only'
    print(f"  ✓ fig3_heterogeneous.{FIGURE_FORMAT} ({tag})")


# ---------------------------------------------------------------------------
# Figure 4: Static Allocation Patterns
# ---------------------------------------------------------------------------

_TARGET_SCENARIOS = [
    ('Low-High', ['low', 'high']),
    ('Extremely_Low-Extremely_High', ['extremely', 'low', 'high']),
    ('Medium-High', ['medium', 'high']),
]

_SCENARIO_COLORS = {
    'Low-High':                      ['steelblue',  'lightsteelblue'],
    'Extremely_Low-Extremely_High':  ['coral',      'lightcoral'],
    'Medium-High':                   ['seagreen',   'lightgreen'],
}


def fig4_allocation_patterns(experiments, output_dir):
    """
    Box plots for allocation distribution across multiple heterogeneous scenarios
    (episode 80).  All box plots appear in a single axis.
    """
    print("\n[Figure 4] Allocation Patterns (Multiple Scenarios)...")

    found = {}
    for exp in experiments:
        if exp['category'] != 'static_heterogeneous':
            continue
        norm = exp['scenario_str'].lower().replace('_', '').replace('-', '').replace(' ', '')
        for tname, keywords in _TARGET_SCENARIOS:
            kws = [k.replace('_', '').replace('-', '') for k in keywords]
            if not all(k in norm for k in kws):
                continue
            if tname == 'Extremely_Low-Extremely_High' and norm.count('extremely') < 2:
                continue
            if tname == 'Low-High' and ('extremely' in norm or 'medium' in norm):
                continue
            if tname == 'Medium-High' and 'extremely' in norm:
                continue
            if tname not in found:
                found[tname] = exp
                print(f"  ✓ Matched '{tname}': {exp['scenario_str']}")
            break

    if not found:
        print("  ⚠️  No matching heterogeneous scenarios found")
        for exp in experiments:
            if exp['category'] == 'static_heterogeneous':
                print(f"        - {exp['scenario_str']}")
        return

    all_data, all_labels, all_colors = [], [], []
    for tname, exp in found.items():
        pair_colors = _SCENARIO_COLORS.get(tname, ['gray', 'lightgray'])
        for idx, key in enumerate(['ep80_action_slice0', 'ep80_action_slice1']):
            if key in exp['data']:
                all_data.append(exp['data'][key]['values'])
                all_labels.append(f'{tname}\nSlice {idx}')
                all_colors.append(pair_colors[idx])

    if not all_data:
        print("  ⚠️  No allocation data found")
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    bp = ax.boxplot(all_data, labels=all_labels, patch_artist=True,
                    boxprops=dict(alpha=0.7),
                    medianprops=dict(color='red', linewidth=2),
                    widths=0.6)
    for patch, color in zip(bp['boxes'], all_colors):
        patch.set_facecolor(color)

    ax.set_ylabel(LABEL_RBS, fontsize=12)
    ax.set_xlabel('Scenario and Slice', fontsize=12)
    ax.set_title('Allocation Distribution Across Heterogeneous Scenarios (Episode 80)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fig4_allocation_patterns.{FIGURE_FORMAT}'),
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ fig4_allocation_patterns.{FIGURE_FORMAT} "
          f"({len(found)} scenarios, {len(all_data)} box plots)")
