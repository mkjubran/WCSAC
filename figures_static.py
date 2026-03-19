"""
figures_static.py — Figures 1–4: training convergence and static scenarios.

    fig1_training_convergence  — fig1a/b/c/d
    fig2_static_homogeneous    — fig2
    fig3_heterogeneous         — fig3
    fig4_allocation_patterns   — fig4

Figures that previously filtered to 'global' reward only now produce one
output file per reward formulation present in the data (e.g. fig2_..._global,
fig2_..._weighted).  Ablation figures that compare formulations are unchanged.
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
    abbrev_profile,
    abbrev_scenario_str,
)
from data_utils import collect_baseline_vectors


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _reward_formulations(experiments, category=None):
    """
    Return a sorted list of distinct reward_formulation values present in
    the experiments (optionally filtered to a specific category).
    Defaults to ['global'] if the field is absent (old JSON).
    """
    filtered = experiments if category is None else [
        e for e in experiments if e['category'] == category
    ]
    found = sorted({e.get('reward_formulation', 'global') for e in filtered})
    return found if found else ['global']


def _rf_suffix(rf):
    """File-name suffix for a reward formulation, e.g. '_global'."""
    return f'_{rf}'


def _plot_grouped_bars(ax, x_pos, labels, sac_vals, sac_stds,
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
        ax.bar(x_pos, sac_vals, yerr=sac_stds, capsize=5,
               color=COLOR_HOMOGENEOUS, alpha=0.8, edgecolor='black', linewidth=1.2)

    return has_baselines


def _bar_chart_metric(sorted_exps, metric, ylabel, title, color, out_path, label_key):
    profiles, vals, stds = [], [], []
    for exp in sorted_exps:
        stats = exp['statistics'].get(metric, {})
        val = stats.get('last_100_mean', stats.get('mean'))
        std = stats.get('last_100_std', stats.get('std'))
        if val is not None and std is not None:
            profiles.append(abbrev_profile(list(exp['scenario'].values())[0]))
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
    ax.set_xticklabels(profiles)
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
# Figure 1: Training Convergence
# ---------------------------------------------------------------------------

def fig1_training_convergence(experiments, output_dir):
    """
    One set of fig1a–1d per reward formulation present in the data.
    Files are named  fig1a_reward_static_global.png,
                     fig1a_reward_static_weighted.png, etc.
    """
    print("\n[Figure 1] Training Convergence...")

    rf_list = _reward_formulations(experiments)

    for rf in rf_list:
        sfx = _rf_suffix(rf)

        homogeneous_exps = sorted(
            [e for e in experiments
             if e['category'] == 'static_homogeneous'
             and e.get('reward_formulation', 'global') == rf],
            key=lambda e: LOAD_ORDER.get(list(e['scenario'].values())[0], 999),
        )
        dynamic_exps = [
            e for e in experiments
            if e['category'] == 'dynamic'
            and e.get('reward_formulation', 'global') == rf
        ]

        rf_title = rf.capitalize()

        # 1a: Reward bar chart
        _bar_chart_metric(
            homogeneous_exps, metric='reward',
            ylabel=LABEL_REWARD,
            title=f'Training Performance: Episode Reward — {rf_title} Reward',
            color=COLOR_REWARD,
            out_path=os.path.join(output_dir,
                                  f'fig1a_reward_static{sfx}.{FIGURE_FORMAT}'),
            label_key=f'fig1a{sfx}',
        )

        # 1b: Beta bar chart
        _bar_chart_metric(
            homogeneous_exps, metric='beta',
            ylabel=LABEL_BETA,
            title=f'Training Performance: QoS Violation — {rf_title} Reward',
            color=COLOR_BETA,
            out_path=os.path.join(output_dir,
                                  f'fig1b_beta_static{sfx}.{FIGURE_FORMAT}'),
            label_key=f'fig1b{sfx}',
        )

        # 1c/1d: Dynamic time-series — first dynamic exp for this formulation
        if not dynamic_exps:
            print(f"  ℹ️  No {rf} dynamic experiments for fig1c/1d")
            continue

        dyn_exp  = dynamic_exps[0]
        pool     = dyn_exp.get('dynamic_profile_set', [])
        pool_str = f" [{abbrev_profile(pool)}]" if pool else ''

        if 'episode_reward' in dyn_exp['data']:
            _single_timeseries(
                data=dyn_exp['data']['episode_reward'],
                xlabel=LABEL_EPISODE, ylabel=LABEL_REWARD,
                title=f'Training Convergence (Dynamic{pool_str}): '
                      f'Episode Reward — {rf_title} Reward',
                color=COLOR_REWARD,
                out_path=os.path.join(output_dir,
                                      f'fig1c_dynamic_reward{sfx}.{FIGURE_FORMAT}'),
                label=f'fig1c{sfx}',
            )

        if 'episode_beta' in dyn_exp['data']:
            _single_timeseries(
                data=dyn_exp['data']['episode_beta'],
                xlabel=LABEL_EPISODE, ylabel=LABEL_BETA,
                title=f'Training Convergence (Dynamic{pool_str}): '
                      f'QoS Performance — {rf_title} Reward',
                color=COLOR_BETA,
                out_path=os.path.join(output_dir,
                                      f'fig1d_dynamic_beta{sfx}.{FIGURE_FORMAT}'),
                label=f'fig1d{sfx}',
            )


# ---------------------------------------------------------------------------
# Figure 2: Static Homogeneous
# ---------------------------------------------------------------------------

def fig2_static_homogeneous(exps_by_cat, baseline_data, output_dir):
    """
    One bar chart per reward formulation present in the data.
    Files: fig2_static_homogeneous_global.png, fig2_static_homogeneous_weighted.png
    """
    print("\n[Figure 2] Static Homogeneous Performance...")

    rf_list = _reward_formulations(
        exps_by_cat['static_homogeneous'])

    for rf in rf_list:
        sfx = _rf_suffix(rf)
        rf_title = rf.capitalize()

        exps = sorted(
            [e for e in exps_by_cat['static_homogeneous']
             if e.get('reward_formulation', 'global') == rf],
            key=lambda e: LOAD_ORDER.get(list(e['scenario'].values())[0], 999),
        )
        if not exps:
            continue

        profiles, sac_betas, sac_stds = [], [], []
        bl_betas = {name: [] for name in BASELINE_NAMES}
        bl_stds  = {name: [] for name in BASELINE_NAMES}

        for exp in exps:
            stats = exp['statistics'].get('beta', {})
            val = stats.get('last_100_mean', stats.get('mean'))
            std = stats.get('last_100_std', stats.get('std'))
            if val is None or std is None:
                continue

            profiles.append(abbrev_profile(list(exp['scenario'].values())[0]))
            sac_betas.append(val)
            sac_stds.append(std)

            betas, stds = collect_baseline_vectors(exp, baseline_data, BASELINE_NAMES)
            for name in BASELINE_NAMES:
                bl_betas[name].append(betas[name])
                bl_stds[name].append(stds[name])

            if exp['run_name'] not in baseline_data:
                print(f"  ℹ️  No baseline for '{exp['run_name']}' [{rf}] — SAC bar only")

        if not profiles:
            continue

        has_any_baseline = any(
            any(v is not None for v in bl_betas[name]) for name in BASELINE_NAMES
        )
        if not has_any_baseline and baseline_data:
            print(f"  ⚠️  [{rf}] Baseline loaded ({len(baseline_data)} entries) "
                  f"but none matched homogeneous run_names.")
            print(f"       Loaded: {sorted(baseline_data.keys())}")
            print(f"       Exps:   {[e['run_name'] for e in exps]}")

        fig, ax = plt.subplots(figsize=(12, 6))
        x_pos = np.arange(len(profiles))
        has_bl = _plot_grouped_bars(ax, x_pos, profiles, sac_betas, sac_stds,
                                    bl_betas, bl_stds)

        ax.set_xlabel('Traffic Profile')
        ax.set_ylabel(LABEL_BETA)
        bl_suffix = ' (SAC vs Baselines)' if has_bl else ''
        ax.set_title(f'Performance Across Static Homogeneous Traffic'
                     f' — {rf_title} Reward{bl_suffix}')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(profiles)
        ax.grid(True, alpha=0.3, axis='y')
        if has_bl:
            ax.legend(loc='best', framealpha=0.9)

        plt.tight_layout()
        fname = f'fig2_static_homogeneous{sfx}.{FIGURE_FORMAT}'
        plt.savefig(os.path.join(output_dir, fname), dpi=FIGURE_DPI,
                    bbox_inches='tight')
        plt.close()
        tag = 'with baselines' if has_bl else 'SAC only'
        print(f"  ✓ {fname} ({tag})")


# ---------------------------------------------------------------------------
# Figure 3: Heterogeneous
# ---------------------------------------------------------------------------

def fig3_heterogeneous(exps_by_cat, baseline_data, output_dir):
    """
    One bar chart per reward formulation present in the data.
    Files: fig3_heterogeneous_global.png, fig3_heterogeneous_weighted.png
    """
    print("\n[Figure 3] Heterogeneous Performance...")

    rf_list = _reward_formulations(
        exps_by_cat['static_heterogeneous'])

    for rf in rf_list:
        sfx = _rf_suffix(rf)
        rf_title = rf.capitalize()

        exps = [e for e in exps_by_cat['static_heterogeneous']
                if e.get('reward_formulation', 'global') == rf]
        if not exps:
            continue

        scenarios, sac_betas, sac_stds = [], [], []
        bl_betas = {name: [] for name in BASELINE_NAMES}
        bl_stds  = {name: [] for name in BASELINE_NAMES}

        for exp in exps:
            stats = exp['statistics'].get('beta', {})
            val = stats.get('last_100_mean', stats.get('mean'))
            std = stats.get('last_100_std', stats.get('std'))
            if val is None or std is None:
                continue

            scenarios.append(abbrev_scenario_str(exp['scenario_str']))
            sac_betas.append(val)
            sac_stds.append(std)

            betas, stds = collect_baseline_vectors(exp, baseline_data, BASELINE_NAMES)
            for name in BASELINE_NAMES:
                bl_betas[name].append(betas[name])
                bl_stds[name].append(stds[name])

            if exp['run_name'] not in baseline_data:
                print(f"  ℹ️  No baseline for '{exp['run_name']}' [{rf}] — SAC bar only")

        if not scenarios:
            continue

        has_any_baseline = any(
            any(v is not None for v in bl_betas[name]) for name in BASELINE_NAMES
        )
        if not has_any_baseline and baseline_data:
            print(f"  ⚠️  [{rf}] Baseline loaded ({len(baseline_data)} entries) "
                  f"but none matched heterogeneous run_names.")
            print(f"       Loaded: {sorted(baseline_data.keys())}")
            print(f"       Exps:   {[e['run_name'] for e in exps]}")

        fig, ax = plt.subplots(figsize=(12, 6))
        x_pos = np.arange(len(scenarios))
        has_bl = _plot_grouped_bars(ax, x_pos, scenarios, sac_betas, sac_stds,
                                    bl_betas, bl_stds)

        ax.set_xlabel('Traffic Scenario')
        ax.set_ylabel(LABEL_BETA)
        bl_suffix = ' (SAC vs Baselines)' if has_bl else ''
        ax.set_title(f'Performance Under Heterogeneous Traffic'
                     f' — {rf_title} Reward{bl_suffix}')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(scenarios)
        ax.grid(True, alpha=0.3, axis='y')
        if has_bl:
            ax.legend(loc='best', framealpha=0.9)

        plt.tight_layout()
        fname = f'fig3_heterogeneous{sfx}.{FIGURE_FORMAT}'
        plt.savefig(os.path.join(output_dir, fname), dpi=FIGURE_DPI,
                    bbox_inches='tight')
        plt.close()
        tag = 'with baselines' if has_bl else 'SAC only'
        print(f"  ✓ {fname} ({tag})")


# ---------------------------------------------------------------------------
# Figure 4: Static Allocation Patterns
# ---------------------------------------------------------------------------

_BOX_COLOR_PAIRS = [
    ('steelblue',    'lightsteelblue'),
    ('coral',        'lightcoral'),
    ('seagreen',     'lightgreen'),
    ('mediumpurple', 'plum'),
    ('goldenrod',    'moccasin'),
]


def fig4_allocation_patterns(experiments, output_dir):
    """
    One box-plot figure per reward formulation present in the data.
    Files: fig4_allocation_patterns_global.png, fig4_allocation_patterns_weighted.png
    """
    print("\n[Figure 4] Allocation Patterns (Multiple Scenarios)...")

    rf_list = _reward_formulations(
        [e for e in experiments if e['category'] == 'static_heterogeneous'])

    for rf in rf_list:
        sfx = _rf_suffix(rf)
        rf_title = rf.capitalize()

        eligible = [
            exp for exp in experiments
            if exp['category'] == 'static_heterogeneous'
            and exp.get('reward_formulation', 'global') == rf
            and 'ep80_action_slice0' in exp['data']
            and 'ep80_action_slice1' in exp['data']
        ]

        if not eligible:
            print(f"  ⚠️  [{rf}] No heterogeneous experiments with ep80 allocation data")
            continue

        all_data, all_labels, all_colors = [], [], []
        for i, exp in enumerate(eligible):
            pair   = _BOX_COLOR_PAIRS[i % len(_BOX_COLOR_PAIRS)]
            sname  = exp['scenario_str']
            slabels = exp.get('slice_labels', [])
            print(f"  ✓ [{rf}] Including '{sname}'")
            for idx, key in enumerate(['ep80_action_slice0', 'ep80_action_slice1']):
                slice_name = slabels[idx] if idx < len(slabels) else f'Slice {idx}'
                all_data.append(exp['data'][key]['values'])
                all_labels.append(f'{abbrev_scenario_str(sname)}\n{slice_name}')
                all_colors.append(pair[idx])

        fig, ax = plt.subplots(figsize=(max(10, len(all_data) * 1.5), 6))
        bp = ax.boxplot(all_data, labels=all_labels, patch_artist=True,
                        boxprops=dict(alpha=0.7),
                        medianprops=dict(color='red', linewidth=2),
                        widths=0.6)
        for patch, color in zip(bp['boxes'], all_colors):
            patch.set_facecolor(color)

        ax.set_ylabel(LABEL_RBS, fontsize=12)
        ax.set_xlabel('Scenario and Slice', fontsize=12)
        ax.set_title(
            f'Allocation Distribution Across Heterogeneous Scenarios (Episode 80)'
            f' — {rf_title} Reward',
            fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        fname = f'fig4_allocation_patterns{sfx}.{FIGURE_FORMAT}'
        plt.savefig(os.path.join(output_dir, fname), dpi=FIGURE_DPI,
                    bbox_inches='tight')
        plt.close()
        print(f"  ✓ {fname} ({len(eligible)} scenarios, {len(all_data)} box plots)")
