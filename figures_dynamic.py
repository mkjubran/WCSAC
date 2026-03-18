"""
figures_dynamic.py — Figure 5 and supporting dynamic-scenario plots.

    fig5_dynamic_scenarios           — 5a/b/c/d/d2/d3/e/f/f2/f3
    fig_per_slice_beta_training      — per-slice β vs global β (full training)
    fig_fairness_evolution           — fairness ratio over training
    fig_episode_per_slice_beta       — per-slice β within a single episode
    fig_slice_beta_boxplot           — final distribution box plot per slice
    fig_actor_loss_comparison        — actor loss comparison
    fig_ablation_reward_formulation_robust  — heterogeneous ablation bar chart
    fig_dynamic_ablation_comparison  — 2×2 dynamic ablation subplots
    fig_dynamic_ablation_bar_chart   — dynamic ablation grouped bar chart

Ablation figures require an ablation_map dict that explicitly maps each
run_name to its reward formulation label (e.g. 'global' or 'uniform').
Pass it via the --ablation-map CLI argument (see step2_generate_figures.py).
No date-based or name-pattern heuristics are used anywhere in this module.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from plot_style import (
    FIGURE_DPI, FIGURE_FORMAT,
    COLOR_BETA, LABEL_BETA, LABEL_DTI, LABEL_RBS, LABEL_EPISODE,
)

_PROFILE_NAMES = {
    0: 'Uniform',
    1: 'Extremely Low',
    2: 'Low',
    3: 'Medium',
    4: 'High',
    5: 'Extremely High',
    6: 'External',
}


# ---------------------------------------------------------------------------
# Figure 5: Dynamic Scenarios
# ---------------------------------------------------------------------------

def fig5_dynamic_scenarios(exps_by_cat, output_dir):
    """
    Generate all dynamic scenario figures (5a–5f3) for every dynamic experiment.

    When there is only one dynamic experiment the output filenames use the
    standard fig5* scheme.  When there are multiple, a short suffix derived
    from the experiment's run_name is appended so files don't overwrite each
    other (e.g. fig5a_dynamic_beta__run_xyz.png).
    """
    print("\n[Figure 5] Dynamic Scenarios...")

    exps = exps_by_cat['dynamic']
    if not exps:
        print("  ⚠️  No dynamic experiments")
        return

    multi = len(exps) > 1

    for exp in exps:
        suffix = f'__{exp["run_name"]}' if multi else ''
        data   = exp['data']
        pool   = exp.get('dynamic_profile_set', [])
        period = exp.get('dynamic_change_period')
        pool_str   = f"[{', '.join(pool)}]" if pool else ''
        period_str = f", T={period} DTIs" if period else ''
        print(f"\n  Processing: {exp['run_name']} — {exp['scenario_str']}{pool_str}{period_str}")

        # 5a: Beta time series
        _fig5a_beta_timeseries(data, output_dir, suffix)

        # 5b: Allocation time series
        _fig5b_allocation_timeseries(exp, output_dir, suffix)

        # Episode 80
        _fig5_episode_profiles(data, output_dir, episode=80,
                                label=f'fig5c{suffix}')
        _fig5_allocation_periods(exp, output_dir, episode=80,
                                 label_prefix=f'fig5d{suffix}')
        _fig5_continuous_allocation(exp, output_dir, episode=80,
                                    label=f'fig5d2{suffix}')
        _fig5_continuous_beta(data, output_dir, episode=80,
                              label=f'fig5d3{suffix}')

        # Episode 160
        _fig5_episode_profiles(data, output_dir, episode=160,
                                label=f'fig5e{suffix}')
        _fig5_allocation_periods(exp, output_dir, episode=160,
                                 label_prefix=f'fig5f{suffix}')
        _fig5_continuous_allocation(exp, output_dir, episode=160,
                                    label=f'fig5f2{suffix}')
        _fig5_continuous_beta(data, output_dir, episode=160,
                              label=f'fig5f3{suffix}')


def _fig5a_beta_timeseries(data, output_dir, suffix=''):
    if 'dti_beta' not in data:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    steps = np.array(data['dti_beta']['steps'])
    betas = np.array(data['dti_beta']['values'])
    ax.plot(steps, betas, color=COLOR_BETA, alpha=0.7, linewidth=1)
    ax.set_xlabel(LABEL_DTI)
    ax.set_ylabel(LABEL_BETA)
    ax.set_title('Dynamic Traffic Adaptation: QoS Performance')
    ax.grid(True, alpha=0.3)
    _save(fig, output_dir, f'fig5a_dynamic_beta{suffix}')


def _fig5b_allocation_timeseries(exp, output_dir, suffix=''):
    data   = exp['data']
    labels = exp.get('slice_labels', [])
    K      = exp.get('K', 2)

    plotted = False
    fig, ax = plt.subplots(figsize=(12, 6))
    for k in range(K):
        key = f'dti_action_slice{k}'
        if key in data:
            slice_name = labels[k] if k < len(labels) else f'Slice {k}'
            ax.plot(np.array(data[key]['steps']),
                    np.array(data[key]['values']),
                    label=slice_name, alpha=0.7, linewidth=1)
            plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel(LABEL_DTI)
    ax.set_ylabel(LABEL_RBS)
    ax.set_title('Dynamic Traffic Adaptation: Resource Allocation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, output_dir, f'fig5b_dynamic_allocation{suffix}')


def _fig5_episode_profiles(data, output_dir, episode, label):
    """
    Active traffic profiles within a single episode.

    Uses the per-episode keys logged directly by step1:
      episode_{N}/active_profile_slice0  →  ep{N}_active_profile_slice0
      episode_{N}/active_profile_slice1  →  ep{N}_active_profile_slice1
    The x-axis index is the position within the episode's values list (0-based DTI).
    Y-axis ticks are derived from the profile integer values actually present in
    the data, mapped through _PROFILE_NAMES where available.
    """
    s0_key = f'ep{episode}_active_profile_slice0'
    s1_key = f'ep{episode}_active_profile_slice1'
    if s0_key not in data or s1_key not in data:
        print(f"  ⚠️  Skipping {label}: '{s0_key}' or '{s1_key}' not in data")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    style = [('-', 'steelblue', (None, None)), ('--', 'coral', (5, 3))]
    all_values = []
    for idx, (key, (ls, color, dashes)) in enumerate(
            zip([s0_key, s1_key], style)):
        values = np.array(data[key]['values'])
        all_values.extend(values.tolist())
        dtis = np.arange(len(values))
        kw = dict(where='post', alpha=0.8, linewidth=2.5, linestyle=ls, color=color)
        if dashes[0] is not None:
            kw['dashes'] = dashes
        ax.step(dtis, values, label=f'Slice {idx}', **kw)

    # Build yticks only from profile IDs actually present in the data
    present_ids = sorted(set(int(v) for v in all_values))
    ax.set_yticks(present_ids)
    ax.set_yticklabels([_PROFILE_NAMES.get(i, str(i)) for i in present_ids])

    ax.set_xlabel(f'DTI (within Episode {episode})')
    ax.set_ylabel('Active Traffic Profile')
    ax.set_title(f'Traffic Profile Changes (Episode {episode})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, output_dir, f'{label}_dynamic_profiles_ep{episode}')


def _fig5_allocation_periods(exp, output_dir, episode, label_prefix):
    """
    Box plots per period within an episode.

    Period boundaries are derived from the steps array logged by step1.
    """
    data   = exp['data']
    labels = exp.get('slice_labels', [])
    K      = exp.get('K', 2)

    s0_key = f'ep{episode}_action_slice0'
    s1_key = f'ep{episode}_action_slice1'
    if s0_key not in data or s1_key not in data:
        return

    slice0 = data[s0_key]['values']
    slice1 = data[s1_key]['values']
    steps  = data[s0_key].get('steps', list(range(len(slice0))))

    # Infer period size from step gaps
    if len(steps) > 1:
        gaps = [steps[i+1] - steps[i] for i in range(len(steps) - 1)]
        period_size = max(set(gaps), key=gaps.count)
        period_size = max(period_size, 1)
    else:
        period_size = 1

    n_points   = len(slice0)
    num_periods = n_points // period_size if period_size > 0 else 1

    s0_name = labels[0] if len(labels) > 0 else 'Slice 0'
    s1_name = labels[1] if len(labels) > 1 else 'Slice 1'

    fig, ax = plt.subplots(figsize=(14, 6))
    box_data, box_labels, positions, colors = [], [], [], []

    for p in range(num_periods):
        start_idx = p * period_size
        end_idx   = (p + 1) * period_size
        s0p = slice0[start_idx:end_idx]
        s1p = slice1[start_idx:end_idx]
        if not s0p or not s1p:
            continue
        step_start = steps[start_idx]
        step_end   = steps[min(end_idx - 1, len(steps) - 1)]
        base = p * 2.5
        box_data   += [s0p, s1p]
        box_labels += [f'{step_start}-{step_end}\n{s0_name}',
                       f'{step_start}-{step_end}\n{s1_name}']
        positions  += [base, base + 1]
        colors     += ['lightblue', 'lightcoral']

    if box_data:
        bp = ax.boxplot(box_data, positions=positions, labels=box_labels,
                        patch_artist=True, widths=0.8)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for med in bp['medians']:
            med.set_color('red')
            med.set_linewidth(2)

    ax.set_xlabel(f'DTI Range within Episode {episode}')
    ax.set_ylabel(LABEL_RBS)
    ax.set_title(f'Allocation Distribution by Period (Episode {episode}, Dynamic)')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    ax.legend(handles=[
        Patch(facecolor='lightblue',  alpha=0.7, label=s0_name),
        Patch(facecolor='lightcoral', alpha=0.7, label=s1_name),
    ], loc='upper right')
    _save(fig, output_dir, f'{label_prefix}_dynamic_allocation_periods_ep{episode}',
          note=f'{num_periods} periods')


def _fig5_continuous_allocation(exp, output_dir, episode, label):
    """Continuous allocation curves for one episode."""
    data   = exp['data']
    labels = exp.get('slice_labels', [])

    s0_key = f'ep{episode}_action_slice0'
    s1_key = f'ep{episode}_action_slice1'
    if s0_key not in data or s1_key not in data:
        return

    steps   = data[s0_key].get('steps', list(range(len(data[s0_key]['values']))))
    s0_name = labels[0] if len(labels) > 0 else 'Slice 0'
    s1_name = labels[1] if len(labels) > 1 else 'Slice 1'

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(steps, data[s0_key]['values'], color='steelblue', alpha=0.8,
            linewidth=1.5, label=s0_name)
    ax.plot(steps, data[s1_key]['values'], color='coral', alpha=0.8,
            linewidth=1.5, label=s1_name)
    ax.set_xlabel(f'DTI (within Episode {episode})', fontsize=12)
    ax.set_ylabel(LABEL_RBS, fontsize=12)
    ax.set_title(f'Continuous Allocation Over Time (Episode {episode})',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    _save(fig, output_dir, f'{label}_continuous_allocation_ep{episode}')


def _fig5_continuous_beta(data, output_dir, episode, label):
    """Continuous beta curve for one episode."""
    ep_key = f'ep{episode}_beta'
    if ep_key not in data:
        print(f"  ⚠️  Skipping {label} — '{ep_key}' not in data")
        return

    steps = data[ep_key].get('steps', list(range(len(data[ep_key]['values']))))
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(steps, data[ep_key]['values'], color='red', alpha=0.8, linewidth=2.0,
            label='β (QoS Violation Ratio)')
    ax.axhline(y=0, color='green', linestyle='--', alpha=0.3, linewidth=1,
               label='Perfect QoS (β=0)')
    ax.set_xlabel(f'DTI (within Episode {episode})', fontsize=12)
    ax.set_ylabel(LABEL_BETA, fontsize=12)
    ax.set_title(f'QoS Violation Ratio Over Time (Episode {episode})',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    _save(fig, output_dir, f'{label}_continuous_beta_ep{episode}')


# ---------------------------------------------------------------------------
# Actor Loss Comparison
# ---------------------------------------------------------------------------

def fig_actor_loss_comparison(experiments, output_dir):
    """Training actor loss comparison: one heterogeneous + one dynamic experiment."""
    print("\n[Figure] Actor Loss Comparison...")

    to_plot = []
    for exp in experiments:
        if exp['category'] == 'static_heterogeneous' and len(to_plot) == 0:
            if 'episode_actor_loss' in exp.get('data', {}):
                to_plot.append(('Heterogeneous', exp, 'coral'))
        elif exp['category'] == 'dynamic' and len(to_plot) <= 1:
            if 'episode_actor_loss' in exp.get('data', {}):
                to_plot.append(('Dynamic', exp, 'steelblue'))
        if len(to_plot) == 2:
            break

    if not to_plot:
        print("  ⚠️  No actor loss data available")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, exp, color in to_plot:
        steps = np.array(exp['data']['episode_actor_loss']['steps'])
        values = np.array(exp['data']['episode_actor_loss']['values'])
        ax.plot(steps, values, alpha=0.3, color=color, linewidth=0.5)
        if len(values) >= 50:
            window = min(50, len(values) // 2)
            ma = np.convolve(values, np.ones(window) / window, mode='valid')
            ax.plot(steps[window - 1:], ma, color=color, linewidth=2,
                    label=f'{label} ({exp["scenario_str"]})')
        else:
            ax.plot(steps, values, color=color, linewidth=2,
                    label=f'{label} ({exp["scenario_str"]})')

    ax.set_xlabel(LABEL_EPISODE)
    ax.set_ylabel('Actor Loss')
    ax.set_title('Training Actor Loss Comparison')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    _save(fig, output_dir, 'fig_actor_loss_comparison')


# ---------------------------------------------------------------------------
# Per-Slice Beta Figures
# ---------------------------------------------------------------------------

def fig_per_slice_beta_training(experiments, output_dir):
    """Per-slice β vs global β over full training (dynamic experiment)."""
    print("\n[Figure] Per-Slice Beta Training...")

    exp = _find_dynamic_with_slices(experiments)
    if not exp:
        return

    data   = exp['data']
    labels = exp.get('slice_labels', [])

    fig, ax = plt.subplots(figsize=(12, 6))
    global_steps = np.array(data['dti_beta']['steps'])
    ax.plot(global_steps, np.array(data['dti_beta']['values']),
            label='Global β', color='black', linewidth=2.0, alpha=0.8)
    for k, color in enumerate(['steelblue', 'coral']):
        key = f'dti_beta_slice{k}'
        if key in data:
            slice_name = labels[k] if k < len(labels) else f'Slice {k}'
            slice_steps = np.array(data[key]['steps'])
            ax.plot(slice_steps, np.array(data[key]['values']),
                    label=f'{slice_name} β', color=color,
                    linewidth=1.5, alpha=0.7, linestyle='--')
    ax.set_xlabel(LABEL_DTI, fontsize=12)
    ax.set_ylabel(LABEL_BETA, fontsize=12)
    ax.set_title('Per-Slice vs Global QoS Violations', fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    _save(fig, output_dir, 'fig_per_slice_beta_training')


def fig_fairness_evolution(experiments, output_dir):
    """Fairness ratio (max β / min β) over training."""
    print("\n[Figure] Fairness Evolution...")

    exp = _find_dynamic_with_slices(experiments)
    if not exp:
        return

    data = exp['data']
    steps = np.array(data['dti_beta_slice0']['steps'])
    beta0 = np.array(data['dti_beta_slice0']['values'])
    beta1 = np.array(data['dti_beta_slice1']['values'])
    fairness = np.clip(
        np.maximum(beta0, beta1) / (np.minimum(beta0, beta1) + 1e-6),
        1.0, 10.0
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(steps, fairness, color='green', linewidth=2.0, alpha=0.8,
            label='Fairness Ratio')
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.5,
               label='Perfect Fairness')
    ax.set_xlabel(LABEL_DTI, fontsize=12)
    ax.set_ylabel('Fairness Ratio (max β / min β)', fontsize=12)
    ax.set_title('Slice Fairness During Training', fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 5.0])
    _save(fig, output_dir, 'fig_fairness_evolution')


def fig_episode_per_slice_beta(experiments, output_dir, episode=80):
    """Per-slice β within a single episode."""
    print(f"\n[Figure] Episode {episode} Per-Slice Beta...")

    ep_key = f'ep{episode}_beta'
    s0_key = f'ep{episode}_beta_slice0'
    s1_key = f'ep{episode}_beta_slice1'

    exp = None
    for e in experiments:
        if e['category'] == 'dynamic':
            if all(k in e['data'] for k in [ep_key, s0_key, s1_key]):
                exp = e
                break

    if not exp:
        print(f"  ⚠️  No dynamic experiment with episode {episode} per-slice beta data")
        return

    data   = exp['data']
    labels = exp.get('slice_labels', [])
    s0_name = labels[0] if len(labels) > 0 else 'Slice 0'
    s1_name = labels[1] if len(labels) > 1 else 'Slice 1'
    steps = data[ep_key].get('steps', list(range(len(data[ep_key]['values']))))
    global_beta = np.array(data[ep_key]['values'])

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(steps, global_beta, label='Global β', color='black', linewidth=2.0, alpha=0.8)
    ax.plot(steps, np.array(data[s0_key]['values']),
            label=f'{s0_name} β', color='steelblue', linewidth=1.5, alpha=0.7, linestyle='--')
    ax.plot(steps, np.array(data[s1_key]['values']),
            label=f'{s1_name} β', color='coral', linewidth=1.5, alpha=0.7, linestyle='--')
    ax.axhline(y=0, color='green', linestyle='--', alpha=0.3, linewidth=1)
    ax.set_xlabel(f'DTI (within Episode {episode})', fontsize=12)
    ax.set_ylabel(LABEL_BETA, fontsize=12)
    ax.set_title(f'Per-Slice QoS Violations (Episode {episode})',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    _save(fig, output_dir, f'fig5_ep{episode}_per_slice_beta')


def fig_slice_beta_boxplot(experiments, output_dir):
    """Final-training distribution of per-slice β (box plot)."""
    print("\n[Figure] Slice Beta Box Plot...")

    exp = _find_dynamic_with_slices(experiments)
    if not exp:
        return

    data   = exp['data']
    labels = exp.get('slice_labels', [])
    s0_name = labels[0] if len(labels) > 0 else 'Slice 0'
    s1_name = labels[1] if len(labels) > 1 else 'Slice 1'
    beta0 = np.array(data['dti_beta_slice0']['values'])
    beta1 = np.array(data['dti_beta_slice1']['values'])
    cutoff = int(len(beta0) * 0.8)
    fb0, fb1 = beta0[cutoff:], beta1[cutoff:]

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot([fb0, fb1], labels=[s0_name, s1_name],
                    patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for med in bp['medians']:
        med.set_color('red')
        med.set_linewidth(2)
    ax.set_ylabel(LABEL_BETA, fontsize=12)
    ax.set_title('Final Performance: Per-Slice QoS Distribution',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.0])
    ax.text(0.02, 0.98,
            f"Slice 0: μ={fb0.mean():.3f}, σ={fb0.std():.3f}\n"
            f"Slice 1: μ={fb1.mean():.3f}, σ={fb1.std():.3f}",
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=10)
    _save(fig, output_dir, 'fig_slice_beta_boxplot')


# ---------------------------------------------------------------------------
# Ablation: Dynamic Comparison
# ---------------------------------------------------------------------------

def fig_dynamic_ablation_comparison(experiments, output_dir, ablation_map):
    """
    2×2 subplots: traffic-range × reward-type.

    ablation_map: dict mapping run_name → reward-type label (e.g. 'global'/'uniform').
    Experiments are grouped first by the number of unique profiles in their
    scenario (traffic range), then by their reward type from ablation_map.
    """
    print("\n[Figure] Dynamic Scenarios Ablation...")

    groups = _group_dynamic_by_range(experiments, ablation_map)
    if groups is None:
        return

    r3, r5 = groups['range_3'], groups['range_5']

    if not all(r3.values()) or not all(r5.values()):
        print("  ⚠️  Missing experiments for complete comparison")
        _debug_dynamic(experiments, ablation_map)
        return

    # Derive axis titles from actual scenario_str values
    def _title(exp, reward_label):
        return f'{exp["scenario_str"]} — {reward_label.capitalize()} Reward'

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    combos = [
        (axes[0, 0], r3['global'],  _title(r3['global'],  'global'),  'steelblue'),
        (axes[0, 1], r3['uniform'], _title(r3['uniform'], 'uniform'), 'coral'),
        (axes[1, 0], r5['global'],  _title(r5['global'],  'global'),  'steelblue'),
        (axes[1, 1], r5['uniform'], _title(r5['uniform'], 'uniform'), 'coral'),
    ]

    for ax, exp, title, color in combos:
        data = exp['data']
        if 'dti_beta' in data:
            steps = np.array(data['dti_beta']['steps'])
            ax.plot(steps, np.array(data['dti_beta']['values']),
                    color=color, alpha=0.7, linewidth=1.5, label='Global β')
            for skey, sc in [('dti_beta_slice0', 'lightskyblue'),
                              ('dti_beta_slice1', 'lightcoral')]:
                if skey in data:
                    sidx = skey.split('slice')[-1]
                    ax.plot(steps, np.array(data[skey]['values']),
                            color=sc, alpha=0.5, linewidth=1, linestyle='--',
                            label=f'β{sidx}')
        ax.set_xlabel('DTI', fontsize=11)
        ax.set_ylabel('β', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

    plt.suptitle('Dynamic Traffic Adaptation: Reward Formulation Comparison',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    _save(fig, output_dir, 'fig_dynamic_ablation_comparison')


def fig_dynamic_ablation_bar_chart(experiments, output_dir, ablation_map):
    """
    Bar chart comparing Global β, β0, β1, JFI across dynamic ranges/rewards.

    ablation_map: dict mapping run_name → reward-type label.
    """
    print("\n[Figure] Dynamic Scenarios Performance Comparison...")

    groups = _group_dynamic_by_range(experiments, ablation_map)
    if groups is None:
        return

    r3, r5 = groups['range_3'], groups['range_5']

    if not all(r3.values()) or not all(r5.values()):
        print("  ⚠️  Missing experiments")
        return

    # Build x-axis labels from the actual scenario_str
    labels = [
        f'{r3["global"]["scenario_str"]}\n{_reward_label(r3["global"], ablation_map).capitalize()}',
        f'{r3["uniform"]["scenario_str"]}\n{_reward_label(r3["uniform"], ablation_map).capitalize()}',
        f'{r5["global"]["scenario_str"]}\n{_reward_label(r5["global"], ablation_map).capitalize()}',
        f'{r5["uniform"]["scenario_str"]}\n{_reward_label(r5["uniform"], ablation_map).capitalize()}',
    ]
    exps_ordered = [r3['global'], r3['uniform'], r5['global'], r5['uniform']]

    g_betas, b0s, b1s, jfis = [], [], [], []
    for exp in exps_ordered:
        stats = exp['statistics'].get('beta', {})
        g_betas.append(stats.get('last_100_mean', stats.get('mean', 0)) or 0)

        b0m, _, b1m, _ = _final_slice_means(exp)
        if b0m is not None:
            b0s.append(b0m)
            b1s.append(b1m)
            s0, s1 = 1 - b0m, 1 - b1m
            denom = 2 * (s0 ** 2 + s1 ** 2)
            jfis.append((s0 + s1) ** 2 / denom if denom > 0 else 0)
        else:
            b0s.append(0)
            b1s.append(0)
            jfis.append(0)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    w = 0.2
    for offset, vals, label, color in [
        (-1.5 * w, g_betas, 'Global β', 'steelblue'),
        (-0.5 * w, b0s,     'β0',        'lightskyblue'),
        (0.5 * w,  b1s,     'β1',        'lightcoral'),
        (1.5 * w,  jfis,    'JFI',       'lightgreen'),
    ]:
        ax.bar(x + offset, vals, w, label=label,
               alpha=0.8, edgecolor='black', linewidth=1.2, color=color)

    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title('Dynamic Traffic: Performance Across Ranges and Reward Formulations',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])
    _save(fig, output_dir, 'fig_dynamic_ablation_bar')


# ---------------------------------------------------------------------------
# Ablation: Heterogeneous Reward Formulation
# ---------------------------------------------------------------------------

def fig_ablation_reward_formulation_robust(experiments, output_dir, ablation_map):
    """
    Grouped bar chart: reward formulation comparison across heterogeneous
    scenarios, showing Global β / β0 / β1 / JFI per reward type.

    ablation_map: dict mapping run_name → reward-type label.
    Experiments that share the same scenario_str are paired by reward type.
    """
    print("\n[Figure] Ablation Study: Reward Formulation Comparison...")

    hetero_exps = [e for e in experiments if e['category'] == 'static_heterogeneous']
    complete = _group_by_reward(hetero_exps, ablation_map)

    if not complete:
        print("  ⚠️  No scenarios with paired reward formulations in ablation_map!")
        _debug_ablation(hetero_exps, ablation_map)
        return

    # Collect all reward-type labels present (preserving insertion order)
    all_rtypes = []
    for pairs in complete.values():
        for rt in pairs:
            if rt not in all_rtypes:
                all_rtypes.append(rt)

    scenario_names = sorted(complete.keys())
    # One list of bar values per reward-type per metric
    data_by_rtype = {rt: {'betas': [], 'b0s': [], 'b1s': [], 'jfis': []}
                     for rt in all_rtypes}

    for sname in scenario_names:
        for rt in all_rtypes:
            exp = complete[sname].get(rt)
            if exp is None:
                # Scenario not available for this reward type
                data_by_rtype[rt]['betas'].append(0)
                data_by_rtype[rt]['b0s'].append(0)
                data_by_rtype[rt]['b1s'].append(0)
                data_by_rtype[rt]['jfis'].append(0)
                continue

            stats = exp['statistics'].get('beta', {})
            beta_val = stats.get('last_100_mean', stats.get('mean', 0)) or 0
            data_by_rtype[rt]['betas'].append(beta_val)

            b0m, _, b1m, _ = _final_slice_means(exp)
            if b0m is not None:
                s0, s1 = 1 - b0m, 1 - b1m
                denom = 2 * (s0 ** 2 + s1 ** 2)
                jfi = (s0 + s1) ** 2 / denom if denom > 0 else 0
                data_by_rtype[rt]['b0s'].append(b0m)
                data_by_rtype[rt]['b1s'].append(b1m)
                data_by_rtype[rt]['jfis'].append(jfi)
            else:
                data_by_rtype[rt]['b0s'].append(0)
                data_by_rtype[rt]['b1s'].append(0)
                data_by_rtype[rt]['jfis'].append(0)

    # 4 metrics × N reward-types bars per scenario
    n_metrics = 4
    n_rtypes = len(all_rtypes)
    n_bars = n_metrics * n_rtypes
    w = 0.8 / n_bars

    # Colour palette: cycle through two hue families (blues then oranges)
    palette = ['#5B9BD5', '#9DC3E6', '#BDD7EE', '#DEEBF7',
               '#ED7D31', '#F4B183', '#FCE4D6', '#92D050']

    fig, ax = plt.subplots(figsize=(max(12, len(scenario_names) * 3), 7))
    x = np.arange(len(scenario_names))

    bar_idx = 0
    for rt in all_rtypes:
        d = data_by_rtype[rt]
        for metric_label, vals in [
            (f'Global β ({rt})', d['betas']),
            (f'β0 ({rt})',        d['b0s']),
            (f'β1 ({rt})',        d['b1s']),
            (f'JFI ({rt})',       d['jfis']),
        ]:
            offset = (bar_idx - n_bars / 2 + 0.5) * w
            color = palette[bar_idx % len(palette)]
            ax.bar(x + offset, vals, w, label=metric_label,
                   color=color, alpha=0.9, edgecolor='black', linewidth=1.0)
            bar_idx += 1

    ax.set_xlabel('Traffic Scenario', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title('Ablation Study: Reward Formulation Comparison\n'
                 'Per-Slice Violation Ratios and Fairness Across Heterogeneous Scenarios',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, rotation=15, ha='right')
    ax.legend(loc='upper left', fontsize=8, ncol=n_rtypes * 2, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])
    ax.text(0.98, 0.02,
            "β = violation ratio (lower is better).\n"
            "JFI (Jain's Fairness Index): 1 = perfect fairness.",
            transform=ax.transAxes, va='bottom', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)
    _save(fig, output_dir, 'fig_ablation_reward_robust')


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _save(fig, output_dir, stem, note=None):
    path = os.path.join(output_dir, f'{stem}.{FIGURE_FORMAT}')
    plt.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    msg = f"  ✓ {stem}.{FIGURE_FORMAT}"
    if note:
        msg += f" ({note})"
    print(msg)


def _find_dynamic_with_slices(experiments):
    for exp in experiments:
        if (exp['category'] == 'dynamic' and
                'dti_beta_slice0' in exp['data'] and
                'dti_beta_slice1' in exp['data']):
            print(f"  Using experiment: {exp['run_name']}")
            return exp
    print("  ⚠️  No dynamic experiment with per-slice beta data found")
    for exp in experiments:
        if exp['category'] == 'dynamic':
            has = 'dti_beta_slice0' in exp['data']
            print(f"        - {exp['run_name']}: has_per_slice={has}")
    return None


def _final_slice_means(exp, cutoff_frac=0.8):
    """Return (b0_mean, b0_std, b1_mean, b1_std) or (None,)*4."""
    d = exp['data']
    if 'dti_beta_slice0' not in d or 'dti_beta_slice1' not in d:
        return None, None, None, None
    b0 = np.array(d['dti_beta_slice0']['values'])
    b1 = np.array(d['dti_beta_slice1']['values'])
    cut = int(len(b0) * cutoff_frac)
    return (float(np.mean(b0[cut:])), float(np.std(b0[cut:])),
            float(np.mean(b1[cut:])), float(np.std(b1[cut:])))


def _build_ablation_map_from_data(experiments):
    """
    Build an ablation_map dict from the reward_formulation field stored
    in each experiment by step1.  Falls back gracefully if the field is absent.

    Returns:
        {run_name: reward_formulation_label, ...}
    """
    mapping = {}
    for exp in experiments:
        rf = exp.get('reward_formulation')
        if rf:
            mapping[exp['run_name']] = rf
    return mapping


def _reward_label(exp, ablation_map):
    """Look up the reward-type label for an experiment from ablation_map."""
    return ablation_map.get(exp['run_name'], 'unknown')


def _group_by_reward(exps, ablation_map):
    """
    Group experiments by (scenario_str, reward_type) using ablation_map.

    Returns:
        {scenario_str: {reward_type: exp, ...}, ...}
        Only scenarios that have at least two distinct reward types are returned.
    """
    groups = {}
    unrecognised = []
    for exp in exps:
        rtype = ablation_map.get(exp['run_name'])
        if rtype is None:
            unrecognised.append(exp['run_name'])
            continue
        key = exp['scenario_str']
        groups.setdefault(key, {})
        groups[key][rtype] = exp

    if unrecognised:
        print(f"  ⚠️  The following run_names are not in ablation_map and will be skipped:")
        for name in unrecognised:
            print(f"       {name}")

    # Keep only scenarios that have more than one reward type
    return {k: v for k, v in groups.items() if len(v) >= 2}


def _group_dynamic_by_range(experiments, ablation_map):
    """
    Split dynamic experiments into groups by number of unique traffic profiles
    (traffic range) and by reward type from ablation_map.

    Returns:
        {'range_3': {rtype: exp, ...}, 'range_5': {rtype: exp, ...}}
        or None if ablation_map is empty/None (ablation figures skipped).
    """
    if not ablation_map:
        print("  ⚠️  No ablation_map provided — skipping ablation figure")
        return None

    dynamic_exps = [e for e in experiments if e['category'] == 'dynamic']

    range_groups = {}
    unrecognised = []

    for exp in dynamic_exps:
        rtype = ablation_map.get(exp['run_name'])
        if rtype is None:
            unrecognised.append(exp['run_name'])
            continue
        n_profiles = len(set(exp['scenario'].values()))
        range_groups.setdefault(n_profiles, {})
        # First experiment seen for each (n_profiles, rtype) wins
        if rtype not in range_groups[n_profiles]:
            range_groups[n_profiles][rtype] = exp

    if unrecognised:
        print(f"  ⚠️  The following dynamic run_names are not in ablation_map:")
        for name in unrecognised:
            print(f"       {name}")

    # Map to the two named groups using the two smallest distinct range sizes
    sorted_ranges = sorted(range_groups.keys())
    if len(sorted_ranges) < 2:
        print(f"  ⚠️  Need dynamic experiments with at least 2 distinct traffic ranges, "
              f"found: {sorted_ranges}")
        return None

    return {
        'range_3': range_groups[sorted_ranges[0]],
        'range_5': range_groups[sorted_ranges[1]],
    }


def _debug_dynamic(experiments, ablation_map):
    print("\n  Available dynamic experiments:")
    for exp in experiments:
        if exp['category'] == 'dynamic':
            n = len(set(exp['scenario'].values()))
            has_s = 'dti_beta_slice0' in exp['data']
            rtype = ablation_map.get(exp['run_name'], '⚠️  NOT IN MAP')
            print(f"    - {exp['run_name']}: {n} profiles, "
                  f"per_slice={has_s}, reward_type={rtype}")


def _debug_ablation(exps, ablation_map):
    print("  Experiments and their ablation_map entries:")
    for exp in exps:
        rtype = ablation_map.get(exp['run_name'], '⚠️  NOT IN MAP')
        print(f"    {exp['run_name']} → {rtype}  (scenario: {exp['scenario_str']})")
