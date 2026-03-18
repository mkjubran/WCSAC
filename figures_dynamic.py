"""
figures_dynamic.py — Figure 5 and supporting dynamic-scenario plots.

    fig5_dynamic_scenarios           — 5a/b/c/d/d2/d3/e/f/f2/f3
    fig_per_slice_beta_training      — per-slice β vs global β (full training)
    fig_fairness_evolution           — fairness ratio over training
    fig_episode_per_slice_beta       — per-slice β within a single episode
    fig_slice_beta_boxplot           — final distribution box plot per slice
    fig_actor_loss_comparison        — actor loss comparison
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
    """Generate all dynamic scenario figures (5a–5f3)."""
    print("\n[Figure 5] Dynamic Scenarios...")

    exps = exps_by_cat['dynamic']
    if not exps:
        print("  ⚠️  No dynamic experiments")
        return

    exp = exps[0]
    data = exp['data']

    # 5a: Beta time series
    _fig5a_beta_timeseries(data, output_dir)

    # 5b: Allocation time series
    _fig5b_allocation_timeseries(data, output_dir)

    # Episode 80
    _fig5_episode_profiles(data, output_dir, episode=80, label='fig5c')
    _fig5_allocation_periods(data, output_dir, episode=80, label_prefix='fig5d')
    _fig5_continuous_allocation(data, output_dir, episode=80, label='fig5d2')
    _fig5_continuous_beta(data, output_dir, episode=80, label='fig5d3')

    # Episode 160
    _fig5_episode_profiles(data, output_dir, episode=160, label='fig5e')
    _fig5_allocation_periods(data, output_dir, episode=160, label_prefix='fig5f')
    _fig5_continuous_allocation(data, output_dir, episode=160, label='fig5f2')
    _fig5_continuous_beta(data, output_dir, episode=160, label='fig5f3')


def _fig5a_beta_timeseries(data, output_dir):
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
    for switch in range(200, int(max(steps)), 200):
        ax.axvline(x=switch, color='gray', linestyle=':', alpha=0.5)
    _save(fig, output_dir, 'fig5a_dynamic_beta')


def _fig5b_allocation_timeseries(data, output_dir):
    fig, ax = plt.subplots(figsize=(12, 6))
    for key in ['dti_action_slice0', 'dti_action_slice1']:
        if key in data:
            slice_num = key.split('slice')[-1]
            ax.plot(np.array(data[key]['steps']),
                    np.array(data[key]['values']),
                    label=f'Slice {slice_num}', alpha=0.7, linewidth=1)
    ax.set_xlabel(LABEL_DTI)
    ax.set_ylabel(LABEL_RBS)
    ax.set_title('Dynamic Traffic Adaptation: Resource Allocation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    if 'dti_beta' in data:
        max_step = max(data['dti_beta']['steps'])
        for switch in range(200, int(max_step), 200):
            ax.axvline(x=switch, color='gray', linestyle=':', alpha=0.5)
    _save(fig, output_dir, 'fig5b_dynamic_allocation')


def _fig5_episode_profiles(data, output_dir, episode, label):
    """Active traffic profiles within a single episode."""
    ep_key = f'ep{episode}_action_slice0'
    if ep_key not in data:
        return
    if ('dti_active_profile_slice0' not in data or
            'dti_active_profile_slice1' not in data):
        return

    ep_start = episode * 2000
    ep_end = (episode + 1) * 2000

    fig, ax = plt.subplots(figsize=(10, 6))
    style = [('-', 'steelblue'), ('--', 'coral')]
    for idx, key in enumerate(['dti_active_profile_slice0', 'dti_active_profile_slice1']):
        ls, color = style[idx]
        all_steps = np.array(data[key]['steps'])
        all_values = np.array(data[key]['values'])
        mask = (all_steps >= ep_start) & (all_steps < ep_end)
        steps = all_steps[mask] - ep_start
        values = all_values[mask]
        kw = dict(where='post', alpha=0.8, linewidth=2.5, linestyle=ls, color=color)
        if idx == 1:
            kw['dashes'] = (5, 3)
        ax.step(steps, values, label=f'Slice {idx}', **kw)

    ax.set_xlabel(f'DTI (within Episode {episode})')
    ax.set_ylabel('Active Traffic Profile')
    ax.set_title(f'Traffic Profile Changes (Episode {episode})')
    ax.set_yticks(list(_PROFILE_NAMES.keys()))
    ax.set_yticklabels(list(_PROFILE_NAMES.values()))
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, output_dir, f'{label}_dynamic_profiles_ep{episode}')


def _fig5_allocation_periods(data, output_dir, episode, label_prefix):
    """Box plots per 200-DTI period within an episode."""
    s0_key = f'ep{episode}_action_slice0'
    s1_key = f'ep{episode}_action_slice1'
    if s0_key not in data or s1_key not in data:
        return

    slice0 = data[s0_key]['values']
    slice1 = data[s1_key]['values']
    period_size = 200
    num_periods = len(slice0) // period_size

    fig, ax = plt.subplots(figsize=(14, 6))
    box_data, labels, positions, colors = [], [], [], []

    for p in range(num_periods):
        start, end = p * period_size, (p + 1) * period_size
        s0p, s1p = slice0[start:end], slice1[start:end]
        if not s0p or not s1p:
            continue
        base = p * 2.5
        box_data += [s0p, s1p]
        labels += [f'{start}-{end-1}\nS0', f'{start}-{end-1}\nS1']
        positions += [base, base + 1]
        colors += ['lightblue', 'lightcoral']

    if box_data:
        bp = ax.boxplot(box_data, positions=positions, labels=labels,
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
        Patch(facecolor='lightblue', alpha=0.7, label='Slice 0'),
        Patch(facecolor='lightcoral', alpha=0.7, label='Slice 1'),
    ], loc='upper right')
    _save(fig, output_dir, f'{label_prefix}_dynamic_allocation_periods_ep{episode}',
          note=f'{num_periods} periods')


def _fig5_continuous_allocation(data, output_dir, episode, label):
    """Continuous allocation curves (TensorBoard-style) for one episode."""
    s0_key = f'ep{episode}_action_slice0'
    s1_key = f'ep{episode}_action_slice1'
    if s0_key not in data or s1_key not in data:
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    dtis = np.arange(len(data[s0_key]['values']))
    ax.plot(dtis, data[s0_key]['values'], color='steelblue', alpha=0.8,
            linewidth=1.5, label='Slice 0')
    ax.plot(dtis, data[s1_key]['values'], color='coral', alpha=0.8,
            linewidth=1.5, label='Slice 1')
    for p in range(1, 10):
        ax.axvline(x=p * 200, color='gray', linestyle=':', alpha=0.5, linewidth=1)
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
    print(f"\n  [DEBUG] 'ep{episode}_beta' in data: {ep_key in data}")
    if ep_key not in data:
        print(f"  ⚠️  Skipping {label} — '{ep_key}' not in data")
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    betas = data[ep_key]['values']
    dtis = np.arange(len(betas))
    ax.plot(dtis, betas, color='red', alpha=0.8, linewidth=2.0,
            label='β (QoS Violation Ratio)')
    for p in range(1, 10):
        ax.axvline(x=p * 200, color='gray', linestyle=':', alpha=0.5, linewidth=1)
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

    data = exp['data']
    steps = np.array(data['dti_beta']['steps'])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(steps, np.array(data['dti_beta']['values']),
            label='Global β', color='black', linewidth=2.0, alpha=0.8)
    for k, (color, name) in enumerate(zip(['steelblue', 'coral'], ['VoIP', 'CBR'])):
        key = f'dti_beta_slice{k}'
        if key in data:
            ax.plot(steps, np.array(data[key]['values']),
                    label=f'Slice {k} β ({name})', color=color,
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

    data = exp['data']
    global_beta = np.array(data[ep_key]['values'])
    dtis = np.arange(len(global_beta))

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dtis, global_beta, label='Global β', color='black', linewidth=2.0, alpha=0.8)
    ax.plot(dtis, np.array(data[s0_key]['values']),
            label='Slice 0 β (VoIP)', color='steelblue', linewidth=1.5, alpha=0.7, linestyle='--')
    ax.plot(dtis, np.array(data[s1_key]['values']),
            label='Slice 1 β (CBR)', color='coral', linewidth=1.5, alpha=0.7, linestyle='--')
    for p in range(1, 10):
        ax.axvline(x=p * 200, color='gray', linestyle=':', alpha=0.5, linewidth=1)
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

    data = exp['data']
    beta0 = np.array(data['dti_beta_slice0']['values'])
    beta1 = np.array(data['dti_beta_slice1']['values'])
    cutoff = int(len(beta0) * 0.8)
    fb0, fb1 = beta0[cutoff:], beta1[cutoff:]

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot([fb0, fb1], labels=['Slice 0 (VoIP)', 'Slice 1 (CBR)'],
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

def fig_dynamic_ablation_comparison(experiments, output_dir):
    """2×2 subplots: [L,M,H] vs [EL,L,M,H,EH] × Global vs Uniform reward."""
    print("\n[Figure] Dynamic Scenarios Ablation...")

    groups = _group_dynamic_by_range(experiments)
    r3, r5 = groups['range_3'], groups['range_5']

    if not all(r3.values()) or not all(r5.values()):
        print("  ⚠️  Missing experiments for complete comparison")
        _debug_dynamic(experiments)
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    combos = [
        (axes[0, 0], r3['global'],  '[L,M,H] Range — Global β Reward',    'steelblue'),
        (axes[0, 1], r3['uniform'], '[L,M,H] Range — Uniform Weighted',    'coral'),
        (axes[1, 0], r5['global'],  '[EL,L,M,H,EH] Range — Global β',      'steelblue'),
        (axes[1, 1], r5['uniform'], '[EL,L,M,H,EH] Range — Uniform',        'coral'),
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

    plt.suptitle('Dynamic Traffic Adaptation: Global vs Uniform Reward',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    _save(fig, output_dir, 'fig_dynamic_ablation_comparison')


def fig_dynamic_ablation_bar_chart(experiments, output_dir):
    """Bar chart comparing Global β, β0, β1, JFI across dynamic ranges/rewards."""
    print("\n[Figure] Dynamic Scenarios Performance Comparison...")

    groups = _group_dynamic_by_range(experiments)
    r3, r5 = groups['range_3'], groups['range_5']

    if not all(r3.values()) or not all(r5.values()):
        print("  ⚠️  Missing experiments")
        return

    labels = ['[L,M,H]\nGlobal', '[L,M,H]\nUniform',
              '[EL,L,M,H,EH]\nGlobal', '[EL,L,M,H,EH]\nUniform']
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
        (-0.5 * w, b0s, 'β0', 'lightskyblue'),
        (0.5 * w, b1s, 'β1', 'lightcoral'),
        (1.5 * w, jfis, 'JFI', 'lightgreen'),
    ]:
        ax.bar(x + offset, vals, w, label=label,
               alpha=0.8, edgecolor='black', linewidth=1.2, color=color)

    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title('Dynamic Traffic: Performance Across Ranges and Reward Formulations',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])
    _save(fig, output_dir, 'fig_dynamic_ablation_bar')


# ---------------------------------------------------------------------------
# Ablation: Heterogeneous Reward Formulation
# ---------------------------------------------------------------------------

def fig_ablation_reward_formulation_robust(experiments, output_dir):
    """
    Grouped bar chart: Global β reward vs Uniform weighted reward across
    heterogeneous scenarios, showing Global β / β0 / β1 / JFI.
    """
    print("\n[Figure] Ablation Study: Reward Formulation Comparison...")

    hetero_exps = [e for e in experiments if e['category'] == 'static_heterogeneous']
    complete = _group_hetero_by_reward(hetero_exps)

    if not complete:
        print("  ⚠️  No scenarios with both Global and Uniform reward formulations!")
        return

    scenario_names = sorted(complete.keys())
    g_betas, g_b0s, g_b1s, g_jfis = [], [], [], []
    u_betas, u_b0s, u_b1s, u_jfis = [], [], [], []

    for sname in scenario_names:
        for vals_list, b0_list, b1_list, jfi_list, exp in [
            (g_betas, g_b0s, g_b1s, g_jfis, complete[sname]['global']),
            (u_betas, u_b0s, u_b1s, u_jfis, complete[sname]['uniform']),
        ]:
            stats = exp['statistics'].get('beta', {})
            beta_val = stats.get('last_100_mean', stats.get('mean', 0)) or 0
            vals_list.append(beta_val)

            b0m, _, b1m, _ = _final_slice_means(exp)
            if b0m is not None:
                b0_list.append(b0m)
                b1_list.append(b1m)
                s0, s1 = 1 - b0m, 1 - b1m
                denom = 2 * (s0 ** 2 + s1 ** 2)
                jfi_list.append((s0 + s1) ** 2 / denom if denom > 0 else 0)
            else:
                b0_list.append(0)
                b1_list.append(0)
                jfi_list.append(0)

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(scenario_names))
    w = 0.09

    bar_specs = [
        (-3.5 * w, g_betas, 'Global β (Global)',  '#5B9BD5'),
        (-2.5 * w, g_b0s,   'β0 (Global)',         '#9DC3E6'),
        (-1.5 * w, g_b1s,   'β1 (Global)',          '#BDD7EE'),
        (-0.5 * w, g_jfis,  'JFI (Global)',         '#DEEBF7'),
        (0.5 * w,  u_betas, 'Global β (Uniform)',  '#ED7D31'),
        (1.5 * w,  u_b0s,   'β0 (Uniform)',         '#F4B183'),
        (2.5 * w,  u_b1s,   'β1 (Uniform)',          '#FCE4D6'),
        (3.5 * w,  u_jfis,  'JFI (Uniform)',         '#92D050'),
    ]
    for offset, vals, label, color in bar_specs:
        ax.bar(x + offset, vals, w, label=label,
               color=color, alpha=0.9, edgecolor='black', linewidth=1.2)

    ax.set_xlabel('Traffic Scenario', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title(
        'Ablation Study: Global β Reward vs. Uniform Weighted Reward\n'
        'Per-Slice Violation Ratios and Fairness Across Heterogeneous Scenarios',
        fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, rotation=15, ha='right')
    ax.legend(loc='upper left', fontsize=9, ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])
    ax.text(0.98, 0.02,
            "Note: β = violation ratio (lower is better).\n"
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


def _group_dynamic_by_range(experiments):
    """Split dynamic experiments into 3-profile and 5-profile groups."""
    range_3 = {'global': None, 'uniform': None}
    range_5 = {'global': None, 'uniform': None}

    for exp in [e for e in experiments if e['category'] == 'dynamic']:
        n = len(set(exp['scenario'].values()))
        run_name = exp['run_name']
        rtype = 'global' if ('20260307' in run_name or '20260317' in run_name) else 'uniform'
        group = range_3 if n == 3 else range_5
        if group[rtype] is None:
            group[rtype] = exp

    return {'range_3': range_3, 'range_5': range_5}


def _group_hetero_by_reward(hetero_exps):
    """Return {scenario_str: {'global': exp, 'uniform': exp}} for complete pairs."""
    scenarios = {}
    for exp in hetero_exps:
        key = exp['scenario_str']
        scenarios.setdefault(key, {'global': None, 'uniform': None})
        run_name = exp['run_name']
        rtype = 'global' if ('20260307' in run_name or '20260317' in run_name) else 'uniform'
        scenarios[key][rtype] = exp
    return {k: v for k, v in scenarios.items()
            if v['global'] is not None and v['uniform'] is not None}


def _debug_dynamic(experiments):
    print("\n  Available dynamic experiments:")
    for exp in experiments:
        if exp['category'] == 'dynamic':
            n = len(set(exp['scenario'].values()))
            has_s = 'dti_beta_slice0' in exp['data']
            print(f"    - {exp['run_name']}: {n} profiles, per_slice={has_s}")
