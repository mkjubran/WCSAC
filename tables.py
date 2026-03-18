"""
tables.py — LaTeX table generation.

    generate_latex_table_static_homogeneous
    generate_latex_table_heterogeneous
    generate_summary_table
    generate_per_slice_beta_table
    generate_ablation_table
    generate_dynamic_ablation_table
"""

import os
import numpy as np

from plot_style import BASELINE_NAMES, LOAD_ORDER
from data_utils import compute_per_slice_stats, collect_baseline_vectors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _beta_str(stats):
    """Format 'mean ± std' from a statistics dict, or return 'N/A'."""
    if not stats:
        return 'N/A'
    mean = stats.get('last_100_mean', stats.get('mean'))
    std = stats.get('last_100_std', stats.get('std'))
    if mean is None or std is None:
        return 'N/A'
    return rf"{mean:.4f} $\pm$ {std:.4f}"


def _save_tex(latex, output_dir, filename):
    path = os.path.join(output_dir, filename)
    with open(path, 'w') as f:
        f.write(latex)
    print(f"  ✓ {filename}")


def _jfi_str(beta0_mean, beta1_mean):
    s0, s1 = 1.0 - beta0_mean, 1.0 - beta1_mean
    denom = 2 * (s0 ** 2 + s1 ** 2)
    jfi = (s0 + s1) ** 2 / denom if denom > 0 else 0.0
    return f"{jfi:.4f}"


def _final_slice_stats(exp_data, cutoff_frac=0.8):
    """
    Returns (b0_mean, b0_std, b1_mean, b1_std) from final cutoff_frac of data,
    or (None,)*4 if per-slice keys are absent.
    """
    if 'dti_beta_slice0' not in exp_data or 'dti_beta_slice1' not in exp_data:
        return None, None, None, None
    b0 = np.array(exp_data['dti_beta_slice0']['values'])
    b1 = np.array(exp_data['dti_beta_slice1']['values'])
    cut = int(len(b0) * cutoff_frac)
    return (float(np.mean(b0[cut:])), float(np.std(b0[cut:])),
            float(np.mean(b1[cut:])), float(np.std(b1[cut:])))


# ---------------------------------------------------------------------------
# Static Homogeneous
# ---------------------------------------------------------------------------

def generate_latex_table_static_homogeneous(exps_by_cat, baseline_data, output_dir):
    """LaTeX table for static homogeneous results (SAC + baselines)."""
    print("\n[Table] Static Homogeneous Results...")

    exps = sorted(
        exps_by_cat['static_homogeneous'],
        key=lambda e: LOAD_ORDER.get(list(e['scenario'].values())[0], 999),
    )
    if not exps:
        print("  ⚠️  No homogeneous experiments")
        return

    rows = []
    for exp in exps:
        profile = list(exp['scenario'].values())[0].replace('_', ' ').title()
        sac_str = _beta_str(exp['statistics'].get('beta'))
        bl_betas, bl_stds = collect_baseline_vectors(exp, baseline_data, BASELINE_NAMES)

        bl_strs = {}
        for name in BASELINE_NAMES:
            bv, sv = bl_betas[name], bl_stds[name]
            if bv is not None and sv is not None:
                bl_strs[name] = rf"{bv:.4f} $\pm$ {sv:.4f}"
            else:
                bl_strs[name] = 'N/A'

        rows.append({
            'profile': profile,
            'sac': sac_str,
            **{name: bl_strs[name] for name in BASELINE_NAMES},
        })

    latex = (
        r"\begin{table}[!t]" + "\n"
        r"\centering" + "\n"
        r"\caption{QoS Performance Across Static Homogeneous Traffic Profiles (Last 100 Episodes)}" + "\n"
        r"\label{tab:static_homogeneous_comparison}" + "\n"
        r"\begin{tabular}{lccccc}" + "\n"
        r"\toprule" + "\n"
        r"\textbf{Profile} & \textbf{SAC} & \textbf{Equal} & \textbf{Proportional} & \textbf{Greedy} & \textbf{Random} \\" + "\n"
        r"\midrule" + "\n"
    )
    for r in rows:
        latex += (
            f"{r['profile']} & {r['sac']} & {r['equal']} & "
            f"{r['proportional']} & {r['greedy']} & {r['random']} \\\\\n"
        )
    latex += r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}" + "\n"

    _save_tex(latex, output_dir, 'table_static_homogeneous.tex')


# ---------------------------------------------------------------------------
# Heterogeneous
# ---------------------------------------------------------------------------

def generate_latex_table_heterogeneous(exps_by_cat, baseline_data, output_dir):
    """LaTeX table for heterogeneous results (SAC + baselines)."""
    print("\n[Table] Heterogeneous Results...")

    exps = exps_by_cat['static_heterogeneous']
    if not exps:
        print("  ⚠️  No heterogeneous experiments")
        return

    rows = []
    for exp in exps:
        sac_str = _beta_str(exp['statistics'].get('beta'))
        bl_betas, bl_stds = collect_baseline_vectors(exp, baseline_data, BASELINE_NAMES)

        bl_strs = {}
        for name in BASELINE_NAMES:
            bv, sv = bl_betas[name], bl_stds[name]
            bl_strs[name] = rf"{bv:.4f} $\pm$ {sv:.4f}" if (bv is not None and sv is not None) else 'N/A'

        rows.append({
            'scenario': exp['scenario_str'],
            'sac': sac_str,
            **{name: bl_strs[name] for name in BASELINE_NAMES},
        })

    latex = (
        r"\begin{table}[!t]" + "\n"
        r"\centering" + "\n"
        r"\caption{QoS Performance Under Heterogeneous Traffic Scenarios (Last 100 Episodes)}" + "\n"
        r"\label{tab:heterogeneous_comparison}" + "\n"
        r"\begin{tabular}{lccccc}" + "\n"
        r"\toprule" + "\n"
        r"\textbf{Scenario} & \textbf{SAC} & \textbf{Equal} & \textbf{Proportional} & \textbf{Greedy} & \textbf{Random} \\" + "\n"
        r"\midrule" + "\n"
    )
    for r in rows:
        latex += (
            f"{r['scenario']} & {r['sac']} & {r['equal']} & "
            f"{r['proportional']} & {r['greedy']} & {r['random']} \\\\\n"
        )
    latex += r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}" + "\n"

    _save_tex(latex, output_dir, 'table_heterogeneous.tex')


# ---------------------------------------------------------------------------
# Summary Table
# ---------------------------------------------------------------------------

def generate_summary_table(experiments, output_dir):
    """LaTeX summary table: all experiments, β, reward, and reward formulation."""
    print("\n[Table] Summary Table...")

    rows = []
    for exp in experiments:
        beta_stats   = exp['statistics'].get('beta', {})
        reward_stats = exp['statistics'].get('reward', {})

        bm = beta_stats.get('last_100_mean',   beta_stats.get('mean'))
        bs = beta_stats.get('last_100_std',    beta_stats.get('std'))
        rm = reward_stats.get('last_100_mean', reward_stats.get('mean'))
        rs = reward_stats.get('last_100_std',  reward_stats.get('std'))

        beta_str   = rf"{bm:.3f} $\pm$ {bs:.3f}" if (bm and bs) else 'N/A'
        reward_str = rf"{rm:.2f} $\pm$ {rs:.2f}" if (rm and rs) else 'N/A'
        rf_label   = exp.get('reward_formulation', 'N/A')

        rows.append((exp['scenario_str'], beta_str, reward_str, rf_label))

    latex = (
        r"\begin{table*}[!t]" + "\n"
        r"\caption{Performance Summary Across Traffic Scenarios (Last 100 Episodes)}" + "\n"
        r"\label{tab:performance_summary}" + "\n"
        r"\centering" + "\n"
        r"\begin{tabular}{@{}lccc@{}}" + "\n"
        r"\toprule" + "\n"
        r"\textbf{Scenario} & \textbf{QoS Violation (β)} & \textbf{Episode Reward} & \textbf{Reward} \\" + "\n"
        r"\midrule" + "\n"
    )
    for scenario, beta, reward, rf in rows:
        latex += f"{scenario} & {beta} & {reward} & {rf} \\\\\n"
    latex += r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table*}" + "\n"

    _save_tex(latex, output_dir, 'summary_table.tex')


# ---------------------------------------------------------------------------
# Per-Slice Beta Table
# ---------------------------------------------------------------------------

def generate_per_slice_beta_table(experiments, output_dir):
    """LaTeX table: per-slice β for the best dynamic experiment."""
    print("\n[Table] Per-Slice Beta Performance...")

    dynamic_exp = None
    for exp in experiments:
        if (exp['category'] == 'dynamic' and
                'dti_beta_slice0' in exp['data'] and
                'dti_beta_slice1' in exp['data']):
            dynamic_exp = exp
            print(f"  Using experiment: {exp['run_name']}")
            break

    if not dynamic_exp:
        print("  ⚠️  No dynamic experiment with per-slice beta data found")
        return

    data = dynamic_exp['data']
    slice_stats = compute_per_slice_stats(data, 'dti_beta_slice')

    if not slice_stats:
        print("  ⚠️  No per-slice beta data available")
        return

    rows = []
    global_stats = dynamic_exp['statistics'].get('beta', {})
    rows.append(('Global', _beta_str(global_stats)))

    # Use slice_labels from the experiment if available, else 'Slice N'
    raw_labels = dynamic_exp.get('slice_labels', [])
    slice_names = {str(i): raw_labels[i] if i < len(raw_labels) else f'Slice {i}'
                   for i in range(dynamic_exp.get('K', 2))}
    for sid in sorted(slice_stats):
        st = slice_stats[sid]
        mean = st.get('last_100_mean', st.get('mean'))
        std = st.get('last_100_std', st.get('std'))
        val = rf"{mean:.4f} $\pm$ {std:.4f}" if (mean is not None and std is not None) else 'N/A'
        rows.append((slice_names.get(sid, f'Slice {sid}'), val))

    if '0' in slice_stats and '1' in slice_stats:
        b0 = slice_stats['0'].get('last_100_mean')
        b1 = slice_stats['1'].get('last_100_mean')
        if b0 is not None and b1 is not None:
            fairness = max(b0, b1) / (min(b0, b1) + 1e-6)
            rows.append(('Fairness (max/min)', f"{fairness:.2f}"))

    latex = (
        r"\begin{table}[!t]" + "\n"
        r"\centering" + "\n"
        r"\caption{Per-Slice QoS Performance Under Dynamic Traffic (Final 20\% of Training)}" + "\n"
        r"\label{tab:per_slice_beta}" + "\n"
        r"\begin{tabular}{lc}" + "\n"
        r"\toprule" + "\n"
        r"\textbf{Metric} & \textbf{β (mean $\pm$ std)} \\" + "\n"
        r"\midrule" + "\n"
    )
    for metric, value in rows:
        latex += f"{metric} & {value} \\\\\n"
    latex += r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}" + "\n"

    _save_tex(latex, output_dir, 'table_per_slice_beta.tex')


# ---------------------------------------------------------------------------
# Ablation Table (Heterogeneous)
# ---------------------------------------------------------------------------

def generate_ablation_table(experiments, output_dir, ablation_map):
    """
    LaTeX table: reward formulation comparison for heterogeneous scenarios,
    showing Global β / β0 / β1 / JFI per reward type (multirow per scenario).

    ablation_map: dict mapping run_name → reward-type label (e.g. 'global'/'uniform').
    Scenarios are sorted alphabetically by scenario_str.
    """
    print("\n[Table] Ablation Study: Reward Formulation Comparison...")

    hetero_exps = [e for e in experiments if e['category'] == 'static_heterogeneous']
    if not hetero_exps:
        print("  ⚠️  No heterogeneous experiments found")
        return

    # Group by (scenario_str, reward_type) using ablation_map
    scenarios = {}
    unrecognised = []
    for exp in hetero_exps:
        rtype = ablation_map.get(exp['run_name'])
        if rtype is None:
            unrecognised.append(exp['run_name'])
            continue
        key = exp['scenario_str']
        scenarios.setdefault(key, {})[rtype] = exp

    if unrecognised:
        print(f"  ⚠️  run_names not in ablation_map (skipped): {unrecognised}")

    # Keep only scenarios with at least 2 reward types
    complete = {k: v for k, v in scenarios.items() if len(v) >= 2}

    if not complete:
        print("  ⚠️  No scenarios with paired reward formulations")
        for k, v in scenarios.items():
            print(f"    {k}: {list(v.keys())}")
        return

    print(f"  Found {len(complete)} complete scenarios")

    # Sort scenarios and collect all reward-type labels (preserving first-seen order)
    scenario_names = sorted(complete.keys())
    all_rtypes = []
    for pairs in complete.values():
        for rt in pairs:
            if rt not in all_rtypes:
                all_rtypes.append(rt)

    rows = []
    for scenario_key in scenario_names:
        for i, rtype in enumerate(all_rtypes):
            exp = complete[scenario_key].get(rtype)
            if exp is None:
                g_str = b0_str = b1_str = jfi = '---'
            else:
                g_str = _beta_str(exp['statistics'].get('beta'))
                b0m, b0s, b1m, b1s = _final_slice_stats(exp['data'])
                if b0m is not None:
                    b0_str = rf"{b0m:.4f} $\pm$ {b0s:.4f}"
                    b1_str = rf"{b1m:.4f} $\pm$ {b1s:.4f}"
                    jfi = _jfi_str(b0m, b1m)
                else:
                    b0_str = b1_str = jfi = '---'

            rows.append({
                'scenario': scenario_key if i == 0 else '',
                'n_rtypes': len(all_rtypes),
                'reward': rtype,
                'global_beta': g_str,
                'beta0': b0_str,
                'beta1': b1_str,
                'jfi': jfi,
                'is_first': i == 0,
                'is_last_in_group': i == len(all_rtypes) - 1,
            })

    latex = (
        r"\begin{table}[!t]" + "\n"
        r"\caption{Ablation Results: Reward Formulation Comparison"
        r" (Mean $\pm$ Std, Last 100 Episodes)}" + "\n"
        r"\label{tab:ablation_results}" + "\n"
        r"\centering" + "\n"
        r"\begin{tabular}{@{}llcccc@{}}" + "\n"
        r"\toprule" + "\n"
        r"\textbf{Scenario} & \textbf{Reward} & \textbf{Global $\beta$}"
        r" & \textbf{$\beta_0$} & \textbf{$\beta_1$} & \textbf{JFI} \\" + "\n"
        r"\midrule" + "\n"
    )

    for i, row in enumerate(rows):
        if row['is_first']:
            latex += f"\\multirow{{{row['n_rtypes']}}}{{*}}{{{row['scenario']}}}\n"
        latex += (
            f"  & {row['reward']}  & {row['global_beta']} & {row['beta0']} "
            f"& {row['beta1']} & {row['jfi']} \\\\\n"
        )
        if row['is_last_in_group'] and i < len(rows) - 1:
            latex += r"\midrule" + "\n"

    latex += r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}" + "\n"
    _save_tex(latex, output_dir, 'table_ablation_results.tex')
    print(f"    Scenarios: {len(complete)}, reward types: {all_rtypes}")


# ---------------------------------------------------------------------------
# Dynamic Ablation Table
# ---------------------------------------------------------------------------

def generate_dynamic_ablation_table(experiments, output_dir, ablation_map):
    """
    LaTeX table: dynamic scenario ablation by traffic range and reward type.

    ablation_map: dict mapping run_name → reward-type label.
    Experiments are grouped by number of unique traffic profiles (range size),
    then by reward type.  Range labels are derived from the actual scenario_str
    values in the data — no hardcoded profile names are used.
    """
    print("\n[Table] Dynamic Scenarios Ablation...")

    if not ablation_map:
        print("  ⚠️  No ablation_map provided — skipping dynamic ablation table")
        return

    dynamic_exps = [e for e in experiments if e['category'] == 'dynamic']
    if not dynamic_exps:
        print("  ⚠️  No dynamic experiments found")
        return

    # Group by (n_profiles, reward_type) using ablation_map
    range_groups = {}   # {n_profiles: {rtype: exp}}
    unrecognised = []

    for exp in dynamic_exps:
        rtype = ablation_map.get(exp['run_name'])
        if rtype is None:
            unrecognised.append(exp['run_name'])
            continue
        n = len(set(exp['scenario'].values()))
        range_groups.setdefault(n, {})
        if rtype not in range_groups[n]:
            range_groups[n][rtype] = exp

    if unrecognised:
        print(f"  ⚠️  run_names not in ablation_map (skipped): {unrecognised}")

    sorted_ranges = sorted(range_groups.keys())
    if len(sorted_ranges) < 2:
        print(f"  ⚠️  Need at least 2 distinct traffic ranges, found: {sorted_ranges}")
        return

    # Collect all reward-type labels (preserving order)
    all_rtypes = []
    for n in sorted_ranges:
        for rt in range_groups[n]:
            if rt not in all_rtypes:
                all_rtypes.append(rt)

    def _row(exp, range_label, rtype, is_first, is_last_in_group, n_rtypes):
        stats = exp['statistics'].get('beta', {})
        g_str = _beta_str(stats)
        b0m, b0s, b1m, b1s = _final_slice_stats(exp['data'])
        if b0m is not None:
            b0_str = rf"{b0m:.4f} $\pm$ {b0s:.4f}"
            b1_str = rf"{b1m:.4f} $\pm$ {b1s:.4f}"
            jfi = _jfi_str(b0m, b1m)
        else:
            b0_str = b1_str = jfi = '---'
        return {
            'range': range_label if is_first else '',
            'n_rtypes': n_rtypes,
            'reward': rtype,
            'global_beta': g_str,
            'beta0': b0_str,
            'beta1': b1_str,
            'jfi': jfi,
            'is_first': is_first,
            'is_last_in_group': is_last_in_group,
        }

    rows = []
    for n in sorted_ranges:
        # Use scenario_str of the first available experiment as the range label
        sample_exp = next(iter(range_groups[n].values()))
        range_label = sample_exp['scenario_str']
        n_rtypes = len(all_rtypes)

        for i, rtype in enumerate(all_rtypes):
            exp = range_groups[n].get(rtype)
            if exp is None:
                # Reward type not present for this range
                rows.append({
                    'range': range_label if i == 0 else '',
                    'n_rtypes': n_rtypes,
                    'reward': rtype,
                    'global_beta': '---',
                    'beta0': '---',
                    'beta1': '---',
                    'jfi': '---',
                    'is_first': i == 0,
                    'is_last_in_group': i == n_rtypes - 1,
                })
            else:
                rows.append(_row(exp, range_label, rtype,
                                 is_first=(i == 0),
                                 is_last_in_group=(i == n_rtypes - 1),
                                 n_rtypes=n_rtypes))

    latex = (
        r"\begin{table}[!t]" + "\n"
        r"\caption{Dynamic Traffic Ablation: Performance Across Traffic Ranges and Reward Formulations"
        r" (Mean $\pm$ Std, Last 100 Episodes)}" + "\n"
        r"\label{tab:dynamic_ablation}" + "\n"
        r"\centering" + "\n"
        r"\begin{tabular}{@{}llcccc@{}}" + "\n"
        r"\toprule" + "\n"
        r"\textbf{Traffic Range} & \textbf{Reward} & \textbf{Global $\beta$}"
        r" & \textbf{$\beta_0$} & \textbf{$\beta_1$} & \textbf{JFI} \\" + "\n"
        r"\midrule" + "\n"
    )

    for i, row in enumerate(rows):
        if row['is_first']:
            latex += f"\\multirow{{{row['n_rtypes']}}}{{*}}{{{row['range']}}}\n"
        latex += (
            f"  & {row['reward']}  & {row['global_beta']} & {row['beta0']} "
            f"& {row['beta1']} & {row['jfi']} \\\\\n"
        )
        if row['is_last_in_group'] and i < len(rows) - 1:
            latex += r"\midrule" + "\n"

    latex += r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}" + "\n"
    _save_tex(latex, output_dir, 'table_dynamic_ablation.tex')
