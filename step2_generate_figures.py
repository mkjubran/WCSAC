"""
step2_generate_figures.py — Entry point for figure and table generation.

Usage:
    python3 step2_generate_figures.py \\
        --data extracted_data.json \\
        --output-dir ./paper_figures

The reward_formulation field is read directly from the JSON produced by step1
(experiments[i]['reward_formulation']).  No --ablation-map file is needed.

Optional overrides:
    --ablation-map  ablation_map.json
        Explicit JSON mapping run_name -> reward-type label.  Overrides the
        values read from the data file.  Use only if step1 data is missing the
        reward_formulation field (e.g. data extracted with an older step1).

    --baseline-dir  ./baseline_results
        Directory with baseline JSON files.

    --include-baselines
        Add baseline bars to figures 2 & 3 and comparison tables.

    --debug
        Print experiment details and exit.
"""

import os
import json
import argparse

import plot_style  # noqa: F401  (side-effect: applies rcParams)

from data_utils import load_extracted_data, load_baseline_results, categorize_experiments
from figures_dynamic import _build_ablation_map_from_data
from figures_static import (
    fig1_training_convergence,
    fig2_static_homogeneous,
    fig3_heterogeneous,
    fig4_allocation_patterns,
)
from figures_dynamic import (
    fig5_dynamic_scenarios,
    fig_actor_loss_comparison,
    fig_per_slice_beta_training,
    fig_fairness_evolution,
    fig_episode_per_slice_beta,
    fig_slice_beta_boxplot,
    fig_ablation_reward_formulation_robust,
    fig_dynamic_ablation_comparison,
    fig_dynamic_ablation_bar_chart,
)
from tables import (
    generate_latex_table_static_homogeneous,
    generate_latex_table_heterogeneous,
    generate_summary_table,
    generate_per_slice_beta_table,
    generate_ablation_table,
    generate_dynamic_ablation_table,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate figures and tables from extracted data')
    parser.add_argument('--data', default='extracted_data.json',
                        help='JSON file produced by step1')
    parser.add_argument('--output-dir', default='./paper_figures',
                        help='Output directory')
    parser.add_argument('--baseline-dir', default='./baseline_results',
                        help='Baseline JSON files directory')
    parser.add_argument('--include-baselines', action='store_true',
                        help='Include baseline comparisons')
    parser.add_argument('--ablation-map', default=None, dest='ablation_map',
                        help='Optional JSON override: run_name -> reward-type label')
    parser.add_argument('--debug', action='store_true',
                        help='Print experiment details and exit')
    return parser.parse_args()


def _load_ablation_map_override(path):
    if not path:
        return {}
    if not os.path.exists(path):
        print(f"  Warning: --ablation-map file not found: {path}")
        return {}
    with open(path, 'r') as f:
        mapping = json.load(f)
    print(f"  Loaded ablation-map override: {len(mapping)} entries from {path}")
    return mapping


def _debug_print(experiments, baseline_data, ablation_map):
    for i, exp in enumerate(experiments):
        print(f"\n[{i+1}] {exp['run_name']}")
        print(f"  category:             {exp['category']}")
        print(f"  scenario_str:         {exp['scenario_str']}")
        print(f"  scenario:             {exp['scenario']}")
        print(f"  slice_labels:         {exp.get('slice_labels')}")
        print(f"  qos_metrics:          {exp.get('qos_metrics')}")
        print(f"  thresholds:           {exp.get('thresholds')}")
        print(f"  reward_formulation:   {exp.get('reward_formulation')}")
        print(f"  slice_weights:        {exp.get('slice_weights')}")
        print(f"  dynamic_profile_set:  {exp.get('dynamic_profile_set')}")
        print(f"  dynamic_change_period:{exp.get('dynamic_change_period')}")
        print(f"  ablation_map entry:   {ablation_map.get(exp['run_name'], 'not in map')}")
        print(f"  data keys:            {list(exp['data'].keys())}")
        if exp['run_name'] in baseline_data:
            print(f"  baselines:            {list(baseline_data[exp['run_name']].keys())}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("STEP 2: GENERATING FIGURES AND TABLES")
    print("=" * 80)

    data = load_extracted_data(args.data)
    experiments = data['experiments']

    baseline_data = {}
    if args.include_baselines:
        baseline_data = load_baseline_results(args.baseline_dir, experiments)

    # Build ablation map: start from data, then apply any explicit override
    ablation_map = _build_ablation_map_from_data(experiments)
    override = _load_ablation_map_override(args.ablation_map)
    ablation_map.update(override)

    recognised = sum(1 for e in experiments
                     if e['run_name'] in ablation_map)
    print(f"\n  Ablation map: {recognised}/{len(experiments)} experiments "
          f"have a reward_formulation label")
    if recognised == 0:
        print("  Ablation figures and tables will be skipped.")

    if args.debug:
        print("\nDEBUG MODE:")
        _debug_print(experiments, baseline_data, ablation_map)
        return

    exps_by_cat = categorize_experiments(experiments)

    # ---- Static figures ----
    print("\n" + "=" * 80)
    print("STATIC FIGURES")
    print("=" * 80)
    fig1_training_convergence(experiments, args.output_dir)
    fig2_static_homogeneous(exps_by_cat, baseline_data, args.output_dir)
    fig3_heterogeneous(exps_by_cat, baseline_data, args.output_dir)
    fig4_allocation_patterns(experiments, args.output_dir)

    # ---- Dynamic figures ----
    print("\n" + "=" * 80)
    print("DYNAMIC FIGURES")
    print("=" * 80)
    fig5_dynamic_scenarios(exps_by_cat, args.output_dir)
    fig_actor_loss_comparison(experiments, args.output_dir)

    # ---- Per-slice beta figures ----
    print("\n" + "=" * 80)
    print("PER-SLICE BETA FIGURES")
    print("=" * 80)
    fig_per_slice_beta_training(experiments, args.output_dir)
    fig_fairness_evolution(experiments, args.output_dir)
    fig_slice_beta_boxplot(experiments, args.output_dir)
    fig_episode_per_slice_beta(experiments, args.output_dir, episode=80)
    fig_episode_per_slice_beta(experiments, args.output_dir, episode=160)

    # ---- Ablation figures ----
    if ablation_map:
        print("\n" + "=" * 80)
        print("ABLATION FIGURES")
        print("=" * 80)
        fig_ablation_reward_formulation_robust(experiments, args.output_dir, ablation_map)
        fig_dynamic_ablation_comparison(experiments, args.output_dir, ablation_map)
        fig_dynamic_ablation_bar_chart(experiments, args.output_dir, ablation_map)

    # ---- Tables ----
    print("\n" + "=" * 80)
    print("TABLES")
    print("=" * 80)
    if baseline_data:
        generate_latex_table_static_homogeneous(exps_by_cat, baseline_data, args.output_dir)
        generate_latex_table_heterogeneous(exps_by_cat, baseline_data, args.output_dir)
    else:
        print("  No baseline data -- skipping comparison tables")
    generate_summary_table(experiments, args.output_dir)
    generate_per_slice_beta_table(experiments, args.output_dir)
    if ablation_map:
        generate_ablation_table(experiments, args.output_dir, ablation_map)
        generate_dynamic_ablation_table(experiments, args.output_dir, ablation_map)

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\nOutput: {args.output_dir}")


if __name__ == "__main__":
    main()
