"""
step2_generate_figures.py — Entry point for figure and table generation.

Usage:
    python3 step2_generate_figures.py --data extracted_data.json \\
                                      --output-dir ./paper_figures

Optional:
    --baseline-dir  ./baseline_results   Directory with baseline JSON files
    --include-baselines                  Add baseline bars to figures 2 & 3
    --debug                              Print experiment details and exit
"""

import os
import argparse

# Apply global style before any figure module imports pyplot
import plot_style  # noqa: F401  (side-effect: sets rcParams)

from data_utils import load_extracted_data, load_baseline_results, categorize_experiments
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
    parser = argparse.ArgumentParser(description='Generate figures from extracted data')
    parser.add_argument('--data', default='extracted_data.json',
                        help='Extracted data JSON file')
    parser.add_argument('--output-dir', default='./paper_figures',
                        help='Output directory for figures')
    parser.add_argument('--baseline-dir', default='./baseline_results',
                        help='Directory containing baseline JSON files')
    parser.add_argument('--include-baselines', action='store_true',
                        help='Include baseline comparisons in figures')
    parser.add_argument('--debug', action='store_true',
                        help='Print debug information and exit')
    return parser.parse_args()


def _debug_print(experiments, baseline_data):
    for i, exp in enumerate(experiments):
        print(f"\n[{i+1}] {exp['run_name']}")
        print(f"  Category: {exp['category']}")
        print(f"  Scenario: {exp['scenario']}")
        print(f"  Data keys: {list(exp['data'].keys())}")
        if exp['run_name'] in baseline_data:
            print(f"  Baselines: {list(baseline_data[exp['run_name']].keys())}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("STEP 2: GENERATING FIGURES")
    print("=" * 80)

    data = load_extracted_data(args.data)
    experiments = data['experiments']

    baseline_data = {}
    if args.include_baselines:
        baseline_data = load_baseline_results(args.baseline_dir, experiments)

    if args.debug:
        print("\nDEBUG MODE:")
        _debug_print(experiments, baseline_data)
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
    print("\n" + "=" * 80)
    print("ABLATION FIGURES")
    print("=" * 80)
    fig_ablation_reward_formulation_robust(experiments, args.output_dir)
    fig_dynamic_ablation_comparison(experiments, args.output_dir)
    fig_dynamic_ablation_bar_chart(experiments, args.output_dir)

    # ---- Tables ----
    print("\n" + "=" * 80)
    print("TABLES")
    print("=" * 80)
    if baseline_data:
        generate_latex_table_static_homogeneous(exps_by_cat, baseline_data, args.output_dir)
        generate_latex_table_heterogeneous(exps_by_cat, baseline_data, args.output_dir)
    else:
        print("  ⚠️  No baseline data — skipping comparison tables")
    generate_summary_table(experiments, args.output_dir)
    generate_per_slice_beta_table(experiments, args.output_dir)
    generate_ablation_table(experiments, args.output_dir)
    generate_dynamic_ablation_table(experiments, args.output_dir)

    print("\n" + "=" * 80)
    print("✓ COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: {args.output_dir}")
    print("\nFigures generated:")
    print("  Training:   fig1a–fig1d")
    print("  Static:     fig2, fig3, fig4")
    print("  Dynamic:    fig5a–fig5f3")
    print("  Per-slice:  fig_per_slice_beta_training, fig_fairness_evolution,")
    print("              fig_slice_beta_boxplot, fig5_ep80/ep160_per_slice_beta")
    print("  Ablation:   fig_ablation_reward_robust, fig_dynamic_ablation_*")
    print("  Actor loss: fig_actor_loss_comparison")
    print("\nTables generated:")
    print("  summary_table.tex, table_per_slice_beta.tex,")
    print("  table_ablation_results.tex, table_dynamic_ablation.tex")
    if baseline_data:
        print("  table_static_homogeneous.tex, table_heterogeneous.tex")


if __name__ == "__main__":
    main()
