"""
data_utils.py — Data loading, experiment categorization, and shared utilities.
"""

import os
import json
import numpy as np


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

    baseline_files = [
        f for f in os.listdir(baseline_dir)
        if f.startswith('baselines_') and f.endswith('.json')
    ]

    if not baseline_files:
        print(f"  No baseline files found in {baseline_dir}")
        return {}

    for filename in baseline_files:
        filepath = os.path.join(baseline_dir, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            run_name = data['metadata']['run_name']
            baseline_data[run_name] = {
                name: results['statistics']
                for name, results in data['baselines'].items()
            }
            print(f"  ✓ Loaded {filename}: {list(data['baselines'].keys())}")
        except Exception as e:
            print(f"  ⚠️  Failed to load {filename}: {e}")

    print(f"  ✓ Loaded baselines for {len(baseline_data)} experiments")
    return baseline_data


def categorize_experiments(experiments):
    """Organize experiments by category and print a debug summary."""
    categories = {
        'static_homogeneous': [],
        'static_heterogeneous': [],
        'dynamic': [],
    }

    for exp in experiments:
        categories[exp['category']].append(exp)

    print(f"\nExperiments by category:")
    print(f"  Homogeneous:  {len(categories['static_homogeneous'])}")
    print(f"  Heterogeneous:{len(categories['static_heterogeneous'])}")
    print(f"  Dynamic:      {len(categories['dynamic'])}")

    print("\n  Homogeneous experiments:")
    for exp in categories['static_homogeneous']:
        profile = list(exp['scenario'].values())[0]
        has_reward = 'episode_reward' in exp['data']
        has_beta = 'episode_beta' in exp['data']
        print(f"    - {profile}: reward={has_reward}, beta={has_beta}")

    return categories


def compute_per_slice_stats(data_dict, metric_prefix='dti_beta_slice'):
    """
    Compute statistics for per-slice metrics.

    Args:
        data_dict: Experiment data dictionary.
        metric_prefix: Prefix for the metric (e.g., 'dti_beta_slice').

    Returns:
        dict: {slice_id: {mean, std, last_100_mean, last_100_std, ...}}
    """
    stats = {}
    slice_keys = [k for k in data_dict if k.startswith(metric_prefix)]

    for key in slice_keys:
        slice_num = key.replace(metric_prefix, '')
        values = np.array(data_dict[key]['values'])
        if len(values) == 0:
            continue

        slice_stats = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'count': len(values),
        }

        window = min(100, int(len(values) * 0.2))
        if len(values) >= window:
            last_n = values[-window:]
            slice_stats['last_100_mean'] = float(np.mean(last_n))
            slice_stats['last_100_std'] = float(np.std(last_n))
            slice_stats['last_100_min'] = float(np.min(last_n))
            slice_stats['last_100_max'] = float(np.max(last_n))

        stats[slice_num] = slice_stats

    return stats


def compute_jains_fairness_index(beta0, beta1):
    """
    Compute Jain's Fairness Index for two slices.

    Uses QoS satisfaction (1 - β) so that higher satisfaction → higher JFI.

    Returns:
        float: JFI in [0, 1], where 1 = perfect fairness.
    """
    s0 = 1.0 - beta0
    s1 = 1.0 - beta1
    numerator = (s0 + s1) ** 2
    denominator = 2 * (s0 ** 2 + s1 ** 2)
    return numerator / denominator if denominator > 0 else 0.0


def get_per_slice_final_stats(exp_data, cutoff_frac=0.8):
    """
    Extract mean/std for slice 0 and slice 1 from the final portion of training.

    Returns:
        tuple: (beta0_mean, beta0_std, beta1_mean, beta1_std) or (None,)*4
    """
    if 'dti_beta_slice0' not in exp_data or 'dti_beta_slice1' not in exp_data:
        return None, None, None, None

    beta0_vals = np.array(exp_data['dti_beta_slice0']['values'])
    beta1_vals = np.array(exp_data['dti_beta_slice1']['values'])
    cutoff = int(len(beta0_vals) * cutoff_frac)

    b0 = beta0_vals[cutoff:]
    b1 = beta1_vals[cutoff:]
    return float(np.mean(b0)), float(np.std(b0)), float(np.mean(b1)), float(np.std(b1))


def collect_baseline_vectors(exp, baseline_data, baseline_names):
    """
    For a given experiment, return dicts of beta values/stds per baseline.
    Missing entries are filled with None.
    """
    betas = {name: None for name in baseline_names}
    stds = {name: None for name in baseline_names}
    run_name = exp['run_name']

    if run_name in baseline_data:
        for name in baseline_names:
            if name in baseline_data[run_name]:
                b_stats = baseline_data[run_name][name]['beta']
                betas[name] = b_stats.get('last_100_mean', b_stats.get('mean'))
                stds[name] = b_stats.get('last_100_std', b_stats.get('std'))

    return betas, stds
