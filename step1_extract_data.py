"""
STEP 1: Extract Essential Data from TensorBoard Runs
Saves to JSON for fast figure generation later.

Usage:
    python3 step1_extract_data.py \\
        --runs-dir ./runs \\
        --configs-dir ./configs \\
        --output extracted_data.json

Output: JSON file with all essential metrics AND rich metadata for labelling
figures and tables in step2 without any hardcoding or heuristics.
"""

import os
import sys
import argparse
import json
import glob
import numpy as np
from pathlib import Path
import re
from datetime import datetime

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("ERROR: tensorboard not available. Install with: pip install tensorboard")
    sys.exit(1)


# ============================================================================
# CONFIG PARSING
# ============================================================================

def _parse_python_list(text):
    """
    Parse a Python list literal from a string, handling multi-line lists and
    quoted strings, integers, and floats.  Returns a list or None on failure.
    """
    # Collect the full bracket contents (handles lists that span multiple lines)
    depth = 0
    buf = []
    recording = False
    for ch in text:
        if ch == '[':
            depth += 1
            recording = True
        if recording:
            buf.append(ch)
        if ch == ']':
            depth -= 1
            if depth == 0:
                break
    if not buf:
        return None
    raw = ''.join(buf)
    # Use ast.literal_eval for safety
    import ast
    try:
        return ast.literal_eval(raw)
    except Exception:
        return None


def _parse_scalar(text, cast):
    """Extract the RHS of 'VAR = VALUE', cast to type, ignore commented lines."""
    for line in text.split('\n'):
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        m = re.match(r'^[A-Z_]+\s*=\s*(.+)', stripped)
        if m:
            try:
                return cast(m.group(1).strip().split('#')[0].strip())
            except Exception:
                pass
    return None


def _extract_variable(content, var_name):
    """
    Extract the value of a module-level assignment 'VAR_NAME = ...'
    from a config file string.  Handles scalars, booleans, lists of lists,
    flat lists, and dicts.  Returns the Python object or None.
    """
    import ast

    # Find the line(s) that define this variable (skip comment lines)
    lines = content.split('\n')
    start_line = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        if re.match(rf'^{re.escape(var_name)}\s*=', stripped):
            start_line = i
            break

    if start_line is None:
        return None

    # Collect the full statement (may span multiple lines if it contains brackets)
    stmt_lines = []
    depth = 0
    for line in lines[start_line:]:
        stmt_lines.append(line)
        depth += line.count('[') + line.count('(') + line.count('{')
        depth -= line.count(']') + line.count(')') + line.count('}')
        if depth <= 0:
            break

    stmt = '\n'.join(stmt_lines)
    # Keep only the RHS
    rhs = stmt.split('=', 1)[1].strip()
    # Strip trailing comment from simple scalars
    if not any(c in rhs for c in ('[', '{', '(')):
        rhs = rhs.split('#')[0].strip()

    try:
        return ast.literal_eval(rhs)
    except Exception:
        return None


def parse_config(config_path):
    """
    Parse all metadata relevant to figure/table labelling from a config file.

    Returns a dict with keys:
        scenario          – {slice_0: profile, slice_1: profile, ...}
        scenario_str      – human-readable scenario label
        category          – 'static_homogeneous' | 'static_heterogeneous' | 'dynamic'
        slice_labels      – list of per-slice display labels, e.g. ['VoIP', 'CBR']
        qos_metrics       – list of lists, per-slice QoS metric names
        thresholds        – list of lists, per-slice thresholds
        qos_directions    – list of lists, 'lower'/'higher' per metric
        qos_table_files   – list of QoS table filenames
        use_multi_metric  – bool
        beta_thresh       – float, the QoS violation ratio target
        reward_formulation – 'global' | 'weighted'
        slice_weights     – list of floats (normalised), or None
        dynamic_profile_set    – list of profiles in the dynamic pool (or [])
        dynamic_change_period  – int, DTIs between profile changes (or None)
        K                 – int, number of slices
        C                 – int, total RB capacity
        N                 – int, TTIs per DTI
        T_max             – int, episode length in DTIs
        num_episodes      – int
        window_size       – int (W)
    """
    try:
        with open(config_path, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"    ⚠️  Cannot read {config_path}: {e}")
        return {}

    result = {}

    # ------------------------------------------------------------------
    # Topology scalars
    # ------------------------------------------------------------------
    result['K']           = _extract_variable(content, 'K')           or 2
    result['C']           = _extract_variable(content, 'C')           or 8
    result['N']           = _extract_variable(content, 'N')           or 8
    result['T_max']       = _extract_variable(content, 'T_MAX')       or 2000
    result['num_episodes']= _extract_variable(content, 'NUM_EPISODES') or 200
    result['window_size'] = _extract_variable(content, 'W')           or 5
    result['beta_thresh'] = _extract_variable(content, 'BETA_THRESH') or 0.2

    # ------------------------------------------------------------------
    # Traffic profiles and scenario
    # ------------------------------------------------------------------
    traffic_profiles = _extract_variable(content, 'TRAFFIC_PROFILES')
    if not traffic_profiles or not isinstance(traffic_profiles, list):
        print(f"    ⚠️  Could not parse TRAFFIC_PROFILES in {config_path}")
        return {}

    scenario = {f'slice_{i}': p.lower().strip()
                for i, p in enumerate(traffic_profiles)}
    result['scenario'] = scenario

    # Categorise
    profiles = list(scenario.values())
    is_dynamic = any('dynamic' in p for p in profiles)
    if is_dynamic:
        category = 'dynamic'
    elif len(set(profiles)) == 1:
        category = 'static_homogeneous'
    else:
        category = 'static_heterogeneous'
    result['category'] = category

    # ------------------------------------------------------------------
    # Dynamic profile pool and change period
    # ------------------------------------------------------------------
    dyn_cfg = _extract_variable(content, 'DYNAMIC_PROFILE_CONFIG')
    if isinstance(dyn_cfg, dict):
        result['dynamic_profile_set']   = dyn_cfg.get('profile_set', [])
        result['dynamic_change_period'] = dyn_cfg.get('change_period')
    else:
        result['dynamic_profile_set']   = []
        result['dynamic_change_period'] = None

    # ------------------------------------------------------------------
    # QoS configuration
    # ------------------------------------------------------------------
    qos_table_files  = _extract_variable(content, 'QOS_TABLE_FILES')  or []
    qos_metrics_multi= _extract_variable(content, 'QOS_METRICS_MULTI')
    thresholds_multi = _extract_variable(content, 'THRESHOLDS_MULTI')
    qos_directions   = _extract_variable(content, 'QOS_METRIC_DIRECTIONS')
    use_multi        = _extract_variable(content, 'USE_MULTI_METRIC_QOS')

    # Fall back to single-metric lists if multi not found
    if qos_metrics_multi is None:
        single = _extract_variable(content, 'QOS_METRICS')
        if isinstance(single, list):
            qos_metrics_multi = [[m] for m in single]

    if thresholds_multi is None:
        single_t = _extract_variable(content, 'THRESHOLDS')
        if isinstance(single_t, list):
            thresholds_multi = [[t] for t in single_t]

    if qos_directions is None:
        K = result['K']
        qos_directions = [['lower'] for _ in range(K)]

    result['qos_table_files']  = qos_table_files if isinstance(qos_table_files, list) else []
    result['qos_metrics']      = qos_metrics_multi  or []
    result['thresholds']       = thresholds_multi   or []
    result['qos_directions']   = qos_directions     or []
    result['use_multi_metric'] = bool(use_multi) if use_multi is not None else bool(qos_metrics_multi)

    # ------------------------------------------------------------------
    # Slice display labels
    # Derived from QOS_TABLE_FILES: 'qos_voip_all_metrics.json' → 'VoIP'
    # Fallback: 'Slice 0', 'Slice 1', ...
    # ------------------------------------------------------------------
    _FILE_TO_LABEL = {
        'voip': 'VoIP',
        'cbr':  'CBR',
        'video':'Video',
    }
    slice_labels = []
    for i, fname in enumerate(result['qos_table_files']):
        fname_lower = fname.lower()
        label = next((v for k, v in _FILE_TO_LABEL.items() if k in fname_lower),
                     f'Slice {i}')
        slice_labels.append(label)
    # Pad if fewer table files than slices
    while len(slice_labels) < result['K']:
        slice_labels.append(f'Slice {len(slice_labels)}')
    result['slice_labels'] = slice_labels

    # ------------------------------------------------------------------
    # Reward formulation
    # ------------------------------------------------------------------
    use_weighted = _extract_variable(content, 'USE_SLICE_WEIGHTED_REWARD')
    slice_weights = _extract_variable(content, 'SLICE_WEIGHTS')

    if use_weighted:
        result['reward_formulation'] = 'weighted'
        if isinstance(slice_weights, list) and sum(slice_weights) > 0:
            total = sum(slice_weights)
            result['slice_weights'] = [w / total for w in slice_weights]
        else:
            result['slice_weights'] = slice_weights
    else:
        result['reward_formulation'] = 'global'
        result['slice_weights'] = slice_weights  # still stored for reference

    # ------------------------------------------------------------------
    # Human-readable scenario string
    # ------------------------------------------------------------------
    if category == 'dynamic':
        pool = result['dynamic_profile_set']
        period = result['dynamic_change_period']
        pool_str = ', '.join(p.title() for p in pool) if pool else 'unknown'
        period_str = f', T={period} DTIs' if period else ''
        result['scenario_str'] = f'Dynamic [{pool_str}]{period_str}'
    else:
        result['scenario_str'] = ' - '.join(p.title() for p in profiles)

    return result


def categorize_scenario(scenario):
    """Categorize scenario dict as homogeneous/heterogeneous/dynamic."""
    profiles = list(scenario.values())
    is_dynamic = any('dynamic' in p for p in profiles)
    if is_dynamic:
        return 'dynamic'
    elif len(set(profiles)) == 1:
        return 'static_homogeneous'
    else:
        return 'static_heterogeneous'


# ============================================================================
# TENSORBOARD EXTRACTION
# ============================================================================

def _tb_series(tb_data, tb_key):
    """Return {'steps': [...], 'values': [...], 'count': N} or None."""
    if tb_key not in tb_data:
        return None
    steps  = [int(v[0])   for v in tb_data[tb_key]]
    values = [float(v[1]) for v in tb_data[tb_key]]
    return {'steps': steps, 'values': values, 'count': len(values)}


def extract_essential_metrics(tb_data, K):
    """
    Extract only the metrics needed for paper figures.
    Slice-indexed keys are generated dynamically from K so the function
    works for any number of slices without hardcoding.

    Returns dict with cleaned, JSON-serialisable data.
    """
    essential = {}

    # ---- Episode-level scalars ----
    for tb_key, out_key in [
        ('episode/reward',   'episode_reward'),
        ('episode/avg_beta', 'episode_beta'),
    ]:
        s = _tb_series(tb_data, tb_key)
        if s:
            essential[out_key] = s

    # ---- DTI-level time series (per-slice, generated from K) ----
    dti_keys = {'dti/beta': 'dti_beta'}
    for k in range(K):
        dti_keys[f'dti/beta_slice{k}']            = f'dti_beta_slice{k}'
        dti_keys[f'dti/action_slice{k}']          = f'dti_action_slice{k}'
        dti_keys[f'dti/active_profile_slice{k}']  = f'dti_active_profile_slice{k}'

    for tb_key, out_key in dti_keys.items():
        s = _tb_series(tb_data, tb_key)
        if s:
            essential[out_key] = s

    # ---- Per-episode snapshots (ep80, ep160) ----
    # These keys are written by the training code for specific episodes.
    # We scan all tags in tb_data for the pattern episode_{N}/... so we
    # capture whatever episodes were actually logged without hardcoding 80/160.
    ep_pattern = re.compile(r'^episode_(\d+)/(.*)')
    seen_ep_keys = set()
    for tag in tb_data:
        m = ep_pattern.match(tag)
        if not m:
            continue
        ep_num  = m.group(1)   # e.g. '80'
        ep_sub  = m.group(2)   # e.g. 'action_slice0'
        out_key = f'ep{ep_num}_{ep_sub}'
        if out_key not in seen_ep_keys:
            s = _tb_series(tb_data, tag)
            if s:
                essential[out_key] = s
                seen_ep_keys.add(out_key)

    # ---- Optional: actor loss (multiple possible tag names) ----
    for tb_key in ['train/actor_loss', 'episode/actor_loss',
                   'Losses/actor_loss', 'Training/actor_loss', 'actor_loss']:
        s = _tb_series(tb_data, tb_key)
        if s:
            essential['episode_actor_loss'] = s
            print(f"    ✓ Found episode_actor_loss as '{tb_key}'")
            break

    return essential


# ============================================================================
# STATISTICS
# ============================================================================

def compute_summary_statistics(data, metric_key, window=100):
    """Compute summary statistics for a metric series."""
    if metric_key not in data:
        return None

    values = np.array(data[metric_key]['values'])

    stats = {
        'mean':   float(np.mean(values)),
        'std':    float(np.std(values)),
        'min':    float(np.min(values)),
        'max':    float(np.max(values)),
        'median': float(np.median(values)),
        'count':  len(values),
    }

    if len(values) >= window:
        last_n = values[-window:]
        stats['last_100_mean'] = float(np.mean(last_n))
        stats['last_100_std']  = float(np.std(last_n))
        stats['last_100_min']  = float(np.min(last_n))
        stats['last_100_max']  = float(np.max(last_n))

    return stats


# ============================================================================
# MAIN EXTRACTION LOOP
# ============================================================================

def extract_all_experiments(runs_dir, configs_dir):
    print("=" * 80)
    print("STEP 1: EXTRACTING ESSENTIAL DATA FROM TENSORBOARD")
    print("=" * 80)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ------------------------------------------------------------------
    # 1. Parse all config files
    # ------------------------------------------------------------------
    print("\n[1/3] Reading config files...")
    config_files = list(Path(configs_dir).glob("*.py"))
    print(f"  Found {len(config_files)} config files")

    config_info = {}
    for config_path in config_files:
        info = parse_config(str(config_path))
        if not info:
            print(f"  ⚠️  Skipping {config_path.name} (parse failed)")
            continue
        run_name = (config_path.name
                    .replace('config_multi_metric_', '')
                    .replace('.py', ''))
        config_info[run_name] = info
        print(f"  ✓ {run_name}: {info['scenario_str']} "
              f"[{info['category']}] reward={info['reward_formulation']}")

    print(f"  Successfully parsed {len(config_info)} configs")

    # ------------------------------------------------------------------
    # 2. Process TensorBoard runs
    # ------------------------------------------------------------------
    print("\n[2/3] Extracting TensorBoard data...")
    run_dirs = [d for d in Path(runs_dir).iterdir() if d.is_dir()]
    print(f"  Found {len(run_dirs)} run directories")

    experiments = []

    for i, run_dir in enumerate(run_dirs, 1):
        run_name = run_dir.name
        print(f"\n  [{i}/{len(run_dirs)}] {run_name}")

        if run_name not in config_info:
            print(f"    ⚠️  No matching config — skipping")
            continue

        info = config_info[run_name]
        K    = info['K']

        print(f"    Scenario:  {info['scenario_str']} ({info['category']})")
        slice_summary = ', '.join(
            f"{info['slice_labels'][k]} ({info['scenario'][f'slice_{k}']})"
            for k in range(K)
        )
        qos_summary = ', '.join(str(info['qos_metrics'][k]) for k in range(K))
        print(f"    Slices:    {slice_summary}")
        print(f"    QoS:       {qos_summary}")
        print(f"    Reward:    {info['reward_formulation']}"
              + (f"  weights={info['slice_weights']}"
                 if info['reward_formulation'] == 'weighted' else ''))

        # Check event files
        event_files = glob.glob(
            os.path.join(str(run_dir), 'events.out.tfevents.*'))
        if not event_files:
            print(f"    ✗ No TensorBoard event files found")
            continue

        # Load TensorBoard
        print(f"    Loading TensorBoard...", end='', flush=True)
        try:
            ea = event_accumulator.EventAccumulator(str(run_dir))
            ea.Reload()
            all_tags = ea.Tags().get('scalars', [])
            tb_data  = {}
            for tag in all_tags:
                try:
                    events = ea.Scalars(tag)
                    tb_data[tag] = [(e.step, e.value) for e in events]
                except Exception:
                    pass
            print(f" ✓ ({len(tb_data)} metrics)")
        except Exception as e:
            print(f" ✗ Error loading TensorBoard: {e}")
            continue

        # Extract series
        print(f"    Extracting series...", end='', flush=True)
        essential_data = extract_essential_metrics(tb_data, K)
        print(f" ✓ ({len(essential_data)} series)")

        # Compute summary statistics
        print(f"    Computing statistics...", end='', flush=True)
        stats = {
            'beta':   compute_summary_statistics(essential_data, 'episode_beta'),
            'reward': compute_summary_statistics(essential_data, 'episode_reward'),
        }
        # Per-slice beta stats (from DTI-level series)
        for k in range(K):
            key = f'dti_beta_slice{k}'
            if key in essential_data:
                stats[f'beta_slice{k}'] = compute_summary_statistics(
                    essential_data, key)
        print(f" ✓")

        experiments.append({
            'run_name':    run_name,
            # Scenario & categorisation
            'scenario':    info['scenario'],
            'scenario_str':info['scenario_str'],
            'category':    info['category'],
            # Per-slice metadata for labelling
            'K':           K,
            'slice_labels':info['slice_labels'],
            'qos_metrics': info['qos_metrics'],
            'thresholds':  info['thresholds'],
            'qos_directions': info['qos_directions'],
            'qos_table_files': info['qos_table_files'],
            'use_multi_metric': info['use_multi_metric'],
            # Reward formulation — used by ablation figures/tables in step2
            # without any date heuristics
            'reward_formulation': info['reward_formulation'],
            'slice_weights':      info['slice_weights'],
            # Dynamic scenario details
            'dynamic_profile_set':   info['dynamic_profile_set'],
            'dynamic_change_period': info['dynamic_change_period'],
            # Environment parameters
            'C':            info['C'],
            'N':            info['N'],
            'T_max':        info['T_max'],
            'num_episodes': info['num_episodes'],
            'window_size':  info['window_size'],
            'beta_thresh':  info['beta_thresh'],
            # Config path for traceability
            'config_path':  info.get('config_path', ''),
            # TensorBoard data and statistics
            'data':        essential_data,
            'statistics':  stats,
        })

    # ------------------------------------------------------------------
    # 3. Summary
    # ------------------------------------------------------------------
    print(f"\n[3/3] Summary...")
    categories = {'static_homogeneous': 0, 'static_heterogeneous': 0, 'dynamic': 0}
    reward_types = {}
    for exp in experiments:
        categories[exp['category']] += 1
        rt = exp['reward_formulation']
        reward_types[rt] = reward_types.get(rt, 0) + 1

    print(f"  Homogeneous:   {categories['static_homogeneous']}")
    print(f"  Heterogeneous: {categories['static_heterogeneous']}")
    print(f"  Dynamic:       {categories['dynamic']}")
    print(f"  Reward types:  {reward_types}")

    return experiments


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract essential data from TensorBoard runs')
    parser.add_argument('--runs-dir',    default='./runs',
                        help='TensorBoard runs directory')
    parser.add_argument('--configs-dir', default='./configs',
                        help='Config files directory (one .py per run)')
    parser.add_argument('--output',      default='extracted_data.json',
                        help='Output JSON file')
    args = parser.parse_args()

    experiments = extract_all_experiments(args.runs_dir, args.configs_dir)

    if not experiments:
        print("\nERROR: No experiments extracted!")
        return

    output_data = {
        'metadata': {
            'extraction_date': datetime.now().isoformat(),
            'runs_dir':        args.runs_dir,
            'configs_dir':     args.configs_dir,
            'num_experiments': len(experiments),
        },
        'experiments': experiments,
    }

    print("\n" + "=" * 80)
    print("SAVING")
    print("=" * 80)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    size = os.path.getsize(args.output)
    print(f"  Saved {len(experiments)} experiments → {args.output}")
    print(f"  File size: {size:,} bytes ({size / 1024:.1f} KB)")

    print("\n" + "=" * 80)
    print("✓ EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"\nNext step:")
    print(f"  python3 step2_generate_figures.py --data {args.output} "
          f"--output-dir ./paper_figures")
    print()
    print("  For ablation figures/tables, the reward_formulation field is now")
    print("  stored per experiment in the JSON — no --ablation-map needed:")
    print("    experiments[i]['reward_formulation']  →  'global' | 'weighted'")
    print("    experiments[i]['slice_weights']        →  [0.5, 0.5] | None")


if __name__ == "__main__":
    main()
