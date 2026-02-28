"""
STEP 1: Extract Essential Data from TensorBoard Runs
Saves to JSON for fast figure generation later

Usage:
    python3 step1_extract_data.py --runs-dir ./runs --configs-dir ./configs --output extracted_data.json

Output: JSON file with all essential metrics for paper figures
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
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("ERROR: tensorboard not available. Install with: pip install tensorboard")
    sys.exit(1)


def parse_scenario_from_config(config_path):
    """Extract traffic scenario from config file."""
    try:
        with open(config_path, 'r') as f:
            content = f.read()
        
        scenario = {}
        pattern = r"^TRAFFIC_PROFILES\s*=\s*\[([^\]]+)\]"
        
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('#'):
                continue
            
            match = re.match(pattern, line)
            if match:
                profiles_str = match.group(1)
                profiles = re.findall(r"['\"]([^'\"]+)['\"]", profiles_str)
                
                for i, prof in enumerate(profiles):
                    scenario[f'slice_{i}'] = prof.lower().strip()
                
                break
        
        return scenario
    
    except Exception as e:
        print(f"    ⚠️  Error parsing {config_path}: {e}")
        return {}


def extract_essential_metrics(tb_data):
    """
    Extract only the metrics needed for paper figures.
    Returns dict with cleaned, numpy-serializable data.
    """
    essential = {}
    
    # Metrics we actually need for figures
    required_metrics = {
        'episode/reward': 'episode_reward',
        'episode/avg_beta': 'episode_beta',
        'dti/beta': 'dti_beta',
        'dti/action_slice0': 'dti_action_slice0',
        'dti/action_slice1': 'dti_action_slice1',
        'dti/active_profile_slice0': 'dti_active_profile_slice0',
        'dti/active_profile_slice1': 'dti_active_profile_slice1',
        'episode_80/action_slice0': 'ep80_action_slice0',
        'episode_80/action_slice1': 'ep80_action_slice1',
        'episode_80/active_profile_slice0': 'ep80_active_profile_slice0',
        'episode_80/active_profile_slice1': 'ep80_active_profile_slice1',
        'episode_160/action_slice0': 'ep160_action_slice0',
        'episode_160/action_slice1': 'ep160_action_slice1',
        'episode_160/active_profile_slice0': 'ep160_active_profile_slice0',
        'episode_160/active_profile_slice1': 'ep160_active_profile_slice1',
    }
    
    for tb_key, output_key in required_metrics.items():
        if tb_key in tb_data:
            # Convert to lists (JSON serializable)
            steps = [int(v[0]) for v in tb_data[tb_key]]
            values = [float(v[1]) for v in tb_data[tb_key]]
            
            essential[output_key] = {
                'steps': steps,
                'values': values,
                'count': len(values)
            }
    
    return essential


def compute_summary_statistics(data, metric_key, window=100):
    """Compute summary statistics for a metric."""
    if metric_key not in data:
        return None
    
    values = np.array(data[metric_key]['values'])
    
    stats = {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'median': float(np.median(values)),
        'count': len(values)
    }
    
    # Last N values
    if len(values) >= window:
        last_n = values[-window:]
        stats['last_100_mean'] = float(np.mean(last_n))
        stats['last_100_std'] = float(np.std(last_n))
        stats['last_100_min'] = float(np.min(last_n))
        stats['last_100_max'] = float(np.max(last_n))
    
    return stats


def categorize_scenario(scenario):
    """Categorize scenario as homogeneous/heterogeneous/dynamic."""
    profiles = list(scenario.values())
    
    is_dynamic = any('dynamic' in p for p in profiles)
    
    if is_dynamic:
        return 'dynamic'
    elif len(set(profiles)) == 1:
        return 'static_homogeneous'
    else:
        return 'static_heterogeneous'


def extract_all_experiments(runs_dir, configs_dir):
    """
    Extract data from all experiments.
    """
    print("="*80)
    print("STEP 1: EXTRACTING ESSENTIAL DATA FROM TENSORBOARD")
    print("="*80)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # First, read all config files
    print("\n[1/3] Reading config files...")
    config_files = list(Path(configs_dir).glob("*.py"))
    print(f"  Found {len(config_files)} config files")
    
    config_info = {}
    for config_path in config_files:
        scenario = parse_scenario_from_config(str(config_path))
        if scenario:
            run_name = config_path.name.replace('config_multi_metric_', '').replace('.py', '')
            config_info[run_name] = {
                'config_path': str(config_path),
                'scenario': scenario,
                'scenario_str': ' - '.join([v.title() for v in scenario.values()]),
                'category': categorize_scenario(scenario)
            }
    
    print(f"  Successfully parsed {len(config_info)} configs")
    
    # Second, process TensorBoard runs
    print("\n[2/3] Extracting TensorBoard data...")
    run_dirs = [d for d in Path(runs_dir).iterdir() if d.is_dir()]
    print(f"  Found {len(run_dirs)} run directories")
    
    experiments = []
    
    for i, run_dir in enumerate(run_dirs, 1):
        run_name = run_dir.name
        
        print(f"\n  [{i}/{len(run_dirs)}] {run_name}")
        
        # Check if we have config
        if run_name not in config_info:
            print(f"    ⚠️  No matching config - skipping")
            continue
        
        info = config_info[run_name]
        print(f"    Scenario: {info['scenario_str']} ({info['category']})")
        
        # Load TensorBoard data
        print(f"    Loading TensorBoard...", end='', flush=True)
        
        event_files = glob.glob(os.path.join(str(run_dir), 'events.out.tfevents.*'))
        if not event_files:
            print(f" ✗ No event files")
            continue
        
        try:
            ea = event_accumulator.EventAccumulator(str(run_dir))
            ea.Reload()
            
            # Get all tags
            all_tags = ea.Tags().get('scalars', [])
            
            # Load essential metrics only
            tb_data = {}
            for tag in all_tags:
                try:
                    events = ea.Scalars(tag)
                    tb_data[tag] = [(e.step, e.value) for e in events]
                except:
                    pass
            
            print(f" ✓ ({len(tb_data)} metrics)")
            
            # Extract only what we need
            print(f"    Extracting essential data...", end='', flush=True)
            essential_data = extract_essential_metrics(tb_data)
            print(f" ✓ ({len(essential_data)} series)")
            
            # Compute summary statistics
            print(f"    Computing statistics...", end='', flush=True)
            stats = {
                'beta': compute_summary_statistics(essential_data, 'episode_beta'),
                'reward': compute_summary_statistics(essential_data, 'episode_reward')
            }
            print(f" ✓")
            
            # Store experiment data
            experiments.append({
                'run_name': run_name,
                'scenario': info['scenario'],
                'scenario_str': info['scenario_str'],
                'category': info['category'],
                'config_path': info['config_path'],
                'data': essential_data,
                'statistics': stats
            })
            
        except Exception as e:
            print(f" ✗ Error: {e}")
            continue
    
    print(f"\n[3/3] Categorizing experiments...")
    categories = {'static_homogeneous': 0, 'static_heterogeneous': 0, 'dynamic': 0}
    for exp in experiments:
        categories[exp['category']] += 1
    
    print(f"  Homogeneous: {categories['static_homogeneous']}")
    print(f"  Heterogeneous: {categories['static_heterogeneous']}")
    print(f"  Dynamic: {categories['dynamic']}")
    
    return experiments


def main():
    parser = argparse.ArgumentParser(description='Extract essential data from TensorBoard')
    parser.add_argument('--runs-dir', type=str, default='./runs', help='TensorBoard runs directory')
    parser.add_argument('--configs-dir', type=str, default='./configs', help='Config files directory')
    parser.add_argument('--output', type=str, default='extracted_data.json', help='Output JSON file')
    
    args = parser.parse_args()
    
    # Extract data
    experiments = extract_all_experiments(args.runs_dir, args.configs_dir)
    
    if not experiments:
        print("\nERROR: No experiments extracted!")
        return
    
    # Prepare output
    output_data = {
        'metadata': {
            'extraction_date': datetime.now().isoformat(),
            'runs_dir': args.runs_dir,
            'configs_dir': args.configs_dir,
            'num_experiments': len(experiments)
        },
        'experiments': experiments
    }
    
    # Save to JSON
    print("\n" + "="*80)
    print("SAVING DATA")
    print("="*80)
    print(f"\nOutput file: {args.output}")
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Get file size
    file_size = os.path.getsize(args.output)
    print(f"File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
    
    print("\n" + "="*80)
    print("✓ EXTRACTION COMPLETE")
    print("="*80)
    print(f"\nExtracted {len(experiments)} experiments")
    print(f"Saved to: {args.output}")
    print(f"\nNext step:")
    print(f"  python3 step2_generate_figures.py --data {args.output} --output-dir ./paper_figures")


if __name__ == "__main__":
    main()
