"""
Check Capacity Violations in TensorBoard Logs

This script scans all TensorBoard event files and checks if the sum of rounded 
actions exceeds the capacity C at any DTI.

Usage:
    python3 check_capacity_violations.py --runs-dir ./runs
    python3 check_capacity_violations.py --runs-dir ./runs --verbose
    python3 check_capacity_violations.py --runs-dir ./runs --export violations.csv
"""

import argparse
import os
from pathlib import Path
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import json


def load_tensorboard_data(event_file):
    """Load data from a TensorBoard event file."""
    try:
        ea = EventAccumulator(str(event_file))
        ea.Reload()
        return ea
    except Exception as e:
        print(f"  ⚠️  Error loading {event_file.name}: {e}")
        return None


def extract_capacity_from_config(run_dir):
    """
    Try to find capacity C from config file.
    Looks for config_multi_metric_*.py in the run directory or parent.
    """
    # Try to find config in run directory or parent
    config_paths = list(run_dir.glob("config*.py"))
    if not config_paths:
        config_paths = list(run_dir.parent.glob("config*.py"))
    
    if not config_paths:
        return None
    
    # Read first config file found
    config_file = config_paths[0]
    try:
        with open(config_file, 'r') as f:
            content = f.read()
        
        # Look for C = value
        for line in content.split('\n'):
            if line.strip().startswith('C =') or line.strip().startswith('C='):
                # Extract number
                parts = line.split('=')
                if len(parts) >= 2:
                    value = parts[1].strip().split('#')[0].strip()
                    return int(value)
    except Exception as e:
        print(f"  ⚠️  Error reading config {config_file}: {e}")
    
    return None


def check_run_for_violations(run_dir, verbose=False, manual_capacity=None):
    """
    Check a single run for capacity violations.
    
    Returns:
        dict with violation statistics
    """
    print(f"\nChecking: {run_dir.name}")
    
    # Find event file
    event_files = list(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        print("  ⚠️  No TensorBoard event files found")
        return None
    
    event_file = event_files[0]
    
    # Load TensorBoard data
    ea = load_tensorboard_data(event_file)
    if ea is None:
        return None
    
    # Check for required tags
    available_tags = ea.Tags()['scalars']
    
    required_tags = ['dti/action_slice0', 'dti/action_slice1']
    missing_tags = [tag for tag in required_tags if tag not in available_tags]
    
    if missing_tags:
        print(f"  ⚠️  Missing required tags: {missing_tags}")
        print(f"      Available tags: {available_tags[:10]}...")
        return None
    
    print(f"  ✓ Found action tags: {required_tags}")
    
    # Extract capacity
    if manual_capacity:
        capacity = manual_capacity
        print(f"  ✓ Using manual capacity C = {capacity}")
    else:
        capacity = extract_capacity_from_config(run_dir)
        if capacity is None:
            print("  ⚠️  Could not determine capacity C from config")
            print("      Use --capacity argument to specify manually")
            return None
        print(f"  ✓ Capacity C = {capacity}")
    
    # Extract action data
    try:
        slice0_scalars = ea.Scalars('dti/action_slice0')
        slice1_scalars = ea.Scalars('dti/action_slice1')
        
        slice0_steps = [s.step for s in slice0_scalars]
        slice0_values = [s.value for s in slice0_scalars]
        
        slice1_steps = [s.step for s in slice1_scalars]
        slice1_values = [s.value for s in slice1_scalars]
        
    except Exception as e:
        print(f"  ⚠️  Error extracting action data: {e}")
        return None
    
    # Find common steps
    steps0 = set(slice0_steps)
    steps1 = set(slice1_steps)
    common_steps = sorted(list(steps0 & steps1))
    
    if not common_steps:
        print("  ⚠️  No common DTI steps found")
        return None
    
    print(f"  ✓ Checking {len(common_steps)} DTI steps")
    
    # Create lookup dictionaries for fast access
    slice0_dict = dict(zip(slice0_steps, slice0_values))
    slice1_dict = dict(zip(slice1_steps, slice1_values))
    
    # Check violations at each DTI
    violations = []
    max_violation = 0
    total_excess = 0
    
    for step in common_steps:
        action0 = slice0_dict[step]
        action1 = slice1_dict[step]
        
        # Round actions (as done in environment)
        rounded0 = round(action0)
        rounded1 = round(action1)
        
        # Check total
        total_allocated = rounded0 + rounded1
        
        if total_allocated > capacity:
            excess = total_allocated - capacity
            violations.append({
                'dti': step,
                'slice0': rounded0,
                'slice1': rounded1,
                'total': total_allocated,
                'capacity': capacity,
                'excess': excess
            })
            max_violation = max(max_violation, excess)
            total_excess += excess
            
            if verbose:
                print(f"    ⚠️  DTI {step}: [{rounded0}, {rounded1}] → {total_allocated} > {capacity} (excess: {excess})")
    
    # Summary
    violation_rate = len(violations) / len(common_steps) * 100 if common_steps else 0
    
    result = {
        'run_name': run_dir.name,
        'capacity': capacity,
        'num_slices': 2,
        'total_dtis': len(common_steps),
        'violations': len(violations),
        'violation_rate': violation_rate,
        'max_excess': max_violation,
        'total_excess': total_excess,
        'avg_excess': total_excess / len(violations) if violations else 0,
        'violation_details': violations[:100]  # Limit to first 100 for JSON export
    }
    
    if violations:
        print(f"  ❌ VIOLATIONS FOUND:")
        print(f"      Total DTIs checked: {len(common_steps)}")
        print(f"      DTIs with violations: {len(violations)} ({violation_rate:.2f}%)")
        print(f"      Max excess: {max_violation} RBs")
        print(f"      Avg excess: {result['avg_excess']:.2f} RBs")
        print(f"      Total excess: {total_excess} RBs")
        if len(violations) <= 10:
            print(f"      Violation DTIs: {[v['dti'] for v in violations[:10]]}")
    else:
        print(f"  ✓ NO VIOLATIONS FOUND")
        print(f"      All {len(common_steps)} DTIs respected capacity C={capacity}")
    
    return result


def export_to_csv(all_results, output_file):
    """Export results to CSV file."""
    import csv
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['Run', 'Capacity', 'Total DTIs', 'Violations', 
                        'Violation Rate (%)', 'Max Excess', 'Avg Excess', 'Total Excess'])
        
        # Data
        for result in all_results:
            if result:
                writer.writerow([
                    result['run_name'],
                    result['capacity'],
                    result['total_dtis'],
                    result['violations'],
                    f"{result['violation_rate']:.2f}",
                    result['max_excess'],
                    f"{result['avg_excess']:.2f}",
                    result['total_excess']
                ])
    
    print(f"\n✓ Results exported to: {output_file}")


def export_to_json(all_results, output_file):
    """Export detailed results to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"✓ Detailed results exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Check for capacity violations in TensorBoard logs')
    parser.add_argument('--runs-dir', type=str, default='./runs',
                       help='Directory containing run subdirectories')
    parser.add_argument('--verbose', action='store_true',
                       help='Print details of each violation')
    parser.add_argument('--export', type=str,
                       help='Export results to CSV file')
    parser.add_argument('--export-json', type=str,
                       help='Export detailed results to JSON file')
    parser.add_argument('--capacity', type=int,
                       help='Manually specify capacity C (if not found in config)')
    
    args = parser.parse_args()
    
    runs_dir = Path(args.runs_dir)
    
    if not runs_dir.exists():
        print(f"Error: Directory not found: {runs_dir}")
        return
    
    print("="*80)
    print("CAPACITY VIOLATION CHECKER")
    print("="*80)
    print(f"Scanning: {runs_dir}")
    
    # Find all run directories (those with event files)
    run_dirs = []
    for path in runs_dir.iterdir():
        if path.is_dir():
            event_files = list(path.glob("events.out.tfevents.*"))
            if event_files:
                run_dirs.append(path)
    
    if not run_dirs:
        print(f"\nNo run directories with TensorBoard files found in {runs_dir}")
        return
    
    print(f"Found {len(run_dirs)} runs to check")
    
    # Check each run
    all_results = []
    runs_with_violations = 0
    
    for run_dir in sorted(run_dirs):
        result = check_run_for_violations(run_dir, verbose=args.verbose, manual_capacity=args.capacity)
        if result:
            all_results.append(result)
            if result['violations'] > 0:
                runs_with_violations += 1
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total runs checked: {len(all_results)}")
    print(f"Runs with violations: {runs_with_violations}")
    print(f"Runs without violations: {len(all_results) - runs_with_violations}")
    
    if runs_with_violations > 0:
        print("\n❌ VIOLATIONS DETECTED:")
        for result in all_results:
            if result['violations'] > 0:
                print(f"  - {result['run_name']}: {result['violations']} violations "
                      f"({result['violation_rate']:.2f}%), max excess = {result['max_excess']}")
    else:
        print("\n✓ NO VIOLATIONS FOUND IN ANY RUN")
    
    # Export results
    if args.export:
        export_to_csv(all_results, args.export)
    
    if args.export_json:
        export_to_json(all_results, args.export_json)
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
