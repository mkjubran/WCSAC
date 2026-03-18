"""
Diagnostic: Check what per-slice data exists in extracted_data.json
"""

import json
import sys

def check_per_slice_data_availability(json_path='extracted_data.json'):
    """
    Check what per-slice data is available for each experiment.
    """
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    experiments = data['experiments']
    
    print("="*80)
    print("PER-SLICE DATA AVAILABILITY CHECK")
    print("="*80)
    
    hetero_exps = [e for e in experiments if e['category'] == 'static_heterogeneous']
    
    print(f"\nFound {len(hetero_exps)} heterogeneous experiments:\n")
    
    for exp in hetero_exps:
        print(f"Experiment: {exp['run_name']}")
        print(f"  Scenario: {exp['scenario_str']}")
        
        # Check TensorBoard-logged metrics
        has_dti_slice0 = 'dti_beta_slice0' in exp['data']
        has_dti_slice1 = 'dti_beta_slice1' in exp['data']
        
        print(f"  TensorBoard per-slice: slice0={has_dti_slice0}, slice1={has_dti_slice1}")
        
        # Check what raw data keys exist
        data_keys = list(exp['data'].keys())
        print(f"  Available data keys ({len(data_keys)}):")
        
        # Look for per-slice related keys
        slice_keys = [k for k in data_keys if 'slice' in k.lower()]
        if slice_keys:
            print(f"    Slice-related keys: {slice_keys}")
        else:
            print(f"    No slice-related keys found")
        
        # Look for traffic/violation keys
        traffic_keys = [k for k in data_keys if 'traffic' in k.lower() or 'violation' in k.lower()]
        if traffic_keys:
            print(f"    Traffic/violation keys: {traffic_keys}")
        
        # Check if we have raw per-slice data that could be used to compute β
        # Common key patterns:
        potential_keys = [
            'dti_traffic_slice0', 'dti_traffic_slice1',
            'dti_violated_traffic_slice0', 'dti_violated_traffic_slice1',
            'dti_violations_slice0', 'dti_violations_slice1'
        ]
        
        available_raw = [k for k in potential_keys if k in exp['data']]
        if available_raw:
            print(f"    ✓ Raw per-slice data available: {available_raw}")
            print(f"      → Can compute β₀, β₁ from this!")
        else:
            print(f"    ✗ No raw per-slice data to compute β from")
        
        print()
    
    print("="*80)
    print("SUMMARY")
    print("="*80)
    
    # Check if any experiment has raw data we can use
    can_compute = []
    cannot_compute = []
    
    for exp in hetero_exps:
        has_tb_slices = 'dti_beta_slice0' in exp['data'] and 'dti_beta_slice1' in exp['data']
        
        if has_tb_slices:
            can_compute.append(exp['run_name'])
        else:
            # Check for raw data
            has_raw = any(k in exp['data'] for k in [
                'dti_traffic_slice0', 'dti_violated_traffic_slice0',
                'dti_violations_slice0'
            ])
            
            if has_raw:
                can_compute.append(f"{exp['run_name']} (needs computation)")
            else:
                cannot_compute.append(exp['run_name'])
    
    print(f"\n✓ Can compute β₀, β₁: {len(can_compute)}")
    for name in can_compute:
        print(f"  - {name}")
    
    print(f"\n✗ Cannot compute β₀, β₁: {len(cannot_compute)}")
    for name in cannot_compute:
        print(f"  - {name}")
    
    if cannot_compute:
        print("\nFor experiments without per-slice data:")
        print("  Option 1: Re-run with per-slice logging enabled")
        print("  Option 2: Show '---' in table/figure (current behavior)")
        print("  Option 3: Use global β as approximation (not recommended)")


if __name__ == "__main__":
    json_path = sys.argv[1] if len(sys.argv) > 1 else 'extracted_data.json'
    check_per_slice_data_availability(json_path)
