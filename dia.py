"""
DIAGNOSTIC: Check Ablation Study Data
Run this to see what data is available for the ablation figure
"""

import json
import sys

def diagnose_ablation_data(json_path='extracted_data.json'):
    """Check what data is available for ablation study."""
    
    print("="*80)
    print("ABLATION STUDY DATA DIAGNOSTIC")
    print("="*80)
    
    # Load data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    experiments = data['experiments']
    
    print(f"\nTotal experiments: {len(experiments)}")
    
    # Target scenarios
    target_scenarios = [
        'Low - High',
        'Extremely_Low - Extremely_High',
        'Medium - High'
    ]
    
    print("\n" + "="*80)
    print("HETEROGENEOUS EXPERIMENTS")
    print("="*80)
    
    hetero_exps = [e for e in experiments if e['category'] == 'static_heterogeneous']
    
    print(f"\nFound {len(hetero_exps)} heterogeneous experiments:")
    
    for i, exp in enumerate(hetero_exps, 1):
        print(f"\n[{i}] {exp['run_name']}")
        print(f"    Scenario: {exp['scenario_str']}")
        
        # Check per-slice data
        has_slice0 = 'dti_beta_slice0' in exp['data']
        has_slice1 = 'dti_beta_slice1' in exp['data']
        print(f"    Has per-slice data: slice0={has_slice0}, slice1={has_slice1}")
        
        # Check global beta stats
        beta_stats = exp['statistics'].get('beta')
        if beta_stats:
            global_mean = beta_stats.get('last_100_mean', beta_stats.get('mean'))
            global_std = beta_stats.get('last_100_std', beta_stats.get('std'))
            print(f"    Global β: {global_mean:.4f} ± {global_std:.4f}" if global_mean else "    Global β: N/A")
        else:
            print(f"    Global β: No statistics")
        
        # If has per-slice data, compute stats
        if has_slice0 and has_slice1:
            beta0_values = exp['data']['dti_beta_slice0']['values']
            beta1_values = exp['data']['dti_beta_slice1']['values']
            
            if len(beta0_values) > 0:
                import numpy as np
                cutoff = int(len(beta0_values) * 0.8)
                final_beta0 = beta0_values[cutoff:]
                final_beta1 = beta1_values[cutoff:]
                
                beta0_mean = np.mean(final_beta0)
                beta1_mean = np.mean(final_beta1)
                
                print(f"    β₀ (final 20%): {beta0_mean:.4f}")
                print(f"    β₁ (final 20%): {beta1_mean:.4f}")
                
                # JFI
                s0 = 1.0 - beta0_mean
                s1 = 1.0 - beta1_mean
                jfi = (s0 + s1)**2 / (2 * (s0**2 + s1**2))
                print(f"    JFI: {jfi:.4f}")
    
    print("\n" + "="*80)
    print("SCENARIO MATCHING CHECK")
    print("="*80)
    
    for target in target_scenarios:
        print(f"\nLooking for: '{target}'")
        
        # Normalize target
        target_norm = target.lower().replace('_', '').replace('-', '').replace(' ', '')
        print(f"  Normalized: '{target_norm}'")
        
        matches = []
        for exp in hetero_exps:
            scenario_norm = exp['scenario_str'].lower().replace('_', '').replace('-', '').replace(' ', '')
            
            if target_norm == scenario_norm:
                matches.append(exp)
                print(f"  ✓ Match: {exp['run_name']} ({exp['scenario_str']})")
        
        if not matches:
            print(f"  ✗ No matches found!")
            print(f"  Available scenarios:")
            for exp in hetero_exps:
                scenario_norm = exp['scenario_str'].lower().replace('_', '').replace('-', '').replace(' ', '')
                print(f"    - '{exp['scenario_str']}' (normalized: '{scenario_norm}')")
    
    print("\n" + "="*80)
    print("REWARD FORMULATION DETECTION")
    print("="*80)
    
    print("\nAssuming:")
    print("  - Experiments WITH per-slice data = Uniform Weighted")
    print("  - Experiments WITHOUT per-slice data = Global Reward")
    
    for target in target_scenarios:
        target_norm = target.lower().replace('_', '').replace('-', '').replace(' ', '')
        
        global_exp = None
        uniform_exp = None
        
        for exp in hetero_exps:
            scenario_norm = exp['scenario_str'].lower().replace('_', '').replace('-', '').replace(' ', '')
            
            if target_norm == scenario_norm:
                has_per_slice = ('dti_beta_slice0' in exp['data'] and 
                                'dti_beta_slice1' in exp['data'])
                
                if has_per_slice:
                    uniform_exp = exp
                else:
                    global_exp = exp
        
        print(f"\n{target}:")
        if global_exp:
            print(f"  ✓ Global Reward: {global_exp['run_name']}")
        else:
            print(f"  ✗ Global Reward: NOT FOUND")
        
        if uniform_exp:
            print(f"  ✓ Uniform Reward: {uniform_exp['run_name']}")
        else:
            print(f"  ✗ Uniform Reward: NOT FOUND")


if __name__ == "__main__":
    import sys
    json_path = sys.argv[1] if len(sys.argv) > 1 else 'extracted_data.json'
    diagnose_ablation_data(json_path)
