"""
Debug script to check why beta = 0

Reads configuration from config.py
Run this to see what's happening with your QoS values
"""

import json
import numpy as np
import config

print("="*70)
print("DEBUGGING BETA CALCULATION")
print("="*70)

# Load configuration
cfg = config.get_config()

print("\nCONFIGURATION:")
print("-"*70)
print(f"K (slices):      {cfg['K']}")
print(f"Thresholds:      {cfg['thresholds']}")
print(f"QoS files:       {cfg['qos_table_files']}")
print(f"QoS metrics:     {cfg['qos_metrics']}")
print(f"Traffic profiles: {cfg['traffic_profiles']}")
print("="*70)

# Process each slice
for k in range(cfg['K']):
    print(f"\n{'█'*70}")
    print(f"SLICE {k+1}")
    print(f"{'█'*70}")
    
    qos_file = cfg['qos_table_files'][k]
    metric = cfg['qos_metrics'][k]
    threshold = cfg['thresholds'][k]
    
    if qos_file is None:
        print("⚠ Using default QoS model (no file)")
        continue
    
    print(f"File:      {qos_file}")
    print(f"Metric:    {metric}")
    print(f"Threshold: {threshold}")
    
    # Load QoS file
    try:
        with open(qos_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {qos_file}")
        continue
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON: {qos_file}")
        continue
    
    print("\n" + "-"*70)
    print("Sample QoS values (typical allocations):")
    print("-"*70)
    print("UE  | RB=1  | RB=2  | RB=3  | RB=4  | RB=5  | RB=6  | RB=7  | RB=8")
    print("-"*70)
    
    for ue in [5, 10, 15, 20, 25, 30, 40, 50, 60, 70]:
        if str(ue) not in data:
            continue
            
        row = f"{ue:2d}  |"
        for rb in range(1, min(9, cfg['C']+1)):
            if str(rb) in data[str(ue)]:
                entry = data[str(ue)][str(rb)]
                
                # Handle multi-metric format
                if metric in entry:
                    value = entry[metric]['mu']
                elif 'mu' in entry:
                    value = entry['mu']
                else:
                    row += "   N/A |"
                    continue
                
                marker = "!" if value > threshold else " "
                row += f" {value:5.1f}{marker}|"
            else:
                row += "   N/A |"
        print(row)
    
    print("-"*70)
    print("! = VIOLATED (value > threshold)")
    print()
    
    # Count violations across all entries
    print("="*70)
    print("VIOLATION ANALYSIS:")
    print("="*70)
    
    count_ok = 0
    count_violated = 0
    values = []
    
    for ue_str in data.keys():
        for rb_str in data[ue_str].keys():
            entry = data[ue_str][rb_str]
            
            # Handle multi-metric format
            if metric in entry:
                value = entry[metric]['mu']
            elif 'mu' in entry:
                value = entry['mu']
            else:
                continue
            
            values.append(value)
            if value <= threshold:
                count_ok += 1
            else:
                count_violated += 1
    
    if len(values) == 0:
        print(f"❌ No data found for metric: {metric}")
        continue
    
    total = count_ok + count_violated
    values = np.array(values)
    
    print(f"\nTotal QoS entries: {total}")
    print(f"Satisfied (≤{threshold}): {count_ok} ({100*count_ok/total:.1f}%)")
    print(f"VIOLATED (>{threshold}): {count_violated} ({100*count_violated/total:.1f}%)")
    
    print(f"\nQoS value statistics:")
    print(f"  Min:    {np.min(values):.4f}")
    print(f"  25th:   {np.percentile(values, 25):.4f}")
    print(f"  Median: {np.median(values):.4f}")
    print(f"  75th:   {np.percentile(values, 75):.4f}")
    print(f"  Mean:   {np.mean(values):.4f}")
    print(f"  Max:    {np.max(values):.4f}")
    
    print("\n" + "="*70)
    print("EXPECTED BEHAVIOR:")
    print("="*70)
    
    violation_pct = 100 * count_violated / total
    
    if violation_pct > 80:
        print(f"⚠ {violation_pct:.0f}% of entries violate threshold!")
        print("⚠ Threshold is too strict - almost everything violates")
        print(f"💡 Suggested threshold: {np.percentile(values, 75):.4f} (75th percentile)")
        print("   This would give ~25% violations")
        
    elif violation_pct > 50:
        print(f"✓ {violation_pct:.0f}% of entries violate threshold")
        print("✓ Good for training - agent will see violations")
        print("✓ Expected β > 0 during training")
        
    elif violation_pct > 20:
        print(f"✓ {violation_pct:.0f}% of entries violate threshold")
        print("✓ Balanced - agent can achieve good QoS")
        print("✓ Expected β ≈ 0.1-0.3 with good allocation")
        
    elif violation_pct > 5:
        print(f"⚠ Only {violation_pct:.0f}% of entries violate threshold")
        print("⚠ Threshold is lenient - easy for agent")
        print("✓ Expected β ≈ 0.0-0.1 with decent allocation")
        
    else:
        print(f"⚠ Only {violation_pct:.0f}% of entries violate threshold")
        print("⚠ Threshold is very high - agent can easily satisfy QoS")
        print("⚠ You may see β = 0 if agent allocates sufficient RBs")
        print(f"💡 Suggested threshold: {np.median(values):.4f} (median)")
        print("   This would give ~50% violations")
    
    print("\n" + "="*70)
    print("SIMULATION (if agent allocates RB=4 uniformly):")
    print("="*70)
    
    # Simulate what happens with RB=4 allocation
    total_traffic_sim = 0
    violated_traffic_sim = 0
    
    print(f"\nAssuming agent allocates RB=4 to all traffic:")
    sample_ues = [10, 15, 20, 25, 30, 35, 40]
    
    for ue in sample_ues:
        if str(ue) not in data or '4' not in data[str(ue)]:
            continue
            
        traffic = ue
        entry = data[str(ue)]['4']
        
        if metric in entry:
            value = entry[metric]['mu']
        elif 'mu' in entry:
            value = entry['mu']
        else:
            continue
        
        total_traffic_sim += traffic
        if value > threshold:
            violated_traffic_sim += traffic
        
        status = 'VIOLATED' if value > threshold else 'OK'
        print(f"  UE={ue:2d}, RB=4: {metric}={value:8.2f} → {status}")
    
    if total_traffic_sim > 0:
        beta_sim = violated_traffic_sim / total_traffic_sim
        print(f"\nSimulated β = {beta_sim:.4f} ({100*beta_sim:.1f}%)")
        
        if beta_sim > 0.5:
            print("✓ With RB=4 allocation, you SHOULD see high β (>0.5)")
        elif beta_sim > 0.1:
            print("✓ With RB=4 allocation, you SHOULD see β ≈ {:.2f}".format(beta_sim))
        else:
            print("⚠ Even with modest RB=4 allocation, violations are rare")
            print("⚠ Agent will easily achieve β ≈ 0")
    
    print("\n" + "="*70)
    print("WHY YOU MIGHT SEE β=0:")
    print("="*70)
    
    # Check high RB allocations
    high_rb_ok = 0
    high_rb_total = 0
    
    for ue_str in data.keys():
        for rb in [6, 7, 8]:
            rb_str = str(rb)
            if rb_str in data[ue_str]:
                entry = data[ue_str][rb_str]
                if metric in entry:
                    value = entry[metric]['mu']
                elif 'mu' in entry:
                    value = entry['mu']
                else:
                    continue
                
                high_rb_total += 1
                if value <= threshold:
                    high_rb_ok += 1
    
    if high_rb_total > 0:
        high_rb_pct = 100 * high_rb_ok / high_rb_total
        print(f"\nWith high RBs (6-8): {high_rb_pct:.0f}% satisfy threshold")
        
        if high_rb_pct > 80:
            print("✓ EXPLANATION: Agent is learning to allocate high RBs (6-8)")
            print("✓ With sufficient RBs, QoS is always satisfied → β=0")
            print("✓ This is actually GOOD performance!")
            print()
            print("If you want to see β > 0 (for research):")
            print(f"  - Lower threshold to {np.percentile(values, 25):.4f}")
            print("  - Or reduce capacity C in config.py")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print("\n✅ Debug complete!")
print("\nTo adjust thresholds, edit config.py:")
print("  THRESHOLDS = [...]")
print("\nThen re-run this script to verify.")
