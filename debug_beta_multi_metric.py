"""
Debug script for Multi-Metric QoS Beta Calculation

Supports both single-metric and multi-metric QoS configurations.
Run this to understand why beta values are what they are.

Usage:
    python3 debug_beta_multi_metric.py
"""

import json
import numpy as np
import config_multi_metric as config

print("="*80)
print("DEBUGGING MULTI-METRIC BETA CALCULATION")
print("="*80)

# Load configuration
cfg = config.get_config()

print("\nCONFIGURATION:")
print("-"*80)
print(f"K (slices):          {cfg['K']}")
print(f"QoS Mode:            {'MULTI-METRIC' if cfg.get('use_multi_metric_qos') else 'SINGLE-METRIC'}")
print(f"QoS files:           {cfg['qos_table_files']}")
print(f"Traffic profiles:    {cfg['traffic_profiles']}")

if cfg.get('use_multi_metric_qos'):
    print(f"\nMulti-Metric Configuration:")
    for k in range(cfg['K']):
        metrics = cfg['qos_metrics_multi'][k]
        thresholds = cfg['thresholds_multi'][k]
        directions = cfg['qos_metric_directions'][k]
        print(f"  Slice {k}:")
        print(f"    Metrics:     {metrics}")
        print(f"    Thresholds:  {thresholds}")
        print(f"    Directions:  {directions}")
else:
    print(f"Single-Metric Configuration:")
    print(f"  QoS metrics:     {cfg['qos_metrics']}")
    print(f"  Thresholds:      {cfg['thresholds']}")

print("="*80)

# Process each slice
for k in range(cfg['K']):
    print(f"\n{'█'*80}")
    print(f"SLICE {k} - {cfg['traffic_profiles'][k].upper()} TRAFFIC")
    print(f"{'█'*80}")
    
    qos_file = cfg['qos_table_files'][k]
    
    if cfg.get('use_multi_metric_qos'):
        metrics = cfg['qos_metrics_multi'][k]
        thresholds = cfg['thresholds_multi'][k]
        directions = cfg['qos_metric_directions'][k]
        n_metrics = len(metrics)
    else:
        metrics = [cfg['qos_metrics'][k]]
        thresholds = [cfg['thresholds'][k]]
        directions = ['lower']
        n_metrics = 1
    
    if qos_file is None:
        print("⚠ Using default QoS model (no file)")
        continue
    
    print(f"\nFile:              {qos_file}")
    print(f"Number of metrics: {n_metrics}")
    
    for i, (metric, threshold, direction) in enumerate(zip(metrics, thresholds, directions)):
        print(f"\n  Metric {i+1}: {metric}")
        print(f"    Threshold:  {threshold}")
        print(f"    Direction:  {direction} ({'≤' if direction == 'lower' else '≥'} threshold)")
    
    # Load QoS file
    try:
        with open(qos_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"\n❌ File not found: {qos_file}")
        continue
    except json.JSONDecodeError:
        print(f"\n❌ Invalid JSON: {qos_file}")
        continue
    
    # Display table for each metric
    for metric_idx, (metric, threshold, direction) in enumerate(zip(metrics, thresholds, directions)):
        print("\n" + "="*80)
        print(f"METRIC {metric_idx+1}: {metric}")
        print(f"Threshold: {threshold} (direction: {direction})")
        print("="*80)
        
        print("\nSample QoS values (typical allocations):")
        print("-"*80)
        print("UE  | RB=1  | RB=2  | RB=3  | RB=4  | RB=5  | RB=6  | RB=7  | RB=8")
        print("-"*80)
        
        for ue in [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80]:
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
                    
                    # Check violation based on direction
                    if direction == 'lower':
                        violated = value > threshold
                    else:  # 'higher'
                        violated = value < threshold
                    
                    marker = "!" if violated else " "
                    row += f" {value:5.1f}{marker}|"
                else:
                    row += "   N/A |"
            print(row)
        
        print("-"*80)
        if direction == 'lower':
            print("! = VIOLATED (value > threshold)")
        else:
            print("! = VIOLATED (value < threshold)")
        
        # Count violations for this metric
        print("\n" + "-"*80)
        print(f"VIOLATION ANALYSIS FOR: {metric}")
        print("-"*80)
        
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
                
                # Check based on direction
                if direction == 'lower':
                    satisfied = value <= threshold
                else:  # 'higher'
                    satisfied = value >= threshold
                
                if satisfied:
                    count_ok += 1
                else:
                    count_violated += 1
        
        if len(values) == 0:
            print(f"❌ No data found for metric: {metric}")
            continue
        
        total = count_ok + count_violated
        values = np.array(values)
        
        print(f"\nTotal QoS entries: {total}")
        if direction == 'lower':
            print(f"Satisfied (≤{threshold}): {count_ok} ({100*count_ok/total:.1f}%)")
            print(f"VIOLATED (>{threshold}): {count_violated} ({100*count_violated/total:.1f}%)")
        else:
            print(f"Satisfied (≥{threshold}): {count_ok} ({100*count_ok/total:.1f}%)")
            print(f"VIOLATED (<{threshold}): {count_violated} ({100*count_violated/total:.1f}%)")
        
        print(f"\nQoS value statistics:")
        print(f"  Min:    {np.min(values):.4f}")
        print(f"  25th:   {np.percentile(values, 25):.4f}")
        print(f"  Median: {np.median(values):.4f}")
        print(f"  75th:   {np.percentile(values, 75):.4f}")
        print(f"  Mean:   {np.mean(values):.4f}")
        print(f"  Max:    {np.max(values):.4f}")
        
        violation_pct = 100 * count_violated / total
        
        print("\n" + "-"*80)
        print("INTERPRETATION:")
        print("-"*80)
        
        if violation_pct > 80:
            print(f"⚠ {violation_pct:.0f}% of entries violate threshold!")
            print("⚠ Threshold is too strict - almost everything violates")
            if direction == 'lower':
                print(f"💡 Suggested threshold: {np.percentile(values, 75):.4f} (75th percentile)")
            else:
                print(f"💡 Suggested threshold: {np.percentile(values, 25):.4f} (25th percentile)")
            
        elif violation_pct > 50:
            print(f"✓ {violation_pct:.0f}% of entries violate threshold")
            print("✓ Good for training - agent will see violations")
            
        elif violation_pct > 20:
            print(f"✓ {violation_pct:.0f}% of entries violate threshold")
            print("✓ Balanced - agent can achieve good QoS")
            
        elif violation_pct > 5:
            print(f"⚠ Only {violation_pct:.0f}% of entries violate threshold")
            print("⚠ Threshold is lenient - easy for agent")
            
        else:
            print(f"⚠ Only {violation_pct:.0f}% of entries violate threshold")
            print("⚠ Threshold is very high - agent can easily satisfy QoS")
    
    # Multi-metric combined analysis
    if n_metrics > 1:
        print("\n" + "="*80)
        print("MULTI-METRIC COMBINED ANALYSIS")
        print("="*80)
        print("\n⚠ ALL metrics must be satisfied for QoS success!")
        
        # Simulate combined satisfaction
        print("\nSimulating combined metric satisfaction:")
        print("-"*80)
        
        combined_ok = 0
        combined_violated = 0
        
        for ue_str in data.keys():
            for rb_str in data[ue_str].keys():
                entry = data[ue_str][rb_str]
                
                # Check ALL metrics
                all_satisfied = True
                metric_values = []
                
                for metric, threshold, direction in zip(metrics, thresholds, directions):
                    if metric in entry:
                        value = entry[metric]['mu']
                    elif 'mu' in entry:
                        value = entry['mu']
                    else:
                        all_satisfied = False
                        break
                    
                    metric_values.append(value)
                    
                    # Check based on direction
                    if direction == 'lower':
                        satisfied = value <= threshold
                    else:  # 'higher'
                        satisfied = value >= threshold
                    
                    if not satisfied:
                        all_satisfied = False
                        break
                
                if len(metric_values) == n_metrics:
                    if all_satisfied:
                        combined_ok += 1
                    else:
                        combined_violated += 1
        
        total_combined = combined_ok + combined_violated
        if total_combined > 0:
            print(f"\nCombined satisfaction (ALL metrics must pass):")
            print(f"  Total entries:     {total_combined}")
            print(f"  ALL satisfied:     {combined_ok} ({100*combined_ok/total_combined:.1f}%)")
            print(f"  ANY violated:      {combined_violated} ({100*combined_violated/total_combined:.1f}%)")
            
            combined_violation_pct = 100 * combined_violated / total_combined
            
            print("\n" + "-"*80)
            print("MULTI-METRIC EXPECTED BEHAVIOR:")
            print("-"*80)
            
            if combined_violation_pct > 80:
                print(f"⚠ {combined_violation_pct:.0f}% violate when ALL metrics checked!")
                print("⚠ Multi-metric QoS is very strict")
                print("✓ Expected high β (> 0.5) during training")
                print("💡 Agent must allocate high RBs to satisfy all metrics")
                
            elif combined_violation_pct > 50:
                print(f"✓ {combined_violation_pct:.0f}% violate when ALL metrics checked")
                print("✓ Multi-metric makes QoS more challenging")
                print("✓ Expected β ≈ 0.3-0.6 during training")
                print("✓ Agent will learn to balance all metrics")
                
            elif combined_violation_pct > 20:
                print(f"✓ {combined_violation_pct:.0f}% violate when ALL metrics checked")
                print("✓ Balanced multi-metric QoS")
                print("✓ Expected β ≈ 0.1-0.3 with good allocation")
                
            else:
                print(f"⚠ Only {combined_violation_pct:.0f}% violate when ALL metrics checked")
                print("⚠ Multi-metric QoS is relatively easy to satisfy")
                print("✓ Expected β ≈ 0.0-0.1 with decent allocation")
    
    # Simulation with RB=4
    print("\n" + "="*80)
    print("SIMULATION (agent allocates RB=4 uniformly):")
    print("="*80)
    
    total_traffic_sim = 0
    violated_traffic_sim = 0
    
    print(f"\nAssuming agent allocates RB=4 to all traffic:")
    sample_ues = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    
    print(f"\n{'UE':<4} | ", end="")
    for metric in metrics:
        print(f"{metric[:20]:^20} | ", end="")
    print("Status")
    print("-" * (10 + 23 * n_metrics + 10))
    
    for ue in sample_ues:
        if str(ue) not in data or '4' not in data[str(ue)]:
            continue
            
        traffic = ue
        entry = data[str(ue)]['4']
        
        # Check all metrics
        all_satisfied = True
        metric_values = []
        
        for metric, threshold, direction in zip(metrics, thresholds, directions):
            if metric in entry:
                value = entry[metric]['mu']
            elif 'mu' in entry:
                value = entry['mu']
            else:
                continue
            
            metric_values.append(value)
            
            # Check based on direction
            if direction == 'lower':
                satisfied = value <= threshold
            else:  # 'higher'
                satisfied = value >= threshold
            
            if not satisfied:
                all_satisfied = False
        
        if len(metric_values) == n_metrics:
            total_traffic_sim += traffic
            if not all_satisfied:
                violated_traffic_sim += traffic
            
            # Print row
            print(f"{ue:<4} | ", end="")
            for i, value in enumerate(metric_values):
                direction = directions[i]
                threshold_val = thresholds[i]
                if direction == 'lower':
                    marker = '✗' if value > threshold_val else '✓'
                else:
                    marker = '✗' if value < threshold_val else '✓'
                print(f"{value:8.2f} {marker:^10} | ", end="")
            
            status = 'VIOLATED' if not all_satisfied else 'OK'
            print(status)
    
    if total_traffic_sim > 0:
        beta_sim = violated_traffic_sim / total_traffic_sim
        print(f"\nSimulated β = {beta_sim:.4f} ({100*beta_sim:.1f}%)")
        print("-" * 80)
        
        if beta_sim > 0.5:
            print("✓ With RB=4 allocation, you SHOULD see high β (>0.5)")
            print("✓ Multi-metric QoS is challenging - this is expected")
        elif beta_sim > 0.1:
            print(f"✓ With RB=4 allocation, you SHOULD see β ≈ {beta_sim:.2f}")
            print("✓ Balanced difficulty - agent can learn good policy")
        else:
            print("⚠ Even with modest RB=4 allocation, violations are rare")
            print("⚠ Agent will easily achieve β ≈ 0")
            if n_metrics > 1:
                print("💡 Consider tightening thresholds for more challenge")

print("\n" + "="*80)
print("SUMMARY & RECOMMENDATIONS")
print("="*80)

print("\n📊 Configuration Mode:")
if cfg.get('use_multi_metric_qos'):
    print("  ✓ MULTI-METRIC mode active")
    print("  ✓ ALL metrics must be satisfied for QoS success")
    print("  ✓ More realistic and challenging than single-metric")
else:
    print("  ✓ SINGLE-METRIC mode (backward compatible)")
    print("  💡 Consider switching to multi-metric for more realistic evaluation")

print("\n📝 Key Insights:")
print("  1. Multi-metric QoS is typically MORE STRICT than single-metric")
print("  2. Higher β values are EXPECTED with multi-metric")
print("  3. Agent needs more RBs to satisfy all metrics simultaneously")

print("\n🔧 To adjust configuration:")
print("  - Edit config_multi_metric.py")
print("  - Modify THRESHOLDS_MULTI for each metric")
print("  - Adjust QOS_METRIC_DIRECTIONS if needed")

print("\n✅ Debug complete!")
print("\nRe-run this script after any configuration changes to verify.")
print("="*80)
