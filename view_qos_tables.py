"""
QoS Table Viewer - Helps you choose appropriate thresholds

This script reads your converted QoS files and displays the mean QoS values
in a tabular format to help you select appropriate thresholds.

Usage:
    python view_qos_tables.py
"""

import json
import sys
from pathlib import Path


def print_qos_table(qos_file, metric_name=None):
    """
    Print QoS table in tabular format showing mean values.
    
    Args:
        qos_file: Path to QoS JSON file
        metric_name: Specific metric to display (None = first available)
    """
    
    # Load QoS file
    try:
        with open(qos_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {qos_file}")
        return
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON file: {qos_file}")
        return
    
    print(f"\n{'='*80}")
    print(f"QoS Table: {qos_file}")
    print(f"{'='*80}")
    
    # Get all traffic values (UEs) and RB values
    ue_values = sorted([int(k) for k in data.keys()])
    
    # Get RB values from first UE
    first_ue = str(ue_values[0])
    rb_values = sorted([int(k) for k in data[first_ue].keys()])
    
    # Check if multi-metric format
    first_rb = str(rb_values[0])
    first_entry = data[first_ue][first_rb]
    
    is_multi_metric = False
    available_metrics = []
    
    if isinstance(first_entry, dict):
        if 'mu' in first_entry:
            # Single metric format
            is_multi_metric = False
        else:
            # Multi-metric format
            is_multi_metric = True
            available_metrics = list(first_entry.keys())
    
    # Determine which metric to display
    if is_multi_metric:
        if metric_name is None:
            metric_name = available_metrics[0]
            print(f"📊 Available metrics: {available_metrics}")
            print(f"📌 Displaying: {metric_name} (first metric)")
        elif metric_name not in available_metrics:
            print(f"❌ Metric '{metric_name}' not found!")
            print(f"📊 Available metrics: {available_metrics}")
            return
        else:
            print(f"📊 Available metrics: {available_metrics}")
            print(f"📌 Displaying: {metric_name}")
    else:
        print(f"📌 Single-metric format")
    
    print()
    
    # Print header
    header = "UE\\RBs"
    for rb in rb_values:
        header += f" | RB={rb:2d}"
    print(header)
    print("-" * len(header))
    
    # Print rows
    for ue in ue_values:
        row = f"UE={ue:2d} "
        
        for rb in rb_values:
            ue_str = str(ue)
            rb_str = str(rb)
            
            if ue_str in data and rb_str in data[ue_str]:
                entry = data[ue_str][rb_str]
                
                if is_multi_metric:
                    if metric_name in entry:
                        value = entry[metric_name]['mu']
                    else:
                        value = None
                else:
                    value = entry['mu']
                
                if value is not None:
                    row += f" | {value:6.2f}"
                else:
                    row += f" |    N/A"
            else:
                row += f" |    N/A"
        
        print(row)
    
    # Print statistics
    print("\n" + "="*80)
    print("📊 STATISTICS (for threshold selection):")
    print("="*80)
    
    # Collect all values
    all_values = []
    for ue in ue_values:
        for rb in rb_values:
            ue_str = str(ue)
            rb_str = str(rb)
            
            if ue_str in data and rb_str in data[ue_str]:
                entry = data[ue_str][rb_str]
                
                if is_multi_metric:
                    if metric_name in entry:
                        all_values.append(entry[metric_name]['mu'])
                else:
                    all_values.append(entry['mu'])
    
    if all_values:
        import numpy as np
        
        print(f"  Min value:        {np.min(all_values):.4f}")
        print(f"  Max value:        {np.max(all_values):.4f}")
        print(f"  Mean value:       {np.mean(all_values):.4f}")
        print(f"  Median value:     {np.median(all_values):.4f}")
        print(f"  25th percentile:  {np.percentile(all_values, 25):.4f}")
        print(f"  75th percentile:  {np.percentile(all_values, 75):.4f}")
        print(f"  90th percentile:  {np.percentile(all_values, 90):.4f}")
        
        print(f"\n💡 THRESHOLD SUGGESTIONS:")
        print(f"  Conservative (low violations): {np.percentile(all_values, 75):.4f}")
        print(f"  Moderate (balanced):           {np.percentile(all_values, 50):.4f}")
        print(f"  Aggressive (high QoS):         {np.percentile(all_values, 25):.4f}")


def compare_metrics(qos_file):
    """
    Compare all metrics in a multi-metric QoS file.
    
    Args:
        qos_file: Path to QoS JSON file
    """
    
    try:
        with open(qos_file, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ Error loading {qos_file}: {e}")
        return
    
    # Check if multi-metric
    first_ue = str(sorted([int(k) for k in data.keys()])[0])
    first_rb = str(sorted([int(k) for k in data[first_ue].keys()])[0])
    first_entry = data[first_ue][first_rb]
    
    if not isinstance(first_entry, dict) or 'mu' in first_entry:
        print(f"ℹ️  {qos_file} is not a multi-metric file")
        return
    
    available_metrics = list(first_entry.keys())
    
    print(f"\n{'='*80}")
    print(f"METRIC COMPARISON: {qos_file}")
    print(f"{'='*80}")
    
    for metric in available_metrics:
        print(f"\n{'─'*80}")
        print_qos_table(qos_file, metric_name=metric)


def suggest_thresholds(voip_file, cbr_file, video_file, 
                       voip_metric='voIPFrameLoss',
                       cbr_metric='cbrFrameDelay', 
                       video_metric='rtVideoStreamingSegmentLoss'):
    """
    Suggest thresholds for all three slices based on their QoS files.
    
    Args:
        voip_file: VoIP QoS file
        cbr_file: CBR QoS file
        video_file: Video QoS file
        voip_metric: Metric to use for VoIP
        cbr_metric: Metric to use for CBR
        video_metric: Metric to use for Video
    """
    
    print("\n" + "="*80)
    print("THRESHOLD SUGGESTION FOR ALL SLICES")
    print("="*80)
    
    files_and_metrics = [
        (voip_file, voip_metric, "VoIP"),
        (cbr_file, cbr_metric, "CBR Video"),
        (video_file, video_metric, "Video Conferencing")
    ]
    
    suggestions = []
    
    for qos_file, metric_name, slice_name in files_and_metrics:
        try:
            with open(qos_file, 'r') as f:
                data = json.load(f)
            
            # Collect all values
            all_values = []
            for ue_str, rb_dict in data.items():
                for rb_str, entry in rb_dict.items():
                    if isinstance(entry, dict):
                        if metric_name in entry:
                            all_values.append(entry[metric_name]['mu'])
                        elif 'mu' in entry:
                            all_values.append(entry['mu'])
            
            if all_values:
                import numpy as np
                
                median = np.median(all_values)
                p75 = np.percentile(all_values, 75)
                
                suggestions.append({
                    'name': slice_name,
                    'metric': metric_name,
                    'conservative': p75,
                    'moderate': median
                })
        except Exception as e:
            print(f"⚠️  Could not process {slice_name}: {e}")
    
    if suggestions:
        print("\n📋 SUGGESTED CONFIG.PY SETTINGS:")
        print("="*80)
        print("\n# Conservative (fewer violations, lower QoS)")
        print("THRESHOLDS = [")
        for s in suggestions:
            print(f"    {s['conservative']:.4f},  # {s['name']} ({s['metric']})")
        print("]")
        
        print("\n# Moderate (balanced)")
        print("THRESHOLDS = [")
        for s in suggestions:
            print(f"    {s['moderate']:.4f},  # {s['name']} ({s['metric']})")
        print("]")
        
        print("\nQOS_METRICS = [")
        for s in suggestions:
            print(f"    '{s['metric']}',  # {s['name']}")
        print("]")


def main():
    """Main function with user interaction"""
    
    print("="*80)
    print("QoS TABLE VIEWER - Threshold Selection Tool")
    print("="*80)
    
    # Check for converted files
    import os
    
    voip_exists = os.path.exists('qos_voip_all_metrics.json')
    cbr_exists = os.path.exists('qos_cbr_all_metrics.json')
    video_exists = os.path.exists('qos_video_all_metrics.json')
    
    if not (voip_exists and cbr_exists and video_exists):
        print("\n⚠️  Converted QoS files not found!")
        print("\nExpected files:")
        print(f"  {'✓' if voip_exists else '✗'} qos_voip_all_metrics.json")
        print(f"  {'✓' if cbr_exists else '✗'} qos_cbr_all_metrics.json")
        print(f"  {'✓' if video_exists else '✗'} qos_video_all_metrics.json")
        print("\nPlease run: python convert_qos_multimetric.py")
        return
    
    print("\n✓ Found all converted QoS files")
    
    # Display VoIP
    print("\n" + "█"*80)
    print("SLICE 1: VoIP")
    print("█"*80)
    print_qos_table('qos_voip_all_metrics.json', 'voIPFrameLoss')
    
    # Display CBR
    print("\n" + "█"*80)
    print("SLICE 2: CBR Video")
    print("█"*80)
    print_qos_table('qos_cbr_all_metrics.json', 'cbrFrameDelay')
    
    # Display Video
    print("\n" + "█"*80)
    print("SLICE 3: Video Conferencing")
    print("█"*80)
    print_qos_table('qos_video_all_metrics.json', 'rtVideoStreamingSegmentLoss')
    
    # Suggest thresholds
    suggest_thresholds(
        'qos_voip_all_metrics.json',
        'qos_cbr_all_metrics.json',
        'qos_video_all_metrics.json',
        voip_metric='voIPFrameLoss',
        cbr_metric='cbrFrameDelay',
        video_metric='rtVideoStreamingSegmentLoss'
    )
    
    print("\n" + "="*80)
    print("✅ Done! Use the suggested thresholds in config.py")
    print("="*80)


if __name__ == "__main__":
    
    import sys
    
    if len(sys.argv) > 1:
        # Custom usage: view specific file and metric
        qos_file = sys.argv[1]
        metric = sys.argv[2] if len(sys.argv) > 2 else None
        
        print_qos_table(qos_file, metric)
        
    else:
        # Default: view all three slices
        main()
