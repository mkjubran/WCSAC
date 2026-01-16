"""
Convert existing QoS JSON files to the format expected by network_env.py
PRESERVES ALL METRICS from your JSON files

Your JSON format:
{
  "meta": {...},
  "data": {
    "rb_1": {
      "ue_5": {
        "metric1": {"mean": X, "std": Y},
        "metric2": {"mean": X, "std": Y},
        ...
      }
    }
  }
}

Expected format (multi-metric):
{
  "5": {
    "1": {
      "metric1": {"mu": X, "sigma": Y},
      "metric2": {"mu": X, "sigma": Y},
      ...
    }
  }
}
"""

import json
import sys
from pathlib import Path


def convert_qos_file_all_metrics(input_file, output_file):
    """
    Convert QoS file preserving ALL metrics.
    
    Args:
        input_file: Path to your JSON file
        output_file: Path to save converted JSON
    """
    # Load your JSON
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Extract meta info
    meta = data['meta']
    print(f"\nConverting {input_file}")
    print(f"  Metrics: {meta['metrics']}")
    print(f"  Units: {meta['units']}")
    
    # Convert to expected format (preserving all metrics)
    converted = {}
    
    for rb_key, ue_data in data['data'].items():
        # Extract RB number: "rb_1" -> "1"
        rb_num = rb_key.split('_')[1]
        
        for ue_key, metrics in ue_data.items():
            # Extract UE number: "ue_5" -> "5"
            ue_num = ue_key.split('_')[1]
            
            # Initialize structure
            if ue_num not in converted:
                converted[ue_num] = {}
            
            if rb_num not in converted[ue_num]:
                converted[ue_num][rb_num] = {}
            
            # Add ALL metrics
            for metric_name, stats in metrics.items():
                converted[ue_num][rb_num][metric_name] = {
                    'mu': stats['mean'],
                    'sigma': stats['std']
                }
    
    # Save converted data
    with open(output_file, 'w') as f:
        json.dump(converted, f, indent=2)
    
    print(f"  Saved to: {output_file}")
    print(f"  Traffic values: {sorted([int(k) for k in converted.keys()])}")
    print(f"  RB values: {sorted([int(k) for k in converted[list(converted.keys())[0]].keys()])}")
    print(f"  Preserved metrics: {list(converted[list(converted.keys())[0]][list(converted[list(converted.keys())[0]].keys()][0]].keys())}")
    
    return converted


def convert_all_preserve_metrics(
    voip_file='voip_qos.json',
    cbr_file='cbr_qos.json', 
    video_file='video_qos.json'
):
    """
    Convert all three files preserving ALL metrics.
    """
    print("=" * 70)
    print("Converting QoS Files (ALL METRICS PRESERVED)")
    print("=" * 70)
    
    # Convert VoIP - all metrics
    voip_data = convert_qos_file_all_metrics(voip_file, 'qos_voip_all_metrics.json')
    print("  ✓ Saved all VoIP metrics")
    
    # Convert CBR - all metrics
    cbr_data = convert_qos_file_all_metrics(cbr_file, 'qos_cbr_all_metrics.json')
    print("  ✓ Saved all CBR metrics")
    
    # Convert Video - all metrics
    video_data = convert_qos_file_all_metrics(video_file, 'qos_video_all_metrics.json')
    print("  ✓ Saved all Video metrics")
    
    print("\n" + "=" * 70)
    print("Conversion Complete!")
    print("=" * 70)
    print("\nNow you can choose which metric to use in config.py:")
    print("\n  # Option 1: Use specific metric per slice")
    print("  K = 3")
    print("  QOS_TABLE_FILES = [")
    print("      'qos_voip_all_metrics.json',")
    print("      'qos_cbr_all_metrics.json',")
    print("      'qos_video_all_metrics.json'")
    print("  ]")
    print("  QOS_METRICS = [")
    print("      'voIPFrameLoss',              # VoIP: use loss")
    print("      'cbrFrameDelay',               # CBR: use delay")
    print("      'rtVideoStreamingSegmentLoss'  # Video: use loss")
    print("  ]")
    print("\n  # Option 2: Use same metric for all slices")
    print("  QOS_METRICS = ['voIPFrameLoss', 'voIPFrameLoss', 'voIPFrameLoss']")
    print("\nThen run: python sac_training.py")


def show_available_metrics(json_file):
    """Show all available metrics in a QoS file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"\nAvailable metrics in {json_file}:")
    for i, metric in enumerate(data['meta']['metrics'], 1):
        unit = data['meta']['units'].get(metric, 'unknown')
        print(f"  {i}. {metric} ({unit})")


def create_example_output():
    """Show example of multi-metric structure"""
    example = {
        "5": {
            "1": {
                "voIPFrameLoss": {"mu": 0.149, "sigma": 0.525},
                "voIPFrameDelay": {"mu": 5.614, "sigma": 3.180},
                "voIPJitter": {"mu": 6.283, "sigma": 12.191},
                "voIPReceivedThroughput": {"mu": 1408.872, "sigma": 261.141}
            }
        }
    }
    
    print("\n" + "=" * 70)
    print("Example multi-metric structure (VoIP, UE=5, RB=1):")
    print("=" * 70)
    print(json.dumps(example, indent=2))
    print("\nNow you can select which metric to use for each slice!")


if __name__ == "__main__":
    
    print("QoS File Converter (Multi-Metric)")
    print("=" * 70)
    
    # Check if files exist
    import os
    
    voip_exists = os.path.exists('voip_qos.json')
    cbr_exists = os.path.exists('cbr_qos.json')
    video_exists = os.path.exists('video_qos.json')
    
    if not (voip_exists and cbr_exists and video_exists):
        print("\n⚠ Missing JSON files!")
        print("Please ensure these files are in the current directory:")
        print("  - voip_qos.json")
        print("  - cbr_qos.json")
        print("  - video_qos.json")
        print("\nOr modify the filenames in the script.")
        sys.exit(1)
    
    # Show available metrics
    print("\nAnalyzing your QoS files...")
    show_available_metrics('voip_qos.json')
    show_available_metrics('cbr_qos.json')
    show_available_metrics('video_qos.json')
    
    print("\n" + "=" * 70)
    print("Starting conversion (preserving ALL metrics)...")
    print("=" * 70)
    
    # Convert all files
    convert_all_preserve_metrics()
    
    # Show example
    create_example_output()
