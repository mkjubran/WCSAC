"""
Test Config Loading

Quick script to test if QOS_TABLE_FILES and QOS_METRICS are being loaded correctly.

Usage:
    python3 test_config_loading.py config_yourfile.py
"""

import sys
import importlib.util

def load_and_print_config(config_path):
    """Load config and print what we find"""
    
    print(f"Loading config from: {config_path}\n")
    
    # Load module
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    # Check all attributes
    all_attrs = dir(config_module)
    
    # Filter to just config variables (no built-ins)
    config_vars = [attr for attr in all_attrs if not attr.startswith('_')]
    
    print(f"All variables found: {len(config_vars)}\n")
    
    # Show uppercase variables
    uppercase_vars = [v for v in config_vars if v.isupper()]
    print(f"Uppercase variables ({len(uppercase_vars)}):")
    for var in sorted(uppercase_vars):
        value = getattr(config_module, var)
        # Shorten long lists
        if isinstance(value, list) and len(value) > 3:
            print(f"  {var:30} = [{value[0]}, {value[1]}, ... ({len(value)} items)]")
        else:
            print(f"  {var:30} = {value}")
    
    # Check specifically for QoS variables
    print(f"\n" + "="*70)
    print("QoS CONFIGURATION CHECK")
    print("="*70)
    
    if hasattr(config_module, 'QOS_TABLE_FILES'):
        qos_files = config_module.QOS_TABLE_FILES
        print(f"\n✓ QOS_TABLE_FILES found:")
        print(f"  Type: {type(qos_files)}")
        print(f"  Value: {qos_files}")
    else:
        print(f"\n✗ QOS_TABLE_FILES NOT FOUND")
    
    if hasattr(config_module, 'QOS_METRICS'):
        qos_metrics = config_module.QOS_METRICS
        print(f"\n✓ QOS_METRICS found:")
        print(f"  Type: {type(qos_metrics)}")
        print(f"  Value: {qos_metrics}")
    else:
        print(f"\n✗ QOS_METRICS NOT FOUND")
    
    # Extract config the same way comprehensive_evaluation does
    print(f"\n" + "="*70)
    print("EXTRACTED CONFIG (same as comprehensive_evaluation.py)")
    print("="*70)
    
    config = {
        key: getattr(config_module, key)
        for key in dir(config_module)
        if key.isupper() and not key.startswith('_')
    }
    
    print(f"\nExtracted {len(config)} parameters")
    
    if 'QOS_TABLE_FILES' in config:
        print(f"✓ QOS_TABLE_FILES in extracted config: {config['QOS_TABLE_FILES']}")
    else:
        print(f"✗ QOS_TABLE_FILES NOT in extracted config")
    
    if 'QOS_METRICS' in config:
        print(f"✓ QOS_METRICS in extracted config: {config['QOS_METRICS']}")
    else:
        print(f"✗ QOS_METRICS NOT in extracted config")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 test_config_loading.py config_file.py")
        sys.exit(1)
    
    config_path = sys.argv[1]
    load_and_print_config(config_path)
