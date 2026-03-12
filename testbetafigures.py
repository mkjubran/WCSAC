"""
Test Script: Verify Beta Figure Generation

This creates synthetic data and tests if the beta figure code works.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Create synthetic beta data
print("Creating synthetic beta data...")
dtis = np.arange(2000)  # 2000 DTIs for episode 80

# Simulate beta values: low with some spikes at transitions
beta_values = []
for dti in dtis:
    # Base low value
    base = 0.05
    
    # Spike at profile transitions (every 200 DTIs)
    if dti % 200 < 10:
        spike = 0.3 * (1 - (dti % 200) / 10)  # Decaying spike
    else:
        spike = 0
    
    # Add some noise
    noise = np.random.uniform(-0.02, 0.02)
    
    beta = np.clip(base + spike + noise, 0, 1)
    beta_values.append(beta)

beta_values = np.array(beta_values)

print(f"Generated {len(beta_values)} beta values")
print(f"  Min: {beta_values.min():.3f}")
print(f"  Max: {beta_values.max():.3f}")
print(f"  Mean: {beta_values.mean():.3f}")

# Test the figure generation code
print("\nGenerating beta figure...")

fig, ax = plt.subplots(figsize=(14, 6))

# Plot continuous beta curve
ax.plot(dtis, beta_values, color='red', alpha=0.8, linewidth=2.0, 
       label='β (QoS Violation Ratio)')

ax.set_xlabel('DTI (within Episode 80)', fontsize=12)
ax.set_ylabel('QoS Violation Ratio (β)', fontsize=12)
ax.set_title('QoS Violation Ratio Over Time (Episode 80)', fontsize=14, fontweight='bold')
ax.legend(loc='best', framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])  # Beta is between 0 and 1

# Add vertical lines at period boundaries (every 200 DTIs)
for period in range(1, 10):
    ax.axvline(x=period * 200, color='gray', linestyle=':', alpha=0.5, linewidth=1)

# Add horizontal line at beta=0 for reference
ax.axhline(y=0, color='green', linestyle='--', alpha=0.3, linewidth=1, label='Perfect QoS (β=0)')

plt.tight_layout()

# Save
output_file = 'test_beta_figure.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Figure saved to: {output_file}")

# Check file exists and has size
if os.path.exists(output_file):
    size = os.path.getsize(output_file)
    print(f"✓ File size: {size:,} bytes")
    
    if size > 10000:
        print("✓ Figure generation code works correctly!")
        print(f"\nYou can view the test figure: {output_file}")
    else:
        print("✗ File is too small - something went wrong")
else:
    print("✗ File was not created")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

if os.path.exists(output_file) and os.path.getsize(output_file) > 10000:
    print("✓ The beta figure generation code works correctly!")
    print("\nIf step2 isn't creating beta figures, the issue is:")
    print("  1. The if-check isn't passing (ep80_beta not in data)")
    print("  2. The code is in wrong location in step2")
    print("  3. step2 is throwing an error but continuing")
    print("\nTo debug:")
    print("  1. Add print statements in step2 before the beta figure code")
    print("  2. Run step2 with verbose output")
    print("  3. Check if any errors are printed")
else:
    print("✗ The figure generation code has an issue")
    print("  Check the error messages above")
