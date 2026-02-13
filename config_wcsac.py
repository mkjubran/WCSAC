"""
WCSAC Configuration File

Extends base config.py with WCSAC-specific parameters for robust RL.
"""

# Import base configuration
from config import *

# ============================================================================
# WCSAC-SPECIFIC PARAMETERS
# ============================================================================

# Robustness Parameters
WCSAC_KAPPA = 0.5  # Robustness trade-off
# κ = 0.0: Standard SAC (no robustness)
# κ = 0.5: Balanced (recommended)
# κ = 1.0: Fully worst-case (very conservative)

WCSAC_UNCERTAINTY_RADIUS = 0.1  # Uncertainty set radius
# Controls size of perturbations to state and reward
# 0.1 = 10% perturbations (recommended for traffic variations)
# 0.05 = 5% (mild uncertainty)
# 0.2 = 20% (high uncertainty)

WCSAC_PESSIMISM_PENALTY = 0.1  # Conservative Q-learning penalty
# Penalty subtracted from Q-values for safety
# 0.0 = No penalty
# 0.1 = Mild conservatism (recommended)
# 0.5 = High conservatism

# ============================================================================
# OVERRIDE TRAINING PARAMETERS (optional)
# ============================================================================

# You may want different settings for WCSAC
# Uncomment and modify as needed:

# NUM_EPISODES = 1500  # WCSAC may need more episodes to converge
# BATCH_SIZE = 256     # Larger batches for stability
# LR_CRITIC = 1e-4     # Lower learning rate for robust critics

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_wcsac_config():
    """Return configuration with WCSAC parameters"""
    cfg = get_config()
    
    # Add WCSAC parameters
    cfg['wcsac_kappa'] = WCSAC_KAPPA
    cfg['wcsac_uncertainty_radius'] = WCSAC_UNCERTAINTY_RADIUS
    cfg['wcsac_pessimism_penalty'] = WCSAC_PESSIMISM_PENALTY
    
    return cfg


def print_wcsac_config():
    """Print configuration with WCSAC parameters"""
    print_config()
    
    print("\n[WCSAC ROBUSTNESS]")
    print(f"  κ (kappa):            {WCSAC_KAPPA} (0=SAC, 1=worst-case)")
    print(f"  Uncertainty radius:   {WCSAC_UNCERTAINTY_RADIUS}")
    print(f"  Pessimism penalty:    {WCSAC_PESSIMISM_PENALTY}")
    print("=" * 70)


if __name__ == "__main__":
    print_wcsac_config()
