"""
config.py — Shared constants, style, and color configuration.
"""

import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# FIGURE STYLE CONFIGURATION
# ============================================================================
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'

sns.set_style("whitegrid")
plt.rcParams.update({
    'figure.dpi': FIGURE_DPI,
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 12,
    'lines.linewidth': 1.5,
    'grid.alpha': 0.3,
})

# Colors
COLOR_REWARD = 'blue'
COLOR_BETA = 'red'
COLOR_HOMOGENEOUS = 'steelblue'
COLOR_HETEROGENEOUS = 'coral'

BASELINE_COLORS = {
    'equal': 'lightcoral',
    'proportional': 'lightgreen',
    'greedy': 'gold',
    'random': 'lightgray',
}

# Axis labels
LABEL_BETA = 'QoS Violation Ratio (β)'
LABEL_REWARD = 'Episode Reward'
LABEL_EPISODE = 'Training Episode'
LABEL_DTI = 'Decision Time Interval (DTI)'
LABEL_RBS = 'Resource Blocks Allocated'

# Traffic profile sort order
LOAD_ORDER = {
    'extremely_low': 0,
    'low': 1,
    'medium': 2,
    'high': 3,
    'extremely_high': 4,
}

BASELINE_NAMES = ['equal', 'proportional', 'greedy', 'random']

# Traffic profile abbreviations used in all figure/table labels
PROFILE_ABBREV = {
    'extremely_low':  'EL',
    'low':            'L',
    'medium':         'M',
    'high':           'H',
    'extremely_high': 'EH',
    'dynamic':        'Dyn',
    'uniform':        'U',
    'external':       'Ext',
}


def abbrev_profile(profile):
    """
    Return the display abbreviation for a traffic profile name or list of names.

    Accepts:
      - a raw profile string:  'extremely_low'  -> 'EL'
      - a list of profiles:    ['low', 'high']  -> 'L, H'
      - a scenario_str from step1 (e.g. 'Extremely_Low - High') ->  'EL - H'

    Unknown profiles are returned unchanged.
    """
    if isinstance(profile, list):
        return ', '.join(abbrev_profile(p) for p in profile)
    key = profile.lower().strip().replace(' ', '_')
    return PROFILE_ABBREV.get(key, profile)


def abbrev_scenario_str(scenario_str):
    """
    Abbreviate every profile token inside a scenario_str.

    e.g. 'Extremely_Low - Extremely_High'  ->  'EL - EH'
         'Low - High'                       ->  'L - H'
         'Dynamic [Low, Medium, High]...'   ->  unchanged (dynamic label)
    """
    if scenario_str.startswith('Dynamic'):
        return scenario_str  # dynamic labels are handled separately
    parts = scenario_str.split(' - ')
    return ' - '.join(abbrev_profile(p.strip()) for p in parts)
