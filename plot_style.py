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
