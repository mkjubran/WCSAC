"""
Extract Comprehensive Results from TensorBoard for Multi-Metric Cross-Layer Paper

This script extracts and analyzes TensorBoard logs to generate results for the paper,
including multi-metric QoS, transport layer metrics, and priority differentiation.

Usage:
    python3 extract_paper_results.py --log-dir results/tensorboard_logs/sac_run1
    python3 extract_paper_results.py --log-dir results/tensorboard_logs/sac_run1 --output-dir paper_figures
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import json

try:
    from tensorboard.backend.event_processing import event_accumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")


# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


def read_tensorboard_data(log_dir):
    """
    Read all relevant metrics from TensorBoard logs.
    
    Returns:
        dict: {tag: [(step, value), ...]}
    """
    if not TENSORBOARD_AVAILABLE:
        print("ERROR: tensorboard package not found.")
        return None
    
    # Find event files
    event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))
    
    if not event_files:
        print(f"No TensorBoard event files found in: {log_dir}")
        return None
    
    print(f"Found {len(event_files)} event file(s)")
    
    # Load event accumulator
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    # Get available tags
    available_tags = ea.Tags().get('scalars', [])
    print(f"\nFound {len(available_tags)} scalar metrics")
    
    # Define metrics to extract
    metrics_of_interest = {
        # RAN metrics
        'episode/reward': 'Episode Reward',
        'episode/avg_beta': 'Multi-Metric QoS Violation (β)',
        'episode/buffer_size': 'Replay Buffer Size',
        'episode/avg_reward_100': 'Avg Reward (100 ep)',
        'episode/avg_beta_100': 'Avg Beta (100 ep)',
        
        # Transport layer metrics (episode level)
        'episode/avg_transport_utilization': 'Transport Utilization (ρ)',
        'episode/avg_transport_penalty': 'Transport Penalty',
        'episode/avg_transport_delay_slice0_ms': 'VoIP Delay (ms)',
        'episode/avg_transport_delay_slice1_ms': 'CBR Delay (ms)',
        'episode/avg_transport_delay_slice2_ms': 'Video Delay (ms)',
        
        # Per-DTI metrics (for detailed analysis)
        'dti/reward': 'DTI Reward',
        'dti/beta': 'DTI Beta',
        'dti/transport_utilization': 'DTI Transport Utilization',
        'dti/transport_stable': 'DTI Transport Stable',
        'dti/transport_penalty': 'DTI Transport Penalty',
        
        # Per-slice transport delays (DTI level)
        'dti/transport_delay_slice0_ms': 'DTI VoIP Delay (ms)',
        'dti/transport_delay_slice1_ms': 'DTI CBR Delay (ms)',
        'dti/transport_delay_slice2_ms': 'DTI Video Delay (ms)',
        
        # Success rates
        'dti/success_rate_slice0': 'VoIP Success Rate',
        'dti/success_rate_slice1': 'CBR Success Rate',
        'dti/success_rate_slice2': 'Video Success Rate',
        
        # Actions and traffic
        'dti/action_slice0': 'VoIP RB Allocation',
        'dti/action_slice1': 'CBR RB Allocation',
        'dti/action_slice2': 'Video RB Allocation',
        'dti/traffic_slice0': 'VoIP Traffic',
        'dti/traffic_slice1': 'CBR Traffic',
        'dti/traffic_slice2': 'Video Traffic',
    }
    
    # Extract data
    data = {}
    for tag, description in metrics_of_interest.items():
        if tag in available_tags:
            events = ea.Scalars(tag)
            data[tag] = {
                'values': [(e.step, e.value) for e in events],
                'description': description
            }
            print(f"  ✓ {tag}: {len(data[tag]['values'])} points")
        else:
            print(f"  ✗ {tag}: Not found")
    
    return data


def compute_statistics(data):
    """Compute comprehensive statistics for paper."""
    
    stats = {}
    
    # Episode-level metrics
    episode_metrics = [
        'episode/reward',
        'episode/avg_beta',
        'episode/avg_transport_utilization',
        'episode/avg_transport_penalty',
        'episode/avg_transport_delay_slice0_ms',
        'episode/avg_transport_delay_slice1_ms',
        'episode/avg_transport_delay_slice2_ms',
    ]
    
    for metric in episode_metrics:
        if metric in data:
            vals = np.array([v[1] for v in data[metric]['values']])
            
            # Overall statistics
            stats[metric] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'min': float(np.min(vals)),
                'max': float(np.max(vals)),
                'median': float(np.median(vals)),
            }
            
            # Last 100 episodes (converged performance)
            if len(vals) >= 100:
                last_100 = vals[-100:]
                stats[metric]['last_100_mean'] = float(np.mean(last_100))
                stats[metric]['last_100_std'] = float(np.std(last_100))
    
    # Priority differentiation analysis
    if all(f'episode/avg_transport_delay_slice{k}_ms' in data for k in range(3)):
        delay_0 = np.array([v[1] for v in data['episode/avg_transport_delay_slice0_ms']['values']])
        delay_1 = np.array([v[1] for v in data['episode/avg_transport_delay_slice1_ms']['values']])
        delay_2 = np.array([v[1] for v in data['episode/avg_transport_delay_slice2_ms']['values']])
        
        # Last 100 episodes
        if len(delay_0) >= 100:
            stats['priority_differentiation'] = {
                'voip_delay_mean': float(np.mean(delay_0[-100:])),
                'cbr_delay_mean': float(np.mean(delay_1[-100:])),
                'video_delay_mean': float(np.mean(delay_2[-100:])),
                'gap_video_voip': float(np.mean(delay_2[-100:]) - np.mean(delay_0[-100:])),
                'gap_cbr_voip': float(np.mean(delay_1[-100:]) - np.mean(delay_0[-100:])),
                'ratio_video_voip': float(np.mean(delay_2[-100:]) / np.mean(delay_0[-100:])) if np.mean(delay_0[-100:]) > 0 else 0,
            }
    
    return stats


def plot_training_convergence(data, output_dir):
    """Generate Figure: Training Convergence (for paper)."""
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Top: Episode Reward
    if 'episode/reward' in data:
        steps = [v[0] for v in data['episode/reward']['values']]
        values = [v[1] for v in data['episode/reward']['values']]
        
        axes[0].plot(steps, values, alpha=0.3, color='blue', linewidth=0.5)
        
        # Moving average
        if len(values) >= 100:
            ma = np.convolve(values, np.ones(100)/100, mode='valid')
            axes[0].plot(steps[99:], ma, color='blue', linewidth=2, label='100-episode MA')
        
        axes[0].set_xlabel('Training Episode')
        axes[0].set_ylabel('Episode Reward')
        axes[0].set_title('Training Convergence: Episode Reward')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Bottom: Multi-Metric Beta
    if 'episode/avg_beta' in data:
        steps = [v[0] for v in data['episode/avg_beta']['values']]
        values = [v[1] for v in data['episode/avg_beta']['values']]
        
        axes[1].plot(steps, values, alpha=0.3, color='red', linewidth=0.5)
        
        # Moving average
        if len(values) >= 100:
            ma = np.convolve(values, np.ones(100)/100, mode='valid')
            axes[1].plot(steps[99:], ma, color='red', linewidth=2, label='100-episode MA')
        
        axes[1].set_xlabel('Training Episode')
        axes[1].set_ylabel('Multi-Metric QoS Violation (β)')
        axes[1].set_title('Training Convergence: Multi-Metric QoS Performance')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0.2, color='green', linestyle='--', label='Target (β < 0.2)', alpha=0.5)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'training_convergence.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_priority_differentiation(data, output_dir):
    """Generate Figure: Priority Differentiation in Transport Layer."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    delays_available = all(f'episode/avg_transport_delay_slice{k}_ms' in data for k in range(3))
    
    if delays_available:
        steps = [v[0] for v in data['episode/avg_transport_delay_slice0_ms']['values']]
        
        delay_0 = [v[1] for v in data['episode/avg_transport_delay_slice0_ms']['values']]
        delay_1 = [v[1] for v in data['episode/avg_transport_delay_slice1_ms']['values']]
        delay_2 = [v[1] for v in data['episode/avg_transport_delay_slice2_ms']['values']]
        
        # Plot with moving average
        window = 50
        if len(delay_0) >= window:
            ma_0 = np.convolve(delay_0, np.ones(window)/window, mode='valid')
            ma_1 = np.convolve(delay_1, np.ones(window)/window, mode='valid')
            ma_2 = np.convolve(delay_2, np.ones(window)/window, mode='valid')
            
            steps_ma = steps[window-1:]
            
            ax.plot(steps_ma, ma_0, color='green', linewidth=2, label='VoIP (Priority 0 - Highest)', marker='o', markevery=50)
            ax.plot(steps_ma, ma_1, color='orange', linewidth=2, label='CBR (Priority 1 - Medium)', marker='s', markevery=50)
            ax.plot(steps_ma, ma_2, color='red', linewidth=2, label='Video (Priority 2 - Lowest)', marker='^', markevery=50)
        
        ax.set_xlabel('Training Episode')
        ax.set_ylabel('Average Transport Delay (ms)')
        ax.set_title('Priority Differentiation in Transport Layer (M/G/1 Queue)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Add annotation showing expected hierarchy
        ax.text(0.98, 0.98, 'Expected: VoIP < CBR < Video',
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'priority_differentiation.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_transport_utilization(data, output_dir):
    """Generate Figure: Transport Utilization Over Training."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'episode/avg_transport_utilization' in data:
        steps = [v[0] for v in data['episode/avg_transport_utilization']['values']]
        values = [v[1] for v in data['episode/avg_transport_utilization']['values']]
        
        # Raw data
        ax.plot(steps, values, alpha=0.3, color='purple', linewidth=0.5)
        
        # Moving average
        window = 50
        if len(values) >= window:
            ma = np.convolve(values, np.ones(window)/window, mode='valid')
            ax.plot(steps[window-1:], ma, color='purple', linewidth=2, label=f'{window}-episode MA')
        
        # Stability threshold
        ax.axhline(y=0.999, color='red', linestyle='--', label='Stability Threshold (ρ = 0.999)', alpha=0.7)
        ax.axhline(y=0.9, color='orange', linestyle='--', label='High Load (ρ = 0.9)', alpha=0.7)
        
        ax.set_xlabel('Training Episode')
        ax.set_ylabel('Transport Layer Utilization (ρ)')
        ax.set_title('Transport Layer Utilization Over Training')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'transport_utilization.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_cross_layer_metrics(data, output_dir):
    """Generate Figure: Cross-Layer Performance (RAN + Transport)."""
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # Top: RAN QoS (Beta)
    if 'episode/avg_beta' in data:
        steps = [v[0] for v in data['episode/avg_beta']['values']]
        values = [v[1] for v in data['episode/avg_beta']['values']]
        
        window = 50
        if len(values) >= window:
            ma = np.convolve(values, np.ones(window)/window, mode='valid')
            axes[0].plot(steps[window-1:], ma, color='blue', linewidth=2, label='Multi-Metric β (RAN)')
        
        axes[0].set_ylabel('RAN QoS Violation (β)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('Cross-Layer Performance: RAN Layer')
    
    # Bottom: Transport Penalty
    if 'episode/avg_transport_penalty' in data:
        steps = [v[0] for v in data['episode/avg_transport_penalty']['values']]
        values = [v[1] for v in data['episode/avg_transport_penalty']['values']]
        
        window = 50
        if len(values) >= window:
            ma = np.convolve(values, np.ones(window)/window, mode='valid')
            axes[1].plot(steps[window-1:], ma, color='red', linewidth=2, label='Transport Penalty')
        
        axes[1].set_xlabel('Training Episode')
        axes[1].set_ylabel('Transport Penalty')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_title('Cross-Layer Performance: Transport Layer')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'cross_layer_performance.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def generate_latex_table_results(stats, output_dir):
    """Generate LaTeX table with results for paper."""
    
    latex = r"""
\begin{table}[!t]
\caption{Training Results Summary (Last 100 Episodes)}
\label{tab:training_results}
\centering
\begin{tabular}{@{}lcc@{}}
\toprule
\textbf{Metric} & \textbf{Mean} & \textbf{Std Dev} \\
\midrule
"""
    
    # Add metrics
    if 'episode/avg_beta' in stats and 'last_100_mean' in stats['episode/avg_beta']:
        beta_mean = stats['episode/avg_beta']['last_100_mean']
        beta_std = stats['episode/avg_beta']['last_100_std']
        latex += f"Multi-Metric QoS Violation (β) & {beta_mean:.3f} & {beta_std:.3f} \\\\\n"
    
    if 'episode/avg_transport_utilization' in stats and 'last_100_mean' in stats['episode/avg_transport_utilization']:
        rho_mean = stats['episode/avg_transport_utilization']['last_100_mean']
        rho_std = stats['episode/avg_transport_utilization']['last_100_std']
        latex += f"Transport Utilization (ρ) & {rho_mean:.3f} & {rho_std:.3f} \\\\\n"
    
    if 'priority_differentiation' in stats:
        pd = stats['priority_differentiation']
        latex += r"\midrule" + "\n"
        latex += r"\multicolumn{3}{@{}l}{\textit{Transport Layer Delays (ms)}} \\" + "\n"
        latex += f"VoIP (Priority 0) & {pd['voip_delay_mean']:.2f} & -- \\\\\n"
        latex += f"CBR (Priority 1) & {pd['cbr_delay_mean']:.2f} & -- \\\\\n"
        latex += f"Video (Priority 2) & {pd['video_delay_mean']:.2f} & -- \\\\\n"
        latex += r"\midrule" + "\n"
        latex += f"Priority Gap (Video - VoIP) & {pd['gap_video_voip']:.2f} ms & -- \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    # Save to file
    output_path = os.path.join(output_dir, 'results_table.tex')
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f"  Saved LaTeX table: {output_path}")
    return latex


def main():
    parser = argparse.ArgumentParser(description='Extract Paper Results from TensorBoard')
    parser.add_argument('--log-dir', type=str, required=True, help='Path to TensorBoard log directory')
    parser.add_argument('--output-dir', type=str, default='paper_results', help='Output directory for figures')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("EXTRACTING PAPER RESULTS FROM TENSORBOARD")
    print("="*80)
    
    # Read data
    print("\n[1/5] Reading TensorBoard Data...")
    data = read_tensorboard_data(args.log_dir)
    
    if not data:
        print("ERROR: No data extracted. Exiting.")
        return
    
    # Compute statistics
    print("\n[2/5] Computing Statistics...")
    stats = compute_statistics(data)
    
    # Save statistics to JSON
    stats_path = os.path.join(args.output_dir, 'statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved statistics: {stats_path}")
    
    # Generate figures
    print("\n[3/5] Generating Figures...")
    plot_training_convergence(data, args.output_dir)
    plot_priority_differentiation(data, args.output_dir)
    plot_transport_utilization(data, args.output_dir)
    plot_cross_layer_metrics(data, args.output_dir)
    
    # Generate LaTeX table
    print("\n[4/5] Generating LaTeX Table...")
    generate_latex_table_results(stats, args.output_dir)
    
    # Print summary
    print("\n[5/5] Summary")
    print("="*80)
    
    if 'priority_differentiation' in stats:
        pd = stats['priority_differentiation']
        print("\nPriority Differentiation (Last 100 Episodes):")
        print(f"  VoIP Delay:  {pd['voip_delay_mean']:.3f} ms")
        print(f"  CBR Delay:   {pd['cbr_delay_mean']:.3f} ms")
        print(f"  Video Delay: {pd['video_delay_mean']:.3f} ms")
        print(f"  Gap (Video - VoIP): {pd['gap_video_voip']:.3f} ms")
        print(f"  Ratio (Video / VoIP): {pd['ratio_video_voip']:.2f}x")
        
        # Check if priority ordering is correct
        if pd['voip_delay_mean'] < pd['cbr_delay_mean'] < pd['video_delay_mean']:
            print("  ✓ Priority ordering CORRECT: VoIP < CBR < Video")
        else:
            print("  ✗ Priority ordering INCORRECT")
    
    print("\n" + "="*80)
    print(f"Results saved to: {args.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
