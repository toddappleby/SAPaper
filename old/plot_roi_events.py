#!/usr/bin/env python
"""
ROI Event Analysis Visualizations

Creates plots comparing behavioral sequences following different lever press types.
This helps visualize what the mouse does differently after:
  - Rewarded presses
  - Timeout presses  
  - Inactive presses

Usage:
    python plot_roi_events.py --input_dir ./roi_analysis --output_dir ./roi_plots
"""

import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class ROIEventPlotter:
    """Create visualizations for ROI event analysis."""
    
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.classified_events = None
        self.sequences = None
        self.summary = None
        
    def load_data(self):
        """Load the CSV files from ROI analysis."""
        print("📂 Loading ROI analysis data...")
        
        try:
            self.classified_events = pl.read_csv(
                self.input_dir / 'classified_lever_events.csv'
            )
            self.sequences = pl.read_csv(
                self.input_dir / 'post_press_sequences.csv'
            )
            self.summary = pl.read_csv(
                self.input_dir / 'roi_summary_by_event_type.csv'
            )
            print("✅ Data loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def plot_head_entry_latency_comparison(self):
        """
        Compare head entry latencies after different press types.
        This shows how quickly mice go to the magazine after different events.
        """
        print("📊 Plotting head entry latency comparison...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Filter out null values
        data = self.classified_events.filter(
            pl.col('next_head_entry_latency').is_not_null()
        )
        
        if len(data) == 0:
            print("⚠️  No head entry latency data available")
            return
        
        # Plot 1: Violin plot
        ax = axes[0]
        event_types = data['event_class'].unique().to_list()
        
        plot_data = []
        labels = []
        
        for event_type in event_types:
            latencies = data.filter(
                pl.col('event_class') == event_type
            )['next_head_entry_latency'].to_list()
            
            if latencies:
                plot_data.append(latencies)
                labels.append(event_type.replace('_', '\n'))
        
        parts = ax.violinplot(plot_data, positions=range(len(plot_data)),
                             showmeans=True, showextrema=True)
        
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Latency to Head Entry (seconds)')
        ax.set_title('Head Entry Latency by Press Type', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Mean with error bars
        ax = axes[1]
        
        summary_data = []
        for event_type in event_types:
            latencies = data.filter(
                pl.col('event_class') == event_type
            )['next_head_entry_latency']
            
            if len(latencies) > 0:
                mean_lat = latencies.mean()
                sem_lat = latencies.std() / np.sqrt(len(latencies))
                
                summary_data.append({
                    'event_type': event_type,
                    'mean': mean_lat,
                    'sem': sem_lat,
                    'n': len(latencies)
                })
        
        if summary_data:
            x_pos = np.arange(len(summary_data))
            means = [d['mean'] for d in summary_data]
            sems = [d['sem'] for d in summary_data]
            labels = [d['event_type'].replace('_', '\n') for d in summary_data]
            
            colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
            bars = ax.bar(x_pos, means, yerr=sems, capsize=5, 
                         color=colors[:len(summary_data)], alpha=0.7)
            
            # Add n values on bars
            for i, (bar, d) in enumerate(zip(bars, summary_data)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + sems[i],
                       f'n={d["n"]}', ha='center', va='bottom', fontsize=10)
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel('Mean Latency (seconds)')
            ax.set_title('Mean Head Entry Latency ± SEM', fontweight='bold', fontsize=14)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = self.output_dir / 'head_entry_latency_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {output_path.name}")
    
    def plot_post_press_behavior_comparison(self):
        """
        Compare what happens in the 10 seconds after different press types.
        """
        print("📊 Plotting post-press behavior comparison...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Get unique event classes
        event_classes = self.sequences['event_class'].unique().to_list()
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        # Plot 1: Head entries after press
        ax = axes[0, 0]
        for i, event_class in enumerate(event_classes):
            data = self.sequences.filter(pl.col('event_class') == event_class)
            
            if len(data) > 0:
                mean = data['post_press_head_entries'].mean()
                sem = data['post_press_head_entries'].std() / np.sqrt(len(data))
                
                ax.bar(i, mean, yerr=sem, capsize=5, 
                      color=colors[i % len(colors)], alpha=0.7,
                      label=event_class.replace('_', ' '))
        
        ax.set_xticks(range(len(event_classes)))
        ax.set_xticklabels([ec.replace('_', '\n') for ec in event_classes], 
                          rotation=45, ha='right')
        ax.set_ylabel('Mean Head Entries (10s window)')
        ax.set_title('Head Entries After Press', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Active presses after press
        ax = axes[0, 1]
        for i, event_class in enumerate(event_classes):
            data = self.sequences.filter(pl.col('event_class') == event_class)
            
            if len(data) > 0:
                mean = data['post_press_active_presses'].mean()
                sem = data['post_press_active_presses'].std() / np.sqrt(len(data))
                
                ax.bar(i, mean, yerr=sem, capsize=5,
                      color=colors[i % len(colors)], alpha=0.7)
        
        ax.set_xticks(range(len(event_classes)))
        ax.set_xticklabels([ec.replace('_', '\n') for ec in event_classes],
                          rotation=45, ha='right')
        ax.set_ylabel('Mean Active Presses (10s window)')
        ax.set_title('Active Presses After Press', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Inactive presses after press
        ax = axes[1, 0]
        for i, event_class in enumerate(event_classes):
            data = self.sequences.filter(pl.col('event_class') == event_class)
            
            if len(data) > 0:
                mean = data['post_press_inactive_presses'].mean()
                sem = data['post_press_inactive_presses'].std() / np.sqrt(len(data))
                
                ax.bar(i, mean, yerr=sem, capsize=5,
                      color=colors[i % len(colors)], alpha=0.7)
        
        ax.set_xticks(range(len(event_classes)))
        ax.set_xticklabels([ec.replace('_', '\n') for ec in event_classes],
                          rotation=45, ha='right')
        ax.set_ylabel('Mean Inactive Presses (10s window)')
        ax.set_title('Inactive Presses After Press', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Combined behavioral profile
        ax = axes[1, 1]
        
        behaviors = ['Head Entries', 'Active Presses', 'Inactive Presses']
        x = np.arange(len(behaviors))
        width = 0.2
        
        for i, event_class in enumerate(event_classes):
            data = self.sequences.filter(pl.col('event_class') == event_class)
            
            if len(data) > 0:
                means = [
                    data['post_press_head_entries'].mean(),
                    data['post_press_active_presses'].mean(),
                    data['post_press_inactive_presses'].mean()
                ]
                
                offset = width * (i - len(event_classes)/2 + 0.5)
                ax.bar(x + offset, means, width, 
                      label=event_class.replace('_', ' '),
                      color=colors[i % len(colors)], alpha=0.7)
        
        ax.set_xticks(x)
        ax.set_xticklabels(behaviors)
        ax.set_ylabel('Mean Count (10s window)')
        ax.set_title('Behavioral Profile After Press', fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = self.output_dir / 'post_press_behavior_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {output_path.name}")
    
    def plot_by_phase(self):
        """Compare event types across experimental phases."""
        print("📊 Plotting comparisons by phase...")
        
        phases = self.summary['phase'].unique().to_list()
        
        if len(phases) == 0:
            print("⚠️  No phase data available")
            return
        
        fig, axes = plt.subplots(len(phases), 2, figsize=(14, 5*len(phases)))
        
        if len(phases) == 1:
            axes = axes.reshape(1, -1)
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        for phase_idx, phase in enumerate(phases):
            phase_data = self.summary.filter(pl.col('phase') == phase)
            event_classes = phase_data['event_class'].to_list()
            
            # Plot 1: Head entry latency
            ax = axes[phase_idx, 0]
            
            means = phase_data['mean_head_entry_latency'].to_list()
            sems = [std / np.sqrt(n) if n > 0 else 0 
                   for std, n in zip(phase_data['std_head_entry_latency'].to_list(),
                                    phase_data['n_events'].to_list())]
            
            x_pos = np.arange(len(event_classes))
            ax.bar(x_pos, means, yerr=sems, capsize=5,
                  color=[colors[i % len(colors)] for i in range(len(event_classes))],
                  alpha=0.7)
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels([ec.replace('_', '\n') for ec in event_classes],
                              rotation=45, ha='right')
            ax.set_ylabel('Head Entry Latency (sec)')
            ax.set_title(f'Phase: {phase} - Head Entry Latency', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Plot 2: Head entries in 10s window
            ax = axes[phase_idx, 1]
            
            means = phase_data['mean_head_entries'].to_list()
            sems = [std / np.sqrt(n) if n > 0 else 0 
                   for std, n in zip(phase_data['std_head_entries'].to_list(),
                                    phase_data['n_events'].to_list())]
            
            ax.bar(x_pos, means, yerr=sems, capsize=5,
                  color=[colors[i % len(colors)] for i in range(len(event_classes))],
                  alpha=0.7)
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels([ec.replace('_', '\n') for ec in event_classes],
                              rotation=45, ha='right')
            ax.set_ylabel('Head Entries (10s window)')
            ax.set_title(f'Phase: {phase} - Post-Press Head Entries', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = self.output_dir / 'comparison_by_phase.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {output_path.name}")
    
    def create_event_timeline_examples(self, n_examples=5):
        """
        Create timeline visualizations showing example sequences for each event type.
        """
        print(f"📊 Creating {n_examples} example timelines per event type...")
        
        event_classes = self.classified_events['event_class'].unique().to_list()
        
        for event_class in event_classes:
            # Get examples
            examples = self.classified_events.filter(
                pl.col('event_class') == event_class
            ).head(n_examples)
            
            if len(examples) == 0:
                continue
            
            fig, axes = plt.subplots(len(examples), 1, figsize=(12, 3*len(examples)))
            
            if len(examples) == 1:
                axes = [axes]
            
            for idx, example in enumerate(examples.iter_rows(named=True)):
                ax = axes[idx]
                
                # Get all events for this session in a 20s window around the press
                press_time = example['time_seconds']
                session = example['filename']
                
                session_events = self.classified_events.filter(
                    pl.col('filename') == session
                )
                
                window_events = session_events.filter(
                    (pl.col('time_seconds') >= press_time - 5) &
                    (pl.col('time_seconds') <= press_time + 15)
                )
                
                # Plot timeline
                times = window_events['time_seconds'].to_list()
                times_rel = [t - press_time for t in times]
                response_types = window_events['response_type'].to_list()
                
                # Color code events
                colors_map = {
                    'active_lever': '#2E86AB',
                    'inactive_lever': '#C73E1D',
                    'head_entry': '#F18F01',
                    'reinforced': '#06A77D',
                    'timeout_start': '#A23B72'
                }
                
                for t, rt in zip(times_rel, response_types):
                    color = colors_map.get(rt, '#888888')
                    ax.axvline(t, color=color, alpha=0.6, linewidth=2)
                    ax.text(t, 0.5, rt.replace('_', '\n'), 
                           rotation=90, va='bottom', ha='center', fontsize=8)
                
                # Mark the trigger press
                ax.axvline(0, color='red', linewidth=3, label='Trigger Press')
                ax.axhline(0, color='black', linewidth=1)
                
                ax.set_xlim(-5, 15)
                ax.set_ylim(0, 1)
                ax.set_xlabel('Time from Press (seconds)')
                ax.set_title(f'Example {idx+1}: {example["subject_original"]} - {example["filename"]}',
                           fontweight='bold')
                ax.legend(loc='upper right')
                ax.set_yticks([])
            
            plt.tight_layout()
            output_path = self.output_dir / f'timeline_examples_{event_class}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Saved: {output_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description='Create visualizations for ROI event analysis'
    )
    parser.add_argument('--input_dir', default='./roi_analysis',
                       help='Directory with ROI analysis CSV files')
    parser.add_argument('--output_dir', default='./roi_plots',
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    plotter = ROIEventPlotter(args.input_dir, args.output_dir)
    
    if not plotter.load_data():
        return 1
    
    # Create all plots
    plotter.plot_head_entry_latency_comparison()
    plotter.plot_post_press_behavior_comparison()
    plotter.plot_by_phase()
    plotter.create_event_timeline_examples()
    
    print("\n" + "=" * 70)
    print("✅ All plots created successfully!")
    print("=" * 70)
    print(f"\nPlots saved to: {args.output_dir}")
    
    return 0


if __name__ == '__main__':
    exit(main())
