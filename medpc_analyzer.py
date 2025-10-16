#!/usr/bin/env python
"""
MedPC Data Analyzer

Interactive tool for analyzing organized MedPC data.
Load the organized data and perform various analyses.

Usage:
    python medpc_analyzer.py --organized_dir ./organized_data
"""

import polars as pl
import numpy as np
from pathlib import Path
import argparse
import json


class MedPCAnalyzer:
    """
    Analyze organized MedPC data with easy-to-use methods.
    """
    
    def __init__(self, organized_dir):
        """
        Load organized MedPC data.
        
        Parameters:
        -----------
        organized_dir : str or Path
            Directory containing organized data from MedPCDataOrganizer
        """
        self.data_dir = Path(organized_dir)
        
        # Load data
        self.sessions = pl.read_csv(self.data_dir / 'session_summary.csv')
        self.time_series = pl.read_csv(self.data_dir / 'time_series_all_events.csv')
        self.time_windows = pl.read_csv(self.data_dir / 'time_window_summary.csv')
        
        # Load metadata
        with open(self.data_dir / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        print(f"✅ Loaded data from {organized_dir}")
        print(f"  • {len(self.sessions)} sessions")
        print(f"  • {len(self.time_series)} events")
        print(f"  • Phases: {', '.join(self.metadata['phases'])}")
        print(f"  • Subjects: {', '.join(self.metadata['subjects'])}")
    
    def get_by_phase(self, phase):
        """
        Get all data for a specific phase.
        
        Parameters:
        -----------
        phase : str
            'SelfAdmin', 'EXT', or 'REI'
            
        Returns:
        --------
        dict with sessions, time_series, and time_windows DataFrames
        """
        return {
            'sessions': self.sessions.filter(pl.col('phase') == phase),
            'time_series': self.time_series.filter(pl.col('phase') == phase),
            'time_windows': self.time_windows.filter(pl.col('phase') == phase),
        }
    
    def get_by_subject(self, subject):
        """
        Get all data for a specific subject.
        
        Parameters:
        -----------
        subject : str
            Subject ID
            
        Returns:
        --------
        dict with sessions, time_series, and time_windows DataFrames
        """
        return {
            'sessions': self.sessions.filter(pl.col('subject') == subject),
            'time_series': self.time_series.filter(pl.col('subject') == subject),
            'time_windows': self.time_windows.filter(pl.col('subject') == subject),
        }
    
    def get_time_window(self, start_min, end_min, phase=None, subject=None):
        """
        Get events within a specific time window.
        
        Parameters:
        -----------
        start_min : float
            Start time in minutes
        end_min : float
            End time in minutes
        phase : str, optional
            Filter by phase
        subject : str, optional
            Filter by subject
            
        Returns:
        --------
        Polars DataFrame with filtered events
        """
        df = self.time_series.filter(
            (pl.col('time_minutes') >= start_min) & 
            (pl.col('time_minutes') < end_min)
        )
        
        if phase:
            df = df.filter(pl.col('phase') == phase)
        
        if subject:
            df = df.filter(pl.col('subject') == subject)
        
        return df
    
    def count_active_presses(self, start_min=0, end_min=None, phase=None, subject=None):
        """
        Count active lever presses in a time window.
        
        Parameters:
        -----------
        start_min : float
            Start time in minutes
        end_min : float, optional
            End time in minutes (None = all remaining time)
        phase : str, optional
            Filter by phase
        subject : str, optional
            Filter by subject
            
        Returns:
        --------
        Polars DataFrame with counts by session
        """
        if end_min is None:
            end_min = self.time_series['time_minutes'].max()
        
        window_data = self.get_time_window(start_min, end_min, phase, subject)
        
        summary = window_data.group_by(['subject', 'phase', 'filename']).agg([
            pl.col('is_active_lever').sum().alias('active_presses'),
            pl.col('is_inactive_lever').sum().alias('inactive_presses'),
            pl.col('is_head_entry').sum().alias('head_entries'),
        ])
        
        return summary
    
    def compare_phases(self, metric='active_lever_total'):
        """
        Compare a metric across different phases.
        
        Parameters:
        -----------
        metric : str
            Column name from session_summary to compare
            
        Returns:
        --------
        Polars DataFrame with comparison
        """
        comparison = self.sessions.group_by('phase').agg([
            pl.col(metric).mean().alias(f'mean_{metric}'),
            pl.col(metric).std().alias(f'std_{metric}'),
            pl.count('subject').alias('n_sessions'),
        ])
        
        return comparison
    
    def get_session_progression(self, subject):
        """
        Get temporal progression of behavior within a single session.
        
        Parameters:
        -----------
        subject : str
            Subject ID
            
        Returns:
        --------
        Polars DataFrame with binned data
        """
        subject_data = self.time_windows.filter(pl.col('subject') == subject)
        return subject_data.sort('start_min')
    
    def get_lever_press_times(self, subject=None, phase=None, lever_type='active'):
        """
        Get timestamps of all lever presses.
        
        Parameters:
        -----------
        subject : str, optional
            Filter by subject
        phase : str, optional
            Filter by phase
        lever_type : str
            'active' or 'inactive'
            
        Returns:
        --------
        Polars DataFrame with press times
        """
        df = self.time_series
        
        if subject:
            df = df.filter(pl.col('subject') == subject)
        
        if phase:
            df = df.filter(pl.col('phase') == phase)
        
        if lever_type == 'active':
            df = df.filter(pl.col('is_active_lever') == True)
        elif lever_type == 'inactive':
            df = df.filter(pl.col('is_inactive_lever') == True)
        
        return df.select(['subject', 'phase', 'filename', 'time_seconds', 'time_minutes', 'response_code'])
    
    def calculate_inter_press_intervals(self, subject, phase=None):
        """
        Calculate time between consecutive active lever presses.
        
        Parameters:
        -----------
        subject : str
            Subject ID
        phase : str, optional
            Filter by phase
            
        Returns:
        --------
        numpy array of inter-press intervals (seconds)
        """
        press_times = self.get_lever_press_times(subject, phase, 'active')
        times = press_times['time_seconds'].to_numpy()
        
        if len(times) < 2:
            return np.array([])
        
        intervals = np.diff(sorted(times))
        return intervals
    
    def export_for_analysis(self, output_path, phase=None, time_window=None):
        """
        Export data in format ready for statistical analysis software.
        
        Parameters:
        -----------
        output_path : str or Path
            Output file path
        phase : str, optional
            Filter by phase
        time_window : tuple, optional
            (start_min, end_min) to filter
        """
        output_path = Path(output_path)
        
        # Start with session summary
        df = self.sessions.clone()
        
        if phase:
            df = df.filter(pl.col('phase') == phase)
        
        # Add time window data if specified
        if time_window:
            start_min, end_min = time_window
            window_label = f"{start_min}-{end_min}min"
            
            tw_data = self.time_windows.filter(
                pl.col('time_window') == window_label
            ).select(['subject', 'filename', 'active_lever_presses', 'inactive_lever_presses', 'head_entries'])
            
            tw_data = tw_data.rename({
                'active_lever_presses': f'active_{window_label}',
                'inactive_lever_presses': f'inactive_{window_label}',
                'head_entries': f'head_entries_{window_label}',
            })
            
            df = df.join(tw_data, on=['subject', 'filename'], how='left')
        
        df.write_csv(output_path)
        print(f"✅ Exported analysis-ready data to {output_path}")
    
    def print_summary(self):
        """Print a comprehensive summary of the dataset."""
        print("\n" + "="*70)
        print("📊 MEDPC DATA ANALYSIS SUMMARY")
        print("="*70)
        
        print(f"\n📁 Dataset Overview:")
        print(f"  • Total sessions: {len(self.sessions)}")
        print(f"  • Total events: {len(self.time_series)}")
        print(f"  • Subjects: {', '.join(self.metadata['subjects'])}")
        print(f"  • Phases: {', '.join(self.metadata['phases'])}")
        
        print(f"\n🧪 By Phase:")
        for phase in ['SelfAdmin', 'EXT', 'REI']:
            phase_sessions = self.sessions.filter(pl.col('phase') == phase)
            if len(phase_sessions) > 0:
                n_sessions = len(phase_sessions)
                mean_active = phase_sessions['active_lever_total'].mean()
                mean_inactive = phase_sessions['inactive_lever_total'].mean()
                mean_reinforcers = phase_sessions['reinforcers'].mean()
                
                print(f"\n  {phase}:")
                print(f"    • Sessions: {n_sessions}")
                print(f"    • Mean active presses: {mean_active:.1f}")
                print(f"    • Mean inactive presses: {mean_inactive:.1f}")
                print(f"    • Mean reinforcers: {mean_reinforcers:.1f}")
        
        print(f"\n👤 By Subject:")
        for subject in sorted(self.metadata['subjects']):
            subject_sessions = self.sessions.filter(pl.col('subject') == subject)
            n_sessions = len(subject_sessions)
            phases = subject_sessions['phase'].unique().to_list()
            
            print(f"\n  {subject}:")
            print(f"    • Sessions: {n_sessions}")
            print(f"    • Phases: {', '.join(phases)}")
            
            for phase in phases:
                phase_data = subject_sessions.filter(pl.col('phase') == phase)
                if len(phase_data) > 0:
                    active = phase_data['active_lever_total'].mean()
                    print(f"    • {phase} avg active presses: {active:.1f}")
        
        print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze organized MedPC data'
    )
    parser.add_argument('--organized_dir', default='./organized_data',
                       help='Directory containing organized MedPC data')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = MedPCAnalyzer(args.organized_dir)
    
    # Print summary
    analyzer.print_summary()
    
    # Example analyses
    print("\n" + "="*70)
    print("💡 EXAMPLE ANALYSES")
    print("="*70)
    
    print("\n1️⃣ Active presses in first 30 minutes by subject:")
    first_30 = analyzer.count_active_presses(start_min=0, end_min=30)
    print(first_30)
    
    print("\n2️⃣ Compare phases (if multiple phases present):")
    if len(analyzer.metadata['phases']) > 1:
        comparison = analyzer.compare_phases('active_lever_total')
        print(comparison)
    else:
        print("  (Only one phase in dataset)")
    
    print("\n3️⃣ Time window breakdown:")
    print(analyzer.time_windows)
    
    print("\n" + "="*70)
    print("💡 PYTHON USAGE EXAMPLES")
    print("="*70)
    print("""
# Load the analyzer
from medpc_analyzer import MedPCAnalyzer
analyzer = MedPCAnalyzer('./organized_data')

# Get specific phase data
selfadmin = analyzer.get_by_phase('SelfAdmin')

# Get specific subject data
mouse_t2 = analyzer.get_by_subject('T2')

# Get first 30 minutes of data
first_30min = analyzer.get_time_window(0, 30)

# Count presses in a window
counts = analyzer.count_active_presses(start_min=0, end_min=30, phase='SelfAdmin')

# Get all active lever press times
press_times = analyzer.get_lever_press_times(subject='T2', lever_type='active')

# Calculate inter-press intervals
intervals = analyzer.calculate_inter_press_intervals(subject='T2')

# Export for statistical software
analyzer.export_for_analysis('analysis_ready.csv', phase='SelfAdmin', time_window=(0, 30))
    """)
    print("="*70)


if __name__ == '__main__':
    main()
