#!/usr/bin/env python
"""
MedPC ROI Event Extractor

This script processes MedPC data files to extract event-triggered sequences
for ROI (Region of Interest) analysis. It distinguishes between:
  - Rewarded active lever presses (Code 2)
  - Timeout lever presses (Code 21) 
  - Inactive lever presses (Codes 1, 20, 23)
  
The goal is to analyze mouse behavior AFTER different types of lever press events
to understand spatial patterns and reward-seeking vs. frustration behaviors.

Usage:
    python medpc_roi_event_extractor.py --data_dir ./data --output_dir ./roi_analysis
"""

import polars as pl
import numpy as np
from pathlib import Path
import argparse
import sys

# Add parent directory to path to import existing parsers
sys.path.append(str(Path(__file__).parent))

try:
    from enhanced_medpc_parser import EnhancedMedPCDataParser
except ImportError:
    print("Warning: Could not import EnhancedMedPCDataParser. Using basic parser.")
    from medpc_parser import MedPCDataParser as EnhancedMedPCDataParser


class MedPCROIEventExtractor:
    """
    Extract event-triggered sequences from MedPC data for ROI analysis.
    """
    
    def __init__(self, data_dir, output_dir):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.parser = EnhancedMedPCDataParser(data_dir)
        self.event_data = None
        self.session_data = None
        
    def load_data(self):
        """Load and parse all MedPC files."""
        print("🔍 Loading MedPC data files...")
        
        files = self.parser.find_files('*.txt')
        print(f"📄 Found {len(files)} files")
        
        if len(files) == 0:
            print("❌ No data files found!")
            return False
        
        # Parse files
        self.parser.parse_all_files()
        
        # Create dataframes
        self.session_data = self.parser.create_dataframe()
        self.event_data = self.parser.create_time_series_dataframe()
        
        print(f"✅ Loaded {len(self.session_data)} sessions")
        print(f"✅ Extracted {len(self.event_data)} events")
        
        return True
    
    def classify_lever_press_events(self):
        """
        Classify each lever press event into categories for ROI analysis.
        
        Categories:
        - rewarded_active: Code 2, followed by reinforcement
        - timeout_active: Code 21, during timeout period (no reinforcement)
        - unrewarded_active: Code 2, but NOT followed by reinforcement
        - inactive: Codes 1, 20, 23
        """
        print("\n🏷️  Classifying lever press events...")
        
        # Filter for lever press events only
        lever_events = self.event_data.filter(
            pl.col('response_code').is_in([1, 2, 20, 21, 23])
        ).sort(['filename', 'time_seconds'])
        
        # Add event classification
        classified_events = []
        
        for session in lever_events['filename'].unique():
            session_events = lever_events.filter(pl.col('filename') == session)
            session_all_events = self.event_data.filter(pl.col('filename') == session)
            
            for row in session_events.iter_rows(named=True):
                event_time = row['time_seconds']
                response_code = row['response_code']
                
                # Initialize classification
                event_class = 'unknown'
                is_rewarded = False
                next_event_type = None
                time_to_next_event = None
                next_head_entry_latency = None
                
                # Classify based on response code
                if response_code in [1, 20, 23]:
                    # Inactive lever press
                    event_class = 'inactive_press'
                    
                elif response_code == 21:
                    # Timeout active lever press (no reward possible)
                    event_class = 'timeout_active_press'
                    
                elif response_code == 2:
                    # Active lever press - check if followed by code 4 (rewarded marker)
                    # In the MedPC program, code 4 is logged immediately after code 2 if rewarded
                    # Look for code 4 within next 0.1 seconds (should be immediate)
                    future_events = session_all_events.filter(
                        (pl.col('time_seconds') >= event_time) &
                        (pl.col('time_seconds') <= event_time + 0.1)
                    )
                    
                    rewarded_marker = future_events.filter(
                        pl.col('response_code') == 4  # Code 4 = Rewarded left lever press
                    )
                    
                    if len(rewarded_marker) > 0:
                        event_class = 'rewarded_active_press'
                        is_rewarded = True
                    else:
                        # Code 2 without code 4 = unrewarded (rare during SelfAdmin)
                        # Common during EXT phase
                        event_class = 'unrewarded_active_press'
                
                # Find the next event after this lever press
                next_events = session_all_events.filter(
                    pl.col('time_seconds') > event_time
                ).sort('time_seconds')
                
                if len(next_events) > 0:
                    next_event = next_events[0]
                    next_event_type = next_event['response_type'][0]
                    time_to_next_event = next_event['time_seconds'][0] - event_time
                
                # Find next head entry specifically
                next_head_entries = session_all_events.filter(
                    (pl.col('time_seconds') > event_time) &
                    (pl.col('response_code') == 6)
                ).sort('time_seconds')
                
                if len(next_head_entries) > 0:
                    next_head_entry_latency = next_head_entries[0]['time_seconds'] - event_time
                
                # Store classified event
                classified_events.append({
                    **row,
                    'event_class': event_class,
                    'is_rewarded': is_rewarded,
                    'next_event_type': next_event_type,
                    'time_to_next_event': time_to_next_event,
                    'next_head_entry_latency': next_head_entry_latency
                })
        
        classified_df = pl.DataFrame(classified_events)
        
        # Print summary
        print("\n📊 Event Classification Summary:")
        print("=" * 50)
        summary = classified_df.group_by('event_class').agg([
            pl.count('event_class').alias('count')
        ]).sort('count', descending=True)
        
        for row in summary.iter_rows(named=True):
            print(f"  {row['event_class']}: {row['count']} events")
        
        return classified_df
    
    def extract_post_press_sequences(self, classified_events, time_window=10):
        """
        Extract behavioral sequences following each lever press.
        
        Parameters:
        -----------
        classified_events : pl.DataFrame
            DataFrame with classified lever press events
        time_window : float
            Time window (seconds) to look ahead after each press
            
        Returns:
        --------
        pl.DataFrame with post-press sequences
        """
        print(f"\n🔄 Extracting {time_window}s post-press sequences...")
        
        sequences = []
        
        for event_row in classified_events.iter_rows(named=True):
            event_time = event_row['time_seconds']
            session = event_row['filename']
            
            # Get all events in the time window after this press
            session_events = self.event_data.filter(pl.col('filename') == session)
            
            post_press_events = session_events.filter(
                (pl.col('time_seconds') > event_time) &
                (pl.col('time_seconds') <= event_time + time_window)
            ).sort('time_seconds')
            
            # Count different event types in the window
            head_entries = len(post_press_events.filter(pl.col('response_code') == 6))
            active_presses = len(post_press_events.filter(pl.col('response_code').is_in([2, 21])))
            inactive_presses = len(post_press_events.filter(pl.col('response_code').is_in([1, 20, 23])))
            
            # Create sequence signature
            event_sequence = []
            for evt in post_press_events.iter_rows(named=True):
                event_sequence.append({
                    'response_type': evt['response_type'],
                    'latency': evt['time_seconds'] - event_time
                })
            
            sequences.append({
                'subject': event_row['subject'],
                'subject_original': event_row['subject_original'],
                'phase': event_row['phase'],
                'filename': event_row['filename'],
                'press_time': event_time,
                'event_class': event_row['event_class'],
                'is_rewarded': event_row['is_rewarded'],
                'post_press_head_entries': head_entries,
                'post_press_active_presses': active_presses,
                'post_press_inactive_presses': inactive_presses,
                'first_head_entry_latency': event_row['next_head_entry_latency'],
                'sequence_length': len(event_sequence),
                'event_sequence': event_sequence  # List of events
            })
        
        return pl.DataFrame(sequences)
    
    def create_roi_summary_by_event_type(self, sequences_df):
        """
        Summarize post-press behavior by event type for ROI analysis.
        """
        print("\n📈 Creating ROI summary by event type...")
        
        # Group by event class and calculate means
        summary = sequences_df.group_by(['phase', 'event_class']).agg([
            pl.count('press_time').alias('n_events'),
            pl.mean('post_press_head_entries').alias('mean_head_entries'),
            pl.std('post_press_head_entries').alias('std_head_entries'),
            pl.mean('post_press_active_presses').alias('mean_active_presses'),
            pl.std('post_press_active_presses').alias('std_active_presses'),
            pl.mean('post_press_inactive_presses').alias('mean_inactive_presses'),
            pl.std('post_press_inactive_presses').alias('std_inactive_presses'),
            pl.mean('first_head_entry_latency').alias('mean_head_entry_latency'),
            pl.std('first_head_entry_latency').alias('std_head_entry_latency'),
        ]).sort(['phase', 'event_class'])
        
        return summary
    
    def save_results(self, classified_events, sequences_df, summary_df):
        """Save all results to CSV files."""
        print(f"\n💾 Saving results to {self.output_dir}...")
        
        # Save classified events (no list columns, should be fine)
        try:
            classified_events.write_csv(self.output_dir / 'classified_lever_events.csv')
            print(f"  ✓ Saved classified_lever_events.csv")
        except Exception as e:
            print(f"  ⚠️  Error saving classified events: {e}")
            # Try saving without problematic columns
            safe_cols = [col for col in classified_events.columns 
                        if classified_events[col].dtype != pl.Object]
            classified_events.select(safe_cols).write_csv(
                self.output_dir / 'classified_lever_events.csv'
            )
            print(f"  ✓ Saved classified_lever_events.csv (simplified)")
        
        # Save sequences (exclude the list column for CSV)
        sequences_simple = sequences_df.select([
            col for col in sequences_df.columns if col != 'event_sequence'
        ])
        try:
            sequences_simple.write_csv(self.output_dir / 'post_press_sequences.csv')
            print(f"  ✓ Saved post_press_sequences.csv")
        except Exception as e:
            print(f"  ⚠️  Error saving sequences: {e}")
            # Try with only non-object columns
            safe_cols = [col for col in sequences_simple.columns 
                        if sequences_simple[col].dtype != pl.Object]
            sequences_simple.select(safe_cols).write_csv(
                self.output_dir / 'post_press_sequences.csv'
            )
            print(f"  ✓ Saved post_press_sequences.csv (simplified)")
        
        # Save summary
        try:
            summary_df.write_csv(self.output_dir / 'roi_summary_by_event_type.csv')
            print(f"  ✓ Saved roi_summary_by_event_type.csv")
        except Exception as e:
            print(f"  ⚠️  Error saving summary: {e}")
            # Try with only non-object columns
            safe_cols = [col for col in summary_df.columns 
                        if summary_df[col].dtype != pl.Object]
            if safe_cols:
                summary_df.select(safe_cols).write_csv(
                    self.output_dir / 'roi_summary_by_event_type.csv'
                )
                print(f"  ✓ Saved roi_summary_by_event_type.csv (simplified)")
        
        # Create a detailed text report
        self.create_text_report(summary_df)
        
        print("\n✅ All results saved successfully!")
    
    def create_text_report(self, summary_df):
        """Create a human-readable text report."""
        report_path = self.output_dir / 'roi_analysis_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("MedPC ROI EVENT ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("This report summarizes behavioral sequences following different\n")
            f.write("types of lever press events for ROI analysis.\n\n")
            
            f.write("Event Types:\n")
            f.write("  • rewarded_active_press: Active press followed by reinforcement\n")
            f.write("  • timeout_active_press: Active press during timeout (no reward)\n")
            f.write("  • unrewarded_active_press: Active press NOT followed by reward\n")
            f.write("  • inactive_press: Inactive lever press\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("SUMMARY BY EVENT TYPE\n")
            f.write("-" * 70 + "\n\n")
            
            for phase in summary_df['phase'].unique():
                f.write(f"\nPhase: {phase}\n")
                f.write("=" * 50 + "\n")
                
                phase_data = summary_df.filter(pl.col('phase') == phase)
                
                for row in phase_data.iter_rows(named=True):
                    f.write(f"\n{row['event_class'].upper()}\n")
                    f.write(f"  N events: {row['n_events']}\n")
                    
                    # Helper function to format values that might be None
                    def format_value(value, default="N/A"):
                        if value is None or (isinstance(value, float) and value != value):  # Check for NaN
                            return default
                        try:
                            return f"{value:.2f}"
                        except:
                            return default
                    
                    mean_he = format_value(row.get('mean_head_entries'))
                    std_he = format_value(row.get('std_head_entries'))
                    f.write(f"  Head entries (10s): {mean_he} ± {std_he}\n")
                    
                    mean_lat = format_value(row.get('mean_head_entry_latency'))
                    std_lat = format_value(row.get('std_head_entry_latency'))
                    f.write(f"  Head entry latency: {mean_lat} ± {std_lat} sec\n")
                    
                    mean_act = format_value(row.get('mean_active_presses'))
                    std_act = format_value(row.get('std_active_presses'))
                    f.write(f"  Active presses (10s): {mean_act} ± {std_act}\n")
                    
                    mean_inact = format_value(row.get('mean_inactive_presses'))
                    std_inact = format_value(row.get('std_inactive_presses'))
                    f.write(f"  Inactive presses (10s): {mean_inact} ± {std_inact}\n")
        
        print(f"  ✓ Saved roi_analysis_report.txt")


def main():
    parser = argparse.ArgumentParser(
        description='Extract event-triggered sequences from MedPC data for ROI analysis'
    )
    parser.add_argument('--data_dir', default='./data',
                       help='Directory containing MedPC data files')
    parser.add_argument('--output_dir', default='./roi_analysis',
                       help='Directory to save analysis outputs')
    parser.add_argument('--time_window', type=float, default=10.0,
                       help='Time window (seconds) to analyze after each press')
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = MedPCROIEventExtractor(args.data_dir, args.output_dir)
    
    # Load data
    if not extractor.load_data():
        return 1
    
    # Classify events
    classified_events = extractor.classify_lever_press_events()
    
    # Extract post-press sequences
    sequences = extractor.extract_post_press_sequences(
        classified_events, 
        time_window=args.time_window
    )
    
    # Create summary
    summary = extractor.create_roi_summary_by_event_type(sequences)
    
    # Save results
    extractor.save_results(classified_events, sequences, summary)
    
    print("\n" + "=" * 70)
    print("✅ ROI event extraction complete!")
    print("=" * 70)
    print(f"\nResults saved to: {args.output_dir}")
    print("\nNext steps for ROI analysis:")
    print("  1. Use 'classified_lever_events.csv' to align with video timestamps")
    print("  2. Use 'post_press_sequences.csv' to analyze behavioral patterns")
    print("  3. Compare rewarded vs. timeout presses in your ROI software")
    
    return 0


if __name__ == '__main__':
    exit(main())