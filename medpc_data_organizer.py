#!/usr/bin/env python
"""
MedPC Data Organizer

Comprehensive tool for organizing MedPC behavioral data by:
- Experimental phase (SelfAdmin, EXT, REI)
- Time windows (e.g., first 30 min, 30-60 min, etc.)
- Subject
- Session

Provides easy-to-manipulate data structures for subsequent analysis.

Usage:
    python medpc_data_organizer.py --data_dir ./data --output_dir ./organized_data
"""

import polars as pl
import numpy as np
from pathlib import Path
import argparse
import re
from datetime import datetime
from typing import Dict, List, Tuple
import json


class MedPCDataOrganizer:
    """
    Organize MedPC output files into structured, analysis-ready format.
    """
    
    # Response code definitions for self-administration paradigm
    RESPONSE_CODES = {
        1: 'inactive_lever',      # Inactive (right) lever press
        2: 'active_lever',        # Active (left) lever press - with Code 4 marker
        4: 'rewarded_marker',     # Marker for rewarded press (follows Code 2)
        5: 'reinforcer_on',       # Reinforcer delivery start
        6: 'head_entry',          # Magazine head entry
        8: 'cue_light_on',        # Cue light on
        10: 'tone_off',           # Tone off
        11: 'house_light_on',     # House light on
        12: 'house_light_off',    # House light off
        13: 'tone_on',            # Tone on
        14: 'tone_off',           # Tone off
        17: 'reinforcer_off',     # Reinforcer delivery end
        18: 'drug_available',     # Timeout end / drug available
        19: 'drug_available',     # Drug available / session ready
        20: 'inactive_lever',     # Inactive lever (alt code)
        21: 'timeout_active',     # Active lever during timeout (NOT reinforced)
        22: 'cue_light_off',      # Cue light off
        23: 'inactive_lever',     # Inactive lever (alt code)
        24: 'timeout_start',      # Timeout period starts
        25: 'active_lever',       # Active lever - reinforced (no Code 4)
        26: 'house_light_on',     # House light on (alt code)
        27: 'house_light_off',    # House light off (alt code)
        30: 'active_lever',       # Active lever - reinforced (no Code 4)
        100: 'session_end',       # Session termination
    }
    
    def __init__(self, data_dir, output_dir=None):
        """
        Initialize the organizer.
        
        Parameters:
        -----------
        data_dir : str or Path
            Directory containing MedPC data files
        output_dir : str or Path
            Directory to save organized outputs (default: data_dir/organized)
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / 'organized'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.raw_files = []
        self.parsed_sessions = []
        self.organized_df = None
        self.time_series_df = None
        
    def find_files(self, pattern="*.txt"):
        """
        Find all MedPC data files in the directory.
        Handles both standard files and files starting with !
        
        Returns:
        --------
        List of file paths
        """
        # Look for files starting with !
        exclamation_files = list(self.data_dir.glob("!*"))
        
        processed_files = []
        
        if exclamation_files:
            print(f"📁 Found {len(exclamation_files)} files starting with '!'")
            
            # Create temp directory for processed files
            temp_dir = self.data_dir / "temp_processed"
            temp_dir.mkdir(exist_ok=True)
            
            for file_path in exclamation_files:
                # Convert format from !YYYY-MM-DD_HHhMMm.Subject XX to YYYY-MM-DD_HHhMMm_Subject_XX.txt
                file_name = file_path.name[1:]  # Remove the !
                new_name = file_name.replace(".", "_") + ".txt"
                new_path = temp_dir / new_name
                
                # Copy content
                with open(file_path, 'r') as src, open(new_path, 'w') as dst:
                    dst.write(src.read())
                
                processed_files.append(new_path)
                print(f"  ✓ Preprocessed: {file_path.name} -> {new_path.name}")
        
        # Also look for regular .txt files
        regular_files = list(self.data_dir.glob(pattern))
        processed_files.extend(regular_files)
        
        self.raw_files = processed_files
        print(f"\n✅ Total files found: {len(self.raw_files)}")
        
        return self.raw_files
    
    def parse_file(self, file_path):
        """
        Parse a single MedPC data file.
        
        Returns:
        --------
        dict with parsed session data
        """
        file_path = Path(file_path)
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        session = {
            'filename': file_path.name,
            'filepath': str(file_path),
        }
        
        # Extract metadata
        session['start_date'] = self._extract_field(content, r'Start Date: (\d{2}/\d{2}/\d{2})')
        session['end_date'] = self._extract_field(content, r'End Date: (\d{2}/\d{2}/\d{2})')
        session['start_time'] = self._extract_field(content, r'Start Time:\s*(\d{1,2}:\d{2}:\d{2})')
        session['end_time'] = self._extract_field(content, r'End Time:\s*(\d{1,2}:\d{2}:\d{2})')
        
        # Extract subject ID
        subject_match = re.search(r'Subject:\s*([A-Za-z0-9]+)', content)
        if subject_match:
            session['subject'] = subject_match.group(1)
        else:
            # Try from filename
            filename_match = re.search(r'Subject[_\s]+([A-Za-z0-9]+)', file_path.name)
            session['subject'] = filename_match.group(1) if filename_match else 'Unknown'
        
        # Extract experiment info
        session['experiment'] = self._extract_field(content, r'Experiment: (\w+)')
        session['group'] = self._extract_field(content, r'Group: (\w+)')
        session['box'] = self._extract_field(content, r'Box: (\d+)', convert_to_int=True)
        
        # Extract MSN and determine phase
        msn = self._extract_field(content, r'MSN: (.+)')
        session['msn'] = msn
        session['phase'] = self._determine_phase(msn, file_path.name)
        
        # Extract single-value variables (F-Z)
        for letter in 'FGHIJKLMNOPQRSTUVWXYZ':
            pattern = rf'{letter}:\s+([\d.]+)'
            match = re.search(pattern, content)
            if match:
                session[f'{letter}_value'] = float(match.group(1))
        
        # Extract array data (T and E arrays are most important)
        session['timestamps'] = self._extract_array(content, 'T')
        session['response_codes'] = self._extract_array(content, 'E')
        
        # Also extract other arrays
        for letter in 'ABCD':
            session[f'{letter}_array'] = self._extract_array(content, letter)
        
        return session
    
    def _extract_field(self, content, pattern, convert_to_int=False):
        """Helper to extract a field using regex."""
        match = re.search(pattern, content)
        if match:
            value = match.group(1).strip()
            return int(value) if convert_to_int else value
        return None
    
    def _determine_phase(self, msn, filename):
        """
        Determine experimental phase from MSN or filename.
        
        Returns:
        --------
        'SelfAdmin', 'EXT', or 'REI'
        """
        msn_upper = msn.upper() if msn else ''
        filename_upper = filename.upper()
        
        # Check MSN first
        if 'REI' in msn_upper or 'REINSTATE' in msn_upper:
            return 'REI'
        elif 'EXT' in msn_upper:
            return 'EXT'
        elif 'SELFADMIN' in msn_upper or 'SA' in msn_upper:
            return 'SelfAdmin'
        
        # Check filename
        if 'REI' in filename_upper or 'REINSTATE' in filename_upper:
            return 'REI'
        elif 'EXT' in filename_upper:
            return 'EXT'
        elif 'SELFADMIN' in filename_upper or 'SA' in filename_upper:
            return 'SelfAdmin'
        
        return 'Unknown'
    
    def _extract_array(self, content, letter):
        """
        Extract array data (e.g., T: timestamps, E: response codes).
        
        Returns:
        --------
        List of float values
        """
        pattern = rf'{letter}:([\s\d.:+-]+?)(?=[A-Z]:|$)'
        match = re.search(pattern, content, re.DOTALL)
        
        if not match:
            return []
        
        array_text = match.group(1).strip()
        values = []
        
        # Parse line by line
        for line in array_text.split('\n'):
            if line.strip():
                # Split by colon to separate index from values
                if ':' in line:
                    # Everything after the colon is the actual data
                    _, data_part = line.split(':', 1)
                else:
                    data_part = line
                
                # Extract numbers, handling negative values
                numbers = re.findall(r'-?\d+\.?\d*', data_part)
                for num_str in numbers:
                    try:
                        val = float(num_str)
                        # Filter out sentinel values
                        if val != -987.987 and val >= 0:
                            values.append(val)
                    except ValueError:
                        continue
        
        return values
    
    def parse_all_files(self):
        """Parse all found files."""
        print(f"\n📖 Parsing {len(self.raw_files)} files...")
        
        for file_path in self.raw_files:
            try:
                session = self.parse_file(file_path)
                self.parsed_sessions.append(session)
                print(f"  ✓ {session['subject']} - {session['phase']} - {file_path.name}")
            except Exception as e:
                print(f"  ✗ Error parsing {file_path.name}: {e}")
        
        print(f"\n✅ Successfully parsed {len(self.parsed_sessions)} sessions")
        return self.parsed_sessions
    
    def create_session_summary(self):
        """
        Create a summary DataFrame with one row per session.
        
        Returns:
        --------
        Polars DataFrame
        """
        print("\n📊 Creating session summary...")
        
        rows = []
        
        for session in self.parsed_sessions:
            # Count events
            response_codes = session['response_codes']
            
            # Active lever presses (codes 2, 21, 25, 30)
            active_presses = sum(1 for code in response_codes if code in [2, 21, 25, 30])
            
            # Active presses during session (codes 2, 25, 30, excluding timeout)
            active_nontimeout = sum(1 for code in response_codes if code in [2, 25, 30])
            
            # Active presses during timeout (code 21)
            active_timeout = sum(1 for code in response_codes if code == 21)
            
            # Inactive lever presses (codes 1, 20, 23)
            inactive_presses = sum(1 for code in response_codes if code in [1, 20, 23])
            
            # Head entries (code 6)
            head_entries = sum(1 for code in response_codes if code == 6)
            
            # Reinforcers (from G value or count rewarded markers)
            reinforcers = session.get('G_value', 0)
            rewarded_markers = sum(1 for code in response_codes if code == 4)
            
            row = {
                'subject': session['subject'],
                'phase': session['phase'],
                'filename': session['filename'],
                'start_date': session['start_date'],
                'start_time': session['start_time'],
                'end_date': session['end_date'],
                'end_time': session['end_time'],
                'box': session['box'],
                'experiment': session['experiment'],
                'group': session['group'],
                'msn': session['msn'],
                'active_lever_total': active_presses,
                'active_lever_nontimeout': active_nontimeout,
                'active_lever_timeout': active_timeout,
                'inactive_lever_total': inactive_presses,
                'head_entries': head_entries,
                'reinforcers': reinforcers,
                'rewarded_markers': rewarded_markers,
                'session_duration_sec': max(session['timestamps']) if session['timestamps'] else 0,
            }
            
            rows.append(row)
        
        df = pl.DataFrame(rows)
        self.organized_df = df
        
        print(f"  ✓ Created summary with {len(df)} sessions")
        return df
    
    def create_time_series(self):
        """
        Create long-format time series DataFrame with individual events.
        
        Returns:
        --------
        Polars DataFrame
        """
        print("\n📈 Creating time series data...")
        
        rows = []
        
        for session in self.parsed_sessions:
            timestamps = session['timestamps']
            response_codes = session['response_codes']
            
            # Ensure equal lengths
            min_len = min(len(timestamps), len(response_codes))
            
            for i in range(min_len):
                time_sec = timestamps[i]
                code = response_codes[i]
                
                # Get response type
                response_type = self.RESPONSE_CODES.get(code, f'unknown_{int(code)}')
                
                # Determine if this is an active lever press
                is_active_lever = code in [2, 21, 25, 30]
                is_inactive_lever = code in [1, 20, 23]
                is_head_entry = code == 6
                
                row = {
                    'subject': session['subject'],
                    'phase': session['phase'],
                    'filename': session['filename'],
                    'time_seconds': time_sec,
                    'time_minutes': time_sec / 60.0,
                    'response_code': int(code),
                    'response_type': response_type,
                    'is_active_lever': is_active_lever,
                    'is_inactive_lever': is_inactive_lever,
                    'is_head_entry': is_head_entry,
                }
                
                rows.append(row)
        
        df = pl.DataFrame(rows)
        self.time_series_df = df
        
        print(f"  ✓ Created time series with {len(df)} events")
        return df
    
    def get_time_window(self, start_min=0, end_min=30, phase=None, subjects=None):
        """
        Extract data for a specific time window.
        
        Parameters:
        -----------
        start_min : float
            Start time in minutes
        end_min : float
            End time in minutes
        phase : str or list
            Phase(s) to include ('SelfAdmin', 'EXT', 'REI')
        subjects : list
            Subject IDs to include
            
        Returns:
        --------
        Polars DataFrame
        """
        if self.time_series_df is None:
            raise ValueError("Time series not created. Call create_time_series() first.")
        
        # Filter by time
        df = self.time_series_df.filter(
            (pl.col('time_minutes') >= start_min) & 
            (pl.col('time_minutes') < end_min)
        )
        
        # Filter by phase
        if phase is not None:
            if isinstance(phase, str):
                phase = [phase]
            df = df.filter(pl.col('phase').is_in(phase))
        
        # Filter by subjects
        if subjects is not None:
            df = df.filter(pl.col('subject').is_in(subjects))
        
        return df
    
    def summarize_by_time_window(self, windows=[(0, 30), (30, 60), (0, 120)]):
        """
        Summarize lever presses by time window for each session.
        
        Parameters:
        -----------
        windows : list of tuples
            Time windows as (start_min, end_min)
            
        Returns:
        --------
        Polars DataFrame
        """
        print(f"\n⏱️  Summarizing data across {len(windows)} time windows...")
        
        summaries = []
        
        for start_min, end_min in windows:
            window_data = self.get_time_window(start_min, end_min)
            
            summary = window_data.group_by(['subject', 'phase', 'filename']).agg([
                pl.col('is_active_lever').sum().alias('active_lever_presses'),
                pl.col('is_inactive_lever').sum().alias('inactive_lever_presses'),
                pl.col('is_head_entry').sum().alias('head_entries'),
            ])
            
            # Add window info
            summary = summary.with_columns([
                pl.lit(f"{start_min}-{end_min}min").alias('time_window'),
                pl.lit(start_min).alias('start_min'),
                pl.lit(end_min).alias('end_min'),
            ])
            
            summaries.append(summary)
        
        combined = pl.concat(summaries)
        
        print(f"  ✓ Created time window summary")
        return combined
    
    def save_organized_data(self):
        """Save all organized data to CSV files."""
        print(f"\n💾 Saving organized data to {self.output_dir}...")
        
        # Save session summary
        if self.organized_df is not None:
            path = self.output_dir / 'session_summary.csv'
            self.organized_df.write_csv(path)
            print(f"  ✓ Saved: session_summary.csv")
        
        # Save time series
        if self.time_series_df is not None:
            path = self.output_dir / 'time_series_all_events.csv'
            self.time_series_df.write_csv(path)
            print(f"  ✓ Saved: time_series_all_events.csv")
        
        # Save time window summaries
        window_summary = self.summarize_by_time_window(
            windows=[(0, 30), (30, 60), (60, 90), (90, 120), (0, 60), (0, 120)]
        )
        path = self.output_dir / 'time_window_summary.csv'
        window_summary.write_csv(path)
        print(f"  ✓ Saved: time_window_summary.csv")
        
        # Save by phase
        for phase in ['SelfAdmin', 'EXT', 'REI']:
            phase_data = self.time_series_df.filter(pl.col('phase') == phase)
            if len(phase_data) > 0:
                path = self.output_dir / f'time_series_{phase}.csv'
                phase_data.write_csv(path)
                print(f"  ✓ Saved: time_series_{phase}.csv")
        
        # Create a metadata file
        self._save_metadata()
        
        print(f"\n✅ All data saved to: {self.output_dir}")
    
    def _save_metadata(self):
        """Save metadata about the dataset."""
        metadata = {
            'created_at': datetime.now().isoformat(),
            'num_sessions': len(self.parsed_sessions),
            'num_events': len(self.time_series_df) if self.time_series_df is not None else 0,
            'phases': list(self.organized_df['phase'].unique()) if self.organized_df is not None else [],
            'subjects': list(self.organized_df['subject'].unique()) if self.organized_df is not None else [],
            'response_code_definitions': self.RESPONSE_CODES,
        }
        
        path = self.output_dir / 'metadata.json'
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✓ Saved: metadata.json")
    
    def print_summary_stats(self):
        """Print summary statistics about the dataset."""
        if self.organized_df is None:
            print("❌ No data to summarize. Run organize() first.")
            return
        
        print("\n" + "="*70)
        print("📊 DATASET SUMMARY")
        print("="*70)
        
        print(f"\n📁 Total Sessions: {len(self.organized_df)}")
        
        print(f"\n👤 Subjects:")
        for subject in sorted(self.organized_df['subject'].unique()):
            count = len(self.organized_df.filter(pl.col('subject') == subject))
            print(f"  • {subject}: {count} sessions")
        
        print(f"\n🧪 Phases:")
        for phase in ['SelfAdmin', 'EXT', 'REI']:
            count = len(self.organized_df.filter(pl.col('phase') == phase))
            if count > 0:
                print(f"  • {phase}: {count} sessions")
        
        print(f"\n📈 Overall Statistics:")
        print(f"  • Total active lever presses: {self.organized_df['active_lever_total'].sum()}")
        print(f"  • Total inactive lever presses: {self.organized_df['inactive_lever_total'].sum()}")
        print(f"  • Total head entries: {self.organized_df['head_entries'].sum()}")
        print(f"  • Total reinforcers: {self.organized_df['reinforcers'].sum()}")
        
        print("\n" + "="*70)
    
    def organize(self):
        """
        Run the complete organization pipeline.
        """
        print("\n" + "="*70)
        print("🚀 MEDPC DATA ORGANIZER")
        print("="*70)
        
        # Step 1: Find files
        self.find_files()
        
        # Step 2: Parse all files
        self.parse_all_files()
        
        # Step 3: Create organized structures
        self.create_session_summary()
        self.create_time_series()
        
        # Step 4: Save everything
        self.save_organized_data()
        
        # Step 5: Print summary
        self.print_summary_stats()
        
        print("\n✅ Organization complete!")
        print(f"📂 Output location: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Organize MedPC data files by phase and time for easy analysis'
    )
    parser.add_argument('--data_dir', default='./data',
                       help='Directory containing MedPC data files')
    parser.add_argument('--output_dir', default=None,
                       help='Directory to save organized outputs (default: data_dir/organized)')
    
    args = parser.parse_args()
    
    # Create organizer and run
    organizer = MedPCDataOrganizer(args.data_dir, args.output_dir)
    organizer.organize()
    
    print("\n" + "="*70)
    print("💡 NEXT STEPS")
    print("="*70)
    print("\nYou can now easily analyze your data:")
    print("  1. session_summary.csv - One row per session with totals")
    print("  2. time_series_*.csv - Individual events with timestamps")
    print("  3. time_window_summary.csv - Binned data by time periods")
    print("\nExample analyses:")
    print("  • Active presses in first 30 min by phase")
    print("  • Compare EXT vs SelfAdmin behavior over time")
    print("  • Track individual subject progression")
    print("="*70)


if __name__ == '__main__':
    main()
