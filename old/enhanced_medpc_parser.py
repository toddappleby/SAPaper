import os
import re
import polars as pl
import numpy as np
from pathlib import Path
import glob

class EnhancedMedPCDataParser:
    def __init__(self, base_dir=None):
        """
        Enhanced MedPC data parser that handles non-numerical subject IDs.
        
        Parameters:
        -----------
        base_dir : str or Path
            Base directory containing MedPC data files
        """
        self.base_dir = Path(base_dir) if base_dir else None
        self.data_files = []
        self.parsed_data = {}
        self.combined_df = None
        self.subject_mapping = {}  # Maps original subject IDs to standardized ones
        
    def find_files(self, pattern="*.txt"):
        """
        Find all MedPC data files matching the pattern in the base directory.
        Handles both standard files and files starting with !
        
        Parameters:
        -----------
        pattern : str
            File pattern to match
                
        Returns:
        --------
        list of Path objects
        """
        if not self.base_dir:
            raise ValueError("Base directory not set")
        
        # Look for files starting with !
        exclamation_files = list(self.base_dir.glob("!*"))
        
        # Create a list to store the file paths
        self.data_files = []
        
        # Process each file that starts with !
        for file_path in exclamation_files:
            file_name = file_path.name
            
            # Check if the file appears to be a MedPC data file
            if "Subject" in file_name:
                # Convert format from !YYYY-MM-DD_HHhMMm.Subject XX to 
                # YYYY-MM-DD_HHhMMm_Subject XX.txt
                new_name = file_name[1:]  # Remove the ! 
                new_name = new_name.replace(".", "_") + ".txt"
                
                # Create or use a temporary directory for processed files
                temp_dir = self.base_dir / "temp_processed"
                temp_dir.mkdir(exist_ok=True)
                
                # Create a new path for the renamed file
                new_path = temp_dir / new_name
                
                # Copy the file content (don't modify the original)
                with open(file_path, 'r') as src_file:
                    content = src_file.read()
                    with open(new_path, 'w') as dest_file:
                        dest_file.write(content)
                
                self.data_files.append(new_path)
                print(f"Preprocessed {file_path.name} -> {new_path.name}")
    
        # If no exclamation mark files found, try the original pattern
        if not self.data_files:
            self.data_files = list(self.base_dir.glob(pattern))
            
        return self.data_files
    
    def extract_subject_id(self, content, filename):
        """
        Extract subject ID from file content or filename, handling both numerical and text IDs.
        
        Parameters:
        -----------
        content : str
            File content
        filename : str
            Filename
            
        Returns:
        --------
        str: Original subject ID as found in the file
        """
        # First try to extract from file content
        subject_match = re.search(r'Subject:\s*([A-Za-z0-9]+)', content)
        if subject_match:
            return subject_match.group(1)
        
        # If not found in content, try to extract from filename
        # Look for patterns like "Subject T1", "Subject 83", etc.
        filename_match = re.search(r'Subject[_\s]+([A-Za-z0-9]+)', filename)
        if filename_match:
            return filename_match.group(1)
        
        # Last resort: look for any alphanumeric string that might be a subject ID
        # This is more speculative but might catch edge cases
        fallback_match = re.search(r'([A-Za-z]+\d+|\d+[A-Za-z]*)', filename)
        if fallback_match:
            return fallback_match.group(1)
        
        return None
    
    def standardize_subject_id(self, original_id):
        """
        Create a standardized subject ID for database consistency while preserving original.
        
        Parameters:
        -----------
        original_id : str
            Original subject ID from the file
            
        Returns:
        --------
        str: Standardized subject ID
        """
        if original_id is None:
            return None
            
        # If it's already a number, keep it as is
        if original_id.isdigit():
            return original_id
        
        # For text-based IDs like "T1", "T2", extract the number and create a mapping
        # This preserves the ability to identify subjects while making them sortable
        text_match = re.match(r'([A-Za-z]+)(\d+)', str(original_id))
        if text_match:
            prefix = text_match.group(1)
            number = text_match.group(2)
            
            # Store the mapping for later reference
            standardized = f"{prefix}{number.zfill(3)}"  # Pad with zeros for sorting
            self.subject_mapping[standardized] = original_id
            return standardized
        
        # If no pattern matches, return as is
        return str(original_id)
    
    def parse_file(self, file_path):
        """
        Parse a single MedPC data file with enhanced subject ID handling.
        
        Parameters:
        -----------
        file_path : str or Path
            Path to the MedPC data file
            
        Returns:
        --------
        dict containing parsed data
        """
        file_path = Path(file_path)
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Extract header information
        header_data = {}
        
        # Extract file name
        header_data['filename'] = file_path.name
        
        # Extract date information
        start_date_match = re.search(r'Start Date: (\d{2}/\d{2}/\d{2})', content)
        if start_date_match:
            header_data['start_date'] = start_date_match.group(1)
        
        end_date_match = re.search(r'End Date: (\d{2}/\d{2}/\d{2})', content)
        if end_date_match:
            header_data['end_date'] = end_date_match.group(1)
        
        # Enhanced subject ID extraction
        original_subject_id = self.extract_subject_id(content, file_path.name)
        if original_subject_id:
            header_data['subject_original'] = original_subject_id
            header_data['subject'] = self.standardize_subject_id(original_subject_id)
        else:
            print(f"Warning: Could not extract subject ID from {file_path.name}")
            header_data['subject_original'] = None
            header_data['subject'] = None
        
        # Extract experiment type
        experiment_match = re.search(r'Experiment: (\w+)', content)
        if experiment_match:
            header_data['experiment'] = experiment_match.group(1)
        
        # Extract group information
        group_match = re.search(r'Group: (\w+)', content)
        if group_match:
            header_data['group'] = group_match.group(1)
        
        # Extract box number
        box_match = re.search(r'Box: (\d+)', content)
        if box_match:
            header_data['box'] = int(box_match.group(1))
        
        # Extract start and end times
        start_time_match = re.search(r'Start Time: (\d{2}:\d{2}:\d{2})', content)
        if start_time_match:
            header_data['start_time'] = start_time_match.group(1)
        
        end_time_match = re.search(r'End Time: (\d{2}:\d{2}:\d{2})', content)
        if end_time_match:
            header_data['end_time'] = end_time_match.group(1)
        
        # Extract MSN (program name) with enhanced phase detection
        msn_match = re.search(r'MSN: (.+)', content)
        if msn_match:
            header_data['msn'] = msn_match.group(1).strip()
            
            # Enhanced experimental phase detection
            msn_upper = header_data['msn'].upper()
            if 'SELFADMIN' in msn_upper and 'EXT' not in msn_upper and 'REI' not in msn_upper:
                header_data['phase'] = 'SelfAdmin'
            elif 'EXT' in msn_upper:
                header_data['phase'] = 'EXT'
            elif 'REI' in msn_upper:
                header_data['phase'] = 'REI'
            else:
                # Try to infer from filename if MSN doesn't contain clear markers
                filename_upper = file_path.name.upper()
                if 'EXT' in filename_upper:
                    header_data['phase'] = 'EXT'
                elif 'REI' in filename_upper:
                    header_data['phase'] = 'REI'
                elif 'SELFADMIN' in filename_upper:
                    header_data['phase'] = 'SelfAdmin'
                else:
                    header_data['phase'] = 'Unknown'
                    print(f"Warning: Could not determine phase for {file_path.name}")
        
        # Extract data arrays
        arrays = {}
        
        # Extract single-letter variables (F-Z)
        for letter in 'FGHIJKLMNOPQRSTUVWXYZ':
            pattern = rf'{letter}:\s+([\d.]+)'
            match = re.search(pattern, content)
            if match:
                arrays[letter] = float(match.group(1))
        
        # Extract multi-dimensional arrays (A-E and T)
        for letter in 'ABCDET':
            pattern = rf'{letter}:([\s\d.:]+?)(?=[A-Z]:|$)'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                array_data = []
                array_text = match.group(1).strip()
                rows = array_text.split('\n')
                for row in rows:
                    if row.strip():
                        values = re.findall(r'(\d+):\s+([\d.\s-]+)', row)
                        if values:
                            indices, data_points = values[0]
                            # Handle negative values (like -987.987)
                            data_points = [float(dp) for dp in data_points.split() if dp.strip()]
                            array_data.extend(data_points)
                arrays[letter] = array_data
        
        # Combine all data
        data = {
            'header': header_data,
            'arrays': arrays
        }
        
        return data
    
    def parse_all_files(self):
        """
        Parse all found MedPC data files.
        
        Returns:
        --------
        dict containing parsed data for all files
        """
        if not self.data_files:
            raise ValueError("No data files found. Call find_files() first.")
        
        for file_path in self.data_files:
            try:
                parsed_data = self.parse_file(file_path)
                self.parsed_data[file_path.name] = parsed_data
                
                subject_info = parsed_data['header'].get('subject_original', 'Unknown')
                phase_info = parsed_data['header'].get('phase', 'Unknown')
                print(f"Successfully parsed {file_path.name} - Subject: {subject_info}, Phase: {phase_info}")
                
            except Exception as e:
                print(f"Error parsing {file_path.name}: {e}")
                import traceback
                traceback.print_exc()
        
        return self.parsed_data
    
    def create_dataframe(self):
        """
        Create a Polars DataFrame from the parsed data with enhanced subject handling.
        
        Returns:
        --------
        Polars DataFrame
        """
        if not self.parsed_data:
            raise ValueError("No parsed data available. Call parse_all_files() first.")
        
        rows = []
        
        for filename, data in self.parsed_data.items():
            header = data['header']
            arrays = data['arrays']
            
            # Create row for basic information
            row = {
                'filename': filename,
                'subject': header.get('subject'),
                'subject_original': header.get('subject_original'),
                'experiment': header.get('experiment'),
                'phase': header.get('phase'),
                'group': header.get('group'),
                'box': header.get('box'),
                'start_date': header.get('start_date'),
                'end_date': header.get('end_date'),
                'start_time': header.get('start_time'),
                'end_time': header.get('end_time'),
                'msn': header.get('msn'),
            }
            
            # Add single-letter variables
            for letter in 'FGHIJKLMNOPQRSTUVWXYZ':
                if letter in arrays and letter != 'T':  # Skip T as it's handled separately
                    row[f'{letter}_value'] = arrays[letter]
            
            # Process timestamp array (T) and response array (E) together
            if 'T' in arrays and 'E' in arrays:
                # Ensure t_array and e_array are lists
                t_array = arrays['T']
                e_array = arrays['E']
                
                if not isinstance(t_array, list):
                    t_array = [t_array]
                if not isinstance(e_array, list):
                    e_array = [e_array]
                
                # Filter out invalid values (like -987.987)
                valid_pairs = [(t, e) for t, e in zip(t_array, e_array) 
                              if t >= 0 and e >= 0 and t != -987.987 and e != -987.987]
                
                if valid_pairs:
                    t_array, e_array = zip(*valid_pairs)
                    t_array = list(t_array)
                    e_array = list(e_array)
                else:
                    t_array, e_array = [], []
                
                # Store these arrays as lists in the row
                row['timestamps'] = t_array
                row['responses'] = e_array
                
                # Calculate additional metrics
                # Active lever presses (response code 2 and 21)
                active_presses = sum(1 for resp in e_array if resp in [2, 21])
                row['active_lever_presses'] = active_presses
                
                # Inactive lever presses (response codes 1, 20, 23)
                inactive_presses = sum(1 for resp in e_array if resp in [1, 20, 23])
                row['inactive_lever_presses'] = inactive_presses
                
                # Head entries (response code 6)
                head_entries = sum(1 for resp in e_array if resp == 6)
                row['head_entries'] = head_entries
                
                # Reinforcers (from G value or count of reinforcement events)
                row['reinforcers'] = arrays.get('G', 0)
                
                # Calculate lever presses in time bins (e.g., 30-minute bins)
                if t_array:
                    bin_size = 30 * 60  # 30 minutes in seconds
                    max_time = max(t_array)
                    num_bins = int(np.ceil(max_time / bin_size))
                    
                    active_bins = [0] * num_bins
                    inactive_bins = [0] * num_bins
                    
                    for i, (time, resp) in enumerate(zip(t_array, e_array)):
                        bin_idx = int(time // bin_size)
                        if bin_idx < num_bins:
                            if resp in [2, 21]:  # Active lever
                                active_bins[bin_idx] += 1
                            elif resp in [1, 20, 23]:  # Inactive lever
                                inactive_bins[bin_idx] += 1
                    
                    row['active_30min_bins'] = active_bins
                    row['inactive_30min_bins'] = inactive_bins
                else:
                    row['active_30min_bins'] = []
                    row['inactive_30min_bins'] = []
            
            rows.append(row)
        
        # Create Polars DataFrame
        df = pl.DataFrame(rows)
        self.combined_df = df
        return df
    
    def create_time_series_dataframe(self):
        """
        Create a long-format time series DataFrame with individual events.
        
        Returns:
        --------
        Polars DataFrame with time series data
        """
        if self.combined_df is None:
            raise ValueError("No combined DataFrame available. Call create_dataframe() first.")
        
        results = []
        
        # Convert Polars DataFrame to dictionaries for easier processing
        for row in self.combined_df.iter_rows(named=True):
            subject = row['subject']
            subject_original = row['subject_original']
            phase = row['phase']
            group = row['group']
            filename = row['filename']
            
            timestamps = row.get('timestamps', [])
            responses = row.get('responses', [])
            
            # Ensure they are lists
            if not isinstance(timestamps, list):
                timestamps = [timestamps] if timestamps is not None else []
            if not isinstance(responses, list):
                responses = [responses] if responses is not None else []
            
            # Make sure arrays are the same length
            min_length = min(len(timestamps), len(responses))
            timestamps = timestamps[:min_length]
            responses = responses[:min_length]
            
            for time, resp in zip(timestamps, responses):
                # Enhanced response type classification
                if resp in [1, 20, 23]:
                    response_type = 'inactive_lever'
                elif resp in [2, 21]:  # Active lever press (normal and during timeout)
                    response_type = 'active_lever'
                elif resp == 6:
                    response_type = 'head_entry'
                elif resp in [5, 17]:  # Reinforcement delivery
                    response_type = 'reinforced'
                elif resp in [11, 12]:  # House light on/off
                    response_type = 'house_light'
                elif resp in [13, 14]:  # Tone on/off
                    response_type = 'tone'
                elif resp == 24:  # Timeout starts
                    response_type = 'timeout_start'
                elif resp == 19:  # Drug available/timeout ends
                    response_type = 'drug_available'
                elif resp == 100:  # Session termination
                    response_type = 'session_end'
                else:
                    response_type = f'other_{int(resp)}'
                
                results.append({
                    'subject': subject,
                    'subject_original': subject_original,
                    'phase': phase,
                    'group': group,
                    'filename': filename,
                    'time_seconds': time,
                    'time_minutes': time / 60,
                    'response_code': resp,
                    'response_type': response_type
                })
        
        return pl.DataFrame(results)
    
    def get_time_window_data(self, start_min=0, end_min=30, subjects=None, phases=None):
        """
        Extract data for a specific time window.
        
        Parameters:
        -----------
        start_min : int
            Start time in minutes
        end_min : int
            End time in minutes
        subjects : list
            List of subjects to include (None for all)
        phases : list
            List of phases to include (None for all)
            
        Returns:
        --------
        Polars DataFrame with filtered data
        """
        # Get time series data
        time_series_df = self.create_time_series_dataframe()
        
        # Filter by time window
        start_sec = start_min * 60
        end_sec = end_min * 60
        
        filtered_df = time_series_df.filter(
            (pl.col('time_seconds') >= start_sec) & 
            (pl.col('time_seconds') < end_sec)
        )
        
        # Filter by subjects if specified
        if subjects is not None:
            # Handle both original and standardized subject IDs
            subject_filter = (
                pl.col('subject').is_in(subjects) | 
                pl.col('subject_original').is_in(subjects)
            )
            filtered_df = filtered_df.filter(subject_filter)
        
        # Filter by phases if specified
        if phases is not None:
            filtered_df = filtered_df.filter(pl.col('phase').is_in(phases))
        
        return filtered_df
    
    def get_subject_summary(self, time_windows=[(0, 30), (30, 60), (0, 60)]):
        """
        Create a summary of lever presses for each subject across different time windows.
        
        Parameters:
        -----------
        time_windows : list of tuples
            List of (start_min, end_min) time windows to analyze
            
        Returns:
        --------
        Polars DataFrame with summary data
        """
        summaries = []
        
        for start_min, end_min in time_windows:
            window_data = self.get_time_window_data(start_min, end_min)
            
            # Group by subject and phase
            summary = window_data.group_by(['subject', 'subject_original', 'phase', 'filename']).agg([
                pl.col('response_type').filter(pl.col('response_type') == 'active_lever').count().alias('active_lever_count'),
                pl.col('response_type').filter(pl.col('response_type') == 'inactive_lever').count().alias('inactive_lever_count'),
                pl.col('response_type').filter(pl.col('response_type') == 'head_entry').count().alias('head_entry_count'),
                pl.col('response_type').filter(pl.col('response_type') == 'reinforced').count().alias('reinforcer_count')
            ])
            
            # Add time window info
            summary = summary.with_columns([
                pl.lit(f"{start_min}-{end_min}min").alias('time_window'),
                pl.lit(start_min).alias('start_min'),
                pl.lit(end_min).alias('end_min')
            ])
            
            summaries.append(summary)
        
        # Combine all summaries
        if summaries:
            return pl.concat(summaries)
        else:
            return pl.DataFrame()
    
    def save_enhanced_data(self, output_path):
        """
        Save processed data with enhanced subject ID handling.
        
        Parameters:
        -----------
        output_path : str or Path
            Directory to save output files
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save subject mapping
        if self.subject_mapping:
            mapping_df = pl.DataFrame([
                {'standardized_id': k, 'original_id': v} 
                for k, v in self.subject_mapping.items()
            ])
            mapping_df.write_csv(output_path / 'subject_id_mapping.csv')
            print(f"Saved subject ID mapping to {output_path / 'subject_id_mapping.csv'}")
        
        if self.combined_df is not None:
            # Save main summary (exclude array columns)
            array_cols = ['timestamps', 'responses', 'active_30min_bins', 'inactive_30min_bins']
            basic_cols = [col for col in self.combined_df.columns if col not in array_cols]
            
            basic_df = self.combined_df.select(basic_cols)
            basic_df.write_csv(output_path / 'session_summary.csv')
            print(f"Saved session summary to {output_path / 'session_summary.csv'}")
            
            # Save time series data
            time_series_df = self.create_time_series_dataframe()
            time_series_df.write_csv(output_path / 'time_series_events.csv')
            print(f"Saved time series data to {output_path / 'time_series_events.csv'}")
            
            # Save subject summaries for different time windows
            subject_summary = self.get_subject_summary()
            if len(subject_summary) > 0:
                subject_summary.write_csv(output_path / 'subject_time_window_summary.csv')
                print(f"Saved subject summaries to {output_path / 'subject_time_window_summary.csv'}")
        
        print(f"\nAll data saved to {output_path}")
        print("\nFiles created:")
        for file in output_path.glob("*.csv"):
            print(f"  - {file.name}")

# Usage example
if __name__ == "__main__":
    # Initialize the enhanced parser
    parser = EnhancedMedPCDataParser('./data')
    
    # Find and parse files
    files = parser.find_files('*.txt')
    print(f"Found {len(files)} files")
    
    # Parse all files
    parser.parse_all_files()
    
    # Create dataframes
    df = parser.create_dataframe()
    print(f"Created dataframe with {len(df)} sessions")
    
    # Show available subjects and phases
    if len(df) > 0:
        subjects = df.select(['subject', 'subject_original']).unique()
        phases = df.select('phase').unique()
        
        print("\nSubjects found:")
        for row in subjects.iter_rows(named=True):
            print(f"  {row['subject_original']} -> {row['subject']}")
        
        print(f"\nPhases found: {phases.to_series().to_list()}")
    
    # Save processed data
    parser.save_enhanced_data('./processed_data')
    
    # Example: Get first 30 minutes of data for all subjects
    first_30min = parser.get_time_window_data(0, 30)
    print(f"\nFirst 30 minutes contains {len(first_30min)} events")