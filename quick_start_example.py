#!/usr/bin/env python
"""
Quick Start Example for MedPC Data Organization System

This script demonstrates the basic workflow for organizing and analyzing
MedPC behavioral data.

Run this after you've organized your data with medpc_data_organizer.py
"""

from medpc_analyzer import MedPCAnalyzer
import numpy as np

def main():
    print("="*70)
    print("🚀 MedPC DATA ANALYSIS - QUICK START EXAMPLE")
    print("="*70)
    
    # =========================================================================
    # STEP 1: Load organized data
    # =========================================================================
    print("\n📂 STEP 1: Loading organized data...")
    analyzer = MedPCAnalyzer('./organized_data')
    
    # =========================================================================
    # STEP 2: Explore your dataset
    # =========================================================================
    print("\n📊 STEP 2: Dataset overview...")
    analyzer.print_summary()
    
    # =========================================================================
    # STEP 3: Analyze specific time windows
    # =========================================================================
    print("\n⏱️  STEP 3: Time window analysis...")
    
    print("\n▶️  First 30 minutes:")
    first_30 = analyzer.count_active_presses(start_min=0, end_min=30)
    print(first_30)
    
    print("\n▶️  Minutes 30-60:")
    min_30_60 = analyzer.count_active_presses(start_min=30, end_min=60)
    print(min_30_60)
    
    # =========================================================================
    # STEP 4: Get lever press timestamps
    # =========================================================================
    print("\n📍 STEP 4: Lever press timing analysis...")
    
    # Get all subjects
    subjects = analyzer.metadata['subjects']
    
    for subject in subjects:
        print(f"\n▶️  Subject {subject}:")
        
        # Get active lever press times
        press_times = analyzer.get_lever_press_times(
            subject=subject, 
            lever_type='active'
        )
        
        if len(press_times) > 0:
            print(f"   • Total active presses: {len(press_times)}")
            print(f"   • First press at: {press_times['time_minutes'][0]:.2f} min")
            print(f"   • Last press at: {press_times['time_minutes'][-1]:.2f} min")
            
            # Calculate inter-press intervals
            intervals = analyzer.calculate_inter_press_intervals(subject=subject)
            if len(intervals) > 0:
                print(f"   • Mean inter-press interval: {intervals.mean():.2f} sec")
                print(f"   • Median inter-press interval: {np.median(intervals):.2f} sec")
                print(f"   • Presses per minute: {60/intervals.mean():.2f}")
    
    # =========================================================================
    # STEP 5: Compare phases (if multiple phases available)
    # =========================================================================
    print("\n🔬 STEP 5: Phase comparison...")
    
    phases = analyzer.metadata['phases']
    print(f"   Available phases: {', '.join(phases)}")
    
    if len(phases) > 1:
        print("\n▶️  Active lever presses by phase:")
        comparison = analyzer.compare_phases('active_lever_total')
        print(comparison)
        
        print("\n▶️  Reinforcers by phase:")
        comparison = analyzer.compare_phases('reinforcers')
        print(comparison)
    else:
        print("   Only one phase present - phase comparison not available")
    
    # =========================================================================
    # STEP 6: Export analysis-ready data
    # =========================================================================
    print("\n💾 STEP 6: Exporting analysis-ready data...")
    
    # Export for each phase
    for phase in phases:
        output_file = f'analysis_{phase}_first30min.csv'
        analyzer.export_for_analysis(
            output_file,
            phase=phase,
            time_window=(0, 30)
        )
    
    # =========================================================================
    # EXAMPLES: Common analysis tasks
    # =========================================================================
    print("\n" + "="*70)
    print("💡 EXAMPLE ANALYSIS TASKS")
    print("="*70)
    
    print("\n1️⃣ Get behavior progression within session:")
    for subject in subjects:
        progression = analyzer.get_session_progression(subject)
        print(f"\n   Subject {subject} progression:")
        print(progression.select(['time_window', 'active_lever_presses']))
    
    print("\n2️⃣ Filter by phase and subject:")
    if 'T2' in subjects:
        t2_data = analyzer.get_by_subject('T2')
        print(f"\n   T2 sessions: {len(t2_data['sessions'])}")
        print(f"   T2 total events: {len(t2_data['time_series'])}")
    
    print("\n3️⃣ Custom time window (e.g., 10-20 minutes):")
    custom_window = analyzer.get_time_window(start_min=10, end_min=20)
    print(f"   Events in 10-20 min window: {len(custom_window)}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)
    print("\nKey outputs generated:")
    print("  • Session summaries by phase")
    print("  • Time window breakdowns")
    print("  • Lever press timing data")
    print("  • Analysis-ready CSV files")
    print("\nNext steps:")
    print("  1. Import CSV files into your statistical software")
    print("  2. Use the time_series files for detailed temporal analysis")
    print("  3. Compare phases using the exported comparison files")
    print("="*70)


if __name__ == '__main__':
    main()
