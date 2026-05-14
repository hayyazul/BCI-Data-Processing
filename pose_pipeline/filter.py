#!/usr/bin/env python3
"""
Pose Estimation Smoothing Pipeline

Takes raw pose estimation CSV data and applies various smoothing techniques
to produce more stable and realistic motion tracking data.

Usage:
    python smooth_poses.py input.csv [options]
    python smooth_poses.py "camera0_20260425_*_poses.csv" [options]
    
Examples:
    python smooth_poses.py pose_data.csv
    python smooth_poses.py "camera0_20260425_*_poses.csv"
    python smooth_poses.py "data/*_poses.csv" -o smoothed/
    python smooth_poses.py "*.csv" --disable-anatomical --min-confidence 50
    python smooth_poses.py camera0_20260425_poses.csv camera0_20260426_poses.csv -o smoothed/
"""

import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import argparse
import sys
import glob
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class PoseSmoothingPipeline:
    def __init__(self, window_size=5, outlier_std=3, butter_cutoff=6, 
                 butter_order=4, min_confidence=30, fps=30,
                 disable_confidence_filter=False, disable_outlier_removal=False,
                 disable_interpolation=False, disable_confidence_weighted=False,
                 disable_butterworth=False, disable_anatomical=False,
                 disable_quality_check=False, enable_elbow_correction=False, 
                 true_upper_arm_length_in=14, true_forearm_length=10):
        """
        Initialize the smoothing pipeline with configurable parameters.
        
        Args:
            window_size: Size of moving window for smoothing
            outlier_std: Number of standard deviations for outlier detection
            butter_cutoff: Cutoff frequency for Butterworth filter (Hz)
            butter_order: Order of Butterworth filter
            min_confidence: Minimum confidence threshold (0-100)
            fps: Frames per second of the video
            disable_*: Flags to disable specific pipeline steps
        """
        self.window_size = window_size
        self.outlier_std = outlier_std
        self.butter_cutoff = butter_cutoff
        self.butter_order = butter_order
        self.min_confidence = min_confidence
        self.fps = fps
        
        # Step control flags
        self.enable_confidence_filter = not disable_confidence_filter
        self.enable_outlier_removal = not disable_outlier_removal
        self.enable_interpolation = not disable_interpolation
        self.enable_confidence_weighted = not disable_confidence_weighted
        self.enable_butterworth = not disable_butterworth
        self.enable_anatomical = not disable_anatomical
        self.enable_quality_check = not disable_quality_check
        self.enable_elbow_correction = enable_elbow_correction
        # Known arm lengths in metres (14 and 10 inches)
        self.TRUE_UPPER_ARM_LENGTH = true_upper_arm_length_in * 0.0254
        self.TRUE_FOREARM_LENGTH = true_forearm_length * 0.0254
          
    def load_data(self, filepath):
        """Load CSV data and prepare for processing"""
        self.df = pd.read_csv(filepath)
        self.df = self.df.sort_values(['frame_idx', 'position_name'])
        
        # Store original data for comparison
        self.original_df = self.df.copy()
        
        print(f"Loaded {len(self.df)} rows of pose data")
        print(f"Frames: {self.df['frame_idx'].nunique()}")
        print(f"Tracking: {', '.join(self.df['position_name'].unique())}")
        return self.df
    
    def filter_low_confidence(self):
        """Replace low confidence measurements with NaN for interpolation"""
        if not self.enable_confidence_filter:
            print("  Skipping confidence filtering (disabled)")
            return self.df
            
        low_conf_mask = self.df['confidence'] < self.min_confidence
        
        n_filtered = low_conf_mask.sum()
        if n_filtered > 0:
            print(f"  Filtering {n_filtered} low confidence points (< {self.min_confidence}%)")
            for coord in ['x', 'y', 'z']:
                self.df.loc[low_conf_mask, coord] = np.nan
        else:
            print("  No low confidence points found")
        
        return self.df
    
    def remove_outliers(self):
        """Detect and remove temporal outliers using velocity threshold"""
        if not self.enable_outlier_removal:
            print("  Skipping outlier removal (disabled)")
            return self.df
            
        outlier_count = 0
        
        for position in self.df['position_name'].unique():
            mask = self.df['position_name'] == position
            position_indices = self.df[mask].index
            
            for coord in ['x', 'y', 'z']:
                series = self.df.loc[position_indices, coord].copy()
                
                velocity = series.diff()
                acceleration = velocity.diff()
                
                vel_threshold = self.outlier_std * velocity.std()
                acc_threshold = self.outlier_std * acceleration.std()
                
                outlier_mask = (np.abs(velocity) > vel_threshold) | \
                              (np.abs(acceleration) > acc_threshold)
                
                outlier_mask.iloc[:2] = False
                
                n_outliers = outlier_mask.sum()
                if n_outliers > 0:
                    outlier_count += n_outliers
                    self.df.loc[position_indices[outlier_mask], coord] = np.nan
        
        print(f"  Removed {outlier_count} outliers ({self.outlier_std} std threshold)")
        return self.df
    
    def interpolate_missing(self):
        """Interpolate NaN values using cubic spline interpolation"""
        if not self.enable_interpolation:
            print("  Skipping interpolation (disabled)")
            return self.df
            
        interpolated_count = 0
        
        for position in self.df['position_name'].unique():
            mask = self.df['position_name'] == position
            position_indices = self.df[mask].index
            
            for coord in ['x', 'y', 'z']:
                series = self.df.loc[position_indices, coord].copy()
                nan_mask = series.isna()
                
                if nan_mask.any():
                    interpolated_count += nan_mask.sum()
                    
                    valid_indices = np.arange(len(series))[~nan_mask]
                    valid_values = series[~nan_mask].values
                    
                    if len(valid_indices) > 3:
                        try:
                            f = interp1d(valid_indices, valid_values, 
                                       kind='cubic', fill_value='extrapolate')
                            interpolated_values = f(np.arange(len(series)))
                            series[nan_mask] = interpolated_values[nan_mask]
                        except Exception:
                            series = series.interpolate(method='linear')
                            series = series.fillna(method='bfill').fillna(method='ffill')
                    else:
                        series = series.interpolate(method='linear')
                        series = series.fillna(method='bfill').fillna(method='ffill')
                    
                    self.df.loc[position_indices, coord] = series
        
        print(f"  Interpolated {interpolated_count} missing values")
        return self.df
    
    def confidence_weighted_smooth(self):
        """Apply confidence-weighted moving average smoothing"""
        if not self.enable_confidence_weighted:
            print("  Skipping confidence-weighted smoothing (disabled)")
            return self.df
            
        for position in self.df['position_name'].unique():
            mask = self.df['position_name'] == position
            position_indices = self.df[mask].index
            
            confidences = self.df.loc[position_indices, 'confidence'].values / 100.0
            
            for coord in ['x', 'y', 'z']:
                values = self.df.loc[position_indices, coord].values
                smoothed = np.zeros_like(values)
                
                for i in range(len(values)):
                    start = max(0, i - self.window_size // 2)
                    end = min(len(values), i + self.window_size // 2 + 1)
                    
                    window_values = values[start:end]
                    window_conf = confidences[start:end]
                    
                    if window_conf.sum() > 0:
                        smoothed[i] = np.average(window_values, weights=window_conf)
                    else:
                        smoothed[i] = window_values.mean()
                
                self.df.loc[position_indices, coord] = smoothed
        
        print(f"  Applied confidence-weighted smoothing (window: {self.window_size})")
        return self.df
    
    def butterworth_filter(self):
        """Apply Butterworth low-pass filter for final temporal smoothing"""
        if not self.enable_butterworth:
            print("  Skipping Butterworth filter (disabled)")
            return self.df
            
        nyquist = self.fps / 2
        normal_cutoff = self.butter_cutoff / nyquist
        
        if normal_cutoff >= 1.0:
            print(f"  Warning: Cutoff frequency ({self.butter_cutoff} Hz) too high for fps ({self.fps})")
            normal_cutoff = 0.99
        
        try:
            b, a = butter(self.butter_order, normal_cutoff, btype='low')
        except Exception as e:
            print(f"  Error designing Butterworth filter: {e}")
            return self.df
        
        for position in self.df['position_name'].unique():
            mask = self.df['position_name'] == position
            position_indices = self.df[mask].index
            
            for coord in ['x', 'y', 'z']:
                signal = self.df.loc[position_indices, coord].values
                
                if len(signal) > 15:
                    try:
                        filtered = filtfilt(b, a, signal)
                        self.df.loc[position_indices, coord] = filtered
                    except Exception as e:
                        print(f"  Warning: Butterworth filter failed for {position}/{coord}: {e}")
        
        print(f"  Applied Butterworth low-pass filter (cutoff: {self.butter_cutoff} Hz, order: {self.butter_order})")
        return self.df

    def correct_elbow_position(self):
        """
        Correct the elbow tag position by extrapolating it along the
        shoulder‑to‑tag ray to exactly match the known upper arm length.
        This compensates for the tag being offset a few inches proximal to the
        true elbow joint.
        """
        if not self.enable_elbow_correction:
            print("  Note: Elbow position correction is disabled. Use --correct-elbow for better anatomical accuracy.")
            return self.df
    
        n_corrected = 0
        shoulder_pos = None
        elbow_pos = None
    
        # We'll work on a copy to avoid chained indexing warnings
        df = self.df.copy()
        # We need to ensure we are operating on the DataFrame after all previous steps
        # (it is already in self.df)
    
        for frame in df['frame_idx'].unique():
            frame_mask = df['frame_idx'] == frame
    
            shoulder_rows = df[frame_mask & (df['position_name'] == 'shoulder')]
            elbow_rows = df[frame_mask & (df['position_name'] == 'elbow')]
    
            if len(shoulder_rows) == 0 or len(elbow_rows) == 0:
                continue
    
            shoulder = shoulder_rows[['x','y','z']].values[0]
            elbow = elbow_rows[['x','y','z']].values[0]
    
            vec = elbow - shoulder
            dist = np.linalg.norm(vec)
    
            if dist < 1e-9:   # degenerate, skip
                continue
    
            direction = vec / dist
            corrected_elbow = shoulder + direction * self.TRUE_UPPER_ARM_LENGTH
    
            elbow_idx = elbow_rows.index[0]
            df.loc[elbow_idx, ['x','y','z']] = corrected_elbow
            n_corrected += 1
    
        self.df = df
        print(f"  Applied elbow position correction to {n_corrected} frames (target upper arm length = {self.TRUE_UPPER_ARM_LENGTH*100:.1f} cm)")
        return self.df
    
    def enforce_anatomical_constraints(self):
        """Enforce reasonable anatomical constraints between connected body parts"""
        if not self.enable_anatomical:
            print("  Skipping anatomical constraints (disabled)")
            return self.df
            
        constraints_applied = 0
        
        for frame in self.df['frame_idx'].unique():
            frame_mask = self.df['frame_idx'] == frame
            
            bracelet = self.df[frame_mask & (self.df['position_name'] == 'bracelet')]
            elbow = self.df[frame_mask & (self.df['position_name'] == 'elbow')]
            shoulder = self.df[frame_mask & (self.df['position_name'] == 'shoulder')]
            
            if len(bracelet) > 0 and len(elbow) > 0:
                bracelet_pos = bracelet[['x', 'y', 'z']].values[0]
                elbow_pos = elbow[['x', 'y', 'z']].values[0]
                
                forearm_length = np.linalg.norm(bracelet_pos - elbow_pos)
                
                if forearm_length > 0.6:
                    direction = (bracelet_pos - elbow_pos) / forearm_length
                    new_bracelet = elbow_pos + direction * 0.4
                    
                    bracelet_idx = bracelet.index[0]
                    self.df.loc[bracelet_idx, ['x', 'y', 'z']] = new_bracelet
                    constraints_applied += 1
                    
            if len(elbow) > 0 and len(shoulder) > 0:
                elbow_pos = elbow[['x', 'y', 'z']].values[0]
                shoulder_pos = shoulder[['x', 'y', 'z']].values[0]
                
                upper_arm_length = np.linalg.norm(elbow_pos - shoulder_pos)
                
                if upper_arm_length > 0.6:
                    direction = (elbow_pos - shoulder_pos) / upper_arm_length
                    new_elbow = shoulder_pos + direction * 0.4
                    
                    elbow_idx = elbow.index[0]
                    self.df.loc[elbow_idx, ['x', 'y', 'z']] = new_elbow
                    constraints_applied += 1
        
        print(f"  Applied {constraints_applied} anatomical constraints")
        return self.df
    
    def final_quality_check(self):
        """Perform final quality check and smoothing on confidence scores"""
        if not self.enable_quality_check:
            print("  Skipping quality check (disabled)")
            return self.df
            
        for position in self.df['position_name'].unique():
            mask = self.df['position_name'] == position
            position_indices = self.df[mask].index
            
            confidences = self.df.loc[position_indices, 'confidence'].values
            smoothed_conf = pd.Series(confidences).rolling(
                window=3, center=True, min_periods=1
            ).mean()
            
            smoothed_conf = smoothed_conf.clip(0, 100)
            self.df.loc[position_indices, 'confidence'] = smoothed_conf
        
        print("  Applied final quality check")
        return self.df
    
    def run_pipeline(self, input_file, output_file, verbose=True):
        """Execute the complete smoothing pipeline"""
        if verbose:
            print(f"\nProcessing: {os.path.basename(input_file)}")
            print("-" * 40)
        
        # Load data
        self.load_data(input_file)
        
        # Process
        print("  1. Confidence filtering...")
        self.filter_low_confidence()
        
        print("  2. Outlier removal...")
        self.remove_outliers()
        
        print("  3. Interpolation...")
        self.interpolate_missing()
        
        print("  4. Confidence-weighted smoothing...")
        self.confidence_weighted_smooth()
        
        print("  5. Butterworth filter...")
        self.butterworth_filter()
        
        print("  6. Anatomical constraints...")
        self.enforce_anatomical_constraints()
        
        print("  7. Quality check...")
        self.final_quality_check()

        print("  8. Elbow position correction...")
        self.correct_elbow_position()
        
        # Save processed data
        self.df.to_csv(output_file, index=False)
        
        if verbose:
            print(f"  ✓ Saved to: {output_file}")
        
        return self.df
    
    def print_statistics(self):
        """Print processing statistics"""
        print("\n  Smoothing Statistics:")
        print("  " + "-" * 40)
        
        for position in self.df['position_name'].unique():
            orig_mask = self.original_df['position_name'] == position
            proc_mask = self.df['position_name'] == position
            
            print(f"  {position}:")
            
            for coord in ['x', 'y', 'z']:
                orig_series = self.original_df.loc[orig_mask, coord]
                proc_series = self.df.loc[proc_mask, coord]
                
                orig_variation = orig_series.diff().abs().sum()
                proc_variation = proc_series.diff().abs().sum()
                
                if orig_variation > 0:
                    reduction = ((orig_variation - proc_variation) / orig_variation) * 100
                    print(f"    {coord}: Variation reduced by {reduction:5.1f}%")
                else:
                    print(f"    {coord}: No variation in original data")


def generate_output_filename(input_path, output_arg):
    """Generate output filename based on input path and output argument"""
    input_path = Path(input_path)
    
    if output_arg:
        output_path = Path(output_arg)
        
        # If output is a directory, place file there with _smoothed suffix
        if output_path.is_dir() or output_arg.endswith('/') or output_arg.endswith('\\'):
            stem = input_path.stem
            suffix = input_path.suffix
            output_path = output_path / f"{stem}_smoothed{suffix}"
        # If output is a file path, use it directly
        else:
            # If it doesn't have an extension, treat as directory
            if not output_path.suffix:
                stem = input_path.stem
                suffix = input_path.suffix
                output_path = output_path / f"{stem}_smoothed{suffix}"
    else:
        # Default: append _smoothed before extension
        stem = input_path.stem
        suffix = input_path.suffix
        output_path = input_path.parent / f"{stem}_smoothed{suffix}"
    
    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    return str(output_path)


def expand_file_patterns(patterns):
    """Expand wildcard patterns and return list of files"""
    files = []
    
    for pattern in patterns:
        # Expand the pattern
        matched_files = glob.glob(pattern)
        
        if not matched_files:
            print(f"⚠️  Warning: No files found matching pattern: {pattern}")
            continue
        
        # Filter to only CSV files
        csv_files = [f for f in matched_files if f.lower().endswith('.csv')]
        
        if not csv_files:
            print(f"⚠️  Warning: No CSV files found matching pattern: {pattern}")
            continue
        
        files.extend(csv_files)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for f in files:
        abs_path = os.path.abspath(f)
        if abs_path not in seen:
            seen.add(abs_path)
            unique_files.append(f)
    
    return unique_files


def create_parser():
    """Create argument parser with detailed help"""
    parser = argparse.ArgumentParser(
        description='Smooth pose estimation data from CSV files (supports wildcards)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single file
  %(prog)s pose_data.csv
  
  # Process with custom output
  %(prog)s pose_data.csv -o smoothed_poses.csv
  
  # Process multiple files with wildcard
  %(prog)s "camera0_20260425_*_poses.csv"
  
  # Process all CSV files in a directory
  %(prog)s "data/*.csv" -o smoothed/
  
  # Process multiple patterns
  %(prog)s "camera0_*.csv" "camera1_*.csv" -o output/
  
  # Custom parameters with wildcard
  %(prog)s "data/*_poses.csv" --min-confidence 50 --window-size 7
  
  # Disable specific steps for all files
  %(prog)s "*.csv" --disable-anatomical --disable-butterworth
  
  # Multiple explicit files
  %(prog)s file1.csv file2.csv file3.csv -o smoothed/

For more information, see the documentation.
        """
    )
    
    # Required arguments
    parser.add_argument(
        'input_patterns',
        type=str,
        nargs='+',
        help='Input CSV file(s) or wildcard pattern(s) (e.g., "camera0_*.csv")'
    )
    
    # Output file/directory
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output file or directory (default: input_smoothed.csv)'
    )
    
    # Filter parameters
    filter_group = parser.add_argument_group('Filter Parameters')
    filter_group.add_argument(
        '--window-size',
        type=int,
        default=5,
        help='Size of smoothing window in frames (default: 5)'
    )
    filter_group.add_argument(
        '--outlier-std',
        type=float,
        default=3.0,
        help='Number of standard deviations for outlier detection (default: 3.0)'
    )
    filter_group.add_argument(
        '--butter-cutoff',
        type=float,
        default=6.0,
        help='Cutoff frequency for Butterworth filter in Hz (default: 6.0)'
    )
    filter_group.add_argument(
        '--butter-order',
        type=int,
        default=4,
        help='Order of Butterworth filter (default: 4)'
    )
    filter_group.add_argument(
        '--min-confidence',
        type=float,
        default=30.0,
        help='Minimum confidence threshold 0-100 (default: 30.0)'
    )
    filter_group.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Frames per second of the video (default: 30)'
    )
    
    # Disable flags
    disable_group = parser.add_argument_group('Disable Pipeline Steps')
    disable_group.add_argument(
        '--disable-confidence-filter',
        action='store_true',
        help='Disable low confidence filtering step'
    )
    disable_group.add_argument(
        '--disable-outlier-removal',
        action='store_true',
        help='Disable outlier removal step'
    )
    disable_group.add_argument(
        '--disable-interpolation',
        action='store_true',
        help='Disable interpolation step'
    )
    disable_group.add_argument(
        '--disable-confidence-weighted',
        action='store_true',
        help='Disable confidence-weighted smoothing step'
    )
    disable_group.add_argument(
        '--disable-butterworth',
        action='store_true',
        help='Disable Butterworth filter step'
    )
    disable_group.add_argument(
        '--disable-anatomical',
        action='store_true',
        help='Disable anatomical constraints step'
    )
    disable_group.add_argument(
        '--disable-quality-check',
        action='store_true',
        help='Disable final quality check step'
    )
    
    # Additional options
    other_group = parser.add_argument_group('Other Options')
    other_group.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )
    other_group.add_argument(
        '--no-stats',
        action='store_true',
        help='Do not print smoothing statistics'
    )
    other_group.add_argument(
        '--list-only',
        action='store_true',
        help='Only list files that would be processed, without processing them'
    )
    # In the "Other Options" group
    other_group.add_argument(
        '--correct-elbow',
        action='store_true',
        help='Extrapolate elbow tag position to known upper arm length'
    )
    other_group.add_argument(
        '--true-upper-arm',
        type=float,
        default=14.0,
        help='True upper arm length in inches (default: 14.0)'
    )
    other_group.add_argument(
        '--true-forearm',
        type=float,
        default=10.0,
        help='True forearm length in inches (default: 10.0)'
    )

    return parser


def main():
    """Main entry point for command-line usage"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate parameters
    if args.min_confidence < 0 or args.min_confidence > 100:
        print("❌ Error: min-confidence must be between 0 and 100")
        sys.exit(1)
    
    if args.window_size < 1:
        print("❌ Error: window-size must be positive")
        sys.exit(1)
    
    if args.outlier_std <= 0:
        print("❌ Error: outlier-std must be positive")
        sys.exit(1)
    
    if args.butter_cutoff <= 0:
        print("❌ Error: butter-cutoff must be positive")
        sys.exit(1)
    
    if args.butter_order < 1:
        print("❌ Error: butter-order must be positive")
        sys.exit(1)
    
    if args.fps < 1:
        print("❌ Error: fps must be positive")
        sys.exit(1)
    
    # Expand file patterns
    input_files = expand_file_patterns(args.input_patterns)
    
    if not input_files:
        print("❌ Error: No CSV files found matching the specified patterns")
        sys.exit(1)
    
    # If --list-only flag is set, just print the files and exit
    if args.list_only:
        print(f"\nFound {len(input_files)} file(s) to process:")
        for i, f in enumerate(input_files, 1):
            print(f"  {i}. {f}")
        return
    
    # Print header
    if not args.quiet:
        print("="*60)
        print("POSE ESTIMATION SMOOTHING PIPELINE")
        print("="*60)
        print(f"\nConfiguration:")
        print(f"  Window size: {args.window_size}")
        print(f"  Outlier std threshold: {args.outlier_std}")
        print(f"  Butterworth cutoff: {args.butter_cutoff} Hz")
        print(f"  Butterworth order: {args.butter_order}")
        print(f"  Min confidence: {args.min_confidence}%")
        print(f"  FPS: {args.fps}")
        print(f"\nEnabled steps:")
        print(f"  Confidence filtering: {not args.disable_confidence_filter}")
        print(f"  Outlier removal: {not args.disable_outlier_removal}")
        print(f"  Interpolation: {not args.disable_interpolation}")
        print(f"  Confidence-weighted smoothing: {not args.disable_confidence_weighted}")
        print(f"  Butterworth filter: {not args.disable_butterworth}")
        print(f"  Anatomical constraints: {not args.disable_anatomical}")
        print(f"  Quality check: {not args.disable_quality_check}")
        print(f"\nFound {len(input_files)} file(s) to process:")
        for f in input_files:
            print(f"  • {f}")
        print("\n" + "="*60)
    
    # Process each file
    successful = 0
    failed = 0
    
    for input_file in input_files:
        try:
            # Generate output filename
            output_file = generate_output_filename(input_file, args.output)
            
            # Initialize pipeline
            pipeline = PoseSmoothingPipeline(
                window_size=args.window_size,
                outlier_std=args.outlier_std,
                butter_cutoff=args.butter_cutoff,
                butter_order=args.butter_order,
                min_confidence=args.min_confidence,
                fps=args.fps,
                disable_confidence_filter=args.disable_confidence_filter,
                disable_outlier_removal=args.disable_outlier_removal,
                disable_interpolation=args.disable_interpolation,
                disable_confidence_weighted=args.disable_confidence_weighted,
                disable_butterworth=args.disable_butterworth,
                disable_anatomical=args.disable_anatomical,
                disable_quality_check=args.disable_quality_check,
                enable_elbow_correction=args.correct_elbow,           # was: args.corrected_elbow
                true_upper_arm_length_in=args.true_upper_arm,         # was missing
                true_forearm_length=args.true_forearm                 # was missing
            )
            
            # Run pipeline
            pipeline.run_pipeline(
                input_file=input_file,
                output_file=output_file,
                verbose=not args.quiet
            )
            
            if not args.quiet and not args.no_stats:
                pipeline.print_statistics()
            
            successful += 1
            
        except Exception as e:
            print(f"\n❌ Error processing {input_file}: {str(e)}")
            if not args.quiet:
                import traceback
                traceback.print_exc()
            failed += 1
    
    # Print summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"  Total files: {len(input_files)}")
    print(f"  Successful:  {successful} ✅")
    if failed > 0:
        print(f"  Failed:      {failed} ❌")
    print("="*60)


if __name__ == "__main__":
    main()