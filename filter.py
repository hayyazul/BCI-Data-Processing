#!/usr/bin/env python3
"""
Pose Estimation Smoothing Pipeline

Takes raw pose estimation CSV data and applies various smoothing techniques
to produce more stable and realistic motion tracking data.

Usage:
    python smooth_poses.py input.csv [options]
    
Examples:
    python smooth_poses.py pose_data.csv
    python smooth_poses.py pose_data.csv -o smoothed_poses.csv
    python smooth_poses.py pose_data.csv --disable-anatomical --min-confidence 50
    python smooth_poses.py pose_data.csv --window-size 7 --butter-cutoff 4
"""

import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import argparse
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class PoseSmoothingPipeline:
    def __init__(self, window_size=5, outlier_std=3, butter_cutoff=6, 
                 butter_order=4, min_confidence=30, fps=30,
                 disable_confidence_filter=False, disable_outlier_removal=False,
                 disable_interpolation=False, disable_confidence_weighted=False,
                 disable_butterworth=False, disable_anatomical=False,
                 disable_quality_check=False):
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
            print("Skipping confidence filtering (disabled)")
            return self.df
            
        low_conf_mask = self.df['confidence'] < self.min_confidence
        
        # Log how many points are being filtered
        n_filtered = low_conf_mask.sum()
        if n_filtered > 0:
            print(f"Filtering {n_filtered} low confidence points (< {self.min_confidence}%)")
            
            # Set coordinates to NaN for low confidence points
            for coord in ['x', 'y', 'z']:
                self.df.loc[low_conf_mask, coord] = np.nan
        else:
            print("No low confidence points found")
        
        return self.df
    
    def remove_outliers(self):
        """Detect and remove temporal outliers using velocity threshold"""
        if not self.enable_outlier_removal:
            print("Skipping outlier removal (disabled)")
            return self.df
            
        outlier_count = 0
        
        for position in self.df['position_name'].unique():
            mask = self.df['position_name'] == position
            position_indices = self.df[mask].index
            
            for coord in ['x', 'y', 'z']:
                series = self.df.loc[position_indices, coord].copy()
                
                # Calculate velocity (first derivative)
                velocity = series.diff()
                
                # Calculate acceleration (second derivative) - better for detecting jerks
                acceleration = velocity.diff()
                
                # Detect outliers using both velocity and acceleration
                vel_threshold = self.outlier_std * velocity.std()
                acc_threshold = self.outlier_std * acceleration.std()
                
                outlier_mask = (np.abs(velocity) > vel_threshold) | \
                              (np.abs(acceleration) > acc_threshold)
                
                # Don't flag the first two frames (no velocity/acceleration data)
                outlier_mask.iloc[:2] = False
                
                n_outliers = outlier_mask.sum()
                if n_outliers > 0:
                    outlier_count += n_outliers
                    # Set outliers to NaN for interpolation
                    self.df.loc[position_indices[outlier_mask], coord] = np.nan
        
        print(f"Removed {outlier_count} outliers ({self.outlier_std} std threshold)")
        return self.df
    
    def interpolate_missing(self):
        """Interpolate NaN values using cubic spline interpolation"""
        if not self.enable_interpolation:
            print("Skipping interpolation (disabled)")
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
                    
                    # Create interpolation function based on valid points
                    valid_indices = np.arange(len(series))[~nan_mask]
                    valid_values = series[~nan_mask].values
                    
                    if len(valid_indices) > 3:  # Need at least 4 points for cubic spline
                        # Use cubic spline interpolation
                        try:
                            f = interp1d(valid_indices, valid_values, 
                                       kind='cubic', fill_value='extrapolate')
                            interpolated_values = f(np.arange(len(series)))
                            
                            # Only replace NaN values
                            series[nan_mask] = interpolated_values[nan_mask]
                        except Exception as e:
                            # Fallback to linear for problematic data
                            series = series.interpolate(method='linear')
                            series = series.fillna(method='bfill').fillna(method='ffill')
                    else:
                        # Fallback to linear interpolation for small gaps
                        series = series.interpolate(method='linear')
                        series = series.fillna(method='bfill').fillna(method='ffill')
                    
                    self.df.loc[position_indices, coord] = series
        
        print(f"Interpolated {interpolated_count} missing values")
        return self.df
    
    def confidence_weighted_smooth(self):
        """Apply confidence-weighted moving average smoothing"""
        if not self.enable_confidence_weighted:
            print("Skipping confidence-weighted smoothing (disabled)")
            return self.df
            
        for position in self.df['position_name'].unique():
            mask = self.df['position_name'] == position
            position_indices = self.df[mask].index
            
            # Get confidence values
            confidences = self.df.loc[position_indices, 'confidence'].values / 100.0
            
            for coord in ['x', 'y', 'z']:
                values = self.df.loc[position_indices, coord].values
                smoothed = np.zeros_like(values)
                
                # Apply confidence-weighted moving average
                for i in range(len(values)):
                    # Define window boundaries
                    start = max(0, i - self.window_size // 2)
                    end = min(len(values), i + self.window_size // 2 + 1)
                    
                    # Extract window
                    window_values = values[start:end]
                    window_conf = confidences[start:end]
                    
                    # Weight by confidence
                    if window_conf.sum() > 0:
                        smoothed[i] = np.average(window_values, weights=window_conf)
                    else:
                        smoothed[i] = window_values.mean()
                
                self.df.loc[position_indices, coord] = smoothed
        
        print(f"Applied confidence-weighted smoothing (window: {self.window_size})")
        return self.df
    
    def butterworth_filter(self):
        """Apply Butterworth low-pass filter for final temporal smoothing"""
        if not self.enable_butterworth:
            print("Skipping Butterworth filter (disabled)")
            return self.df
            
        # Design the filter
        nyquist = self.fps / 2
        normal_cutoff = self.butter_cutoff / nyquist
        
        # Ensure cutoff is valid
        if normal_cutoff >= 1.0:
            print(f"Warning: Cutoff frequency ({self.butter_cutoff} Hz) is too high for fps ({self.fps}). Reducing cutoff.")
            normal_cutoff = 0.99
        
        try:
            b, a = butter(self.butter_order, normal_cutoff, btype='low')
        except Exception as e:
            print(f"Error designing Butterworth filter: {e}")
            print("Skipping Butterworth filter")
            return self.df
        
        for position in self.df['position_name'].unique():
            mask = self.df['position_name'] == position
            position_indices = self.df[mask].index
            
            for coord in ['x', 'y', 'z']:
                signal = self.df.loc[position_indices, coord].values
                
                # Need at least 15 points for the filter to work well
                if len(signal) > 15:
                    # Apply zero-phase filtering
                    try:
                        filtered = filtfilt(b, a, signal)
                        self.df.loc[position_indices, coord] = filtered
                    except Exception as e:
                        print(f"Warning: Butterworth filter failed for {position}/{coord}: {e}")
        
        print(f"Applied Butterworth low-pass filter (cutoff: {self.butter_cutoff} Hz, order: {self.butter_order})")
        return self.df
    
    def enforce_anatomical_constraints(self):
        """Enforce reasonable anatomical constraints between connected body parts"""
        if not self.enable_anatomical:
            print("Skipping anatomical constraints (disabled)")
            return self.df
            
        constraints_applied = 0
        
        # Group by frame to work with complete poses
        for frame in self.df['frame_idx'].unique():
            frame_mask = self.df['frame_idx'] == frame
            
            # Get positions for this frame
            bracelet = self.df[frame_mask & (self.df['position_name'] == 'bracelet')]
            elbow = self.df[frame_mask & (self.df['position_name'] == 'elbow')]
            shoulder = self.df[frame_mask & (self.df['position_name'] == 'shoulder')]
            
            if len(bracelet) > 0 and len(elbow) > 0:
                # Calculate current bone lengths
                bracelet_pos = bracelet[['x', 'y', 'z']].values[0]
                elbow_pos = elbow[['x', 'y', 'z']].values[0]
                
                forearm_length = np.linalg.norm(bracelet_pos - elbow_pos)
                
                # Apply gentle constraints (forearm length ~0.3-0.5m typically)
                if forearm_length > 0.6:  # Unusually long forearm
                    direction = (bracelet_pos - elbow_pos) / forearm_length
                    new_bracelet = elbow_pos + direction * 0.4  # Scale to reasonable length
                    
                    bracelet_idx = bracelet.index[0]
                    self.df.loc[bracelet_idx, ['x', 'y', 'z']] = new_bracelet
                    constraints_applied += 1
                    
            if len(elbow) > 0 and len(shoulder) > 0:
                elbow_pos = elbow[['x', 'y', 'z']].values[0]
                shoulder_pos = shoulder[['x', 'y', 'z']].values[0]
                
                upper_arm_length = np.linalg.norm(elbow_pos - shoulder_pos)
                
                # Apply gentle constraints (upper arm length ~0.3-0.5m typically)
                if upper_arm_length > 0.6:
                    direction = (elbow_pos - shoulder_pos) / upper_arm_length
                    new_elbow = shoulder_pos + direction * 0.4
                    
                    elbow_idx = elbow.index[0]
                    self.df.loc[elbow_idx, ['x', 'y', 'z']] = new_elbow
                    constraints_applied += 1
        
        print(f"Applied {constraints_applied} anatomical constraints")
        return self.df
    
    def final_quality_check(self):
        """Perform final quality check and smoothing on confidence scores"""
        if not self.enable_quality_check:
            print("Skipping quality check (disabled)")
            return self.df
            
        # Smooth confidence scores slightly
        for position in self.df['position_name'].unique():
            mask = self.df['position_name'] == position
            position_indices = self.df[mask].index
            
            # Apply mild smoothing to confidence scores
            confidences = self.df.loc[position_indices, 'confidence'].values
            smoothed_conf = pd.Series(confidences).rolling(
                window=3, center=True, min_periods=1
            ).mean()
            
            # Ensure confidence stays in valid range
            smoothed_conf = smoothed_conf.clip(0, 100)
            self.df.loc[position_indices, 'confidence'] = smoothed_conf
        
        print("Applied final quality check")
        return self.df
    
    def run_pipeline(self, input_file, output_file, verbose=True):
        """Execute the complete smoothing pipeline"""
        if verbose:
            print("="*60)
            print("POSE ESTIMATION SMOOTHING PIPELINE")
            print("="*60)
            print(f"Input:  {input_file}")
            print(f"Output: {output_file}")
            print("="*60)
            print("\nConfiguration:")
            print(f"  Window size: {self.window_size}")
            print(f"  Outlier std threshold: {self.outlier_std}")
            print(f"  Butterworth cutoff: {self.butter_cutoff} Hz")
            print(f"  Butterworth order: {self.butter_order}")
            print(f"  Min confidence: {self.min_confidence}%")
            print(f"  FPS: {self.fps}")
            print("\nEnabled steps:")
            print(f"  Confidence filtering: {self.enable_confidence_filter}")
            print(f"  Outlier removal: {self.enable_outlier_removal}")
            print(f"  Interpolation: {self.enable_interpolation}")
            print(f"  Confidence-weighted smoothing: {self.enable_confidence_weighted}")
            print(f"  Butterworth filter: {self.enable_butterworth}")
            print(f"  Anatomical constraints: {self.enable_anatomical}")
            print(f"  Quality check: {self.enable_quality_check}")
            print("\n" + "="*60)
        
        # Step 1: Load data
        self.load_data(input_file)
        
        # Step 2: Process
        if verbose:
            print("\nProcessing steps:")
            
        print(f"\n1. {'[SKIP]' if not self.enable_confidence_filter else ''} Confidence filtering...")
        self.filter_low_confidence()
        
        print(f"\n2. {'[SKIP]' if not self.enable_outlier_removal else ''} Outlier removal...")
        self.remove_outliers()
        
        print(f"\n3. {'[SKIP]' if not self.enable_interpolation else ''} Interpolation...")
        self.interpolate_missing()
        
        print(f"\n4. {'[SKIP]' if not self.enable_confidence_weighted else ''} Confidence-weighted smoothing...")
        self.confidence_weighted_smooth()
        
        print(f"\n5. {'[SKIP]' if not self.enable_butterworth else ''} Butterworth filter...")
        self.butterworth_filter()
        
        print(f"\n6. {'[SKIP]' if not self.enable_anatomical else ''} Anatomical constraints...")
        self.enforce_anatomical_constraints()
        
        print(f"\n7. {'[SKIP]' if not self.enable_quality_check else ''} Quality check...")
        self.final_quality_check()
        
        # Save processed data
        self.df.to_csv(output_file, index=False)
        
        if verbose:
            print("\n" + "="*60)
            print(f"✅ Processing complete! Output saved to: {output_file}")
            print("="*60)
            
            # Print statistics
            self.print_statistics()
        
        return self.df
    
    def print_statistics(self):
        """Print processing statistics"""
        print("\n" + "="*60)
        print("SMOOTHING STATISTICS")
        print("="*60)
        
        for position in self.df['position_name'].unique():
            orig_mask = self.original_df['position_name'] == position
            proc_mask = self.df['position_name'] == position
            
            print(f"\n{position.upper()}:")
            print("-" * 40)
            
            for coord in ['x', 'y', 'z']:
                orig_series = self.original_df.loc[orig_mask, coord]
                proc_series = self.df.loc[proc_mask, coord]
                
                # Calculate smoothness metric (total variation)
                orig_variation = orig_series.diff().abs().sum()
                proc_variation = proc_series.diff().abs().sum()
                
                if orig_variation > 0:
                    reduction = ((orig_variation - proc_variation) / orig_variation) * 100
                    print(f"  {coord}: Variation reduced by {reduction:5.1f}%")
                else:
                    print(f"  {coord}: No variation in original data")
        
        print("\n" + "="*60)


def create_parser():
    """Create argument parser with detailed help"""
    parser = argparse.ArgumentParser(
        description='Smooth pose estimation data from CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s pose_data.csv
  %(prog)s pose_data.csv -o smoothed_poses.csv
  %(prog)s pose_data.csv --min-confidence 50 --window-size 7
  %(prog)s pose_data.csv --disable-anatomical --disable-butterworth
  %(prog)s pose_data.csv --butter-cutoff 4 --outlier-std 2.5

For more information, see the documentation.
        """
    )
    
    # Required arguments
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input CSV file with pose estimation data'
    )
    
    # Output file
    parser.add_argument(
        '-o', '--output-file',
        type=str,
        default=None,
        help='Path to output CSV file (default: input_file_smoothed.csv)'
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
    
    return parser


def main():
    """Main entry point for command-line usage"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate input file exists
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {args.input_file}")
        sys.exit(1)
    
    # Generate output filename if not provided
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        # Append '_smoothed' before the extension
        stem = input_path.stem
        suffix = input_path.suffix
        output_path = input_path.parent / f"{stem}_smoothed{suffix}"
    
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
    
    # Initialize pipeline with command-line arguments
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
        disable_quality_check=args.disable_quality_check
    )
    
    # Run the pipeline
    try:
        smoothed_df = pipeline.run_pipeline(
            input_file=str(input_path),
            output_file=str(output_path),
            verbose=not args.quiet
        )
        
        if not args.quiet and not args.no_stats:
            pass  # Statistics are already printed in run_pipeline
            
        print(f"\n✅ Successfully processed {len(smoothed_df)} data points")
        
    except Exception as e:
        print(f"\n❌ Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()