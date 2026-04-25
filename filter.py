import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

class PoseSmoothingPipeline:
    def __init__(self, window_size=5, outlier_std=3, butter_cutoff=6, 
                 butter_order=4, min_confidence=30, fps=30):
        """
        Initialize the smoothing pipeline with configurable parameters.
        
        Args:
            window_size: Size of moving window for smoothing
            outlier_std: Number of standard deviations for outlier detection
            butter_cutoff: Cutoff frequency for Butterworth filter (Hz)
            butter_order: Order of Butterworth filter
            min_confidence: Minimum confidence threshold (0-100)
            fps: Frames per second of the video
        """
        self.window_size = window_size
        self.outlier_std = outlier_std
        self.butter_cutoff = butter_cutoff
        self.butter_order = butter_order
        self.min_confidence = min_confidence
        self.fps = fps
        
    def load_data(self, filepath):
        """Load CSV data and prepare for processing"""
        self.df = pd.read_csv(filepath)
        self.df = self.df.sort_values(['frame_idx', 'position_name'])
        
        # Store original data for comparison
        self.original_df = self.df.copy()
        
        print(f"Loaded {len(self.df)} rows of pose data")
        print(f"Tracking: {self.df['position_name'].unique()}")
        return self.df
    
    def filter_low_confidence(self):
        """Replace low confidence measurements with NaN for interpolation"""
        low_conf_mask = self.df['confidence'] < self.min_confidence
        
        # Log how many points are being filtered
        n_filtered = low_conf_mask.sum()
        if n_filtered > 0:
            print(f"Filtering {n_filtered} low confidence points (< {self.min_confidence}%)")
            
            # Set coordinates to NaN for low confidence points
            for coord in ['x', 'y', 'z']:
                self.df.loc[low_conf_mask, coord] = np.nan
        
        return self.df
    
    def remove_outliers(self):
        """Detect and remove temporal outliers using velocity threshold"""
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
        
        print(f"Removed {outlier_count} outliers")
        return self.df
    
    def interpolate_missing(self):
        """Interpolate NaN values using cubic spline interpolation"""
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
                        f = interp1d(valid_indices, valid_values, 
                                   kind='cubic', fill_value='extrapolate')
                        interpolated_values = f(np.arange(len(series)))
                        
                        # Only replace NaN values
                        series[nan_mask] = interpolated_values[nan_mask]
                    else:
                        # Fallback to linear interpolation for small gaps
                        series = series.interpolate(method='linear')
                    
                    self.df.loc[position_indices, coord] = series
        
        print(f"Interpolated {interpolated_count} missing values")
        return self.df
    
    def confidence_weighted_smooth(self):
        """Apply confidence-weighted moving average smoothing"""
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
        
        print("Applied confidence-weighted smoothing")
        return self.df
    
    def butterworth_filter(self):
        """Apply Butterworth low-pass filter for final temporal smoothing"""
        # Design the filter
        nyquist = self.fps / 2
        normal_cutoff = self.butter_cutoff / nyquist
        
        # Ensure cutoff is valid
        if normal_cutoff >= 1.0:
            print(f"Warning: Cutoff frequency ({self.butter_cutoff} Hz) is too high for fps ({self.fps}). Reducing cutoff.")
            normal_cutoff = 0.99
            
        b, a = butter(self.butter_order, normal_cutoff, btype='low')
        
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
        
        print(f"Applied Butterworth low-pass filter (cutoff: {self.butter_cutoff} Hz)")
        return self.df
    
    def enforce_anatomical_constraints(self):
        """Enforce reasonable anatomical constraints between connected body parts"""
        # Define expected bone lengths (you can calibrate these from your data)
        # For now, we'll use gentle constraints
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
                # Relaxed constraints to avoid over-constraining
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
        
        return self.df
    
    def run_pipeline(self, input_file, output_file, verbose=True):
        """Execute the complete smoothing pipeline"""
        if verbose:
            print("="*60)
            print("POSE ESTIMATION SMOOTHING PIPELINE")
            print("="*60)
        
        # Step 1: Load data
        self.load_data(input_file)
        
        if verbose:
            print("\n1. Filtering low confidence measurements...")
        self.filter_low_confidence()
        
        if verbose:
            print("\n2. Removing temporal outliers...")
        self.remove_outliers()
        
        if verbose:
            print("\n3. Interpolating missing values...")
        self.interpolate_missing()
        
        if verbose:
            print("\n4. Applying confidence-weighted smoothing...")
        self.confidence_weighted_smooth()
        
        if verbose:
            print("\n5. Applying Butterworth temporal filter...")
        self.butterworth_filter()
        
        if verbose:
            print("\n6. Enforcing anatomical constraints...")
        self.enforce_anatomical_constraints()
        
        if verbose:
            print("\n7. Final quality check...")
        self.final_quality_check()
        
        # Save processed data
        self.df.to_csv(output_file, index=False)
        
        if verbose:
            print("\n" + "="*60)
            print(f"Processing complete! Output saved to: {output_file}")
            print("="*60)
            
            # Print statistics
            self.print_statistics()
        
        return self.df
    
    def print_statistics(self):
        """Print processing statistics"""
        print("\nPROCESSING STATISTICS:")
        print("-"*30)
        
        for position in self.df['position_name'].unique():
            orig_mask = self.original_df['position_name'] == position
            proc_mask = self.df['position_name'] == position
            
            # Calculate smoothness metric (total variation)
            for coord in ['x', 'y', 'z']:
                orig_variation = self.original_df.loc[orig_mask, coord].diff().abs().sum()
                proc_variation = self.df.loc[proc_mask, coord].diff().abs().sum()
                
                reduction = ((orig_variation - proc_variation) / orig_variation) * 100
                print(f"{position:12s} {coord}: Variation reduced by {reduction:5.1f}%")
        
        print("-"*30)
    
    def visualize_results(self, save_plot=None):
        """Create visualization comparing original and smoothed data"""
        try:
            import matplotlib.pyplot as plt
            
            positions = self.df['position_name'].unique()
            coords = ['x', 'y', 'z']
            
            fig, axes = plt.subplots(len(positions), len(coords), 
                                    figsize=(15, 4*len(positions)))
            
            if len(positions) == 1:
                axes = axes.reshape(1, -1)
            
            for i, position in enumerate(positions):
                orig_mask = self.original_df['position_name'] == position
                proc_mask = self.df['position_name'] == position
                
                time = self.df[proc_mask]['video_time_s'].values
                
                for j, coord in enumerate(coords):
                    ax = axes[i, j]
                    
                    # Plot original data
                    ax.plot(time, self.original_df.loc[orig_mask, coord].values, 
                           'r-', alpha=0.5, label='Original', linewidth=1)
                    
                    # Plot smoothed data
                    ax.plot(time, self.df.loc[proc_mask, coord].values, 
                           'b-', label='Smoothed', linewidth=2)
                    
                    ax.set_title(f'{position} - {coord}')
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel(f'{coord} (m)')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plot:
                plt.savefig(save_plot, dpi=150, bbox_inches='tight')
                print(f"Plot saved to: {save_plot}")
            
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for visualization")

# Main execution
if __name__ == "__main__":
    # Initialize the pipeline with tuned parameters
    pipeline = PoseSmoothingPipeline(
        window_size=5,        # Size of smoothing window
        outlier_std=3.0,      # Standard deviations for outlier detection
        butter_cutoff=8.0,    # Hz - cutoff for low-pass filter
        butter_order=4,       # Order of Butterworth filter
        min_confidence=40,    # Minimum confidence percentage
        fps=30               # Frames per second
    )
    
    # Run the pipeline
    smoothed_df = pipeline.run_pipeline(
        input_file='pose_data.csv',      # Your input CSV file
        output_file='pose_data_smoothed.csv',  # Output file
        verbose=True
    )
    
    # Optional: Visualize results
    # pipeline.visualize_results(save_plot='smoothing_comparison.png')
    
    # Optional: Quick preview of results
    print("\nPREVIEW OF SMOOTHED DATA:")
    print(smoothed_df.head(15))