# vertical_oscillation_analyzer.py

"""
Vertical oscillation analysis for running biomechanics.

This module analyzes the vertical movement of a runner's center of mass (CoM) during
the gait cycle. Vertical oscillation is a key efficiency metric in running performance,
with excessive vertical movement indicating energy waste and potential inefficiency.

Biomechanical Context:
- Optimal vertical oscillation: 6-9 cm for recreational runners
- Elite runners typically show 4-7 cm vertical oscillation
- Excessive oscillation (>12 cm) indicates inefficient running form
- Related to stride mechanics, cadence, and energy expenditure
- Connected to ground contact time and flight phase dynamics

Key Metrics:
- Peak-to-peak vertical displacement of center of mass
- Oscillation frequency (cycles per second)
- Efficiency ratings based on established biomechanical ranges
"""

import logging
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Union, Tuple
from scipy import signal
from scipy.stats import zscore

# Configure module logger
logger = logging.getLogger(__name__)

# Type aliases
Landmark = Tuple[float, float, float, float]  # (x, y, z, visibility)
LandmarksDict = Dict[str, Landmark]
OscillationResult = Dict[str, Union[float, str, bool]]
QualityMetrics = Dict[str, Union[str, float, int]]


class VerticalOscillationAnalyzer:
    """
    Analyzes vertical oscillation patterns in running gait using center of mass tracking.
    
    This analyzer maintains a sliding window of center of mass height measurements to
    calculate vertical oscillation metrics including displacement magnitude, frequency,
    and efficiency ratings. Uses hip midpoint as CoM approximation, which provides
    good correlation with full-body center of mass for running analysis.
    
    Key Features:
    - Real-time analysis with configurable sliding window
    - Robust frequency detection using multiple methods
    - Data quality assessment and filtering
    - Biomechanically-informed efficiency ratings
    - Outlier detection and smoothing
    
    Attributes:
        frame_rate (float): Video frame rate for temporal calculations
        window_size (int): Number of frames maintained for analysis
        com_heights (deque): Sliding window of center of mass heights
        frame_count (int): Total frames processed
        min_data_points (int): Minimum data required for analysis
        scale_factor (float): Conversion factor to real-world units (cm)
    """
    
    def __init__(
        self,
        frame_rate: float = 60.0,
        window_size: int = 30,
        scale_factor: float = 100.0,
        smoothing_enabled: bool = True
    ) -> None:
        """
        Initialize vertical oscillation analyzer.
        
        Parameters:
        -----------
        frame_rate : float, default=60.0
            Video frame rate in fps for temporal analysis.
            Critical for accurate frequency calculations.
            
        window_size : int, default=30
            Number of frames in sliding analysis window.
            Larger windows provide smoother analysis but less responsiveness.
            30 frames ≈ 0.5 seconds at 60fps.
            
        scale_factor : float, default=100.0
            Conversion factor from normalized coordinates to centimeters.
            Assumes normalized MediaPipe coordinates and typical body proportions.
            May require calibration based on camera distance and setup.
            
        smoothing_enabled : bool, default=True
            Whether to apply smoothing to reduce measurement noise.
            Recommended for real-world video analysis.
            
        Raises:
        -------
        ValueError
            If frame_rate <= 0 or window_size < 5
        """
        # Input validation
        if frame_rate <= 0:
            raise ValueError(f"Frame rate must be positive, got {frame_rate}")
        if window_size < 5:
            raise ValueError(f"Window size must be ≥5 for meaningful analysis, got {window_size}")
        if scale_factor <= 0:
            raise ValueError(f"Scale factor must be positive, got {scale_factor}")
            
        # Core parameters
        self.frame_rate = frame_rate
        self.window_size = window_size
        self.scale_factor = scale_factor
        self.smoothing_enabled = smoothing_enabled
        
        # Data storage and tracking
        self.com_heights = deque(maxlen=window_size)
        self.raw_heights = deque(maxlen=window_size)  # Store unsmoothed data
        self.frame_count = 0
        self.min_data_points = min(10, window_size)  # Minimum data for analysis
        
        # Quality control parameters
        self.outlier_threshold = 2.5  # Z-score threshold for outlier detection
        self.max_instantaneous_change = 0.05  # Maximum reasonable frame-to-frame change
        
        # Efficiency thresholds (cm) based on biomechanical research
        self.efficiency_thresholds = {
            'excellent': 7.0,
            'good': 9.0,
            'moderate': 12.0,
            'poor': float('inf')
        }
        
        logger.info(f"VerticalOscillationAnalyzer initialized: "
                   f"frame_rate={frame_rate}, window_size={window_size}, "
                   f"scale_factor={scale_factor}")
    
    def update(self, landmarks: LandmarksDict) -> OscillationResult:
        """
        Process new frame and calculate vertical oscillation metrics.
        
        Parameters:
        -----------
        landmarks : LandmarksDict
            MediaPipe pose landmarks containing hip positions.
            Required keys: 'left_hip', 'right_hip'
            Format: {landmark_name: (x, y, z, visibility)}
            
        Returns:
        --------
        OscillationResult
            Dictionary containing:
            - 'vertical_oscillation_cm' (float): Peak-to-peak displacement in cm
            - 'oscillation_frequency' (float): Oscillation frequency in Hz
            - 'efficiency_rating' (str): Biomechanical efficiency assessment
            - 'avg_com_height' (float): Average CoM height in normalized coords
            - 'data_quality' (str): Quality assessment of current data
            - 'frames_analyzed' (int): Number of frames used in analysis
            - 'calculation_successful' (bool): Whether analysis completed successfully
        """
        result = self._initialize_result()
        
        try:
            # Extract and validate center of mass height
            com_height = self._extract_com_height(landmarks)
            if com_height is None:
                result['data_quality'] = 'invalid_landmarks'
                return result
            
            # Quality control and smoothing
            if self._is_outlier(com_height):
                logger.debug(f"Outlier detected: {com_height:.4f}, using interpolation")
                com_height = self._interpolate_missing_value()
            
            # Store data
            self.raw_heights.append(com_height)
            processed_height = self._apply_smoothing(com_height) if self.smoothing_enabled else com_height
            self.com_heights.append(processed_height)
            self.frame_count += 1
            
            # Check if sufficient data available
            if len(self.com_heights) < self.min_data_points:
                result['data_quality'] = 'insufficient_data'
                result['frames_analyzed'] = len(self.com_heights)
                return result
            
            # Perform oscillation analysis
            return self._analyze_oscillation()
            
        except KeyError as e:
            logger.error(f"Missing required landmark: {e}")
            result['data_quality'] = 'missing_landmarks'
            return result
        except Exception as e:
            logger.exception(f"Error in vertical oscillation analysis: {e}")
            result['data_quality'] = 'analysis_error'
            return result
    
    def _extract_com_height(self, landmarks: LandmarksDict) -> Optional[float]:
        """
        Extract center of mass height from hip landmarks.
        
        Parameters:
        -----------
        landmarks : LandmarksDict
            Pose landmarks dictionary
            
        Returns:
        --------
        Optional[float]
            Center of mass Y-coordinate, or None if extraction fails
        """
        required_landmarks = ['left_hip', 'right_hip']
        
        # Validate landmark availability and visibility
        for landmark_name in required_landmarks:
            if landmark_name not in landmarks:
                logger.warning(f"Required landmark '{landmark_name}' not found")
                return None
            
            # Check visibility (assuming visibility is 4th element)
            if len(landmarks[landmark_name]) >= 4 and landmarks[landmark_name][3] < 0.5:
                logger.debug(f"Low visibility for {landmark_name}: {landmarks[landmark_name][3]}")
                return None
        
        try:
            # Calculate hip center as CoM approximation
            left_hip_y = landmarks['left_hip'][1]
            right_hip_y = landmarks['right_hip'][1]
            com_height = (left_hip_y + right_hip_y) / 2.0
            
            # Sanity check - CoM should be within reasonable range
            if not 0.0 <= com_height <= 1.0:
                logger.warning(f"CoM height outside expected range: {com_height}")
                return None
                
            return com_height
            
        except (IndexError, TypeError) as e:
            logger.error(f"Error extracting CoM height: {e}")
            return None
    
    def _is_outlier(self, new_height: float) -> bool:
        """
        Detect if new measurement is an outlier.
        
        Uses both statistical (Z-score) and temporal (frame-to-frame change) criteria.
        
        Parameters:
        -----------
        new_height : float
            New CoM height measurement
            
        Returns:
        --------
        bool
            True if measurement appears to be an outlier
        """
        if len(self.com_heights) < 3:
            return False  # Not enough data for outlier detection
        
        # Check for excessive frame-to-frame change
        if len(self.com_heights) > 0:
            instantaneous_change = abs(new_height - self.com_heights[-1])
            if instantaneous_change > self.max_instantaneous_change:
                return True
        
        # Statistical outlier detection using Z-score
        if len(self.com_heights) >= 5:
            heights_array = np.array(list(self.com_heights))
            recent_heights = np.append(heights_array[-4:], new_height)
            z_scores = np.abs(zscore(recent_heights))
            
            if z_scores[-1] > self.outlier_threshold:
                return True
        
        return False
    
    def _interpolate_missing_value(self) -> float:
        """
        Interpolate CoM height when outlier detected.
        
        Returns:
        --------
        float
            Interpolated CoM height value
        """
        if len(self.com_heights) < 2:
            return self.com_heights[-1] if self.com_heights else 0.5
        
        # Simple linear interpolation from last two valid points
        if len(self.com_heights) >= 2:
            return (self.com_heights[-1] + self.com_heights[-2]) / 2.0
        
        return self.com_heights[-1]
    
    def _apply_smoothing(self, new_height: float) -> float:
        """
        Apply temporal smoothing to reduce measurement noise.
        
        Uses exponential moving average for real-time processing.
        
        Parameters:
        -----------
        new_height : float
            Raw CoM height measurement
            
        Returns:
        --------
        float
            Smoothed CoM height
        """
        if len(self.com_heights) == 0:
            return new_height
        
        # Exponential moving average with alpha = 0.3
        alpha = 0.3
        return alpha * new_height + (1 - alpha) * self.com_heights[-1]
    
    def _analyze_oscillation(self) -> OscillationResult:
        """
        Perform comprehensive vertical oscillation analysis.
        
        Returns:
        --------
        OscillationResult
            Complete oscillation analysis results
        """
        heights_array = np.array(self.com_heights)
        
        # Basic statistics
        avg_height = np.mean(heights_array)
        min_height = np.min(heights_array)
        max_height = np.max(heights_array)
        
        # Calculate vertical oscillation in centimeters
        vertical_oscillation_cm = (max_height - min_height) * self.scale_factor
        
        # Calculate oscillation frequency
        oscillation_frequency = self._calculate_frequency_advanced(heights_array, avg_height)
        
        # Assess efficiency and data quality
        efficiency_rating = self._get_efficiency_rating(vertical_oscillation_cm)
        data_quality = self._assess_data_quality(heights_array)
        
        return {
            'vertical_oscillation_cm': round(vertical_oscillation_cm, 2),
            'oscillation_frequency': round(oscillation_frequency, 2),
            'efficiency_rating': efficiency_rating,
            'avg_com_height': round(avg_height, 4),
            'min_com_height': round(min_height, 4),
            'max_com_height': round(max_height, 4),
            'data_quality': data_quality,
            'frames_analyzed': len(self.com_heights),
            'calculation_successful': True,
            'std_deviation': round(np.std(heights_array), 4),
            'smoothness_score': self._calculate_smoothness_score(heights_array)
        }
    
    def _calculate_frequency_advanced(self, heights: np.ndarray, avg_height: float) -> float:
        """
        Calculate oscillation frequency using multiple robust methods.
        
        Combines peak detection and spectral analysis for accuracy.
        
        Parameters:
        -----------
        heights : np.ndarray
            Array of CoM height measurements
        avg_height : float
            Average CoM height for reference
            
        Returns:
        --------
        float
            Oscillation frequency in Hz
        """
        if len(heights) < 5:
            return 0.0
        
        # Method 1: Peak detection
        freq_peaks = self._frequency_from_peaks(heights, avg_height)
        
        # Method 2: Zero-crossing analysis (more robust for noisy data)
        freq_zero_cross = self._frequency_from_zero_crossings(heights, avg_height)
        
        # Method 3: Spectral analysis (if sufficient data)
        freq_spectral = 0.0
        if len(heights) >= 15:
            freq_spectral = self._frequency_from_spectrum(heights)
        
        # Combine methods with weighting
        frequencies = [f for f in [freq_peaks, freq_zero_cross, freq_spectral] if f > 0]
        
        if not frequencies:
            return 0.0
        
        # Use median for robustness
        return float(np.median(frequencies))
    
    def _frequency_from_peaks(self, heights: np.ndarray, avg_height: float) -> float:
        """Calculate frequency using peak detection."""
        # Find peaks above average
        peaks, _ = signal.find_peaks(heights, height=avg_height, distance=2)
        
        if len(peaks) < 2:
            return 0.0
        
        # Calculate time span and frequency
        time_span = len(heights) / self.frame_rate
        num_cycles = len(peaks)
        
        return num_cycles / time_span if time_span > 0 else 0.0
    
    def _frequency_from_zero_crossings(self, heights: np.ndarray, avg_height: float) -> float:
        """Calculate frequency using zero-crossing analysis."""
        # Center data around average
        centered = heights - avg_height
        
        # Count zero crossings (sign changes)
        zero_crossings = np.sum(np.diff(np.sign(centered)) != 0)
        
        if zero_crossings < 2:
            return 0.0
        
        # Each complete cycle has 2 zero crossings
        time_span = len(heights) / self.frame_rate
        num_cycles = zero_crossings / 2.0
        
        return num_cycles / time_span if time_span > 0 else 0.0
    
    def _frequency_from_spectrum(self, heights: np.ndarray) -> float:
        """Calculate frequency using spectral analysis."""
        try:
            # Apply window to reduce spectral leakage
            windowed = heights * signal.windows.hann(len(heights))
            
            # Compute power spectral density
            freqs, psd = signal.periodogram(windowed, fs=self.frame_rate)
            
            # Find dominant frequency (exclude DC component)
            if len(freqs) > 1:
                peak_idx = np.argmax(psd[1:]) + 1  # Skip DC component
                return float(freqs[peak_idx])
            
        except Exception as e:
            logger.debug(f"Spectral analysis failed: {e}")
        
        return 0.0
    
    def _get_efficiency_rating(self, oscillation_cm: float) -> str:
        """
        Determine biomechanical efficiency rating.
        
        Based on established research ranges for running efficiency.
        
        Parameters:
        -----------
        oscillation_cm : float
            Vertical oscillation in centimeters
            
        Returns:
        --------
        str
            Efficiency rating: 'excellent', 'good', 'moderate', or 'poor'
        """
        for rating, threshold in self.efficiency_thresholds.items():
            if oscillation_cm <= threshold:
                return rating
        
        return 'poor'  # Fallback
    
    def _assess_data_quality(self, heights: np.ndarray) -> str:
        """
        Assess overall quality of the height data.
        
        Parameters:
        -----------
        heights : np.ndarray
            Array of CoM height measurements
            
        Returns:
        --------
        str
            Quality assessment: 'excellent', 'good', 'moderate', or 'poor'
        """
        if len(heights) < self.min_data_points:
            return 'insufficient'
        
        # Calculate data quality metrics
        std_dev = np.std(heights)
        smoothness = self._calculate_smoothness_score(heights)
        
        # Quality assessment based on multiple criteria
        if std_dev > 0.05 or smoothness < 0.5:
            return 'poor'
        elif std_dev > 0.03 or smoothness < 0.7:
            return 'moderate'
        elif std_dev > 0.02 or smoothness < 0.85:
            return 'good'
        else:
            return 'excellent'
    
    def _calculate_smoothness_score(self, heights: np.ndarray) -> float:
        """
        Calculate smoothness score based on second derivative.
        
        Lower second derivative indicates smoother data.
        
        Parameters:
        -----------
        heights : np.ndarray
            Array of height measurements
            
        Returns:
        --------
        float
            Smoothness score between 0 and 1 (higher = smoother)
        """
        if len(heights) < 3:
            return 0.0
        
        # Calculate second derivative (curvature)
        second_derivative = np.diff(heights, n=2)
        curvature = np.mean(np.abs(second_derivative))
        
        # Convert to score (lower curvature = higher score)
        smoothness_score = 1.0 / (1.0 + curvature * 100)
        
        return float(np.clip(smoothness_score, 0.0, 1.0))
    
    def _initialize_result(self) -> OscillationResult:
        """Initialize default result dictionary."""
        return {
            'vertical_oscillation_cm': 0.0,
            'oscillation_frequency': 0.0,
            'efficiency_rating': 'insufficient_data',
            'avg_com_height': 0.0,
            'min_com_height': 0.0,
            'max_com_height': 0.0,
            'data_quality': 'insufficient',
            'frames_analyzed': 0,
            'calculation_successful': False,
            'std_deviation': 0.0,
            'smoothness_score': 0.0
        }
    
    def reset_analysis(self) -> None:
        """Reset analyzer for new analysis session."""
        logger.info("Resetting vertical oscillation analyzer")
        self.com_heights.clear()
        self.raw_heights.clear()
        self.frame_count = 0
    
    def get_analysis_summary(self) -> Dict[str, Union[str, float, int]]:
        """
        Get comprehensive analysis summary.
        
        Returns:
        --------
        Dict containing current analysis state and parameters
        """
        if len(self.com_heights) == 0:
            return {'status': 'no_data', 'frames_processed': self.frame_count}
        
        heights_array = np.array(self.com_heights)
        
        return {
            'status': 'active',
            'frames_processed': self.frame_count,
            'data_points': len(self.com_heights),
            'window_utilization': len(self.com_heights) / self.window_size,
            'current_oscillation_cm': (np.max(heights_array) - np.min(heights_array)) * self.scale_factor,
            'data_std_dev': np.std(heights_array),
            'smoothness_score': self._calculate_smoothness_score(heights_array),
            'frame_rate': self.frame_rate,
            'window_size': self.window_size
        }

def vertical_oscillation_wrapper(
    landmarks: LandmarksDict,
    detector_instance: Optional[VerticalOscillationAnalyzer]    
) -> OscillationResult:
    """
    Standalone wrapper function for vertical oscillation detection.
    
    Provides functional interface to VerticalOscillationAnalyzer class
    for integration with existing analysis pipelines.
    
    Args:
        detector: Initialized VerticalOscillationAnalyzer instance
        landmarks: Dictionary containing pose landmarks
        
    Returns:
        OscillationResult: Analysis results from detector
    """

        # Create or retrieve detector instance
    if detector_instance is not None:
        detector = detector_instance
    else:
        # Use function attribute to maintain detector across calls
        if not hasattr(vertical_oscillation_wrapper, '_default_detector'):
            vertical_oscillation_wrapper._default_detector = VerticalOscillationAnalyzer()
            logger.info("Created default VerticalOscillationAnalyzer instance")
        detector = vertical_oscillation_wrapper._default_detector

    return detector.get_analysis_summary(landmarks)

# Example usage and testing
def example_usage():
    """Demonstrate VerticalOscillationAnalyzer usage."""
    
    # Initialize analyzer
    analyzer = VerticalOscillationAnalyzer(
        frame_rate=30.0,
        window_size=20,
        scale_factor=150.0,  # Adjusted for camera distance
        smoothing_enabled=True
    )
    
    # Simulate running data with realistic oscillation pattern
    time_points = np.linspace(0, 2, 60)  # 2 seconds at 30fps
    
    for i, t in enumerate(time_points):
        # Simulate vertical oscillation with some noise
        base_height = 0.4
        oscillation = 0.03 * np.sin(2 * np.pi * 1.5 * t)  # 1.5 Hz oscillation
        noise = 0.005 * np.random.randn()
        
        simulated_landmarks = {
            'left_hip': (0.3, base_height + oscillation + noise, 0.0, 0.95),
            'right_hip': (0.7, base_height + oscillation + noise + 0.002, 0.0, 0.95)
        }
        
        result = analyzer.update(simulated_landmarks)
        
        # Print results every 10 frames
        if i % 10 == 9:
            print(f"Frame {i+1}:")
            print(f"  Vertical Oscillation: {result['vertical_oscillation_cm']:.1f} cm")
            print(f"  Frequency: {result['oscillation_frequency']:.2f} Hz")
            print(f"  Efficiency: {result['efficiency_rating']}")
            print(f"  Data Quality: {result['data_quality']}")
            print()
    
    # Final summary
    summary = analyzer.get_analysis_summary()
    print("Analysis Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    example_usage()