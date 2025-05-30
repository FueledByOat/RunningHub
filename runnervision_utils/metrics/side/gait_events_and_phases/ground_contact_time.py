# ground_contact_time.py

"""
Analyzes ground contact time and related metrics for running gait analysis.

This module tracks foot contact with the ground over time to calculate:
- Ground contact duration for each foot
- Contact time asymmetry between feet
- Running cadence
- Efficiency ratings based on contact time

Ground contact is determined by analyzing foot height relative to an estimated
ground level over a sliding window of frames.
"""

import logging
import numpy as np
from collections import deque
from typing import Dict, Tuple, Optional, Any, Union

logger = logging.getLogger(__name__)

# Type aliases for clarity and consistency
Landmark = Tuple[float, float, float, float]  # (x, y, z, visibility)
LandmarksDict = Dict[str, Landmark]
GroundContactResult = Dict[str, Union[float, str, int]]


class GroundContactTimeAnalyzer:
    """
    Analyzes ground contact time for running gait analysis.
    
    This class maintains internal history of foot positions and contact states
    to calculate ground contact metrics over time. It uses a sliding window
    approach to estimate ground level and detect contact/lift-off events.
    
    The analyzer tracks:
    - Individual foot contact durations
    - Contact time asymmetry between feet  
    - Running cadence (steps per minute)
    - Efficiency ratings based on contact time benchmarks
    
    Attributes:
        frame_rate (float): Video frame rate in fps
        history_size (int): Number of frames to maintain for analysis
        foot_height_threshold (float): Height threshold for contact detection
        frame_count (int): Total frames processed
        step_count (int): Total steps detected across both feet
    """
    
    def __init__(
        self, 
        frame_rate: float = 60.0, 
        history_size: int = 60, 
        foot_height_threshold: float = 0.02
    ):
        """
        Initialize the ground contact time analyzer.
        
        Args:
            frame_rate: Video frame rate in fps. Used for time calculations.
            history_size: Number of frames to maintain in sliding window for 
                         ground level estimation and contact analysis.
            foot_height_threshold: Normalized height threshold above estimated 
                                 ground level for determining foot contact.
                                 Smaller values are more sensitive.
        
        Raises:
            ValueError: If parameters are outside valid ranges.
        """
        # Validate input parameters
        if frame_rate <= 0:
            raise ValueError(f"Frame rate must be positive, got {frame_rate}")
        if history_size <= 0:
            raise ValueError(f"History size must be positive, got {history_size}")
        if foot_height_threshold < 0:
            raise ValueError(f"Foot height threshold must be non-negative, got {foot_height_threshold}")
            
        self.frame_rate = frame_rate
        self.history_size = history_size
        self.foot_height_threshold = foot_height_threshold
        
        # Sliding windows for foot height tracking
        self.left_foot_heights = deque(maxlen=history_size)
        self.right_foot_heights = deque(maxlen=history_size)
        
        # Contact state tracking
        self.left_contact_states = deque(maxlen=history_size)
        self.right_contact_states = deque(maxlen=history_size)
        
        # Contact period tracking
        self.left_contact_start: Optional[int] = None
        self.right_contact_start: Optional[int] = None
        
        # Store recent contact durations for averaging
        self.left_contact_times = deque(maxlen=10)
        self.right_contact_times = deque(maxlen=10)
        
        # Frame and step counters
        self.frame_count = 0
        self.step_count = 0
        
        logger.info(f"Initialized GroundContactTimeAnalyzer with frame_rate={frame_rate}, "
                   f"history_size={history_size}, threshold={foot_height_threshold}")
    
    def update(self, landmarks: LandmarksDict) -> GroundContactResult:
        """
        Update analyzer with new frame data and calculate ground contact metrics.
        
        Processes a new frame of landmark data to:
        1. Extract foot positions
        2. Update contact state tracking
        3. Detect contact/lift-off transitions
        4. Calculate current metrics
        
        Args:
            landmarks: Dictionary containing pose landmarks. Must include 
                      'left_ankle' and 'right_ankle' keys with (x,y,z,visibility) tuples.
                      Y-coordinates should be normalized (0.0-1.0) with 0 at top.
        
        Returns:
            GroundContactResult: Dictionary containing calculated metrics:
                - left_foot_contact_time_ms: Average left foot contact time
                - right_foot_contact_time_ms: Average right foot contact time  
                - avg_contact_time_ms: Overall average contact time
                - contact_time_ratio: Left/right contact time ratio
                - efficiency_rating: Qualitative efficiency assessment
                - cadence_spm: Running cadence in steps per minute
                - total_steps_detected: Total steps counted
                - data_quality: 'sufficient' or 'insufficient'
                - calculation_successful: True if calculation completed
        """
        # Initialize result structure
        result: GroundContactResult = {
            'left_foot_contact_time_ms': 0.0,
            'right_foot_contact_time_ms': 0.0,
            'avg_contact_time_ms': 0.0,
            'contact_time_ratio': 1.0,
            'efficiency_rating': 'insufficient_data',
            'cadence_spm': 0.0,
            'total_steps_detected': 0,
            'data_quality': 'insufficient',
            'calculation_successful': False
        }
        
        # Validate required landmarks
        required_landmarks = ['left_ankle', 'right_ankle']
        for landmark_name in required_landmarks:
            if landmark_name not in landmarks:
                logger.warning(f"Required landmark '{landmark_name}' not found for ground contact analysis.")
                return result
        
        try:
            # Extract foot positions (y-coordinate for height)
            left_ankle_y = landmarks['left_ankle'][1]
            right_ankle_y = landmarks['right_ankle'][1]
            
            # Update height history
            self.left_foot_heights.append(left_ankle_y)
            self.right_foot_heights.append(right_ankle_y)
            self.frame_count += 1
            
            # Estimate ground levels for each foot
            left_ground_level = self._estimate_ground_level(self.left_foot_heights)
            right_ground_level = self._estimate_ground_level(self.right_foot_heights)
            
            # Determine current contact states
            left_contact = left_ankle_y <= (left_ground_level + self.foot_height_threshold)
            right_contact = right_ankle_y <= (right_ground_level + self.foot_height_threshold)
            
            # Update contact state history
            self.left_contact_states.append(left_contact)
            self.right_contact_states.append(right_contact)
            
            # Track contact period transitions
            self._update_contact_tracking('left', left_contact)
            self._update_contact_tracking('right', right_contact)
            
            # Calculate metrics if sufficient data available
            if len(self.left_foot_heights) >= 10:
                result = self._calculate_metrics()
                result['calculation_successful'] = True
            else:
                logger.debug(f"Insufficient data for analysis: {len(self.left_foot_heights)}/10 frames")
                
        except KeyError as e:
            logger.error(f"Missing landmark data during ground contact analysis: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error during ground contact analysis: {e}")
            
        return result
    
    def _estimate_ground_level(self, foot_heights: deque) -> float:
        """
        Estimate ground level from recent foot height positions.
        
        Uses the 10th percentile of recent foot positions as ground level
        estimate, which is robust to outliers and gait variations.
        
        Args:
            foot_heights: Deque of recent foot height values
            
        Returns:
            float: Estimated ground level (y-coordinate)
        """
        if len(foot_heights) < 5:
            return min(foot_heights) if foot_heights else 0.0
        
        # Use 10th percentile as robust ground level estimate
        heights_array = np.array(foot_heights)
        ground_level = np.percentile(heights_array, 10)
        
        return ground_level
    
    def _update_contact_tracking(self, foot: str, is_contact: bool) -> None:
        """
        Track contact period transitions for each foot.
        
        Detects contact start/end events and calculates contact durations.
        Updates step count and contact time history.
        
        Args:
            foot: Either 'left' or 'right'
            is_contact: Current contact state for the specified foot
        """
        if foot == 'left':
            contact_start = self.left_contact_start
            contact_times = self.left_contact_times
        elif foot == 'right':
            contact_start = self.right_contact_start
            contact_times = self.right_contact_times
        else:
            logger.warning(f"Invalid foot identifier: {foot}")
            return
        
        # Detect contact start
        if is_contact and contact_start is None:
            if foot == 'left':
                self.left_contact_start = self.frame_count
            else:
                self.right_contact_start = self.frame_count
            logger.debug(f"{foot.capitalize()} foot contact started at frame {self.frame_count}")
        
        # Detect contact end and calculate duration
        elif not is_contact and contact_start is not None:
            contact_duration_frames = self.frame_count - contact_start
            contact_duration_ms = (contact_duration_frames / self.frame_rate) * 1000
            
            # Store contact time and increment step counter
            contact_times.append(contact_duration_ms)
            self.step_count += 1
            
            logger.debug(f"{foot.capitalize()} foot contact ended. Duration: {contact_duration_ms:.1f}ms")
            
            # Reset contact start tracking
            if foot == 'left':
                self.left_contact_start = None
            else:
                self.right_contact_start = None
    
    def _calculate_metrics(self) -> GroundContactResult:
        """
        Calculate comprehensive ground contact time metrics.
        
        Computes averages, ratios, cadence, and efficiency ratings from
        the accumulated contact time data.
        
        Returns:
            GroundContactResult: Dictionary of calculated metrics
        """
        # Calculate average contact times
        left_avg = np.mean(self.left_contact_times) if self.left_contact_times else 0.0
        right_avg = np.mean(self.right_contact_times) if self.right_contact_times else 0.0
        
        # Overall average contact time
        if left_avg > 0 or right_avg > 0:
            avg_contact_time_ms = (left_avg + right_avg) / 2
        else:
            avg_contact_time_ms = 0.0
        
        # Contact time asymmetry ratio
        if right_avg > 0:
            contact_time_ratio = left_avg / right_avg
        else:
            contact_time_ratio = 1.0 if left_avg == 0 else float('inf')
        
        # Calculate running cadence
        time_span_minutes = len(self.left_foot_heights) / (self.frame_rate * 60)
        total_steps = len(self.left_contact_times) + len(self.right_contact_times)
        
        if time_span_minutes > 0:
            cadence_spm = total_steps / time_span_minutes
        else:
            cadence_spm = 0.0
        
        # Determine efficiency rating
        efficiency_rating = self._assess_efficiency(avg_contact_time_ms)
        
        return {
            'left_foot_contact_time_ms': left_avg,
            'right_foot_contact_time_ms': right_avg,
            'avg_contact_time_ms': avg_contact_time_ms,
            'contact_time_ratio': contact_time_ratio,
            'efficiency_rating': efficiency_rating,
            'cadence_spm': cadence_spm,
            'total_steps_detected': total_steps,
            'data_quality': 'sufficient',
            'calculation_successful': True
        }
    
    def _assess_efficiency(self, contact_time_ms: float) -> str:
        """
        Determine running efficiency rating based on ground contact time.
        
        Based on research indicating that elite runners typically have
        shorter ground contact times:
        - Excellent: ≤180ms (elite level)
        - Good: 181-220ms (competitive recreational)  
        - Moderate: 221-280ms (recreational)
        - Poor: >280ms (inefficient)
        
        Args:
            contact_time_ms: Average ground contact time in milliseconds
            
        Returns:
            str: Efficiency rating category
        """
        if contact_time_ms <= 180:
            return 'excellent'
        elif contact_time_ms <= 220:
            return 'good'
        elif contact_time_ms <= 280:
            return 'moderate'
        else:
            return 'poor'
    
    def reset(self) -> None:
        """
        Reset the analyzer state for processing a new sequence.
        
        Clears all accumulated data while preserving configuration parameters.
        Useful when starting analysis of a new video or runner.
        """
        self.left_foot_heights.clear()
        self.right_foot_heights.clear()
        self.left_contact_states.clear()
        self.right_contact_states.clear()
        
        self.left_contact_start = None
        self.right_contact_start = None
        self.left_contact_times.clear()
        self.right_contact_times.clear()
        
        self.frame_count = 0
        self.step_count = 0
        
        logger.info("GroundContactTimeAnalyzer state reset")


def ground_contact_wrapper(landmarks: LandmarksDict, detector_instance: Optional[GroundContactTimeAnalyzer] ) -> GroundContactResult:
    """
    Standalone wrapper function for ground contact analysis.
    
    Provides a functional interface to the GroundContactTimeAnalyzer class
    for integration with existing analysis pipelines.
    
    Args:
        analyzer: Initialized GroundContactTimeAnalyzer instance
        landmarks: Dictionary containing pose landmarks
        
    Returns:
        GroundContactResult: Analysis results from the analyzer
    """
        # Create or retrieve detector instance
    if detector_instance is not None:
        detector = detector_instance
    else:
        # Use function attribute to maintain detector across calls
        if not hasattr(ground_contact_wrapper, '_default_detector'):
            ground_contact_wrapper._default_detector = GroundContactTimeAnalyzer()
            logger.info("Created default GroundContactTimeAnalyzer instance")
        detector = ground_contact_wrapper._default_detector

    return analyzer.update(landmarks)


if __name__ == "__main__":
    print("Testing GroundContactTimeAnalyzer...")
    
    # Configure logging for testing
    logging.basicConfig(level=logging.INFO)
    
    # Initialize analyzer
    analyzer = GroundContactTimeAnalyzer(frame_rate=30.0, history_size=30)
    
    # Simulate running gait data
    import math
    
    print("\nSimulating running gait data...")
    for frame in range(100):
        # Simulate alternating foot contact pattern
        left_phase = math.sin(frame * 0.2) * 0.05 + 0.85  # Ground at ~0.9
        right_phase = math.sin(frame * 0.2 + math.pi) * 0.05 + 0.85
        
        sample_landmarks = {
            'left_ankle': (0.4, left_phase, 0, 0.95),
            'right_ankle': (0.6, right_phase, 0, 0.95)
        }
        
        result = analyzer.update(sample_landmarks)
        
        # Print results every 20 frames
        if frame % 20 == 0 and frame > 0:
            print(f"\nFrame {frame} Results:")
            print(f"  Left contact time: {result['left_foot_contact_time_ms']:.1f}ms")
            print(f"  Right contact time: {result['right_foot_contact_time_ms']:.1f}ms")
            print(f"  Average contact time: {result['avg_contact_time_ms']:.1f}ms")
            print(f"  Contact ratio: {result['contact_time_ratio']:.2f}")
            print(f"  Efficiency: {result['efficiency_rating']}")
            print(f"  Cadence: {result['cadence_spm']:.1f} spm")
            print(f"  Steps detected: {result['total_steps_detected']}")
    
    print(f"\nFinal analysis complete. Total steps: {analyzer.step_count}")