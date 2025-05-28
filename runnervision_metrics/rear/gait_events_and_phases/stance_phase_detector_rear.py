# stance_phase_detector_rear.py

"""
Rear-view stance phase detection for running biomechanics analysis.

This module provides functionality to detect stance phase (ground contact) during running
from rear-view camera angles using MediaPipe Pose landmarks. The detector uses vertical
foot position relative to calibrated ground and swing zones to determine when feet are
in contact with the ground.

Key Features:
- Automatic calibration from observed foot motion
- Robust detection using multiple foot landmarks
- Confidence scoring for detection reliability
- Handles missing or low-visibility landmarks gracefully
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union

# Configure module logger
logger = logging.getLogger(__name__)

# Type aliases for clarity and maintainability
Landmark = Tuple[float, float, float, float]  # (x, y, z, visibility)
LandmarksDict = Dict[str, Landmark]
StanceResult = Dict[str, Union[bool, str, float, None]]
FootLandmarksConfig = Dict[str, List[str]]


class StancePhaseDetectorRear:
    """
    Detects stance phase from rear-view running analysis using foot landmark positions.
    
    This detector works by:
    1. Calibrating ground and swing zones from observed foot motion
    2. Comparing current foot positions to these zones
    3. Determining which foot (if any) is in ground contact
    4. Providing confidence scores for detection reliability
    
    The detector assumes normalized MediaPipe coordinates where:
    - Y=0 is top of frame, Y=1 is bottom of frame
    - Higher Y values indicate lower vertical positions (closer to ground)
    """
    
    def __init__(self, 
                 calibration_frames_total: int = 90,
                 ground_zone_percentage: float = 0.15,
                 visibility_threshold: float = 0.5,
                 foot_landmarks_to_use: Optional[FootLandmarksConfig] = None) -> None:
        """
        Initialize the rear-view stance phase detector.
        
        Args:
            calibration_frames_total: Number of frames for calibration phase.
                Recommended: 90-150 frames (3-5 seconds at 30fps) to capture
                full range of motion during running.
            ground_zone_percentage: Percentage of foot's vertical range that defines
                ground contact zone. Lower values = stricter ground detection.
                Typical range: 0.10-0.25.
            visibility_threshold: Minimum MediaPipe visibility score (0.0-1.0) 
                required for landmark to be considered reliable.
            foot_landmarks_to_use: Custom landmark configuration for each foot.
                If None, uses default foot_index and heel landmarks.
                Format: {'left': ['landmark1', 'landmark2'], 'right': [...]}
        
        Raises:
            ValueError: If parameters are outside valid ranges.
        """
        # Validate input parameters
        if not 30 <= calibration_frames_total <= 300:
            raise ValueError("calibration_frames_total should be between 30-300 frames")
        if not 0.05 <= ground_zone_percentage <= 0.5:
            raise ValueError("ground_zone_percentage should be between 0.05-0.5")
        if not 0.1 <= visibility_threshold <= 1.0:
            raise ValueError("visibility_threshold should be between 0.1-1.0")
        
        # Configuration parameters
        self.calibration_frames_total = calibration_frames_total
        self.ground_zone_percentage = ground_zone_percentage  
        self.visibility_threshold = visibility_threshold
        
        # Calibration state
        self.frames_calibrated = 0
        self._foot_y_samples_normalized: List[float] = []
        
        # Calibrated thresholds (set during calibration)
        self.calibrated_overall_max_y: Optional[float] = None  # Ground level proxy
        self.calibrated_overall_min_y: Optional[float] = None  # Peak swing proxy
        self.ground_contact_entry_threshold_y: Optional[float] = None  # Stance zone entry
        
        # Configure foot landmarks to monitor
        if foot_landmarks_to_use is None:
            self.foot_landmarks_to_check: FootLandmarksConfig = {
                'left': ['left_foot_index', 'left_heel'],
                'right': ['right_foot_index', 'right_heel']
            }
        else:
            self.foot_landmarks_to_check = foot_landmarks_to_use
            
        logger.info(f"Initialized StancePhaseDetectorRear: calibration_frames={calibration_frames_total}, "
                   f"ground_zone_pct={ground_zone_percentage}, visibility_threshold={visibility_threshold}")
    
    def _get_lowest_point_of_foot(self, landmarks: LandmarksDict, side_key: str) -> Optional[float]:
        """
        Get the lowest Y coordinate for a foot, considering landmark visibility.
        
        Args:
            landmarks: Dictionary of pose landmarks
            side_key: 'left' or 'right' foot identifier
            
        Returns:
            Lowest (highest Y value) visible point of the foot, or None if no
            landmarks are visible above threshold.
            
        Note:
            In normalized coordinates, max(Y) represents the lowest visual point
            since Y increases downward from top of frame.
        """
        foot_y_values = []
        
        if side_key not in self.foot_landmarks_to_check:
            logger.warning(f"Unknown foot side_key: {side_key}")
            return None
            
        for landmark_name in self.foot_landmarks_to_check[side_key]:
            if (landmark_name in landmarks and 
                landmarks[landmark_name][3] >= self.visibility_threshold):
                foot_y_values.append(landmarks[landmark_name][1])  # Y coordinate
                
        if not foot_y_values:
            logger.debug(f"No visible landmarks for {side_key} foot above threshold {self.visibility_threshold}")
            return None
            
        return max(foot_y_values)  # Lowest point on screen (highest Y value)
    
    def _collect_calibration_data(self, landmarks: LandmarksDict) -> None:
        """
        Collect foot position data during calibration phase.
        
        Stores the lowest point of each visible foot to build understanding
        of the runner's range of motion and ground contact patterns.
        
        Args:
            landmarks: Current frame's pose landmarks
        """
        left_lowest_y = self._get_lowest_point_of_foot(landmarks, 'left')
        right_lowest_y = self._get_lowest_point_of_foot(landmarks, 'right')
        
        # Collect all available foot position data
        for foot_y, side in [(left_lowest_y, 'left'), (right_lowest_y, 'right')]:
            if foot_y is not None:
                self._foot_y_samples_normalized.append(foot_y)
                logger.debug(f"Calibration frame {self.frames_calibrated}: {side} foot Y={foot_y:.3f}")
    
    def _finalize_calibration(self) -> None:
        """
        Calculate detection thresholds from collected calibration data.
        
        Uses robust statistical methods (percentiles) to establish:
        - Ground level (where feet contact during stance)
        - Peak swing level (highest point during swing phase)  
        - Ground contact threshold (boundary for stance detection)
        
        Handles edge cases like insufficient data or minimal foot motion.
        """
        min_required_samples = self.calibration_frames_total * 0.5
        
        if (not self._foot_y_samples_normalized or 
            len(self._foot_y_samples_normalized) < min_required_samples):
            logger.warning(f"Insufficient calibration data: {len(self._foot_y_samples_normalized)} samples "
                          f"(need {min_required_samples}). Using default values.")
            self._set_default_calibration_values()
            return
            
        try:
            # Use percentiles for robustness against outliers
            self.calibrated_overall_max_y = np.percentile(self._foot_y_samples_normalized, 95)  # Ground level
            self.calibrated_overall_min_y = np.percentile(self._foot_y_samples_normalized, 5)   # Peak swing
            
            # Validate calibration makes sense
            self._validate_and_adjust_calibration()
            
            # Calculate stance detection threshold
            height_range = self.calibrated_overall_max_y - self.calibrated_overall_min_y
            
            if height_range <= 0.02:  # Very small motion range
                logger.warning(f"Small foot motion range detected: {height_range:.3f}. "
                              "Detection may be sensitive to noise.")
                self.ground_contact_entry_threshold_y = self.calibrated_overall_max_y - 0.015
            else:
                # Threshold is ground_zone_percentage up from the lowest point
                self.ground_contact_entry_threshold_y = (
                    self.calibrated_overall_max_y - (height_range * self.ground_zone_percentage)
                )
            
            logger.info(f"Calibration complete: Ground={self.calibrated_overall_max_y:.3f}, "
                       f"Peak={self.calibrated_overall_min_y:.3f}, "
                       f"Threshold={self.ground_contact_entry_threshold_y:.3f}, "
                       f"Range={height_range:.3f}, Samples={len(self._foot_y_samples_normalized)}")
                       
        except Exception as e:
            logger.error(f"Calibration failed: {e}", exc_info=True)
            self._set_default_calibration_values()
    
    def _set_default_calibration_values(self) -> None:
        """Set reasonable default values when calibration fails or has insufficient data."""
        self.calibrated_overall_max_y = 0.90  # Assume ground near bottom of frame
        self.calibrated_overall_min_y = 0.50  # Assume swing peak at mid-frame
        height_range = self.calibrated_overall_max_y - self.calibrated_overall_min_y
        self.ground_contact_entry_threshold_y = (
            self.calibrated_overall_max_y - (height_range * self.ground_zone_percentage)
        )
        logger.info("Using default calibration values")
    
    def _validate_and_adjust_calibration(self) -> None:
        """
        Ensure calibration values are reasonable and adjust if necessary.
        
        Checks that ground level is actually lower than peak swing level,
        and adjusts if the detected range seems inverted or too small.
        """
        min_reasonable_range = 0.05
        
        if self.calibrated_overall_max_y <= self.calibrated_overall_min_y + min_reasonable_range:
            logger.warning("Calibration issue: ground not sufficiently below peak swing. Adjusting.")
            
            if self._foot_y_samples_normalized:
                # Use most extreme observed values
                self.calibrated_overall_max_y = max(self._foot_y_samples_normalized)
                self.calibrated_overall_min_y = min(
                    self.calibrated_overall_max_y - 0.1,  # Ensure minimum range
                    min(self._foot_y_samples_normalized)
                )
            else:
                self._set_default_calibration_values()
    
    def _calculate_stance_confidence(self, 
                                   foot_y: float, 
                                   is_in_stance: bool, 
                                   other_foot_y: Optional[float] = None) -> float:
        """
        Calculate confidence score for stance/flight detection.
        
        Args:
            foot_y: Y position of the foot in question
            is_in_stance: Whether foot is detected as being in stance
            other_foot_y: Y position of other foot (for flight confidence calculation)
            
        Returns:
            Confidence score between 0.0-1.0, where:
            - 0.5+ indicates stance phase confidence
            - <0.5 indicates flight phase confidence
            - Higher values = more confident
        """
        if is_in_stance:
            # Confidence based on depth into stance zone
            stance_zone_height = self.calibrated_overall_max_y - self.ground_contact_entry_threshold_y
            if stance_zone_height > 1e-5:
                depth_ratio = (foot_y - self.ground_contact_entry_threshold_y) / stance_zone_height
                confidence = 0.5 + min(max(depth_ratio, 0), 1) * 0.49  # Scale 0.5-0.99
            else:
                confidence = 0.6  # At threshold line
        else:
            # Flight confidence based on clearance from stance threshold
            flight_zone_height = self.ground_contact_entry_threshold_y - self.calibrated_overall_min_y
            if flight_zone_height > 1e-5 and foot_y < self.ground_contact_entry_threshold_y:
                clearance_ratio = (self.ground_contact_entry_threshold_y - foot_y) / flight_zone_height
                confidence = 0.5 + min(max(clearance_ratio, 0), 1) * 0.49
            else:
                confidence = 0.2  # Near threshold but not clearly in flight
                
        return round(confidence, 3)
    
    def detect_stance_phase(self, landmarks: LandmarksDict) -> StanceResult:
        """
        Detect stance phase from current frame landmarks.
        
        Args:
            landmarks: Dictionary of MediaPipe pose landmarks with format:
                      {landmark_name: (x, y, z, visibility), ...}
                      
        Returns:
            Dictionary containing:
            - 'is_stance_phase' (bool): True if either foot is in ground contact
            - 'stance_foot' (str|None): 'left', 'right', or None
            - 'confidence' (float): Detection confidence score (0.0-1.0)
            - 'debug' (str, optional): Debug information for troubleshooting
            
        Note:
            During calibration phase, always returns is_stance_phase=False
            with debug info indicating calibration status.
        """
        # Handle calibration phase
        if self.frames_calibrated < self.calibration_frames_total:
            self._collect_calibration_data(landmarks)
            self.frames_calibrated += 1
            
            if self.frames_calibrated == self.calibration_frames_total:
                self._finalize_calibration()
                logger.info("Calibration phase completed, stance detection now active")
                
            return {
                'is_stance_phase': False, 
                'stance_foot': None, 
                'confidence': 0.0,
                'debug': f"calibrating ({self.frames_calibrated}/{self.calibration_frames_total})"
            }
        
        # Ensure calibration was successful
        if self.ground_contact_entry_threshold_y is None:
            logger.error("Stance detector not properly calibrated")
            return {
                'is_stance_phase': False, 
                'stance_foot': None, 
                'confidence': 0.0,
                'debug': "calibration_failed"
            }
        
        # Get current foot positions
        left_lowest_y = self._get_lowest_point_of_foot(landmarks, 'left')
        right_lowest_y = self._get_lowest_point_of_foot(landmarks, 'right')
        
        # Handle missing landmark data
        if left_lowest_y is None and right_lowest_y is None:
            logger.debug("No visible foot landmarks in current frame")
            return {
                'is_stance_phase': False, 
                'stance_foot': None, 
                'confidence': 0.0,
                'debug': "no_foot_data"
            }
        
        # Determine stance status for each foot
        left_in_stance = (left_lowest_y is not None and 
                         left_lowest_y >= self.ground_contact_entry_threshold_y)
        right_in_stance = (right_lowest_y is not None and 
                          right_lowest_y >= self.ground_contact_entry_threshold_y)
        
        # Determine overall phase and primary stance foot
        if left_in_stance and right_in_stance:
            # Double support - choose the lower foot as primary
            is_stance_phase = True
            stance_foot = 'left' if left_lowest_y > right_lowest_y else 'right'
            active_foot_y = left_lowest_y if stance_foot == 'left' else right_lowest_y
            logger.debug(f"Double support detected, primary foot: {stance_foot}")
        elif left_in_stance:
            is_stance_phase = True
            stance_foot = 'left'
            active_foot_y = left_lowest_y
        elif right_in_stance:
            is_stance_phase = True
            stance_foot = 'right' 
            active_foot_y = right_lowest_y
        else:
            # Flight phase
            is_stance_phase = False
            stance_foot = None
            # Use higher foot for confidence calculation
            active_foot_y = min(
                y for y in [left_lowest_y, right_lowest_y] if y is not None
            )
        
        # Calculate confidence score
        confidence = self._calculate_stance_confidence(
            active_foot_y, is_stance_phase, 
            right_lowest_y if stance_foot == 'left' else left_lowest_y
        )
        
        result = {
            'is_stance_phase': is_stance_phase,
            'stance_foot': stance_foot, 
            'confidence': confidence
        }
        
        logger.debug(f"Stance detection: {result}, foot_y={active_foot_y:.3f}")
        return result
    
    def reset_calibration(self) -> None:
        """
        Reset calibration state to allow re-calibration with new data.
        
        Useful when switching between different runners or significantly
        different running conditions.
        """
        self.frames_calibrated = 0
        self._foot_y_samples_normalized.clear()
        self.calibrated_overall_max_y = None
        self.calibrated_overall_min_y = None
        self.ground_contact_entry_threshold_y = None
        logger.info("Calibration state reset")
    
    @property
    def is_calibrated(self) -> bool:
        """Check if detector has completed calibration."""
        return (self.frames_calibrated >= self.calibration_frames_total and 
                self.ground_contact_entry_threshold_y is not None)
    
    def get_calibration_info(self) -> Dict[str, Any]:
        """
        Get current calibration parameters and status.
        
        Returns:
            Dictionary with calibration state information for debugging
            and validation purposes.
        """
        return {
            'is_calibrated': self.is_calibrated,
            'frames_calibrated': self.frames_calibrated,
            'total_calibration_frames': self.calibration_frames_total,
            'samples_collected': len(self._foot_y_samples_normalized),
            'ground_level_y': self.calibrated_overall_max_y,
            'peak_swing_y': self.calibrated_overall_min_y, 
            'stance_threshold_y': self.ground_contact_entry_threshold_y,
            'ground_zone_percentage': self.ground_zone_percentage,
            'visibility_threshold': self.visibility_threshold
        }


def detect_stance_phase_rear(landmarks: LandmarksDict, 
                           detector_instance: Optional[StancePhaseDetectorRear] = None) -> StanceResult:
    """
    Convenience wrapper function for rear-view stance phase detection.
    
    This function provides a simple interface for stance detection while managing
    the detector instance lifecycle. It's designed to be called frame-by-frame
    in a video processing pipeline.
    
    Args:
        landmarks: Dictionary of MediaPipe pose landmarks for current frame
        detector_instance: Optional pre-configured detector instance. If None,
                         a new detector with default settings will be created
                         and stored for subsequent calls.
                         
    Returns:
        Stance detection results dictionary (see StancePhaseDetectorRear.detect_stance_phase)
        
    Example:
        >>> landmarks = get_mediapipe_landmarks(frame)
        >>> result = detect_stance_phase_rear(landmarks)
        >>> if result['is_stance_phase']:
        >>>     print(f"Stance foot: {result['stance_foot']}, confidence: {result['confidence']}")
        
    Note:
        This wrapper maintains a single detector instance across calls using a
        function attribute. For multiple concurrent analyses, create separate
        StancePhaseDetectorRear instances directly.
    """
    # Create or retrieve detector instance
    if detector_instance is not None:
        detector = detector_instance
    else:
        # Use function attribute to maintain detector across calls
        if not hasattr(detect_stance_phase_rear, '_default_detector'):
            detect_stance_phase_rear._default_detector = StancePhaseDetectorRear()
            logger.info("Created default StancePhaseDetectorRear instance")
        detector = detect_stance_phase_rear._default_detector
    
    return detector.detect_stance_phase(landmarks)


# Example usage and testing
if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("Testing StancePhaseDetectorRear...")
    
    # Create test detector
    detector = StancePhaseDetectorRear(calibration_frames_total=30, ground_zone_percentage=0.2)
    
    # Simulate calibration with test data
    test_landmarks = {
        'left_foot_index': (0.3, 0.8, 0, 0.95),   # Ground contact
        'left_heel': (0.25, 0.82, 0, 0.90),
        'right_foot_index': (0.7, 0.6, 0, 0.95),  # Swing phase
        'right_heel': (0.75, 0.58, 0, 0.90)
    }
    
    print("\nCalibration phase:")
    for i in range(30):
        # Vary foot positions to simulate running motion
        test_landmarks['left_foot_index'] = (0.3, 0.8 - 0.3 * np.sin(i * 0.2), 0, 0.95)
        test_landmarks['right_foot_index'] = (0.7, 0.8 - 0.3 * np.sin(i * 0.2 + np.pi), 0, 0.95)
        
        result = detector.detect_stance_phase(test_landmarks)
        if i % 10 == 0:
            print(f"Frame {i}: {result}")
    
    print(f"\nCalibration info: {detector.get_calibration_info()}")
    
    # Test stance detection
    print("\nStance detection phase:")
    
    # Test left foot stance
    test_landmarks['left_foot_index'] = (0.3, 0.85, 0, 0.95)  # Ground contact
    test_landmarks['right_foot_index'] = (0.7, 0.6, 0, 0.95)  # Swing
    result = detector.detect_stance_phase(test_landmarks)
    print(f"Left stance: {result}")
    
    # Test right foot stance  
    test_landmarks['left_foot_index'] = (0.3, 0.6, 0, 0.95)   # Swing
    test_landmarks['right_foot_index'] = (0.7, 0.85, 0, 0.95) # Ground contact
    result = detector.detect_stance_phase(test_landmarks)
    print(f"Right stance: {result}")
    
    # Test flight phase
    test_landmarks['left_foot_index'] = (0.3, 0.6, 0, 0.95)   # Both in swing
    test_landmarks['right_foot_index'] = (0.7, 0.6, 0, 0.95)
    result = detector.detect_stance_phase(test_landmarks)
    print(f"Flight phase: {result}")
    
    # Test wrapper function
    print(f"\nWrapper function test: {detect_stance_phase_rear(test_landmarks)}")
    
    print("\nTesting complete!")