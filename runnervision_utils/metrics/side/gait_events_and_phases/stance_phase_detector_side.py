# stance_phase_detector_side.py

"""
Side-view stance phase detection for running biomechanics analysis.

This module provides stance phase detection capabilities for runners captured from 
a side view using MediaPipe BlazePose landmarks. The detector uses ground contact 
proximity and optional velocity analysis to determine when each foot is in contact 
with the ground during the running gait cycle.

Key Features:
- Automatic calibration of ground level and runner dimensions
- Dynamic threshold adjustment based on runner's apparent height
- Confidence scoring for stance phase predictions
- Support for both left and right foot detection
- Robust handling of landmark visibility variations

Biomechanical Context:
- Stance phase: Period when foot is in contact with ground (~40% of gait cycle)
- Flight phase: Period when both feet are airborne (~60% of gait cycle)
- Critical for analyzing ground reaction forces, impact loading, and propulsion
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

# Configure module logger
logger = logging.getLogger(__name__)

# Type aliases for clarity
Landmark = Tuple[float, float, float, float]  # (x, y, z, visibility)
LandmarksDict = Dict[str, Landmark]
StanceResult = Dict[str, Union[bool, str, float, None]]
CalibrationData = Dict[str, Union[float, List[float], None]]


class StancePhaseDetectorSide:
    """
    Detects stance phase from side-view running footage using MediaPipe BlazePose landmarks.
    
    This detector performs automatic calibration during initial frames to establish ground
    level and runner dimensions, then uses proximity-based detection with optional velocity
    analysis to determine foot-ground contact phases.
    
    The detection algorithm accounts for:
    - Varying runner heights and camera distances through dynamic scaling
    - Landmark visibility variations in real-world footage  
    - Single vs double stance phases (rare in running but possible at transitions)
    - Confidence scoring based on detection certainty
    
    Attributes:
        calibration_frames (int): Number of frames used for initial calibration
        stance_threshold_ratio (float): Stance detection sensitivity as ratio of runner height
        visibility_threshold (float): Minimum landmark visibility score (0.0-1.0)
        ground_level_normalized (Optional[float]): Calibrated ground Y-coordinate
        avg_runner_height_normalized (Optional[float]): Calibrated runner height in normalized coords
        frame_count (int): Current frame number for calibration tracking
    """
    
    def __init__(
        self, 
        calibration_frames: int = 90,
        stance_threshold_ratio: float = 0.018,
        visibility_threshold: float = 0.5
    ) -> None:
        """
        Initialize the stance phase detector with calibration parameters.
        
        Parameters:
        -----------
        calibration_frames : int, default=90
            Number of initial frames used for ground level and runner height calibration.
            More frames provide better calibration but delay analysis start. 
            90 frames ≈ 3 seconds at 30fps.
            
        stance_threshold_ratio : float, default=0.018
            Sensitivity threshold for stance detection as ratio of runner's apparent height.
            Smaller values = stricter detection (foot must be very close to ground).
            Typical range: 0.01-0.03 (1-3% of runner height).
            
        visibility_threshold : float, default=0.5
            Minimum MediaPipe landmark visibility score for landmark to be considered valid.
            Range: 0.0-1.0, where 1.0 = fully visible, 0.0 = not detected.
            
        Note:
        -----
        All coordinate calculations assume normalized MediaPipe coordinates (0.0-1.0)
        where (0,0) is top-left and (1,1) is bottom-right of the image frame.
        """
        # Validation of input parameters
        if calibration_frames < 30:
            logger.warning(f"Calibration frames ({calibration_frames}) may be too low. "
                         f"Recommend ≥30 for stable calibration.")
        if not 0.005 <= stance_threshold_ratio <= 0.05:
            logger.warning(f"Stance threshold ratio ({stance_threshold_ratio}) outside "
                         f"typical range (0.005-0.05). May affect detection accuracy.")
        if not 0.0 <= visibility_threshold <= 1.0:
            raise ValueError(f"Visibility threshold must be 0.0-1.0, got {visibility_threshold}")
            
        # Core detection parameters
        self.calibration_frames = calibration_frames
        self.stance_threshold_ratio = stance_threshold_ratio
        self.visibility_threshold = visibility_threshold
        
        # Calibration state tracking
        self.frame_count = 0
        self.ground_y_samples_normalized: List[float] = []
        self.runner_height_samples_normalized: List[float] = []
        
        # Calibrated values (None until calibration complete)
        self.ground_level_normalized: Optional[float] = None
        self.avg_runner_height_normalized: Optional[float] = None
        
        # Velocity-based detection (optional enhancement)
        self.foot_y_history: Dict[str, List[float]] = {'left': [], 'right': []}
        self.max_history_len = 3  # Frames for velocity calculation buffer
        self.foot_velocity_threshold_normalized = 0.01  # Velocity threshold for stability detection
        
        # Landmark configuration for foot and head detection
        self.foot_landmark_names = {
            'right': ['right_heel', 'right_foot_index', 'right_ankle'],
            'left': ['left_heel', 'left_foot_index', 'left_ankle']
        }
        self.head_landmark_names = ['right_ear', 'left_ear', 'nose']
        
        logger.info(f"StancePhaseDetectorSide initialized: "
                   f"calibration_frames={calibration_frames}, "
                   f"threshold_ratio={stance_threshold_ratio}, "
                   f"visibility_threshold={visibility_threshold}")

    def detect_stance_phase_side(self, landmarks: LandmarksDict) -> StanceResult:
        """
        Detect stance phase from side-view MediaPipe landmarks.
        
        This is the main detection method that handles calibration during initial frames,
        then performs stance phase analysis for subsequent frames.
        
        Parameters:
        -----------
        landmarks : LandmarksDict
            Dictionary of MediaPipe pose landmarks with format:
            {landmark_name: (x, y, z, visibility), ...}
            Required landmarks: foot landmarks (heel, foot_index) and head landmarks
            
        Returns:
        --------
        StanceResult
            Dictionary containing:
            - 'is_stance_phase' (bool): True if any foot is in stance phase
            - 'stance_foot' (str|None): Which foot is in stance ('left', 'right', or None)
            - 'confidence' (float): Detection confidence score (0.0-1.0)
            - 'debug_info' (str): Status information for calibration/debugging
            
        Note:
        -----
        During calibration phase (first N frames), returns is_stance_phase=False
        with debug_info indicating calibration status.
        """
        # Handle calibration phase
        if self.frame_count < self.calibration_frames:
            self._collect_calibration_data(landmarks)
            self.frame_count += 1
            
            if self.frame_count == self.calibration_frames:
                self._finalize_calibration()
                logger.info("Stance phase detector calibration completed. Detection now active.")
                
            return {
                'is_stance_phase': False,
                'stance_foot': None,
                'confidence': 0.0,
                'debug_info': f"calibrating ({self.frame_count}/{self.calibration_frames})"
            }
        
        # Verify calibration completed successfully
        if self.ground_level_normalized is None or self.avg_runner_height_normalized is None:
            logger.error("Calibration incomplete. Cannot perform stance detection.")
            return {
                'is_stance_phase': False,
                'stance_foot': None,
                'confidence': 0.0,
                'debug_info': "calibration_failed"
            }
        
        # Perform stance phase detection
        return self._perform_stance_detection(landmarks)
    
    def _collect_calibration_data(self, landmarks: LandmarksDict) -> None:
        """
        Collect ground level and runner height data during calibration frames.
        
        This method accumulates data points from multiple frames to establish robust
        baseline measurements for ground level and runner dimensions.
        
        Parameters:
        -----------
        landmarks : LandmarksDict
            Current frame's pose landmarks
            
        Collects:
        ---------
        - Ground level: Y-coordinate of lowest visible foot landmarks
        - Runner height: Distance from head landmarks to foot landmarks
        """
        try:
            # 1. Collect ground level samples (lowest foot positions)
            lowest_foot_y_this_frame = []
            
            for side in ['left', 'right']:
                foot_landmarks = self.foot_landmark_names[side][:2]  # heel and foot_index only
                
                for landmark_name in foot_landmarks:
                    if (landmark_name in landmarks and 
                        landmarks[landmark_name][3] >= self.visibility_threshold):
                        lowest_foot_y_this_frame.append(landmarks[landmark_name][1])
            
            if lowest_foot_y_this_frame:
                ground_sample = max(lowest_foot_y_this_frame)  # Max Y = lowest point in image
                self.ground_y_samples_normalized.append(ground_sample)
                
            # 2. Collect runner height samples (head to foot distance)
            head_y_candidates = []
            for landmark_name in self.head_landmark_names:
                if (landmark_name in landmarks and 
                    landmarks[landmark_name][3] >= self.visibility_threshold):
                    head_y_candidates.append(landmarks[landmark_name][1])
            
            if head_y_candidates and lowest_foot_y_this_frame:
                min_head_y = min(head_y_candidates)  # Highest head point
                max_foot_y = max(lowest_foot_y_this_frame)  # Lowest foot point
                
                runner_height = max_foot_y - min_head_y  # Positive = valid height
                
                # Sanity check: runner should occupy reasonable portion of frame
                if runner_height > 0.1:  # At least 10% of image height
                    self.runner_height_samples_normalized.append(runner_height)
                    
        except (KeyError, IndexError, TypeError) as e:
            logger.debug(f"Error during calibration data collection: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error in calibration data collection: {e}")
    
    def _finalize_calibration(self) -> None:
        """
        Calculate final calibration values from collected sample data.
        
        Uses robust statistical methods (percentiles, median) to establish ground level
        and runner height measurements that are resistant to outliers and measurement noise.
        
        Sets:
        -----
        - self.ground_level_normalized: Calibrated ground Y-coordinate
        - self.avg_runner_height_normalized: Calibrated average runner height
        """
        # Calculate ground level using percentile for robustness
        if self.ground_y_samples_normalized:
            # Use 85th percentile to handle occasional foot lift during calibration
            self.ground_level_normalized = np.percentile(
                self.ground_y_samples_normalized, 85
            )
            logger.info(f"Calibrated ground level: {self.ground_level_normalized:.3f} "
                       f"(from {len(self.ground_y_samples_normalized)} samples)")
        else:
            logger.warning("No ground level samples collected. Using default fallback.")
            self.ground_level_normalized = 0.9  # Conservative default (bottom 10% of frame)
        
        # Calculate runner height using median for robustness
        if self.runner_height_samples_normalized:
            self.avg_runner_height_normalized = np.median(
                self.runner_height_samples_normalized
            )
            logger.info(f"Calibrated runner height: {self.avg_runner_height_normalized:.3f} "
                       f"(from {len(self.runner_height_samples_normalized)} samples)")
        else:
            logger.warning("No runner height samples collected. Using default fallback.")
            self.avg_runner_height_normalized = 0.7  # Default: 70% of frame height
        
        # Log calibration summary
        dynamic_threshold = self.avg_runner_height_normalized * self.stance_threshold_ratio
        logger.info(f"Calibration complete. Dynamic stance threshold: {dynamic_threshold:.4f}")
    
    def _perform_stance_detection(self, landmarks: LandmarksDict) -> StanceResult:
        """
        Perform stance phase detection using calibrated parameters.
        
        Parameters:
        -----------
        landmarks : LandmarksDict
            Current frame's pose landmarks
            
        Returns:
        --------
        StanceResult
            Detection results with stance phase status, active foot, and confidence
        """
        # Calculate dynamic stance threshold based on calibrated runner height
        dynamic_stance_threshold = (
            self.avg_runner_height_normalized * self.stance_threshold_ratio
        )
        
        stance_candidates = []
        
        # Analyze each foot for stance phase
        for side in ['left', 'right']:
            candidate_result = self._analyze_foot_stance(
                landmarks, side, dynamic_stance_threshold
            )
            stance_candidates.append(candidate_result)
        
        # Determine overall stance phase and active foot
        return self._resolve_stance_phase(stance_candidates)
    
    def _analyze_foot_stance(
        self, 
        landmarks: LandmarksDict, 
        side: str, 
        threshold: float
    ) -> Dict[str, Union[str, bool, float]]:
        """
        Analyze stance phase for a single foot.
        
        Parameters:
        -----------
        landmarks : LandmarksDict
            Current frame landmarks
        side : str
            Foot side to analyze ('left' or 'right')
        threshold : float
            Dynamic stance detection threshold
            
        Returns:
        --------
        Dict containing side, stance status, confidence, and foot position data
        """
        relevant_landmarks = self.foot_landmark_names[side][:2]  # heel and foot_index
        foot_points_y = []
        
        # Check landmark visibility and collect Y coordinates
        all_landmarks_visible = True
        for landmark_name in relevant_landmarks:
            if (landmark_name not in landmarks or 
                landmarks[landmark_name][3] < self.visibility_threshold):
                all_landmarks_visible = False
                break
            foot_points_y.append(landmarks[landmark_name][1])
        
        # Return failed detection if landmarks not visible
        if not all_landmarks_visible or not foot_points_y:
            return {
                'side': side,
                'in_stance': False,
                'confidence': 0.0,
                'lowest_y': float('inf'),
                'reason': 'landmarks_not_visible'
            }
        
        # Find lowest point of foot (maximum Y coordinate)
        lowest_y_of_foot = max(foot_points_y)
        
        # Calculate distance from calibrated ground level
        distance_from_ground = abs(self.ground_level_normalized - lowest_y_of_foot)
        
        # Determine if foot is in stance based on proximity to ground
        is_in_stance = distance_from_ground <= threshold
        
        # Calculate confidence score (1.0 = on ground, 0.0 = far from ground)
        confidence = 0.0
        if threshold > 1e-6:  # Avoid division by zero
            confidence = max(0.0, 1.0 - (distance_from_ground / (threshold * 2)))
        
        # Reduce confidence if not in stance
        if not is_in_stance:
            confidence *= 0.5
        
        return {
            'side': side,
            'in_stance': is_in_stance,
            'confidence': confidence,
            'lowest_y': lowest_y_of_foot,
            'distance_from_ground': distance_from_ground
        }
    
    def _resolve_stance_phase(
        self, 
        stance_candidates: List[Dict[str, Union[str, bool, float]]]
    ) -> StanceResult:
        """
        Resolve overall stance phase from individual foot analyses.
        
        Handles cases of single stance, double stance (rare in running), and flight phase.
        Prioritizes the foot that is most confidently in contact with ground.
        
        Parameters:
        -----------
        stance_candidates : List[Dict]
            Results from individual foot stance analyses
            
        Returns:
        --------
        StanceResult
            Final stance phase determination
        """
        left_candidate = next(c for c in stance_candidates if c['side'] == 'left')
        right_candidate = next(c for c in stance_candidates if c['side'] == 'right')
        
        is_stance_phase = False
        stance_foot = None
        final_confidence = 0.0
        
        # Case 1: Both feet detected in stance (rare in running, common in walking)
        if left_candidate['in_stance'] and right_candidate['in_stance']:
            logger.debug("Double stance detected - unusual for running analysis")
            
            # Choose foot based on which is lower (more planted) and confidence
            if left_candidate['lowest_y'] > right_candidate['lowest_y']:  # Left is lower
                if left_candidate['confidence'] > right_candidate['confidence'] * 0.8:
                    stance_foot = 'left'
                    final_confidence = left_candidate['confidence']
                else:
                    stance_foot = 'right'
                    final_confidence = right_candidate['confidence']
            else:  # Right is lower or equal
                if right_candidate['confidence'] > left_candidate['confidence'] * 0.8:
                    stance_foot = 'right'
                    final_confidence = right_candidate['confidence']
                else:
                    stance_foot = 'left'
                    final_confidence = left_candidate['confidence']
            
            is_stance_phase = True
            
        # Case 2: Left foot only in stance
        elif left_candidate['in_stance']:
            is_stance_phase = True
            stance_foot = 'left'
            final_confidence = left_candidate['confidence']
            
        # Case 3: Right foot only in stance  
        elif right_candidate['in_stance']:
            is_stance_phase = True
            stance_foot = 'right'
            final_confidence = right_candidate['confidence']
            
        # Case 4: No stance phase (flight phase)
        else:
            # Confidence represents how close the closest foot was to stance
            final_confidence = max(left_candidate['confidence'], right_candidate['confidence'])
        
        return {
            'is_stance_phase': is_stance_phase,
            'stance_foot': stance_foot,
            'confidence': final_confidence,
            'debug_info': 'detection_complete'
        }
    
    def _get_foot_vertical_velocity(self, side: str, current_lowest_y: float) -> float:
        """
        Estimate vertical velocity of foot for enhanced stance detection.
        
        This method is currently not used in main detection but provides foundation
        for velocity-based stance refinement.
        
        Parameters:
        -----------
        side : str
            Foot side ('left' or 'right')
        current_lowest_y : float
            Current frame's lowest Y coordinate for the foot
            
        Returns:
        --------
        float
            Estimated vertical velocity in normalized coordinates per frame
            Positive = moving down, Negative = moving up
        """
        self.foot_y_history[side].append(current_lowest_y)
        
        # Maintain history buffer
        if len(self.foot_y_history[side]) > self.max_history_len:
            self.foot_y_history[side].pop(0)
        
        if len(self.foot_y_history[side]) < 2:
            return 0.0  # Insufficient history
        
        # Simple velocity calculation (could be enhanced with linear regression)
        velocity = self.foot_y_history[side][-1] - self.foot_y_history[side][-2]
        return velocity
    
    def get_calibration_status(self) -> CalibrationData:
        """
        Get current calibration status and parameters.
        
        Returns:
        --------
        CalibrationData
            Dictionary containing calibration progress and computed values
        """
        return {
            'frames_processed': self.frame_count,
            'calibration_complete': self.frame_count >= self.calibration_frames,
            'ground_level_normalized': self.ground_level_normalized,
            'runner_height_normalized': self.avg_runner_height_normalized,
            'ground_samples_count': len(self.ground_y_samples_normalized),
            'height_samples_count': len(self.runner_height_samples_normalized),
            'dynamic_threshold': (
                self.avg_runner_height_normalized * self.stance_threshold_ratio
                if self.avg_runner_height_normalized else None
            )
        }
    
    def reset_calibration(self) -> None:
        """
        Reset detector to perform new calibration.
        
        Useful when switching to new video or significantly different camera angle.
        """
        logger.info("Resetting stance phase detector calibration")
        
        self.frame_count = 0
        self.ground_y_samples_normalized.clear()
        self.runner_height_samples_normalized.clear()
        self.ground_level_normalized = None
        self.avg_runner_height_normalized = None
        
        # Clear velocity history
        for side in ['left', 'right']:
            self.foot_y_history[side].clear()

def stance_detector_side_wrapper( 
    landmarks: LandmarksDict,
    detector_instance: Optional[StancePhaseDetectorSide] = None
) -> StanceResult:
    """
    Standalone wrapper function for stance phase detection.
    
    Provides functional interface to StancePhaseDetectorSide class
    for integration with existing analysis pipelines.
    
    Args:
        detector: Initialized StancePhaseDetectorVelocity instance
        landmarks: Dictionary containing pose landmarks
        
    Returns:
        StanceResult: Analysis results from detector
    """

            # Create or retrieve detector instance
    if detector_instance is not None:
        detector = detector_instance
    else:
        # Use function attribute to maintain detector across calls
        if not hasattr(stance_detector_side_wrapper, '_default_detector'):
            stance_detector_side_wrapper._default_detector = StancePhaseDetectorSide()
            logger.info("Created default StancePhaseDetectorSide instance")
        detector = stance_detector_side_wrapper._default_detector
    return detector.detect_stance_phase_side(landmarks)


# Example usage and testing functions
def example_usage():
    """Demonstrate basic usage of StancePhaseDetectorSide."""
    
    # Initialize detector with custom parameters
    detector = StancePhaseDetectorSide(
        calibration_frames=60,      # 2 seconds at 30fps
        stance_threshold_ratio=0.02, # 2% of runner height
        visibility_threshold=0.6     # Require 60% visibility
    )
    
    # Example landmark data (normalized coordinates)
    sample_landmarks = {
        'left_heel': (0.3, 0.85, 0.0, 0.9),
        'left_foot_index': (0.32, 0.87, 0.0, 0.85),
        'left_ankle': (0.31, 0.82, 0.0, 0.95),
        'right_heel': (0.6, 0.75, 0.0, 0.9),
        'right_foot_index': (0.62, 0.77, 0.0, 0.85),
        'right_ankle': (0.61, 0.72, 0.0, 0.95),
        'nose': (0.45, 0.15, 0.0, 0.95),
        'left_ear': (0.42, 0.18, 0.0, 0.9),
        'right_ear': (0.48, 0.18, 0.0, 0.9)
    }
    
    # Process frame
    result = detector.detect_stance_phase_side(sample_landmarks)
    
    print("Stance Detection Result:")
    print(f"  Stance Phase: {result['is_stance_phase']}")
    print(f"  Stance Foot: {result['stance_foot']}")
    print(f"  Confidence: {result['confidence']:.3f}")
    print(f"  Status: {result['debug_info']}")
    
    # Check calibration status
    calibration_status = detector.get_calibration_status()
    print(f"\nCalibration Status: {calibration_status}")


if __name__ == "__main__":
    example_usage()