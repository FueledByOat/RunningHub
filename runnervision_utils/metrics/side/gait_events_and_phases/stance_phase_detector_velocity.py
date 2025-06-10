# stance_phase_detector_velocity.py

"""
Detects stance phase in running gait using velocity analysis.

This module analyzes ankle velocities to determine which foot is in stance phase
(ground contact) during running. Stance phase is characterized by lower vertical
velocity and proximity to ground level.

The detector uses a sliding window approach to calculate foot velocities and
combines velocity with position data to identify the supporting foot.
"""

import logging
import numpy as np
from collections import deque
from typing import Dict, Tuple, Optional, Union, List

logger = logging.getLogger(__name__)

# Type aliases
Landmark = Tuple[float, float, float, float]  # (x, y, z, visibility)
LandmarksDict = Dict[str, Landmark]
StancePhaseResult = Dict[str, Union[bool, str, float]]


class StancePhaseDetectorVelocity:
    """
    Detects stance phase in running gait using velocity analysis.
    
    This class maintains a sliding window of ankle positions to calculate
    vertical velocities and determine which foot is currently in stance phase.
    
    Stance phase detection combines:
    - Vertical velocity analysis (lower velocity indicates ground contact)
    - Relative foot height comparison
    - Confidence scoring based on velocity differences
    
    The detector is optimized for running gait where clear velocity differences
    exist between stance and swing phases.
    
    Attributes:
        frame_rate (float): Video frame rate in fps
        velocity_window (int): Number of frames for velocity calculation
        velocity_threshold (float): Velocity threshold for stance detection
    """
    
    def __init__(
        self, 
        frame_rate: float = 30.0, 
        velocity_window: int = 3,
        velocity_threshold: float = 0.1
    ):
        """
        Initialize stance phase detector.
        
        Args:
            frame_rate: Video frame rate in fps. Used for velocity calculations.
            velocity_window: Number of frames to maintain for velocity analysis.
                           Larger windows provide smoother velocity estimates
                           but reduce temporal resolution.
            velocity_threshold: Velocity threshold below which foot is considered
                              in stance phase. Units are normalized coordinates
                              per second.
        
        Raises:
            ValueError: If parameters are outside valid ranges.
        """
        if frame_rate <= 0:
            raise ValueError(f"Frame rate must be positive, got {frame_rate}")
        if velocity_window < 2:
            raise ValueError(f"Velocity window must be >= 2, got {velocity_window}")
        if velocity_threshold < 0:
            raise ValueError(f"Velocity threshold must be non-negative, got {velocity_threshold}")
            
        self.frame_rate = frame_rate
        self.velocity_window = velocity_window
        self.velocity_threshold = velocity_threshold
        
        # Sliding window for ankle position history
        self.ankle_positions = deque(maxlen=velocity_window)
        
        logger.info(f"Initialized StancePhaseDetectorVelocity with frame_rate={frame_rate}, "
                   f"velocity_window={velocity_window}, threshold={velocity_threshold}")
    
    def update(self, landmarks: LandmarksDict) -> StancePhaseResult:
        """
        Update detector with new landmarks and analyze stance phase.
        
        Processes new frame data to:
        1. Calculate foot velocities over the sliding window
        2. Compare relative foot positions
        3. Determine stance foot and confidence
        4. Assess if currently in stance phase
        
        Args:
            landmarks: Dictionary containing pose landmarks. Must include
                      'left_ankle' and 'right_ankle' keys with (x,y,z,visibility) tuples.
                      Y-coordinates should be normalized (0.0-1.0).
        
        Returns:
            StancePhaseResult: Dictionary containing:
                - is_stance_phase (bool): True if in stance phase
                - stance_foot (str): 'left', 'right', or 'unknown'
                - left_foot_velocity (float): Left foot vertical velocity
                - right_foot_velocity (float): Right foot vertical velocity
                - confidence (float): Detection confidence (0.0-1.0)
                - calculation_successful (bool): True if calculation completed
        """
        # Initialize result structure
        result: StancePhaseResult = {
            'is_stance_phase': False,
            'stance_foot': 'unknown',
            'left_foot_velocity': 0.0,
            'right_foot_velocity': 0.0,
            'confidence': 0.0,
            'calculation_successful': False
        }
        
        # Validate required landmarks
        required_landmarks = ['left_ankle', 'right_ankle']
        for landmark_name in required_landmarks:
            if landmark_name not in landmarks:
                logger.warning(f"Required landmark '{landmark_name}' not found for stance phase detection.")
                return result
        
        try:
            # Store current ankle positions
            ankle_data = {
                'left_ankle': landmarks['left_ankle'],
                'right_ankle': landmarks['right_ankle']
            }
            self.ankle_positions.append(ankle_data)
            
            # Need at least 2 positions for velocity calculation
            if len(self.ankle_positions) < 2:
                logger.debug("Insufficient data for velocity calculation")
                return result
            
            # Calculate velocities
            left_velocity, right_velocity = self._calculate_velocities()
            
            # Determine stance foot and confidence
            stance_foot, confidence = self._determine_stance_foot(
                landmarks, left_velocity, right_velocity
            )
            
            # Assess if in stance phase
            stance_velocity = left_velocity if stance_foot == 'left' else right_velocity
            is_stance = stance_velocity < self.velocity_threshold
            
            result.update({
                'is_stance_phase': is_stance,
                'stance_foot': stance_foot,
                'left_foot_velocity': left_velocity,
                'right_foot_velocity': right_velocity,
                'confidence': confidence,
                'calculation_successful': True
            })
            
            logger.debug(f"Stance detection: {stance_foot} foot, velocity={stance_velocity:.3f}, "
                        f"confidence={confidence:.2f}")
            
        except KeyError as e:
            logger.error(f"Missing landmark data during stance phase detection: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error during stance phase detection: {e}")
            
        return result
    
    def _calculate_velocities(self) -> Tuple[float, float]:
        """
        Calculate average vertical velocities for both feet.
        
        Computes velocities over the available position history using
        frame-to-frame differences and temporal smoothing.
        
        Returns:
            Tuple[float, float]: (left_velocity, right_velocity) in normalized units/second
        """
        dt = 1.0 / self.frame_rate
        recent_positions = list(self.ankle_positions)
        
        left_velocities: List[float] = []
        right_velocities: List[float] = []
        
        # Calculate frame-to-frame velocities
        for i in range(len(recent_positions) - 1):
            curr_left_y = recent_positions[i+1]['left_ankle'][1]
            prev_left_y = recent_positions[i]['left_ankle'][1]
            curr_right_y = recent_positions[i+1]['right_ankle'][1]
            prev_right_y = recent_positions[i]['right_ankle'][1]
            
            # Calculate absolute vertical velocities
            left_dy = abs((curr_left_y - prev_left_y) / dt)
            right_dy = abs((curr_right_y - prev_right_y) / dt)
            
            left_velocities.append(left_dy)
            right_velocities.append(right_dy)
        
        # Return average velocities
        avg_left_velocity = np.mean(left_velocities) if left_velocities else 0.0
        avg_right_velocity = np.mean(right_velocities) if right_velocities else 0.0
        
        return avg_left_velocity, avg_right_velocity
    
    def _determine_stance_foot(
        self, 
        landmarks: LandmarksDict, 
        left_velocity: float, 
        right_velocity: float
    ) -> Tuple[str, float]:
        """
        Determine which foot is in stance phase and calculate confidence.
        
        Combines velocity analysis with relative foot height to score
        each foot's likelihood of being in stance phase.
        
        Args:
            landmarks: Current frame landmarks
            left_velocity: Left foot vertical velocity
            right_velocity: Right foot vertical velocity
        
        Returns:
            Tuple[str, float]: (stance_foot, confidence_score)
        """
        # Get current foot heights
        current_left_height = landmarks['left_ankle'][1]
        current_right_height = landmarks['right_ankle'][1]
        
        # Score each foot for stance likelihood
        # Higher score = more likely to be stance foot
        # Formula: inverse velocity * height bonus
        epsilon = 0.001  # Prevent division by zero
        
        left_velocity_score = 1.0 / (left_velocity + epsilon)
        right_velocity_score = 1.0 / (right_velocity + epsilon)
        
        # Height bonus: favor lower foot (closer to ground)
        left_height_bonus = 1.0 if current_left_height >= current_right_height else 0.7
        right_height_bonus = 1.0 if current_right_height >= current_left_height else 0.7
        
        left_total_score = left_velocity_score * left_height_bonus
        right_total_score = right_velocity_score * right_height_bonus
        
        # Determine stance foot and calculate confidence
        total_score = left_total_score + right_total_score
        
        if left_total_score > right_total_score:
            stance_foot = 'left'
            confidence = left_total_score / total_score if total_score > 0 else 0.5
        else:
            stance_foot = 'right'
            confidence = right_total_score / total_score if total_score > 0 else 0.5
        
        return stance_foot, confidence
    
    def reset(self) -> None:
        """
        Reset detector state for processing a new sequence.
        
        Clears position history while preserving configuration parameters.
        """
        self.ankle_positions.clear()
        logger.info("StancePhaseDetectorVelocity state reset")
    
    def set_velocity_threshold(self, threshold: float) -> None:
        """
        Update velocity threshold for stance detection.
        
        Args:
            threshold: New velocity threshold (normalized units/second)
            
        Raises:
            ValueError: If threshold is negative
        """
        if threshold < 0:
            raise ValueError(f"Velocity threshold must be non-negative, got {threshold}")
            
        self.velocity_threshold = threshold
        logger.info(f"Velocity threshold updated to {threshold}")


def stance_detector_velocity_wrapper(
    landmarks: LandmarksDict,
    detector_instance: Optional[StancePhaseDetectorVelocity] = None
) -> StancePhaseResult:
    """
    Standalone wrapper function for stance phase detection.
    
    Provides functional interface to StancePhaseDetectorVelocity class
    for integration with existing analysis pipelines.
    
    Args:
        detector: Initialized StancePhaseDetectorVelocity instance
        landmarks: Dictionary containing pose landmarks
        
    Returns:
        StancePhaseResult: Analysis results from detector
    """
        # Create or retrieve detector instance
    if detector_instance is not None:
        detector = detector_instance
    else:
        # Use function attribute to maintain detector across calls
        if not hasattr(stance_detector_velocity_wrapper, '_default_detector'):
            stance_detector_velocity_wrapper._default_detector = StancePhaseDetectorVelocity()
            logger.info("Created default StancePhaseDetectorVelocity instance")
        detector = stance_detector_velocity_wrapper._default_detector

    return detector.update(landmarks)


if __name__ == "__main__":
    print("Testing StancePhaseDetectorVelocity...")
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize detector
    detector = StancePhaseDetectorVelocity(frame_rate=30.0, velocity_window=5)
    
    # Simulate running gait with alternating stance phases
    import math
    
    print("\nSimulating running gait data...")
    for frame in range(50):
        # Simulate alternating foot movement
        time = frame * 0.1
        left_y = 0.8 + 0.1 * math.sin(time * 2)  # Left foot cycle
        right_y = 0.8 + 0.1 * math.sin(time * 2 + math.pi)  # Right foot offset
        
        sample_landmarks = {
            'left_ankle': (0.4, left_y, 0, 0.95),
            'right_ankle': (0.6, right_y, 0, 0.95)
        }
        
        result = detector.update(sample_landmarks)
        
        # Print results every 10 frames
        if frame % 10 == 0 and frame > 0:
            print(f"\nFrame {frame} Results:")
            print(f"  Stance phase: {result['is_stance_phase']}")
            print(f"  Stance foot: {result['stance_foot']}")
            print(f"  Left velocity: {result['left_foot_velocity']:.3f}")
            print(f"  Right velocity: {result['right_foot_velocity']:.3f}")
            print(f"  Confidence: {result['confidence']:.2f}")
    
    print("\nTesting complete.")