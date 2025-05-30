# stride_length.py

"""
Estimates stride length and related biomechanical metrics for running analysis.

This module calculates stride length using temporal tracking of foot positions,
optimized for side-view analysis. It provides both instantaneous estimates and
smoothed temporal measurements with confidence scoring.
"""

import logging
import numpy as np
from typing import Dict, Tuple, List, Optional, Any

logger = logging.getLogger(__name__)

# Define type aliases for clarity
Landmark = Tuple[float, float, float, float]
LandmarksDict = Dict[str, Landmark]

# Define the expected output structure
StrideAnalysisResult = Dict[str, Optional[Any]]

# Define temporal tracking data structure
TemporalTrackingData = Dict[str, Any]

def estimate_stride_length(
    landmarks: LandmarksDict,
    frame_index: Optional[int] = None,
    height_cm: Optional[float] = None,
    temporal_tracking: Optional[TemporalTrackingData] = None,
    fps: float = 30.0,
    threshold_proportion: float = 0.25
) -> Tuple[StrideAnalysisResult, Optional[TemporalTrackingData]]:
    """
    Estimate stride length based on foot positions with temporal tracking,
    optimized for side-view analysis.
    
    This function analyzes foot movement patterns to detect touchdown events
    and calculate stride metrics. It uses both instantaneous estimates from
    current pose and temporal tracking for improved accuracy.
    
    Parameters:
    -----------
    landmarks : LandmarksDict
        Dictionary containing pose landmarks with (x, y, z, visibility) coordinates.
        Required keys: 'right_foot_index', 'right_ankle', 'right_knee', 'right_hip'.
        Optional keys: 'right_shoulder', 'nose', 'right_ear', 'right_eye' for height estimation.
        Coordinates are typically normalized (0.0-1.0).
        
    frame_index : Optional[int], default=None
        Current frame index for temporal tracking. If None, only instantaneous
        estimates will be calculated.
        
    height_cm : Optional[float], default=None
        Runner's height in centimeters. If provided, improves scale factor accuracy
        and enables normalized stride length calculation.
        
    temporal_tracking : Optional[TemporalTrackingData], default=None
        Previous temporal tracking data to maintain state across frames.
        If None, new tracking data will be initialized.
        
    fps : float, default=30.0
        Video frame rate in frames per second, used for timing calculations.
        
    threshold_proportion : float, default=0.25
        Velocity threshold proportion for touchdown detection. Lower values
        make touchdown detection more sensitive.
    
    Returns:
    --------
    Tuple[StrideAnalysisResult, Optional[TemporalTrackingData]]
        A tuple containing:
        - StrideAnalysisResult: Dictionary with stride analysis metrics
        - TemporalTrackingData: Updated temporal tracking data (None if frame_index is None)
        
    StrideAnalysisResult contains:
        - "instantaneous_estimate_cm" (Optional[float]): Current stride estimate from pose
        - "stride_length_cm" (Optional[float]): Smoothed stride length from temporal tracking
        - "normalized_stride_length" (Optional[float]): Stride length relative to height (0.0-2.0 typical range)
        - "stride_frequency" (Optional[float]): Cadence in strides per minute
        - "assessment" (Optional[str]): Stride appropriateness ("optimal", "too_short", "too_long", "insufficient_data")
        - "confidence" (float): Measurement confidence (0.0-1.0)
        - "calculation_successful" (bool): True if basic calculations completed
    """
    
    # Initialize result structure
    result: StrideAnalysisResult = {
        'instantaneous_estimate_cm': None,
        'stride_length_cm': None,
        'normalized_stride_length': None,
        'stride_frequency': None,
        'assessment': None,
        'confidence': 0.5,
        'calculation_successful': False
    }
    
    # Check required landmarks
    required_landmarks = ['right_foot_index', 'right_ankle', 'right_knee', 'right_hip']
    missing_landmarks = [lm for lm in required_landmarks if lm not in landmarks]
    
    if missing_landmarks:
        for lm_name in missing_landmarks:
            logger.warning(f"Required landmark '{lm_name}' not found for stride length calculation.")
        result['confidence'] = 0.0
        result['assessment'] = "insufficient_data"
        return result, temporal_tracking
    
    try:
        # Calculate scale factor for coordinate-to-centimeter conversion
        scale_factor = _calculate_side_view_scale_factor(landmarks, height_cm)
        
        # Initialize or update temporal tracking if frame_index provided
        updated_tracking = None
        if frame_index is not None:
            updated_tracking = _initialize_or_update_temporal_tracking(
                landmarks, frame_index, temporal_tracking, fps, threshold_proportion, scale_factor
            )
            
            # Extract temporal results if available
            if updated_tracking and len(updated_tracking['stride_lengths']) > 0:
                result['stride_length_cm'] = (
                    sum(updated_tracking['stride_lengths']) / len(updated_tracking['stride_lengths'])
                )
                result['confidence'] = 0.85
                result['instantaneous_estimate_cm'] = updated_tracking['stride_lengths'][-1]
                
                # Calculate stride frequency if we have timing data
                if len(updated_tracking['stride_times']) > 0:
                    avg_stride_time_frames = (
                        sum(updated_tracking['stride_times']) / len(updated_tracking['stride_times'])
                    )
                    stride_time_seconds = avg_stride_time_frames / fps
                    result['stride_frequency'] = 60.0 / stride_time_seconds if stride_time_seconds > 0 else None
        
        # Calculate instantaneous estimate if not available from temporal tracking
        if result['instantaneous_estimate_cm'] is None:
            instantaneous_estimate = _calculate_instantaneous_stride_estimate(landmarks, scale_factor)
            result['instantaneous_estimate_cm'] = instantaneous_estimate
            
            # Use instantaneous estimate if no temporal data available
            if result['stride_length_cm'] is None:
                result['stride_length_cm'] = instantaneous_estimate
                result['confidence'] = 0.6
        
        # Calculate normalized stride length
        if result['stride_length_cm'] is not None:
            runner_height = height_cm or _estimate_height_from_side_view(landmarks, scale_factor)
            if runner_height and runner_height > 0:
                result['normalized_stride_length'] = result['stride_length_cm'] / runner_height
        
        # Assess stride length appropriateness
        if result['normalized_stride_length'] is not None:
            result['assessment'] = _assess_stride_length(result['normalized_stride_length'])
        
        result['calculation_successful'] = True
        
    except Exception as e:
        logger.exception(f"Unexpected error during stride length calculation: {e}")
        result['confidence'] = 0.0
        result['assessment'] = "calculation_error"
    
    return result, updated_tracking


def _calculate_side_view_scale_factor(landmarks: LandmarksDict, height_cm: Optional[float] = None) -> float:
    """
    Calculate scale factor for converting normalized coordinates to centimeters.
    
    Parameters:
    -----------
    landmarks : LandmarksDict
        Dictionary containing pose landmarks
    height_cm : Optional[float]
        Known runner height in cm for accurate scaling
        
    Returns:
    --------
    float
        Scale factor for coordinate conversion
    """
    try:
        if height_cm and height_cm > 0:
            # Use known height for precise scaling
            head_landmarks = ['nose', 'right_ear', 'right_eye']
            head_y = float('inf')
            
            for head_lm in head_landmarks:
                if head_lm in landmarks:
                    head_y = min(head_y, landmarks[head_lm][1])
                    break
            
            if head_y < float('inf') and 'right_ankle' in landmarks:
                ankle_y = landmarks['right_ankle'][1]
                body_height_coords = ankle_y - head_y
                
                if body_height_coords > 0:
                    # Head-to-ankle is approximately 85% of total height
                    return height_cm / (body_height_coords * 0.85)
        
        # Estimate scale using body proportions
        if all(lm in landmarks for lm in ['right_hip', 'right_knee', 'right_ankle']):
            # Calculate leg length (thigh + shin)
            thigh_length = np.sqrt(
                (landmarks['right_hip'][0] - landmarks['right_knee'][0])**2 + 
                (landmarks['right_hip'][1] - landmarks['right_knee'][1])**2
            )
            
            shin_length = np.sqrt(
                (landmarks['right_knee'][0] - landmarks['right_ankle'][0])**2 + 
                (landmarks['right_knee'][1] - landmarks['right_ankle'][1])**2
            )
            
            leg_length = thigh_length + shin_length
            
            if leg_length > 0:
                # Leg length is approximately 48% of body height
                estimated_height_coords = leg_length / 0.48
                return 170 / estimated_height_coords  # Use 170cm as average height
        
        # Fallback using torso length
        if all(lm in landmarks for lm in ['right_hip', 'right_shoulder']):
            torso_length = np.sqrt(
                (landmarks['right_hip'][0] - landmarks['right_shoulder'][0])**2 + 
                (landmarks['right_hip'][1] - landmarks['right_shoulder'][1])**2
            )
            
            if torso_length > 0:
                # Torso is approximately 30% of body height
                estimated_height_coords = torso_length / 0.30
                return 170 / estimated_height_coords
        
    except Exception as e:
        logger.warning(f"Error calculating scale factor, using default: {e}")
    
    # Default scale factor
    return 200.0


def _initialize_or_update_temporal_tracking(
    landmarks: LandmarksDict,
    frame_index: int,
    temporal_tracking: Optional[TemporalTrackingData],
    fps: float,
    threshold_proportion: float,
    scale_factor: float
) -> TemporalTrackingData:
    """
    Initialize or update temporal tracking data for stride detection.
    
    Parameters:
    -----------
    landmarks : LandmarksDict
        Current frame landmarks
    frame_index : int
        Current frame number
    temporal_tracking : Optional[TemporalTrackingData]
        Previous tracking data
    fps : float
        Video frame rate
    threshold_proportion : float
        Velocity threshold for touchdown detection
    scale_factor : float
        Coordinate to centimeter conversion factor
        
    Returns:
    --------
    TemporalTrackingData
        Updated temporal tracking data
    """
    # Initialize tracking if not provided
    if temporal_tracking is None:
        temporal_tracking = {
            'foot_positions': [],
            'stride_events': [],
            'stride_lengths': [],
            'foot_y_history': [],
            'stride_times': [],
            'last_touchdown_frame': None,
            'foot_velocity': [],
            'last_foot_y': None
        }
    
    # Extract current foot position
    right_foot = np.array([landmarks['right_foot_index'][0], landmarks['right_foot_index'][1]])
    
    # Update foot position history
    temporal_tracking['foot_y_history'].append(right_foot[1])
    max_history = int(2 * fps)  # 2 seconds of history
    
    if len(temporal_tracking['foot_y_history']) > max_history:
        temporal_tracking['foot_y_history'] = temporal_tracking['foot_y_history'][-max_history:]
    
    # Calculate foot velocity
    if temporal_tracking['last_foot_y'] is not None:
        y_velocity = right_foot[1] - temporal_tracking['last_foot_y']
        temporal_tracking['foot_velocity'].append(y_velocity)
        
        # Keep velocity history manageable
        if len(temporal_tracking['foot_velocity']) > 10:
            temporal_tracking['foot_velocity'] = temporal_tracking['foot_velocity'][-10:]
    
    temporal_tracking['last_foot_y'] = right_foot[1]
    
    # Detect foot touchdown
    foot_touchdown_detected = _detect_foot_touchdown(
        temporal_tracking, right_foot, threshold_proportion
    )
    
    # Process touchdown event
    if foot_touchdown_detected:
        min_frames_between_touchdowns = int(0.5 * fps)  # Minimum 0.5 seconds between touchdowns
        
        if (temporal_tracking['last_touchdown_frame'] is None or 
            frame_index - temporal_tracking['last_touchdown_frame'] > min_frames_between_touchdowns):
            
            temporal_tracking['stride_events'].append({
                'frame': frame_index,
                'position': right_foot.copy(),
                'type': 'touchdown'
            })
            temporal_tracking['last_touchdown_frame'] = frame_index
            
            # Calculate stride metrics if we have enough data
            _update_stride_metrics(temporal_tracking, scale_factor, fps)
    
    return temporal_tracking


def _detect_foot_touchdown(
    temporal_tracking: TemporalTrackingData,
    current_foot_pos: np.ndarray,
    threshold_proportion: float
) -> bool:
    """
    Detect foot touchdown based on position and velocity patterns.
    
    Parameters:
    -----------
    temporal_tracking : TemporalTrackingData
        Current tracking data
    current_foot_pos : np.ndarray
        Current foot position [x, y]
    threshold_proportion : float
        Velocity threshold proportion
        
    Returns:
    --------
    bool
        True if touchdown detected
    """
    if len(temporal_tracking['foot_velocity']) < 3 or len(temporal_tracking['foot_y_history']) < 10:
        return False
    
    try:
        # Calculate weighted average of recent velocity
        recent_velocities = temporal_tracking['foot_velocity'][-3:]
        weights = [1, 2, 3]  # More weight to recent values
        recent_velocity = sum(v * w for v, w in zip(recent_velocities, weights)) / sum(weights)
        
        # Check if foot is in lower position range
        y_values = temporal_tracking['foot_y_history']
        y_threshold = sorted(y_values)[-len(y_values)//4]  # Lower quartile
        
        # Touchdown conditions:
        # 1. Foot is in lower position range
        # 2. Velocity is near zero (transition from downward to upward)
        velocity_threshold = threshold_proportion * 0.04  # Adjust based on threshold_proportion
        
        return (current_foot_pos[1] > y_threshold and 
                abs(recent_velocity) < velocity_threshold)
        
    except Exception as e:
        logger.warning(f"Error in touchdown detection: {e}")
        return False


def _update_stride_metrics(
    temporal_tracking: TemporalTrackingData,
    scale_factor: float,
    fps: float
) -> None:
    """
    Update stride length and timing metrics based on recent touchdown events.
    
    Parameters:
    -----------
    temporal_tracking : TemporalTrackingData
        Tracking data to update
    scale_factor : float
        Coordinate to centimeter conversion factor
    fps : float
        Video frame rate
    """
    try:
        # Get recent touchdown events
        touchdown_events = [e for e in temporal_tracking['stride_events'] if e['type'] == 'touchdown']
        
        if len(touchdown_events) >= 2:
            last_two = touchdown_events[-2:]
            frames_diff = last_two[1]['frame'] - last_two[0]['frame']
            
            # Validate reasonable stride timing (0.5-2 seconds)
            min_frames = int(0.5 * fps)
            max_frames = int(2.0 * fps)
            
            if min_frames <= frames_diff <= max_frames:
                # Calculate horizontal distance between touchdowns
                stride_distance = abs(last_two[1]['position'][0] - last_two[0]['position'][0])
                stride_length_cm = stride_distance * scale_factor
                
                # Add to measurements
                temporal_tracking['stride_lengths'].append(stride_length_cm)
                temporal_tracking['stride_times'].append(frames_diff)
                
                # Keep only recent measurements (last 5)
                max_measurements = 5
                if len(temporal_tracking['stride_lengths']) > max_measurements:
                    temporal_tracking['stride_lengths'] = temporal_tracking['stride_lengths'][-max_measurements:]
                    temporal_tracking['stride_times'] = temporal_tracking['stride_times'][-max_measurements:]
    
    except Exception as e:
        logger.warning(f"Error updating stride metrics: {e}")


def _calculate_instantaneous_stride_estimate(landmarks: LandmarksDict, scale_factor: float) -> float:
    """
    Calculate instantaneous stride estimate from current pose.
    
    Parameters:
    -----------
    landmarks : LandmarksDict
        Current pose landmarks
    scale_factor : float
        Coordinate to centimeter conversion factor
        
    Returns:
    --------
    float
        Instantaneous stride estimate in centimeters
    """
    try:
        # Calculate horizontal leg extension from hip to ankle
        hip_pos = np.array([landmarks['right_hip'][0], landmarks['right_hip'][1]])
        ankle_pos = np.array([landmarks['right_ankle'][0], landmarks['right_ankle'][1]])
        
        horizontal_leg_extension = abs(hip_pos[0] - ankle_pos[0])
        
        # Stride length is typically 0.8-1.0x the leg extension
        stride_multiplier = 0.9
        instantaneous_estimate_cm = horizontal_leg_extension * scale_factor * stride_multiplier
        
        return max(instantaneous_estimate_cm, 0.0)  # Ensure non-negative
    
    except Exception as e:
        logger.warning(f"Error calculating instantaneous stride estimate: {e}")
        return 0.0


def _estimate_height_from_side_view(landmarks: LandmarksDict, scale_factor: float) -> Optional[float]:
    """
    Estimate runner's height from side-view landmarks.
    
    Parameters:
    -----------
    landmarks : LandmarksDict
        Pose landmarks
    scale_factor : float
        Coordinate to centimeter conversion factor
        
    Returns:
    --------
    Optional[float]
        Estimated height in cm, or None if estimation fails
    """
    try:
        height_estimates = []
        weights = []
        
        # Method 1: Head to ankle distance
        head_landmarks = ['nose', 'right_ear', 'right_eye']
        head_y = float('inf')
        
        for head_lm in head_landmarks:
            if head_lm in landmarks:
                head_y = min(head_y, landmarks[head_lm][1])
                break
        
        if head_y < float('inf') and 'right_ankle' in landmarks:
            ankle_y = landmarks['right_ankle'][1]
            visible_height_coords = ankle_y - head_y
            
            if visible_height_coords > 0:
                # Convert to cm (visible height is ~85% of total height)
                height_estimate = (visible_height_coords * scale_factor) / 0.85
                height_estimates.append(height_estimate)
                weights.append(1.0)
        
        # Method 2: Leg length proportions
        if all(lm in landmarks for lm in ['right_hip', 'right_knee', 'right_ankle']):
            thigh_length = np.sqrt(
                (landmarks['right_hip'][0] - landmarks['right_knee'][0])**2 + 
                (landmarks['right_hip'][1] - landmarks['right_knee'][1])**2
            )
            
            shin_length = np.sqrt(
                (landmarks['right_knee'][0] - landmarks['right_ankle'][0])**2 + 
                (landmarks['right_knee'][1] - landmarks['right_ankle'][1])**2
            )
            
            leg_length = thigh_length + shin_length
            
            if leg_length > 0:
                # Leg length is ~48% of body height
                height_from_leg = (leg_length * scale_factor) / 0.48
                height_estimates.append(height_from_leg)
                weights.append(0.9)
        
        # Calculate weighted average
        if height_estimates:
            weighted_height = sum(h * w for h, w in zip(height_estimates, weights)) / sum(weights)
            return max(weighted_height, 100.0)  # Minimum reasonable height
        
        return 170.0  # Default average height
    
    except Exception as e:
        logger.warning(f"Error estimating height: {e}")
        return 170.0


def _assess_stride_length(normalized_stride_length: float) -> str:
    """
    Assess stride length appropriateness based on normalized value.
    
    Parameters:
    -----------
    normalized_stride_length : float
        Stride length as proportion of height
        
    Returns:
    --------
    str
        Assessment: "optimal", "too_short", or "too_long"
    """
    # Based on running biomechanics research:
    # - Recreational runners: 0.8-1.0 (normalized)
    # - Efficient range: 0.85-0.95
    # - Elite runners may go higher with speed
    
    if normalized_stride_length < 0.75:
        return "too_short"
    elif normalized_stride_length > 1.15:
        return "too_long"
    else:
        return "optimal"


if __name__ == "__main__":
    print("Testing stride length estimation module...")
    
    # Test case 1: Basic stride estimation
    sample_landmarks = {
        'right_hip': (0.45, 0.4, 0, 0.99),
        'right_knee': (0.48, 0.6, 0, 0.95),
        'right_ankle': (0.52, 0.8, 0, 0.95),
        'right_foot_index': (0.55, 0.85, 0, 0.90),
        'right_shoulder': (0.42, 0.25, 0, 0.95),
        'nose': (0.40, 0.1, 0, 0.90)
    }
    
    result, tracking = estimate_stride_length(sample_landmarks, height_cm=175.0)
    print(f"\nBasic stride estimation:\n{result}")
    
    # Test case 2: With temporal tracking
    result_temporal, tracking_updated = estimate_stride_length(
        sample_landmarks, 
        frame_index=1, 
        height_cm=175.0, 
        temporal_tracking=tracking
    )
    print(f"\nWith temporal tracking:\n{result_temporal}")
    
    # Test case 3: Missing landmarks
    incomplete_landmarks = {
        'right_hip': (0.45, 0.4, 0, 0.99),
        'right_knee': (0.48, 0.6, 0, 0.95)
        # Missing ankle and foot
    }
    
    result_incomplete, _ = estimate_stride_length(incomplete_landmarks)
    print(f"\nMissing landmarks test:\n{result_incomplete}")