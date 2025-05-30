# trunk_angle.py

"""
Calculates trunk forward lean angle for running biomechanics analysis.

This metric measures the forward lean of the runner's trunk relative to vertical,
which is crucial for efficient running mechanics. Optimal forward lean (5-10°) 
helps with momentum and reduces braking forces, while excessive or insufficient 
lean can lead to inefficiencies and injury risk.
"""

import logging
import math
from typing import Dict, Tuple, Optional, Any

logger = logging.getLogger(__name__)

# Define type aliases consistent with foot_crossover.py
Landmark = Tuple[float, float, float, float]  # (x, y, z, visibility)
LandmarksDict = Dict[str, Landmark]
TrunkAngleResult = Dict[str, Optional[Any]]  # Values can be float, bool, or str

def calculate_trunk_angle(
    landmarks: LandmarksDict,
    smoothing_factor: float = 0.3,
    previous_angle: Optional[float] = None,
    estimated_speed_mps: Optional[float] = None
) -> TrunkAngleResult:
    """
    Calculate trunk forward lean angle relative to vertical from side view.
    
    The trunk angle is calculated using hip and shoulder midpoints to determine
    the trunk's deviation from vertical. Positive angles indicate forward lean,
    negative angles indicate backward lean. Temporal smoothing is applied to
    reduce noise between frames.
    
    Parameters:
    -----------
    landmarks : LandmarksDict
        Dictionary containing 3D coordinates and visibility of pose landmarks.
        Required keys: 'left_hip', 'right_hip', 'left_shoulder', 'right_shoulder'
        Optional keys: 'neck' (for improved upper trunk representation)
        Each landmark is a tuple: (x, y, z, visibility)
        
    smoothing_factor : float, default=0.3
        Temporal smoothing factor (0.0-1.0). Higher values provide more smoothing
        but slower response to actual changes. 0.0 = no smoothing, 1.0 = maximum smoothing.
        
    previous_angle : Optional[float], default=None
        Previous frame's trunk angle for temporal smoothing. If None, no smoothing applied.
        
    estimated_speed_mps : Optional[float], default=None
        Estimated running speed in meters per second. Used to adjust optimal range
        recommendations based on running pace.
    
    Returns:
    --------
    TrunkAngleResult
        Dictionary containing:
        - "angle_degrees" (Optional[float]): Forward lean angle in degrees. 
            Positive = forward lean, negative = backward lean. None if calculation fails.
        - "is_optimal" (Optional[bool]): True if angle is within optimal range (5-10°). 
            None if calculation fails.
        - "assessment" (Optional[str]): Categorical assessment of trunk position.
            Values: 'backward_lean', 'insufficient_forward_lean', 'optimal_forward_lean',
            'moderate_forward_lean', 'excessive_forward_lean'. None if calculation fails.
        - "assessment_detail" (Optional[str]): Detailed explanation and recommendations.
            None if calculation fails.
        - "confidence" (Optional[float]): Confidence score (0.0-1.0) based on landmark
            visibility and measurement reliability. None if calculation fails.
        - "calculation_successful" (bool): True if metrics were calculated successfully.
    """
    
    required_landmarks = ['left_hip', 'right_hip', 'left_shoulder', 'right_shoulder']
    
    # Initialize default return values for failure case
    result: TrunkAngleResult = {
        "angle_degrees": None,
        "is_optimal": None,
        "assessment": None,
        "assessment_detail": None,
        "confidence": None,
        "calculation_successful": False
    }
    
    # Check for presence of required landmarks
    for lm_name in required_landmarks:
        if lm_name not in landmarks:
            logger.warning(f"Required landmark '{lm_name}' not found for trunk angle calculation.")
            return result
    
    try:
        # Calculate hip midpoint
        hip_center_x = (landmarks['left_hip'][0] + landmarks['right_hip'][0]) / 2
        hip_center_y = (landmarks['left_hip'][1] + landmarks['right_hip'][1]) / 2
        
        # Calculate shoulder midpoint
        shoulder_center_x = (landmarks['left_shoulder'][0] + landmarks['right_shoulder'][0]) / 2
        shoulder_center_y = (landmarks['left_shoulder'][1] + landmarks['right_shoulder'][1]) / 2
        
        # Use neck point if available for more accurate upper trunk representation
        if 'neck' in landmarks:
            neck_x = landmarks['neck'][0]
            neck_y = landmarks['neck'][1]
            
            # Average shoulder midpoint and neck for upper trunk
            upper_trunk_x = (shoulder_center_x + neck_x) / 2
            upper_trunk_y = (shoulder_center_y + neck_y) / 2
        else:
            upper_trunk_x = shoulder_center_x
            upper_trunk_y = shoulder_center_y
        
        # Calculate trunk vector from hip to upper trunk
        trunk_vector_x = upper_trunk_x - hip_center_x
        trunk_vector_y = hip_center_y - upper_trunk_y  # Invert y-axis for typical image coordinates
        
        # Check for degenerate case (points too close together)
        trunk_length = math.sqrt(trunk_vector_x**2 + trunk_vector_y**2)
        if trunk_length < 1e-5:
            logger.warning("Hip and shoulder points are too close together for reliable trunk angle calculation.")
            return result
        
        # Calculate angle with vertical (y-axis) using arctan2
        # arctan2(x, y) gives angle from positive y-axis to vector (x, y)
        angle_radians = math.atan2(trunk_vector_x, trunk_vector_y)
        angle_degrees = math.degrees(angle_radians)
        
        # Apply temporal smoothing if previous angle provided
        if previous_angle is not None:
            # Handle angle wrapping near ±180 degrees
            angle_diff = angle_degrees - previous_angle
            if angle_diff > 180:
                angle_diff -= 360
            elif angle_diff < -180:
                angle_diff += 360
            
            # Apply exponential moving average
            angle_degrees = previous_angle + (angle_diff * (1 - smoothing_factor))
        
        # Calculate confidence based on landmark visibility and measurement quality
        confidence = _calculate_trunk_confidence(landmarks, angle_degrees, previous_angle)
        
        # Define optimal trunk lean range based on biomechanical research
        min_optimal = 5.0  # degrees
        max_optimal = 10.0  # degrees
        
        # Assess trunk position
        is_optimal = min_optimal <= angle_degrees <= max_optimal
        
        # Determine assessment category
        if angle_degrees < -2:
            assessment = "backward_lean"
        elif angle_degrees < min_optimal:
            assessment = "insufficient_forward_lean"
        elif angle_degrees <= max_optimal:
            assessment = "optimal_forward_lean"
        elif angle_degrees <= 15:
            assessment = "moderate_forward_lean"
        else:
            assessment = "excessive_forward_lean"
        
        # Generate detailed assessment with speed-based recommendations
        assessment_detail = _generate_assessment_detail(assessment, angle_degrees, estimated_speed_mps)
        
        # Populate successful result
        result.update({
            "angle_degrees": angle_degrees,
            "is_optimal": is_optimal,
            "assessment": assessment,
            "assessment_detail": assessment_detail,
            "confidence": confidence,
            "calculation_successful": True
        })
        
    except KeyError as e:
        logger.error(f"Missing landmark during trunk angle calculation: {e}", exc_info=True)
        
    except Exception as e:
        logger.exception(f"Unexpected error during trunk angle calculation: {e}")
    
    return result


def _calculate_trunk_confidence(
    landmarks: LandmarksDict, 
    current_angle: float, 
    previous_angle: Optional[float]
) -> float:
    """
    Calculate confidence score for trunk angle measurement.
    
    Parameters:
    -----------
    landmarks : LandmarksDict
        Dictionary containing pose landmarks with visibility scores
    current_angle : float
        Current calculated trunk angle in degrees
    previous_angle : Optional[float]
        Previous frame's angle for temporal consistency check
        
    Returns:
    --------
    float
        Confidence score between 0.0 and 1.0
    """
    # Base confidence from landmark visibility (4th element of tuple)
    hip_visibility = (landmarks['left_hip'][3] + landmarks['right_hip'][3]) / 2
    shoulder_visibility = (landmarks['left_shoulder'][3] + landmarks['right_shoulder'][3]) / 2
    
    # Weighted average (hips typically more reliable than shoulders)
    base_confidence = (hip_visibility * 0.6) + (shoulder_visibility * 0.4)
    
    # Apply penalties for potential issues
    confidence = base_confidence
    
    # Penalty for dramatic angle changes between frames (likely noise)
    if previous_angle is not None:
        angle_change = abs(current_angle - previous_angle)
        if angle_change > 20:  # Degrees per frame
            confidence *= 0.7
            logger.debug(f"Large angle change detected: {angle_change:.1f}°")
    
    # Penalty for physiologically implausible angles
    if abs(current_angle) > 30:  # Extreme lean angles
        confidence *= 0.8
        logger.debug(f"Extreme trunk angle detected: {current_angle:.1f}°")
    
    # Check for potential occlusion
    if _is_trunk_potentially_occluded(landmarks):
        confidence *= 0.8
        logger.debug("Potential trunk occlusion detected")
    
    return max(min(confidence, 1.0), 0.0)  # Clamp to [0, 1]


def _is_trunk_potentially_occluded(landmarks: LandmarksDict) -> bool:
    """
    Check if trunk landmarks might be occluded by arms or other factors.
    
    Parameters:
    -----------
    landmarks : LandmarksDict
        Dictionary containing pose landmarks
        
    Returns:
    --------
    bool
        True if occlusion is likely detected
    """
    # Check if both elbows are present and potentially occluding trunk
    elbow_landmarks = ['left_elbow', 'right_elbow']
    shoulder_landmarks = ['left_shoulder', 'right_shoulder']
    
    if all(lm in landmarks for lm in elbow_landmarks + shoulder_landmarks):
        # Check if elbows are positioned in front of shoulders (potential occlusion)
        left_elbow_forward = landmarks['left_elbow'][0] < landmarks['left_shoulder'][0]
        right_elbow_forward = landmarks['right_elbow'][0] < landmarks['right_shoulder'][0]
        
        # Both arms crossed in front suggests potential trunk occlusion
        if left_elbow_forward and right_elbow_forward:
            return True
    
    # Check for very low visibility scores (additional occlusion indicator)
    trunk_landmarks = ['left_hip', 'right_hip', 'left_shoulder', 'right_shoulder']
    avg_visibility = sum(landmarks[lm][3] for lm in trunk_landmarks) / len(trunk_landmarks)
    
    if avg_visibility < 0.5:
        return True
    
    return False


def _generate_assessment_detail(
    assessment: str, 
    angle_degrees: float, 
    estimated_speed_mps: Optional[float]
) -> str:
    """
    Generate detailed assessment text with recommendations.
    
    Parameters:
    -----------
    assessment : str
        Categorical assessment of trunk position
    angle_degrees : float
        Calculated trunk angle in degrees
    estimated_speed_mps : Optional[float]
        Estimated running speed for context-specific advice
        
    Returns:
    --------
    str
        Detailed assessment with recommendations
    """
    # Generate speed-specific modifier
    speed_modifier = ""
    if estimated_speed_mps is not None:
        if estimated_speed_mps > 5.5:  # ~3:00 min/km pace (fast)
            speed_modifier = " (Note: Faster speeds typically benefit from slightly increased forward lean)"
        elif estimated_speed_mps < 2.7:  # ~6:00 min/km pace (easy)
            speed_modifier = " (Note: Slower speeds typically require less forward lean)"
    
    # Assessment details with specific recommendations
    details = {
        "backward_lean": (
            f"Backward trunk lean detected ({angle_degrees:.1f}°). This creates braking forces "
            "and reduces efficiency. Focus on leaning forward slightly from the ankles, not the waist."
        ),
        "insufficient_forward_lean": (
            f"Forward lean ({angle_degrees:.1f}°) is less than optimal range (5-10°). "
            f"Consider increasing forward lean slightly from the ankles{speed_modifier}."
        ),
        "optimal_forward_lean": (
            f"Trunk angle ({angle_degrees:.1f}°) is within optimal range (5-10°) for efficient running{speed_modifier}."
        ),
        "moderate_forward_lean": (
            f"Forward lean ({angle_degrees:.1f}°) is slightly higher than optimal range (5-10°). "
            f"This may be appropriate for acceleration or uphill running{speed_modifier}."
        ),
        "excessive_forward_lean": (
            f"Forward lean ({angle_degrees:.1f}°) is excessive. This increases stress on the lower back "
            "and hamstrings. Try running more upright with forward lean initiated from the ankles."
        )
    }
    
    return details.get(assessment, f"Trunk angle: {angle_degrees:.1f}°")


if __name__ == "__main__":
    print("Testing calculate_trunk_angle module...")
    
    # Example 1: Optimal forward lean
    sample_landmarks_optimal: LandmarksDict = {
        'left_hip': (0.45, 0.6, 0, 0.95),
        'right_hip': (0.55, 0.6, 0, 0.95),
        'left_shoulder': (0.46, 0.4, 0, 0.90),
        'right_shoulder': (0.54, 0.4, 0, 0.90),
        'neck': (0.50, 0.35, 0, 0.85)
    }
    
    result_optimal = calculate_trunk_angle(sample_landmarks_optimal)
    print(f"\nOptimal Forward Lean Example:\n{result_optimal}")
    
    # Example 2: Excessive forward lean
    sample_landmarks_excessive: LandmarksDict = {
        'left_hip': (0.45, 0.6, 0, 0.95),
        'right_hip': (0.55, 0.6, 0, 0.95),
        'left_shoulder': (0.35, 0.4, 0, 0.90),  # Shoulders way forward
        'right_shoulder': (0.45, 0.4, 0, 0.90)
    }
    
    result_excessive = calculate_trunk_angle(sample_landmarks_excessive, estimated_speed_mps=3.0)
    print(f"\nExcessive Forward Lean Example:\n{result_excessive}")
    
    # Example 3: Backward lean (inefficient)
    sample_landmarks_backward: LandmarksDict = {
        'left_hip': (0.45, 0.6, 0, 0.95),
        'right_hip': (0.55, 0.6, 0, 0.95),
        'left_shoulder': (0.52, 0.4, 0, 0.90),  # Shoulders behind hips
        'right_shoulder': (0.58, 0.4, 0, 0.90)
    }
    
    result_backward = calculate_trunk_angle(sample_landmarks_backward)
    print(f"\nBackward Lean Example:\n{result_backward}")
    
    # Example 4: Missing required landmark
    sample_landmarks_missing: LandmarksDict = {
        'left_hip': (0.45, 0.6, 0, 0.95),
        # Missing right_hip
        'left_shoulder': (0.46, 0.4, 0, 0.90),
        'right_shoulder': (0.54, 0.4, 0, 0.90)
    }
    
    result_missing = calculate_trunk_angle(sample_landmarks_missing)
    print(f"\nMissing Landmark Example:\n{result_missing}")
    
    # Example 5: With temporal smoothing
    previous_angle = 8.2
    result_smoothed = calculate_trunk_angle(
        sample_landmarks_optimal, 
        smoothing_factor=0.5, 
        previous_angle=previous_angle
    )
    print(f"\nTemporal Smoothing Example (previous: {previous_angle}°):\n{result_smoothed}")