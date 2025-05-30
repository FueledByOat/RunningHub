# knee_angle.py

"""
Calculates knee flexion/extension angles from pose landmarks.

This module analyzes knee joint angles to assess running biomechanics,
identify potential overstriding, and evaluate leg extension patterns
that may impact performance or injury risk.
"""

import logging
import math
from typing import Dict, Tuple, Optional, Any, Literal

logger = logging.getLogger(__name__)

# Type aliases for clarity and consistency
Landmark = Tuple[float, float, float, float]  # (x, y, z, visibility)
LandmarksDict = Dict[str, Landmark]
KneeAngleResult = Dict[str, Optional[Any]]
Side = Literal['left', 'right']

# Configuration constants
MIN_SEGMENT_LENGTH = 1e-10  # Minimum segment length to avoid division by zero
STRAIGHT_KNEE_ANGLE = 180.0  # Degrees for fully extended knee
BENT_KNEE_ANGLE = 0.0       # Degrees for fully flexed knee


def calculate_knee_angle(
    landmarks: LandmarksDict,
    side: Side
) -> KneeAngleResult:
    """
    Calculate knee flexion/extension angle from hip-knee-ankle landmarks.
    
    Computes the interior angle at the knee joint using vector dot product.
    Returns extension angle where 180° represents fully straight leg and
    smaller values indicate increasing knee flexion.
    
    Parameters:
    -----------
    landmarks : LandmarksDict
        Dictionary containing pose landmarks with (x, y, z, visibility) coordinates.
        Required keys: '{side}_hip', '{side}_knee', '{side}_ankle'
        
    side : Side
        Leg side to analyze ('left' or 'right')
        
    Returns:
    --------
    KneeAngleResult
        Dictionary containing:
        - "knee_angle" (Optional[float]): Knee extension angle in degrees (180° = straight)
        - "knee_flexion" (Optional[float]): Knee flexion from straight position in degrees
        - "leg_extension_assessment" (Optional[str]): Qualitative assessment of leg extension
        - "side" (str): Which leg was analyzed
        - "calculation_successful" (bool): True if angle calculated successfully
    """
    
    required_landmarks = [f'{side}_hip', f'{side}_knee', f'{side}_ankle']
    
    # Initialize default return values
    result: KneeAngleResult = {
        "knee_angle": None,
        "knee_flexion": None,
        "leg_extension_assessment": None,
        "side": side,
        "calculation_successful": False
    }
    
    # Validate required landmarks presence
    for landmark_name in required_landmarks:
        if landmark_name not in landmarks:
            logger.warning(f"Required landmark '{landmark_name}' not found for knee angle calculation.")
            return result
    
    try:
        # Extract landmark coordinates (x, y only)
        hip = landmarks[f'{side}_hip'][:2]
        knee = landmarks[f'{side}_knee'][:2]
        ankle = landmarks[f'{side}_ankle'][:2]
        
        # Calculate knee angle
        knee_angle = _calculate_joint_angle(hip, knee, ankle)
        
        if knee_angle is None:
            return result
        
        # Calculate flexion from straight position
        knee_flexion = STRAIGHT_KNEE_ANGLE - knee_angle
        
        # Assess leg extension
        leg_assessment = _assess_leg_extension(knee_angle)
        
        # Update result with calculated values
        result.update({
            "knee_angle": knee_angle,
            "knee_flexion": knee_flexion,
            "leg_extension_assessment": leg_assessment,
            "calculation_successful": True
        })
        
        logger.debug(f"{side.capitalize()} knee analysis: angle={knee_angle:.1f}°, "
                    f"flexion={knee_flexion:.1f}°, assessment={leg_assessment}")
        
    except KeyError as e:
        logger.error(f"Missing landmark during knee angle calculation: {e}", exc_info=True)
        
    except Exception as e:
        logger.exception(f"Unexpected error during knee angle calculation: {e}")
    
    return result


def _calculate_joint_angle(
    proximal: Tuple[float, float],
    joint: Tuple[float, float],
    distal: Tuple[float, float]
) -> Optional[float]:
    """
    Calculate interior angle at a joint using vector dot product method.
    
    Parameters:
    -----------
    proximal : Tuple[float, float]
        Coordinates of proximal landmark (e.g., hip)
    joint : Tuple[float, float]
        Coordinates of joint center (e.g., knee)
    distal : Tuple[float, float]
        Coordinates of distal landmark (e.g., ankle)
        
    Returns:
    --------
    Optional[float]
        Joint angle in degrees, or None if calculation fails
    """
    try:
        # Calculate leg segment vectors
        proximal_to_joint = [joint[0] - proximal[0], joint[1] - proximal[1]]
        joint_to_distal = [distal[0] - joint[0], distal[1] - joint[1]]
        
        # Calculate vector magnitudes
        proximal_magnitude = math.sqrt(proximal_to_joint[0]**2 + proximal_to_joint[1]**2)
        distal_magnitude = math.sqrt(joint_to_distal[0]**2 + joint_to_distal[1]**2)
        
        # Check for zero-length segments
        if proximal_magnitude < MIN_SEGMENT_LENGTH or distal_magnitude < MIN_SEGMENT_LENGTH:
            logger.warning("Near-zero leg segment length detected, cannot calculate knee angle")
            return None
        
        # Calculate dot product
        dot_product = (proximal_to_joint[0] * joint_to_distal[0] + 
                      proximal_to_joint[1] * joint_to_distal[1])
        
        # Calculate cosine of angle
        cos_angle = dot_product / (proximal_magnitude * distal_magnitude)
        cos_angle = max(min(cos_angle, 1.0), -1.0)  # Clamp to valid range [-1, 1]
        
        # Calculate angle in radians then convert to degrees
        angle_rad = math.acos(cos_angle)
        interior_angle = math.degrees(angle_rad)
        
        # Convert to extension angle (complement of interior angle)
        # Interior angle of 0° = straight leg (180° extension)
        # Interior angle of 90° = bent leg (90° extension)
        extension_angle = STRAIGHT_KNEE_ANGLE - interior_angle
        
        return extension_angle
        
    except Exception as e:
        logger.error(f"Error in joint angle calculation: {e}")
        return None


def _assess_leg_extension(knee_angle: float) -> str:
    """
    Provide qualitative assessment of leg extension based on knee angle.
    
    Parameters:
    -----------
    knee_angle : float
        Knee extension angle in degrees
        
    Returns:
    --------
    str
        Qualitative assessment of leg extension
    """
    if knee_angle >= 165:
        return "fully_extended"
    elif knee_angle >= 145:
        return "well_extended"  
    elif knee_angle >= 120:
        return "moderately_extended"
    elif knee_angle >= 90:
        return "moderately_flexed"
    elif knee_angle >= 45:
        return "well_flexed"
    else:
        return "highly_flexed"


def calculate_bilateral_knee_angles(landmarks: LandmarksDict) -> Dict[str, KneeAngleResult]:
    """
    Calculate knee angles for both legs simultaneously.
    
    Parameters:
    -----------
    landmarks : LandmarksDict
        Dictionary containing pose landmarks for both legs
        
    Returns:
    --------
    Dict[str, KneeAngleResult]
        Dictionary with 'left' and 'right' keys containing respective knee analyses
    """
    results = {}
    
    for side in ['left', 'right']:
        results[side] = calculate_knee_angle(landmarks, side)
    
    return results


def analyze_knee_asymmetry(
    left_result: KneeAngleResult, 
    right_result: KneeAngleResult
) -> Dict[str, Any]:
    """
    Analyze asymmetry between left and right knee angles.
    
    Parameters:
    -----------
    left_result : KneeAngleResult
        Left knee analysis result
    right_result : KneeAngleResult
        Right knee analysis result
        
    Returns:
    --------
    Dict[str, Any]
        Asymmetry analysis including angle difference and assessment
    """
    if not (left_result['calculation_successful'] and right_result['calculation_successful']):
        return {
            "angle_difference": None,
            "asymmetry_assessment": "insufficient_data",
            "recommendations": ["Ensure both legs are visible for asymmetry analysis"]
        }
    
    left_angle = left_result['knee_angle']
    right_angle = right_result['knee_angle']
    angle_difference = abs(left_angle - right_angle)
    
    # Assess asymmetry level
    if angle_difference < 5:
        assessment = "minimal_asymmetry"
        recommendations = ["Knee angles are well-matched between legs"]
    elif angle_difference < 10:
        assessment = "mild_asymmetry"
        recommendations = ["Minor asymmetry detected - monitor for consistency"]
    elif angle_difference < 20:
        assessment = "moderate_asymmetry" 
        recommendations = ["Significant asymmetry detected - consider form assessment"]
    else:
        assessment = "high_asymmetry"
        recommendations = ["High asymmetry detected - consider biomechanical evaluation"]
    
    return {
        "angle_difference": angle_difference,
        "asymmetry_assessment": assessment,
        "recommendations": recommendations,
        "left_angle": left_angle,
        "right_angle": right_angle
    }


if __name__ == "__main__":
    print("Testing knee_angle module...")
    
    # Test case 1: Well-extended knee
    extended_landmarks: LandmarksDict = {
        'right_hip': (0.3, 0.4, 0, 0.99),
        'right_knee': (0.32, 0.6, 0, 0.99),
        'right_ankle': (0.34, 0.8, 0, 0.99)
    }
    
    result_extended = calculate_knee_angle(extended_landmarks, 'right')
    print(f"\nExtended Knee Results:")
    print(f"Knee angle: {result_extended['knee_angle']:.1f}°")
    print(f"Knee flexion: {result_extended['knee_flexion']:.1f}°")
    print(f"Assessment: {result_extended['leg_extension_assessment']}")
    
    # Test case 2: Flexed knee (bent leg)
    flexed_landmarks: LandmarksDict = {
        'left_hip': (0.3, 0.4, 0, 0.99),
        'left_knee': (0.35, 0.55, 0, 0.99),
        'left_ankle': (0.25, 0.7, 0, 0.99)  # Ankle pulled back, creating bend
    }
    
    result_flexed = calculate_knee_angle(flexed_landmarks, 'left')
    print(f"\nFlexed Knee Results:")
    print(f"Knee angle: {result_flexed['knee_angle']:.1f}°")
    print(f"Knee flexion: {result_flexed['knee_flexion']:.1f}°")
    print(f"Assessment: {result_flexed['leg_extension_assessment']}")
    
    # Test case 3: Bilateral analysis
    bilateral_landmarks: LandmarksDict = {
        'left_hip': (0.25, 0.4, 0, 0.99),
        'left_knee': (0.23, 0.6, 0, 0.99),
        'left_ankle': (0.21, 0.8, 0, 0.99),
        'right_hip': (0.35, 0.4, 0, 0.99),
        'right_knee': (0.37, 0.6, 0, 0.99),
        'right_ankle': (0.32, 0.75, 0, 0.99)  # Slightly more flexed
    }
    
    bilateral_results = calculate_bilateral_knee_angles(bilateral_landmarks)
    asymmetry_analysis = analyze_knee_asymmetry(
        bilateral_results['left'], 
        bilateral_results['right']
    )
    
    print(f"\nBilateral Analysis:")
    print(f"Left knee: {bilateral_results['left']['knee_angle']:.1f}°")
    print(f"Right knee: {bilateral_results['right']['knee_angle']:.1f}°")
    print(f"Angle difference: {asymmetry_analysis['angle_difference']:.1f}°")
    print(f"Asymmetry assessment: {asymmetry_analysis['asymmetry_assessment']}")
    
    # Test case 4: Missing landmarks
    incomplete_landmarks: LandmarksDict = {
        'right_hip': (0.3, 0.4, 0, 0.99),
        'right_knee': (0.32, 0.6, 0, 0.99)
        # Missing ankle landmark
    }
    
    result_incomplete = calculate_knee_angle(incomplete_landmarks, 'right')
    print(f"\nIncomplete Data Results:")
    print(f"Calculation successful: {result_incomplete['calculation_successful']}")
    
    # Test case 5: Zero-length segment (overlapping landmarks)
    zero_length_landmarks: LandmarksDict = {
        'left_hip': (0.3, 0.4, 0, 0.99),
        'left_knee': (0.3, 0.4, 0, 0.99),  # Same as hip
        'left_ankle': (0.34, 0.8, 0, 0.99)
    }
    
    result_zero = calculate_knee_angle(zero_length_landmarks, 'left')
    print(f"\nZero-length Segment Results:")
    print(f"Calculation successful: {result_zero['calculation_successful']}")
    print(f"Knee angle: {result_zero['knee_angle']}")