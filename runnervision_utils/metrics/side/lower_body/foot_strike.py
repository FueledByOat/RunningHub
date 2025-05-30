# foot_strike.py

"""
Analyzes foot strike patterns during running from side-view pose landmarks.

This module determines whether a runner exhibits heel strike, midfoot strike, 
or forefoot strike patterns. The strike pattern analysis is crucial for:
- Understanding impact forces and injury risk
- Optimizing running efficiency
- Identifying biomechanical compensations
- Guiding footwear and training recommendations

Foot strike patterns have different biomechanical implications:
- Heel strike: Most common, may increase impact forces and loading rates
- Midfoot strike: Often provides balance between impact absorption and propulsion
- Forefoot strike: May reduce impact forces but increases calf/Achilles tendon loading
"""

import logging
import math
from typing import Dict, Tuple, Optional, Any, Union

logger = logging.getLogger(__name__)

# Define type aliases for clarity and consistency
Landmark = Tuple[float, float, float, float]  # (x, y, z, visibility)
LandmarksDict = Dict[str, Landmark]
StancePhaseInfo = Dict[str, Union[bool, float, str]]  # Stance phase analysis results
FootStrikeResult = Dict[str, Union[str, float, Optional[float]]]

def calculate_foot_strike(
    landmarks: LandmarksDict, 
    stance_phase: StancePhaseInfo,
    vertical_threshold_heel: float = 0.015,
    vertical_threshold_forefoot: float = -0.01,
    confidence_scaling_heel: float = 0.03,
    confidence_scaling_forefoot: float = 0.02,
    angle_contradiction_penalty: float = 0.7
) -> FootStrikeResult:
    """
    Determine foot strike pattern (heel, midfoot, forefoot) from side-view pose landmarks.
    
    This function analyzes the relative positions of the heel and toe at foot contact
    to classify the strike pattern. Multiple biomechanical indicators are used for
    robust classification including vertical position differences, foot angles, and
    ankle angles.
    
    Parameters:
    -----------
    landmarks : LandmarksDict
        Dictionary containing 3D coordinates and visibility of detected pose landmarks.
        Required keys: 'right_heel', 'right_foot_index', 'right_ankle'.
        Each landmark is a tuple: (x, y, z, visibility).
        Coordinates are typically normalized (0.0-1.0).
        
    stance_phase : StancePhaseInfo
        Dictionary containing stance phase analysis results.
        Must include 'is_stance_phase' (bool) indicating if foot is in contact with ground.
        
    vertical_threshold_heel : float, default=0.015
        Minimum vertical difference (heel_y - toe_y) to classify as heel strike.
        Positive values indicate heel is lower than toe in image coordinates.
        
    vertical_threshold_forefoot : float, default=-0.01
        Maximum vertical difference (heel_y - toe_y) to classify as forefoot strike.
        Negative values indicate toe is lower than heel in image coordinates.
        
    confidence_scaling_heel : float, default=0.03
        Scaling factor for heel strike confidence calculation.
        Higher values result in lower confidence for same vertical difference.
        
    confidence_scaling_forefoot : float, default=0.02
        Scaling factor for forefoot strike confidence calculation.
        Higher values result in lower confidence for same vertical difference.
        
    angle_contradiction_penalty : float, default=0.7
        Multiplier applied to confidence when foot angle contradicts strike pattern.
        Should be between 0.0 and 1.0.
    
    Returns:
    --------
    FootStrikeResult
        Dictionary containing:
        - "strike_pattern" (str): One of 'heel', 'midfoot', 'forefoot', or 'not_applicable'
        - "confidence" (float): Confidence score (0.0-1.0) for the classification
        - "vertical_difference" (float): Heel Y - Toe Y position difference
        - "foot_angle" (float): Foot angle relative to horizontal (degrees)
        - "ankle_angle" (float): Ankle dorsiflexion/plantarflexion angle (degrees)
        - "landing_stiffness" (str): One of 'stiff', 'moderate', 'compliant', or 'not_applicable'
        - "calculation_successful" (bool): True if analysis completed, False if failed
        
    Notes:
    ------
    - Analysis only performed during stance phase (foot contact with ground)
    - Coordinate system assumes higher Y values = lower position in image
    - Positive foot angle = heel lower than toe (dorsiflexion)
    - Negative foot angle = toe lower than heel (plantarflexion)
    - Ankle angle interpretation: negative = stiff landing, positive = compliant landing
    """
    
    required_landmarks = ['right_heel', 'right_foot_index', 'right_ankle']
    
    # Initialize default return values for failure/non-applicable cases
    result: FootStrikeResult = {
        "strike_pattern": 'not_applicable',
        "confidence": 0.0,
        "vertical_difference": 0.0,
        "foot_angle": 0.0,
        "ankle_angle": 0.0,
        "landing_stiffness": 'not_applicable',
        "calculation_successful": False
    }
    
    # Validate stance phase information
    if not isinstance(stance_phase, dict) or 'is_stance_phase' not in stance_phase:
        logger.warning("Invalid stance_phase parameter. Must be dict with 'is_stance_phase' key.")
        return result
    
    # Check if foot is in stance phase
    if not stance_phase['is_stance_phase']:
        logger.debug("Foot not in stance phase. Foot strike analysis not applicable.")
        result["calculation_successful"] = True  # Not an error, just not applicable
        return result
    
    # Check for presence of all required landmarks
    for landmark_name in required_landmarks:
        if landmark_name not in landmarks:
            logger.warning(f"Required landmark '{landmark_name}' not found for foot strike calculation.")
            return result
    
    try:
        # Extract landmark coordinates
        right_heel = landmarks['right_heel']
        right_foot_index = landmarks['right_foot_index']
        right_ankle = landmarks['right_ankle']
        
        # Extract individual coordinate components
        right_heel_x, right_heel_y = right_heel[0], right_heel[1]
        right_foot_index_x, right_foot_index_y = right_foot_index[0], right_foot_index[1]
        right_ankle_x, right_ankle_y = right_ankle[0], right_ankle[1]
        
        # Calculate primary indicator: vertical difference between heel and toe
        # Positive = heel lower than toe, Negative = toe lower than heel
        vertical_difference = right_heel_y - right_foot_index_y
        
        # Calculate foot angle relative to horizontal
        # This provides additional biomechanical context
        foot_angle = _calculate_foot_angle(
            right_heel_x, right_heel_y, 
            right_foot_index_x, right_foot_index_y
        )
        
        # Calculate ankle angle (dorsiflexion/plantarflexion)
        # Useful for assessing landing stiffness
        ankle_angle = _calculate_ankle_angle(
            right_ankle_x, right_ankle_y,
            right_heel_x, right_heel_y
        )
        
        # Classify strike pattern using vertical difference as primary indicator
        strike_pattern, confidence = _classify_strike_pattern(
            vertical_difference,
            vertical_threshold_heel,
            vertical_threshold_forefoot,
            confidence_scaling_heel,
            confidence_scaling_forefoot
        )
        
        # Apply secondary validation using foot angle
        confidence = _apply_angle_validation(
            strike_pattern, foot_angle, confidence, angle_contradiction_penalty
        )
        
        # Classify landing stiffness based on ankle angle
        landing_stiffness = _classify_landing_stiffness(ankle_angle)
        
        # Update result with calculated values
        result.update({
            "strike_pattern": strike_pattern,
            "confidence": confidence,
            "vertical_difference": vertical_difference,
            "foot_angle": foot_angle,
            "ankle_angle": ankle_angle,
            "landing_stiffness": landing_stiffness,
            "calculation_successful": True
        })
        
        logger.debug(f"Foot strike analysis completed: {strike_pattern} "
                    f"(confidence: {confidence:.2f}, vertical_diff: {vertical_difference:.3f})")
        
    except KeyError as e:
        logger.error(f"Missing landmark during foot strike calculation: {e}", exc_info=True)
        
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid data type in foot strike calculation: {e}", exc_info=True)
        
    except Exception as e:
        logger.exception(f"Unexpected error during foot strike calculation: {e}")
    
    return result


def _calculate_foot_angle(heel_x: float, heel_y: float, toe_x: float, toe_y: float) -> float:
    """
    Calculate the angle of the foot relative to horizontal.
    
    Parameters:
    -----------
    heel_x, heel_y : float
        Heel landmark coordinates
    toe_x, toe_y : float
        Toe (foot index) landmark coordinates
        
    Returns:
    --------
    float
        Foot angle in degrees. Positive = heel lower than toe, Negative = toe lower than heel
    """
    dx = toe_x - heel_x
    dy = toe_y - heel_y
    
    if abs(dx) < 1e-6:  # Avoid division by zero
        return 0.0
    
    return math.degrees(math.atan2(dy, dx))


def _calculate_ankle_angle(ankle_x: float, ankle_y: float, heel_x: float, heel_y: float) -> float:
    """
    Calculate ankle angle (dorsiflexion/plantarflexion).
    
    Parameters:
    -----------
    ankle_x, ankle_y : float
        Ankle landmark coordinates
    heel_x, heel_y : float
        Heel landmark coordinates
        
    Returns:
    --------
    float
        Ankle angle in degrees. Negative = more vertical shin (stiff), Positive = angled shin (compliant)
    """
    dx = heel_x - ankle_x
    dy = heel_y - ankle_y
    
    if abs(dx) < 1e-6:  # Avoid division by zero
        return 0.0
    
    return math.degrees(math.atan2(dy, dx))


def _classify_strike_pattern(
    vertical_difference: float,
    heel_threshold: float,
    forefoot_threshold: float,
    heel_scaling: float,
    forefoot_scaling: float
) -> Tuple[str, float]:
    """
    Classify strike pattern based on vertical difference between heel and toe.
    
    Parameters:
    -----------
    vertical_difference : float
        Heel Y - Toe Y position difference
    heel_threshold : float
        Threshold for heel strike classification
    forefoot_threshold : float
        Threshold for forefoot strike classification
    heel_scaling : float
        Confidence scaling factor for heel strike
    forefoot_scaling : float
        Confidence scaling factor for forefoot strike
        
    Returns:
    --------
    Tuple[str, float]
        Strike pattern classification and confidence score
    """
    if vertical_difference > heel_threshold:
        # Heel significantly lower than toe
        pattern = "heel"
        confidence = min(1.0, vertical_difference / heel_scaling)
        
    elif vertical_difference < forefoot_threshold:
        # Toe significantly lower than heel
        pattern = "forefoot"
        confidence = min(1.0, abs(vertical_difference) / forefoot_scaling)
        
    else:
        # Between thresholds - classified as midfoot
        pattern = "midfoot"
        # Higher confidence when closer to center of midfoot range
        midfoot_range = heel_threshold - forefoot_threshold
        distance_from_center = abs(vertical_difference - (heel_threshold + forefoot_threshold) / 2)
        confidence = 1.0 - min(1.0, (distance_from_center * 2) / midfoot_range)
    
    return pattern, confidence


def _apply_angle_validation(
    strike_pattern: str, 
    foot_angle: float, 
    confidence: float, 
    penalty: float
) -> float:
    """
    Apply secondary validation using foot angle to adjust confidence.
    
    Parameters:
    -----------
    strike_pattern : str
        Primary strike pattern classification
    foot_angle : float
        Calculated foot angle in degrees
    confidence : float
        Initial confidence score
    penalty : float
        Penalty multiplier for contradicting angles
        
    Returns:
    --------
    float
        Adjusted confidence score
    """
    # Check for contradicting signals between vertical difference and foot angle
    if strike_pattern == "heel" and foot_angle < -5:
        # Heel strike but foot angle suggests toe is lower - contradiction
        logger.debug(f"Heel strike classification contradicted by foot angle ({foot_angle:.1f}°)")
        confidence *= penalty
        
    elif strike_pattern == "forefoot" and foot_angle > 5:
        # Forefoot strike but foot angle suggests heel is lower - contradiction
        logger.debug(f"Forefoot strike classification contradicted by foot angle ({foot_angle:.1f}°)")
        confidence *= penalty
    
    return confidence


def _classify_landing_stiffness(ankle_angle: float) -> str:
    """
    Classify landing stiffness based on ankle angle at foot contact.
    
    Parameters:
    -----------
    ankle_angle : float
        Ankle angle in degrees
        
    Returns:
    --------
    str
        Landing stiffness classification: 'stiff', 'moderate', or 'compliant'
    """
    if ankle_angle < -15:
        return "stiff"
    elif ankle_angle > 5:
        return "compliant"
    else:
        return "moderate"


if __name__ == "__main__":
    print("Testing calculate_foot_strike module...")
    
    # Mock stance phase data for testing
    stance_active = {"is_stance_phase": True}
    stance_inactive = {"is_stance_phase": False}
    
    # Example 1: Clear heel strike pattern
    sample_landmarks_heel: LandmarksDict = {
        'right_heel': (0.5, 0.75, 0, 0.95),      # Heel lower in image
        'right_foot_index': (0.55, 0.73, 0, 0.95),  # Toe higher in image
        'right_ankle': (0.48, 0.70, 0, 0.95)     # Ankle position
    }
    result_heel = calculate_foot_strike(sample_landmarks_heel, stance_active)
    print(f"\nHeel Strike Test:\n{result_heel}")
    
    # Example 2: Clear forefoot strike pattern
    sample_landmarks_forefoot: LandmarksDict = {
        'right_heel': (0.5, 0.73, 0, 0.95),      # Heel higher in image
        'right_foot_index': (0.55, 0.76, 0, 0.95),  # Toe lower in image
        'right_ankle': (0.48, 0.70, 0, 0.95)     # Ankle position
    }
    result_forefoot = calculate_foot_strike(sample_landmarks_forefoot, stance_active)
    print(f"\nForefoot Strike Test:\n{result_forefoot}")
    
    # Example 3: Midfoot strike pattern
    sample_landmarks_midfoot: LandmarksDict = {
        'right_heel': (0.5, 0.745, 0, 0.95),     # Heel and toe at similar height
        'right_foot_index': (0.55, 0.744, 0, 0.95),
        'right_ankle': (0.48, 0.70, 0, 0.95)
    }
    result_midfoot = calculate_foot_strike(sample_landmarks_midfoot, stance_active)
    print(f"\nMidfoot Strike Test:\n{result_midfoot}")
    
    # Example 4: Not in stance phase
    result_no_stance = calculate_foot_strike(sample_landmarks_heel, stance_inactive)
    print(f"\nNo Stance Phase Test:\n{result_no_stance}")
    
    # Example 5: Missing landmark
    sample_landmarks_missing: LandmarksDict = {
        'right_heel': (0.5, 0.75, 0, 0.95),
        # Missing 'right_foot_index'
        'right_ankle': (0.48, 0.70, 0, 0.95)
    }
    result_missing = calculate_foot_strike(sample_landmarks_missing, stance_active)
    print(f"\nMissing Landmark Test:\n{result_missing}")
    
    # Example 6: Test with custom thresholds
    result_custom = calculate_foot_strike(
        sample_landmarks_heel, 
        stance_active,
        vertical_threshold_heel=0.01,  # More sensitive heel detection
        confidence_scaling_heel=0.02   # Higher confidence scaling
    )
    print(f"\nCustom Thresholds Test:\n{result_custom}")