# foot_landing_position.py

"""
Calculates horizontal foot landing position relative to the body's center of mass.

This metric evaluates whether a runner's foot lands under, ahead of, or behind their
center of mass during the stance phase. Landing position affects:
- Running efficiency and energy expenditure
- Impact forces and injury risk
- Propulsive forces and stride mechanics
- Overall biomechanical efficiency

Key biomechanical implications:
- Landing under CoM: Generally more efficient, reduces braking forces
- Landing ahead of CoM (overstriding): Increases braking forces, may reduce efficiency
- Landing behind CoM: Uncommon but may indicate compensatory patterns
"""

import logging
from typing import Dict, Tuple, Optional, Union

logger = logging.getLogger(__name__)

# Define type aliases for clarity and consistency
Landmark = Tuple[float, float, float, float]  # (x, y, z, visibility)
LandmarksDict = Dict[str, Landmark]
StancePhaseInfo = Dict[str, Union[bool, str, float]]  # Stance phase analysis results
FootLandingResult = Dict[str, Union[float, bool, str]]

def calculate_foot_landing_position(
    landmarks: LandmarksDict,
    stance_phase: StancePhaseInfo,
    tolerance_cm: float = 5.0,
    body_scale_factor: float = 100.0,
    use_hip_center: bool = True
) -> FootLandingResult:
    """
    Calculate horizontal distance from foot landing to center of mass during stance phase.
    
    This function determines where the foot lands relative to the body's center of mass,
    which is a key indicator of running efficiency and biomechanical quality. The analysis
    uses the hip center as a proxy for center of mass, which is biomechanically appropriate
    for running gait analysis.
    
    Parameters:
    -----------
    landmarks : LandmarksDict
        Dictionary containing 3D coordinates and visibility of detected pose landmarks.
        Required keys depend on use_hip_center:
        - If use_hip_center=True: 'left_hip', 'right_hip'
        - Always requires ankle landmarks based on stance_foot from stance_phase
        Each landmark is a tuple: (x, y, z, visibility).
        
    stance_phase : StancePhaseInfo
        Dictionary containing stance phase analysis results.
        Required keys:
        - 'is_stance_phase' (bool): Whether foot is in contact with ground
        - 'stance_foot' (str): Which foot is in stance ('left' or 'right')
        
    tolerance_cm : float, default=5.0
        Tolerance in centimeters for considering foot landing "under" center of mass.
        Smaller values are stricter in defining optimal landing position.
        
    body_scale_factor : float, default=100.0
        Scaling factor to convert normalized coordinates to centimeters.
        Assumes landmarks are normalized (0.0-1.0) and body proportions.
        May need adjustment based on coordinate system and body size.
        
    use_hip_center : bool, default=True
        Whether to use hip center as center of mass proxy.
        If False, would require additional implementation for full CoM calculation.
        
    Returns:
    --------
    FootLandingResult
        Dictionary containing:
        - "distance_cm" (float): Horizontal distance in cm from foot to CoM
            Negative = foot behind CoM, Positive = foot ahead of CoM (overstriding)
        - "is_under_com" (bool): True if foot lands under CoM within tolerance
        - "position_category" (str): 'under', 'ahead', 'behind', or 'not_applicable'
        - "center_of_mass_x" (Optional[float]): Calculated CoM x-coordinate
        - "foot_position_x" (Optional[float]): Foot landing x-coordinate
        - "calculation_successful" (bool): True if analysis completed successfully
        
    Notes:
    ------
    - Analysis only performed during stance phase
    - Hip center serves as reasonable CoM approximation for running analysis
    - Distance calculation assumes normalized coordinate system
    - Positive distance indicates potential overstriding
    """
    
    # Initialize default return values
    result: FootLandingResult = {
        "distance_cm": 0.0,
        "is_under_com": False,
        "position_category": 'not_applicable',
        "center_of_mass_x": None,
        "foot_position_x": None,
        "calculation_successful": False
    }
    
    # Validate stance phase information
    if not isinstance(stance_phase, dict):
        logger.warning("Invalid stance_phase parameter. Must be dictionary.")
        return result
        
    required_stance_keys = ['is_stance_phase', 'stance_foot']
    for key in required_stance_keys:
        if key not in stance_phase:
            logger.warning(f"Required stance_phase key '{key}' not found.")
            return result
    
    # Check if foot is in stance phase
    if not stance_phase['is_stance_phase']:
        logger.debug("Foot not in stance phase. Landing position analysis not applicable.")
        result["calculation_successful"] = True  # Not an error, just not applicable
        return result
    
    try:
        # Calculate center of mass approximation
        center_of_mass_x = _calculate_center_of_mass(landmarks, use_hip_center)
        if center_of_mass_x is None:
            return result
            
        # Get stance foot position
        stance_foot = stance_phase['stance_foot']
        foot_position_x = _get_stance_foot_position(landmarks, stance_foot)
        if foot_position_x is None:
            return result
            
        # Calculate horizontal distance
        horizontal_distance = foot_position_x - center_of_mass_x
        distance_cm = horizontal_distance * body_scale_factor
        
        # Determine if foot is under center of mass within tolerance
        is_under_com = abs(distance_cm) <= tolerance_cm
        
        # Categorize landing position
        position_category = _categorize_landing_position(distance_cm, tolerance_cm)
        
        # Update result with calculated values
        result.update({
            "distance_cm": distance_cm,
            "is_under_com": is_under_com,
            "position_category": position_category,
            "center_of_mass_x": center_of_mass_x,
            "foot_position_x": foot_position_x,
            "calculation_successful": True
        })
        
        logger.debug(f"Foot landing position calculated: {position_category} "
                    f"({distance_cm:.1f}cm from CoM)")
        
    except KeyError as e:
        logger.error(f"Missing landmark during foot landing position calculation: {e}", exc_info=True)
        
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid data type in foot landing position calculation: {e}", exc_info=True)
        
    except Exception as e:
        logger.exception(f"Unexpected error during foot landing position calculation: {e}")
    
    return result


def _calculate_center_of_mass(landmarks: LandmarksDict, use_hip_center: bool) -> Optional[float]:
    """
    Calculate center of mass x-coordinate approximation.
    
    Parameters:
    -----------
    landmarks : LandmarksDict
        Dictionary containing pose landmarks
    use_hip_center : bool
        Whether to use hip center as CoM approximation
        
    Returns:
    --------
    Optional[float]
        Center of mass x-coordinate, or None if calculation fails
    """
    if use_hip_center:
        required_landmarks = ['left_hip', 'right_hip']
        
        for landmark_name in required_landmarks:
            if landmark_name not in landmarks:
                logger.warning(f"Required landmark '{landmark_name}' not found for CoM calculation.")
                return None
        
        try:
            left_hip_x = landmarks['left_hip'][0]
            right_hip_x = landmarks['right_hip'][0]
            return (left_hip_x + right_hip_x) / 2.0
            
        except (IndexError, TypeError) as e:
            logger.error(f"Error accessing hip landmark coordinates: {e}")
            return None
    else:
        # Placeholder for more sophisticated CoM calculation
        # Would incorporate upper body, arms, head contributions
        logger.warning("Full center of mass calculation not implemented. Using hip center fallback.")
        return _calculate_center_of_mass(landmarks, True)


def _get_stance_foot_position(landmarks: LandmarksDict, stance_foot: str) -> Optional[float]:
    """
    Get the x-coordinate of the stance foot ankle.
    
    Parameters:
    -----------
    landmarks : LandmarksDict
        Dictionary containing pose landmarks
    stance_foot : str
        Which foot is in stance ('left' or 'right')
        
    Returns:
    --------
    Optional[float]
        Foot ankle x-coordinate, or None if not found
    """
    if stance_foot not in ['left', 'right']:
        logger.warning(f"Invalid stance_foot value: '{stance_foot}'. Must be 'left' or 'right'.")
        return None
    
    ankle_landmark_name = f"{stance_foot}_ankle"
    
    if ankle_landmark_name not in landmarks:
        logger.warning(f"Stance foot landmark '{ankle_landmark_name}' not found.")
        return None
    
    try:
        return landmarks[ankle_landmark_name][0]
    except (IndexError, TypeError) as e:
        logger.error(f"Error accessing {ankle_landmark_name} coordinates: {e}")
        return None


def _categorize_landing_position(distance_cm: float, tolerance_cm: float) -> str:
    """
    Categorize foot landing position relative to center of mass.
    
    Parameters:
    -----------
    distance_cm : float
        Horizontal distance from foot to CoM in centimeters
    tolerance_cm : float
        Tolerance for "under" classification
        
    Returns:
    --------
    str
        Position category: 'under', 'ahead', or 'behind'
    """
    if abs(distance_cm) <= tolerance_cm:
        return 'under'
    elif distance_cm > tolerance_cm:
        return 'ahead'  # Potential overstriding
    else:
        return 'behind'  # Uncommon but possible


if __name__ == "__main__":
    print("Testing calculate_foot_landing_position module...")
    
    # Mock stance phase data for testing
    stance_left = {"is_stance_phase": True, "stance_foot": "left"}
    stance_right = {"is_stance_phase": True, "stance_foot": "right"}
    stance_inactive = {"is_stance_phase": False, "stance_foot": "left"}
    
    # Example 1: Foot landing under center of mass (optimal)
    sample_landmarks_under: LandmarksDict = {
        'left_hip': (0.45, 0.5, 0, 0.95),
        'right_hip': (0.55, 0.5, 0, 0.95),     # Hip center at x=0.5
        'left_ankle': (0.48, 0.8, 0, 0.95),    # Left foot close to center
        'right_ankle': (0.62, 0.8, 0, 0.95)
    }
    result_under = calculate_foot_landing_position(sample_landmarks_under, stance_left)
    print(f"\nFoot Under CoM Test:\n{result_under}")
    
    # Example 2: Foot landing ahead of center of mass (overstriding)
    sample_landmarks_ahead: LandmarksDict = {
        'left_hip': (0.45, 0.5, 0, 0.95),
        'right_hip': (0.55, 0.5, 0, 0.95),     # Hip center at x=0.5
        'left_ankle': (0.35, 0.8, 0, 0.95),    # Left foot well ahead
        'right_ankle': (0.62, 0.8, 0, 0.95)
    }
    result_ahead = calculate_foot_landing_position(sample_landmarks_ahead, stance_left)
    print(f"\nFoot Ahead of CoM Test (Overstriding):\n{result_ahead}")
    
    # Example 3: Foot landing behind center of mass
    sample_landmarks_behind: LandmarksDict = {
        'left_hip': (0.45, 0.5, 0, 0.95),
        'right_hip': (0.55, 0.5, 0, 0.95),     # Hip center at x=0.5
        'left_ankle': (0.58, 0.8, 0, 0.95),    # Left foot behind center
        'right_ankle': (0.62, 0.8, 0, 0.95)
    }
    result_behind = calculate_foot_landing_position(sample_landmarks_behind, stance_left)
    print(f"\nFoot Behind CoM Test:\n{result_behind}")
    
    # Example 4: Not in stance phase
    result_no_stance = calculate_foot_landing_position(sample_landmarks_under, stance_inactive)
    print(f"\nNo Stance Phase Test:\n{result_no_stance}")
    
    # Example 5: Missing landmark
    sample_landmarks_missing: LandmarksDict = {
        'left_hip': (0.45, 0.5, 0, 0.95),
        # Missing 'right_hip'
        'left_ankle': (0.48, 0.8, 0, 0.95),
        'right_ankle': (0.62, 0.8, 0, 0.95)
    }
    result_missing = calculate_foot_landing_position(sample_landmarks_missing, stance_left)
    print(f"\nMissing Landmark Test:\n{result_missing}")
    
    # Example 6: Custom tolerance testing
    result_strict = calculate_foot_landing_position(
        sample_landmarks_under, 
        stance_left,
        tolerance_cm=2.0  # Stricter tolerance
    )
    print(f"\nStrict Tolerance Test:\n{result_strict}")
    
    # Example 7: Right foot stance
    result_right_foot = calculate_foot_landing_position(sample_landmarks_under, stance_right)
    print(f"\nRight Foot Stance Test:\n{result_right_foot}")