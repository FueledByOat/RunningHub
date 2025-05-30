# pelvic_tilt.py
"""
Calculates lateral pelvic tilt angle in the frontal plane during running analysis.
This metric identifies pelvic orientation deviations that can indicate hip abductor
weakness, leg length discrepancies, or compensation patterns.
"""
import logging
import numpy as np
from typing import Dict, Tuple, Optional, Any

logger = logging.getLogger(__name__)

# Define type aliases for consistency
Landmark = Tuple[float, float, float, float]  # (x, y, z, visibility)
LandmarksDict = Dict[str, Landmark]
PelvicTiltResult = Dict[str, Optional[Any]]

def calculate_pelvic_tilt(
    landmarks: LandmarksDict,
    coordinate_system: str = "vision_standard"
) -> PelvicTiltResult:
    """
    Calculate lateral pelvic tilt angle in the frontal plane during running.
    
    Measures lateral pelvic tilt (frontal plane) which can indicate:
    - Hip abductor weakness (primarily gluteus medius)
    - Leg length discrepancy (functional or anatomical)
    - Compensation patterns for other biomechanical issues
    - Potential IT band, low back, or knee injury risk
    
    Parameters:
    -----------
    landmarks : LandmarksDict
        A dictionary containing the 3D coordinates and visibility of detected pose landmarks.
        Expected keys for this function: 'left_hip', 'right_hip'.
        Each landmark is a tuple: (x, y, z, visibility). Coordinates are typically 
        normalized (0.0-1.0) relative to image dimensions.
        
    coordinate_system : str, default="vision_standard"
        Coordinate system convention:
        - "vision_standard": Y increases downward (typical for computer vision)
        - "clinical_standard": Y increases upward (typical for clinical analysis)
        
    Returns:
    --------
    PelvicTiltResult
        A dictionary containing:
        - "tilt_angle_degrees" (Optional[float]): Lateral pelvic tilt angle in degrees.
            Positive values indicate right side elevated.
            Negative values indicate left side elevated.
            None if calculation fails.
        - "elevated_side" (Optional[str]): Side that is elevated ("left", "right", "neutral").
            None if calculation fails.
        - "severity" (Optional[str]): Clinical severity classification ("normal", "mild", "moderate", "severe").
            None if calculation fails.
        - "normalized_tilt" (Optional[float]): Tilt normalized by hip distance for relative assessment.
            None if calculation fails.
        - "calculation_successful" (bool): True if metrics were calculated, False if essential landmarks were missing.
        
    Notes:
    ------
    Clinical severity thresholds:
    - Normal range: ±2° during stance phase
    - Mild tilt: 2-5° (potential early intervention)
    - Moderate: 5-10° (intervention recommended)  
    - Severe: >10° (significant dysfunction)
    
    This measures frontal plane motion only and differs from anterior/posterior pelvic tilt
    (sagittal plane), which requires side-view analysis.
    
    Best Practice:
    - Apply during single-leg stance phases for most accurate assessment
    - Consider multiple cycles for reliable clinical interpretation
    - Account for camera positioning and potential parallax effects
    """
    
    required_landmarks = ['left_hip', 'right_hip']
    
    # Initialize default return values for failure case
    result: PelvicTiltResult = {
        "tilt_angle_degrees": None,
        "elevated_side": None,
        "severity": None,
        "normalized_tilt": None,
        "calculation_successful": False
    }
    
    # Check for presence of all required landmarks
    for lm_name in required_landmarks:
        if lm_name not in landmarks:
            logger.warning(f"Required landmark '{lm_name}' not found for pelvic tilt calculation.")
            return result
    
    # Validate landmark visibility
    left_hip = landmarks['left_hip']
    right_hip = landmarks['right_hip']
    
    if left_hip[3] < 0.5 or right_hip[3] < 0.5:
        logger.warning("Hip landmarks have low visibility scores, pelvic tilt calculation may be unreliable.")
    
    try:
        # Extract hip coordinates
        left_hip_x, left_hip_y = left_hip[0], left_hip[1]
        right_hip_x, right_hip_y = right_hip[0], right_hip[1]
        
        # Calculate horizontal distance between hips for normalization
        hip_distance = abs(right_hip_x - left_hip_x)
        
        if hip_distance < 0.01:  # Prevent division by very small numbers
            logger.warning("Hip distance too small for reliable tilt calculation.")
            return result
        
        # Calculate tilt angle using arctangent
        # Positive = right side elevated, negative = left side elevated
        tilt_angle = np.degrees(np.arctan2(right_hip_y - left_hip_y, right_hip_x - left_hip_x))
        
        # Apply coordinate system correction
        if coordinate_system == "clinical_standard":
            tilt_angle = -tilt_angle
        
        # Determine severity based on clinical thresholds
        abs_tilt = abs(tilt_angle)
        if abs_tilt <= 2:
            severity = "normal"
        elif abs_tilt <= 5:
            severity = "mild"
        elif abs_tilt <= 10:
            severity = "moderate"
        else:
            severity = "severe"
        
        # Determine elevated side
        if abs_tilt <= 2:
            elevated_side = "neutral"
        else:
            elevated_side = "right" if tilt_angle > 0 else "left"
        
        # Calculate normalized tilt for relative assessment
        reference_angle = np.degrees(np.arctan2(0.1, hip_distance))  # 10% of hip distance as reference
        normalized_tilt = tilt_angle / reference_angle if reference_angle != 0 else 0
        
        # Update result with successful calculations
        result.update({
            "tilt_angle_degrees": round(tilt_angle, 2),
            "elevated_side": elevated_side,
            "severity": severity,
            "normalized_tilt": round(normalized_tilt, 3),
            "calculation_successful": True
        })
        
        logger.debug(f"Pelvic tilt calculated: {tilt_angle:.2f}° ({elevated_side}, {severity})")
        
    except (KeyError, IndexError, TypeError, ZeroDivisionError) as e:
        logger.error(f"Error calculating pelvic tilt: {e}")
        return result
    
    return result


def analyze_pelvic_tilt_sequence(
    landmark_sequence: list[LandmarksDict],
    coordinate_system: str = "vision_standard",
    stance_phases: Optional[list[bool]] = None
) -> Dict[str, Any]:
    """
    Analyze pelvic tilt across a sequence of frames, optionally filtering for stance phases.
    
    Parameters:
    -----------
    landmark_sequence : list[LandmarksDict]
        List of landmark dictionaries for each frame
    coordinate_system : str, default="vision_standard"
        Coordinate system convention for tilt calculation
    stance_phases : Optional[list[bool]], default=None
        Boolean list indicating stance phases for filtering
        
    Returns:
    --------
    Dict[str, Any]
        Summary statistics and frame-by-frame results
    """
    
    if stance_phases and len(stance_phases) != len(landmark_sequence):
        logger.warning("Stance phases list length doesn't match landmark sequence length")
        stance_phases = None
    
    frame_results = []
    valid_tilts = []
    
    for i, landmarks in enumerate(landmark_sequence):
        if stance_phases and not stance_phases[i]:
            continue
            
        result = calculate_pelvic_tilt(landmarks, coordinate_system)
        frame_results.append(result)
        
        if result["calculation_successful"] and result["tilt_angle_degrees"] is not None:
            valid_tilts.append(result["tilt_angle_degrees"])
    
    # Calculate summary statistics
    summary = {
        "total_frames_analyzed": len(frame_results),
        "valid_calculations": len(valid_tilts),
        "mean_tilt_angle": round(sum(valid_tilts) / len(valid_tilts), 2) if valid_tilts else None,
        "max_tilt_angle": round(max(valid_tilts, key=abs), 2) if valid_tilts else None,
        "tilt_range": round(max(valid_tilts) - min(valid_tilts), 2) if valid_tilts else None,
        "frame_results": frame_results
    }
    
    logger.info(f"Pelvic tilt sequence analysis: {len(valid_tilts)}/{len(landmark_sequence)} valid frames")
    
    return summary


if __name__ == "__main__":
    print("Testing calculate_pelvic_tilt module...")
    
    # Example 1: Normal/neutral pelvic alignment
    sample_landmarks_neutral: LandmarksDict = {
        'left_hip': (0.4, 0.5, 0, 0.99),
        'right_hip': (0.6, 0.5, 0, 0.99)  # Level hips = 0° tilt
    }
    results_neutral = calculate_pelvic_tilt(sample_landmarks_neutral)
    print(f"\nResults for Neutral Case:\n{results_neutral}")
    
    # Example 2: Mild right side elevation
    sample_landmarks_mild_right: LandmarksDict = {
        'left_hip': (0.4, 0.52, 0, 0.99),  # Left hip slightly lower
        'right_hip': (0.6, 0.50, 0, 0.99)  # Creates ~5.7° right elevation
    }
    results_mild_right = calculate_pelvic_tilt(sample_landmarks_mild_right)
    print(f"\nResults for Mild Right Elevation:\n{results_mild_right}")
    
    # Example 3: Severe left side elevation
    sample_landmarks_severe_left: LandmarksDict = {
        'left_hip': (0.4, 0.45, 0, 0.99),  # Left hip much higher
        'right_hip': (0.6, 0.50, 0, 0.99)  # Creates ~14° left elevation
    }
    results_severe_left = calculate_pelvic_tilt(sample_landmarks_severe_left)
    print(f"\nResults for Severe Left Elevation:\n{results_severe_left}")
    
    # Example 4: Missing landmark (error case)
    sample_landmarks_missing: LandmarksDict = {
        'left_hip': (0.4, 0.5, 0, 0.99)  # Missing right_hip
    }
    results_missing = calculate_pelvic_tilt(sample_landmarks_missing)
    print(f"\nResults for Missing Landmark:\n{results_missing}")
    
    # Example 5: Clinical coordinate system
    sample_landmarks_clinical: LandmarksDict = {
        'left_hip': (0.4, 0.48, 0, 0.99),
        'right_hip': (0.6, 0.52, 0, 0.99)  # Test coordinate system correction
    }
    results_clinical = calculate_pelvic_tilt(sample_landmarks_clinical, coordinate_system="clinical_standard")
    print(f"\nResults for Clinical Coordinate System:\n{results_clinical}")