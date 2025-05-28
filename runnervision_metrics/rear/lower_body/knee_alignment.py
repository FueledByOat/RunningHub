# knee_alignment.py
"""
Calculates knee alignment patterns during running to detect valgus (knock-knee) 
or varus (bow-leg) deviations that may indicate injury risk factors.
"""
import logging
from typing import Dict, Tuple, Optional, Any

logger = logging.getLogger(__name__)

# Define type aliases
Landmark = Tuple[float, float, float, float]  # (x, y, z, visibility)
LandmarksDict = Dict[str, Landmark]
KneeAlignmentResult = Dict[str, Optional[Any]]

def calculate_knee_alignment(
    landmarks: LandmarksDict,
    threshold: float = 0.1
) -> KneeAlignmentResult:
    """
    Assess knee alignment during running to detect valgus (knock-knee) or varus (bow-leg) patterns.
    
    Dynamic knee valgus is particularly concerning in runners as it indicates:
    - Potential weakness in hip abductors/external rotators
    - Excessive foot pronation
    - Risk factor for patellofemoral pain syndrome, ACL injuries, and IT band syndrome
    
    Parameters:
    -----------
    landmarks : LandmarksDict
        A dictionary containing the 3D coordinates and visibility of detected pose landmarks.
        Expected keys: 'left_hip', 'left_knee', 'left_ankle', 'right_hip', 'right_knee', 'right_ankle'.
        Each landmark is a tuple: (x, y, z, visibility).
        
    threshold : float, default=0.1
        Threshold as proportion of hip width for classification (0.1 = 10% of hip width).
        Values above this threshold indicate concerning alignment deviation.
        
    Returns:
    --------
    KneeAlignmentResult
        A dictionary containing:
        - "left_knee_valgus" (Optional[bool]): True if left knee shows valgus pattern. None if calculation fails.
        - "left_knee_varus" (Optional[bool]): True if left knee shows varus pattern. None if calculation fails.
        - "right_knee_valgus" (Optional[bool]): True if right knee shows valgus pattern. None if calculation fails.
        - "right_knee_varus" (Optional[bool]): True if right knee shows varus pattern. None if calculation fails.
        - "left_normalized_deviation" (Optional[float]): Left knee deviation normalized by hip width. None if calculation fails.
        - "right_normalized_deviation" (Optional[float]): Right knee deviation normalized by hip width. None if calculation fails.
        - "severity_left" (Optional[str]): Clinical severity for left knee ("normal", "mild", "moderate", "severe"). None if calculation fails.
        - "severity_right" (Optional[str]): Clinical severity for right knee ("normal", "mild", "moderate", "severe"). None if calculation fails.
        - "calculation_successful" (bool): True if metrics were calculated, False if essential landmarks were missing.
        
    Notes:
    ------
    Alignment assessment (from posterior view):
    - Valgus (knock-knee): Knee deviates toward midline
    - Varus (bow-leg): Knee deviates away from midline
    
    Severity thresholds (as proportion of hip width):
    - Normal: < 10% deviation
    - Mild: 10-15% deviation
    - Moderate: 15-20% deviation  
    - Severe: > 20% deviation
    
    Best Practice:
    - Analyze during single-leg stance phases for most accurate assessment
    - Consider multiple gait cycles for reliable patterns
    - Account for natural anatomical variations
    """
    
    required_landmarks = [
        'left_hip', 'left_knee', 'left_ankle',
        'right_hip', 'right_knee', 'right_ankle'
    ]
    
    # Initialize default return values
    result: KneeAlignmentResult = {
        "left_knee_valgus": None,
        "left_knee_varus": None,
        "right_knee_valgus": None,
        "right_knee_varus": None,
        "left_normalized_deviation": None,
        "right_normalized_deviation": None,
        "severity_left": None,
        "severity_right": None,
        "calculation_successful": False
    }
    
    # Check for required landmarks
    for lm_name in required_landmarks:
        if lm_name not in landmarks:
            logger.warning(f"Required landmark '{lm_name}' not found for knee alignment calculation.")
            return result
    
    # Validate visibility
    low_vis_landmarks = [lm for lm in required_landmarks if landmarks[lm][3] < 0.5]
    if low_vis_landmarks:
        logger.warning(f"Low visibility landmarks detected: {low_vis_landmarks}")
    
    try:
        # Extract coordinates
        left_hip_x = landmarks['left_hip'][0]
        left_knee_x = landmarks['left_knee'][0]
        left_ankle_x = landmarks['left_ankle'][0]
        
        right_hip_x = landmarks['right_hip'][0]
        right_knee_x = landmarks['right_knee'][0]
        right_ankle_x = landmarks['right_ankle'][0]
        
        # Calculate hip width for normalization
        hip_width = abs(right_hip_x - left_hip_x)
        if hip_width < 0.01:
            logger.warning("Hip width too small for reliable knee alignment calculation.")
            return result
        
        # Left leg alignment analysis
        left_hip_to_ankle_x = left_ankle_x - left_hip_x
        if abs(left_hip_to_ankle_x) > 0.001:  # Avoid near-zero divisions
            left_expected_knee_x = left_hip_x + left_hip_to_ankle_x * 0.5
            left_deviation = left_knee_x - left_expected_knee_x
            left_normalized_deviation = left_deviation / hip_width
        else:
            left_normalized_deviation = 0
        
        # Right leg alignment analysis  
        right_hip_to_ankle_x = right_ankle_x - right_hip_x
        if abs(right_hip_to_ankle_x) > 0.001:
            right_expected_knee_x = right_hip_x + right_hip_to_ankle_x * 0.5
            right_deviation = right_knee_x - right_expected_knee_x
            right_normalized_deviation = right_deviation / hip_width
        else:
            right_normalized_deviation = 0
        
        # Determine valgus/varus patterns (from posterior view)
        left_valgus = left_normalized_deviation < -threshold
        left_varus = left_normalized_deviation > threshold
        right_valgus = right_normalized_deviation > threshold
        right_varus = right_normalized_deviation < -threshold
        
        # Determine severity
        def get_severity(deviation: float) -> str:
            abs_dev = abs(deviation)
            if abs_dev < threshold:
                return "normal"
            elif abs_dev < threshold * 1.5:
                return "mild"
            elif abs_dev < threshold * 2:
                return "moderate"
            else:
                return "severe"
        
        severity_left = get_severity(left_normalized_deviation)
        severity_right = get_severity(right_normalized_deviation)
        
        # Update result with successful calculations
        result.update({
            "left_knee_valgus": left_valgus,
            "left_knee_varus": left_varus,
            "right_knee_valgus": right_valgus,
            "right_knee_varus": right_varus,
            "left_normalized_deviation": round(left_normalized_deviation, 3),
            "right_normalized_deviation": round(right_normalized_deviation, 3),
            "severity_left": severity_left,
            "severity_right": severity_right,
            "calculation_successful": True
        })
        
        logger.debug(f"Knee alignment calculated - Left: {severity_left}, Right: {severity_right}")
        
    except (KeyError, IndexError, TypeError, ZeroDivisionError) as e:
        logger.error(f"Error calculating knee alignment: {e}")
        return result
    
    return result


def analyze_knee_alignment_sequence(
    landmark_sequence: list[LandmarksDict],
    threshold: float = 0.1,
    stance_phases: Optional[list[bool]] = None
) -> Dict[str, Any]:
    """
    Analyze knee alignment across multiple frames with optional stance phase filtering.
    
    Parameters:
    -----------
    landmark_sequence : list[LandmarksDict]
        List of landmark dictionaries for each frame
    threshold : float, default=0.1
        Alignment deviation threshold
    stance_phases : Optional[list[bool]], default=None
        Boolean list for stance phase filtering
        
    Returns:
    --------
    Dict[str, Any]
        Summary statistics and frame-by-frame results
    """
    
    if stance_phases and len(stance_phases) != len(landmark_sequence):
        logger.warning("Stance phases list length mismatch")
        stance_phases = None
    
    frame_results = []
    left_deviations = []
    right_deviations = []
    
    for i, landmarks in enumerate(landmark_sequence):
        if stance_phases and not stance_phases[i]:
            continue
            
        result = calculate_knee_alignment(landmarks, threshold)
        frame_results.append(result)
        
        if result["calculation_successful"]:
            if result["left_normalized_deviation"] is not None:
                left_deviations.append(result["left_normalized_deviation"])
            if result["right_normalized_deviation"] is not None:
                right_deviations.append(result["right_normalized_deviation"])
    
    # Calculate summary statistics
    summary = {
        "total_frames_analyzed": len(frame_results),
        "valid_calculations": len([r for r in frame_results if r["calculation_successful"]]),
        "left_mean_deviation": round(sum(left_deviations) / len(left_deviations), 3) if left_deviations else None,
        "right_mean_deviation": round(sum(right_deviations) / len(right_deviations), 3) if right_deviations else None,
        "left_max_deviation": round(max(left_deviations, key=abs), 3) if left_deviations else None,
        "right_max_deviation": round(max(right_deviations, key=abs), 3) if right_deviations else None,
        "frame_results": frame_results
    }
    
    logger.info(f"Knee alignment sequence analysis: {summary['valid_calculations']}/{len(landmark_sequence)} valid frames")
    
    return summary


if __name__ == "__main__":
    print("Testing calculate_knee_alignment module...")
    
    # Example 1: Normal alignment
    sample_landmarks_normal: LandmarksDict = {
        'left_hip': (0.4, 0.4, 0, 0.99),
        'left_knee': (0.4, 0.6, 0, 0.99),  # Aligned with hip-ankle line
        'left_ankle': (0.4, 0.8, 0, 0.99),
        'right_hip': (0.6, 0.4, 0, 0.99),
        'right_knee': (0.6, 0.6, 0, 0.99),  # Aligned with hip-ankle line
        'right_ankle': (0.6, 0.8, 0, 0.99)
    }
    results_normal = calculate_knee_alignment(sample_landmarks_normal)
    print(f"\nResults for Normal Alignment:\n{results_normal}")
    
    # Example 2: Left knee valgus (knock-knee)
    sample_landmarks_valgus: LandmarksDict = {
        'left_hip': (0.4, 0.4, 0, 0.99),
        'left_knee': (0.45, 0.6, 0, 0.99),  # Knee deviated medially
        'left_ankle': (0.4, 0.8, 0, 0.99),
        'right_hip': (0.6, 0.4, 0, 0.99),
        'right_knee': (0.55, 0.6, 0, 0.99),  # Right knee also valgus
        'right_ankle': (0.6, 0.8, 0, 0.99)
    }
    results_valgus = calculate_knee_alignment(sample_landmarks_valgus)
    print(f"\nResults for Valgus Pattern:\n{results_valgus}")
    
    # Example 3: Severe varus (bow-leg)
    sample_landmarks_varus: LandmarksDict = {
        'left_hip': (0.4, 0.4, 0, 0.99),
        'left_knee': (0.35, 0.6, 0, 0.99),  # Knee deviated laterally
        'left_ankle': (0.4, 0.8, 0, 0.99),
        'right_hip': (0.6, 0.4, 0, 0.99),
        'right_knee': (0.65, 0.6, 0, 0.99),  # Right knee also varus
        'right_ankle': (0.6, 0.8, 0, 0.99)
    }
    results_varus = calculate_knee_alignment(sample_landmarks_varus)
    print(f"\nResults for Varus Pattern:\n{results_varus}")
    
    # Example 4: Missing landmarks
    sample_landmarks_missing: LandmarksDict = {
        'left_hip': (0.4, 0.4, 0, 0.99),
        'right_hip': (0.6, 0.4, 0, 0.99)
        # Missing knee and ankle landmarks
    }
    results_missing = calculate_knee_alignment(sample_landmarks_missing)
    print(f"\nResults for Missing Landmarks:\n{results_missing}")
    
    # Example 5: Low visibility case
    sample_landmarks_low_vis: LandmarksDict = {
        'left_hip': (0.4, 0.4, 0, 0.3),      # Low visibility
        'left_knee': (0.42, 0.6, 0, 0.2),    # Very low visibility
        'left_ankle': (0.4, 0.8, 0, 0.4),
        'right_hip': (0.6, 0.4, 0, 0.3),
        'right_knee': (0.58, 0.6, 0, 0.2),
        'right_ankle': (0.6, 0.8, 0, 0.4)
    }
    results_low_vis = calculate_knee_alignment(sample_landmarks_low_vis)
    print(f"\nResults for Low Visibility:\n{results_low_vis}")