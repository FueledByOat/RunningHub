# hip_drop.py
"""
Calculates hip drop (Trendelenburg gait) during running analysis.
This metric identifies lateral pelvic tilt during single-leg support phases,
which can indicate hip abductor weakness and potential injury risk.
"""
import logging
from typing import Dict, Tuple, Optional, Any

logger = logging.getLogger(__name__)

# Define type aliases for consistency with other modules
Landmark = Tuple[float, float, float, float]  # (x, y, z, visibility)
LandmarksDict = Dict[str, Landmark]
HipDropResult = Dict[str, Optional[Any]]  # Using Any for mixed value types

def calculate_hip_drop(
    landmarks: LandmarksDict,
    threshold: float = 0.015
) -> HipDropResult:
    """
    Detect hip drop (Trendelenburg gait) during running stance phase.
    
    Hip drop occurs when the pelvis tilts laterally during single-leg support,
    indicating potential weakness in hip abductor muscles (primarily gluteus medius).
    This analysis is most accurate when applied to frames during single-leg stance
    phases rather than flight phases.
    
    Parameters:
    -----------
    landmarks : LandmarksDict
        A dictionary containing the 3D coordinates and visibility of detected pose landmarks.
        Expected keys for this function: 'left_hip', 'right_hip'.
        Each landmark is a tuple: (x, y, z, visibility). Coordinates are typically 
        normalized (0.0-1.0) relative to image dimensions.
        
    threshold : float, default=0.015
        The minimum difference in normalized hip height to classify as hip drop.
        Represents approximately 1-2% of image height. Should be adjusted based on:
        - Camera angle and distance from subject
        - Image resolution and quality
        - Clinical sensitivity requirements
        
    Returns:
    --------
    HipDropResult
        A dictionary containing:
        - "hip_drop_value" (Optional[float]): Raw hip height difference in normalized coordinates.
            Positive values indicate right hip is lower (dropped).
            Negative values indicate left hip is lower (dropped).
            None if calculation fails.
        - "hip_drop_direction" (Optional[str]): Direction of hip drop ("left", "right", "neutral").
            None if calculation fails.
        - "severity" (Optional[str]): Clinical severity classification ("none", "mild", "moderate", "severe").
            None if calculation fails.
        - "calculation_successful" (bool): True if metrics were calculated, False if essential landmarks were missing.
        
    Notes:
    ------
    Clinical severity thresholds (approximate conversions from degrees to normalized coordinates):
    - None/Neutral: < 1.5% image height (< ~2°)
    - Mild: 1.5-3% image height (~2-5°)
    - Moderate: 3-5% image height (~5-8°)
    - Severe: > 5% image height (> ~8°)
    
    These thresholds assume typical camera positioning and may need adjustment
    for different recording setups or clinical protocols.
    
    Best Practice:
    - Apply this analysis only during identified single-leg stance phases
    - Consider multiple frames/cycles for reliable assessment
    - Account for natural body asymmetries in interpretation
    """
    
    required_landmarks = ['left_hip', 'right_hip']
    
    # Initialize default return values for failure case
    result: HipDropResult = {
        "hip_drop_value": None,
        "hip_drop_direction": None,
        "severity": None,
        "calculation_successful": False
    }
    
    # Check for presence of all required landmarks
    for lm_name in required_landmarks:
        if lm_name not in landmarks:
            logger.warning(f"Required landmark '{lm_name}' not found for hip drop calculation.")
            return result
    
    # Validate landmark visibility (assuming visibility > 0.5 indicates reliable detection)
    left_hip = landmarks['left_hip']
    right_hip = landmarks['right_hip']
    
    if left_hip[3] < 0.5 or right_hip[3] < 0.5:
        logger.warning("Hip landmarks have low visibility scores, hip drop calculation may be unreliable.")
        # Continue calculation but log the warning
    
    try:
        # Extract Y coordinates (vertical position)
        left_hip_y = left_hip[1]
        right_hip_y = right_hip[1]
        
        # Calculate hip height difference 
        # Positive value = right hip is lower (dropped)
        # Negative value = left hip is lower (dropped)
        hip_drop = right_hip_y - left_hip_y
        
        # Determine direction and severity based on clinical thresholds
        if abs(hip_drop) < threshold:
            direction = "neutral"
            severity = "none"
        else:
            direction = "right" if hip_drop > 0 else "left"
            
            # Clinical severity classification
            abs_drop = abs(hip_drop)
            if abs_drop < 0.03:  # ~2-5 degrees
                severity = "mild"
            elif abs_drop < 0.05:  # ~5-8 degrees
                severity = "moderate"
            else:  # > ~8 degrees
                severity = "severe"
        
        # Update result with successful calculations
        result.update({
            "hip_drop_value": hip_drop,
            "hip_drop_direction": direction,
            "severity": severity,
            "calculation_successful": True
        })
        
        logger.debug(f"Hip drop calculated: {hip_drop:.4f} ({direction}, {severity})")
        
    except (KeyError, IndexError, TypeError) as e:
        logger.error(f"Error calculating hip drop: {e}")
        return result
    
    return result


def analyze_hip_drop_sequence(
    landmark_sequence: list[LandmarksDict],
    threshold: float = 0.015,
    stance_phases: Optional[list[bool]] = None
) -> Dict[str, Any]:
    """
    Analyze hip drop across a sequence of frames, optionally filtering for stance phases.
    
    Parameters:
    -----------
    landmark_sequence : list[LandmarksDict]
        List of landmark dictionaries for each frame
    threshold : float, default=0.015
        Hip drop detection threshold
    stance_phases : Optional[list[bool]], default=None
        Boolean list indicating stance phases. If provided, only stance phase
        frames will be analyzed. Should match length of landmark_sequence.
        
    Returns:
    --------
    Dict[str, Any]
        Summary statistics including mean, max, and frame-by-frame results
    """
    
    if stance_phases and len(stance_phases) != len(landmark_sequence):
        logger.warning("Stance phases list length doesn't match landmark sequence length")
        stance_phases = None
    
    frame_results = []
    valid_drops = []
    
    for i, landmarks in enumerate(landmark_sequence):
        # Skip non-stance phases if specified
        if stance_phases and not stance_phases[i]:
            continue
            
        result = calculate_hip_drop(landmarks, threshold)
        frame_results.append(result)
        
        if result["calculation_successful"] and result["hip_drop_value"] is not None:
            valid_drops.append(result["hip_drop_value"])
    
    # Calculate summary statistics
    summary = {
        "total_frames_analyzed": len(frame_results),
        "valid_calculations": len(valid_drops),
        "mean_hip_drop": sum(valid_drops) / len(valid_drops) if valid_drops else None,
        "max_hip_drop": max(valid_drops, key=abs) if valid_drops else None,
        "frame_results": frame_results
    }
    
    logger.info(f"Hip drop sequence analysis: {len(valid_drops)}/{len(landmark_sequence)} valid frames")
    
    return summary


if __name__ == "__main__":
    print("Testing calculate_hip_drop module...")
    
    # Example 1: Normal/neutral hip alignment
    sample_landmarks_neutral: LandmarksDict = {
        'left_hip': (0.4, 0.5, 0, 0.99),
        'right_hip': (0.6, 0.5, 0, 0.99)  # Same Y coordinates = no hip drop
    }
    results_neutral = calculate_hip_drop(sample_landmarks_neutral)
    print(f"\nResults for Neutral Case:\n{results_neutral}")
    
    # Example 2: Mild right hip drop
    sample_landmarks_mild_right: LandmarksDict = {
        'left_hip': (0.4, 0.48, 0, 0.99),
        'right_hip': (0.6, 0.50, 0, 0.99)  # Right hip 0.02 lower = mild drop
    }
    results_mild_right = calculate_hip_drop(sample_landmarks_mild_right)
    print(f"\nResults for Mild Right Hip Drop:\n{results_mild_right}")
    
    # Example 3: Severe left hip drop
    sample_landmarks_severe_left: LandmarksDict = {
        'left_hip': (0.4, 0.56, 0, 0.99),  # Left hip 0.06 lower = severe drop
        'right_hip': (0.6, 0.50, 0, 0.99)
    }
    results_severe_left = calculate_hip_drop(sample_landmarks_severe_left)
    print(f"\nResults for Severe Left Hip Drop:\n{results_severe_left}")
    
    # Example 4: Missing landmark (error case)
    sample_landmarks_missing: LandmarksDict = {
        'right_hip': (0.6, 0.5, 0, 0.99)  # Missing left_hip
    }
    results_missing = calculate_hip_drop(sample_landmarks_missing)
    print(f"\nResults for Missing Landmark:\n{results_missing}")
    
    # Example 5: Low visibility landmarks
    sample_landmarks_low_vis: LandmarksDict = {
        'left_hip': (0.4, 0.48, 0, 0.3),   # Low visibility
        'right_hip': (0.6, 0.52, 0, 0.4)  # Low visibility, moderate drop
    }
    results_low_vis = calculate_hip_drop(sample_landmarks_low_vis)
    print(f"\nResults for Low Visibility Case:\n{results_low_vis}")