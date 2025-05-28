# ankle_inversion.py

"""
Calculates ankle inversion/eversion patterns during running gait analysis.

This metric helps identify biomechanical inefficiencies related to foot strike
patterns that may lead to injury. Excessive inversion is linked to lateral ankle
sprains and insufficient shock absorption, while excessive eversion is associated
with medial tibial stress syndrome (shin splints) and plantar fasciitis.
"""

import logging
import numpy as np
from typing import Dict, Tuple, Optional, Any

logger = logging.getLogger(__name__)

# Define type aliases for consistency with foot_crossover.py
Landmark = Tuple[float, float, float, float]
LandmarksDict = Dict[str, Landmark]
AnkleInversionResult = Dict[str, Optional[Any]]

def calculate_ankle_inversion(
    landmarks: LandmarksDict,
    inversion_threshold: float = 0.03
) -> AnkleInversionResult:
    """
    Measures ankle inversion/eversion patterns during running from rear view analysis.
    
    Inversion occurs when the ankle rolls outward (supination), while eversion 
    occurs when the ankle rolls inward (pronation). Measurements are normalized
    by hip width for consistent comparison across different body types.
    
    From rear view perspective:
    - Left foot: heel left of ankle = inversion, heel right of ankle = eversion
    - Right foot: heel right of ankle = inversion, heel left of ankle = eversion
    
    Parameters:
    -----------
    landmarks : LandmarksDict
        Dictionary containing 3D coordinates and visibility of detected pose landmarks.
        Expected keys: 'left_hip', 'right_hip', 'left_ankle', 'right_ankle',
                      'left_heel', 'right_heel'.
        Optional keys: 'left_foot_index', 'right_foot_index' for enhanced analysis.
        Each landmark is a tuple: (x, y, z, visibility).
    
    inversion_threshold : float, default=0.03
        Threshold as proportion of hip width for classifying inversion/eversion.
        Values above this threshold indicate abnormal patterns.
        Clinical standard is approximately 3% of hip width.
    
    Returns:
    --------
    AnkleInversionResult
        Dictionary containing:
        - "left_inversion_value" (Optional[float]): Raw inversion measurement for left foot.
            Positive = inversion, negative = eversion. None if calculation fails.
        - "right_inversion_value" (Optional[float]): Raw inversion measurement for right foot.
            Positive = inversion, negative = eversion. None if calculation fails.
        - "left_normalized" (Optional[float]): Left inversion normalized by hip width.
            None if calculation fails.
        - "right_normalized" (Optional[float]): Right inversion normalized by hip width.
            None if calculation fails.
        - "left_pattern" (Optional[str]): Classification as "inversion", "eversion", or "neutral".
            None if calculation fails.
        - "right_pattern" (Optional[str]): Classification as "inversion", "eversion", or "neutral".
            None if calculation fails.
        - "left_severity" (Optional[str]): Severity level: "normal", "mild", "moderate", "severe".
            None if calculation fails.
        - "right_severity" (Optional[str]): Severity level: "normal", "mild", "moderate", "severe".
            None if calculation fails.
        - "left_foot_angle" (Optional[float]): Foot axis angle in degrees if toe landmarks available.
            None if landmarks unavailable or calculation fails.
        - "right_foot_angle" (Optional[float]): Foot axis angle in degrees if toe landmarks available.
            None if landmarks unavailable or calculation fails.
        - "calculation_successful" (bool): True if core metrics calculated, False otherwise.
    """
    
    required_landmarks = [
        'left_hip', 'right_hip', 'left_ankle', 'right_ankle',
        'left_heel', 'right_heel'
    ]
    
    # Initialize default return values for failure case
    result: AnkleInversionResult = {
        "left_inversion_value": None,
        "right_inversion_value": None,
        "left_normalized": None,
        "right_normalized": None,
        "left_pattern": None,
        "right_pattern": None,
        "left_severity": None,
        "right_severity": None,
        "left_foot_angle": None,
        "right_foot_angle": None,
        "calculation_successful": False
    }
    
    # Check for presence of all required landmarks
    for lm_name in required_landmarks:
        if lm_name not in landmarks:
            logger.warning(f"Required landmark '{lm_name}' not found for ankle inversion calculation.")
            return result
    
    try:
        # Extract landmark coordinates
        left_ankle_x = landmarks['left_ankle'][0]
        left_heel_x = landmarks['left_heel'][0]
        right_ankle_x = landmarks['right_ankle'][0]
        right_heel_x = landmarks['right_heel'][0]
        
        # Calculate hip width for normalization
        left_hip_x = landmarks['left_hip'][0]
        right_hip_x = landmarks['right_hip'][0]
        hip_width = abs(right_hip_x - left_hip_x)
        
        if hip_width < 1e-5:
            logger.warning("Hip width is near zero. Ankle inversion measurement might be unreliable.",
                         extra={"landmarks": landmarks})
            # Continue with calculation but normalization will be affected
        
        # Calculate inversion values (positive = inversion, negative = eversion)
        # From rear view:
        # Left foot: heel to the left of ankle = inversion, heel to the right = eversion
        # Right foot: heel to the right of ankle = inversion, heel to the left = eversion
        left_inversion = left_ankle_x - left_heel_x
        right_inversion = right_heel_x - right_ankle_x
        
        # Normalize by hip width for better comparison across runners
        left_normalized = left_inversion / hip_width if hip_width > 1e-5 else 0
        right_normalized = right_inversion / hip_width if hip_width > 1e-5 else 0
        
        # Calculate foot axis angles if toe landmarks are available
        left_foot_angle = None
        right_foot_angle = None
        
        left_foot_index_available = 'left_foot_index' in landmarks
        right_foot_index_available = 'right_foot_index' in landmarks
        
        if left_foot_index_available:
            try:
                left_ankle_y = landmarks['left_ankle'][1]
                left_heel_y = landmarks['left_heel'][1]
                left_foot_angle = np.degrees(np.arctan2(left_ankle_y - left_heel_y,
                                                       left_ankle_x - left_heel_x))
            except Exception as e:
                logger.debug(f"Could not calculate left foot angle: {e}")
        
        if right_foot_index_available:
            try:
                right_ankle_y = landmarks['right_ankle'][1]
                right_heel_y = landmarks['right_heel'][1]
                right_foot_angle = np.degrees(np.arctan2(right_ankle_y - right_heel_y,
                                                        right_heel_x - right_ankle_x))
            except Exception as e:
                logger.debug(f"Could not calculate right foot angle: {e}")
        
        # Classify patterns based on clinical thresholds
        def get_pattern(normalized_value: float) -> str:
            """Classify inversion pattern based on normalized value."""
            if normalized_value > inversion_threshold:
                return "inversion"
            elif normalized_value < -inversion_threshold:
                return "eversion"
            else:
                return "neutral"
        
        def get_severity(normalized_value: float) -> str:
            """Determine severity level based on normalized value."""
            abs_value = abs(normalized_value)
            if abs_value < inversion_threshold:
                return "normal"
            elif abs_value < inversion_threshold * 2:
                return "mild"
            elif abs_value < inversion_threshold * 3:
                return "moderate"
            else:
                return "severe"
        
        # Apply classifications
        left_pattern = get_pattern(left_normalized)
        right_pattern = get_pattern(right_normalized)
        left_severity = get_severity(left_normalized)
        right_severity = get_severity(right_normalized)
        
        # Update result with calculated values
        result.update({
            "left_inversion_value": left_inversion,
            "right_inversion_value": right_inversion,
            "left_normalized": left_normalized,
            "right_normalized": right_normalized,
            "left_pattern": left_pattern,
            "right_pattern": right_pattern,
            "left_severity": left_severity,
            "right_severity": right_severity,
            "left_foot_angle": left_foot_angle,
            "right_foot_angle": right_foot_angle,
            "calculation_successful": True
        })
        
    except KeyError as e:
        logger.error(f"Missing landmark during ankle inversion calculation: {e}", exc_info=True)
    
    except Exception as e:
        logger.exception(f"An unexpected error occurred during ankle inversion calculation: {e}")
    
    return result


if __name__ == "__main__":
    print("Testing calculate_ankle_inversion module...")
    
    # Example 1: Normal neutral stance
    sample_landmarks_neutral: LandmarksDict = {
        'left_hip': (0.4, 0.5, 0, 0.99),
        'right_hip': (0.6, 0.5, 0, 0.99),  # hip_width = 0.2
        'left_ankle': (0.42, 0.8, 0, 0.95),
        'left_heel': (0.42, 0.85, 0, 0.95),  # aligned with ankle
        'right_ankle': (0.58, 0.8, 0, 0.95),
        'right_heel': (0.58, 0.85, 0, 0.95),  # aligned with ankle
        'left_foot_index': (0.42, 0.9, 0, 0.90),
        'right_foot_index': (0.58, 0.9, 0, 0.90)
    }
    results_neutral = calculate_ankle_inversion(sample_landmarks_neutral)
    print(f"\nResults for Neutral Stance:\n{results_neutral}")
    
    # Example 2: Left inversion, right eversion
    sample_landmarks_mixed: LandmarksDict = {
        'left_hip': (0.4, 0.5, 0, 0.99),
        'right_hip': (0.6, 0.5, 0, 0.99),  # hip_width = 0.2
        'left_ankle': (0.42, 0.8, 0, 0.95),
        'left_heel': (0.40, 0.85, 0, 0.95),  # heel left of ankle = inversion
        'right_ankle': (0.58, 0.8, 0, 0.95),
        'right_heel': (0.60, 0.85, 0, 0.95),  # heel right of ankle = eversion
    }
    results_mixed = calculate_ankle_inversion(sample_landmarks_mixed)
    print(f"\nResults for Mixed Pattern (Left Inversion, Right Eversion):\n{results_mixed}")
    
    # Example 3: Severe bilateral inversion
    sample_landmarks_severe: LandmarksDict = {
        'left_hip': (0.4, 0.5, 0, 0.99),
        'right_hip': (0.6, 0.5, 0, 0.99),  # hip_width = 0.2
        'left_ankle': (0.42, 0.8, 0, 0.95),
        'left_heel': (0.35, 0.85, 0, 0.95),  # significant inversion
        'right_ankle': (0.58, 0.8, 0, 0.95),
        'right_heel': (0.51, 0.85, 0, 0.95),  # significant inversion
    }
    results_severe = calculate_ankle_inversion(sample_landmarks_severe)
    print(f"\nResults for Severe Bilateral Inversion:\n{results_severe}")
    
    # Example 4: Missing required landmark
    sample_landmarks_missing: LandmarksDict = {
        'left_hip': (0.4, 0.5, 0, 0.99),
        'right_hip': (0.6, 0.5, 0, 0.99),
        'left_ankle': (0.42, 0.8, 0, 0.95),
        # 'left_heel' missing
        'right_ankle': (0.58, 0.8, 0, 0.95),
        'right_heel': (0.58, 0.85, 0, 0.95),
    }
    results_missing = calculate_ankle_inversion(sample_landmarks_missing)
    print(f"\nResults for Missing Landmark:\n{results_missing}")
    
    # Example 5: Zero hip width edge case
    sample_landmarks_zero_hip: LandmarksDict = {
        'left_hip': (0.5, 0.45, 0, 0.99),
        'right_hip': (0.5, 0.55, 0, 0.99),  # hip_width = 0
        'left_ankle': (0.48, 0.8, 0, 0.95),
        'left_heel': (0.46, 0.85, 0, 0.95),
        'right_ankle': (0.52, 0.8, 0, 0.95),
        'right_heel': (0.54, 0.85, 0, 0.95),
    }
    results_zero_hip = calculate_ankle_inversion(sample_landmarks_zero_hip)
    print(f"\nResults for Zero Hip Width:\n{results_zero_hip}")