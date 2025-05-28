# arm_swing_mechanics.py

"""
Analyzes arm swing mechanics during running from rear view perspective.

Efficient arm swing should move primarily in the sagittal plane, maintain
symmetrical timing and amplitude, preserve ~90° elbow flexion, counter-rotate
with opposite leg, and avoid excessive midline crossing. Poor mechanics can
lead to energy inefficiency and compensatory movement patterns.
"""

import logging
import numpy as np
from typing import Dict, Tuple, Optional, Any

logger = logging.getLogger(__name__)

# Type aliases
Landmark = Tuple[float, float, float, float]
LandmarksDict = Dict[str, Landmark]
ArmSwingResult = Dict[str, Optional[Any]]

def calculate_arm_swing_mechanics(
    landmarks: LandmarksDict,
    symmetry_threshold: float = 0.05,
    rotation_threshold: float = 0.03
) -> ArmSwingResult:
    """
    Analyzes arm swing mechanics during running from rear view perspective.
    
    Evaluates vertical symmetry, elbow angles, crossover patterns, and shoulder
    stability. Measurements are normalized by hip width for consistent comparison
    across body types.
    
    Parameters:
    -----------
    landmarks : LandmarksDict
        Dictionary containing 3D coordinates and visibility of detected landmarks.
        Required keys: 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                      'left_wrist', 'right_wrist', 'left_hip', 'right_hip'.
        Each landmark is a tuple: (x, y, z, visibility).
    
    symmetry_threshold : float, default=0.05
        Threshold as proportion of hip width for assessing arm height symmetry.
        Values above indicate asymmetrical arm positioning.
    
    rotation_threshold : float, default=0.03
        Threshold as proportion of hip width for detecting excessive shoulder rotation.
        Values above indicate unstable shoulder mechanics.
    
    Returns:
    --------
    ArmSwingResult
        Dictionary containing:
        - "vertical_elbow_diff" (Optional[float]): Absolute vertical difference between elbows.
        - "normalized_vertical_diff" (Optional[float]): Vertical difference normalized by hip width.
        - "left_elbow_angle" (Optional[float]): Left elbow flexion angle in degrees.
        - "right_elbow_angle" (Optional[float]): Right elbow flexion angle in degrees.
        - "normalized_shoulder_diff" (Optional[float]): Shoulder height difference normalized by hip width.
        - "normalized_shoulder_width" (Optional[float]): Shoulder width normalized by hip width.
        - "arm_height_symmetry" (Optional[str]): Classification: "good", "moderate", "poor".
        - "elbow_angle_left" (Optional[str]): Classification: "optimal", "too_straight", "too_bent".
        - "elbow_angle_right" (Optional[str]): Classification: "optimal", "too_straight", "too_bent".
        - "left_wrist_crossover" (Optional[bool]): True if left wrist crosses midline.
        - "right_wrist_crossover" (Optional[bool]): True if right wrist crosses midline.
        - "shoulder_rotation" (Optional[str]): Classification: "stable", "excessive".
        - "calculation_successful" (bool): True if metrics calculated successfully.
    """
    
    required_landmarks = [
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip'
    ]
    
    # Initialize default return values
    result: ArmSwingResult = {
        "vertical_elbow_diff": None,
        "normalized_vertical_diff": None,
        "left_elbow_angle": None,
        "right_elbow_angle": None,
        "normalized_shoulder_diff": None,
        "normalized_shoulder_width": None,
        "arm_height_symmetry": None,
        "elbow_angle_left": None,
        "elbow_angle_right": None,
        "left_wrist_crossover": None,
        "right_wrist_crossover": None,
        "shoulder_rotation": None,
        "calculation_successful": False
    }
    
    # Check for required landmarks
    for lm_name in required_landmarks:
        if lm_name not in landmarks:
            logger.warning(f"Required landmark '{lm_name}' not found for arm swing analysis.")
            return result
    
    try:
        # Extract landmark coordinates
        left_shoulder = landmarks['left_shoulder']
        right_shoulder = landmarks['right_shoulder']
        left_elbow = landmarks['left_elbow']
        right_elbow = landmarks['right_elbow']
        left_wrist = landmarks['left_wrist']
        right_wrist = landmarks['right_wrist']
        
        # Calculate hip width for normalization
        left_hip_x = landmarks['left_hip'][0]
        right_hip_x = landmarks['right_hip'][0]
        hip_width = abs(right_hip_x - left_hip_x)
        
        if hip_width < 1e-5:
            logger.warning("Hip width is near zero. Arm swing measurements might be unreliable.",
                         extra={"landmarks": landmarks})
        
        # 1. Vertical symmetry - detect arms at different heights
        vertical_diff = abs(left_elbow[1] - right_elbow[1])
        normalized_vertical_diff = vertical_diff / hip_width if hip_width > 1e-5 else 0
        
        # 2. Crossover detection - arms crossing midline from rear view
        shoulder_midpoint_x = (left_shoulder[0] + right_shoulder[0]) / 2
        left_wrist_crossover = left_wrist[0] > shoulder_midpoint_x
        right_wrist_crossover = right_wrist[0] < shoulder_midpoint_x
        
        # 3. Elbow angle calculation
        def calculate_angle(a: Landmark, b: Landmark, c: Landmark) -> float:
            """Calculate angle between three points (b is the vertex)."""
            ba = np.array([a[0] - b[0], a[1] - b[1], a[2] - b[2]])
            bc = np.array([c[0] - b[0], c[1] - b[1], c[2] - b[2]])
            
            # Handle zero vectors
            ba_norm = np.linalg.norm(ba)
            bc_norm = np.linalg.norm(bc)
            
            if ba_norm < 1e-10 or bc_norm < 1e-10:
                logger.debug("Very small vector in angle calculation, returning 0")
                return 0.0
            
            cosine = np.dot(ba, bc) / (ba_norm * bc_norm)
            cosine = max(min(cosine, 1.0), -1.0)  # Clip to avoid numerical errors
            return np.degrees(np.arccos(cosine))
        
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        # 4. Shoulder rotation detection
        shoulder_height_diff = abs(left_shoulder[1] - right_shoulder[1])
        normalized_shoulder_diff = shoulder_height_diff / hip_width if hip_width > 1e-5 else 0
        
        # 5. Shoulder width stability
        shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
        normalized_shoulder_width = shoulder_width / hip_width if hip_width > 1e-5 else 0
        
        # Classification functions
        def classify_arm_symmetry(norm_diff: float) -> str:
            """Classify arm height symmetry."""
            if norm_diff < symmetry_threshold:
                return "good"
            elif norm_diff < symmetry_threshold * 2:
                return "moderate"
            else:
                return "poor"
        
        def classify_elbow_angle(angle: float) -> str:
            """Classify elbow angle optimality."""
            if 80 <= angle <= 110:
                return "optimal"
            elif angle > 110:
                return "too_straight"
            else:
                return "too_bent"
        
        def classify_shoulder_rotation(norm_diff: float) -> str:
            """Classify shoulder rotation stability."""
            return "stable" if norm_diff < rotation_threshold else "excessive"
        
        # Apply classifications
        arm_height_symmetry = classify_arm_symmetry(normalized_vertical_diff)
        elbow_angle_left = classify_elbow_angle(left_elbow_angle)
        elbow_angle_right = classify_elbow_angle(right_elbow_angle)
        shoulder_rotation = classify_shoulder_rotation(normalized_shoulder_diff)
        
        # Update result with calculated values
        result.update({
            "vertical_elbow_diff": vertical_diff,
            "normalized_vertical_diff": normalized_vertical_diff,
            "left_elbow_angle": left_elbow_angle,
            "right_elbow_angle": right_elbow_angle,
            "normalized_shoulder_diff": normalized_shoulder_diff,
            "normalized_shoulder_width": normalized_shoulder_width,
            "arm_height_symmetry": arm_height_symmetry,
            "elbow_angle_left": elbow_angle_left,
            "elbow_angle_right": elbow_angle_right,
            "left_wrist_crossover": left_wrist_crossover,
            "right_wrist_crossover": right_wrist_crossover,
            "shoulder_rotation": shoulder_rotation,
            "calculation_successful": True
        })
        
    except KeyError as e:
        logger.error(f"Missing landmark during arm swing calculation: {e}", exc_info=True)
    
    except Exception as e:
        logger.exception(f"An unexpected error occurred during arm swing calculation: {e}")
    
    return result


if __name__ == "__main__":
    print("Testing calculate_arm_swing_mechanics module...")
    
    # Example 1: Optimal arm swing
    sample_landmarks_optimal: LandmarksDict = {
        'left_hip': (0.4, 0.5, 0, 0.99),
        'right_hip': (0.6, 0.5, 0, 0.99),  # hip_width = 0.2
        'left_shoulder': (0.38, 0.3, 0, 0.95),
        'right_shoulder': (0.62, 0.3, 0, 0.95),
        'left_elbow': (0.35, 0.4, 0, 0.90),
        'right_elbow': (0.65, 0.4, 0, 0.90),  # symmetric height
        'left_wrist': (0.33, 0.45, 0, 0.85),
        'right_wrist': (0.67, 0.45, 0, 0.85)   # no crossover
    }
    results_optimal = calculate_arm_swing_mechanics(sample_landmarks_optimal)
    print(f"\nResults for Optimal Arm Swing:\n{results_optimal}")
    
    # Example 2: Asymmetric arm heights with crossover
    sample_landmarks_asymmetric: LandmarksDict = {
        'left_hip': (0.4, 0.5, 0, 0.99),
        'right_hip': (0.6, 0.5, 0, 0.99),
        'left_shoulder': (0.38, 0.3, 0, 0.95),
        'right_shoulder': (0.62, 0.3, 0, 0.95),
        'left_elbow': (0.35, 0.35, 0, 0.90),   # higher than right
        'right_elbow': (0.65, 0.45, 0, 0.90),
        'left_wrist': (0.52, 0.4, 0, 0.85),    # crosses midline
        'right_wrist': (0.48, 0.5, 0, 0.85)    # crosses midline
    }
    results_asymmetric = calculate_arm_swing_mechanics(sample_landmarks_asymmetric)
    print(f"\nResults for Asymmetric with Crossover:\n{results_asymmetric}")
    
    # Example 3: Poor elbow angles
    sample_landmarks_poor_angles: LandmarksDict = {
        'left_hip': (0.4, 0.5, 0, 0.99),
        'right_hip': (0.6, 0.5, 0, 0.99),
        'left_shoulder': (0.38, 0.3, 0, 0.95),
        'right_shoulder': (0.62, 0.3, 0, 0.95),
        'left_elbow': (0.35, 0.4, 0, 0.90),
        'right_elbow': (0.65, 0.4, 0, 0.90),
        'left_wrist': (0.32, 0.5, 0, 0.85),    # too straight arm
        'right_wrist': (0.66, 0.35, 0, 0.85)    # too bent arm
    }
    results_poor_angles = calculate_arm_swing_mechanics(sample_landmarks_poor_angles)
    print(f"\nResults for Poor Elbow Angles:\n{results_poor_angles}")
    
    # Example 4: Missing landmark
    sample_landmarks_missing: LandmarksDict = {
        'left_hip': (0.4, 0.5, 0, 0.99),
        'right_hip': (0.6, 0.5, 0, 0.99),
        'left_shoulder': (0.38, 0.3, 0, 0.95),
        # 'right_shoulder' missing
        'left_elbow': (0.35, 0.4, 0, 0.90),
        'right_elbow': (0.65, 0.4, 0, 0.90),
        'left_wrist': (0.33, 0.45, 0, 0.85),
        'right_wrist': (0.67, 0.45, 0, 0.85)
    }
    results_missing = calculate_arm_swing_mechanics(sample_landmarks_missing)
    print(f"\nResults for Missing Landmark:\n{results_missing}")
    
    # Example 5: Excessive shoulder rotation
    sample_landmarks_rotation: LandmarksDict = {
        'left_hip': (0.4, 0.5, 0, 0.99),
        'right_hip': (0.6, 0.5, 0, 0.99),
        'left_shoulder': (0.38, 0.25, 0, 0.95),  # elevated
        'right_shoulder': (0.62, 0.35, 0, 0.95),  # lowered
        'left_elbow': (0.35, 0.4, 0, 0.90),
        'right_elbow': (0.65, 0.4, 0, 0.90),
        'left_wrist': (0.33, 0.45, 0, 0.85),
        'right_wrist': (0.67, 0.45, 0, 0.85)
    }
    results_rotation = calculate_arm_swing_mechanics(sample_landmarks_rotation)
    print(f"\nResults for Excessive Shoulder Rotation:\n{results_rotation}")