# foot_crossover.py

"""
Calculates foot crossover and distance from the body's midline.

This metric helps identify if a runner's feet are crossing the midline
or landing too close to it, which can be indicative of certain biomechanical
inefficiencies or an increased risk of injury.
"""

import logging
from typing import Dict, Tuple, Optional, Any

logger = logging.getLogger(__name__) 

# Define a type alias for a single landmark's coordinates (x, y, z, visibility)
Landmark = Tuple[float, float, float, float]
# Define a type alias for the dictionary of all landmarks
LandmarksDict = Dict[str, Landmark]

# Define a type alias for the expected output structure of this function
FootCrossoverResult = Dict[str, Optional[Any]] # Using Any for values that can be bool or float

def calculate_foot_crossover(
    landmarks: LandmarksDict, 
    threshold_proportion: float = 0.25
) -> FootCrossoverResult:
    """
    Checks for feet being too close to or crossing the body's midline (from rear view).

    The midline is calculated as the center point between the hips. Distances are
    normalized by hip width. Crossover is determined if a foot is within a certain
    proportional distance (threshold_proportion) of the midline on its own side,
    or if it has crossed to the other side.

    Parameters:
    -----------
    landmarks : LandmarksDict
        A dictionary containing the 3D coordinates and visibility of detected pose landmarks.
        Expected keys for this function: 'left_hip', 'right_hip', 
                                        'left_foot_index', 'right_foot_index'.
        Each landmark is a tuple: (x, y, z, visibility). Coordinates are typically normalized (0.0-1.0).
    
    threshold_proportion : float, default=0.25
        The proportion of hip width used as a threshold. If a foot is closer to the
        midline than this proportion of the hip width (on its own side), or has
        crossed the midline, it's flagged as a crossover.
        Lower values are stricter (less deviation from a wider track allowed).

    Returns:
    --------
    FootCrossoverResult
        A dictionary containing:
        - "left_foot_crossover" (Optional[bool]): True if the left foot crosses or is too close to midline, else False. None if calculation fails.
        - "right_foot_crossover" (Optional[bool]): True if the right foot crosses or is too close, else False. None if calculation fails.
        - "left_distance_from_midline" (Optional[float]): The X-axis distance of the left foot from the hip center.
            Negative values indicate the foot is to the viewer's left of the midline.
            Positive values indicate the foot is to the viewer's right of the midline.
            None if calculation fails.
        - "right_distance_from_midline" (Optional[float]): The X-axis distance of the right foot from the hip center.
            Interpreted same as left_distance_from_midline.
            None if calculation fails.
        - "calculation_successful" (bool): True if metrics were calculated, False if essential landmarks were missing.
    """
    
    required_landmarks = [
        'left_hip', 'right_hip', 
        'left_foot_index', 'right_foot_index'
    ]
    
    # Initialize default return values for failure case
    result: FootCrossoverResult = {
        "left_foot_crossover": None,
        "right_foot_crossover": None,
        "left_distance_from_midline": None,
        "right_distance_from_midline": None,
        "calculation_successful": False
    }

    # Check for presence of all required landmarks
    for lm_name in required_landmarks:
        if lm_name not in landmarks:
            logger.warning(f"Required landmark '{lm_name}' not found for foot crossover calculation.")            # print(f"Warning: Required landmark '{lm_name}' not found for foot crossover calculation.")
            return result

    try:
        # Calculate hip center and width for reference
        left_hip_x = landmarks['left_hip'][0]
        right_hip_x = landmarks['right_hip'][0]
        
        # Ensure right hip is to the right of left hip (positive width)
        # This assumes standard image coordinates where x increases to the right.
        # If landmarks are from a mirror view or different system, this might need adjustment.
        if left_hip_x > right_hip_x:
            left_hip_x, right_hip_x = right_hip_x, left_hip_x # Swap if needed

        hip_center_x = (left_hip_x + right_hip_x) / 2
        hip_width = right_hip_x - left_hip_x # Should be positive now

        if hip_width < 1e-5: # Avoid division by zero or extreme sensitivity with tiny hip width
            logger.warning("Hip width is near zero. Crossover detection might be unreliable.", 
                    extra={"landmarks": landmarks}) # Optional: add extra context
            # In this case, any deviation might be a crossover.
            # Or, decide to return failure/defaults. For now, proceed cautiously.
            # If hip_width is effectively zero, the threshold distance becomes zero.
            # Calculation can proceed, threshold_distance will be 0.

        # Get foot positions
        left_foot_x = landmarks['left_foot_index'][0]
        right_foot_x = landmarks['right_foot_index'][0]
        
        # Calculate distances from center line
        # For left foot: negative distance means it's to the left of hip_center_x (expected side)
        #                positive distance means it's to the right of hip_center_x (crossed over)
        left_distance = left_foot_x - hip_center_x
        
        # For right foot: positive distance means it's to the right of hip_center_x (expected side)
        #                 negative distance means it's to the left of hip_center_x (crossed over)
        right_distance = right_foot_x - hip_center_x
        
        # Determine the threshold distance in normalized units
        threshold_distance = threshold_proportion * hip_width
        
        # Check for crossover
        # Left foot crosses if it's to the right of (midline - threshold_distance_on_left_side)
        # This means its distance from midline is greater than -threshold_distance
        # e.g., if threshold_distance is 0.05, crossover if left_distance > -0.05
        # This covers foot being too close on its own side (e.g. -0.02) or crossed (e.g. +0.01)
        crossover_left = left_distance > -threshold_distance
        
        # Right foot crosses if it's to the left of (midline + threshold_distance_on_right_side)
        # This means its distance from midline is less than +threshold_distance
        # e.g., if threshold_distance is 0.05, crossover if right_distance < 0.05
        # This covers foot being too close on its own side (e.g. +0.02) or crossed (e.g. -0.01)
        crossover_right = right_distance < threshold_distance
        
        result["left_foot_crossover"] = crossover_left
        result["right_foot_crossover"] = crossover_right
        result["left_distance_from_midline"] = left_distance
        result["right_distance_from_midline"] = right_distance
        result["calculation_successful"] = True

    except KeyError as e:
        # This is a fallback, though the initial check should prevent most KeyErrors.
        logger.error(f"Missing landmark during foot crossover calculation: {e}", exc_info=True)       

    except Exception as e:
        # Catch any other unexpected errors during calculation
        logger.exception(f"An unexpected error occurred during foot crossover calculation: {e}")        # Result already initialized to failure state
                
    return result

if __name__ == "__main__":
    print("Testing calculate_foot_crossover module...")

    # Example 1: Clear crossover for both feet
    sample_landmarks_crossover: LandmarksDict = {
        'left_hip': (0.4, 0.5, 0, 0.99),
        'right_hip': (0.6, 0.5, 0, 0.99), # hip_center_x = 0.5, hip_width = 0.2
        'left_foot_index': (0.52, 0.8, 0, 0.95), # left_distance = 0.02. Threshold_dist = 0.25 * 0.2 = 0.05.  0.02 > -0.05 (True)
        'right_foot_index': (0.48, 0.8, 0, 0.95) # right_distance = -0.02. -0.02 < 0.05 (True)
    }
    results_crossover = calculate_foot_crossover(sample_landmarks_crossover)
    print(f"\nResults for Crossover Case:\n{results_crossover}")

    # Example 2: No crossover, feet well apart
    sample_landmarks_no_crossover: LandmarksDict = {
        'left_hip': (0.4, 0.5, 0, 0.99),
        'right_hip': (0.6, 0.5, 0, 0.99), # hip_center_x = 0.5, hip_width = 0.2
        'left_foot_index': (0.35, 0.8, 0, 0.95), # left_distance = -0.15. -0.15 > -0.05 (False)
        'right_foot_index': (0.65, 0.8, 0, 0.95)  # right_distance = 0.15. 0.15 < 0.05 (False)
    }
    results_no_crossover = calculate_foot_crossover(sample_landmarks_no_crossover)
    print(f"\nResults for No Crossover Case:\n{results_no_crossover}")

    # Example 3: Feet close to midline but not crossing (depends on threshold)
    # threshold_proportion = 0.25, hip_width = 0.2, threshold_distance = 0.05
    sample_landmarks_close: LandmarksDict = {
        'left_hip': (0.4, 0.5, 0, 0.99),
        'right_hip': (0.6, 0.5, 0, 0.99),
        'left_foot_index': (0.46, 0.8, 0, 0.95), # left_distance = -0.04. -0.04 > -0.05 (True, flagged as "crossover" or too close)
        'right_foot_index': (0.54, 0.8, 0, 0.95) # right_distance = 0.04.  0.04 < 0.05 (True, flagged as "crossover" or too close)
    }
    results_close = calculate_foot_crossover(sample_landmarks_close)
    print(f"\nResults for Close to Midline Case (threshold_proportion=0.25):\n{results_close}")

    # Example 4: Missing landmark
    sample_landmarks_missing: LandmarksDict = {
        'left_hip': (0.4, 0.5, 0, 0.99),
        # 'right_hip' is missing
        'left_foot_index': (0.4, 0.8, 0, 0.95),
        'right_foot_index': (0.6, 0.8, 0, 0.95)
    }
    results_missing = calculate_foot_crossover(sample_landmarks_missing)
    print(f"\nResults for Missing Landmark Case:\n{results_missing}")

    # Example 5: Hips vertically aligned (hip_width = 0)
    sample_landmarks_zero_hip_width: LandmarksDict = {
        'left_hip': (0.5, 0.45, 0, 0.99),
        'right_hip': (0.5, 0.55, 0, 0.99), # hip_center_x = 0.5, hip_width = 0
        'left_foot_index': (0.48, 0.8, 0, 0.95), # left_distance = -0.02. Threshold_dist = 0. -0.02 > 0 (False)
        'right_foot_index': (0.52, 0.8, 0, 0.95) # right_distance = 0.02.  0.02 < 0 (False)
    }
    # In this zero hip_width case, crossover means literally on the other side of the exact midline point.
    # crossover_left = left_distance > 0 (left foot is to the right of hip_center_x)
    # crossover_right = right_distance < 0 (right foot is to the left of hip_center_x)
    results_zero_hip_width = calculate_foot_crossover(sample_landmarks_zero_hip_width)
    print(f"\nResults for Zero Hip Width Case:\n{results_zero_hip_width}")