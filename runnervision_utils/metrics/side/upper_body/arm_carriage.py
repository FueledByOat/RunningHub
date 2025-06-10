# arm_carriage.py

"""
Analyzes running arm carriage and swing mechanics from pose landmarks.

This module evaluates arm position, elbow angle, hand placement, and swing amplitude
to identify biomechanical inefficiencies that may impact running performance or 
increase injury risk. Analysis focuses on the right arm when visible from right-side view.
"""

import logging
import math
from collections import deque
from typing import Dict, Tuple, Optional, Any, List
import numpy as np

logger = logging.getLogger(__name__)

# Type aliases for clarity and consistency
Landmark = Tuple[float, float, float, float]  # (x, y, z, visibility)
LandmarksDict = Dict[str, Landmark]
ArmCarriageResult = Dict[str, Optional[Any]]

# Configuration constants
DEFAULT_FRAME_RATE = 30.0
DEFAULT_HISTORY_SIZE = 60
OPTIMAL_ELBOW_ANGLE_MIN = 85.0
OPTIMAL_ELBOW_ANGLE_MAX = 115.0
OPTIMAL_SWING_AMPLITUDE = 52.5  # degrees
MIN_FRAMES_FOR_AMPLITUDE = 15
MIDLINE_CROSS_THRESHOLD = -15.0  # degrees


def calculate_arm_carriage(
    landmarks: LandmarksDict,
    frame_rate: float = DEFAULT_FRAME_RATE,
    arm_history: Optional[deque] = None
) -> ArmCarriageResult:
    """
    Analyzes arm carriage mechanics from pose landmarks for a single frame.
    
    Evaluates elbow angle, hand position, and upper arm positioning to assess
    running form efficiency. For swing amplitude analysis, historical data
    is required via the arm_history parameter.
    
    Parameters:
    -----------
    landmarks : LandmarksDict
        Dictionary containing pose landmarks with (x, y, z, visibility) coordinates.
        Required keys: 'right_shoulder', 'right_elbow', 'right_wrist'
        Optional keys: 'right_hip' (for hand position assessment)
        
    frame_rate : float, default=30.0
        Video frame rate in fps (currently unused but maintained for future features)
        
    arm_history : Optional[deque], default=None
        Historical arm position data for swing amplitude calculation.
        If None, swing amplitude will not be calculated.
    
    Returns:
    --------
    ArmCarriageResult
        Dictionary containing:
        - "upper_arm_angle" (Optional[float]): Angle of upper arm relative to vertical in degrees
        - "elbow_angle" (Optional[float]): Interior elbow angle in degrees (0°=fully bent, 180°=straight)
        - "hand_position" (Optional[str]): Hand position assessment ('optimal', 'too_high', 'too_low')
        - "arm_swing_amplitude" (Optional[float]): Normalized swing amplitude (1.0 = optimal)
        - "crosses_midline" (Optional[bool]): True if arm crosses body midline
        - "overall_assessment" (str): Overall form assessment
        - "recommendations" (List[str]): List of improvement recommendations
        - "calculation_successful" (bool): True if analysis completed successfully
    """
    
    # Required landmarks for arm analysis
    required_landmarks = ['right_shoulder', 'right_elbow', 'right_wrist']
    
    # Initialize default return values
    result: ArmCarriageResult = {
        "upper_arm_angle": None,
        "elbow_angle": None,
        "hand_position": None,
        "arm_swing_amplitude": None,
        "crosses_midline": None,
        "overall_assessment": "insufficient_data",
        "recommendations": ["Ensure right arm is visible in the video for proper analysis"],
        "calculation_successful": False
    }
    
    # Validate required landmarks presence
    for landmark_name in required_landmarks:
        if landmark_name not in landmarks:
            logger.warning(f"Required landmark '{landmark_name}' not found for arm carriage analysis.")
            return result
    
    try:
        # Extract landmark coordinates
        shoulder = landmarks['right_shoulder'][:2]  # (x, y)
        elbow = landmarks['right_elbow'][:2]
        wrist = landmarks['right_wrist'][:2]
        
        # Calculate arm angles and positions
        upper_arm_angle = _calculate_upper_arm_angle(shoulder, elbow)
        elbow_angle = _calculate_elbow_angle(shoulder, elbow, wrist)
        hand_position = _analyze_hand_position(landmarks, shoulder, wrist)
        
        # Calculate swing amplitude if history is available
        swing_amplitude = None
        if arm_history is not None and len(arm_history) >= MIN_FRAMES_FOR_AMPLITUDE:
            swing_amplitude = _calculate_swing_amplitude(arm_history)
        
        # Check for midline crossing
        crosses_midline = upper_arm_angle < MIDLINE_CROSS_THRESHOLD
        
        # Generate assessment and recommendations
        overall_assessment, recommendations = _generate_assessment(
            upper_arm_angle, elbow_angle, hand_position, swing_amplitude, crosses_midline
        )
        
        # Update result with calculated values
        result.update({
            "upper_arm_angle": upper_arm_angle,
            "elbow_angle": elbow_angle,
            "hand_position": hand_position,
            "arm_swing_amplitude": swing_amplitude,
            "crosses_midline": crosses_midline,
            "overall_assessment": overall_assessment,
            "recommendations": recommendations,
            "calculation_successful": True
        })
        
        logger.debug(f"Arm carriage analysis completed: elbow={elbow_angle:.1f}°, "
                    f"hand_position={hand_position}, crosses_midline={crosses_midline}")
                    
    except KeyError as e:
        logger.error(f"Missing landmark during arm carriage calculation: {e}", exc_info=True)
        
    except Exception as e:
        logger.exception(f"Unexpected error during arm carriage calculation: {e}")
    
    return result


def _calculate_upper_arm_angle(shoulder: Tuple[float, float], elbow: Tuple[float, float]) -> float:
    """
    Calculate upper arm angle relative to vertical axis.
    
    Parameters:
    -----------
    shoulder : Tuple[float, float]
        Shoulder landmark coordinates (x, y)
    elbow : Tuple[float, float]
        Elbow landmark coordinates (x, y)
        
    Returns:
    --------
    float
        Angle in degrees where 0° is vertical, positive values indicate rightward lean
    """
    upper_arm_dx = elbow[0] - shoulder[0]
    upper_arm_dy = shoulder[1] - elbow[1]  # Inverted y-axis for image coordinates
    
    return math.degrees(math.atan2(upper_arm_dx, upper_arm_dy))


def _calculate_elbow_angle(
    shoulder: Tuple[float, float], 
    elbow: Tuple[float, float], 
    wrist: Tuple[float, float]
) -> float:
    """
    Calculate interior elbow angle using vector dot product.
    
    Parameters:
    -----------
    shoulder : Tuple[float, float]
        Shoulder landmark coordinates (x, y)
    elbow : Tuple[float, float]
        Elbow landmark coordinates (x, y)
    wrist : Tuple[float, float]
        Wrist landmark coordinates (x, y)
        
    Returns:
    --------
    float
        Interior elbow angle in degrees (0° = fully bent, 180° = straight)
    """
    # Calculate arm segment vectors
    upper_arm = [elbow[0] - shoulder[0], elbow[1] - shoulder[1]]
    forearm = [wrist[0] - elbow[0], wrist[1] - elbow[1]]
    
    # Calculate dot product
    dot_product = upper_arm[0] * forearm[0] + upper_arm[1] * forearm[1]
    
    # Calculate vector magnitudes
    upper_arm_magnitude = math.sqrt(upper_arm[0]**2 + upper_arm[1]**2)
    forearm_magnitude = math.sqrt(forearm[0]**2 + forearm[1]**2)
    
    # Handle zero-length vectors
    if upper_arm_magnitude < 1e-10 or forearm_magnitude < 1e-10:
        logger.warning("Near-zero arm segment length detected, using default elbow angle")
        return 90.0
    
    # Calculate angle using dot product formula
    cos_angle = dot_product / (upper_arm_magnitude * forearm_magnitude)
    cos_angle = max(min(cos_angle, 1.0), -1.0)  # Clamp to valid range [-1, 1]
    
    angle_rad = math.acos(cos_angle)
    return math.degrees(angle_rad)


def _analyze_hand_position(
    landmarks: LandmarksDict, 
    shoulder: Tuple[float, float], 
    wrist: Tuple[float, float]
) -> str:
    """
    Analyze hand vertical position relative to body landmarks.
    
    Parameters:
    -----------
    landmarks : LandmarksDict
        Full landmarks dictionary for reference points
    shoulder : Tuple[float, float]
        Shoulder coordinates (x, y)
    wrist : Tuple[float, float]
        Wrist coordinates (x, y)
        
    Returns:
    --------
    str
        Hand position assessment: 'optimal', 'too_high', or 'too_low'
    """
    shoulder_y = shoulder[1]
    wrist_y = wrist[1]
    
    # Get hip position for reference
    if 'right_hip' in landmarks:
        hip_y = landmarks['right_hip'][1]
    else:
        # Estimate hip position as 30% below shoulder
        hip_y = shoulder_y + abs(shoulder_y * 0.3)
        logger.debug("Hip landmark not available, using estimated position")
    
    # Assess vertical hand position (remember inverted y-coordinates)
    if wrist_y < shoulder_y:  # Hand above shoulder
        return 'too_high'
    elif wrist_y > hip_y:  # Hand below hip
        return 'too_low'
    else:
        return 'optimal'


def _calculate_swing_amplitude(arm_history: deque) -> Optional[float]:
    """
    Calculate arm swing amplitude from historical wrist angle data.
    
    Uses percentile-based analysis to determine swing range while filtering outliers.
    
    Parameters:
    -----------
    arm_history : deque
        Historical arm position data containing wrist angles
        
    Returns:
    --------
    Optional[float]
        Normalized swing amplitude where 1.0 represents optimal swing,
        or None if insufficient data
    """
    try:
        # Extract wrist angles from history
        angles = [pos.get('wrist_angle', 0) for pos in arm_history if 'wrist_angle' in pos]
        
        if len(angles) < MIN_FRAMES_FOR_AMPLITUDE:
            return None
            
        angles_array = np.array(angles)
        
        # Use percentiles to determine swing range (filters outliers)
        p10 = np.percentile(angles_array, 10)  # Lower swing bound
        p90 = np.percentile(angles_array, 90)  # Upper swing bound
        
        swing_range_degrees = p90 - p10
        
        # Normalize against optimal swing amplitude
        normalized_amplitude = swing_range_degrees / OPTIMAL_SWING_AMPLITUDE
        
        logger.debug(f"Swing amplitude calculated: {swing_range_degrees:.1f}° "
                    f"(normalized: {normalized_amplitude:.2f})")
        
        return normalized_amplitude
        
    except Exception as e:
        logger.error(f"Error calculating swing amplitude: {e}")
        return None


def _generate_assessment(
    upper_arm_angle: float,
    elbow_angle: float,
    hand_position: str,
    swing_amplitude: Optional[float],
    crosses_midline: bool
) -> Tuple[str, List[str]]:
    """
    Generate overall assessment and recommendations based on arm carriage metrics.
    
    Parameters:
    -----------
    upper_arm_angle : float
        Upper arm angle relative to vertical
    elbow_angle : float
        Interior elbow angle in degrees
    hand_position : str
        Hand position assessment
    swing_amplitude : Optional[float]
        Normalized swing amplitude
    crosses_midline : bool
        Whether arm crosses body midline
        
    Returns:
    --------
    Tuple[str, List[str]]
        Overall assessment string and list of recommendations
    """
    recommendations = []
    issues = []
    
    # Analyze elbow angle
    if elbow_angle < OPTIMAL_ELBOW_ANGLE_MIN:
        issues.append("elbow_too_bent")
        recommendations.append(f"Open elbow angle slightly (aim for {OPTIMAL_ELBOW_ANGLE_MIN:.0f}-{OPTIMAL_ELBOW_ANGLE_MAX:.0f}°)")
    elif elbow_angle > OPTIMAL_ELBOW_ANGLE_MAX:
        issues.append("elbow_too_straight")
        recommendations.append(f"Increase elbow bend (aim for {OPTIMAL_ELBOW_ANGLE_MIN:.0f}-{OPTIMAL_ELBOW_ANGLE_MAX:.0f}°)")
    
    # Analyze hand position
    if hand_position == 'too_high':
        issues.append("hand_position_high")
        recommendations.append("Keep hands below shoulder height during arm swing")
    elif hand_position == 'too_low':
        issues.append("hand_position_low")
        recommendations.append("Avoid dropping hands below hip level during arm swing")
    
    # Analyze swing amplitude
    if swing_amplitude is not None:
        if swing_amplitude < 0.7:  # Less than 70% of optimal
            issues.append("insufficient_swing")
            recommendations.append("Increase arm swing amplitude for better running efficiency")
        elif swing_amplitude > 1.4:  # More than 140% of optimal
            issues.append("excessive_swing")
            recommendations.append("Consider reducing excessive arm swing to conserve energy")
    
    # Check for midline crossing
    if crosses_midline:
        issues.append("crosses_midline")
        recommendations.append("Avoid swinging right arm across body midline")
    
    # Determine overall assessment
    if not issues:
        overall_assessment = "optimal"
        recommendations.insert(0, "Arm carriage is good - maintain current form")
    elif len(issues) == 1:
        overall_assessment = f"suboptimal_{issues[0]}"
    else:
        overall_assessment = "multiple_issues"
        recommendations.insert(0, "Multiple arm carriage issues detected")
    
    return overall_assessment, recommendations


class ArmCarriageAnalyzer:
    """
    Stateful analyzer for running arm carriage mechanics with historical tracking.
    
    Maintains internal history of arm positions to enable swing amplitude analysis
    and trend detection. Designed for frame-by-frame analysis of video data.
    
    Attributes:
    -----------
    frame_rate : float
        Video frame rate in fps
    history_size : int
        Maximum number of frames to retain in history
    frame_count : int
        Total number of frames processed
    """
    
    def __init__(self, frame_rate: float = DEFAULT_FRAME_RATE, history_size: int = DEFAULT_HISTORY_SIZE):
        """
        Initialize the arm carriage analyzer with specified parameters.
        
        Parameters:
        -----------
        frame_rate : float, default=30.0
            Video frame rate in fps
        history_size : int, default=60
            Number of frames to maintain for swing analysis (typically 2 seconds of data)
        """
        self.frame_rate = frame_rate
        self.history_size = history_size
        self.frame_count = 0
        
        # Historical data storage
        self.right_arm_positions = deque(maxlen=history_size)
        
        logger.info(f"ArmCarriageAnalyzer initialized: frame_rate={frame_rate}, "
                   f"history_size={history_size}")
    
    def update(self, landmarks: LandmarksDict) -> ArmCarriageResult:
        """
        Process new frame data and update arm carriage analysis.
        
        Stores current frame data in history and performs comprehensive analysis
        including swing amplitude calculation when sufficient history is available.
        
        Parameters:
        -----------
        landmarks : LandmarksDict
            Dictionary containing pose landmarks for current frame
            
        Returns:
        --------
        ArmCarriageResult
            Comprehensive arm carriage analysis results
        """
        self.frame_count += 1
        
        # Store current frame data if landmarks are available
        if self._has_required_landmarks(landmarks):
            self._store_frame_data(landmarks)
        
        # Perform analysis using functional approach
        result = calculate_arm_carriage(landmarks, self.frame_rate, self.right_arm_positions)
        
        # Add analyzer-specific metadata
        result['frames_analyzed'] = len(self.right_arm_positions)
        result['total_frames_processed'] = self.frame_count
        
        logger.debug(f"Frame {self.frame_count} processed, "
                    f"history size: {len(self.right_arm_positions)}")
        
        return result
    
    def _has_required_landmarks(self, landmarks: LandmarksDict) -> bool:
        """Check if required landmarks are present for analysis."""
        required = ['right_shoulder', 'right_elbow', 'right_wrist']
        return all(landmark in landmarks for landmark in required)
    
    def _store_frame_data(self, landmarks: LandmarksDict) -> None:
        """
        Store current frame arm position data for historical analysis.
        
        Parameters:
        -----------
        landmarks : LandmarksDict
            Current frame landmarks
        """
        try:
            shoulder = landmarks['right_shoulder'][:2]
            elbow = landmarks['right_elbow'][:2]
            wrist = landmarks['right_wrist'][:2]
            
            # Calculate wrist angle relative to shoulder for swing analysis
            wrist_dx = wrist[0] - shoulder[0]
            wrist_dy = shoulder[1] - wrist[1]  # Inverted y for image coordinates
            wrist_angle = math.degrees(math.atan2(wrist_dy, wrist_dx))
            
            # Store frame data
            frame_data = {
                'shoulder': shoulder,
                'elbow': elbow,
                'wrist': wrist,
                'wrist_angle': wrist_angle,
                'frame': self.frame_count
            }
            
            self.right_arm_positions.append(frame_data)
            
        except Exception as e:
            logger.error(f"Error storing frame data: {e}")
    
    def get_analysis_summary(self) -> str:
        """
        Generate a comprehensive text summary of current arm carriage state.
        
        Returns:
        --------
        str
            Formatted summary of arm carriage analysis
        """
        if len(self.right_arm_positions) == 0:
            return "No arm carriage data available for analysis."
        
        # Get latest landmarks for current analysis
        latest_pos = self.right_arm_positions[-1]
        landmarks = {
            'right_shoulder': (*latest_pos['shoulder'], 0, 0.99),
            'right_elbow': (*latest_pos['elbow'], 0, 0.99),
            'right_wrist': (*latest_pos['wrist'], 0, 0.99)
        }
        
        # Get current analysis
        analysis = calculate_arm_carriage(landmarks, self.frame_rate, self.right_arm_positions)
        
        # Generate formatted summary
        summary_lines = [
            "RIGHT ARM CARRIAGE ANALYSIS:",
            f"• Frames analyzed: {len(self.right_arm_positions)}/{self.frame_count}",
        ]
        
        if analysis['elbow_angle'] is not None:
            elbow = analysis['elbow_angle']
            status = "optimal" if OPTIMAL_ELBOW_ANGLE_MIN <= elbow <= OPTIMAL_ELBOW_ANGLE_MAX else \
                    "too bent" if elbow < OPTIMAL_ELBOW_ANGLE_MIN else "too straight"
            summary_lines.append(f"• Elbow angle: {elbow:.1f}° ({status})")
        
        if analysis['hand_position'] is not None:
            summary_lines.append(f"• Hand position: {analysis['hand_position']}")
        
        if analysis['arm_swing_amplitude'] is not None:
            amp = analysis['arm_swing_amplitude']
            status = "optimal" if 0.8 <= amp <= 1.2 else \
                    "insufficient" if amp < 0.8 else "excessive"
            summary_lines.append(f"• Swing amplitude: {amp:.2f} ({status})")
        
        if analysis.get('crosses_midline'):
            summary_lines.append("• ⚠ Arm crosses body midline")
        
        if analysis['recommendations']:
            summary_lines.extend([
                "",
                "RECOMMENDATIONS:"
            ])
            summary_lines.extend([f"{i}. {rec}" for i, rec in enumerate(analysis['recommendations'], 1)])
        
        return "\n".join(summary_lines)
    
    def reset(self) -> None:
        """Reset analyzer state and clear all historical data."""
        self.right_arm_positions.clear()
        self.frame_count = 0
        logger.info("ArmCarriageAnalyzer state reset")


if __name__ == "__main__":
    print("Testing arm_carriage module...")
    
    # Test case 1: Optimal arm carriage
    optimal_landmarks: LandmarksDict = {
        'right_shoulder': (0.3, 0.3, 0, 0.99),
        'right_elbow': (0.35, 0.45, 0, 0.99),
        'right_wrist': (0.32, 0.6, 0, 0.99),
        'right_hip': (0.31, 0.65, 0, 0.99)
    }
    
    result_optimal = calculate_arm_carriage(optimal_landmarks)
    print(f"\nOptimal Case Results:")
    print(f"Elbow angle: {result_optimal['elbow_angle']:.1f}°")
    print(f"Hand position: {result_optimal['hand_position']}")
    print(f"Assessment: {result_optimal['overall_assessment']}")
    
    # Test case 2: Problematic arm carriage
    problematic_landmarks: LandmarksDict = {
        'right_shoulder': (0.3, 0.3, 0, 0.99),
        'right_elbow': (0.25, 0.35, 0, 0.99),  # Crosses midline
        'right_wrist': (0.22, 0.25, 0, 0.99),  # Hand too high, very bent elbow
        'right_hip': (0.31, 0.65, 0, 0.99)
    }
    
    result_problematic = calculate_arm_carriage(problematic_landmarks)
    print(f"\nProblematic Case Results:")
    print(f"Elbow angle: {result_problematic['elbow_angle']:.1f}°")
    print(f"Hand position: {result_problematic['hand_position']}")
    print(f"Crosses midline: {result_problematic['crosses_midline']}")
    print(f"Assessment: {result_problematic['overall_assessment']}")
    print("Recommendations:")
    for i, rec in enumerate(result_problematic['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    # Test case 3: Missing landmarks
    incomplete_landmarks: LandmarksDict = {
        'right_shoulder': (0.3, 0.3, 0, 0.99)
        # Missing elbow and wrist
    }
    
    result_incomplete = calculate_arm_carriage(incomplete_landmarks)
    print(f"\nIncomplete Data Results:")
    print(f"Calculation successful: {result_incomplete['calculation_successful']}")
    print(f"Assessment: {result_incomplete['overall_assessment']}")
    
    # Test case 4: Analyzer with history
    print(f"\nTesting ArmCarriageAnalyzer with history...")
    analyzer = ArmCarriageAnalyzer(history_size=10)
    
    # Simulate multiple frames
    for frame in range(15):
        # Simulate varying wrist positions for swing analysis
        wrist_x = 0.32 + 0.05 * math.sin(frame * 0.4)  # Oscillating swing
        test_landmarks = {
            'right_shoulder': (0.3, 0.3, 0, 0.99),
            'right_elbow': (0.35, 0.45, 0, 0.99),
            'right_wrist': (wrist_x, 0.6, 0, 0.99),
            'right_hip': (0.31, 0.65, 0, 0.99)
        }
        
        result = analyzer.update(test_landmarks)
        
        if frame == 14:  # Print final result
            print(f"Final swing amplitude: {result['arm_swing_amplitude']:.2f}")
            print(f"Frames in history: {result['frames_analyzed']}")
    
    print("\nDetailed Summary:")
    print(analyzer.get_analysis_summary())