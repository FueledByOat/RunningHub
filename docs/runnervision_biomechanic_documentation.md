# RunnerVision Biomechanic Documentation  
  
Found 24 biomechanic modules.  
  
### Module: rating_utils  
**File**: `runnervision_utils\reports\rating_utils.py`  
  
#### Functions  
##### get_rating_from_score(score, thresholds)  
Determines the rating category based on a score and defined thresholds.

Args:
    score (float): The score to evaluate (0-100).
    thresholds (dict): A dictionary with keys 'optimal', 'good', 'fair' defining the lower bound of each category.

Returns:
    tuple: A tuple containing the rating text (str) and rating key (str).  
  
##### rate_cadence(cadence)  
Rates running cadence (in steps per minute).  
  
##### rate_trunk_angle(angle)  
Rates forward trunk lean in degrees.  
  
##### rate_knee_symmetry(diff_percent)  
Rates knee symmetry based on percentage difference.  
  
##### rate_crossover(crossover_percent)  
Rates foot crossover percentage (lower is better).  
  
---  
  
### Module: text_generation  
**File**: `runnervision_utils\reports\text_generation.py`  
  
#### Functions  
##### generate_rear_view_summary_from_llm(summary_data, lm_model)  
Generates a text summary for the rear view analysis using a local language model
and formats the markdown response into HTML.  
  
##### generate_side_view_summary_from_llm(summary_data, lm_model)  
Generates a text summary for the side view analysis using a local language model.

Args:
    summary_data (dict): A dictionary of summarized metrics for the side view.

Returns:
    str: A raw markdown string with the generated analysis and recommendations.  
  
---  
  
### Module: stance_phase_detector_rear  
**File**: `runnervision_utils\metrics\rear\gait_events_and_phases\stance_phase_detector_rear.py`  
  
Rear-view stance phase detection for running biomechanics analysis.

This module provides functionality to detect stance phase (ground contact) during running
from rear-view camera angles using MediaPipe Pose landmarks. The detector uses vertical
foot position relative to calibrated ground and swing zones to determine when feet are
in contact with the ground.

Key Features:
- Automatic calibration from observed foot motion
- Robust detection using multiple foot landmarks
- Confidence scoring for detection reliability
- Handles missing or low-visibility landmarks gracefully  
  
#### Classes  
##### StancePhaseDetectorRear  
Detects stance phase from rear-view running analysis using foot landmark positions.

This detector works by:
1. Calibrating ground and swing zones from observed foot motion
2. Comparing current foot positions to these zones
3. Determining which foot (if any) is in ground contact
4. Providing confidence scores for detection reliability

The detector assumes normalized MediaPipe coordinates where:
- Y=0 is top of frame, Y=1 is bottom of frame
- Higher Y values indicate lower vertical positions (closer to ground)  
  
**Methods**:  
- `__init__(calibration_frames_total, ground_zone_percentage, visibility_threshold, foot_landmarks_to_use)` -> None  
  - Initialize the rear-view stance phase detector.  
- `_get_lowest_point_of_foot(landmarks, side_key)` -> Optional[float]  
  - Get the lowest Y coordinate for a foot, considering landmark visibility.  
- `_collect_calibration_data(landmarks)` -> None  
  - Collect foot position data during calibration phase.  
- `_finalize_calibration()` -> None  
  - Calculate detection thresholds from collected calibration data.  
- `_set_default_calibration_values()` -> None  
  - Set reasonable default values when calibration fails or has insufficient data.  
- `_validate_and_adjust_calibration()` -> None  
  - Ensure calibration values are reasonable and adjust if necessary.  
- `_calculate_stance_confidence(foot_y, is_in_stance, other_foot_y)` -> float  
  - Calculate confidence score for stance/flight detection.  
- `detect_stance_phase(landmarks)` -> StanceResult  
  - Detect stance phase from current frame landmarks.  
- `reset_calibration()` -> None  
  - Reset calibration state to allow re-calibration with new data.  
- `is_calibrated()` -> bool  
  - Check if detector has completed calibration.  
- `get_calibration_info()` -> Dict[(str, Any)]  
  - Get current calibration parameters and status.  
  
#### Functions  
##### detect_stance_phase_rear(landmarks, detector_instance) -> StanceResult  
Convenience wrapper function for rear-view stance phase detection.

This function provides a simple interface for stance detection while managing
the detector instance lifecycle. It's designed to be called frame-by-frame
in a video processing pipeline.

Args:
    landmarks: Dictionary of MediaPipe pose landmarks for current frame
    detector_instance: Optional pre-configured detector instance. If None,
                     a new detector with default settings will be created
                     and stored for subsequent calls.
                     
Returns:
    Stance detection results dictionary (see StancePhaseDetectorRear.detect_stance_phase)
    
Example:
    >>> landmarks = get_mediapipe_landmarks(frame)
    >>> result = detect_stance_phase_rear(landmarks)
    >>> if result['is_stance_phase']:
    >>>     print(f"Stance foot: {result['stance_foot']}, confidence: {result['confidence']}")
    
Note:
    This wrapper maintains a single detector instance across calls using a
    function attribute. For multiple concurrent analyses, create separate
    StancePhaseDetectorRear instances directly.  
  
---  
  
### Module: step_width  
**File**: `runnervision_utils\metrics\rear\gait_events_and_phases\step_width.py`  
  
#### Functions  
##### calculate_step_width(landmarks)  
Distance between both feet.  
  
---  
  
### Module: stride_symmetry  
**File**: `runnervision_utils\metrics\rear\gait_events_and_phases\stride_symmetry.py`  
  
#### Functions  
##### calculate_stride_symmetry(landmarks)  
Compare stride or timing parameters over a cycle (requires frame history).
Placeholder uses foot x-delta.  
  
---  
  
### Module: ankle_inversion  
**File**: `runnervision_utils\metrics\rear\lower_body\ankle_inversion.py`  
  
Calculates ankle inversion/eversion patterns during running gait analysis.

This metric helps identify biomechanical inefficiencies related to foot strike
patterns that may lead to injury. Excessive inversion is linked to lateral ankle
sprains and insufficient shock absorption, while excessive eversion is associated
with medial tibial stress syndrome (shin splints) and plantar fasciitis.  
  
#### Functions  
##### calculate_ankle_inversion(landmarks, inversion_threshold) -> AnkleInversionResult  
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
  
##### get_pattern(normalized_value) -> str  
Classify inversion pattern based on normalized value.  
  
##### get_severity(normalized_value) -> str  
Determine severity level based on normalized value.  
  
---  
  
### Module: foot_crossover  
**File**: `runnervision_utils\metrics\rear\lower_body\foot_crossover.py`  
  
Calculates foot crossover and distance from the body's midline.

This metric helps identify if a runner's feet are crossing the midline
or landing too close to it, which can be indicative of certain biomechanical
inefficiencies or an increased risk of injury.  
  
#### Functions  
##### calculate_foot_crossover(landmarks, threshold_proportion) -> FootCrossoverResult  
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
  
---  
  
### Module: hip_drop  
**File**: `runnervision_utils\metrics\rear\lower_body\hip_drop.py`  
  
Calculates hip drop (Trendelenburg gait) during running analysis.
This metric identifies lateral pelvic tilt during single-leg support phases,
which can indicate hip abductor weakness and potential injury risk.  
  
#### Functions  
##### calculate_hip_drop(landmarks, threshold) -> HipDropResult  
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
  
##### analyze_hip_drop_sequence(landmark_sequence, threshold, stance_phases) -> Dict[(str, Any)]  
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
  
---  
  
### Module: knee_alignment  
**File**: `runnervision_utils\metrics\rear\lower_body\knee_alignment.py`  
  
Calculates knee alignment patterns during running to detect valgus (knock-knee) 
or varus (bow-leg) deviations that may indicate injury risk factors.  
  
#### Functions  
##### calculate_knee_alignment(landmarks, threshold) -> KneeAlignmentResult  
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
  
##### analyze_knee_alignment_sequence(landmark_sequence, threshold, stance_phases) -> Dict[(str, Any)]  
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
  
##### get_severity(deviation) -> str  
  
---  
  
### Module: pelvic_tilt  
**File**: `runnervision_utils\metrics\rear\lower_body\pelvic_tilt.py`  
  
Calculates lateral pelvic tilt angle in the frontal plane during running analysis.
This metric identifies pelvic orientation deviations that can indicate hip abductor
weakness, leg length discrepancies, or compensation patterns.  
  
#### Functions  
##### calculate_pelvic_tilt(landmarks, coordinate_system) -> PelvicTiltResult  
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
  
##### analyze_pelvic_tilt_sequence(landmark_sequence, coordinate_system, stance_phases) -> Dict[(str, Any)]  
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
  
---  
  
### Module: arm_swing_mechanics  
**File**: `runnervision_utils\metrics\rear\upper_body\arm_swing_mechanics.py`  
  
Analyzes arm swing mechanics during running from rear view perspective.

Efficient arm swing should move primarily in the sagittal plane, maintain
symmetrical timing and amplitude, preserve ~90° elbow flexion, counter-rotate
with opposite leg, and avoid excessive midline crossing. Poor mechanics can
lead to energy inefficiency and compensatory movement patterns.  
  
#### Functions  
##### calculate_arm_swing_mechanics(landmarks, symmetry_threshold, rotation_threshold) -> ArmSwingResult  
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
  
##### calculate_angle(a, b, c) -> float  
Calculate angle between three points (b is the vertex).  
  
##### classify_arm_symmetry(norm_diff) -> str  
Classify arm height symmetry.  
  
##### classify_elbow_angle(angle) -> str  
Classify elbow angle optimality.  
  
##### classify_shoulder_rotation(norm_diff) -> str  
Classify shoulder rotation stability.  
  
---  
  
### Module: ground_contact_time  
**File**: `runnervision_utils\metrics\side\gait_events_and_phases\ground_contact_time.py`  
  
Analyzes ground contact time and related metrics for running gait analysis.

This module tracks foot contact with the ground over time to calculate:
- Ground contact duration for each foot
- Contact time asymmetry between feet
- Running cadence
- Efficiency ratings based on contact time

Ground contact is determined by analyzing foot height relative to an estimated
ground level over a sliding window of frames.  
  
#### Classes  
##### GroundContactTimeAnalyzer  
Analyzes ground contact time for running gait analysis.

This class maintains internal history of foot positions and contact states
to calculate ground contact metrics over time. It uses a sliding window
approach to estimate ground level and detect contact/lift-off events.

The analyzer tracks:
- Individual foot contact durations
- Contact time asymmetry between feet  
- Running cadence (steps per minute)
- Efficiency ratings based on contact time benchmarks

Attributes:
    frame_rate (float): Video frame rate in fps
    history_size (int): Number of frames to maintain for analysis
    foot_height_threshold (float): Height threshold for contact detection
    frame_count (int): Total frames processed
    step_count (int): Total steps detected across both feet  
  
**Methods**:  
- `__init__(frame_rate, history_size, foot_height_threshold)`  
  - Initialize the ground contact time analyzer.  
- `update(landmarks)` -> GroundContactResult  
  - Update analyzer with new frame data and calculate ground contact metrics.  
- `_estimate_ground_level(foot_heights)` -> float  
  - Estimate ground level from recent foot height positions.  
- `_update_contact_tracking(foot, is_contact)` -> None  
  - Track contact period transitions for each foot.  
- `_calculate_metrics()` -> GroundContactResult  
  - Calculate comprehensive ground contact time metrics.  
- `_assess_efficiency(contact_time_ms)` -> str  
  - Determine running efficiency rating based on ground contact time.  
- `reset()` -> None  
  - Reset the analyzer state for processing a new sequence.  
  
#### Functions  
##### ground_contact_wrapper(landmarks, detector_instance) -> GroundContactResult  
Standalone wrapper function for ground contact analysis.

Provides a functional interface to the GroundContactTimeAnalyzer class
for integration with existing analysis pipelines.

Args:
    analyzer: Initialized GroundContactTimeAnalyzer instance
    landmarks: Dictionary containing pose landmarks
    
Returns:
    GroundContactResult: Analysis results from the analyzer  
  
---  
  
### Module: stance_phase_detector_side  
**File**: `runnervision_utils\metrics\side\gait_events_and_phases\stance_phase_detector_side.py`  
  
Side-view stance phase detection for running biomechanics analysis.

This module provides stance phase detection capabilities for runners captured from 
a side view using MediaPipe BlazePose landmarks. The detector uses ground contact 
proximity and optional velocity analysis to determine when each foot is in contact 
with the ground during the running gait cycle.

Key Features:
- Automatic calibration of ground level and runner dimensions
- Dynamic threshold adjustment based on runner's apparent height
- Confidence scoring for stance phase predictions
- Support for both left and right foot detection
- Robust handling of landmark visibility variations

Biomechanical Context:
- Stance phase: Period when foot is in contact with ground (~40% of gait cycle)
- Flight phase: Period when both feet are airborne (~60% of gait cycle)
- Critical for analyzing ground reaction forces, impact loading, and propulsion  
  
#### Classes  
##### StancePhaseDetectorSide  
Detects stance phase from side-view running footage using MediaPipe BlazePose landmarks.

This detector performs automatic calibration during initial frames to establish ground
level and runner dimensions, then uses proximity-based detection with optional velocity
analysis to determine foot-ground contact phases.

The detection algorithm accounts for:
- Varying runner heights and camera distances through dynamic scaling
- Landmark visibility variations in real-world footage  
- Single vs double stance phases (rare in running but possible at transitions)
- Confidence scoring based on detection certainty

Attributes:
    calibration_frames (int): Number of frames used for initial calibration
    stance_threshold_ratio (float): Stance detection sensitivity as ratio of runner height
    visibility_threshold (float): Minimum landmark visibility score (0.0-1.0)
    ground_level_normalized (Optional[float]): Calibrated ground Y-coordinate
    avg_runner_height_normalized (Optional[float]): Calibrated runner height in normalized coords
    frame_count (int): Current frame number for calibration tracking  
  
**Methods**:  
- `__init__(calibration_frames, stance_threshold_ratio, visibility_threshold)` -> None  
  - Initialize the stance phase detector with calibration parameters.  
- `detect_stance_phase_side(landmarks)` -> StanceResult  
  - Detect stance phase from side-view MediaPipe landmarks.  
- `_collect_calibration_data(landmarks)` -> None  
  - Collect ground level and runner height data during calibration frames.  
- `_finalize_calibration()` -> None  
  - Calculate final calibration values from collected sample data.  
- `_perform_stance_detection(landmarks)` -> StanceResult  
  - Perform stance phase detection using calibrated parameters.  
- `_analyze_foot_stance(landmarks, side, threshold)` -> Dict[(str, Union[str, bool, float])]  
  - Analyze stance phase for a single foot.  
- `_resolve_stance_phase(stance_candidates)` -> StanceResult  
  - Resolve overall stance phase from individual foot analyses.  
- `_get_foot_vertical_velocity(side, current_lowest_y)` -> float  
  - Estimate vertical velocity of foot for enhanced stance detection.  
- `get_calibration_status()` -> CalibrationData  
  - Get current calibration status and parameters.  
- `reset_calibration()` -> None  
  - Reset detector to perform new calibration.  
  
#### Functions  
##### stance_detector_side_wrapper(landmarks, detector_instance) -> StanceResult  
Standalone wrapper function for stance phase detection.

Provides functional interface to StancePhaseDetectorSide class
for integration with existing analysis pipelines.

Args:
    detector: Initialized StancePhaseDetectorVelocity instance
    landmarks: Dictionary containing pose landmarks
    
Returns:
    StanceResult: Analysis results from detector  
  
##### example_usage()  
Demonstrate basic usage of StancePhaseDetectorSide.  
  
---  
  
### Module: stance_phase_detector_velocity  
**File**: `runnervision_utils\metrics\side\gait_events_and_phases\stance_phase_detector_velocity.py`  
  
Detects stance phase in running gait using velocity analysis.

This module analyzes ankle velocities to determine which foot is in stance phase
(ground contact) during running. Stance phase is characterized by lower vertical
velocity and proximity to ground level.

The detector uses a sliding window approach to calculate foot velocities and
combines velocity with position data to identify the supporting foot.  
  
#### Classes  
##### StancePhaseDetectorVelocity  
Detects stance phase in running gait using velocity analysis.

This class maintains a sliding window of ankle positions to calculate
vertical velocities and determine which foot is currently in stance phase.

Stance phase detection combines:
- Vertical velocity analysis (lower velocity indicates ground contact)
- Relative foot height comparison
- Confidence scoring based on velocity differences

The detector is optimized for running gait where clear velocity differences
exist between stance and swing phases.

Attributes:
    frame_rate (float): Video frame rate in fps
    velocity_window (int): Number of frames for velocity calculation
    velocity_threshold (float): Velocity threshold for stance detection  
  
**Methods**:  
- `__init__(frame_rate, velocity_window, velocity_threshold)`  
  - Initialize stance phase detector.  
- `update(landmarks)` -> StancePhaseResult  
  - Update detector with new landmarks and analyze stance phase.  
- `_calculate_velocities()` -> Tuple[(float, float)]  
  - Calculate average vertical velocities for both feet.  
- `_determine_stance_foot(landmarks, left_velocity, right_velocity)` -> Tuple[(str, float)]  
  - Determine which foot is in stance phase and calculate confidence.  
- `reset()` -> None  
  - Reset detector state for processing a new sequence.  
- `set_velocity_threshold(threshold)` -> None  
  - Update velocity threshold for stance detection.  
  
#### Functions  
##### stance_detector_velocity_wrapper(landmarks, detector_instance) -> StancePhaseResult  
Standalone wrapper function for stance phase detection.

Provides functional interface to StancePhaseDetectorVelocity class
for integration with existing analysis pipelines.

Args:
    detector: Initialized StancePhaseDetectorVelocity instance
    landmarks: Dictionary containing pose landmarks
    
Returns:
    StancePhaseResult: Analysis results from detector  
  
---  
  
### Module: vertical_oscillation_analyzer  
**File**: `runnervision_utils\metrics\side\gait_events_and_phases\vertical_oscillation_analyzer.py`  
  
Vertical oscillation analysis for running biomechanics.

This module analyzes the vertical movement of a runner's center of mass (CoM) during
the gait cycle. Vertical oscillation is a key efficiency metric in running performance,
with excessive vertical movement indicating energy waste and potential inefficiency.

Biomechanical Context:
- Optimal vertical oscillation: 6-9 cm for recreational runners
- Elite runners typically show 4-7 cm vertical oscillation
- Excessive oscillation (>12 cm) indicates inefficient running form
- Related to stride mechanics, cadence, and energy expenditure
- Connected to ground contact time and flight phase dynamics

Key Metrics:
- Peak-to-peak vertical displacement of center of mass
- Oscillation frequency (cycles per second)
- Efficiency ratings based on established biomechanical ranges  
  
#### Classes  
##### VerticalOscillationAnalyzer  
Analyzes vertical oscillation patterns in running gait using center of mass tracking.

This analyzer maintains a sliding window of center of mass height measurements to
calculate vertical oscillation metrics including displacement magnitude, frequency,
and efficiency ratings. Uses hip midpoint as CoM approximation, which provides
good correlation with full-body center of mass for running analysis.

Key Features:
- Real-time analysis with configurable sliding window
- Robust frequency detection using multiple methods
- Data quality assessment and filtering
- Biomechanically-informed efficiency ratings
- Outlier detection and smoothing

Attributes:
    frame_rate (float): Video frame rate for temporal calculations
    window_size (int): Number of frames maintained for analysis
    com_heights (deque): Sliding window of center of mass heights
    frame_count (int): Total frames processed
    min_data_points (int): Minimum data required for analysis
    scale_factor (float): Conversion factor to real-world units (cm)  
  
**Methods**:  
- `__init__(frame_rate, window_size, scale_factor, smoothing_enabled)` -> None  
  - Initialize vertical oscillation analyzer.  
- `update(landmarks)` -> OscillationResult  
  - Process new frame and calculate vertical oscillation metrics.  
- `_extract_com_height(landmarks)` -> Optional[float]  
  - Extract center of mass height from hip landmarks.  
- `_is_outlier(new_height)` -> bool  
  - Detect if new measurement is an outlier.  
- `_interpolate_missing_value()` -> float  
  - Interpolate CoM height when outlier detected.  
- `_apply_smoothing(new_height)` -> float  
  - Apply temporal smoothing to reduce measurement noise.  
- `_analyze_oscillation()` -> OscillationResult  
  - Perform comprehensive vertical oscillation analysis.  
- `_calculate_frequency_advanced(heights, avg_height)` -> float  
  - Calculate oscillation frequency using multiple robust methods.  
- `_frequency_from_peaks(heights, avg_height)` -> float  
  - Calculate frequency using peak detection.  
- `_frequency_from_zero_crossings(heights, avg_height)` -> float  
  - Calculate frequency using zero-crossing analysis.  
- `_frequency_from_spectrum(heights)` -> float  
  - Calculate frequency using spectral analysis.  
- `_get_efficiency_rating(oscillation_cm)` -> str  
  - Determine biomechanical efficiency rating.  
- `_assess_data_quality(heights)` -> str  
  - Assess overall quality of the height data.  
- `_calculate_smoothness_score(heights)` -> float  
  - Calculate smoothness score based on second derivative.  
- `_initialize_result()` -> OscillationResult  
  - Initialize default result dictionary.  
- `reset_analysis()` -> None  
  - Reset analyzer for new analysis session.  
- `get_analysis_summary()` -> Dict[(str, Union[str, float, int])]  
  - Get comprehensive analysis summary.  
  
#### Functions  
##### vertical_oscillation_wrapper(landmarks, detector_instance) -> OscillationResult  
Standalone wrapper function for vertical oscillation detection.

Provides functional interface to VerticalOscillationAnalyzer class
for integration with existing analysis pipelines.

Args:
    detector: Initialized VerticalOscillationAnalyzer instance
    landmarks: Dictionary containing pose landmarks
    
Returns:
    OscillationResult: Analysis results from detector  
  
##### example_usage()  
Demonstrate VerticalOscillationAnalyzer usage.  
  
---  
  
### Module: foot_landing  
**File**: `runnervision_utils\metrics\side\lower_body\foot_landing.py`  
  
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
  
#### Functions  
##### calculate_foot_landing_position(landmarks, stance_phase, tolerance_cm, body_scale_factor, use_hip_center) -> FootLandingResult  
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
  
##### _calculate_center_of_mass(landmarks, use_hip_center) -> Optional[float]  
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
  
##### _get_stance_foot_position(landmarks, stance_foot) -> Optional[float]  
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
  
##### _categorize_landing_position(distance_cm, tolerance_cm) -> str  
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
  
---  
  
### Module: foot_strike  
**File**: `runnervision_utils\metrics\side\lower_body\foot_strike.py`  
  
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
  
#### Functions  
##### calculate_foot_strike(landmarks, stance_phase, vertical_threshold_heel, vertical_threshold_forefoot, confidence_scaling_heel, confidence_scaling_forefoot, angle_contradiction_penalty) -> FootStrikeResult  
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
  
##### _calculate_foot_angle(heel_x, heel_y, toe_x, toe_y) -> float  
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
  
##### _calculate_ankle_angle(ankle_x, ankle_y, heel_x, heel_y) -> float  
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
  
##### _classify_strike_pattern(vertical_difference, heel_threshold, forefoot_threshold, heel_scaling, forefoot_scaling) -> Tuple[(str, float)]  
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
  
##### _apply_angle_validation(strike_pattern, foot_angle, confidence, penalty) -> float  
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
  
##### _classify_landing_stiffness(ankle_angle) -> str  
Classify landing stiffness based on ankle angle at foot contact.

Parameters:
-----------
ankle_angle : float
    Ankle angle in degrees
    
Returns:
--------
str
    Landing stiffness classification: 'stiff', 'moderate', or 'compliant'  
  
---  
  
### Module: knee_angle  
**File**: `runnervision_utils\metrics\side\lower_body\knee_angle.py`  
  
Calculates knee flexion/extension angles from pose landmarks.

This module analyzes knee joint angles to assess running biomechanics,
identify potential overstriding, and evaluate leg extension patterns
that may impact performance or injury risk.  
  
#### Functions  
##### calculate_knee_angle(landmarks, side) -> KneeAngleResult  
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
  
##### _calculate_joint_angle(proximal, joint, distal) -> Optional[float]  
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
  
##### _assess_leg_extension(knee_angle) -> str  
Provide qualitative assessment of leg extension based on knee angle.

Parameters:
-----------
knee_angle : float
    Knee extension angle in degrees
    
Returns:
--------
str
    Qualitative assessment of leg extension  
  
##### calculate_bilateral_knee_angles(landmarks) -> Dict[(str, KneeAngleResult)]  
Calculate knee angles for both legs simultaneously.

Parameters:
-----------
landmarks : LandmarksDict
    Dictionary containing pose landmarks for both legs
    
Returns:
--------
Dict[str, KneeAngleResult]
    Dictionary with 'left' and 'right' keys containing respective knee analyses  
  
##### analyze_knee_asymmetry(left_result, right_result) -> Dict[(str, Any)]  
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
  
---  
  
### Module: stride_length  
**File**: `runnervision_utils\metrics\side\lower_body\stride_length.py`  
  
Estimates stride length and related biomechanical metrics for running analysis.

This module calculates stride length using temporal tracking of foot positions,
optimized for side-view analysis. It provides both instantaneous estimates and
smoothed temporal measurements with confidence scoring.  
  
#### Functions  
##### estimate_stride_length(landmarks, frame_index, height_cm, temporal_tracking, fps, threshold_proportion) -> Tuple[(StrideAnalysisResult, Optional[TemporalTrackingData])]  
Estimate stride length based on foot positions with temporal tracking,
optimized for side-view analysis.

This function analyzes foot movement patterns to detect touchdown events
and calculate stride metrics. It uses both instantaneous estimates from
current pose and temporal tracking for improved accuracy.

Parameters:
-----------
landmarks : LandmarksDict
    Dictionary containing pose landmarks with (x, y, z, visibility) coordinates.
    Required keys: 'right_foot_index', 'right_ankle', 'right_knee', 'right_hip'.
    Optional keys: 'right_shoulder', 'nose', 'right_ear', 'right_eye' for height estimation.
    Coordinates are typically normalized (0.0-1.0).
    
frame_index : Optional[int], default=None
    Current frame index for temporal tracking. If None, only instantaneous
    estimates will be calculated.
    
height_cm : Optional[float], default=None
    Runner's height in centimeters. If provided, improves scale factor accuracy
    and enables normalized stride length calculation.
    
temporal_tracking : Optional[TemporalTrackingData], default=None
    Previous temporal tracking data to maintain state across frames.
    If None, new tracking data will be initialized.
    
fps : float, default=30.0
    Video frame rate in frames per second, used for timing calculations.
    
threshold_proportion : float, default=0.25
    Velocity threshold proportion for touchdown detection. Lower values
    make touchdown detection more sensitive.

Returns:
--------
Tuple[StrideAnalysisResult, Optional[TemporalTrackingData]]
    A tuple containing:
    - StrideAnalysisResult: Dictionary with stride analysis metrics
    - TemporalTrackingData: Updated temporal tracking data (None if frame_index is None)
    
StrideAnalysisResult contains:
    - "instantaneous_estimate_cm" (Optional[float]): Current stride estimate from pose
    - "stride_length_cm" (Optional[float]): Smoothed stride length from temporal tracking
    - "normalized_stride_length" (Optional[float]): Stride length relative to height (0.0-2.0 typical range)
    - "stride_frequency" (Optional[float]): Cadence in strides per minute
    - "assessment" (Optional[str]): Stride appropriateness ("optimal", "too_short", "too_long", "insufficient_data")
    - "confidence" (float): Measurement confidence (0.0-1.0)
    - "calculation_successful" (bool): True if basic calculations completed  
  
##### _calculate_side_view_scale_factor(landmarks, height_cm) -> float  
Calculate scale factor for converting normalized coordinates to centimeters.

Parameters:
-----------
landmarks : LandmarksDict
    Dictionary containing pose landmarks
height_cm : Optional[float]
    Known runner height in cm for accurate scaling
    
Returns:
--------
float
    Scale factor for coordinate conversion  
  
##### _initialize_or_update_temporal_tracking(landmarks, frame_index, temporal_tracking, fps, threshold_proportion, scale_factor) -> TemporalTrackingData  
Initialize or update temporal tracking data for stride detection.

Parameters:
-----------
landmarks : LandmarksDict
    Current frame landmarks
frame_index : int
    Current frame number
temporal_tracking : Optional[TemporalTrackingData]
    Previous tracking data
fps : float
    Video frame rate
threshold_proportion : float
    Velocity threshold for touchdown detection
scale_factor : float
    Coordinate to centimeter conversion factor
    
Returns:
--------
TemporalTrackingData
    Updated temporal tracking data  
  
##### _detect_foot_touchdown(temporal_tracking, current_foot_pos, threshold_proportion) -> bool  
Detect foot touchdown based on position and velocity patterns.

Parameters:
-----------
temporal_tracking : TemporalTrackingData
    Current tracking data
current_foot_pos : np.ndarray
    Current foot position [x, y]
threshold_proportion : float
    Velocity threshold proportion
    
Returns:
--------
bool
    True if touchdown detected  
  
##### _update_stride_metrics(temporal_tracking, scale_factor, fps) -> None  
Update stride length and timing metrics based on recent touchdown events.

Parameters:
-----------
temporal_tracking : TemporalTrackingData
    Tracking data to update
scale_factor : float
    Coordinate to centimeter conversion factor
fps : float
    Video frame rate  
  
##### _calculate_instantaneous_stride_estimate(landmarks, scale_factor) -> float  
Calculate instantaneous stride estimate from current pose.

Parameters:
-----------
landmarks : LandmarksDict
    Current pose landmarks
scale_factor : float
    Coordinate to centimeter conversion factor
    
Returns:
--------
float
    Instantaneous stride estimate in centimeters  
  
##### _estimate_height_from_side_view(landmarks, scale_factor) -> Optional[float]  
Estimate runner's height from side-view landmarks.

Parameters:
-----------
landmarks : LandmarksDict
    Pose landmarks
scale_factor : float
    Coordinate to centimeter conversion factor
    
Returns:
--------
Optional[float]
    Estimated height in cm, or None if estimation fails  
  
##### _assess_stride_length(normalized_stride_length) -> str  
Assess stride length appropriateness based on normalized value.

Parameters:
-----------
normalized_stride_length : float
    Stride length as proportion of height
    
Returns:
--------
str
    Assessment: "optimal", "too_short", or "too_long"  
  
---  
  
### Module: trunk_angle  
**File**: `runnervision_utils\metrics\side\lower_body\trunk_angle.py`  
  
Calculates trunk forward lean angle for running biomechanics analysis.

This metric measures the forward lean of the runner's trunk relative to vertical,
which is crucial for efficient running mechanics. Optimal forward lean (5-10°) 
helps with momentum and reduces braking forces, while excessive or insufficient 
lean can lead to inefficiencies and injury risk.  
  
#### Functions  
##### calculate_trunk_angle(landmarks, smoothing_factor, previous_angle, estimated_speed_mps) -> TrunkAngleResult  
Calculate trunk forward lean angle relative to vertical from side view.

The trunk angle is calculated using hip and shoulder midpoints to determine
the trunk's deviation from vertical. Positive angles indicate forward lean,
negative angles indicate backward lean. Temporal smoothing is applied to
reduce noise between frames.

Parameters:
-----------
landmarks : LandmarksDict
    Dictionary containing 3D coordinates and visibility of pose landmarks.
    Required keys: 'left_hip', 'right_hip', 'left_shoulder', 'right_shoulder'
    Optional keys: 'neck' (for improved upper trunk representation)
    Each landmark is a tuple: (x, y, z, visibility)
    
smoothing_factor : float, default=0.3
    Temporal smoothing factor (0.0-1.0). Higher values provide more smoothing
    but slower response to actual changes. 0.0 = no smoothing, 1.0 = maximum smoothing.
    
previous_angle : Optional[float], default=None
    Previous frame's trunk angle for temporal smoothing. If None, no smoothing applied.
    
estimated_speed_mps : Optional[float], default=None
    Estimated running speed in meters per second. Used to adjust optimal range
    recommendations based on running pace.

Returns:
--------
TrunkAngleResult
    Dictionary containing:
    - "angle_degrees" (Optional[float]): Forward lean angle in degrees. 
        Positive = forward lean, negative = backward lean. None if calculation fails.
    - "is_optimal" (Optional[bool]): True if angle is within optimal range (5-10°). 
        None if calculation fails.
    - "assessment" (Optional[str]): Categorical assessment of trunk position.
        Values: 'backward_lean', 'insufficient_forward_lean', 'optimal_forward_lean',
        'moderate_forward_lean', 'excessive_forward_lean'. None if calculation fails.
    - "assessment_detail" (Optional[str]): Detailed explanation and recommendations.
        None if calculation fails.
    - "confidence" (Optional[float]): Confidence score (0.0-1.0) based on landmark
        visibility and measurement reliability. None if calculation fails.
    - "calculation_successful" (bool): True if metrics were calculated successfully.  
  
##### _calculate_trunk_confidence(landmarks, current_angle, previous_angle) -> float  
Calculate confidence score for trunk angle measurement.

Parameters:
-----------
landmarks : LandmarksDict
    Dictionary containing pose landmarks with visibility scores
current_angle : float
    Current calculated trunk angle in degrees
previous_angle : Optional[float]
    Previous frame's angle for temporal consistency check
    
Returns:
--------
float
    Confidence score between 0.0 and 1.0  
  
##### _is_trunk_potentially_occluded(landmarks) -> bool  
Check if trunk landmarks might be occluded by arms or other factors.

Parameters:
-----------
landmarks : LandmarksDict
    Dictionary containing pose landmarks
    
Returns:
--------
bool
    True if occlusion is likely detected  
  
##### _generate_assessment_detail(assessment, angle_degrees, estimated_speed_mps) -> str  
Generate detailed assessment text with recommendations.

Parameters:
-----------
assessment : str
    Categorical assessment of trunk position
angle_degrees : float
    Calculated trunk angle in degrees
estimated_speed_mps : Optional[float]
    Estimated running speed for context-specific advice
    
Returns:
--------
str
    Detailed assessment with recommendations  
  
---  
  
### Module: arm_carriage  
**File**: `runnervision_utils\metrics\side\upper_body\arm_carriage.py`  
  
Analyzes running arm carriage and swing mechanics from pose landmarks.

This module evaluates arm position, elbow angle, hand placement, and swing amplitude
to identify biomechanical inefficiencies that may impact running performance or 
increase injury risk. Analysis focuses on the right arm when visible from right-side view.  
  
#### Classes  
##### ArmCarriageAnalyzer  
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
  
**Methods**:  
- `__init__(frame_rate, history_size)`  
  - Initialize the arm carriage analyzer with specified parameters.  
- `update(landmarks)` -> ArmCarriageResult  
  - Process new frame data and update arm carriage analysis.  
- `_has_required_landmarks(landmarks)` -> bool  
  - Check if required landmarks are present for analysis.  
- `_store_frame_data(landmarks)` -> None  
  - Store current frame arm position data for historical analysis.  
- `get_analysis_summary()` -> str  
  - Generate a comprehensive text summary of current arm carriage state.  
- `reset()` -> None  
  - Reset analyzer state and clear all historical data.  
  
#### Functions  
##### calculate_arm_carriage(landmarks, frame_rate, arm_history) -> ArmCarriageResult  
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
  
##### _calculate_upper_arm_angle(shoulder, elbow) -> float  
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
  
##### _calculate_elbow_angle(shoulder, elbow, wrist) -> float  
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
  
##### _analyze_hand_position(landmarks, shoulder, wrist) -> str  
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
  
##### _calculate_swing_amplitude(arm_history) -> Optional[float]  
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
  
##### _generate_assessment(upper_arm_angle, elbow_angle, hand_position, swing_amplitude, crosses_midline) -> Tuple[(str, List[str])]  
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
  
---  
  
### Module: base_report_generator  
**File**: `runnervision_utils\reports\report_generators\base_report_generator.py`  
  
#### Classes  
##### BaseReportGenerator  
**Methods**:  
- `__init__(metrics_df, session_id, reports_dir, metadata, report_file_path)`  
- `_add_html_head(html_content, report_title)`  
- `_add_metric_box(html_content, title, value_str, unit, std_dev_str, rating_text, rating_key, progress_percent, sub_text)`  
  - Generates the HTML for a single metric box.  
- `_get_series_stats(df, col_name, drop_na_val)`  
- `_generate_session_info_section(html_content)`  
- `_generate_main_report_structure(html_content, report_title_suffix)`  
- `generate_html_file(output_filename_base)`  
  - Generates the HTML report and saves it to a file.  
- `view_name()`  
- `_generate_metrics_summary_section(html_content)`  
- `_generate_specialized_sections(html_content, summary_data)`  
  - Hook for view-specific sections beyond basic summary and plots.  
- `_generate_plots_section(html_content)`  
- `_generate_recommendations_section(html_content)`  
- `_generate_overall_assessment_section(html_content, summary_data)`  
  
---  
  
### Module: rear_report_generator  
**File**: `runnervision_utils\reports\report_generators\rear_report_details\rear_report_generator.py`  
  
#### Classes  
##### RearViewReportGenerator  
**Methods**:  
- `view_name()`  
- `_generate_metrics_summary_section(html_content)`  
- `_generate_specialized_sections(html_content, summary_data)`  
  - No specialized sections for rear view in this version.  
- `_generate_recommendations_section(html_content)`  
  - Generates recommendations using the Language Model.  
- `_generate_overall_assessment_section(html_content, summary_data)`  
  - Overall assessment can be integrated or removed as needed.  
- `_generate_plots_section(html_content)`  
- `_save_rear_metric_plots()`  
  - Create and save plots of running metrics.  
  
#### Functions  
##### add_summary_metric(col_name, title, unit, val_format, is_categorical)  
  
---  
  
### Module: side_report_generator  
**File**: `runnervision_utils\reports\report_generators\side_report_details\side_report_generator.py`  
  
#### Classes  
##### SideViewReportGenerator  
**Methods**:  
- `view_name()`  
- `_generate_metrics_summary_section(html_content)`  
- `_generate_specialized_sections(html_content, summary_data)`  
- `_generate_bilateral_comparison_section(html_content, summary_data)`  
- `_generate_plots_section(html_content)`  
- `_save_side_metric_plots()`  
- `_generate_recommendations_section(html_content)`  
- `_generate_overall_assessment_section(html_content, summary_data)`  
  
#### Functions  
##### add_summary_metric(col_name, title, unit, val_format, is_categorical)  
  
---  
  
