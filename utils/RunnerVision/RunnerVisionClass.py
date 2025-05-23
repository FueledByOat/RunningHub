# RunnverVision.py
"""
Core implementation for RunnerVision using BlazePose for runner biomechanics analysis
"""
import os
import math
from datetime import datetime
import argparse
import csv
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque


class RunnerVisionAnalyzer:
    def __init__(self):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # Use highest accuracy model
            enable_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Define key pose landmarks for running analysis
        self.key_points = {
            'left_ear': self.mp_pose.PoseLandmark.LEFT_EAR,
            'right_ear': self.mp_pose.PoseLandmark.RIGHT_EAR,
            'nose': self.mp_pose.PoseLandmark.NOSE,
            'left_shoulder': self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            'right_shoulder': self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            'left_elbow': self.mp_pose.PoseLandmark.LEFT_ELBOW,
            'right_elbow': self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            'left_wrist': self.mp_pose.PoseLandmark.LEFT_WRIST, 
            'right_wrist': self.mp_pose.PoseLandmark.RIGHT_WRIST,
            'left_hip': self.mp_pose.PoseLandmark.LEFT_HIP,
            'right_hip': self.mp_pose.PoseLandmark.RIGHT_HIP,
            'left_knee': self.mp_pose.PoseLandmark.LEFT_KNEE,
            'right_knee': self.mp_pose.PoseLandmark.RIGHT_KNEE,
            'left_ankle': self.mp_pose.PoseLandmark.LEFT_ANKLE,
            'right_ankle': self.mp_pose.PoseLandmark.RIGHT_ANKLE,
            'left_heel': self.mp_pose.PoseLandmark.LEFT_HEEL,
            'right_heel': self.mp_pose.PoseLandmark.RIGHT_HEEL,
            'left_foot_index': self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
            'right_foot_index': self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
        }
        
        # Side Metrics storage
        self.side_metrics = {
            'timestamp': [],
            'frame_number': [],
            'strike_pattern': [],  # heel, midfoot, forefoot
            'strike_confidence' : [],
            'vertical_difference' : [],
            'strike_foot_angle' : [],
            'strike_ankle_angle' : [],
            'strike_landing_stiffness' : [],
            'foot_landing_position_category': [],  # cm relative to center of mass
            'foot_landing_distance_from_center_in_cm' : [],
            'foot_landing_is_under_center_of_mass' : [],
            'trunk_angle_degrees': [],  # degrees
            'trunk_angle_is_optimal' : [],
            'trunk_angle_assessment' : [],
            'trunk_angle_assessment_detail' : [],
            'trunk_angle_confidence' : [],
            'upper_arm_angle' : [],
            'elbow_angle' : [],
            'hand_position' : [],
            'arm_swing_amplitude' : [],
            'arm_swing_symmetry' : [],
            'arm_swing_overall_assessment' : [],
            'arm_swing_recommendations' : [],
            'knee_angle_left': [],  # degrees
            'knee_angle_right': [],  # degrees
            'stride_instantaneous_estimate_cm' : [],
            'stride_length_cm': [],  # estimated in cm
            'normalized_stride_length' : [],
            'stride_frequency' : [],
            'stride_assessment' : [],
            'stride_confidence' : [],
            'stance_phase_detected': [],  # boolean for whether foot is on ground
            'stance_foot' : [],
            'stance_confidence' : [],
            # 'ground_contact_time' : [],
            # 'flight_time' : [],
            # 'duty_factor' : [],
            # 'peak_knee_flexion_during_stance' : [],
            # 'knee_flexion_at_foot_strike' : [],
            # 'hip_extension_at_toe_off' : [],
            # 'anke_dorsiflextion_mid_stance' : [],
            # 'saggital_pelvic_rotation' : [],
            # 'ship_ankle_at_foot_strike' : [],
        }
        # Rear Metrics storage
        self.rear_metrics = {
            'timestamp': [],
            'frame_number': [],
            'left_foot_crossover': [],  # Boolean for left foot crossing the midline
            'right_foot_crossover': [],  # Boolean for right foot crossing the midline
            'left_distance_from_midline': [],  # unitless
            'right_distance_from_midline': [], # unitless
            'hip_drop_value': [],  # unitless, can be used to determine severity
            'hip_drop_direction': [],  # "left" / "right" / "neutral"
            'hip_drop_severity' : [], # details in function
            'pelvic_tilt_angle': [],  # Degrees (positive = right hip lower, negative = left hip lower)
            'pelvic_tilt_elevated_side' : [],
            'pelvic_tilt_severity' : [],
            'pelvic_tilt_normalized' : [],
            'left_knee_valgus': [],  # boolean indicator of inward alignment
            'left_knee_varus' : [],
            'right_knee_valgus': [],  # boolean indicator of inward alignment
            'right_knee_varus' : [],
            'left_knee_normalized_deviation' : [],
            'right_knee_normalized_deviation' : [],
            'knee_severity_left' : [],
            'knee_severity_right' : [],
            'left_ankle_inversion_value': [],  # Normalized X-difference (heel - ankle), unitless
            'right_ankle_inversion_value': [],  # Normalized X-difference (heel - ankle), unitless
            'left_ankle_normalized_value' : [],
            'right_ankle_normalized_value' : [],            
            'left_ankle_pattern' : [],           
            'right_ankle_pattern' : [],
            'left_ankle_severity' : [],             
            'right_ankle_severity' : [],
            'left_ankle_angle' : [],             
            'right_ankle_angle' : [],
            'step_width': [],  # Approx. cm (if image width is normalized to body width ~1 meter)
            'symmetry': [],  # Unitless ratio (0 = perfect symmetry; positive = right dominant)
            'vertical_elbow_diff' : [],
            'normalized_vertical_diff' : [],                   
            'left_elbow_angle' : [],
            'right_elbow_angle' : [],     
            'normalized_shoulder_diff' : [],
            'normalized_shoulder_width' : [],     
            'arm_height_symmetry' : [],
            'elbow_angle_left' : [],     
            'elbow_angle_right' : [],
            'left_wrist_crossover' : [],   
            'right_wrist_crossover' : [],
            'shoulder_rotation' : [],   
            'stance_phase_detected': [],  # boolean for whether foot is on ground
            'stance_foot' : [],
            'stance_confidence' : []                                    
        }
        
    def process_video(self, video_path, output_path="videos"): 
        """Process video and extract running biomechanics data."""
        side = True if 'side' in video_path.lower() else False
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Set up output video if needed
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        frame_count = 0
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
                
            # Convert BGR image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process image with BlazePose
            results = self.pose.process(image_rgb)
            
            # Skip if no pose detected
            if not results.pose_landmarks:
                continue
            
            # Extract running metrics
            if side:
                # print('extracting side metrics')
                self.extract_side_metrics(results.pose_landmarks, frame_count, cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0)
            else:
                # print('extracting rear metrics')
                self.extract_rear_metrics(results.pose_landmarks, frame_count, cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0)

            # Draw pose landmarks and metrics on image
            if side:
                # print('drawing side image')
                annotated_image = self.draw_side_analysis(image.copy(), results.pose_landmarks, frame_count)
            else:
                # print('drawing rear image')
                annotated_image = self.draw_rear_analysis(image.copy(), results.pose_landmarks, frame_count)


            if output_path:
                out.write(annotated_image)
                
            frame_count += 1
            
            # Display in-progress frames (comment out for faster processing)
            # cv2.imshow('RunnerVision Analysis', annotated_image)
            # if cv2.waitKey(5) & 0xFF == 27:  # ESC to exit
            #     break
                
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        # Convert metrics to DataFrame for analysis
        if side:
            # for i, thing in self.side_metrics.items():
            #     print(i, len(thing))
            #     for item in thing:
            #         print(item)
            #         break
            return pd.DataFrame(self.side_metrics)
        else:
            return pd.DataFrame(self.rear_metrics)
    
    def extract_side_metrics(self, landmarks, frame_number, timestamp):
        """Extract running biomechanics metrics from a single frame
        for a video shot from the side of a runner, AKA the sagittal plane."""

        # Get normalized landmark positions
        landmark_coords = {}
        for name, landmark_id in self.key_points.items():
            landmark = landmarks.landmark[landmark_id]
            landmark_coords[name] = (landmark.x, landmark.y, landmark.z, landmark.visibility)
        
        # Instantiate Stance Class
        # Detect stance phase
        stance_phase = self.detect_stance_phase_side(landmark_coords)
        self.side_metrics['stance_phase_detected'].append(stance_phase['is_stance_phase'])
        self.side_metrics['stance_foot'].append(stance_phase['stance_foot'])
        self.side_metrics['stance_confidence'].append(stance_phase['confidence'])

        # Store basic timestamp data
        self.side_metrics['timestamp'].append(timestamp)
        self.side_metrics['frame_number'].append(frame_number)
        
        # Calculate foot strike metrics
        foot_strike = self.calculate_foot_strike(landmark_coords, stance_phase = stance_phase)
        self.side_metrics['strike_pattern'].append(foot_strike['strike_pattern'])
        self.side_metrics['strike_confidence'].append(foot_strike['confidence'])
        self.side_metrics['vertical_difference'].append(foot_strike['vertical_difference'])
        self.side_metrics['strike_foot_angle'].append(foot_strike['foot_angle'])
        self.side_metrics['strike_ankle_angle'].append(foot_strike['ankle_angle'])
        self.side_metrics['strike_landing_stiffness'].append(foot_strike['landing_stiffness'])
        
        # Calculate foot landing position relative to center of mass
        foot_position = self.calculate_foot_landing_position(landmark_coords, stance_phase = stance_phase)
        self.side_metrics['foot_landing_position_category'].append(foot_position['position_category'])
        self.side_metrics['foot_landing_distance_from_center_in_cm'].append(foot_position['distance_cm'])
        self.side_metrics['foot_landing_is_under_center_of_mass'].append(foot_position['is_under_com'])

        # Calculate trunk angle / trunk lean
        trunk_angle = self.calculate_trunk_angle(landmark_coords)
        self.side_metrics['trunk_angle_degrees'].append(trunk_angle['angle_degrees'])
        self.side_metrics['trunk_angle_is_optimal'].append(trunk_angle['is_optimal'])
        self.side_metrics['trunk_angle_assessment'].append(trunk_angle['assessment'])
        self.side_metrics['trunk_angle_assessment_detail'].append(trunk_angle['assessment_detail'])
        self.side_metrics['trunk_angle_confidence'].append(trunk_angle['confidence'])
        
        # Calculate arm carriage
        arm_angle = self.analyze_arm_carriage(landmark_coords)
        self.side_metrics['upper_arm_angle'].append(arm_angle['upper_arm_angle'])
        self.side_metrics['elbow_angle'].append(arm_angle['elbow_angle'])
        self.side_metrics['hand_position'].append(arm_angle['hand_position'])
        self.side_metrics['arm_swing_amplitude'].append(arm_angle['arm_swing_amplitude'])
        self.side_metrics['arm_swing_symmetry'].append(arm_angle['arm_swing_symmetry'])
        self.side_metrics['arm_swing_overall_assessment'].append(arm_angle['overall_assessment'])
        self.side_metrics['arm_swing_recommendations'].append(arm_angle['recommendations'])
        
        # Calculate knee angle
        knee_angle_right = self.calculate_knee_angle(landmark_coords, 'right')
        knee_angle_left = self.calculate_knee_angle(landmark_coords, 'left')
        self.side_metrics['knee_angle_left'].append(knee_angle_left)
        self.side_metrics['knee_angle_right'].append(knee_angle_right)

        # Estimate stride length
        stride_length = self.estimate_stride_length(landmark_coords)
        self.side_metrics['stride_instantaneous_estimate_cm'].append(stride_length['instantaneous_estimate_cm'])
        self.side_metrics['stride_length_cm'].append(stride_length['stride_length_cm'])
        self.side_metrics['normalized_stride_length'].append(stride_length['normalized_stride_length'])
        self.side_metrics['stride_frequency'].append(stride_length['stride_frequency'])
        self.side_metrics['stride_assessment'].append(stride_length['assessment'])
        self.side_metrics['stride_confidence'].append(stride_length['confidence'])




    def calculate_foot_strike(self, landmarks, stance_phase):
        """
        Determine foot strike pattern (heel, midfoot, forefoot) from side view.
        
        Foot strike patterns affect running biomechanics:
        - Heel strike: Common but may increase impact forces (higher loading rates)
        - Midfoot: Often balances impact absorption and propulsion efficiency
        - Forefoot: May reduce impact forces but increases calf/Achilles loading
        
        Parameters:
        -----------
        landmarks : dict
            Dictionary containing body landmark coordinates
        
        Returns:
        --------
        dict
            Analysis of foot strike pattern with classification and measurements
            "strike_pattern": strike_pattern,
            "confidence": confidence,
            "vertical_difference": vertical_difference,
            "foot_angle": foot_angle,
            "ankle_angle": ankle_angle,
            "landing_stiffness": stiffness
        """

        # Check if in stance phase
        stance_info = stance_phase
        
        if not stance_info['is_stance_phase']:
            return {
                "strike_pattern": 'not_applicable',
                "confidence": 0,
                "vertical_difference": 0,
                "foot_angle": 0,
                "ankle_angle": 0,
                "landing_stiffness": 'not_applicable'
        }

        # Extract relevant landmarks
        right_heel_y = landmarks['right_heel'][1]
        right_foot_index_y = landmarks['right_foot_index'][1]
        # Include ankle for calculating foot angle
        right_ankle_y = landmarks['right_ankle'][1]
        right_ankle_x = landmarks['right_ankle'][0]
        right_heel_x = landmarks['right_heel'][0]
        right_foot_index_x = landmarks['right_foot_index'][0]
        
        # Vertical difference between heel and toe (primary indicator)
        # Note: In most coordinate systems, higher Y value = lower position in image
        vertical_difference = right_heel_y - right_foot_index_y
        
        # Calculate foot angle relative to ground for additional context
        # A positive angle means heel is lower than toe (dorsiflexion)
        # A negative angle means toe is lower than heel (plantarflexion)
        dx_heel_toe = right_foot_index_x - right_heel_x
        dy_heel_toe = right_foot_index_y - right_heel_y
        
        if dx_heel_toe != 0:  # Avoid division by zero
            foot_angle = np.degrees(np.arctan2(dy_heel_toe, dx_heel_toe))
        else:
            foot_angle = 0
        
        # Calculate ankle angle (dorsiflexion/plantarflexion)
        dx_ankle_heel = right_heel_x - right_ankle_x
        dy_ankle_heel = right_heel_y - right_ankle_y
        
        if dx_ankle_heel != 0:  # Avoid division by zero
            ankle_angle = np.degrees(np.arctan2(dy_ankle_heel, dx_ankle_heel))
        else:
            ankle_angle = 0
        
        # Determine strike pattern with refined thresholds
        # Use multiple metrics for more robust classification
        if vertical_difference > 0.015:  # Heel significantly lower than toe
            strike_pattern = "heel"
            confidence = min(1.0, vertical_difference / 0.03)  # Scale confidence
        elif vertical_difference < -0.01:  # Toe significantly lower than heel
            strike_pattern = "forefoot"
            confidence = min(1.0, abs(vertical_difference) / 0.02)
        else:
            strike_pattern = "midfoot"
            confidence = 1.0 - min(1.0, abs(vertical_difference) / 0.015)
        
        # Adjust confidence based on foot angle as secondary validation
        if strike_pattern == "heel" and foot_angle < -5:
            # Contradicting signals
            confidence *= 0.7
        elif strike_pattern == "forefoot" and foot_angle > 5:
            # Contradicting signals
            confidence *= 0.7
        
        # Classify landing stiffness based on ankle angle
        # Stiff landing = more vertical shin angle at contact
        stiffness = "moderate"
        if ankle_angle < -15:
            stiffness = "stiff"
        elif ankle_angle > 5:
            stiffness = "compliant"
        
        return {
            "strike_pattern": strike_pattern,
            "confidence": confidence,
            "vertical_difference": vertical_difference,
            "foot_angle": foot_angle,
            "ankle_angle": ankle_angle,
            "landing_stiffness": stiffness
        }
    
    def calculate_foot_landing_position(self, landmarks, stance_phase, tolerance_cm=5.0):
        """
        Calculate horizontal distance from foot landing to center of mass.
        
        Args:
            landmarks: Dictionary containing pose landmarks with x,y coordinates
            tolerance_cm: Tolerance in cm for considering a foot "under" center of mass
            
        Returns:
            dict: Contains:
                - 'distance_cm': Horizontal distance in cm (negative = behind CoM, positive = ahead of CoM)
                - 'is_under_com': Boolean indicating if foot is landing under CoM within tolerance
                - 'position_category': Categorical assessment ('under', 'ahead', 'behind')
        """
        # Center of mass approximation - using hip midpoint
        # More accurate CoM would include upper body contribution
        hip_center_x = (landmarks['left_hip'][0] + landmarks['right_hip'][0]) / 2
        
        # Check if in stance phase
        stance_info = stance_phase
        
        if not stance_info['is_stance_phase']:
            return {
                'distance_cm': 0.0,
                'is_under_com': False,
                'position_category': 'not_applicable'
            }
        
        # Use the foot identified as being in stance phase
        foot_x = landmarks[stance_info['stance_foot'] + '_ankle'][0]
        
        # Calculate horizontal distance (converted to cm based on body proportions)
        # Assuming the coordinate system is normalized by height or has known scale
        distance_cm = (foot_x - hip_center_x) * 100
        
        # Determine if foot is landing under center of mass within tolerance
        is_under_com = abs(distance_cm) <= tolerance_cm
        
        # Categorize the landing position
        if is_under_com:
            position_category = 'under'
        elif distance_cm > tolerance_cm:
            position_category = 'ahead'  # Foot is ahead of center of mass (overstriding)
        else:
            position_category = 'behind'  # Foot is behind center of mass
        
        return {
            'distance_cm': distance_cm,
            'is_under_com': is_under_com,
            'position_category': position_category
        }

    
    def calculate_trunk_angle(self, landmarks, smoothing_factor=0.3):
        """
        Calculate trunk forward lean angle relative to vertical.
        
        Args:
            landmarks: Dictionary containing pose landmarks with x,y coordinates
            smoothing_factor: Factor for temporal smoothing (0-1, higher = more smoothing)
            
        Returns:
            dict: Contains:
                - 'angle_degrees': Forward lean angle in degrees
                - 'is_optimal': Boolean indicating if angle is within optimal range
                - 'assessment': Categorical assessment of trunk position
                - 'confidence': Confidence score for the measurement
        """
        # For more accurate trunk angle, we can use multiple points
        # Hip midpoint
        hip_center_x = (landmarks['left_hip'][0] + landmarks['right_hip'][0]) / 2
        hip_center_y = (landmarks['left_hip'][1] + landmarks['right_hip'][1]) / 2
        
        # Shoulder midpoint
        shoulder_center_x = (landmarks['left_shoulder'][0] + landmarks['right_shoulder'][0]) / 2
        shoulder_center_y = (landmarks['left_shoulder'][1] + landmarks['right_shoulder'][1]) / 2
        
        # Optional: Include neck point for more accurate trunk line
        if 'neck' in landmarks:
            neck_x = landmarks['neck'][0]
            neck_y = landmarks['neck'][1]
            
            # Average the shoulder midpoint and neck for upper trunk representation
            upper_trunk_x = (shoulder_center_x + neck_x) / 2
            upper_trunk_y = (shoulder_center_y + neck_y) / 2
        else:
            # Use shoulder midpoint if neck isn't available
            upper_trunk_x = shoulder_center_x
            upper_trunk_y = shoulder_center_y
        
        # Calculate trunk vector
        trunk_vector_x = upper_trunk_x - hip_center_x
        trunk_vector_y = hip_center_y - upper_trunk_y  # Note: y-axis is typically inverted in image coordinates
        
        # Calculate angle with vertical (y-axis)
        # arctan2 gives angle in range (-π, π) radians
        angle = np.degrees(np.arctan2(trunk_vector_x, trunk_vector_y))
        
        # Apply temporal smoothing if previous measurements exist
        if hasattr(self, 'previous_trunk_angle'):
            angle = (self.previous_trunk_angle * smoothing_factor) + (angle * (1 - smoothing_factor))
        
        # Store for next frame's smoothing
        self.previous_trunk_angle = angle
        
        # Ensure angle is positive for forward lean
        # Negative values would represent backward lean
        # Note: May need to adjust based on coordinate system
        
        # Calculate confidence based on landmark visibility
        # This depends on how landmark confidence is provided in your implementation
        landmark_confidence = self._estimate_trunk_landmark_confidence(landmarks)
        
        # Define optimal trunk lean range (based on biomechanical research)
        # Research shows 5-10° is optimal for most runners
        min_optimal = 5.0  # degrees
        max_optimal = 10.0  # degrees
        
        # Assess trunk position
        is_optimal = min_optimal <= angle <= max_optimal
        
        if angle < 0:
            assessment = "backward_lean"
        elif angle < min_optimal:
            assessment = "insufficient_forward_lean"
        elif angle <= max_optimal:
            assessment = "optimal_forward_lean"
        elif angle <= 15:
            assessment = "moderate_forward_lean"
        else:
            assessment = "excessive_forward_lean"
        
        # Provide detailed assessment and recommendations
        running_speed_modifier = ""
        if hasattr(self, 'estimated_speed_mps'):
            # Adjust expectations based on running speed
            if self.estimated_speed_mps > 5.5:  # ~5:00 min/mile pace
                running_speed_modifier = " (Note: Faster speeds typically benefit from slightly increased forward lean)"
            elif self.estimated_speed_mps < 2.7:  # ~10:00 min/mile pace
                running_speed_modifier = " (Note: Slower speeds typically require less forward lean)"
        
        assessment_detail = {
            "backward_lean": "Backward trunk lean detected. This is inefficient and may indicate overstriding.",
            "insufficient_forward_lean": f"Forward lean is less than optimal range (5-10°). Consider increasing forward lean slightly from the ankles, not waist{running_speed_modifier}.",
            "optimal_forward_lean": f"Trunk angle is within optimal range (5-10°){running_speed_modifier}.",
            "moderate_forward_lean": f"Forward lean is slightly higher than optimal range (5-10°). This may be appropriate for sprinting or hill climbing{running_speed_modifier}.",
            "excessive_forward_lean": "Forward lean is excessive. This may increase stress on the lower back and hamstrings. Try running more upright with forward lean from ankles."
        }.get(assessment, "")
        
        return {
            'angle_degrees': angle,
            'is_optimal': is_optimal,
            'assessment': assessment,
            'assessment_detail': assessment_detail,
            'confidence': landmark_confidence
        }

    def _estimate_trunk_landmark_confidence(self, landmarks):
        """
        Estimate confidence in trunk angle measurement based on landmark quality.
        
        Args:
            landmarks: Dictionary containing pose landmarks with confidence values
            
        Returns:
            float: Confidence score (0-1)
        """
        # Default confidence if not provided with landmarks
        if not hasattr(landmarks.get('left_hip', {}), 'confidence'):
            return 0.8  # Reasonable default
        
        # Average confidence of relevant landmarks
        hip_confidence = (landmarks['left_hip'].get('confidence', 0.5) + 
                        landmarks['right_hip'].get('confidence', 0.5)) / 2
        
        shoulder_confidence = (landmarks['left_shoulder'].get('confidence', 0.5) + 
                            landmarks['right_shoulder'].get('confidence', 0.5)) / 2
        
        # Weighted average (shoulders are typically less reliable than hips)
        confidence = (hip_confidence * 0.6) + (shoulder_confidence * 0.4)
        
        # Apply penalties for edge cases
        if self._is_trunk_occluded(landmarks):
            confidence *= 0.7  # Reduce confidence when trunk is likely occluded
        
        # Check if angle is physiologically plausible
        if hasattr(self, 'previous_trunk_angle'):
            angle_change = abs(self.previous_trunk_angle - landmarks.get('trunk_angle', self.previous_trunk_angle))
            if angle_change > 20:  # Dramatic change unlikely between frames
                confidence *= 0.8
        
        return min(max(confidence, 0.0), 1.0)  # Clamp between 0 and 1

    def _is_trunk_occluded(self, landmarks):
        """
        Check if trunk landmarks might be occluded or unreliable.
        
        Args:
            landmarks: Dictionary containing pose landmarks
            
        Returns:
            bool: True if occlusion is likely
        """
        # Check if arms might be occluding trunk
        if ('left_elbow' in landmarks and 'right_elbow' in landmarks and
            'left_shoulder' in landmarks and 'right_shoulder' in landmarks):
            # Check if elbows are in front of shoulders (potential occlusion)
            left_elbow_forward = landmarks['left_elbow'][0] < landmarks['left_shoulder'][0]
            right_elbow_forward = landmarks['right_elbow'][0] < landmarks['right_shoulder'][0]
            
            if left_elbow_forward and right_elbow_forward:
                return True
        
        # Additional checks could be added here
        
        return False
    
    def analyze_arm_carriage(self, landmarks, frame_index=None):
        """
        Analyze arm carriage and swing mechanics during running.
        
        Args:
            landmarks: Dictionary containing pose landmarks with x,y coordinates
            frame_index: Current frame index (optional, for temporal analysis)
            
        Returns:
            dict: Contains:
                - 'upper_arm_angle': Angle of upper arm relative to vertical (degrees)
                - 'elbow_angle': Elbow flexion angle (degrees)
                - 'hand_position': Assessment of hand position (e.g., 'optimal', 'too_high')
                - 'arm_swing_amplitude': Estimated swing amplitude (if temporal data available)
                - 'arm_swing_symmetry': Assessment of left/right symmetry (if both visible)
                - 'overall_assessment': Overall assessment of arm carriage
                - 'recommendations': Specific recommendations for improvement
        """
        result = {
            'upper_arm_angle': None,
            'elbow_angle': None,
            'hand_position': None,
            'arm_swing_amplitude': None,
            'arm_swing_symmetry': None,
            'overall_assessment': None,
            'recommendations': []
        }
        
        # Initialize confidence tracking
        visible_points = 0
        total_points = 0
        
        # Track which arms are visible for analysis
        visible_arms = []
        if all(point in landmarks for point in ['right_shoulder', 'right_elbow', 'right_wrist']):
            visible_arms.append('right')
            visible_points += 3
        
        if all(point in landmarks for point in ['left_shoulder', 'left_elbow', 'left_wrist']):
            visible_arms.append('left')
            visible_points += 3
        
        total_points = 6  # Total possible arm points
        
        # If no arms are visible with sufficient landmarks, return limited analysis
        if not visible_arms:
            return {
                'upper_arm_angle': None,
                'elbow_angle': None,
                'overall_assessment': 'insufficient_data',
                'recommendations': ['Ensure arms are visible in the video for proper analysis']
            }
        
        # Analysis for each visible arm
        for arm_side in visible_arms:
            prefix = arm_side
            
            # Extract landmarks for current arm
            shoulder = landmarks[f'{prefix}_shoulder']
            elbow = landmarks[f'{prefix}_elbow']
            wrist = landmarks[f'{prefix}_wrist']
            
            # 1. Calculate upper arm angle relative to vertical
            upper_arm_dx = elbow[0] - shoulder[0]
            upper_arm_dy = shoulder[1] - elbow[1]  # Inverted y-axis in image coordinates
            
            upper_arm_angle = np.degrees(np.arctan2(upper_arm_dx, upper_arm_dy))
            
            # Store in result, prioritizing right arm (typically more visible in right-side footage)
            if result['upper_arm_angle'] is None or arm_side == 'right':
                result['upper_arm_angle'] = upper_arm_angle
            
            # 2. Calculate elbow flexion angle
            elbow_angle = self._calculate_elbow_angle(landmarks, arm_side)
            
            # Store in result, prioritizing right arm
            if result['elbow_angle'] is None or arm_side == 'right':
                result['elbow_angle'] = elbow_angle
            
            # 3. Analyze hand position relative to body
            # - Check if hands cross midline
            # - Check if hands rise above shoulder height
            # - Check if hands drop below hip height
            hip_y = landmarks[f'{prefix}_hip'][1]
            shoulder_y = shoulder[1]
            wrist_y = wrist[1]
            
            # Approximate body midline using hip and shoulder midpoints
            hip_center_x = (landmarks['left_hip'][0] + landmarks['right_hip'][0]) / 2 if all(k in landmarks for k in ['left_hip', 'right_hip']) else None
            shoulder_center_x = (landmarks['left_shoulder'][0] + landmarks['right_shoulder'][0]) / 2 if all(k in landmarks for k in ['left_shoulder', 'right_shoulder']) else None
            
            if hip_center_x is not None and shoulder_center_x is not None:
                midline_x = (hip_center_x + shoulder_center_x) / 2
                
                # Check if hand crosses midline (considering which side the arm is on)
                crosses_midline = False
                if (arm_side == 'right' and wrist[0] < midline_x) or (arm_side == 'left' and wrist[0] > midline_x):
                    crosses_midline = True
                    result['recommendations'].append(f"Avoid crossing hands over body midline during arm swing")
            
            # Check vertical hand position
            hand_position = 'optimal'
            if wrist_y < shoulder_y:  # Hand higher than shoulder
                hand_position = 'too_high'
                result['recommendations'].append("Keep hands below shoulder height during arm swing")
            elif wrist_y > hip_y:  # Hand lower than hip
                hand_position = 'too_low'
                result['recommendations'].append("Avoid dropping hands below hip level during arm swing")
            
            # Store in result (prioritizing right arm for single-sided analysis)
            if result['hand_position'] is None or arm_side == 'right':
                result['hand_position'] = hand_position
        
        # 4. Temporal analysis of arm swing (if tracking data available)
        if frame_index is not None:
            # Initialize arm position history if it doesn't exist
            if not hasattr(self, 'arm_position_history'):
                self.arm_position_history = {
                    'right': [],
                    'left': []
                }
            
            # Store arm positions
            for arm_side in visible_arms:
                if arm_side == 'right' and 'right_wrist' in landmarks and 'right_shoulder' in landmarks:
                    self.arm_position_history['right'].append({
                        'frame': frame_index,
                        'wrist_pos': landmarks['right_wrist'],
                        'elbow_pos': landmarks['right_elbow'],
                        'shoulder_pos': landmarks['right_shoulder']
                    })
                elif arm_side == 'left' and 'left_wrist' in landmarks and 'left_shoulder' in landmarks:
                    self.arm_position_history['left'].append({
                        'frame': frame_index,
                        'wrist_pos': landmarks['left_wrist'],
                        'elbow_pos': landmarks['left_elbow'],
                        'shoulder_pos': landmarks['left_shoulder']
                    })
            
            # Limit history size (keep last 60 frames - approximately 2 seconds at 30fps)
            max_history = 60
            for arm_side in ['right', 'left']:
                if len(self.arm_position_history[arm_side]) > max_history:
                    self.arm_position_history[arm_side] = self.arm_position_history[arm_side][-max_history:]
            
            # Calculate arm swing amplitude (if we have enough history)
            for arm_side in visible_arms:
                if len(self.arm_position_history[arm_side]) >= 15:  # Need enough frames for reliable measurement
                    # Find extremes of anterior and posterior swing
                    anterior_positions = []
                    posterior_positions = []
                    
                    for pos in self.arm_position_history[arm_side]:
                        # Compute angle relative to body
                        dx = pos['wrist_pos'][0] - pos['shoulder_pos'][0]
                        if arm_side == 'right':
                            # Right arm: posterior is positive x, anterior is negative x
                            if dx < 0:
                                anterior_positions.append(dx)
                            else:
                                posterior_positions.append(dx)
                        else:
                            # Left arm: posterior is negative x, anterior is positive x
                            if dx > 0:
                                anterior_positions.append(dx)
                            else:
                                posterior_positions.append(dx)
                    
                    # Calculate swing amplitude if we have both anterior and posterior positions
                    if anterior_positions and posterior_positions:
                        # Get average of extremes to filter out noise
                        anterior_extreme = sorted(anterior_positions)[:5]  # 5 most extreme values
                        posterior_extreme = sorted(posterior_positions, reverse=True)[:5]  # 5 most extreme values
                        
                        anterior_avg = sum(anterior_extreme) / len(anterior_extreme)
                        posterior_avg = sum(posterior_extreme) / len(posterior_extreme)
                        
                        # Amplitude is the difference between extremes
                        swing_amplitude = abs(posterior_avg - anterior_avg)
                        
                        # Convert to degrees for easier interpretation
                        # Approximating the arc length using shoulder as pivot
                        shoulder_width = abs(landmarks['right_shoulder'][0] - landmarks['left_shoulder'][0]) \
                            if all(k in landmarks for k in ['right_shoulder', 'left_shoulder']) else 0.5  # Default value
                        
                        # Normalize by shoulder width
                        normalized_amplitude = swing_amplitude / shoulder_width
                        
                        # Store result
                        result['arm_swing_amplitude'] = normalized_amplitude
                        
                        # Assess swing amplitude
                        if normalized_amplitude < 0.6:
                            result['recommendations'].append("Increase arm swing amplitude for better running efficiency")
                        elif normalized_amplitude > 1.4:
                            result['recommendations'].append("Consider reducing excessive arm swing to conserve energy")
        
        # 5. Analyze arm swing symmetry (if both arms visible)
        if 'right' in visible_arms and 'left' in visible_arms:
            # Compare angles between left and right arms
            left_upper_arm_dx = landmarks['left_elbow'][0] - landmarks['left_shoulder'][0]
            left_upper_arm_dy = landmarks['left_shoulder'][1] - landmarks['left_elbow'][1]
            left_angle = np.degrees(np.arctan2(left_upper_arm_dx, left_upper_arm_dy))
            
            right_upper_arm_dx = landmarks['right_elbow'][0] - landmarks['right_shoulder'][0]
            right_upper_arm_dy = landmarks['right_shoulder'][1] - landmarks['right_elbow'][1]
            right_angle = np.degrees(np.arctan2(right_upper_arm_dx, right_upper_arm_dy))
            
            # Compare elbow angles
            left_elbow_angle = self._calculate_elbow_angle(landmarks, 'left')
            right_elbow_angle = self._calculate_elbow_angle(landmarks, 'right')
            
            # Check for significant asymmetry
            angle_difference = abs(left_angle - right_angle)
            elbow_difference = abs(left_elbow_angle - right_elbow_angle)
            
            if angle_difference > 20 or elbow_difference > 25:
                result['arm_swing_symmetry'] = 'asymmetrical'
                result['recommendations'].append("Work on arm swing symmetry for better running economy")
            else:
                result['arm_swing_symmetry'] = 'symmetrical'
        
        # 6. Overall assessment of arm carriage
        overall_issues = []
        
        # Check elbow angle (90° ± 15° is generally optimal)
        if result['elbow_angle'] is not None:
            if result['elbow_angle'] < 75:
                overall_issues.append("elbow_too_bent")
                result['recommendations'].append("Open elbow angle slightly (aim for 90-100°)")
            elif result['elbow_angle'] > 115:
                overall_issues.append("elbow_too_straight")
                result['recommendations'].append("Increase elbow bend (aim for 90-100°)")
        
        # Check hand position
        if result['hand_position'] != 'optimal':
            overall_issues.append(f"hand_position_{result['hand_position']}")
        
        # Check arm swing amplitude
        if result['arm_swing_amplitude'] is not None:
            if result['arm_swing_amplitude'] < 0.6:
                overall_issues.append("insufficient_swing")
            elif result['arm_swing_amplitude'] > 1.4:
                overall_issues.append("excessive_swing")
        
        # Determine overall assessment
        if not overall_issues:
            result['overall_assessment'] = "optimal"
        elif len(overall_issues) == 1:
            result['overall_assessment'] = f"suboptimal_{overall_issues[0]}"
        else:
            result['overall_assessment'] = "multiple_issues"
        
        # Remove duplicate recommendations
        result['recommendations'] = list(set(result['recommendations']))
        
        # Add general guidance based on overall assessment
        if result['overall_assessment'] == "optimal":
            result['recommendations'].insert(0, "Arm carriage is good - maintain current form")
        elif "multiple_issues" in result['overall_assessment']:
            result['recommendations'].insert(0, "Multiple arm carriage issues detected - focus on the specific recommendations below")
        
        return result

    def _calculate_elbow_angle(self, landmarks, side):
        """
        Calculate elbow flexion angle.
        
        Args:
            landmarks: Dictionary containing pose landmarks
            side: 'left' or 'right' to specify which arm
            
        Returns:
            float: Elbow angle in degrees (180 = fully straight)
        """
        # Extract relevant landmarks
        shoulder = landmarks[f'{side}_shoulder']
        elbow = landmarks[f'{side}_elbow']
        wrist = landmarks[f'{side}_wrist']
        
        # Calculate vectors
        upper_arm = [elbow[0] - shoulder[0], elbow[1] - shoulder[1]]
        forearm = [wrist[0] - elbow[0], wrist[1] - elbow[1]]
        
        # Calculate dot product
        dot_product = upper_arm[0] * forearm[0] + upper_arm[1] * forearm[1]
        
        # Calculate magnitudes
        upper_arm_magnitude = (upper_arm[0]**2 + upper_arm[1]**2)**0.5
        forearm_magnitude = (forearm[0]**2 + forearm[1]**2)**0.5
        
        # Calculate angle in radians, then convert to degrees
        if upper_arm_magnitude == 0 or forearm_magnitude == 0:
            return 0
        
        cos_angle = dot_product / (upper_arm_magnitude * forearm_magnitude)
        # Ensure value is in valid range for arccos
        cos_angle = max(min(cos_angle, 1.0), -1.0)
        
        angle_rad = math.acos(cos_angle)
        angle_deg = math.degrees(angle_rad)
        
        # Adjust to standard biomechanical convention
        # 180 degrees = straight arm, 0 degrees = fully flexed
        return 180 - angle_deg

    def get_arm_carriage_summary(self, running_state=None):
        """
        Generate a human-readable summary of arm carriage analysis.
        
        Args:
            running_state: Optional dictionary with additional running metrics
            
        Returns:
            str: Summary of arm carriage analysis with recommendations
        """
        if not hasattr(self, 'arm_carriage_results') or not self.arm_carriage_results:
            return "Insufficient data for arm carriage analysis."
        
        # Get most recent arm carriage analysis
        latest = self.arm_carriage_results[-1]
        
        # Create summary
        summary = "ARM CARRIAGE ANALYSIS:\n"
        
        # Add elbow angle assessment
        if latest.get('elbow_angle') is not None:
            elbow_angle = latest['elbow_angle']
            summary += f"• Elbow angle: {elbow_angle:.1f}° "
            
            if 85 <= elbow_angle <= 105:
                summary += "(optimal)\n"
            elif elbow_angle < 85:
                summary += "(too bent)\n"
            else:
                summary += "(too straight)\n"
        
        # Add hand position assessment
        if latest.get('hand_position') is not None:
            summary += f"• Hand position: "
            
            if latest['hand_position'] == 'optimal':
                summary += "Good (within recommended range)\n"
            elif latest['hand_position'] == 'too_high':
                summary += "Too high (above shoulder level)\n"
            elif latest['hand_position'] == 'too_low':
                summary += "Too low (below hip level)\n"
        
        # Add arm swing amplitude if available
        if latest.get('arm_swing_amplitude') is not None:
            amplitude = latest['arm_swing_amplitude']
            amplitude_percent = amplitude * 100  # Convert to percentage for readability
            
            summary += f"• Arm swing amplitude: {amplitude_percent:.1f}% of shoulder width "
            
            if 60 <= amplitude_percent <= 140:
                summary += "(optimal range)\n"
            elif amplitude_percent < 60:
                summary += "(insufficient - energy inefficient)\n"
            else:
                summary += "(excessive - may waste energy)\n"
        
        # Add symmetry information if available
        if latest.get('arm_swing_symmetry') is not None:
            summary += f"• Arm swing symmetry: {latest['arm_swing_symmetry'].capitalize()}\n"
        
        # Add recommendations
        if latest.get('recommendations'):
            summary += "\nRECOMMENDATIONS:\n"
            for i, rec in enumerate(latest['recommendations'], 1):
                summary += f"{i}. {rec}\n"
        
        # Add context based on running state
        if running_state and 'speed_mps' in running_state:
            speed = running_state['speed_mps']
            
            if speed > 5.0:  # Faster running
                summary += "\nNote: At faster speeds (current: {:.1f} m/s), slightly increased arm drive is beneficial.\n".format(speed)
            elif speed < 3.0:  # Slower running
                summary += "\nNote: At slower speeds (current: {:.1f} m/s), focus on relaxed, efficient arm movement.\n".format(speed)
        
        return summary
    
    def calculate_knee_angle(self, landmarks, side):
        """
        Calculate knee angle (extension/flexion).
        
        Args:
            landmarks: Dictionary containing pose landmarks
            side: 'left' or 'right' to specify which leg
            
        Returns:
            float: Knee angle in degrees (180 = fully straight)
        """
        # Get coordinates for hip, knee, and ankle
        hip = landmarks[f'{side}_hip']
        knee = landmarks[f'{side}_knee']
        ankle = landmarks[f'{side}_ankle']
        
        # Calculate vectors
        hip_to_knee = [knee[0] - hip[0], knee[1] - hip[1]]
        knee_to_ankle = [ankle[0] - knee[0], ankle[1] - knee[1]]
        
        # Calculate dot product
        dot_product = hip_to_knee[0] * knee_to_ankle[0] + hip_to_knee[1] * knee_to_ankle[1]
        
        # Calculate magnitudes
        hip_knee_magnitude = (hip_to_knee[0]**2 + hip_to_knee[1]**2)**0.5
        knee_ankle_magnitude = (knee_to_ankle[0]**2 + knee_to_ankle[1]**2)**0.5
        
        # Calculate angle in radians, then convert to degrees
        if hip_knee_magnitude == 0 or knee_ankle_magnitude == 0:
            return 0
        
        cos_angle = dot_product / (hip_knee_magnitude * knee_ankle_magnitude)
        # Ensure value is in valid range for arccos
        cos_angle = max(min(cos_angle, 1.0), -1.0)
        
        angle_rad = math.acos(cos_angle)
        angle_deg = math.degrees(angle_rad)
        
        # Return complementary angle - knee is typically measured as extension angle
        return 180 - angle_deg
    
    def estimate_stride_length(self, landmarks, frame_index=None, height_cm=None):
        """
        Estimate stride length based on foot positions with temporal tracking,
        optimized for side-view analysis.
        
        Args:
            landmarks: Dictionary containing pose landmarks with x,y coordinates
            frame_index: Current frame index for temporal tracking
            height_cm: Runner's height in cm (if known)
            
        Returns:
            dict: Contains:
                - 'instantaneous_estimate_cm': Current estimate based on foot positions
                - 'stride_length_cm': Smoothed stride length estimate (if temporal data available)
                - 'normalized_stride_length': Stride length relative to runner's height
                - 'stride_frequency': Estimated cadence (if temporal data available)
                - 'assessment': Assessment of stride length appropriateness
                - 'confidence': Confidence in the measurement (0-1)
        """
        result = {
            'instantaneous_estimate_cm': None,
            'stride_length_cm': None,
            'normalized_stride_length': None,
            'stride_frequency': None,
            'assessment': None,
            'confidence': 0.5  # Default confidence
        }
        
        # Check if necessary landmarks are available - for side view, we need at least one foot and ankle
        required_landmarks = ['right_foot_index', 'right_ankle', 'right_knee', 'right_hip']
        if not all(key in landmarks for key in required_landmarks):
            result['confidence'] = 0
            result['assessment'] = "insufficient_data"
            return result
        
        # 1. Calculate body height scale factor for side view
        scale_factor = self._calculate_side_view_scale_factor(landmarks, height_cm)
        
        # 2. Initialize or update temporal tracking
        if frame_index is not None:
            if not hasattr(self, 'side_view_tracking'):
                self.side_view_tracking = {
                    'foot_positions': [],        # List of foot position records
                    'stride_events': [],         # List of detected stride events
                    'current_stride_length': 0,  # Current best estimate
                    'stride_lengths': [],        # Recent stride length measurements
                    'last_foot_y': None,         # Last foot y position
                    'foot_y_history': [],        # Recent foot y positions for peak detection
                    'stride_times': [],          # Time between strides in frames
                    'step_detected': False,      # Flag for step detection
                    'last_touchdown_frame': None, # Last frame where foot touched down
                    'foot_velocity': []          # Track foot velocity for detecting touchdown
                }
                
            # Add current foot position to history
            right_foot = np.array([landmarks['right_foot_index'][0], landmarks['right_foot_index'][1]])
            right_ankle = np.array([landmarks['right_ankle'][0], landmarks['right_ankle'][1]])
            
            # Store foot Y position for vertical movement analysis
            self.side_view_tracking['foot_y_history'].append(right_foot[1])
            
            # Keep history manageable
            max_history = 60  # About 2 seconds at 30fps
            if len(self.side_view_tracking['foot_y_history']) > max_history:
                self.side_view_tracking['foot_y_history'] = self.side_view_tracking['foot_y_history'][-max_history:]
            
            # Calculate foot velocity (for touchdown detection)
            if self.side_view_tracking['last_foot_y'] is not None:
                y_velocity = right_foot[1] - self.side_view_tracking['last_foot_y']
                self.side_view_tracking['foot_velocity'].append(y_velocity)
                
                # Keep velocity history manageable
                if len(self.side_view_tracking['foot_velocity']) > 10:
                    self.side_view_tracking['foot_velocity'] = self.side_view_tracking['foot_velocity'][-10:]
            
            self.side_view_tracking['last_foot_y'] = right_foot[1]
            
            # 3. Detect foot touchdown (using velocity and position)
            # In a side view, touchdown happens when:
            #   - Foot is near lowest position
            #   - Vertical velocity transitions from downward to upward
            
            foot_touchdown_detected = False
            
            # Check for near lowest point and velocity transition
            if len(self.side_view_tracking['foot_velocity']) >= 3:
                # Calculate weighted average of recent velocity (more weight to recent)
                recent_velocity = sum(v * (i+1) for i, v in enumerate(self.side_view_tracking['foot_velocity'][-3:])) / 6
                
                # Check if foot is near lowest position (lower quartile of y values)
                y_values = self.side_view_tracking['foot_y_history']
                if len(y_values) > 10:
                    y_threshold = sorted(y_values)[-len(y_values)//4]  # Lower quartile
                    
                    # Foot touchdown detected when:
                    # - Currently in lower range of position
                    # - Recent velocity showing transition from downward to slowing/upward
                    if right_foot[1] > y_threshold and -0.01 < recent_velocity < 0.01:
                        foot_touchdown_detected = True
            
            # Record touchdown event
            if foot_touchdown_detected and (self.side_view_tracking['last_touchdown_frame'] is None or 
                                        frame_index - self.side_view_tracking['last_touchdown_frame'] > 15):
                self.side_view_tracking['stride_events'].append({
                    'frame': frame_index,
                    'position': right_foot.copy(),
                    'type': 'touchdown'
                })
                self.side_view_tracking['last_touchdown_frame'] = frame_index
                
                # Calculate stride length and frequency if we have at least 2 touchdowns
                if len(self.side_view_tracking['stride_events']) >= 2:
                    # Get last two touchdown events
                    last_two = [e for e in self.side_view_tracking['stride_events'] if e['type'] == 'touchdown'][-2:]
                    
                    if len(last_two) >= 2:
                        frames_diff = last_two[1]['frame'] - last_two[0]['frame']
                        
                        # Reasonable frame range for a stride (0.5-2 seconds at 30fps)
                        if 15 <= frames_diff <= 60:
                            # Calculate horizontal distance between touchdowns
                            stride_distance = abs(last_two[1]['position'][0] - last_two[0]['position'][0])
                            
                            # Convert to cm using scale factor
                            stride_length_cm = stride_distance * scale_factor
                            
                            # Add to recent measurements
                            self.side_view_tracking['stride_lengths'].append(stride_length_cm)
                            self.side_view_tracking['stride_times'].append(frames_diff)
                            
                            # Keep only recent measurements
                            if len(self.side_view_tracking['stride_lengths']) > 5:
                                self.side_view_tracking['stride_lengths'] = self.side_view_tracking['stride_lengths'][-5:]
                                self.side_view_tracking['stride_times'] = self.side_view_tracking['stride_times'][-5:]
                            
                            # Calculate stride frequency (cadence)
                            # Convert frames to seconds (assuming 30fps)
                            avg_stride_time = sum(self.side_view_tracking['stride_times']) / len(self.side_view_tracking['stride_times'])
                            stride_time_seconds = avg_stride_time / 30.0
                            stride_frequency = 60.0 / stride_time_seconds  # strides per minute
                            
                            # Store in result
                            result['stride_frequency'] = stride_frequency
                            
                            # Use the average of recent measurements for stability
                            result['stride_length_cm'] = sum(self.side_view_tracking['stride_lengths']) / len(self.side_view_tracking['stride_lengths'])
                            result['confidence'] = 0.85  # Higher confidence with temporal data
                            
                            # Also calculate instantaneous estimate from the most recent stride
                            result['instantaneous_estimate_cm'] = stride_length_cm
        
        # If we don't have temporal data, use leg length as a fallback instantaneous estimate
        if result['instantaneous_estimate_cm'] is None:
            # Estimate from leg extension
            right_hip = np.array([landmarks['right_hip'][0], landmarks['right_hip'][1]])
            right_knee = np.array([landmarks['right_knee'][0], landmarks['right_knee'][1]])
            right_ankle = np.array([landmarks['right_ankle'][0], landmarks['right_ankle'][1]])
            
            # Calculate horizontal distance from hip to ankle
            horizontal_leg_extension = abs(right_hip[0] - right_ankle[0])
            
            # Use this as a proportion of stride length (typically 0.8-1.0x)
            instantaneous_estimate_cm = horizontal_leg_extension * scale_factor * 0.9
            
            result['instantaneous_estimate_cm'] = instantaneous_estimate_cm
            
            # If no stride length from temporal tracking, use this estimate
            if result['stride_length_cm'] is None:
                result['stride_length_cm'] = instantaneous_estimate_cm
                result['confidence'] = 0.6  # Moderate confidence in this estimate
        
        # 5. Normalize stride length by height
        if height_cm:
            # Calculate stride length as percentage of height
            result['normalized_stride_length'] = result['stride_length_cm'] / height_cm
        else:
            # Estimate height from landmarks if possible
            estimated_height = self._estimate_height_from_side_view(landmarks)
            if estimated_height:
                result['normalized_stride_length'] = result['stride_length_cm'] / estimated_height
        
        # 6. Assess stride length appropriateness
        if result['normalized_stride_length'] is not None:
            # Runner's speed affects optimal stride length
            speed_assessment = "moderate"  # Default assumption
            if hasattr(self, 'estimated_speed_mps'):
                if self.estimated_speed_mps < 2.5:  # Slow jog
                    speed_assessment = "slow"
                elif self.estimated_speed_mps > 4.5:  # Fast run
                    speed_assessment = "fast"
            
            # Based on research, normalized stride length typically ranges from:
            # - 0.7-0.85 for slower running
            # - 0.85-1.0 for moderate running
            # - 1.0-1.2+ for faster running
            nsl = result['normalized_stride_length']
            assessment = "optimal"
            
            if speed_assessment == "slow":
                if nsl < 0.65:
                    assessment = "too_short"
                elif nsl > 0.9:
                    assessment = "too_long"
            elif speed_assessment == "moderate":
                if nsl < 0.8:
                    assessment = "too_short"
                elif nsl > 1.05:
                    assessment = "too_long"
            else:  # fast
                if nsl < 0.9:
                    assessment = "too_short"
                elif nsl > 1.3:
                    assessment = "too_long"
            
            result['assessment'] = assessment
        
        return result
    
    def _calculate_side_view_scale_factor(self, landmarks, height_cm=None):
        """
        Calculate scale factor for converting normalized coordinates to centimeters,
        optimized for side-view analysis.
        
        Args:
            landmarks: Dictionary containing pose landmarks
            height_cm: Runner's height in cm (if known)
            
        Returns:
            float: Scale factor for converting coordinates to centimeters
        """
        # If height is known, use it for scaling
        if height_cm:
            # For side view, use the vertical distance from head to ankle
            head_y = landmarks.get('nose', [0, float('inf')])[1]
            if head_y == float('inf'):
                head_y = min(landmarks.get('right_ear', [0, float('inf')])[1],
                            landmarks.get('right_eye', [0, float('inf')])[1])
            
            ankle_y = landmarks['right_ankle'][1]
            
            body_height_coords = ankle_y - head_y
            
            # Height in real units divided by height in coordinate units
            # Multiply by 0.85 since head-to-ankle is ~85% of total height in side view
            return height_cm / (body_height_coords * 0.85) if body_height_coords > 0 else 200
        
        # Otherwise, estimate scale using body proportions visible in side view
        else:
            # Leg length is approximately 0.48 of body height
            thigh_length = np.sqrt((landmarks['right_hip'][0] - landmarks['right_knee'][0])**2 + 
                                (landmarks['right_hip'][1] - landmarks['right_knee'][1])**2)
            
            shin_length = np.sqrt((landmarks['right_knee'][0] - landmarks['right_ankle'][0])**2 + 
                                (landmarks['right_knee'][1] - landmarks['right_ankle'][1])**2)
            
            leg_length = thigh_length + shin_length
            
            if leg_length > 0:
                # Estimated height based on leg length
                estimated_height_coords = leg_length / 0.48
                
                # Use 170cm as default average height for scale conversion
                return 170 / estimated_height_coords
            
            # Torso length is approximately 0.30 of body height
            torso_length = np.sqrt((landmarks['right_hip'][0] - landmarks['right_shoulder'][0])**2 + 
                                (landmarks['right_hip'][1] - landmarks['right_shoulder'][1])**2)
            
            if torso_length > 0:
                estimated_height_coords = torso_length / 0.30
                return 170 / estimated_height_coords
            
            # Default scale factor if all else fails
            return 200

    def _estimate_height_from_side_view(self, landmarks):
        """
        Estimate the runner's height in cm from side-view landmarks.
        
        Args:
            landmarks: Dictionary containing pose landmarks
            
        Returns:
            float: Estimated height in cm, or None if estimation fails
        """
        height_estimates = []
        weights = []
        
        # Method 1: Use visible body height (head to ankle)
        head_y = landmarks.get('nose', [0, float('inf')])[1]
        if head_y == float('inf'):
            head_y = min(landmarks.get('right_ear', [0, float('inf')])[1],
                        landmarks.get('right_eye', [0, float('inf')])[1])
        
        ankle_y = landmarks['right_ankle'][1]
        
        if head_y < float('inf'):
            visible_height_coords = ankle_y - head_y
            if visible_height_coords > 0:
                # Convert to cm (assuming visible height is about 85% of total height)
                height_estimates.append(visible_height_coords / 0.85)
                weights.append(1.0)  # Base weight
        
        # Method 2: Use leg length proportions (leg length is approximately 0.48 of body height)
        thigh_length = np.sqrt((landmarks['right_hip'][0] - landmarks['right_knee'][0])**2 + 
                            (landmarks['right_hip'][1] - landmarks['right_knee'][1])**2)
        
        shin_length = np.sqrt((landmarks['right_knee'][0] - landmarks['right_ankle'][0])**2 + 
                            (landmarks['right_knee'][1] - landmarks['right_ankle'][1])**2)
        
        leg_length = thigh_length + shin_length
        
        if leg_length > 0:
            height_from_leg = leg_length / 0.48
            height_estimates.append(height_from_leg)
            weights.append(0.9)  # Leg length is a good height predictor
        
        # Method 3: Use torso length (torso is approximately 0.30 of body height)
        torso_length = np.sqrt((landmarks['right_hip'][0] - landmarks['right_shoulder'][0])**2 + 
                            (landmarks['right_hip'][1] - landmarks['right_shoulder'][1])**2)
        
        if torso_length > 0:
            height_from_torso = torso_length / 0.30
            height_estimates.append(height_from_torso)
            weights.append(0.8)  # Torso can vary but is still useful
        
        # Calculate weighted average if we have estimates
        if height_estimates:
            weighted_height = sum(h * w for h, w in zip(height_estimates, weights)) / sum(weights)
            # Convert from coordinate units to cm using reference height of 170cm
            return weighted_height * 170
        
        # Default to average height if estimation fails
        return 170  # Average adult height in cm

    def estimate_running_speed(self, landmarks, frame_index=None, ground_truth_speed=None):
        """
        Estimate running speed from side-view landmarks.
        Can be used alongside stride analysis to improve assessment.
        
        Args:
            landmarks: Dictionary containing pose landmarks
            frame_index: Current frame index for temporal tracking
            ground_truth_speed: Known speed in m/s (if available)
            
        Returns:
            float: Estimated speed in meters per second
        """
        # Initialize speed tracking if needed
        if not hasattr(self, 'speed_tracking'):
            self.speed_tracking = {
                'position_history': [],    # Position history
                'time_history': [],        # Frame indices
                'speed_estimates': [],     # Recent speed estimates
                'calibration_factor': 1.0  # Calibration if ground truth is provided
            }
        
        # Calculate center of mass position (approximation)
        hip_pos = np.array(landmarks['right_hip'])
        shoulder_pos = np.array(landmarks['right_shoulder'])
        com_position = (hip_pos * 0.6 + shoulder_pos * 0.4)  # Weight more toward hip
        
        # Store position and time
        if frame_index is not None:
            self.speed_tracking['position_history'].append(com_position.copy())
            self.speed_tracking['time_history'].append(frame_index)
            
            # Keep history manageable
            max_history = 30  # About 1 second at 30fps
            if len(self.speed_tracking['position_history']) > max_history:
                self.speed_tracking['position_history'] = self.speed_tracking['position_history'][-max_history:]
                self.speed_tracking['time_history'] = self.speed_tracking['time_history'][-max_history:]
            
            # Need at least 15 frames (0.5 seconds) for reliable speed estimation
            if len(self.speed_tracking['position_history']) >= 15:
                # Calculate horizontal displacement
                start_pos = self.speed_tracking['position_history'][0]
                end_pos = self.speed_tracking['position_history'][-1]
                displacement = abs(end_pos[0] - start_pos[0])
                
                # Calculate time elapsed (assume 30fps)
                time_elapsed = (self.speed_tracking['time_history'][-1] - self.speed_tracking['time_history'][0]) / 30.0
                
                if time_elapsed > 0:
                    # Convert displacement to real-world units
                    # Get scale factor from stride analysis
                    if hasattr(self, '_calculate_side_view_scale_factor'):
                        scale_factor = self._calculate_side_view_scale_factor(landmarks)
                        displacement_meters = displacement * scale_factor / 100  # Convert cm to meters
                        
                        # Calculate speed
                        speed_mps = displacement_meters / time_elapsed
                        
                        # Apply calibration if ground truth was provided
                        speed_mps *= self.speed_tracking['calibration_factor']
                        
                        # Store speed estimate
                        self.speed_tracking['speed_estimates'].append(speed_mps)
                        
                        # Keep only recent estimates
                        if len(self.speed_tracking['speed_estimates']) > 5:
                            self.speed_tracking['speed_estimates'] = self.speed_tracking['speed_estimates'][-5:]
                        
                        # Update calibration if ground truth is provided
                        if ground_truth_speed is not None and ground_truth_speed > 0:
                            current_estimate = sum(self.speed_tracking['speed_estimates']) / len(self.speed_tracking['speed_estimates'])
                            if current_estimate > 0:
                                self.speed_tracking['calibration_factor'] = ground_truth_speed / current_estimate
                        
                        # Return average of recent speed estimates
                        self.estimated_speed_mps = sum(self.speed_tracking['speed_estimates']) / len(self.speed_tracking['speed_estimates'])
                        return self.estimated_speed_mps
        
        # Return last estimated speed if available
        if hasattr(self, 'estimated_speed_mps'):
            return self.estimated_speed_mps
        
        # Default to moderate running speed if no estimate available
        return 3.0  # ~3 m/s is approximately 10.8 km/h or 6.7 mph

    def _calculate_height_scale_factor(self, landmarks, height_cm=172):
        """
        Calculate scale factor for converting normalized coordinates to centimeters.
        
        Args:
            landmarks: Dictionary containing pose landmarks
            height_cm: Runner's height in cm (if known)
            
        Returns:
            float: Scale factor for converting coordinates to centimeters
        """
        # If height is known, use it for scaling
        if height_cm:
            # Estimate visible body height in coordinate space
            # Using head to ankle vertical distance as proxy for height
            head_y = min(landmarks['nose'][1], landmarks.get('left_ear', [0, float('inf')])[1], 
                        landmarks.get('right_ear', [0, float('inf')])[1])
            ankle_y = max(landmarks['left_ankle'][1], landmarks['right_ankle'][1])
            
            body_height_coords = ankle_y - head_y
            
            # Height in real units divided by height in coordinate units
            # Multiply by 0.9 since head-to-ankle is ~90% of total height
            return height_cm / (body_height_coords * 0.9) if body_height_coords > 0 else 200
        
        # Otherwise, estimate scale using body proportions
        else:
            # Hip width is approximately 0.191 of body height (anatomical average)
            hip_width_coords = abs(landmarks['left_hip'][0] - landmarks['right_hip'][0])
            
            # Shoulder width is approximately 0.259 of body height
            shoulder_width_coords = abs(landmarks['left_shoulder'][0] - landmarks['right_shoulder'][0])
            
            # Use average of hip and shoulder proportions to estimate height
            est_height_from_hips = hip_width_coords / 0.191 if hip_width_coords > 0 else 0
            est_height_from_shoulders = shoulder_width_coords / 0.259 if shoulder_width_coords > 0 else 0
            
            # Take the average if both are available, otherwise use the non-zero one
            if est_height_from_hips > 0 and est_height_from_shoulders > 0:
                estimated_height_coords = (est_height_from_hips + est_height_from_shoulders) / 2
            else:
                estimated_height_coords = max(est_height_from_hips, est_height_from_shoulders)
            
            # Use 170cm as default average height if estimation fails
            if estimated_height_coords <= 0:
                return 200  # Default scale factor
            
            # 170cm is average height, adjust scale accordingly
            return 170 / estimated_height_coords
        
    def _determine_gait_phase(self, landmarks):
        """
        Determine the current gait cycle phase based on foot positions.
        
        Args:
            landmarks: Dictionary containing pose landmarks
            
        Returns:
            str: Current gait phase (left_stance, right_stance, double_support, flight)
        """
        # Extract vertical (y) positions - higher value means lower position in image
        left_ankle_y = landmarks['left_ankle'][1]
        right_ankle_y = landmarks['right_ankle'][1]
        left_knee_y = landmarks['left_knee'][1]
        right_knee_y = landmarks['right_knee'][1]
        
        # Calculate vertical distances
        left_foot_height = left_ankle_y - left_knee_y
        right_foot_height = right_ankle_y - right_knee_y
        
        # Determine which foot is likely in contact with ground
        # Threshold for considering a foot in contact with ground (proportion of leg length)
        contact_threshold = 0.05
        
        left_foot_contact = left_foot_height > contact_threshold
        right_foot_contact = right_foot_height > contact_threshold
        
        # Check horizontal feet separation to improve detection
        feet_separation = abs(landmarks['left_foot_index'][0] - landmarks['right_foot_index'][0])
        
        # Feet close together might indicate stance-to-swing transition or flight phase
        if feet_separation < 0.1:  # Threshold as proportion of image width
            # Check if both feet are relatively high (flight phase)
            if not left_foot_contact and not right_foot_contact:
                return "flight"
        
        # Determine phase based on foot contact
        if left_foot_contact and right_foot_contact:
            return "double_support"
        elif left_foot_contact:
            return "left_stance"
        elif right_foot_contact:
            return "right_stance"
        else:
            # No foot contact detected - flight phase
            return "flight"

    def _estimate_height_from_landmarks(self, landmarks):
        """
        Estimate the runner's height in cm from landmarks.
        
        Args:
            landmarks: Dictionary containing pose landmarks
            
        Returns:
            float: Estimated height in cm, or None if estimation fails
        """
        # Method 1: Use visible body height (head to ankle)
        # Find highest point (minimum y value)
        head_points = [landmarks.get(point, [0, float('inf')])[1] for point in 
                    ['nose', 'left_ear', 'right_ear', 'left_eye', 'right_eye']]
        head_y = min([y for y in head_points if y < float('inf')])
        
        # Find lowest point (maximum y value)
        foot_points = [landmarks.get(point, [0, 0])[1] for point in 
                    ['left_ankle', 'right_ankle', 'left_foot_index', 'right_foot_index']]
        foot_y = max(foot_points)
        
        # Calculate height in coordinate space
        visible_height_coords = foot_y - head_y
        
        # Method 2: Use known body proportions
        # Hip width is approximately 0.191 of body height
        hip_width_coords = abs(landmarks['left_hip'][0] - landmarks['right_hip'][0])
        height_from_hips = hip_width_coords / 0.191 if hip_width_coords > 0 else 0
        
        # Shoulder width is approximately 0.259 of body height
        shoulder_width_coords = abs(landmarks['left_shoulder'][0] - landmarks['right_shoulder'][0])
        height_from_shoulders = shoulder_width_coords / 0.259 if shoulder_width_coords > 0 else 0
        
        # Method 3: Use leg length proportions
        # Leg length is approximately 0.48 of body height
        left_leg_length = ((landmarks['left_hip'][0] - landmarks['left_knee'][0])**2 + 
                        (landmarks['left_hip'][1] - landmarks['left_knee'][1])**2)**0.5 + \
                        ((landmarks['left_knee'][0] - landmarks['left_ankle'][0])**2 + 
                        (landmarks['left_knee'][1] - landmarks['left_ankle'][1])**2)**0.5
        
        right_leg_length = ((landmarks['right_hip'][0] - landmarks['right_knee'][0])**2 + 
                        (landmarks['right_hip'][1] - landmarks['right_knee'][1])**2)**0.5 + \
                        ((landmarks['right_knee'][0] - landmarks['right_ankle'][0])**2 + 
                        (landmarks['right_knee'][1] - landmarks['right_ankle'][1])**2)**0.5
        
        avg_leg_length = (left_leg_length + right_leg_length) / 2
        height_from_legs = avg_leg_length / 0.48 if avg_leg_length > 0 else 0
        
        # Combine estimates (weighted average)
        height_estimates = []
        weights = []
        
        if visible_height_coords > 0:
            # Convert to cm (assuming visible height is about 90% of total height)
            height_estimates.append(visible_height_coords * 170 / 0.9)
            weights.append(1.0)  # Base weight
        
        if height_from_hips > 0:
            height_estimates.append(height_from_hips * 170)
            weights.append(0.7)  # Hip width can vary among individuals
        
        if height_from_shoulders > 0:
            height_estimates.append(height_from_shoulders * 170)
            weights.append(0.8)  # Shoulder width more reliable than hip width
        
        if height_from_legs > 0:
            height_estimates.append(height_from_legs * 170)
            weights.append(0.9)  # Leg length is a good height predictor
        
        # Calculate weighted average if we have estimates
        if height_estimates:
            weighted_height = sum(h * w for h, w in zip(height_estimates, weights)) / sum(weights)
            return weighted_height
        
        # Default to average height if estimation fails
        return 170  # Average adult height in cm

    def get_stride_length_assessment(self, speed_mps=None):
        """
        Generate a human-readable assessment of stride length.
        
        Args:
            speed_mps: Current running speed in meters per second (if known)
            
        Returns:
            str: Assessment text with recommendations
        """
        if not hasattr(self, 'stride_data') or not self.stride_data:
            return "Insufficient data for stride length assessment."
        
        # Get latest stride data
        latest = self.stride_data[-1]
        
        # Start building assessment
        assessment = "STRIDE LENGTH ASSESSMENT:\n"
        
        # Add stride length measurement
        if latest.get('stride_length_cm'):
            sl_cm = latest['stride_length_cm']
            assessment += f"• Stride length: {sl_cm:.1f} cm"
            
            # Add normalized value if available
            if latest.get('normalized_stride_length'):
                nsl = latest['normalized_stride_length']
                assessment += f" ({nsl:.2f} × height)\n"
            else:
                assessment += "\n"
        
        # Add cadence if available
        if latest.get('stride_frequency'):
            cadence = latest['stride_frequency']
            assessment += f"• Cadence: {cadence:.1f} strides/minute"
            
            # Add cadence assessment
            if cadence < 160:
                assessment += " (below optimal range of 170-180)\n"
            elif cadence > 190:
                assessment += " (above optimal range of 170-180)\n"
            else:
                assessment += " (within optimal range)\n"
        
        # Add stride length assessment based on speed
        current_speed = speed_mps if speed_mps else getattr(self, 'estimated_speed_mps', None)
        
        if current_speed and latest.get('normalized_stride_length'):
            nsl = latest['normalized_stride_length']
            assessment += f"• At your current speed ({current_speed:.2f} m/s): "
            
            # Adjust optimal range based on speed
            if current_speed < 2.5:  # Slow jogging
                if nsl < 0.65:
                    assessment += "Stride length is too short. Try extending slightly while maintaining cadence.\n"
                elif nsl > 0.9:
                    assessment += "Stride length is longer than typical. Consider increasing cadence rather than stride length.\n"
                else:
                    assessment += "Stride length is appropriate for this pace.\n"
                    
            elif current_speed < 4.0:  # Moderate running
                if nsl < 0.8:
                    assessment += "Stride length is shorter than optimal. Consider a slight increase in stride length.\n"
                elif nsl > 1.05:
                    assessment += "Stride length is longer than optimal. May be overstriding - try increasing cadence instead.\n"
                else:
                    assessment += "Stride length is in the optimal range for this speed.\n"
                    
            else:  # Fast running
                if nsl < 0.9:
                    assessment += "Stride length is too short for this speed. Work on power development for longer strides.\n"
                elif nsl > 1.3:
                    assessment += "Stride length is very long. Ensure you're not overstriding with foot landing ahead of center of mass.\n"
                else:
                    assessment += "Stride length is appropriate for faster running.\n"
        
        # Add specific recommendations
        assessment += "\nRECOMMENDATIONS:\n"
        
        if latest.get('assessment') == "too_short":
            assessment += "1. Focus on hip extension strength and mobility\n"
            assessment += "2. Consider plyometric exercises to improve stride power\n"
            assessment += "3. Practice running drills like high knees and butt kicks\n"
        elif latest.get('assessment') == "too_long":
            assessment += "1. Work on increasing cadence (steps per minute)\n"
            assessment += "2. Focus on landing with foot closer to center of mass\n"
            assessment += "3. Try metronome training at 170-180 bpm\n"
        else:
            assessment += "1. Maintain current stride mechanics\n"
            assessment += "2. For efficiency, focus on the relationship between stride length and cadence\n"
        
        # Add stride length-cadence relationship note
        assessment += "\nNOTE: Optimal stride length varies with speed. As speed increases, both stride length "
        assessment += "and cadence should increase, with elite runners typically maintaining cadence in the "
        assessment += "170-190 steps/minute range across a wide range of speeds."
        
        return assessment
    
    class StancePhaseDetectorSide:
        def __init__(self, calibration_frames=90, stance_threshold_ratio=0.018, visibility_threshold=0.5):
            """Initialize the stance phase detector.
            Args:
                calibration_frames (int): Number of frames for ground and runner height calibration.
                stance_threshold_ratio (float): Threshold for stance detection, as a ratio of the
                                                runner's apparent height in normalized coordinates.
                                                (e.g., 0.04 means 4% of runner's apparent height).
                visibility_threshold (float): Minimum visibility score for a landmark to be considered.
            """
            self.calibration_frames = calibration_frames
            self.stance_threshold_ratio = stance_threshold_ratio
            self.visibility_threshold = visibility_threshold
            self.frame_count = 0

            self.ground_y_samples_normalized = []
            self.runner_height_samples_normalized = [] # Store normalized runner height during calibration

            self.ground_level_normalized = None
            self.avg_runner_height_normalized = None # Runner's height in normalized [0,1] coordinates

            # For optional velocity calculation
            self.foot_y_history = {'left': [], 'right': []}
            self.max_history_len = 3 # Frames for velocity calculation buffer
            self.foot_velocity_threshold_normalized = 0.01 # Normalized velocity; tune this
            
            self.foot_landmark_names = {
                'right': ['right_heel', 'right_foot_index', 'right_ankle'], # Ankle for reference
                'left': ['left_heel', 'left_foot_index', 'left_ankle']
            }
            self.head_landmark_names = ['right_ear', 'left_ear', 'nose']


        def _collect_calibration_data(self, landmarks):
            """Collects data for ground level and runner's apparent height during calibration."""
            
            # 1. Collect ground y-coordinates (lowest point of any visible foot)
            lowest_foot_y_this_frame = []
            for side in ['left', 'right']:
                for lm_name in self.foot_landmark_names[side][:2]: # Heel and foot_index
                    if lm_name in landmarks and landmarks[lm_name][3] >= self.visibility_threshold:
                        lowest_foot_y_this_frame.append(landmarks[lm_name][1])
            
            if lowest_foot_y_this_frame:
                self.ground_y_samples_normalized.append(max(lowest_foot_y_this_frame))

            # 2. Collect runner's apparent height (ears/nose to lowest foot)
            head_y_candidates = []
            for lm_name in self.head_landmark_names:
                if lm_name in landmarks and landmarks[lm_name][3] >= self.visibility_threshold:
                    head_y_candidates.append(landmarks[lm_name][1])
            
            if not head_y_candidates or not lowest_foot_y_this_frame:
                return # Not enough data in this frame

            min_head_y_normalized = min(head_y_candidates)
            max_foot_y_normalized = max(lowest_foot_y_this_frame) # Use the already found lowest foot y

            runner_height_normalized_this_frame = max_foot_y_normalized - min_head_y_normalized
            if runner_height_normalized_this_frame > 0.1: # Basic sanity check (runner is at least 10% of image height)
                self.runner_height_samples_normalized.append(runner_height_normalized_this_frame)

        def _finalize_calibration(self):
            """Calculates ground level and average runner height from collected samples."""
            if self.ground_y_samples_normalized:
                # Using median or percentile for robustness
                self.ground_level_normalized = np.percentile(self.ground_y_samples_normalized, 85) # e.g., 85th percentile
                print(f"Calibrated ground_level_normalized: {self.ground_level_normalized:.3f}")
            else:
                print("Warning: Could not calibrate ground level.")
                # Fallback or raise error if critical
                self.ground_level_normalized = 0.9 # Default if no data

            if self.runner_height_samples_normalized:
                self.avg_runner_height_normalized = np.median(self.runner_height_samples_normalized)
                print(f"Calibrated avg_runner_height_normalized: {self.avg_runner_height_normalized:.3f}")
            else:
                print("Warning: Could not calibrate runner's apparent height.")
                self.avg_runner_height_normalized = 0.7 # Default (runner occupies 70% of frame height)
                
        def _get_foot_vertical_velocity(self, side, current_lowest_y):
            """Estimates vertical velocity of the foot (normalized units per frame)."""
            self.foot_y_history[side].append(current_lowest_y)
            if len(self.foot_y_history[side]) > self.max_history_len:
                self.foot_y_history[side].pop(0)

            if len(self.foot_y_history[side]) < 2:
                return 0.0 # Not enough history

            # Simple difference; more advanced could be slope of linear regression over history
            velocity = self.foot_y_history[side][-1] - self.foot_y_history[side][-2]
            return velocity

        def detect_stance_phase_side(self, landmarks):
            """
            Detect if a runner is in stance phase from side view landmarks.
            Relies on normalized coordinates from BlazePose.
            """
            if self.frame_count < self.calibration_frames:
                self._collect_calibration_data(landmarks)
                self.frame_count += 1
                if self.frame_count == self.calibration_frames:
                    self._finalize_calibration()
                return {'is_stance_phase': False, 'stance_foot': None, 'confidence': 0.0, 'debug_info': "calibrating"}

            if self.ground_level_normalized is None or self.avg_runner_height_normalized is None:
                # This case should ideally be handled by forcing calibration or having robust defaults
                return {'is_stance_phase': False, 'stance_foot': None, 'confidence': 0.0, 'debug_info': "calibration_not_complete"}

            # Calculate dynamic stance threshold based on runner's apparent height
            # This threshold is in normalized units
            dynamic_stance_threshold_normalized = self.avg_runner_height_normalized * self.stance_threshold_ratio
            
            stance_candidates = []

            for side in ['left', 'right']:
                foot_points_y = []
                relevant_landmarks = self.foot_landmark_names[side][:2] # Heel and foot_index

                all_landmarks_visible_for_side = True
                for lm_name in relevant_landmarks:
                    if not (lm_name in landmarks and landmarks[lm_name][3] >= self.visibility_threshold):
                        all_landmarks_visible_for_side = False
                        break
                    foot_points_y.append(landmarks[lm_name][1])
                
                if not all_landmarks_visible_for_side or not foot_points_y:
                    stance_candidates.append({'side': side, 'in_stance': False, 'confidence': 0.0, 'lowest_y': float('inf')})
                    continue

                lowest_y_of_foot = max(foot_points_y) # Max because Y increases downwards (normalized)
                distance_from_ground_normalized = abs(self.ground_level_normalized - lowest_y_of_foot)
                
                # Check 1: Proximity to ground
                is_near_ground = distance_from_ground_normalized <= dynamic_stance_threshold_normalized
                
                # Check 2: Low vertical velocity (optional, but recommended)
                # vertical_velocity = self._get_foot_vertical_velocity(side, lowest_y_of_foot)
                # is_stable_vertically = abs(vertical_velocity) < self.foot_velocity_threshold_normalized
                # is_in_stance = is_near_ground and is_stable_vertically
                
                is_in_stance = is_near_ground # Keeping it simpler for now, add velocity later if needed

                confidence = 0.0
                if dynamic_stance_threshold_normalized > 1e-6: # Avoid division by zero
                    confidence = max(0.0, 1.0 - (distance_from_ground_normalized / (dynamic_stance_threshold_normalized * 2)))
                
                if not is_in_stance: # If not in stance by primary criteria, confidence should be low reflecting that
                    confidence *= 0.5 

                stance_candidates.append({
                    'side': side, 
                    'in_stance': is_in_stance, 
                    'confidence': confidence, 
                    'lowest_y': lowest_y_of_foot,
                    # 'debug_vel': vertical_velocity # if using
                })

            # Determine overall stance phase and foot
            # This logic prioritizes a single stance foot, which is typical for running.
            left_candidate = next(c for c in stance_candidates if c['side'] == 'left')
            right_candidate = next(c for c in stance_candidates if c['side'] == 'right')

            is_stance_phase = False
            stance_foot = None
            final_confidence = 0.0
            # debug_info = {'left': left_candidate, 'right': right_candidate, 'threshold': dynamic_stance_threshold_normalized}


            if left_candidate['in_stance'] and right_candidate['in_stance']:
                # Both detected in stance: choose based on which is lower (more planted) or higher confidence
                # This is rare in running (should be flight or single stance)
                # Could indicate walking, start/stop, or error. For pure running, one should be chosen.
                if left_candidate['lowest_y'] > right_candidate['lowest_y']: # Left is lower
                    if left_candidate['confidence'] > right_candidate['confidence'] * 0.8: # Bias for the lower one
                        stance_foot = 'left'
                        final_confidence = left_candidate['confidence']
                    else:
                        stance_foot = 'right'
                        final_confidence = right_candidate['confidence']
                else: # Right is lower or same height
                    if right_candidate['confidence'] > left_candidate['confidence'] * 0.8:
                        stance_foot = 'right'
                        final_confidence = right_candidate['confidence']
                    else:
                        stance_foot = 'left'
                        final_confidence = left_candidate['confidence']
                is_stance_phase = True # If both in stance, phase is true.
            elif left_candidate['in_stance']:
                is_stance_phase = True
                stance_foot = 'left'
                final_confidence = left_candidate['confidence']
            elif right_candidate['in_stance']:
                is_stance_phase = True
                stance_foot = 'right'
                final_confidence = right_candidate['confidence']
            else:
                # No stance phase, confidence is how close the closest foot was (max of non-stance confidences)
                final_confidence = max(left_candidate['confidence'], right_candidate['confidence'])

            return {
                'is_stance_phase': is_stance_phase, 
                'stance_foot': stance_foot, 
                'confidence': final_confidence,
                # 'debug_info': debug_info # For inspecting values
            }
    
    def detect_stance_phase_side(self, landmarks):
        """Attempting a wrapper around G rewrite"""
    
        if not hasattr(self, '_stance_detector_side'):
            self._stance_detector_side = self.StancePhaseDetectorSide()
            # Pass image height if available
            # if hasattr(self, 'image_height'):
            #     self._stance_detector.image_height = self.image_height
        
        # Use the detector to determine stance phase
        return self._stance_detector_side.detect_stance_phase_side(landmarks)

    def draw_side_analysis(self, image, landmarks, frame_number):
        """Draw pose landmarks and metrics on image."""
        # Draw skeleton
        self.mp_drawing.draw_landmarks(
            image,
            landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        # Add metrics text
        h, w, _ = image.shape
        metrics_text = [
            f"Frame: {frame_number}",
            "",
            "FOOT STRIKE AND LANDING METRICS",
            f"Foot Strike: {self.side_metrics['strike_pattern'][-1]}",
            f"Foot Strike Confidence: {self.side_metrics['strike_confidence'][-1]}",
            # f"Foot Strike Foot Angle: {self.side_metrics['strike_foot_angle'][-1]:.2f}",
            # f"Foot Strike Ankle Angle: {self.side_metrics['strike_ankle_angle'][-1]:.2f}",
            # f"Foot Strike Landing Stiffness: {self.side_metrics['strike_landing_stiffness'][-1]}",
            # f"Foot Landing Category: {self.side_metrics['foot_landing_position_category'][-1]:}",
            f"Foot Landing Under CoM: {self.side_metrics['foot_landing_is_under_center_of_mass'][-1]:.1f}",
            f"Foot Distance in cm: {self.side_metrics['foot_landing_distance_from_center_in_cm'][-1]:.2f} cm",
            "",
            "KNEE METRICS",
            f"Knee Angle Left: {self.side_metrics['knee_angle_left'][-1]:.1f}*",
            f"Knee Angle Right: {self.side_metrics['knee_angle_right'][-1]:.1f}*",
            "",
            "STRIDE METRICS",
            f"Stride Instant Estimate: {self.side_metrics['stride_instantaneous_estimate_cm'][-1]:.2f}",
            f"Stride Length: {self.side_metrics['stride_length_cm'][-1]:.2f} cm",
            f"Stride Length Normalized: {self.side_metrics['normalized_stride_length'][-1]:.2f}",
            f"Stride Frequency / Cadence: {self.side_metrics['stride_frequency'][-1]}",
            "",
            "TRUNK AND ARM METRICS",
            f"Trunk Angle: {self.side_metrics['trunk_angle_degrees'][-1]:.2f}*",
            f"Trunk Angle Optimal?: {self.side_metrics['trunk_angle_is_optimal'][-1]}",
            f"Trunk Angle Assessment: {self.side_metrics['trunk_angle_assessment'][-1]}",
            f"Trunk Angle Confidence: {self.side_metrics['trunk_angle_confidence'][-1]:.1f}",
            f"Upper Arm Angle: {self.side_metrics['upper_arm_angle'][-1]:.1f}*",
            f"Elbow Angle: {self.side_metrics['elbow_angle'][-1]:.1f}*",
            f"Hand Position: {self.side_metrics['hand_position'][-1]}",
            f"Arm Swing Amplitude: {self.side_metrics['arm_swing_amplitude'][-1]}",
            f"Arm Swing Symmetry: {self.side_metrics['arm_swing_symmetry'][-1]}",
            "",
            "STANCE METRICS",
            f"Stance Detected: {self.side_metrics['stance_phase_detected'][-1]}",
    
        ]
        
        for i, text in enumerate(metrics_text):
            cv2.putText(image, text, (10, 30 + i*30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return image
    
    def save_side_metrics(self, output_path):
        """Save extracted metrics to CSV file."""
        df = pd.DataFrame(self.side_metrics)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Metrics saved to {output_path}")
        return df
    
    def merge_side_watch_data(self, metrics_df, watch_data_path):
        """Merge extracted metrics with watch data based on timestamp."""
        watch_df = pd.read_csv(watch_data_path)
        
        # Assuming watch data has a timestamp column
        # Interpolate watch data to match video frames
        merged_df = pd.merge_asof(
            metrics_df.sort_values('timestamp'),
            watch_df.sort_values('timestamp'),
            on='timestamp',
            direction='nearest'
        )
        
        return merged_df
    
    def extract_rear_metrics(self, landmarks, frame_number, timestamp):
        """Extract running biomechanics metrics from a single frame
        for a video shot from the side of a runner, AKA the frontal plane."""
        # Get normalized landmark positions
        landmark_coords = {}
        for name, landmark_id in self.key_points.items():
            landmark = landmarks.landmark[landmark_id]
            landmark_coords[name] = (landmark.x, landmark.y, landmark.z, landmark.visibility)
        
        # Detect stance phase rear
        stance_phase_rear = self.detect_stance_phase_rear(landmark_coords)
        self.rear_metrics['stance_phase_detected'].append(stance_phase_rear['is_stance_phase'])
        self.rear_metrics['stance_foot'].append(stance_phase_rear['stance_foot'])
        self.rear_metrics['stance_confidence'].append(stance_phase_rear['confidence'])

        # Store basic timestamp data
        self.rear_metrics['timestamp'].append(timestamp)
        self.rear_metrics['frame_number'].append(frame_number)
        
        # Calculate foot crossover
        foot_crossover = self.calculate_foot_crossover(landmark_coords)
        self.rear_metrics['left_foot_crossover'].append(foot_crossover["left_foot_crossover"])
        self.rear_metrics['right_foot_crossover'].append(foot_crossover["right_foot_crossover"])
        self.rear_metrics['left_distance_from_midline'].append(foot_crossover["left_distance_from_midline"])
        self.rear_metrics['right_distance_from_midline'].append(foot_crossover["right_distance_from_midline"])
        
        # Calculate hip drop
        hip_drop = self.calculate_hip_drop(landmark_coords)
        self.rear_metrics['hip_drop_value'].append(hip_drop["hip_drop_value"])
        self.rear_metrics['hip_drop_direction'].append(hip_drop["hip_drop_direction"])
        self.rear_metrics['hip_drop_severity'].append(hip_drop["severity"])

        # Calculate pelic tilt angle
        pelvic_tilt = self.calculate_pelvic_tilt(landmark_coords)
        self.rear_metrics['pelvic_tilt_angle'].append(pelvic_tilt["tilt_angle_degrees"])
        self.rear_metrics['pelvic_tilt_elevated_side'].append(pelvic_tilt["elevated_side"])
        self.rear_metrics['pelvic_tilt_severity'].append(pelvic_tilt["severity"])
        self.rear_metrics['pelvic_tilt_normalized'].append(pelvic_tilt["normalized_tilt"])
        
        # Calculate knee_alignment
        knee_alignment = self.calculate_knee_alignment(landmark_coords)
        self.rear_metrics['left_knee_valgus'].append(knee_alignment["left_knee_valgus"])
        self.rear_metrics['right_knee_valgus'].append(knee_alignment["right_knee_valgus"])   
        self.rear_metrics['left_knee_varus'].append(knee_alignment["left_knee_varus"])
        self.rear_metrics['right_knee_varus'].append(knee_alignment["right_knee_varus"]) 
        self.rear_metrics['left_knee_normalized_deviation'].append(knee_alignment["left_normalized_deviation"])
        self.rear_metrics['right_knee_normalized_deviation'].append(knee_alignment["right_normalized_deviation"])    
        self.rear_metrics['knee_severity_left'].append(knee_alignment["severity_left"])
        self.rear_metrics['knee_severity_right'].append(knee_alignment["severity_right"]) 

        # Calculate ankle_inversion
        ankle_inversion = self.calculate_ankle_inversion(landmark_coords)
        self.rear_metrics['left_ankle_inversion_value'].append(ankle_inversion["left_inversion_value"])
        self.rear_metrics['right_ankle_inversion_value'].append(ankle_inversion["right_inversion_value"])
        self.rear_metrics['left_ankle_normalized_value'].append(ankle_inversion["left_normalized"])
        self.rear_metrics['right_ankle_normalized_value'].append(ankle_inversion["right_normalized"])
        self.rear_metrics['left_ankle_pattern'].append(ankle_inversion["left_pattern"])
        self.rear_metrics['right_ankle_pattern'].append(ankle_inversion["right_pattern"])
        self.rear_metrics['left_ankle_severity'].append(ankle_inversion["left_severity"])
        self.rear_metrics['right_ankle_severity'].append(ankle_inversion["right_severity"])
        self.rear_metrics['left_ankle_angle'].append(ankle_inversion["left_foot_angle"])
        self.rear_metrics['right_ankle_angle'].append(ankle_inversion["right_foot_angle"])

        # Estimate step_width
        step_width = self.calculate_step_width(landmark_coords)
        self.rear_metrics['step_width'].append(step_width)
        
        # Detect stride_symmetry
        stride_symmetry = self.calculate_stride_symmetry(landmark_coords)
        self.rear_metrics['symmetry'].append(stride_symmetry)

        # Detect arm_swing_symmetry,
        arm_swing_mechanics = self.calculate_arm_swing_mechanics(landmark_coords)
        self.rear_metrics['vertical_elbow_diff'].append(arm_swing_mechanics['vertical_elbow_diff'])
        self.rear_metrics['normalized_vertical_diff'].append(arm_swing_mechanics['normalized_vertical_diff'])
        self.rear_metrics['left_elbow_angle'].append(arm_swing_mechanics['left_elbow_angle'])
        self.rear_metrics['right_elbow_angle'].append(arm_swing_mechanics['right_elbow_angle'])
        self.rear_metrics['normalized_shoulder_diff'].append(arm_swing_mechanics['normalized_shoulder_diff'])
        self.rear_metrics['normalized_shoulder_width'].append(arm_swing_mechanics['normalized_shoulder_width'])
        self.rear_metrics['arm_height_symmetry'].append(arm_swing_mechanics['arm_height_symmetry'])
        self.rear_metrics['elbow_angle_left'].append(arm_swing_mechanics['elbow_angle_left'])
        self.rear_metrics['elbow_angle_right'].append(arm_swing_mechanics['elbow_angle_right'])
        self.rear_metrics['left_wrist_crossover'].append(arm_swing_mechanics['left_wrist_crossover'])
        self.rear_metrics['right_wrist_crossover'].append(arm_swing_mechanics['right_wrist_crossover'])
        self.rear_metrics['shoulder_rotation'].append(arm_swing_mechanics['shoulder_rotation'])

    def calculate_foot_crossover(self, landmarks, threshold=0.25):
        """
        Check for feet being too close to or crossing the body's midline (from rear view).
        
        From rear view perspective:
        - Ideal: Left foot stays in left lane, right foot stays in right lane
        - Issue: Feet cross or get too close to midline (medial foot placement)
        
        Future consideration would be calculating the angle from the hip marker to the knee for extreme values.

        Parameters:
        -----------
        landmarks : dict
            Dictionary containing body landmark coordinates with keys like 'left_hip', 'right_foot_index', etc.
        threshold : float, default=0.25
            Proportion of hip width that determines acceptable proximity to midline.
            Lower values are stricter (less crossover allowed).
        
        Returns:
        --------
        dict
            Information about foot crossover including boolean flags and distances from midline.
        """
        # Calculate hip center and width for reference
        left_hip_x = landmarks['left_hip'][0]
        right_hip_x = landmarks['right_hip'][0]
        hip_center_x = (left_hip_x + right_hip_x) / 2
        hip_width = abs(left_hip_x - right_hip_x)
        
        # Get foot positions
        left_foot_x = landmarks['left_foot_index'][0]
        right_foot_x = landmarks['right_foot_index'][0]
        
        # Calculate distances from center line (negative = right of midline, positive = left of midline)
        left_distance = left_foot_x - hip_center_x
        right_distance = right_foot_x - hip_center_x
        
        # Check if feet cross or get too close to midline beyond the threshold
        # From rear view: Left foot crossing over = left foot is left of ideal position (too close to or past midline)
        # From rear view: Right foot crossing over = right foot is right of ideal position (too close to or past midline)
        crossover_left = left_distance > -threshold * hip_width  # Left foot too far to the left (too close to midline)
        crossover_right = right_distance < threshold * hip_width  # Right foot too far to the right (too close to midline)
        
        return {
            "left_foot_crossover": crossover_left,
            "right_foot_crossover": crossover_right,
            "left_distance_from_midline": left_distance,
            "right_distance_from_midline": right_distance
        }
    
    def calculate_hip_drop(self, landmarks, threshold=0.015):
        """
        Detect hip drop (Trendelenburg gait) during running stance phase.
        
        Hip drop occurs when the pelvis tilts laterally during single-leg support,
        indicating potential weakness in hip abductor muscles (primarily gluteus medius).
        
        Parameters:
        -----------
        landmarks : dict
            Dictionary containing body landmark coordinates with keys like 'left_hip', 'right_hip'.
            Coordinates should be normalized (0-1 range) relative to image dimensions.
        threshold : float, default=0.015
            The minimum difference in normalized hip height to classify as hip drop.
            Typically 1-2% of image height is appropriate for detection.
            Can be adjusted based on camera angle and distance.
        
        Returns:
        --------
        dict
            Contains the hip drop value (positive = right hip drops) and classification.
            
        Notes:
        ------
        For clinical assessment:
        - Mild: < 3° drop (roughly 0.01-0.02 in normalized coordinates)
        - Moderate: 3-10° drop (roughly 0.02-0.05)
        - Severe: > 10° drop (roughly > 0.05)
        
        This function should ideally be used during single-leg stance phases
        for accurate assessment, not during flight phases.
        """
        left_hip_y = landmarks['left_hip'][1]
        right_hip_y = landmarks['right_hip'][1]
        
        # Calculate hip height difference (positive = right hip is lower/dropped)
        hip_drop = right_hip_y - left_hip_y
        
        # Determine severity based on clinical thresholds
        if abs(hip_drop) < threshold:
            direction = "neutral"
            severity = "none"
        else:
            direction = "right" if hip_drop > 0 else "left"
            if abs(hip_drop) < 0.03:
                severity = "mild"
            elif abs(hip_drop) < 0.05:
                severity = "moderate"
            else:
                severity = "severe"
        
        return {
            "hip_drop_value": hip_drop,
            "hip_drop_direction": direction,
            "severity": severity
        }

    
    def calculate_pelvic_tilt(self, landmarks):
        """
        Calculate lateral pelvic tilt angle in the frontal plane during running.
        
        Measures lateral pelvic tilt (frontal plane) which can indicate:
        - Hip abductor weakness (primarily gluteus medius)
        - Leg length discrepancy (functional or anatomical)
        - Compensation patterns for other biomechanical issues
        - Potential IT band, low back, or knee injury risk
        
        Parameters:
        -----------
        landmarks : dict
            Dictionary containing body landmark coordinates with keys like 'left_hip', 'right_hip'.
            Coordinates should be normalized (0-1 range) relative to image dimensions.
        
        Returns:
        --------
        dict
            Contains tilt angle in degrees, clinical interpretation and normalized values.
            Positive angles indicate right side elevated, negative angles indicate left side elevated.
        
        Notes:
        ------
        Clinical reference:
        - Normal range: ±2° during stance phase
        - Mild tilt: 2-5° (potential early intervention)
        - Moderate: 5-10° (intervention recommended)
        - Severe: >10° (significant dysfunction)
        
        This measures frontal plane motion only and differs from anterior/posterior pelvic tilt 
        (sagittal plane), which requires side-view analysis.
        """
        # Extract hip coordinates
        left_hip_x, left_hip_y = landmarks['left_hip'][0], landmarks['left_hip'][1]
        right_hip_x, right_hip_y = landmarks['right_hip'][0], landmarks['right_hip'][1]
        
        # Calculate horizontal distance between hips for normalization
        hip_distance = abs(right_hip_x - left_hip_x)
        
        # Calculate tilt angle (positive = right side up, negative = left side up)
        tilt_angle = np.degrees(np.arctan2(right_hip_y - left_hip_y,
                                            right_hip_x - left_hip_x))
        
        # Apply coordinate system correction if needed
        # Note: In many vision systems, y increases downward, so we may need to negate
        # Uncomment if your coordinate system has y increasing upward
        # tilt_angle = -tilt_angle
        
        # Determine severity
        if abs(tilt_angle) <= 2:
            severity = "normal"
        elif abs(tilt_angle) <= 5:
            severity = "mild"
        elif abs(tilt_angle) <= 10:
            severity = "moderate"
        else:
            severity = "severe"
        
        # Determine elevated side
        if abs(tilt_angle) <= 2:
            elevated_side = "neutral"
        else:
            elevated_side = "left" if tilt_angle > 0 else "right"
        
        return {
            "tilt_angle_degrees": tilt_angle,
            "elevated_side": elevated_side,
            "severity": severity,
            "normalized_tilt": tilt_angle / (np.arctan2(0.1, hip_distance) * 180/np.pi)
        }

    
    def calculate_knee_alignment(self, landmarks):
        """
        Assess knee alignment during running to detect valgus (knock-knee) or varus (bow-leg) patterns.
        
        Dynamic knee valgus is particularly concerning in runners as it indicates:
        - Potential weakness in hip abductors/external rotators
        - Excessive foot pronation
        - Risk factor for patellofemoral pain syndrome, ACL injuries, and IT band syndrome
        
        Parameters:
        -----------
        landmarks : dict
            Dictionary containing body landmark coordinates with keys like 'left_hip', 'left_knee', etc.
        
        Returns:
        --------
        dict
            Contains assessment of knee alignment patterns and deviation measurements.
        """
        # Extract landmark coordinates
        left_hip_x = landmarks['left_hip'][0]
        left_knee_x = landmarks['left_knee'][0]
        left_ankle_x = landmarks['left_ankle'][0]
        
        right_hip_x = landmarks['right_hip'][0]
        right_knee_x = landmarks['right_knee'][0]
        right_ankle_x = landmarks['right_ankle'][0]
        
        # Calculate alignment metrics
        # For left leg (viewed from behind):
        # - Valgus = knee is more medial (right) than the hip-ankle line
        # - Varus = knee is more lateral (left) than the hip-ankle line
        left_hip_to_ankle_x = left_ankle_x - left_hip_x
        if left_hip_to_ankle_x != 0:  # Avoid division by zero
            left_expected_knee_x = left_hip_x + left_hip_to_ankle_x * 0.5  # Simplified linear interpolation
            left_deviation = left_knee_x - left_expected_knee_x
            left_hip_width = abs(left_hip_x - right_hip_x)
            # Normalize by hip width to account for different runner sizes and camera distances
            left_normalized_deviation = left_deviation / left_hip_width if left_hip_width else 0
        else:
            left_normalized_deviation = 0
        
        # For right leg (viewed from behind):
        # - Valgus = knee is more medial (left) than the hip-ankle line
        # - Varus = knee is more lateral (right) than the hip-ankle line
        right_hip_to_ankle_x = right_ankle_x - right_hip_x
        if right_hip_to_ankle_x != 0:  # Avoid division by zero
            right_expected_knee_x = right_hip_x + right_hip_to_ankle_x * 0.5  # Simplified linear interpolation
            right_deviation = right_knee_x - right_expected_knee_x
            right_hip_width = abs(left_hip_x - right_hip_x)
            right_normalized_deviation = right_deviation / right_hip_width if right_hip_width else 0
        else:
            right_normalized_deviation = 0
        
        # Clinical thresholds for classification
        threshold = 0.1  # 10% of hip width as a threshold for concern
        
        # Determine alignment patterns (from rear view)
        left_valgus = left_normalized_deviation < -threshold  # Knee is too far medial (right)
        left_varus = left_normalized_deviation > threshold    # Knee is too far lateral (left)
        
        right_valgus = right_normalized_deviation > threshold  # Knee is too far medial (left)
        right_varus = right_normalized_deviation < -threshold  # Knee is too far lateral (right)
        
        # Return comprehensive assessment
        return {
            "left_knee_valgus": left_valgus,
            "left_knee_varus": left_varus,
            "right_knee_valgus": right_valgus,
            "right_knee_varus": right_varus,
            "left_normalized_deviation": left_normalized_deviation,
            "right_normalized_deviation": right_normalized_deviation,
            "severity_left": "normal" if abs(left_normalized_deviation) < threshold else 
                            "mild" if abs(left_normalized_deviation) < threshold*1.5 else
                            "moderate" if abs(left_normalized_deviation) < threshold*2 else "severe",
            "severity_right": "normal" if abs(right_normalized_deviation) < threshold else 
                            "mild" if abs(right_normalized_deviation) < threshold*1.5 else
                            "moderate" if abs(right_normalized_deviation) < threshold*2 else "severe"
        }
    
    def calculate_ankle_inversion(self, landmarks):
        """
        Measure ankle inversion/eversion patterns during running.
        
        Inversion = ankle rolls outward (supination)
        Eversion = ankle rolls inward (pronation)
        
        From rear view:
        - Excessive inversion is linked to lateral ankle sprains and insufficient shock absorption
        - Excessive eversion is linked to medial tibial stress syndrome (shin splints) and plantar fasciitis
        
        Parameters:
        -----------
        landmarks : dict
            Dictionary containing body landmark coordinates with keys for ankle and heel positions
        
        Returns:
        --------
        dict
            Analysis of ankle inversion/eversion patterns including normalized measurements and clinical assessment
        """
        # Extract landmark coordinates
        left_ankle_x = landmarks['left_ankle'][0]
        left_heel_x = landmarks['left_heel'][0]
        left_foot_index_x = landmarks.get('left_foot_index', landmarks.get('left_toe', [0, 0]))[0]
        
        right_ankle_x = landmarks['right_ankle'][0]
        right_heel_x = landmarks['right_heel'][0]
        right_foot_index_x = landmarks.get('right_foot_index', landmarks.get('right_toe', [0, 0]))[0]
        
        # Calculate hip width for normalization
        hip_width = abs(landmarks['left_hip'][0] - landmarks['right_hip'][0])
        
        # Calculate inversion values (positive = inversion, negative = eversion)
        # From rear view:
        # Left foot: heel to the left of ankle = inversion, heel to the right = eversion
        # Right foot: heel to the right of ankle = inversion, heel to the left = eversion
        left_inversion = left_ankle_x - left_heel_x  
        right_inversion = right_heel_x - right_ankle_x
        
        # Normalize by hip width for better comparison across runners
        left_normalized = left_inversion / hip_width if hip_width else 0
        right_normalized = right_inversion / hip_width if hip_width else 0
        
        # Advanced: Calculate foot axis angle if toe landmarks are available
        if left_foot_index_x and right_foot_index_x:
            left_foot_angle = np.degrees(np.arctan2(landmarks['left_ankle'][1] - landmarks['left_heel'][1],
                                                left_ankle_x - left_heel_x))
            right_foot_angle = np.degrees(np.arctan2(landmarks['right_ankle'][1] - landmarks['right_heel'][1],
                                                right_heel_x - right_ankle_x))
        else:
            left_foot_angle = right_foot_angle = None
        
        # Classify based on clinical thresholds
        # Threshold values based on normalized measurements
        inversion_threshold = 0.03  # 3% of hip width
        
        # Determine patterns
        left_pattern = "neutral"
        if left_normalized > inversion_threshold:
            left_pattern = "inversion"
        elif left_normalized < -inversion_threshold:
            left_pattern = "eversion"
            
        right_pattern = "neutral"
        if right_normalized > inversion_threshold:
            right_pattern = "inversion"
        elif right_normalized < -inversion_threshold:
            right_pattern = "eversion"
        
        # Determine severity
        def get_severity(value):
            abs_value = abs(value)
            if abs_value < inversion_threshold:
                return "normal"
            elif abs_value < inversion_threshold*2:
                return "mild"
            elif abs_value < inversion_threshold*3:
                return "moderate"
            else:
                return "severe"
        
        return {
            "left_inversion_value": left_inversion,
            "right_inversion_value": right_inversion,
            "left_normalized": left_normalized,
            "right_normalized": right_normalized,
            "left_pattern": left_pattern,
            "right_pattern": right_pattern,
            "left_severity": get_severity(left_normalized),
            "right_severity": get_severity(right_normalized),
            "left_foot_angle": left_foot_angle,
            "right_foot_angle": right_foot_angle
        }

    
    def calculate_step_width(self, landmarks):
        """Distance between both feet."""
        left_foot_x = landmarks['left_foot_index'][0]
        right_foot_x = landmarks['right_foot_index'][0]
        
        step_width = abs(left_foot_x - right_foot_x) * 100  # Rough cm conversion
        return step_width

        
    def calculate_stride_symmetry(self, landmarks):
        """Compare stride or timing parameters over a cycle (requires frame history).
         Placeholder uses foot x-delta."""
        left_stride = landmarks['left_foot_index'][0] - landmarks['left_heel'][0]
        right_stride = landmarks['right_foot_index'][0] - landmarks['right_heel'][0]
        
        symmetry = (right_stride - left_stride) / max(abs(right_stride), abs(left_stride) + 1e-6)
        
        return symmetry

    def calculate_arm_swing_mechanics(self, landmarks):
        """
        Analyze arm swing mechanics during running from rear view.
        
        Efficient arm swing should:
        - Move primarily in sagittal plane (front-to-back)
        - Be symmetrical in timing and amplitude
        - Maintain roughly 90° elbow flexion
        - Counter-rotate with opposite leg
        - Not cross midline excessively
        
        Parameters:
        -----------
        landmarks : dict
            Dictionary containing body landmark coordinates
        
        Returns:
        --------
        dict
            Comprehensive analysis of arm swing mechanics
        """
        # Extract relevant landmarks
        left_shoulder = landmarks['left_shoulder']
        right_shoulder = landmarks['right_shoulder']
        left_elbow = landmarks['left_elbow']
        right_elbow = landmarks['right_elbow']
        left_wrist = landmarks['left_wrist'] 
        right_wrist = landmarks['right_wrist']
        
        # 1. Vertical symmetry - detect if arms are at different heights
        vertical_diff = abs(left_elbow[1] - right_elbow[1])
        hip_width = abs(landmarks['left_hip'][0] - landmarks['right_hip'][0])
        normalized_vertical_diff = vertical_diff / hip_width if hip_width else 0
        
        # 2. Crossover detection - arms crossing midline (from rear view)
        shoulder_midpoint_x = (left_shoulder[0] + right_shoulder[0]) / 2
        left_wrist_crossover = left_wrist[0] > shoulder_midpoint_x
        right_wrist_crossover = right_wrist[0] < shoulder_midpoint_x
        
        # 3. Elbow angle (flexion) calculation
        def calculate_angle(a, b, c):
            """Calculate angle between three points (b is the vertex)"""
            ba = np.array(a) - np.array(b)
            bc = np.array(c) - np.array(b)
            cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            cosine = max(min(cosine, 1.0), -1.0)  # Clip to avoid numerical errors
            return np.degrees(np.arccos(cosine))
        
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        # 4. Shoulder rotation (limited from rear view, but can detect excessive movement)
        shoulder_height_diff = abs(left_shoulder[1] - right_shoulder[1])
        normalized_shoulder_diff = shoulder_height_diff / hip_width if hip_width else 0
        
        # 5. Shoulder width stability (should be relatively constant during run)
        shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
        normalized_shoulder_width = shoulder_width / hip_width if hip_width else 0
        
        # Classification based on clinical thresholds
        sym_threshold = 0.05  # 5% of hip width
        
        return {
            # Raw measurements
            "vertical_elbow_diff": vertical_diff,
            "normalized_vertical_diff": normalized_vertical_diff,
            "left_elbow_angle": left_elbow_angle,
            "right_elbow_angle": right_elbow_angle, 
            "normalized_shoulder_diff": normalized_shoulder_diff,
            "normalized_shoulder_width": normalized_shoulder_width,
            
            # Clinical assessments
            "arm_height_symmetry": "good" if normalized_vertical_diff < sym_threshold else 
                                "moderate" if normalized_vertical_diff < sym_threshold*2 else "poor",
            "elbow_angle_left": "optimal" if 80 <= left_elbow_angle <= 110 else 
                                "too_straight" if left_elbow_angle > 110 else "too_bent",
            "elbow_angle_right": "optimal" if 80 <= right_elbow_angle <= 110 else 
                                "too_straight" if right_elbow_angle > 110 else "too_bent",
            "left_wrist_crossover": left_wrist_crossover,
            "right_wrist_crossover": right_wrist_crossover,
            "shoulder_rotation": "stable" if normalized_shoulder_diff < 0.03 else "excessive",
        }
    
    class StancePhaseDetectorRear:
        def __init__(self, calibration_frames_total=90, 
                    ground_zone_percentage=0.15, # Defines the bottom X% of foot's RoM as ground zone
                    visibility_threshold=0.5,
                    foot_landmarks_to_use=None): # Allows customization of landmarks
            """
            Initializes the rear-view stance phase detector.

            Args:
                calibration_frames_total (int): Number of frames to use for calibration.
                ground_zone_percentage (float): The bottom percentage of the foot's observed vertical 
                                            range of motion that defines the ground contact zone.
                                            (e.g., 0.3 means bottom 30%).
                visibility_threshold (float): Minimum landmark visibility to be considered.
                foot_landmarks_to_use (dict, optional): Specify which landmarks define the 'bottom'
                                                        of each foot. Defaults to heel and foot_index.
                                                        Example: {'left': ['left_ankle'], 'right': ['right_ankle']}
            """
            self.calibration_frames_total = calibration_frames_total
            self.ground_zone_percentage = ground_zone_percentage
            self.visibility_threshold = visibility_threshold
            
            self.frames_calibrated = 0
            
            # Stores all observed lowest Y positions of all visible feet during calibration
            self._foot_y_samples_normalized = [] 
            
            self.calibrated_overall_max_y = None # Lowest point any foot reached (ground plane proxy)
            self.calibrated_overall_min_y = None # Highest point any foot reached (peak swing proxy)
            self.ground_contact_entry_threshold_y = None # Y value above which foot is considered in stance zone

            if foot_landmarks_to_use is None:
                self.foot_landmarks_to_check = {
                    'left': ['left_foot_index', 'left_heel'],
                    'right': ['right_foot_index', 'right_heel']
                }
            else:
                self.foot_landmarks_to_check = foot_landmarks_to_use
            
            # Optional: For velocity checks (can be added later for more robustness)
            # self._foot_y_history = {'left': [], 'right': []}
            # self._max_velocity_history = 3
            # self._velocity_threshold_normalized = 0.008 # Needs tuning

        def _get_lowest_point_of_foot(self, landmarks, side_key):
            """Gets the lowest Y coordinate for a given foot, considering visibility."""
            foot_y_values = []
            if side_key not in self.foot_landmarks_to_check:
                return None

            for lm_name in self.foot_landmarks_to_check[side_key]:
                if lm_name in landmarks and landmarks[lm_name][3] >= self.visibility_threshold: # landmarks[lm_name] = (x,y,z,visibility)
                    foot_y_values.append(landmarks[lm_name][1])
            
            if not foot_y_values:
                return None 
            return max(foot_y_values) # Max Y is lowest on screen (normalized 0-1, 1 is bottom)

        def _collect_calibration_data(self, landmarks):
            """Collects the lowest Y position of each visible foot in the current frame."""
            left_lowest_y = self._get_lowest_point_of_foot(landmarks, 'left')
            right_lowest_y = self._get_lowest_point_of_foot(landmarks, 'right')

            if left_lowest_y is not None:
                self._foot_y_samples_normalized.append(left_lowest_y)
            if right_lowest_y is not None:
                self._foot_y_samples_normalized.append(right_lowest_y)

        def _finalize_calibration(self):
            """Calculates calibration parameters from the collected foot Y samples."""
            if not self._foot_y_samples_normalized or len(self._foot_y_samples_normalized) < self.calibration_frames_total * 0.5: # Require at least half of calibration frames to have some data
                print("Warning: Insufficient data for rear stance detection calibration. Using broad defaults.")
                self.calibrated_overall_max_y = 0.90 # Assumed ground
                self.calibrated_overall_min_y = 0.50 # Assumed peak swing
            else:
                # Use percentiles for robustness against extreme outliers
                self.calibrated_overall_max_y = np.percentile(self._foot_y_samples_normalized, 95) # Robust lowest point
                self.calibrated_overall_min_y = np.percentile(self._foot_y_samples_normalized, 5)  # Robust highest point

            # Ensure max_y (ground) is actually lower than min_y (peak swing)
            if self.calibrated_overall_max_y <= self.calibrated_overall_min_y + 0.05: # Add small buffer
                print("Warning: Rear stance calibration issue - foot motion range too small or ground not lower than peak swing. Adjusting.")
                # If very little motion or inverted, use the most extreme sample for max_y and estimate min_y
                if self._foot_y_samples_normalized:
                    self.calibrated_overall_max_y = max(self._foot_y_samples_normalized)
                    self.calibrated_overall_min_y = min(self.calibrated_overall_max_y - 0.1, min(self._foot_y_samples_normalized)) # Ensure some range
                else: # No samples at all
                    self.calibrated_overall_max_y = 0.90
                    self.calibrated_overall_min_y = 0.50


            height_range = self.calibrated_overall_max_y - self.calibrated_overall_min_y
            
            if height_range <= 0.02: # If very small range (e.g., standing still, or bad calibration)
                print("Warning: Very small foot motion range detected in rear view calibration. Threshold may be sensitive.")
                # Set threshold very close to the detected "ground"
                self.ground_contact_entry_threshold_y = self.calibrated_overall_max_y - 0.015 
            else:
                # The stance zone starts this much *above* the lowest point (max_y)
                # Or, equivalently, (1 - ground_zone_percentage) of the range from the highest point (min_y)
                self.ground_contact_entry_threshold_y = self.calibrated_overall_max_y - (height_range * self.ground_zone_percentage)

            print(f"Rear Stance Calibrated: OverallMaxY={self.calibrated_overall_max_y:.3f} (Ground), "
                f"OverallMinY={self.calibrated_overall_min_y:.3f} (Peak Swing), "
                f"ContactEntryThresholdY={self.ground_contact_entry_threshold_y:.3f}")

        def detect_stance_phase(self, landmarks):
            """
            Detects stance phase from rear view landmarks.

            Args:
                landmarks (dict): Dictionary of landmark coordinates (x,y,z,visibility).
            
            Returns:
                dict: {'is_stance_phase': bool, 'stance_foot': str|None, 'confidence': float}
            """
            if self.frames_calibrated < self.calibration_frames_total:
                self._collect_calibration_data(landmarks)
                self.frames_calibrated += 1
                if self.frames_calibrated == self.calibration_frames_total:
                    self._finalize_calibration()
                return {'is_stance_phase': False, 'stance_foot': None, 'confidence': 0.0, 'debug': "calibrating"}

            if self.ground_contact_entry_threshold_y is None: # Calibration failed or not yet run
                print("Error: Rear stance detector not calibrated.")
                return {'is_stance_phase': False, 'stance_foot': None, 'confidence': 0.0, 'debug': "not_calibrated"}

            left_lowest_y = self._get_lowest_point_of_foot(landmarks, 'left')
            right_lowest_y = self._get_lowest_point_of_foot(landmarks, 'right')

            is_stance_phase = False
            stance_foot = None
            confidence = 0.0 # Default confidence for no stance / flight

            # Check for missing landmarks for either foot
            if left_lowest_y is None and right_lowest_y is None:
                return {'is_stance_phase': False, 'stance_foot': None, 'confidence': 0.0, 'debug': "no_foot_data"}

            # Determine if each foot is in the stance zone
            # A foot is in stance if its Y value is at or below the entry threshold
            left_in_stance = (left_lowest_y is not None) and (left_lowest_y >= self.ground_contact_entry_threshold_y)
            right_in_stance = (right_lowest_y is not None) and (right_lowest_y >= self.ground_contact_entry_threshold_y)
            
            # Determine overall phase and stance foot
            if left_in_stance and right_in_stance:
                is_stance_phase = True
                # Both feet in stance zone (double support or error) - choose the truly lower one
                stance_foot = 'left' if left_lowest_y > right_lowest_y else 'right'
                # Calculate confidence based on how deep the chosen foot is in the stance zone
                active_foot_y = left_lowest_y if stance_foot == 'left' else right_lowest_y
            elif left_in_stance:
                is_stance_phase = True
                stance_foot = 'left'
                active_foot_y = left_lowest_y
            elif right_in_stance:
                is_stance_phase = True
                stance_foot = 'right'
                active_foot_y = right_lowest_y
            
            # Calculate confidence
            if is_stance_phase and active_foot_y is not None:
                # Zone for confidence: from ground_contact_entry_threshold_y to calibrated_overall_max_y
                stance_zone_height = self.calibrated_overall_max_y - self.ground_contact_entry_threshold_y
                if stance_zone_height > 1e-5: # Avoid division by zero if threshold is at max_y
                    depth_ratio = (active_foot_y - self.ground_contact_entry_threshold_y) / stance_zone_height
                    confidence = 0.5 + min(max(depth_ratio, 0), 1) * 0.49 # Scale from 0.5 (at threshold) to 0.99 (at max_y)
                else: # Foot is at the threshold which is also the max_y
                    confidence = 0.6 # Reasonably confident it's on the exact line
            else: # No stance phase (flight)
                # Confidence for flight: how far is the *highest* of the two feet from the stance threshold?
                # (Higher foot is min_y of the two)
                if left_lowest_y is not None or right_lowest_y is not None:
                    highest_current_foot_y = min(left_lowest_y if left_lowest_y is not None else float('inf'), 
                                                right_lowest_y if right_lowest_y is not None else float('inf'))
                    if highest_current_foot_y < self.ground_contact_entry_threshold_y: # If it's above the stance threshold
                        flight_zone_height = self.ground_contact_entry_threshold_y - self.calibrated_overall_min_y
                        if flight_zone_height > 1e-5:
                            clearance_ratio = (self.ground_contact_entry_threshold_y - highest_current_foot_y) / flight_zone_height
                            confidence = 0.5 + min(max(clearance_ratio, 0), 1) * 0.49 # Confident it's in flight
                        else:
                            confidence = 0.6 # Clearly in flight, but small range
                    else:
                        confidence = 0.2 # Near threshold but not in stance
                else:
                    confidence = 0.1 # No foot data but not in stance

            return {'is_stance_phase': is_stance_phase, 'stance_foot': stance_foot, 'confidence': round(confidence, 3)}
    
    def detect_stance_phase_rear(self, landmarks):
        """Attempting a wrapper around G rewrite"""

        if not hasattr(self, '_stance_detector_rear'):
            self._stance_detector_rear = self.StancePhaseDetectorRear()
        
        return self._stance_detector_rear.detect_stance_phase(landmarks)


    def draw_rear_analysis(self, image, landmarks, frame_number):
        """Draw pose landmarks and metrics on image."""
        # Draw skeleton
        self.mp_drawing.draw_landmarks(
            image,
            landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        # Add metrics text
        h, w, _ = image.shape
        metrics_text_left = [
            f"Frame: {frame_number}",
            "",
            "HIP METRICS",
            f"Hip Drop Direction: {self.rear_metrics['hip_drop_direction'][-1]}",
            f"Hip Drop Value: {self.rear_metrics['hip_drop_value'][-1]:.3f}",
            f"Hip Drop Severity: {self.rear_metrics['hip_drop_severity'][-1]}",
            "",
            "PELVIC TILT METRICS",
            f"Pelvic Tilt Angle: {self.rear_metrics['pelvic_tilt_angle'][-1]:.3f}*",
            f"Pelvic Tilt Elevated Side: {self.rear_metrics['pelvic_tilt_elevated_side'][-1]}",
            f"Pelvic Tilt Severity: {self.rear_metrics['pelvic_tilt_severity'][-1]}",
            f"Pelvic Tilt Normalized: {self.rear_metrics['pelvic_tilt_normalized'][-1]:.3f}*",
            "",
            f"Step Width: {self.rear_metrics['step_width'][-1]:.1f} cm",
            "",
            "FOOT CROSSOVER METRICS",
            f"Left Foot Crossover: {self.rear_metrics['left_foot_crossover'][-1]}",
            f"Right Foot Crossover: {self.rear_metrics['right_foot_crossover'][-1]}",
            f"Left Foot Distance from Mid: {self.rear_metrics['left_distance_from_midline'][-1]:.3f}",
            f"Right Foot Distance from Mid: {self.rear_metrics['right_distance_from_midline'][-1]:.3f}",   
            "",
            "KNEE METRICS",
            f"Left Knee Valgus: {self.rear_metrics['left_knee_valgus'][-1]}",
            f"Left Knee Varus: {self.rear_metrics['left_knee_varus'][-1]}",
            f"Right Knee Valgus: {self.rear_metrics['right_knee_valgus'][-1]}",
            f"Right Knee Varus: {self.rear_metrics['right_knee_varus'][-1]}",
            f"Left Knee Normalized Deviation: {self.rear_metrics['left_knee_normalized_deviation'][-1]:.3f}*",
            f"Right Knee Normalized Deviation: {self.rear_metrics['right_knee_normalized_deviation'][-1]:.3f}*",
            f"Knee Severity Left: {self.rear_metrics['knee_severity_left'][-1]}",
            f"Knee Severity Right: {self.rear_metrics['knee_severity_right'][-1]}",
        ]

        metrics_text_right = [
            "UPPER BODY / ARM METRICS",
            f"Verticial Elbow Difference: {self.rear_metrics['vertical_elbow_diff'][-1]}",
            f"Normalized Vertical Difference: {self.rear_metrics['normalized_vertical_diff'][-1]:.3f}",
            f"Left Elbow Angle: {self.rear_metrics['left_elbow_angle'][-1]:.3f}",
            f"Right Elbow Angle: {self.rear_metrics['right_elbow_angle'][-1]:.3f}",
            f"Normalized Shoulder Difference: {self.rear_metrics['normalized_shoulder_diff'][-1]:.3f}",
            f"Normalized Shoulder Width: {self.rear_metrics['normalized_shoulder_width'][-1]:.3f}",
            f"Arm Height Symmetry: {self.rear_metrics['arm_height_symmetry'][-1]}",
            f"Elbow Angle Left: {self.rear_metrics['elbow_angle_left'][-1]}",
            f"Elbow Angle Right: {self.rear_metrics['elbow_angle_right'][-1]}",
            f"Left Wrist Crossover: {self.rear_metrics['left_wrist_crossover'][-1]}",
            f"Right Wrist Crossover: {self.rear_metrics['right_wrist_crossover'][-1]}",
            f"Shoulder Rotation: {self.rear_metrics['shoulder_rotation'][-1]}",
            "",
            "ANKLE METRICS",
            f"Left Ankle Inversion: {self.rear_metrics['left_ankle_inversion_value'][-1]:.3f}",
            f"Right Ankle Inversion: {self.rear_metrics['right_ankle_inversion_value'][-1]:.3f}",
            f"Left Ankle Inversion Normalized: {self.rear_metrics['left_ankle_normalized_value'][-1]:.3f}",
            f"Right Ankle Inversion Normalized : {self.rear_metrics['right_ankle_normalized_value'][-1]:.3f}",
            f"Left Ankle Inversion Pattern: {self.rear_metrics['left_ankle_pattern'][-1]}",
            f"Right Ankle Inversion Pattern: {self.rear_metrics['right_ankle_pattern'][-1]}",
            f"Left Ankle Inversion Severity: {self.rear_metrics['left_ankle_severity'][-1]}",
            f"Right Ankle Inversion Severity: {self.rear_metrics['right_ankle_severity'][-1]}",
            f"Left Ankle Inversion Angle: {self.rear_metrics['left_ankle_angle'][-1]:.3f}",
            f"Right Ankle Inversion Angle: {self.rear_metrics['right_ankle_angle'][-1]:.3f}",
            "",
            "STANCE METRICS",
            f"Stance Phase: {self.rear_metrics['stance_phase_detected'][-1]}",
            f"Stance Foot:  {self.rear_metrics['stance_foot'][-1]}",
            f"Stance Confidence: {self.rear_metrics['stance_confidence'][-1]}",
        
        ]

        # Text on the left side of vertical video text 
        for i, text in enumerate(metrics_text_left):
            cv2.putText(image, text, (10, 30 + i*30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        # Text on the Right side of vertical video text 
        for i, text in enumerate(metrics_text_right):
            cv2.putText(image, text, (image.shape[1] - 450, 30 + i*30),  # Adjust the x-coordinate as needed
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return image
    
    def save_rear_metrics(self, output_path):
        """Save extracted metrics to CSV file."""
        df = pd.DataFrame(self.rear_metrics)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Metrics saved to {output_path}")
        return df
    
    def merge_rear_watch_data(self, metrics_df, watch_data_path):
        """Merge extracted metrics with watch data based on timestamp."""
        watch_df = pd.read_csv(watch_data_path)
        
        # Assuming watch data has a timestamp column
        # Interpolate watch data to match video frames
        merged_df = pd.merge_asof(
            metrics_df.sort_values('timestamp'),
            watch_df.sort_values('timestamp'),
            on='timestamp',
            direction='nearest'
        )
        
        return merged_df

def main():
    parser = argparse.ArgumentParser(description='Runner Vision Biomechanics Analysis')
    parser.add_argument('--video', required=True, help='Path to input video file')
    parser.add_argument('--output_video', help='Path to output annotated video')
    parser.add_argument('--output_metrics', help='Path to output metrics CSV')
    parser.add_argument('--watch_data', help='Path to watch metrics CSV')
    
    args = parser.parse_args()
    
    analyzer = RunnerVisionAnalyzer()
    
    # Process video and extract metrics
    metrics_df = analyzer.process_video(args.video, args.output_video)
    
    # Save metrics
    if args.output_metrics:
        analyzer.save_metrics(args.output_metrics)
    
    # Merge with watch data if provided
    if args.watch_data:
        merged_df = analyzer.merge_rear_watch_data(metrics_df, args.watch_data)
        merged_df = analyzer.merge_side_watch_data(metrics_df, args.watch_data)
        merged_output = args.output_metrics.replace('.csv', '_merged.csv') if args.output_metrics else 'merged_metrics.csv'
        merged_df.to_csv(merged_output, index=False)
        print(f"Merged metrics saved to {merged_output}")
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()