# RunnverVision.py
"""
Core implementation for RunnerVision using BlazePose for runner biomechanics analysis
"""
from runnervision_utils.metrics import rear_metrics, side_metrics

import os
import math
from datetime import datetime
import argparse
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque
from typing import Dict, List, Tuple, Optional


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
            # 'arm_swing_symmetry' : [],
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
            'stance_phase_detected_velocity': [],
            'stance_foot_velocity': [],
            'avg_contact_time_ms' : [],
            'ground_contact_efficiency_rating' : [],
            'ground_contact_cadence_spm' : [],
            'vertical_oscillation_cm' : [],
            'vertical_oscillation_efficiency_rating' : [],
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
    
# -------------------------------------
# Side Metrics Sections
# -------------------------------------

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
        stance_phase = side_metrics.stance_detector_side_wrapper(landmark_coords)
        self.side_metrics['stance_phase_detected'].append(stance_phase['is_stance_phase'])
        self.side_metrics['stance_foot'].append(stance_phase['stance_foot'])
        self.side_metrics['stance_confidence'].append(stance_phase['confidence'])

        # Store basic timestamp data
        self.side_metrics['timestamp'].append(timestamp)
        self.side_metrics['frame_number'].append(frame_number)
        
        # Calculate foot strike metrics
        foot_strike = side_metrics.calculate_foot_strike(landmark_coords, stance_phase = stance_phase)
        self.side_metrics['strike_pattern'].append(foot_strike['strike_pattern'])
        self.side_metrics['strike_confidence'].append(foot_strike['confidence'])
        self.side_metrics['vertical_difference'].append(foot_strike['vertical_difference'])
        self.side_metrics['strike_foot_angle'].append(foot_strike['foot_angle'])
        self.side_metrics['strike_ankle_angle'].append(foot_strike['ankle_angle'])
        self.side_metrics['strike_landing_stiffness'].append(foot_strike['landing_stiffness'])
        
        # Calculate foot landing position relative to center of mass
        foot_position = side_metrics.calculate_foot_landing_position(landmark_coords, stance_phase = stance_phase)
        self.side_metrics['foot_landing_position_category'].append(foot_position['position_category'])
        self.side_metrics['foot_landing_distance_from_center_in_cm'].append(foot_position['distance_cm'])
        self.side_metrics['foot_landing_is_under_center_of_mass'].append(foot_position['is_under_com'])

        # Calculate trunk angle / trunk lean
        trunk_angle = side_metrics.calculate_trunk_angle(landmark_coords)
        self.side_metrics['trunk_angle_degrees'].append(trunk_angle['angle_degrees'])
        self.side_metrics['trunk_angle_is_optimal'].append(trunk_angle['is_optimal'])
        self.side_metrics['trunk_angle_assessment'].append(trunk_angle['assessment'])
        self.side_metrics['trunk_angle_assessment_detail'].append(trunk_angle['assessment_detail'])
        self.side_metrics['trunk_angle_confidence'].append(trunk_angle['confidence'])
        
        # Calculate arm carriage
        arm_angle = side_metrics.calculate_arm_carriage(landmark_coords)
        self.side_metrics['upper_arm_angle'].append(arm_angle['upper_arm_angle'])
        self.side_metrics['elbow_angle'].append(arm_angle['elbow_angle'])
        self.side_metrics['hand_position'].append(arm_angle['hand_position'])
        self.side_metrics['arm_swing_amplitude'].append(arm_angle['arm_swing_amplitude'])
        # self.side_metrics['arm_swing_symmetry'].append(arm_angle['arm_swing_symmetry'])
        self.side_metrics['arm_swing_overall_assessment'].append(arm_angle['overall_assessment'])
        self.side_metrics['arm_swing_recommendations'].append(arm_angle['recommendations'])
        
        # Calculate knee angle
        knee_angle_right = side_metrics.calculate_knee_angle(landmark_coords, 'right')
        knee_angle_left = side_metrics.calculate_knee_angle(landmark_coords, 'left')
        self.side_metrics['knee_angle_left'].append(knee_angle_left)
        self.side_metrics['knee_angle_right'].append(knee_angle_right)

        # Estimate stride length
        stride_length = side_metrics.estimate_stride_length(landmark_coords)
        self.side_metrics['stride_instantaneous_estimate_cm'].append(stride_length['instantaneous_estimate_cm'])
        self.side_metrics['stride_length_cm'].append(stride_length['stride_length_cm'])
        self.side_metrics['normalized_stride_length'].append(stride_length['normalized_stride_length'])
        self.side_metrics['stride_frequency'].append(stride_length['stride_frequency'])
        self.side_metrics['stride_assessment'].append(stride_length['assessment'])
        self.side_metrics['stride_confidence'].append(stride_length['confidence'])

        # New Metrics
        vertical_oscillation_metrics = side_metrics.vertical_oscillation_wrapper(landmark_coords)
        self.side_metrics['vertical_oscillation_cm'].append(vertical_oscillation_metrics['vertical_oscillation_cm'])
        self.side_metrics['vertical_oscillation_efficiency_rating'].append(vertical_oscillation_metrics['efficiency_rating'])
        ground_contact_metrics = side_metrics.ground_contact_wrapper(landmark_coords)
        self.side_metrics['avg_contact_time_ms'].append(ground_contact_metrics['avg_contact_time_ms'])
        self.side_metrics['ground_contact_cadence_spm'].append(ground_contact_metrics['cadence_spm'])
        self.side_metrics['ground_contact_efficiency_rating'].append(ground_contact_metrics['efficiency_rating'])
        stand_phase_detector_velocity = side_metrics.stance_detector_velocity_wrapper(landmark_coords)
        self.side_metrics['stance_phase_detected_velocity'].append(stand_phase_detector_velocity['is_stance_phase'])
        self.side_metrics['stance_foot_velocity'].append(stand_phase_detector_velocity['stance_foot'])


# -------------------------------------
# Side Metric Functions and Classes
# -------------------------------------

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
    
    class ArmCarriageAnalyzer:
        """
        Analyzes arm carriage focusing on the right arm (visible from right-side view).
        Maintains internal history for swing amplitude analysis.
        """
        
        def __init__(self, frame_rate: float = 30.0, history_size: int = 60):
            """
            Initialize the arm carriage analyzer.
            
            Args:
                frame_rate: Video frame rate (fps)
                history_size: Number of frames to maintain for swing analysis
            """
            self.frame_rate = frame_rate
            self.history_size = history_size
            
            # Store right arm positions for amplitude analysis
            self.right_arm_positions = deque(maxlen=history_size)
            self.right_wrist_angles = deque(maxlen=history_size)  # Angle relative to shoulder
            
            self.frame_count = 0
        
        def update(self, landmarks: Dict) -> Dict:
            """
            Update with new frame data and analyze arm carriage.
            
            Args:
                landmarks: Dictionary containing pose landmarks with x,y coordinates
                
            Returns:
                dict: Arm carriage analysis results
            """
            self.frame_count += 1
            
            # Check if right arm is visible (prioritize right side for right-side view)
            if not all(point in landmarks for point in ['right_shoulder', 'right_elbow', 'right_wrist']):
                return self._insufficient_data_response()
            
            # Extract right arm landmarks
            shoulder = landmarks['right_shoulder']
            elbow = landmarks['right_elbow']
            wrist = landmarks['right_wrist']
            
            # Store position for amplitude analysis
            self._store_arm_position(shoulder, elbow, wrist)
            
            # Calculate current frame metrics
            upper_arm_angle = self._calculate_upper_arm_angle(shoulder, elbow)
            elbow_angle = self._calculate_elbow_angle(shoulder, elbow, wrist)
            hand_position = self._analyze_hand_position(landmarks, shoulder, wrist)
            
            # Calculate swing amplitude (requires history)
            swing_amplitude = self._calculate_swing_amplitude()
            
            # Generate overall assessment and recommendations
            overall_assessment, recommendations = self._generate_assessment(
                upper_arm_angle, elbow_angle, hand_position, swing_amplitude
            )
            
            return {
                'upper_arm_angle': upper_arm_angle,
                'elbow_angle': elbow_angle,
                'hand_position': hand_position,
                'arm_swing_amplitude': swing_amplitude,
                'overall_assessment': overall_assessment,
                'recommendations': recommendations,
                'data_quality': 'sufficient',
                'frames_analyzed': len(self.right_arm_positions)
            }
        
        def _store_arm_position(self, shoulder, elbow, wrist):
            """Store arm position data for amplitude analysis."""
            # Calculate wrist angle relative to shoulder (horizontal reference)
            wrist_dx = wrist[0] - shoulder[0]
            wrist_dy = shoulder[1] - wrist[1]  # Inverted y for image coordinates
            
            # Angle from horizontal (0° = straight out to right, 90° = straight up)
            wrist_angle = math.degrees(math.atan2(wrist_dy, wrist_dx))
            
            self.right_arm_positions.append({
                'shoulder': shoulder,
                'elbow': elbow,
                'wrist': wrist,
                'wrist_angle': wrist_angle,
                'frame': self.frame_count
            })
            
            self.right_wrist_angles.append(wrist_angle)
        
        def _calculate_upper_arm_angle(self, shoulder, elbow):
            """Calculate upper arm angle relative to vertical."""
            upper_arm_dx = elbow[0] - shoulder[0]
            upper_arm_dy = shoulder[1] - elbow[1]  # Inverted y-axis
            
            return math.degrees(math.atan2(upper_arm_dx, upper_arm_dy))
        
        def _calculate_elbow_angle(self, shoulder, elbow, wrist):
            """Calculate elbow flexion angle."""
            # Calculate vectors
            upper_arm = [elbow[0] - shoulder[0], elbow[1] - shoulder[1]]
            forearm = [wrist[0] - elbow[0], wrist[1] - elbow[1]]
            
            # Calculate dot product
            dot_product = upper_arm[0] * forearm[0] + upper_arm[1] * forearm[1]
            
            # Calculate magnitudes
            upper_arm_magnitude = (upper_arm[0]**2 + upper_arm[1]**2)**0.5
            forearm_magnitude = (forearm[0]**2 + forearm[1]**2)**0.5
            
            if upper_arm_magnitude == 0 or forearm_magnitude == 0:
                return 90.0  # Default reasonable value
            
            cos_angle = dot_product / (upper_arm_magnitude * forearm_magnitude)
            cos_angle = max(min(cos_angle, 1.0), -1.0)  # Clamp to valid range
            
            angle_rad = math.acos(cos_angle)
            angle_deg = math.degrees(angle_rad)
            
            # Return the interior angle (0° = fully bent, 180° = straight)
            return angle_deg
        
        def _analyze_hand_position(self, landmarks, shoulder, wrist):
            """Analyze hand position relative to body."""
            # Check vertical position relative to shoulder and hip
            shoulder_y = shoulder[1]
            wrist_y = wrist[1]
            
            # Get hip position if available
            if 'right_hip' in landmarks:
                hip_y = landmarks['right_hip'][1]
            else:
                # Estimate hip position
                hip_y = shoulder_y + (shoulder_y * 0.3)  # Rough estimate
            
            # Check vertical hand position
            if wrist_y < shoulder_y:  # Hand higher than shoulder (remember inverted y)
                return 'too_high'
            elif wrist_y > hip_y:  # Hand lower than hip
                return 'too_low'
            else:
                return 'optimal'
        
        def _calculate_swing_amplitude(self):
            """Calculate arm swing amplitude from wrist angle variations."""
            if len(self.right_wrist_angles) < 15:  # Need sufficient data
                return None
            
            # Convert deque to array for easier analysis
            angles = np.array(self.right_wrist_angles)
            
            # Remove outliers to get cleaner swing pattern
            # Use percentiles to find swing range
            p10 = np.percentile(angles, 10)  # Lower bound of swing
            p90 = np.percentile(angles, 90)  # Upper bound of swing
            
            # Calculate swing amplitude as the range between 10th and 90th percentiles
            swing_range_degrees = p90 - p10
            
            # Convert to normalized amplitude
            # A typical efficient arm swing is about 45-60 degrees total range
            # Normalize so 1.0 represents optimal swing
            normalized_amplitude = swing_range_degrees / 52.5  # 52.5° is middle of optimal range
            
            return normalized_amplitude
        
        def _generate_assessment(self, upper_arm_angle, elbow_angle, hand_position, swing_amplitude):
            """Generate overall assessment and recommendations."""
            recommendations = []
            issues = []
            
            # Analyze elbow angle (90-110° is generally optimal for running)
            if elbow_angle < 75:
                issues.append("elbow_too_bent")
                recommendations.append("Open elbow angle slightly (aim for 90-110°)")
            elif elbow_angle > 125:
                issues.append("elbow_too_straight")
                recommendations.append("Increase elbow bend (aim for 90-110°)")
            
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
            
            # Check for crossing midline (based on upper arm angle)
            if upper_arm_angle < -15:  # Right arm swinging too far left
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
        
        def _insufficient_data_response(self):
            """Return response when insufficient data is available."""
            return {
                'upper_arm_angle': None,
                'elbow_angle': None,
                'hand_position': None,
                'arm_swing_amplitude': None,
                'overall_assessment': 'insufficient_data',
                'recommendations': ['Ensure right arm is visible in the video for proper analysis'],
                'data_quality': 'insufficient',
                'frames_analyzed': 0
            }
        
        def get_detailed_summary(self) -> str:
            """Generate a detailed summary of the latest arm analysis."""
            if len(self.right_arm_positions) == 0:
                return "No arm carriage data available."
            
            # Get current state by running analysis on most recent position
            latest_pos = self.right_arm_positions[-1]
            landmarks = {
                'right_shoulder': latest_pos['shoulder'],
                'right_elbow': latest_pos['elbow'],
                'right_wrist': latest_pos['wrist']
            }
            
            # Get latest analysis
            analysis = self.update(landmarks)
            
            summary = "RIGHT ARM CARRIAGE ANALYSIS:\n"
            summary += f"• Frames analyzed: {analysis['frames_analyzed']}\n"
            
            if analysis['elbow_angle'] is not None:
                elbow = analysis['elbow_angle']
                summary += f"• Elbow angle: {elbow:.1f}° "
                if 85 <= elbow <= 115:
                    summary += "(optimal)\n"
                elif elbow < 85:
                    summary += "(too bent)\n"
                else:
                    summary += "(too straight)\n"
            
            if analysis['hand_position'] is not None:
                summary += f"• Hand position: {analysis['hand_position']}\n"
            
            if analysis['arm_swing_amplitude'] is not None:
                amp = analysis['arm_swing_amplitude']
                summary += f"• Swing amplitude: {amp:.2f} "
                if 0.8 <= amp <= 1.2:
                    summary += "(optimal)\n"
                elif amp < 0.8:
                    summary += "(insufficient)\n"
                else:
                    summary += "(excessive)\n"
            
            if analysis['recommendations']:
                summary += "\nRECOMMENDATIONS:\n"
                for i, rec in enumerate(analysis['recommendations'], 1):
                    summary += f"{i}. {rec}\n"
            
            return summary
    
    def arm_carriage_wrapper(self, landmarks):
        """Attempting a wrapper around G rewrite"""
    
        if not hasattr(self, '_arm_carriage_analyzer_side'):
            self._arm_carriage_analyzer_side = self.ArmCarriageAnalyzer()
            # Pass image height if available
            # if hasattr(self, 'image_height'):
            #     self._stance_detector.image_height = self.image_height
        
        # Use the detector to determine stance phase
        return self._arm_carriage_analyzer_side.update(landmarks)



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


    class VerticalOscillationAnalyzer:
        """
        Analyzes vertical oscillation of runner's center of mass.
        Maintains internal history of necessary data points.
        """
        
        def __init__(self, frame_rate: float = 60.0, window_size: int = 30):
            """
            Initialize the vertical oscillation analyzer.
            
            Args:
                frame_rate: Video frame rate (fps)
                window_size: Number of frames to maintain for analysis
            """
            self.frame_rate = frame_rate
            self.window_size = window_size
            self.com_heights = deque(maxlen=window_size)
            self.frame_count = 0
            
        def update(self, landmarks: Dict) -> Dict:
            """
            Update with new frame data and calculate vertical oscillation metrics.
            
            Args:
                landmarks: Dictionary containing pose landmarks with x,y coordinates
                
            Returns:
                dict: Vertical oscillation analysis results
            """
            # Calculate center of mass height (hip midpoint approximation)
            hip_center_y = (landmarks['left_hip'][1] + landmarks['right_hip'][1]) / 2
            self.com_heights.append(hip_center_y)
            self.frame_count += 1
            
            # Need sufficient data for meaningful analysis
            if len(self.com_heights) < min(10, self.window_size):
                return self._insufficient_data_response()
            
            heights_array = np.array(self.com_heights)
            
            # Calculate basic statistics
            avg_height = np.mean(heights_array)
            min_height = np.min(heights_array)
            max_height = np.max(heights_array)
            
            # Calculate vertical oscillation (peak-to-peak in cm)
            vertical_oscillation_cm = (max_height - min_height) * 100
            
            # Calculate oscillation frequency
            oscillation_frequency = self._calculate_frequency(heights_array, avg_height)
            
            # Determine efficiency rating
            efficiency_rating = self._get_efficiency_rating(vertical_oscillation_cm)
            
            return {
                'vertical_oscillation_cm': vertical_oscillation_cm,
                'oscillation_frequency': oscillation_frequency,
                'efficiency_rating': efficiency_rating,
                'avg_com_height': avg_height,
                'min_com_height': min_height,
                'max_com_height': max_height,
                'data_quality': 'sufficient'
            }
        
        def _calculate_frequency(self, heights: np.ndarray, avg_height: float) -> float:
            """Calculate oscillation frequency using simple peak detection."""
            if len(heights) < 5:
                return 0.0
                
            # Simple peak detection - count transitions above average
            peaks = 0
            above_avg = heights > avg_height
            
            for i in range(1, len(above_avg)):
                if above_avg[i] and not above_avg[i-1]:  # Transition to above average
                    peaks += 1
            
            # Convert to frequency (oscillations per second)
            time_span = len(heights) / self.frame_rate
            return peaks / time_span if time_span > 0 else 0.0
        
        def _get_efficiency_rating(self, oscillation_cm: float) -> str:
            """Determine efficiency rating based on vertical oscillation."""
            if oscillation_cm <= 7.0:
                return 'excellent'
            elif oscillation_cm <= 9.0:
                return 'good'
            elif oscillation_cm <= 12.0:
                return 'moderate'
            else:
                return 'poor'
        
        def _insufficient_data_response(self) -> Dict:
            """Return response when insufficient data is available."""
            return {
                'vertical_oscillation_cm': 0.0,
                'oscillation_frequency': 0.0,
                'efficiency_rating': 'insufficient_data',
                'avg_com_height': 0.0,
                'min_com_height': 0.0,
                'max_com_height': 0.0,
                'data_quality': 'insufficient'
            }

    def vertical_oscillation_wrapper(self, landmarks):
        """Attempting a wrapper around G rewrite"""
    
        if not hasattr(self, '_vertical_oscillation_detector_side'):
            self._vertical_oscillation_detector_side = self.VerticalOscillationAnalyzer()
            # Pass image height if available
            # if hasattr(self, 'image_height'):
            #     self._stance_detector.image_height = self.image_height
        
        # Use the detector to determine stance phase
        return self._vertical_oscillation_detector_side.update(landmarks)

    class GroundContactTimeAnalyzer:
        """
        Analyzes ground contact time for running gait.
        Maintains internal history of foot positions and contact states.
        """
        
        def __init__(self, frame_rate: float = 60.0, history_size: int = 60, 
                    foot_height_threshold: float = 0.02):
            """
            Initialize the ground contact time analyzer.
            
            Args:
                frame_rate: Video frame rate (fps)
                history_size: Number of frames to maintain for contact analysis
                foot_height_threshold: Threshold for determining foot contact
            """
            self.frame_rate = frame_rate
            self.history_size = history_size
            self.foot_height_threshold = foot_height_threshold
            
            # Store foot heights and contact states
            self.left_foot_heights = deque(maxlen=history_size)
            self.right_foot_heights = deque(maxlen=history_size)
            self.left_contact_states = deque(maxlen=history_size)
            self.right_contact_states = deque(maxlen=history_size)
            
            # Track current contact periods
            self.left_contact_start = None
            self.right_contact_start = None
            self.left_contact_times = deque(maxlen=10)  # Store recent contact durations
            self.right_contact_times = deque(maxlen=10)
            
            self.frame_count = 0
            self.step_count = 0
        
        def update(self, landmarks: Dict) -> Dict:
            """
            Update with new frame data and calculate ground contact metrics.
            
            Args:
                landmarks: Dictionary containing pose landmarks with x,y coordinates
                
            Returns:
                dict: Ground contact time analysis results
            """
            # Extract current foot positions
            left_ankle_y = landmarks['left_ankle'][1]
            right_ankle_y = landmarks['right_ankle'][1]
            
            self.left_foot_heights.append(left_ankle_y)
            self.right_foot_heights.append(right_ankle_y)
            self.frame_count += 1
            
            # Determine ground contact states
            left_ground_level = self._get_ground_level(self.left_foot_heights)
            right_ground_level = self._get_ground_level(self.right_foot_heights)
            
            left_contact = left_ankle_y <= (left_ground_level + self.foot_height_threshold)
            right_contact = right_ankle_y <= (right_ground_level + self.foot_height_threshold)
            
            self.left_contact_states.append(left_contact)
            self.right_contact_states.append(right_contact)
            
            # Track contact periods
            self._update_contact_tracking('left', left_contact)
            self._update_contact_tracking('right', right_contact)
            
            # Calculate metrics if sufficient data
            if len(self.left_foot_heights) < 10:
                return self._insufficient_data_response()
            
            return self._calculate_metrics()
        
        def _get_ground_level(self, foot_heights: deque) -> float:
            """Estimate ground level from recent foot positions."""
            if len(foot_heights) < 5:
                return min(foot_heights) if foot_heights else 0.0
            
            # Use 10th percentile as ground level estimate
            heights_array = np.array(foot_heights)
            return np.percentile(heights_array, 10)
        
        def _update_contact_tracking(self, foot: str, is_contact: bool):
            """Track contact periods for each foot."""
            if foot == 'left':
                contact_start = self.left_contact_start
                contact_times = self.left_contact_times
            else:
                contact_start = self.right_contact_start
                contact_times = self.right_contact_times
            
            if is_contact and contact_start is None:
                # Start of contact period
                if foot == 'left':
                    self.left_contact_start = self.frame_count
                else:
                    self.right_contact_start = self.frame_count
            elif not is_contact and contact_start is not None:
                # End of contact period
                contact_duration_frames = self.frame_count - contact_start
                contact_duration_ms = (contact_duration_frames / self.frame_rate) * 1000
                contact_times.append(contact_duration_ms)
                self.step_count += 1
                
                if foot == 'left':
                    self.left_contact_start = None
                else:
                    self.right_contact_start = None
        
        def _calculate_metrics(self) -> Dict:
            """Calculate ground contact time metrics."""
            # Calculate average contact times
            left_avg = np.mean(self.left_contact_times) if self.left_contact_times else 0.0
            right_avg = np.mean(self.right_contact_times) if self.right_contact_times else 0.0
            
            avg_contact_time_ms = (left_avg + right_avg) / 2 if (left_avg > 0 or right_avg > 0) else 0.0
            
            # Contact time ratio
            if right_avg > 0:
                contact_time_ratio = left_avg / right_avg
            else:
                contact_time_ratio = 1.0 if left_avg == 0 else float('inf')
            
            # Calculate cadence
            time_span_minutes = len(self.left_foot_heights) / (self.frame_rate * 60)
            total_steps = len(self.left_contact_times) + len(self.right_contact_times)
            cadence_spm = total_steps / time_span_minutes if time_span_minutes > 0 else 0.0
            
            # Efficiency rating
            efficiency_rating = self._get_efficiency_rating(avg_contact_time_ms)
            
            return {
                'left_foot_contact_time_ms': left_avg,
                'right_foot_contact_time_ms': right_avg,
                'avg_contact_time_ms': avg_contact_time_ms,
                'contact_time_ratio': contact_time_ratio,
                'efficiency_rating': efficiency_rating,
                'cadence_spm': cadence_spm,
                'total_steps_detected': total_steps,
                'data_quality': 'sufficient'
            }
        
        def _get_efficiency_rating(self, contact_time_ms: float) -> str:
            """Determine efficiency rating based on contact time."""
            if contact_time_ms <= 180:
                return 'excellent'
            elif contact_time_ms <= 220:
                return 'good'
            elif contact_time_ms <= 280:
                return 'moderate'
            else:
                return 'poor'
        
        def _insufficient_data_response(self) -> Dict:
            """Return response when insufficient data is available."""
            return {
                'left_foot_contact_time_ms': 0.0,
                'right_foot_contact_time_ms': 0.0,
                'avg_contact_time_ms': 0.0,
                'contact_time_ratio': 1.0,
                'efficiency_rating': 'insufficient_data',
                'cadence_spm': 0.0,
                'total_steps_detected': 0,
                'data_quality': 'insufficient'
            }

    def ground_contact_wrapper(self, landmarks):
        """Attempting a wrapper around G rewrite"""
    
        if not hasattr(self, '_ground_contact_detector_side'):
            self._ground_contact_detector_side = self.GroundContactTimeAnalyzer()
            # Pass image height if available
            # if hasattr(self, 'image_height'):
            #     self._stance_detector.image_height = self.image_height
        
        # Use the detector to determine stance phase
        return self._ground_contact_detector_side.update(landmarks)

    class StancePhaseDetectorVelocity:
        """
        Helper class to detect stance phase using velocity analysis.
        Maintains minimal history for velocity calculations.
        """
        
        def __init__(self, frame_rate: float = 30.0, velocity_window: int = 3):
            """
            Initialize stance phase detector.
            
            Args:
                frame_rate: Video frame rate (fps)
                velocity_window: Number of frames for velocity calculation
            """
            self.frame_rate = frame_rate
            self.velocity_window = velocity_window
            self.ankle_positions = deque(maxlen=velocity_window)
            
        def update(self, landmarks: Dict) -> Dict:
            """
            Update with new landmarks and detect stance phase.
            
            Args:
                landmarks: Dictionary containing pose landmarks
                
            Returns:
                dict: Stance phase detection results
            """
            # Store ankle positions for velocity calculation
            ankle_data = {
                'left_ankle': landmarks['left_ankle'],
                'right_ankle': landmarks['right_ankle']
            }
            self.ankle_positions.append(ankle_data)
            
            if len(self.ankle_positions) < 2:
                return {
                    'is_stance_phase': False,
                    'stance_foot': 'unknown',
                    'left_foot_velocity': 0.0,
                    'right_foot_velocity': 0.0,
                    'confidence': 0.0
                }
            
            # Calculate velocities
            dt = 1.0 / self.frame_rate
            recent_positions = list(self.ankle_positions)
            
            left_velocities = []
            right_velocities = []
            
            for i in range(len(recent_positions) - 1):
                left_dy = recent_positions[i+1]['left_ankle'][1] - recent_positions[i]['left_ankle'][1]
                right_dy = recent_positions[i+1]['right_ankle'][1] - recent_positions[i]['right_ankle'][1]
                
                left_velocities.append(abs(left_dy / dt))
                right_velocities.append(abs(right_dy / dt))
            
            avg_left_velocity = np.mean(left_velocities)
            avg_right_velocity = np.mean(right_velocities)
            
            # Determine stance foot (lower velocity + lower position)
            current_left_height = landmarks['left_ankle'][1]
            current_right_height = landmarks['right_ankle'][1]
            
            # Score each foot for stance likelihood
            left_score = (1.0 / (avg_left_velocity + 0.001)) * (1.0 if current_left_height <= current_right_height else 0.5)
            right_score = (1.0 / (avg_right_velocity + 0.001)) * (1.0 if current_right_height <= current_left_height else 0.5)
            
            if left_score > right_score:
                stance_foot = 'left'
                confidence = left_score / (left_score + right_score)
            else:
                stance_foot = 'right'
                confidence = right_score / (left_score + right_score)
            
            # Determine if in stance phase (low velocity threshold)
            stance_velocity = avg_left_velocity if stance_foot == 'left' else avg_right_velocity
            is_stance = stance_velocity < 0.1  # Adjust threshold as needed
            
            return {
                'is_stance_phase': is_stance,
                'stance_foot': stance_foot,
                'left_foot_velocity': avg_left_velocity,
                'right_foot_velocity': avg_right_velocity,
                'confidence': confidence
            }

    def stance_detector_velocity_wrapper(self, landmarks):
        """Attempting a wrapper around G rewrite"""
    
        if not hasattr(self, 'stance_detector_velocity_side'):
            self.stance_detector_velocity_side = self.StancePhaseDetectorVelocity()
            # Pass image height if available
            # if hasattr(self, 'image_height'):
            #     self._stance_detector.image_height = self.image_height
        
        # Use the detector to determine stance phase
        return self.stance_detector_velocity_side.update(landmarks)

# -------------------------------------
# End Side Metrics Functions and Classes
# -------------------------------------

    def draw_side_analysis(self, image, landmarks, frame_number):
        """Draw pose landmarks and metrics on image."""
        # Draw skeleton
        self.mp_drawing.draw_landmarks(
            image,
            landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )

# -------------------------------------
# Side Video Display Metrics 
# -------------------------------------

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
            # f"Arm Swing Symmetry: {self.side_metrics['arm_swing_symmetry'][-1]}",
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
    
# -------------------------------------
# Rear Metrics Sections
# -------------------------------------

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