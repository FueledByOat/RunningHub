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
        
# -------------------------------------
# Pose Landmarks Initialized and Defined
# -------------------------------------

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

# -------------------------------------
# Side Metrics Initialized
# -------------------------------------

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

# -------------------------------------
# Rear Metrics Initialized
# -------------------------------------

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

# -------------------------------------
# Video Processing Section
# -------------------------------------   
        
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
# Side Metrics Section
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
        self.side_metrics['knee_angle_left'].append(knee_angle_left['knee_angle'])
        self.side_metrics['knee_angle_right'].append(knee_angle_right['knee_angle'])

        # Estimate stride length
        stride_length = side_metrics.estimate_stride_length(landmark_coords)[0]
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

# Moved to runnervision_utils/metrics

# -------------------------------------
# Side Video Display Metrics 
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

    def extract_rear_metrics(self, landmarks, frame_number, timestamp):
        """Extract running biomechanics metrics from a single frame
        for a video shot from the side of a runner, AKA the frontal plane."""
        # Get normalized landmark positions
        landmark_coords = {}
        for name, landmark_id in self.key_points.items():
            landmark = landmarks.landmark[landmark_id]
            landmark_coords[name] = (landmark.x, landmark.y, landmark.z, landmark.visibility)
        
        # Detect stance phase rear
        stance_phase_rear = rear_metrics.detect_stance_phase_rear(landmark_coords)
        self.rear_metrics['stance_phase_detected'].append(stance_phase_rear['is_stance_phase'])
        self.rear_metrics['stance_foot'].append(stance_phase_rear['stance_foot'])
        self.rear_metrics['stance_confidence'].append(stance_phase_rear['confidence'])

        # Store basic timestamp data
        self.rear_metrics['timestamp'].append(timestamp)
        self.rear_metrics['frame_number'].append(frame_number)
        
        # Calculate foot crossover
        foot_crossover = rear_metrics.calculate_foot_crossover(landmark_coords)
        self.rear_metrics['left_foot_crossover'].append(foot_crossover["left_foot_crossover"])
        self.rear_metrics['right_foot_crossover'].append(foot_crossover["right_foot_crossover"])
        self.rear_metrics['left_distance_from_midline'].append(foot_crossover["left_distance_from_midline"])
        self.rear_metrics['right_distance_from_midline'].append(foot_crossover["right_distance_from_midline"])
        
        # Calculate hip drop
        hip_drop = rear_metrics.calculate_hip_drop(landmark_coords)
        self.rear_metrics['hip_drop_value'].append(hip_drop["hip_drop_value"])
        self.rear_metrics['hip_drop_direction'].append(hip_drop["hip_drop_direction"])
        self.rear_metrics['hip_drop_severity'].append(hip_drop["severity"])

        # Calculate pelic tilt angle
        pelvic_tilt = rear_metrics.calculate_pelvic_tilt(landmark_coords)
        self.rear_metrics['pelvic_tilt_angle'].append(pelvic_tilt["tilt_angle_degrees"])
        self.rear_metrics['pelvic_tilt_elevated_side'].append(pelvic_tilt["elevated_side"])
        self.rear_metrics['pelvic_tilt_severity'].append(pelvic_tilt["severity"])
        self.rear_metrics['pelvic_tilt_normalized'].append(pelvic_tilt["normalized_tilt"])
        
        # Calculate knee_alignment
        knee_alignment = rear_metrics.calculate_knee_alignment(landmark_coords)
        self.rear_metrics['left_knee_valgus'].append(knee_alignment["left_knee_valgus"])
        self.rear_metrics['right_knee_valgus'].append(knee_alignment["right_knee_valgus"])   
        self.rear_metrics['left_knee_varus'].append(knee_alignment["left_knee_varus"])
        self.rear_metrics['right_knee_varus'].append(knee_alignment["right_knee_varus"]) 
        self.rear_metrics['left_knee_normalized_deviation'].append(knee_alignment["left_normalized_deviation"])
        self.rear_metrics['right_knee_normalized_deviation'].append(knee_alignment["right_normalized_deviation"])    
        self.rear_metrics['knee_severity_left'].append(knee_alignment["severity_left"])
        self.rear_metrics['knee_severity_right'].append(knee_alignment["severity_right"]) 

        # Calculate ankle_inversion
        ankle_inversion = rear_metrics.calculate_ankle_inversion(landmark_coords)
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
        step_width = rear_metrics.calculate_step_width(landmark_coords)
        self.rear_metrics['step_width'].append(step_width)
        
        # Detect stride_symmetry
        stride_symmetry = rear_metrics.calculate_stride_symmetry(landmark_coords)
        self.rear_metrics['symmetry'].append(stride_symmetry)

        # Detect arm_swing_symmetry,
        arm_swing_mechanics = rear_metrics.calculate_arm_swing_mechanics(landmark_coords)
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

# -------------------------------------
# Rear Metric Functions and Classes
# -------------------------------------

# Moved to runnervision_utils/metrics

# -------------------------------------
# Rear Video Display Metrics 
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