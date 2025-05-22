# runnervision_utils.py

"""
Run analysis script for RunnerVision

This script processes recorded video and watch data to analyze running biomechanics
"""

import argparse
import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2
from datetime import datetime
from glob import glob

# Import the RunnerVisionAnalyzer class from the core implementation
from utils.RunnerVision.RunnerVisionClass import RunnerVisionAnalyzer

class RunAnalyzer:
    def __init__(self, session_date=None, session_id=None):
        """Initialize the run analyzer with session details."""
        self.base_dir = os.path.dirname((os.path.abspath(__file__)))
        self.base_dir = os.getcwd()

        # Set session details
        if session_date is None:
            self.session_date = datetime.now().strftime('%Y-%m-%d')
        else:
            self.session_date = session_date
            
        self.session_id = self.session_date
        
        # Initialize paths
        self.videos_upload_dir = os.path.join(self.base_dir, "uploads" )
        self.videos_output_dir = os.path.join(self.base_dir, 'videos')
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.processed_dir = os.path.join(self.base_dir, 'processed')
        self.reports_dir = os.path.join(self.base_dir, 'reports')
        
        # Create directories if they don't exist
        for directory in [self.data_dir, self.processed_dir, self.reports_dir]:
            os.makedirs(directory, exist_ok=True)
            
        # Initialize the analyzer
        self.analyzer = RunnerVisionAnalyzer()
        
    def find_session_files(self):
        """Find all files related to the specified session."""
        if not os.path.exists(self.videos_upload_dir):
            raise FileNotFoundError(f"Video directory not found: {self.videos_upload_dir}")
        
        all_files = os.listdir(self.videos_upload_dir)

        # if self.session_id:
        #     # Filter files by session ID
        #     session_files = [f for f in all_files if self.session_id in f]
        #     if not session_files:
        #         raise FileNotFoundError(f"No files found for session ID: {self.session_id}")
        # else:
        #     # Find the most recent session
        #     session_prefixes = set()
        #     for filename in all_files:
        #         if filename.startswith('session_'):
        #             parts = filename.split('_')
        #             if len(parts) >= 3:
        #                 prefix = f"session_{parts[1]}_{parts[2]}"
        #                 session_prefixes.add(prefix)
            
        #     if not session_prefixes:
        #         raise FileNotFoundError("No session files found")
            
        #     # Get the latest session
        #     self.session_id = sorted(session_prefixes)[-1]
        #     session_files = [f for f in all_files if self.session_id in f]
        
        # Identify side and rear videos
        self.side_video = next((os.path.join(self.videos_upload_dir, f) for f in all_files if "side" in f.lower()), None)
        self.rear_video = next((os.path.join(self.videos_upload_dir, f) for f in all_files if "rear" in f.lower()), None)
        
        # Look for metadata file
        self.metadata_file = next((os.path.join(self.videos_upload_dir, f) for f in all_files if "metadata.txt" in f), None)
        
        # Look for watch data file in the data directory
        if os.path.exists(self.data_dir):
            data_files = os.listdir(self.data_dir)
            self.watch_data = next((os.path.join(self.data_dir, f) for f in data_files 
                                   if f.startswith(self.session_id) and f.endswith('.csv')), None)
        else:
            self.watch_data = None
        
        print(f"Found session: {self.session_id}")
        print(f"Side video: {self.side_video}")
        print(f"Rear video: {self.rear_video}")
        print(f"Metadata: {self.metadata_file}")
        print(f"Watch data: {self.watch_data}")
        
        return bool(self.side_video or self.rear_video)
    
    def load_metadata(self):
        """Load session metadata if available."""
        if not self.metadata_file or not os.path.exists(self.metadata_file):
            print("No metadata file found")
            self.metadata = {}
            return
        
        # Parse metadata file
        self.metadata = {}
        with open(self.metadata_file, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    self.metadata[key.strip()] = value.strip()
        
        print("Metadata loaded:")
        for key, value in self.metadata.items():
            print(f"  {key}: {value}")
    
    def process_videos(self):
        """Process both side and rear videos for analysis."""
        # Process side view
        if self.side_video:
            print(f"Processing side video: {self.side_video}")
            side_output = os.path.join(self.videos_output_dir, f"{self.session_id}_side_processed.mp4")
            self.side_metrics = self.analyzer.process_video(self.side_video, side_output)
            
            # Save side metrics
            side_metrics_path = os.path.join(self.data_dir, f"{self.session_id}_side_metrics.csv")
            self.side_metrics.to_csv(side_metrics_path, index=False)
            print(f"Side metrics saved to: {side_metrics_path}")
        else:
            self.side_metrics = None
        
        # Process rear view
        if self.rear_video:
            print(f"Processing rear video: {self.rear_video}")
            rear_output = os.path.join(self.videos_output_dir, f"{self.session_id}_rear_processed.mp4")
            
            # Create a new analyzer instance for rear view
            # This ensures metrics from side view aren't carried over
            rear_analyzer = RunnerVisionAnalyzer()
            self.rear_metrics = rear_analyzer.process_video(self.rear_video, rear_output)
            
            # Save rear metrics
            rear_metrics_path = os.path.join(self.data_dir, f"{self.session_id}_rear_metrics.csv")
            self.rear_metrics.to_csv(rear_metrics_path, index=False)
            print(f"Rear metrics saved to: {rear_metrics_path}")
        else:
            self.rear_metrics = None
        
        # # Merge with watch data if available
        # if self.watch_data and os.path.exists(self.watch_data):
        #     print(f"Merging with watch data: {self.watch_data}")
        #     self.merge_with_watch_data()
    
    def merge_with_watch_data(self):
        """Merge video metrics with watch data."""
        watch_df = pd.read_csv(self.watch_data)
        
        # Merge with side metrics if available
        if self.side_metrics is not None:
            # Make sure the timestamp columns are compatible
            # Watch data may have a different time format
            
            # Assuming watch data has a timestamp column
            # Interpolate watch data to match video frames
            merged_df = pd.merge_asof(
                self.side_metrics.sort_values('timestamp'),
                watch_df.sort_values('timestamp'),
                on='timestamp',
                direction='nearest'
            )
            
            # Save merged data
            merged_path = os.path.join(self.data_dir, f"{self.session_id}_merged_metrics.csv")
            merged_df.to_csv(merged_path, index=False)
            print(f"Merged metrics saved to: {merged_path}")
            
            # Update side metrics with merged data
            self.side_metrics = merged_df
    
    def generate_side_report(self, output_format='html'):
        """Generate an analysis report from the processed data."""
        if self.side_metrics.empty:
            print("No metrics available to generate report")
            return
        
        print("Generating analysis report...")
        
        # Create report filename
        report_file = os.path.join(self.reports_dir, f"{self.session_id}_side_angle_report.{output_format}")
        
        if output_format == 'html':
            self._generate_side_html_report(report_file)
        elif output_format == 'pdf':
            self._generate_pdf_report(report_file)
        else:
            print(f"Unsupported output format: {output_format}")
            return
        
        print(f"Report generated: {report_file}")

    def generate_rear_report(self, output_format='html'):
        """Generate an analysis report from the processed data."""
        if self.rear_metrics.empty:
            print("No metrics available to generate report")
            return
        
        print("Generating analysis report...")
        
        # Create report filename
        report_file = os.path.join(self.reports_dir, f"{self.session_id}_rear_angle_report.{output_format}")
        
        if output_format == 'html':
            self._generate_rear_html_report(report_file)
        elif output_format == 'pdf':
            self._generate_pdf_report(report_file)
        else:
            print(f"Unsupported output format: {output_format}")
            return
        
        print(f"Report generated: {report_file}")
    
    def _generate_rear_html_report(self, report_file):

        
        """Generate an HTML report with interactive plots."""
        # Create HTML content
        html_content = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<img src='RunnerVisionLogo_transparent.png' alt='RunnerVision Logo' width='503' height='195' style='display: block; margin: auto;'>",
            "    <title>Analysis Report</title>",
            "    <style>",
            "        body { font-family: Arial, sans-serif; margin: 20px; }",
            "        .container { max-width: 1200px; margin: 0 auto; }",
            "        .header { text-align: center; margin-bottom: 30px; }",
            "        .section { margin-bottom: 40px; }",
            "        .metric-box { background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }",
            "        .metric-value { font-size: 24px; font-weight: bold; }",
            "        .row { display: flex; justify-content: space-between; flex-wrap: wrap; }",
            "        .column { flex: 1; padding: 10px; min-width: 300px; }",
            "        .chart-container { width: 100%; height: 400px; margin-bottom: 30px; }",
            "        img { max-width: 100%; height: auto; }",
            "        .rating { font-size: 18px; font-weight: bold; }",
            "        .rating-optimal { color: #2ecc71; }",
            "        .rating-good { color: #3498db; }",
            "        .rating-fair { color: #f39c12; }",
            "        .rating-needs-work { color: #e74c3c; }",
            "        .metric-comparison { display: flex; align-items: center; margin-top: 10px; }",
            "        .metric-comparison-item { flex: 1; text-align: center; }",
            "        .comparison-divider { font-size: 24px; margin: 0 15px; }",
            "        .data-table { width: 100%; border-collapse: collapse; margin-top: 20px; }",
            "        .data-table th, .data-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "        .data-table th { background-color: #f2f2f2; }",
            "        .data-table tr:nth-child(even) { background-color: #f9f9f9; }",
            "        .progress-container { width: 100%; background-color: #e0e0e0; border-radius: 5px; margin-top: 10px; }",
            "        .progress-bar { height: 20px; border-radius: 5px; text-align: center; line-height: 20px; color: white; }",
            "    </style>",
            "</head>",
            "<body>",
            "    <div class='container'>",
            f"        <div class='header'><h1>Running Analysis Report</h1><h2>Session: {self.session_id}</h2></div>"
        ]
        
        # Add metadata section if available
        if hasattr(self, 'metadata') and self.metadata:
            html_content.extend([
                "        <div class='section'>",
                "            <h2>Session Information</h2>",
                "            <div class='metric-box'>",
                "                <table>",
            ])
            
            for key, value in self.metadata.items():
                html_content.append(f"                    <tr><td><strong>{key}:</strong></td><td>{value}</td></tr>")
            
            html_content.extend([
                "                </table>",
                "            </div>",
                "        </div>",
            ])
        
        # Generate metrics summary
        if self.rear_metrics is not None:
            html_content.extend([
                "        <div class='section'>",
                "            <h2>Running Metrics Summary</h2>",
                "            <div class='row'>",
            ])
            
            # Calculate % of frames during left sided strides the left foot is violating the crossover threshold 
            
            left_frames_count = len(self.rear_metrics[self.rear_metrics['stance_foot'] == 'left'])
            left_frames_and_crossover_count = len(self.rear_metrics[(self.rear_metrics['left_foot_crossover'] == True )
                                                                    & (self.rear_metrics['stance_foot'] == 'left')
                                                                    & (self.rear_metrics['stance_phase_detected'] == True)])
            left_leg_crossover_percent = ((left_frames_and_crossover_count / float(left_frames_count)) * 100)
            
            # Calculate % of frames during right sided strides the right foot is violating the crossover threshold 

            right_frames_count = len(self.rear_metrics[self.rear_metrics['stance_foot'] == 'right'])
            right_frames_and_crossover_count = len(self.rear_metrics[(self.rear_metrics['right_foot_crossover'] == True )
                                                                     & (self.rear_metrics['stance_foot'] == 'right')
                                                                     & (self.rear_metrics['stance_phase_detected'] == True)])
            right_leg_crossover_percent = ((right_frames_and_crossover_count / float(right_frames_count)) * 100)
          
            avg_metrics = {
                'Left Crossover Percentage': f"{left_leg_crossover_percent:.1f}%",
                'Right Crossover Percentage': f"{right_leg_crossover_percent:.1f}%",          
                'Hip Drop Values': f"{self.rear_metrics['hip_drop_value'].mean():.4f}cm ± {self.rear_metrics['hip_drop_value'].std():.4}cm",
                'Pelvic Tilt Angle': f"{self.rear_metrics['pelvic_tilt_angle'].mean():.2f}° ± {self.rear_metrics['pelvic_tilt_angle'].std():.2f}°",
                'Stride Symmetry': f"{self.rear_metrics['symmetry'].mean():.2f}%",
                'Shoulder Rotation': f"{self.rear_metrics['shoulder_rotation'].mode()[0]}"
            }
            
            # Add watch data metrics if available
            if 'vertical_oscillation' in self.rear_metrics.columns:
                avg_metrics['Vertical Oscillation'] = f"{self.rear_metrics['vertical_oscillation'].mean():.1f} cm"
            if 'ground_contact_time' in self.rear_metrics.columns:
                avg_metrics['Ground Contact Time'] = f"{self.rear_metrics['ground_contact_time'].mean():.0f} ms"
            if 'stride_length' in self.rear_metrics.columns:
                avg_metrics['Stride Length'] = f"{self.rear_metrics['stride_length'].mean():.1f} cm"
            if 'cadence' in self.rear_metrics.columns:
                avg_metrics['Cadence'] = f"{self.rear_metrics['cadence'].mean():.0f} spm"
            
            # Add metric boxes
            col_count = 0
            for metric, value in avg_metrics.items():
                if col_count % 3 == 0:
                    if col_count > 0:
                        html_content.append("            </div>")  # Close previous row
                    html_content.append("            <div class='row'>")  # Start new row
                
                html_content.extend([
                    "                <div class='column'>",
                    "                    <div class='metric-box'>",
                    f"                        <div>{metric}</div>",
                    f"                        <div class='metric-value'>{value}</div>",
                    "                    </div>",
                    "                </div>",
                ])
                col_count += 1
            
            html_content.append("            </div>")  # Close final row
            html_content.append("        </div>")  # Close section
        
        # Generate plots
        if self.rear_metrics is not None:
            # Save plots to files
            plot_files = self._save_rear_metric_plots()
            
            html_content.extend([
                "        <div class='section'>",
                "            <h2>Running Metrics Visualization</h2>",
                # "            <div class='row'>",
            ])
            
            # Add plots to report
            for plot_file in plot_files:
                rel_path = os.path.relpath(plot_file, os.path.dirname(report_file))
                html_content.extend([
                    "                <div class='column'>",
                    f"                    <img src='{rel_path}' alt='Running Metrics Plot'>",
                    "                </div>",
                ])
            
            html_content.extend([
                "            </div>",
                "        </div>",
            ])
        
        # Add gait analysis section
        if self.rear_metrics is not None:
            html_content.extend([
                "        <div class='section'>",
                "            <h2>Gait Analysis</h2>",
                "            <div class='metric-box'>",
                "                <h3>Form Recommendations</h3>",
                "                <ul>",
            ])
            
            # Generate recommendations based on metrics
            recommendations = self._generate_rear_recommendations()
            for rec in recommendations:
                html_content.append(f"                    <li>{rec}</li>")
            
            html_content.extend([
                "                </ul>",
                "            </div>",
                "        </div>",
            ])
        
        # Close HTML
        html_content.extend([
            "    </div>",
            "</body>",
            "</html>",
        ])
        
        # Write HTML to file
        with open(report_file, 'w') as f:
            f.write("\n".join(html_content))
    
    
    def _generate_side_html_report(self, report_file):
        """Generate an HTML report with interactive plots."""
        # Create HTML content
        html_content = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<img src='RunnerVisionLogo_transparent.png' alt='RunnerVision Logo' width='503' height='195' style='display: block; margin: auto;'>",            "    <title>Analysis Report</title>",
            "    <style>",
            "        body { font-family: Arial, sans-serif; margin: 20px; }",
            "        .container { max-width: 1200px; margin: 0 auto; }",
            "        .header { text-align: center; margin-bottom: 30px; }",
            "        .section { margin-bottom: 40px; }",
            "        .metric-box { background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }",
            "        .metric-value { font-size: 24px; font-weight: bold; }",
            "        .row { display: flex; justify-content: space-between; flex-wrap: wrap; }",
            "        .column { flex: 1; padding: 10px; min-width: 300px; }",
            "        .chart-container { width: 100%; height: 400px; margin-bottom: 30px; }",
            "        img { max-width: 100%; height: auto; }",
            "        .rating { font-size: 18px; font-weight: bold; }",
            "        .rating-optimal { color: #2ecc71; }",
            "        .rating-good { color: #3498db; }",
            "        .rating-fair { color: #f39c12; }",
            "        .rating-needs-work { color: #e74c3c; }",
            "        .metric-comparison { display: flex; align-items: center; margin-top: 10px; }",
            "        .metric-comparison-item { flex: 1; text-align: center; }",
            "        .comparison-divider { font-size: 24px; margin: 0 15px; }",
            "        .data-table { width: 100%; border-collapse: collapse; margin-top: 20px; }",
            "        .data-table th, .data-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "        .data-table th { background-color: #f2f2f2; }",
            "        .data-table tr:nth-child(even) { background-color: #f9f9f9; }",
            "        .progress-container { width: 100%; background-color: #e0e0e0; border-radius: 5px; margin-top: 10px; }",
            "        .progress-bar { height: 20px; border-radius: 5px; text-align: center; line-height: 20px; color: white; }",
            "    </style>",
            "</head>",
            "<body>",
            "    <div class='container'>",
            f"        <div class='header'><h1>Running Analysis Report</h1><h2>Session: {self.session_id}</h2></div>"
        ]
        
        # Add metadata section if available
        if hasattr(self, 'metadata') and self.metadata:
            html_content.extend([
                "        <div class='section'>",
                "            <h2>Session Information</h2>",
                "            <div class='metric-box'>",
                "                <table>",
            ])
            
            for key, value in self.metadata.items():
                html_content.append(f"                    <tr><td><strong>{key}:</strong></td><td>{value}</td></tr>")
            
            html_content.extend([
                "                </table>",
                "            </div>",
                "        </div>",
            ])
        
        # Generate metrics summary
        if self.side_metrics is not None:
            html_content.extend([
                "        <div class='section'>",
                "            <h2>Running Metrics Summary</h2>",
                "            <div class='row'>",
            ])
            
            # Calculate average metrics
            ## Foot Strike
            foot_strike_counts = self.side_metrics['strike_pattern'].value_counts().drop("not_applicable", errors='ignore')
            primary_foot_strike = foot_strike_counts.idxmax() if not foot_strike_counts.empty else "N/A"
            foot_strike_percent = (foot_strike_counts.max() / foot_strike_counts.sum() * 100) if not foot_strike_counts.empty else 0
            
            ## Foot Landing Position
            foot_landing_counts = self.side_metrics['foot_landing_position_category'].value_counts().drop("not_applicable", errors='ignore')
            primary_foot_landing = foot_landing_counts.idxmax() if not foot_landing_counts.empty else "N/A"
            foot_landing_percent = (foot_landing_counts.max() / foot_landing_counts.sum() * 100) if not foot_landing_counts.empty else 0

            # ## Calculate arm swing metrics
            # arm_swing_amplitude_avg = self.side_metrics['arm_swing_amplitude'].mean() if 'arm_swing_amplitude' in self.side_metrics.columns else None
            # arm_swing_symmetry_counts = self.side_metrics['arm_swing_symmetry'].value_counts()['symmetrical'] if 'arm_swing_symmetry' in self.side_metrics.columns else None
            # arm_swing_symmetry_avg = (arm_swing_symmetry_counts / len(self.side_metrics['arm_swing_symmetry'])) * 100

            ## Calculate stance phase percentage
            stance_phase_percent = (self.side_metrics['stance_phase_detected'].sum() / len(self.side_metrics) * 100) if 'stance_phase_detected' in self.side_metrics.columns else None
            
            ## Calculate stride metrics
            stride_length_avg = self.side_metrics['stride_length_cm'].mean() if 'stride_length_cm' in self.side_metrics.columns else None
            normalized_stride_avg = self.side_metrics['normalized_stride_length'].mean() if 'normalized_stride_length' in self.side_metrics.columns else None
            # stride_frequency_avg = self.side_metrics['stride_frequency'].mean() if 'stride_frequency' in self.side_metrics.columns else None
            
            ## Calculate landing stiffness
            # landing_stiffness_avg = self.side_metrics['strike_landing_stiffness'].mean() if 'strike_landing_stiffness' in self.side_metrics.columns else None

            landing_stiffness_counts = self.side_metrics['strike_landing_stiffness'].value_counts().drop("not_applicable", errors='ignore')
            primary_landing_stiffness = landing_stiffness_counts.idxmax() if not landing_stiffness_counts.empty else "N/A"
            landing_stiffness_percent = (landing_stiffness_counts.max() / landing_stiffness_counts.sum() * 100) if not landing_stiffness_counts.empty else 0

            avg_metrics = {
                'Foot Strike Pattern': f"{primary_foot_strike} ({foot_strike_percent:.1f}%)",
                'Foot Landing Pattern': f"{primary_foot_landing} ({foot_landing_percent:.1f}%)",
                'Trunk Angle': f"{self.side_metrics['trunk_angle_degrees'].mean():.1f}° ± {self.side_metrics['trunk_angle_degrees'].std():.1f}°",
                'Left Knee Angle': f"{self.side_metrics['knee_angle_left'].mean():.1f}° ± {self.side_metrics['knee_angle_left'].std():.1f}°",
                'Right Knee Angle': f"{self.side_metrics['knee_angle_right'].mean():.1f}° ± {self.side_metrics['knee_angle_right'].std():.1f}°"
            }
            
            # # Add arm swing metrics if available
            # if arm_swing_amplitude_avg is not None:
            #     avg_metrics['Arm Swing Amplitude'] = f"{arm_swing_amplitude_avg:.1f}°"
            # if arm_swing_symmetry_avg is not None:
            #     avg_metrics['Arm Swing Symmetry'] = f"{arm_swing_symmetry_avg:.1f}%" if arm_swing_symmetry_avg <= 100 else "100%"
            
            # Add stance phase metrics if available
            if stance_phase_percent is not None:
                avg_metrics['Stance Phase'] = f"{stance_phase_percent:.1f}% of gait cycle"
            
            # Add stride metrics if available
            if stride_length_avg is not None:
                avg_metrics['Stride Length'] = f"{stride_length_avg:.1f} cm"
            if normalized_stride_avg is not None:
                avg_metrics['Normalized Stride'] = f"{normalized_stride_avg:.2f} × height"
            # if stride_frequency_avg is not None:
            #     avg_metrics['Stride Frequency'] = f"{stride_frequency_avg:.1f} strides/min"
            
            # Add landing stiffness if available
            if landing_stiffness_counts is not None:
                # avg_metrics['Landing Stiffness'] = f"{landing_stiffness_avg:.1f}/10"
                avg_metrics['Landing Stiffness'] = f"{primary_landing_stiffness} ({landing_stiffness_percent:.1f}%)"
                
            # Add watch data metrics if available
            if 'vertical_oscillation' in self.side_metrics.columns:
                avg_metrics['Vertical Oscillation'] = f"{self.side_metrics['vertical_oscillation'].mean():.1f} cm"
            if 'ground_contact_time' in self.side_metrics.columns:
                avg_metrics['Ground Contact Time'] = f"{self.side_metrics['ground_contact_time'].mean():.0f} ms"
            if 'cadence' in self.side_metrics.columns:
                avg_metrics['Cadence'] = f"{self.side_metrics['cadence'].mean():.0f} spm"
            
            # Add metric boxes
            col_count = 0
            for metric, value in avg_metrics.items():
                if col_count % 3 == 0:
                    if col_count > 0:
                        html_content.append("            </div>")  # Close previous row
                    html_content.append("            <div class='row'>")  # Start new row
                
                html_content.extend([
                    "                <div class='column'>",
                    "                    <div class='metric-box'>",
                    f"                        <div>{metric}</div>",
                    f"                        <div class='metric-value'>{value}</div>",
                    "                    </div>",
                    "                </div>",
                ])
                col_count += 1
            
            html_content.append("            </div>")  # Close final row
            html_content.append("        </div>")  # Close section
        
        # Generate advanced metrics sections
        if self.side_metrics is not None:
            # Knee Angle Comparison (Left vs Right)
            html_content.extend([
                "        <div class='section'>",
                "            <h2>Bilateral Comparison</h2>",
                "            <div class='metric-box'>",
                "                <h3>Knee Angle Symmetry</h3>",
                "                <div class='metric-comparison'>",
                "                    <div class='metric-comparison-item'>",
                f"                        <div>Left Knee</div>",
                f"                        <div class='metric-value'>{self.side_metrics['knee_angle_left'].mean():.1f}°</div>",
                "                    </div>",
                "                    <div class='comparison-divider'>vs</div>",
                "                    <div class='metric-comparison-item'>",
                f"                        <div>Right Knee</div>",
                f"                        <div class='metric-value'>{self.side_metrics['knee_angle_right'].mean():.1f}°</div>",
                "                    </div>",
                "                </div>",
            ])
            
            # Calculate knee angle difference
            left_avg = self.side_metrics['knee_angle_left'].mean()
            right_avg = self.side_metrics['knee_angle_right'].mean()
            diff_percent = abs(left_avg - right_avg) / ((left_avg + right_avg) / 2) * 100
            
            # Add symmetry rating
            if diff_percent < 5:
                symmetry_rating = "Excellent symmetry"
                symmetry_class = "rating-optimal"
            elif diff_percent < 10:
                symmetry_rating = "Good symmetry"
                symmetry_class = "rating-good"
            elif diff_percent < 15:
                symmetry_rating = "Fair symmetry"
                symmetry_class = "rating-fair"
            else:
                symmetry_rating = "Needs improvement"
                symmetry_class = "rating-needs-work"
            
            html_content.extend([
                f"                <div>Difference: {abs(left_avg - right_avg):.1f}° ({diff_percent:.1f}%)</div>",
                f"                <div class='rating {symmetry_class}'>{symmetry_rating}</div>",
                "            </div>",
                "        </div>",
            ])
            
            # Arm Swing Analysis
            if 'arm_swing_amplitude' in self.side_metrics.columns and 'arm_swing_symmetry' in self.side_metrics.columns:
                arm_swing_amp = self.side_metrics['arm_swing_amplitude'].mean()
                
                arm_swing_symmetry_counts = self.side_metrics['arm_swing_symmetry'].value_counts()['symmetrical'] if 'arm_swing_symmetry' in self.side_metrics.columns else None
                arm_swing_sym = (arm_swing_symmetry_counts / len(self.side_metrics['arm_swing_symmetry'])) * 100

                # arm_swing_sym = self.side_metrics['arm_swing_symmetry'].mean()
                
                # Determine rating for arm swing
                if arm_swing_amp > 45:
                    arm_amp_rating = "Optimal"
                    arm_amp_class = "rating-optimal"
                elif arm_swing_amp > 35:
                    arm_amp_rating = "Good"
                    arm_amp_class = "rating-good"
                elif arm_swing_amp > 25:
                    arm_amp_rating = "Fair"
                    arm_amp_class = "rating-fair"
                else:
                    arm_amp_rating = "Limited"
                    arm_amp_class = "rating-needs-work"
                    
                if arm_swing_sym > 90:
                    arm_sym_rating = "Excellent symmetry"
                    arm_sym_class = "rating-optimal"
                elif arm_swing_sym > 80:
                    arm_sym_rating = "Good symmetry"
                    arm_sym_class = "rating-good"
                elif arm_swing_sym > 70:
                    arm_sym_rating = "Fair symmetry"
                    arm_sym_class = "rating-fair"
                else:
                    arm_sym_rating = "Asymmetrical"
                    arm_sym_class = "rating-needs-work"
                
                html_content.extend([
                    "        <div class='section'>",
                    "            <h2>Arm Swing Analysis</h2>",
                    "            <div class='row'>",
                    "                <div class='column'>",
                    "                    <div class='metric-box'>",
                    "                        <h3>Arm Swing Amplitude</h3>",
                    f"                        <div class='metric-value'>{arm_swing_amp:.1f}°</div>",
                    f"                        <div class='rating {arm_amp_class}'>{arm_amp_rating}</div>",
                    "                        <div class='progress-container'>",
                    f"                            <div class='progress-bar' style='width: {min(arm_swing_amp / 60 * 100, 100)}%; background-color: {'#2ecc71' if arm_swing_amp > 45 else '#f39c12' if arm_swing_amp > 30 else '#e74c3c'};'>{arm_swing_amp:.1f}°</div>",
                    "                        </div>",
                    "                    </div>",
                    "                </div>",
                    "                <div class='column'>",
                    "                    <div class='metric-box'>",
                    "                        <h3>Arm Swing Symmetry</h3>",
                    f"                        <div class='metric-value'>{arm_swing_sym:.1f}%</div>",
                    f"                        <div class='rating {arm_sym_class}'>{arm_sym_rating}</div>",
                    "                        <div class='progress-container'>",
                    f"                            <div class='progress-bar' style='width: {min(arm_swing_sym, 100)}%; background-color: {'#2ecc71' if arm_swing_sym > 90 else '#3498db' if arm_swing_sym > 80 else '#f39c12' if arm_swing_sym > 70 else '#e74c3c'};'>{arm_swing_sym:.1f}%</div>",
                    "                        </div>",
                    "                    </div>",
                    "                </div>",
                    "            </div>",
                    "        </div>",
                ])
            
            # Stride Analysis Section
            if 'stride_length_cm' in self.side_metrics.columns and 'stride_frequency' in self.side_metrics.columns:
                stride_length = self.side_metrics['stride_length_cm'].mean()
                stride_freq = self.side_metrics['stride_frequency'].mean()
                norm_stride = self.side_metrics['normalized_stride_length'].mean() if 'normalized_stride_length' in self.side_metrics.columns else None
                
                html_content.extend([
                    "        <div class='section'>",
                    "            <h2>Stride Analysis</h2>",
                    "            <div class='row'>",
                    "                <div class='column'>",
                    "                    <div class='metric-box'>",
                    "                        <h3>Stride Length</h3>",
                    f"                        <div class='metric-value'>{stride_length:.1f} cm</div>",
                ])
                
                if norm_stride is not None:
                    html_content.extend([
                        f"                        <div>Normalized: {norm_stride:.2f} × height</div>",
                    ])
                    
                    # Add stride length rating
                    if 1.2 <= norm_stride <= 1.4:
                        html_content.extend([
                            f"                        <div class='rating rating-optimal'>Optimal Range</div>",
                        ])
                    elif 1.0 <= norm_stride < 1.2 or 1.4 < norm_stride <= 1.6:
                        html_content.extend([
                            f"                        <div class='rating rating-good'>Good Range</div>",
                        ])
                    else:
                        html_content.extend([
                            f"                        <div class='rating rating-needs-work'>Outside Optimal Range</div>",
                        ])
                
                html_content.extend([
                    "                    </div>",
                    "                </div>",
                    "                <div class='column'>",
                    "                    <div class='metric-box'>",
                    "                        <h3>Stride Frequency</h3>",
                    f"                        <div class='metric-value'>{stride_freq:.1f} strides/min</div>",
                    f"                        <div>Cadence: {stride_freq * 2:.1f} steps/min</div>",
                ])
                
                # Add cadence rating
                cadence = stride_freq * 2
                if 170 <= cadence <= 190:
                    html_content.extend([
                        f"                        <div class='rating rating-optimal'>Optimal Cadence</div>",
                    ])
                elif 160 <= cadence < 170 or 190 < cadence <= 200:
                    html_content.extend([
                        f"                        <div class='rating rating-good'>Good Cadence</div>",
                    ])
                else:
                    html_content.extend([
                        f"                        <div class='rating rating-needs-work'>Outside Optimal Range</div>",
                    ])
                
                html_content.extend([
                    "                    </div>",
                    "                </div>",
                    "            </div>",
                    "        </div>",
                ])
            
            # Stance Phase and Landing Stiffness Analysis
            if 'stance_phase_detected' in self.side_metrics.columns or 'strike_landing_stiffness' in self.side_metrics.columns:
                html_content.extend([
                    "        <div class='section'>",
                    "            <h2>Ground Contact Analysis</h2>",
                    "            <div class='row'>",
                ])
                
                # Stance Phase Analysis
                if 'stance_phase_detected' in self.side_metrics.columns and 'stance_foot' in self.side_metrics.columns:
                    stance_data = {}
                    for foot in self.side_metrics['stance_foot'].unique():
                        if foot not in ['not_applicable', 'unknown', None]:
                            foot_stance_frames = self.side_metrics[self.side_metrics['stance_foot'] == foot]
                            stance_data[foot] = len(foot_stance_frames)
                    
                    # Create stance phase distribution
                    if stance_data:
                        total_stance_frames = sum(stance_data.values())
                        html_content.extend([
                            "                <div class='column'>",
                            "                    <div class='metric-box'>",
                            "                        <h3>Stance Phase Distribution</h3>",
                        ])
                        
                        for foot, count in stance_data.items():
                            percentage = count / total_stance_frames * 100
                            html_content.extend([
                                "                        <div class='progress-container'>",
                                f"                            <div>{foot}: {percentage:.1f}%</div>",
                                f"                            <div class='progress-bar' style='width: {percentage}%; background-color: {'#3498db' if foot == 'left' else '#e74c3c'};'>{count} frames</div>",
                                "                        </div>",
                            ])
                        
                        html_content.extend([
                            "                    </div>",
                            "                </div>",
                        ])
                
                # Landing Stiffness Analysis
                if 'strike_landing_stiffness' in self.side_metrics.columns:
                    # landing_stiffness = self.side_metrics['strike_landing_stiffness'].mean()

                    landing_stiffness_counts = self.side_metrics['strike_landing_stiffness'].value_counts().drop("not_applicable", errors='ignore')
                    primary_landing_stiffness = landing_stiffness_counts.idxmax() if not landing_stiffness_counts.empty else "N/A"
                    landing_stiffness = (landing_stiffness_counts.max() / landing_stiffness_counts.sum() * 10) if not landing_stiffness_counts.empty else 0
                    
                    # Determine rating
                    if 4 <= landing_stiffness <= 6:
                        stiffness_rating = "Optimal stiffness"
                        stiffness_class = "rating-optimal"
                    elif 3 <= landing_stiffness < 4 or 6 < landing_stiffness <= 7:
                        stiffness_rating = "Good stiffness"
                        stiffness_class = "rating-good"
                    elif 2 <= landing_stiffness < 3 or 7 < landing_stiffness <= 8:
                        stiffness_rating = "Fair stiffness"
                        stiffness_class = "rating-fair"
                    else:
                        stiffness_rating = "Suboptimal stiffness"
                        stiffness_class = "rating-needs-work"
                    
                    html_content.extend([
                        "                <div class='column'>",
                        "                    <div class='metric-box'>",
                        "                        <h3>Landing Stiffness</h3>",
                        f"                        <div class='metric-value'>{landing_stiffness:.1f}/10</div>",
                        f"                        <div class='rating {stiffness_class}'>{stiffness_rating}</div>",
                        "                        <div class='progress-container'>",
                        f"                            <div class='progress-bar' style='width: {landing_stiffness * 10}%; background-color: {'#e74c3c' if landing_stiffness < 3 else '#f39c12' if landing_stiffness < 4 else '#2ecc71' if landing_stiffness <= 6 else '#f39c12' if landing_stiffness <= 7 else '#e74c3c'};'>{landing_stiffness:.1f}/10</div>",
                        "                        </div>",
                        "                        <div><small>0 = Very soft landing, 10 = Very stiff landing</small></div>",
                        "                    </div>",
                        "                </div>",
                    ])
                
                html_content.extend([
                    "            </div>",
                    "        </div>",
                ])
        
        # Generate plots
        if self.side_metrics is not None:
            # Save plots to files
            plot_files = self._save_side_metric_plots()
            
            html_content.extend([
                "        <div class='section'>",
                "            <h2>Running Metrics Visualization</h2>",
                # "            <div class='row'>",
            ])
            
            # Add plots to report
            for i, plot_file in enumerate(plot_files):
                rel_path = os.path.relpath(plot_file, os.path.dirname(report_file))
                # if i % 2 == 0 and i > 1:
                html_content.extend([
                    "            <div class='row'>",
                "                <div class='column'>",
                f"                    <img src='{rel_path}' alt='Running Metrics Plot'>",
                "                </div>",
            ])                   
                # else:
                #     html_content.extend([
                #         "                <div class='column'>",
                #         f"                    <img src='{rel_path}' alt='Running Metrics Plot'>",
                #         "                </div>",
                #     ])
            
            html_content.extend([
                "            </div>",
                "        </div>",
            ])
        
        # Add gait analysis section
        if self.side_metrics is not None:
            html_content.extend([
                "        <div class='section'>",
                "            <h2>Gait Analysis</h2>",
                "            <div class='metric-box'>",
                "                <h3>Form Recommendations</h3>",
                "                <ul>",
            ])
            
            # Generate recommendations based on metrics
            recommendations = self._generate_side_recommendations()
            for rec in recommendations:
                html_content.append(f"                    <li>{rec}</li>")
            
            html_content.extend([
                "                </ul>",
                "            </div>",
                "        </div>",
            ])
        
        # Add Data Table Section for detailed metrics
        if self.side_metrics is not None:
            html_content.extend([
                "        <div class='section'>",
                "            <h2>Detailed Metrics</h2>",
                "            <div class='metric-box'>",
                "                <button onclick=\"toggleTable()\" style=\"padding: 8px 16px; margin-bottom: 10px; background-color: #3498db; color: white; border: none; border-radius: 4px; cursor: pointer;\">Show/Hide Detailed Data</button>",
                "                <div id=\"metricsTable\" style=\"display: none; overflow-x: auto;\">",
                "                    <table class='data-table'>",
                "                        <thead>",
                "                            <tr>",
                "                                <th>Frame</th>",
                "                                <th>Strike Pattern</th>",
                "                                <th>Foot Landing</th>",
                "                                <th>Trunk Angle</th>",
                "                                <th>L Knee Angle</th>",
                "                                <th>R Knee Angle</th>",
            ])
            
            # Add additional columns if available
            if 'arm_swing_amplitude' in self.side_metrics.columns:
                html_content.append("                                <th>Arm Swing</th>")
            if 'strike_landing_stiffness' in self.side_metrics.columns:
                html_content.append("                                <th>Landing Stiffness</th>")
            if 'stride_frequency' in self.side_metrics.columns:
                html_content.append("                                <th>Stride Freq</th>")
            if 'stride_length_cm' in self.side_metrics.columns:
                html_content.append("                                <th>Stride Length</th>")
            if 'stance_phase_detected' in self.side_metrics.columns:
                html_content.append("                                <th>Stance Phase</th>")
            
            html_content.append("                            </tr>")
            html_content.append("                        </thead>")
            html_content.append("                        <tbody>")
            
            # Add sample data rows (limited to max 20 rows to keep the table manageable)
            sample_indices = np.linspace(0, len(self.side_metrics) - 1, min(20, len(self.side_metrics))).astype(int)
            for i in sample_indices:
                row = self.side_metrics.iloc[i]
                html_content.append("                            <tr>")
                html_content.append(f"                                <td>{row['frame_number']}</td>")
                html_content.append(f"                                <td>{row['strike_pattern']}</td>")
                html_content.append(f"                                <td>{row['foot_landing_position_category']}</td>")
                html_content.append(f"                                <td>{row['trunk_angle_degrees']:.1f}°</td>")
                html_content.append(f"                                <td>{row['knee_angle_left']:.1f}°</td>")
                html_content.append(f"                                <td>{row['knee_angle_right']:.1f}°</td>")
                
                # Add additional columns if available
                ## Commenting out as these are having issues populating and blocking troubleshooting
                # if 'arm_swing_amplitude' in self.side_metrics.columns and not self.side_metrics['arm_swing_amplitude'].empty:
                #     html_content.append(f"                                <td>{row['arm_swing_amplitude']:.1f}°</td>")
                # if 'strike_landing_stiffness' in self.side_metrics.columns:
                #     html_content.append(f"                                <td>{row['strike_landing_stiffness']:.1f}</td>")
                # if 'stride_frequency' in self.side_metrics.columns:
                #     html_content.append(f"                                <td>{row['stride_frequency']:.1f}</td>")
                # if 'stride_length_cm' in self.side_metrics.columns:
                #     html_content.append(f"                                <td>{row['stride_length_cm']:.1f} cm</td>")
                # if 'stance_phase_detected' in self.side_metrics.columns:
                #     html_content.append(f"                                <td>{'Yes' if row['stance_phase_detected'] else 'No'}</td>")
                
                html_content.append("                            </tr>")
            
            html_content.extend([
                "                        </tbody>",
                "                    </table>",
                "                </div>",
                "                <script>",
                "                    function toggleTable() {",
                "                        var table = document.getElementById('metricsTable');",
                "                        if (table.style.display === 'none') {",
                "                            table.style.display = 'block';",
                "                        } else {",
                "                            table.style.display = 'none';",
                "                        }",
                "                    }",
                "                </script>",
                "            </div>",
                "        </div>",
            ])
        
        # Add Summary and Conclusion Section
        if self.side_metrics is not None:
            # Calculate overall performance score based on multiple metrics
            score_components = []
            
            # Assess foot strike pattern
            foot_strike_counts = self.side_metrics['strike_pattern'].value_counts().drop("not_applicable", errors='ignore')
            primary_foot_strike = foot_strike_counts.idxmax() if not foot_strike_counts.empty else None
            if primary_foot_strike == 'midfoot':
                score_components.append(10)  # Ideal foot strike
            elif primary_foot_strike == 'forefoot':
                score_components.append(8)  # Good but may cause calf strain
            elif primary_foot_strike == 'heel':
                score_components.append(6)  # Suboptimal but common
                
            # Assess trunk angle
            trunk_mean = self.side_metrics['trunk_angle_degrees'].mean()
            if 85 <= 100 - trunk_mean <= 95:
                score_components.append(10)  # Optimal upright posture
            elif 80 <= 100 - trunk_mean < 85 or 95 < 100 - trunk_mean <= 100:
                score_components.append(8)  # Slightly off optimal
            else:
                score_components.append(5)  # Suboptimal
            
            # Assess knee symmetry
            if 'knee_angle_left' in self.side_metrics.columns and 'knee_angle_right' in self.side_metrics.columns:
                left_avg = self.side_metrics['knee_angle_left'].mean()
                right_avg = self.side_metrics['knee_angle_right'].mean()
                diff_percent = abs(left_avg - right_avg) / ((left_avg + right_avg) / 2) * 100
                
                if diff_percent < 5:
                    score_components.append(10)  # Excellent symmetry
                elif diff_percent < 10:
                    score_components.append(8)  # Good symmetry
                elif diff_percent < 15:
                    score_components.append(6)  # Fair symmetry
                else:
                    score_components.append(4)  # Poor symmetry
            
            # # Assess arm swing if available
            if 'arm_swing_amplitude' in self.side_metrics.columns and not self.side_metrics['arm_swing_amplitude'].empty:
                arm_swing_amp = self.side_metrics['arm_swing_amplitude'].mean()
                if arm_swing_amp > 45:
                    score_components.append(10)  # Optimal
                elif arm_swing_amp > 35:
                    score_components.append(8)  # Good
                elif arm_swing_amp > 25:
                    score_components.append(6)  # Fair
                else:
                    score_components.append(4)  # Limited
            
            # Calculate overall score if we have components
            if score_components:
                overall_score = sum(score_components) / len(score_components) * 10
                
                # Determine performance category
                if overall_score >= 90:
                    performance_category = "Excellent"
                    performance_class = "rating-optimal"
                elif overall_score >= 80:
                    performance_category = "Good"
                    performance_class = "rating-good"
                elif overall_score >= 70:
                    performance_category = "Fair"
                    performance_class = "rating-fair"
                else:
                    performance_category = "Needs Improvement"
                    performance_class = "rating-needs-work"
                
                html_content.extend([
                    "        <div class='section'>",
                    "            <h2>Overall Assessment</h2>",
                    "            <div class='metric-box'>",
                    "                <div style='text-align: center;'>",
                    f"                    <div class='metric-value'>{overall_score:.1f}/100</div>",
                    f"                    <div class='rating {performance_class}'>{performance_category}</div>",
                    "                </div>",
                    "                <div class='progress-container' style='height: 30px; margin: 20px 0;'>",
                    f"                    <div class='progress-bar' style='width: {overall_score}%; height: 30px; background-color: {'#2ecc71' if overall_score >= 90 else '#3498db' if overall_score >= 80 else '#f39c12' if overall_score >= 70 else '#e74c3c'};'>{overall_score:.1f}%</div>",
                    "                </div>",
                    "                <h3>Key Strengths</h3>",
                    "                <ul>",
                ])
                
                # Add strengths based on metrics
                strengths = []
                
                if primary_foot_strike == 'midfoot':
                    strengths.append("Efficient midfoot strike pattern")
                elif primary_foot_strike == 'forefoot':
                    strengths.append("Powerful forefoot strike pattern")
                    
                if 85 <= 100 - trunk_mean <= 95:
                    strengths.append("Excellent upright posture")
                    
                if 'knee_angle_left' in self.side_metrics.columns and 'knee_angle_right' in self.side_metrics.columns:
                    if diff_percent < 5:
                        strengths.append("Exceptional bilateral symmetry")
                        
                if 'arm_swing_amplitude' in self.side_metrics.columns and arm_swing_amp > 45:
                    strengths.append("Optimal arm swing mechanics")
                    
                if 'stride_frequency' in self.side_metrics.columns:
                    stride_freq = self.side_metrics['stride_frequency'].mean()
                    cadence = stride_freq * 2
                    if 170 <= cadence <= 190:
                        strengths.append(f"Ideal cadence ({cadence:.1f} steps/min)")
                
                # If no strengths identified, add generic message
                if not strengths:
                    strengths.append("Consistent running form")
                    
                # Add up to 3 strengths
                for strength in strengths[:3]:
                    html_content.append(f"                    <li>{strength}</li>")
                
                html_content.extend([
                    "                </ul>",
                    "                <h3>Areas for Improvement</h3>",
                    "                <ul>",
                ])
                
                # Add areas for improvement based on metrics
                improvements = []
                
                if primary_foot_strike == 'heel':
                    improvements.append("Work on transitioning from heel strike to midfoot landing")
                    
                if not (85 <= 100 - trunk_mean <= 95):
                    if 100 - trunk_mean < 85:
                        improvements.append("Focus on more upright posture to reduce forward lean")
                    else:
                        improvements.append("Avoid excessive backward lean during running")
                        
                if 'knee_angle_left' in self.side_metrics.columns and 'knee_angle_right' in self.side_metrics.columns and diff_percent > 10:
                    improvements.append("Address bilateral asymmetry between left and right knee angles")
                    
                if 'arm_swing_amplitude' in self.side_metrics.columns and arm_swing_amp < 35:
                    improvements.append("Increase arm swing amplitude for better running economy")
                    
                if 'stride_frequency' in self.side_metrics.columns:
                    stride_freq = self.side_metrics['stride_frequency'].mean()
                    cadence = stride_freq * 2
                    if cadence < 160:
                        improvements.append(f"Increase cadence (currently {cadence:.1f} steps/min) to reduce overstriding")
                    elif cadence > 200:
                        improvements.append(f"Consider slightly lower cadence (currently {cadence:.1f} steps/min) for distance running")
                
                # If no improvements identified, add generic message
                if not improvements:
                    improvements.append("Continue maintaining current form while gradually increasing training volume")
                    
                # Add up to 3 areas for improvement
                for improvement in improvements[:3]:
                    html_content.append(f"                    <li>{improvement}</li>")
                
                html_content.extend([
                    "                </ul>",
                    "            </div>",
                    "        </div>",
                ])
        
        # Close HTML
        html_content.extend([
            "    </div>",
            "</body>",
            "</html>",
        ])
        
        # Write HTML to file
        with open(report_file, 'w') as f:
            f.write("\n".join(html_content))
    
    # def _generate_pdf_report(self, report_file):
    #     """Generate a PDF report with plots and analysis."""
    #     # This is a placeholder - in a real implementation,
    #     # you would use a library like reportlab or weasyprint to generate PDF
    #     print("PDF report generation is not yet implemented")
        
    #     # For now, generate HTML report and save it with .pdf extension
    #     html_file = report_file.replace('.pdf', '.html')
    #     self._generate_html_report(html_file)
    #     print(f"Generated HTML report instead: {html_file}")
    
    def _save_rear_metric_plots(self):
        """Create and save plots of running metrics."""
        if self.rear_metrics is None:
            return []
        
        # Create a directory for plots
        plots_dir = os.path.join(self.reports_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        plot_files = []
        
        # Plot 1: Foot distance from midline
        plt.figure(figsize=(10, 6))
        plt.plot(self.rear_metrics['frame_number'], -self.rear_metrics['left_distance_from_midline'], 'bo-', label='Left Foot')
        plt.plot(self.rear_metrics['frame_number'], -self.rear_metrics['right_distance_from_midline'], 'ro-', label='Right Foot')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        plt.xlabel('Frame Number')
        plt.ylabel('Distance from Midline')
        plt.title('Foot Distance from Midline Over Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(plots_dir, f"{self.session_id}_foot_distance_midline.png")
        plt.savefig(plot_file)
        plot_files.append(plot_file)
        plt.close()

        # Plot 3: Hip drop with direction indicator
        plt.figure(figsize=(10, 6))
        colors = []
        for direction in self.rear_metrics['hip_drop_direction']:
            if direction == 'left':
                colors.append('blue')
            elif direction == 'right':
                colors.append('red')
            else:  # neutral
                colors.append('green')

        plt.bar(self.rear_metrics['frame_number'], self.rear_metrics['hip_drop_value'], color=colors)
        plt.xlabel('Frame Number')
        plt.ylabel('Hip Drop Value (m)')
        plt.title('Hip Drop with Direction')

        # Create custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', label='Left Drop'),
            Patch(facecolor='red', label='Right Drop'),
            Patch(facecolor='green', label='Neutral')
        ]
        plt.legend(handles=legend_elements)
        plt.grid(True, axis='y')
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(plots_dir, f"{self.session_id}_hip_drop.png")
        plt.savefig(plot_file)
        plot_files.append(plot_file)
        plt.close()

        # Plot 4: Pelvic tilt angle with elevated side indicator
        plt.figure(figsize=(10, 6))
        markers = []
        for side in self.rear_metrics['pelvic_tilt_elevated_side']:
            if side == 'left':
                markers.append('^')  # triangle up
            elif side == 'right':
                markers.append('v')  # triangle down
            else:  # neutral
                markers.append('o')  # circle

        for i, (x, y, marker) in enumerate(zip(self.rear_metrics['frame_number'], self.rear_metrics['pelvic_tilt_angle'], markers)):
            if self.rear_metrics['pelvic_tilt_elevated_side'][i] == 'left':
                plt.plot(x, y, marker, color='blue', markersize=10, label='Left Elevated' if i == 0 else "")
            elif self.rear_metrics['pelvic_tilt_elevated_side'][i] == 'right':
                plt.plot(x, y, marker, color='red', markersize=10, label='Right Elevated' if i == 1 else "")
            else:  # neutral
                plt.plot(x, y, marker, color='green', markersize=10, label='Neutral' if i == 2 else "")

        plt.plot(self.rear_metrics['frame_number'], self.rear_metrics['pelvic_tilt_angle'], 'k--', alpha=0.5)
        plt.axhline(y=3, color='g', linestyle='--', alpha=0.7, label='Optimal')
        plt.axhline(y=6, color='y', linestyle='--', alpha=0.5, label='Moderate Limit')
        plt.axhline(y=10, color='r', linestyle='--', alpha=0.5, label='Severe Limit')
        plt.xlabel('Frame Number')
        plt.ylabel('Pelvic Tilt Angle (degrees)')
        plt.title('Pelvic Tilt Angle with Elevated Side')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(plots_dir, f"{self.session_id}_pelvic_tilt.png")
        plt.savefig(plot_file)
        plot_files.append(plot_file)
        plt.close()

        # # Plot 5: Combined biomechanics plot
        # plt.figure(figsize=(12, 8))

        # # Normalize values for comparison
        # max_dist = max(max(abs(min(self.rear_metrics['left_distance_from_midline'])), max(self.rear_metrics['right_distance_from_midline'])))
        # normalized_left = [x / max_dist for x in self.rear_metrics['left_distance_from_midline']]
        # normalized_right = [x / max_dist for x in self.rear_metrics['right_distance_from_midline']]
        # normalized_hip = [x / max(abs(min(self.rear_metrics['hip_drop_value'])), max(self.rear_metrics['hip_drop_value'])) for x in self.side_metrics['hip_drop_value']]
        # normalized_pelvic = [x / max(abs(min(self.rear_metrics['pelvic_tilt_angle'])), max(self.rear_metrics['pelvic_tilt_angle'])) for x in self.side_metrics['pelvic_tilt_angle']]

        # plt.plot(self.rear_metrics['frame_number'], normalized_left, 'b-', label='Left Foot Position (norm)')
        # plt.plot(self.rear_metrics['frame_number'], normalized_right, 'r-', label='Right Foot Position (norm)')
        # plt.plot(self.rear_metrics['frame_number'], normalized_hip, 'g-', label='Hip Drop (norm)')
        # plt.plot(self.rear_metrics['frame_number'], normalized_pelvic, 'y-', label='Pelvic Tilt (norm)')

        # # Add stride phase indicators (assuming frame 2 is mid-stride)
        # plt.axvline(x=2, color='purple', linestyle='--', alpha=0.5, label='Mid-stride')

        # plt.xlabel('Frame Number')
        # plt.ylabel('Normalized Values')
        # plt.title('Combined Running Biomechanics')
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        
        # # Save plot
        # plot_file = os.path.join(plots_dir, f"{self.session_id}_rear_initial_combined.png")
        # plt.savefig(plot_file)
        # plot_files.append(plot_file)
        # plt.close()

        
        return plot_files
    
    def _save_side_metric_plots(self):
        """Create and save plots of running metrics."""
        if self.side_metrics is None:
            return []
        
        # Create a directory for plots
        plots_dir = os.path.join(self.reports_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        plot_files = []
        
        # Plot 1: Foot strike pattern over time
        plt.figure(figsize=(10, 6))
        foot_strike_mapping = {'heel': 1, 'midfoot': 2, 'forefoot': 3}
        numeric_foot_strike = self.side_metrics['strike_pattern'].map(foot_strike_mapping)
        plt.plot(self.side_metrics['frame_number'], numeric_foot_strike, 'o')
        plt.yticks([1, 2, 3], ['Heel', 'Midfoot', 'Forefoot'])
        plt.xlabel('Frame Number')
        plt.ylabel('Foot Strike Pattern')
        plt.title('Foot Strike Pattern Over Time')
        plt.grid(True)
        
        # Save plot
        plot_file = os.path.join(plots_dir, f"{self.session_id}_foot_strike.png")
        plt.savefig(plot_file)
        plot_files.append(plot_file)
        plt.close()
        
        # Plot 2: Trunk angle
        plt.figure(figsize=(10, 6))
        plt.plot(self.side_metrics['frame_number'], self.side_metrics['trunk_angle_degrees'], 'b-', label='Trunk Angle')
        # Add reference lines for optimal range
        plt.axhline(y=100 - 90, color='g', linestyle='--', alpha=0.7, label='Optimal')
        plt.axhline(y=100 - 85, color='y', linestyle='--', alpha=0.5, label='Lower Limit')
        plt.axhline(y=100 - 95, color='y', linestyle='--', alpha=0.5, label='Upper Limit')
        plt.xlabel('Frame Number')
        plt.ylabel('Angle (degrees)')
        plt.title('Trunk Angle Over Time')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plot_file = os.path.join(plots_dir, f"{self.session_id}_trunk_angle.png")
        plt.savefig(plot_file)
        plot_files.append(plot_file)
        plt.close()
        
        # Plot 3: Foot Landing Over Time
        plt.figure(figsize=(10, 6))
        foot_landing_mapping = {'behind': 1, 'under': 2, 'ahead': 3}
        numeric_foot_landing = self.side_metrics['foot_landing_position_category'].map(foot_landing_mapping)
        plt.plot(self.side_metrics['frame_number'], numeric_foot_landing, 'o')
        plt.yticks([1, 2, 3], ['Behind', 'Under', 'Ahead'])
        plt.xlabel('Frame Number')
        plt.ylabel('Foot Landing Pattern')
        plt.title('Foot Landing Pattern Over Time')
        plt.grid(True)
        
        # Save plot
        plot_file = os.path.join(plots_dir, f"{self.session_id}_foot_landing.png")
        plt.savefig(plot_file)
        plot_files.append(plot_file)
        plt.close()
        
        # Plot 4: Knee Angle Comparison (Left vs Right)
        if 'knee_angle_left' in self.side_metrics.columns and 'knee_angle_right' in self.side_metrics.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(self.side_metrics['frame_number'], self.side_metrics['knee_angle_left'], 'b-', label='Left Knee')
            plt.plot(self.side_metrics['frame_number'], self.side_metrics['knee_angle_right'], 'r-', label='Right Knee')
            plt.xlabel('Frame Number')
            plt.ylabel('Knee Angle (degrees)')
            plt.title('Knee Angle Comparison')
            plt.legend()
            plt.grid(True)
            
            # Save plot
            plot_file = os.path.join(plots_dir, f"{self.session_id}_knee_angles.png")
            plt.savefig(plot_file)
            plot_files.append(plot_file)
            plt.close()
        
        # # Plot 5: Arm Swing Analysis
        # if 'arm_swing_amplitude' in self.side_metrics.columns:
        #     plt.figure(figsize=(10, 6))
        #     plt.plot(self.side_metrics['frame_number'], self.side_metrics['arm_swing_amplitude'], 'g-')
        #     plt.xlabel('Frame Number')
        #     plt.ylabel('Amplitude (degrees)')
        #     plt.title('Arm Swing Amplitude')
        #     plt.grid(True)
            
        #     # Add reference lines for optimal zones
        #     plt.axhline(y=45, color='g', linestyle='--', alpha=0.7, label='Optimal Zone')
        #     plt.axhline(y=35, color='y', linestyle='--', alpha=0.5, label='Good Zone')
        #     plt.axhline(y=25, color='r', linestyle='--', alpha=0.5, label='Minimal Zone')
        #     plt.legend()
            
        #     # Save plot
        #     plot_file = os.path.join(plots_dir, f"{self.session_id}_arm_swing.png")
        #     plt.savefig(plot_file)
        #     plot_files.append(plot_file)
        #     plt.close()
        
        # # Plot 6: Stride Length and Frequency
        # if 'stride_length_cm' in self.side_metrics.columns and 'stride_frequency' in self.side_metrics.columns:
        #     fig, ax1 = plt.subplots(figsize=(10, 6))
            
        #     color = 'tab:blue'
        #     ax1.set_xlabel('Frame Number')
        #     ax1.set_ylabel('Stride Length (cm)', color=color)
        #     ax1.plot(self.side_metrics['frame_number'], self.side_metrics['stride_length_cm'], color=color)
        #     ax1.tick_params(axis='y', labelcolor=color)
            
        #     ax2 = ax1.twinx()  # Create a second y-axis sharing the same x-axis
        #     color = 'tab:red'
        #     ax2.set_ylabel('Stride Frequency (strides/min)', color=color)
        #     ax2.plot(self.side_metrics['frame_number'], self.side_metrics['stride_frequency'], color=color)
        #     ax2.tick_params(axis='y', labelcolor=color)
            
        #     fig.tight_layout()
        #     plt.title('Stride Length and Frequency')
            
        #     # Save plot
        #     plot_file = os.path.join(plots_dir, f"{self.session_id}_stride_metrics.png")
        #     plt.savefig(plot_file)
        #     plot_files.append(plot_file)
        #     plt.close()
        
        # Plot 7: Landing Stiffness Analysis
        if 'strike_landing_stiffness' in self.side_metrics.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(self.side_metrics['frame_number'], self.side_metrics['strike_landing_stiffness'], 'o')
            plt.xlabel('Frame Number')
            plt.ylabel('Stiffness Rating (0-10)')
            plt.title('Landing Stiffness Over Time')
            
            # Add reference bands for optimal zones
            plt.yticks([1, 2, 3], ['not_applicable', 'Stiff', 'Compliant'])
            
            plt.ylim(0, 3)
            plt.legend()
            plt.grid(True)
            
            # Save plot
            plot_file = os.path.join(plots_dir, f"{self.session_id}_landing_stiffness.png")
            plt.savefig(plot_file)
            plot_files.append(plot_file)
            plt.close()

        return plot_files
    

    def _generate_rear_recommendations(self):
        """Generate running form recommendations based on metrics."""
        if self.rear_metrics is None:
            return []
        
        recommendations = []
        
        # Add general recommendations if list is short
        if len(recommendations) < 3:
            recommendations.append(
                "Maintain a consistent running cadence between 170-180 steps per minute for optimal efficiency. "
                "A metronome app can help you achieve this rhythm."
            )
            
            recommendations.append(
                "Focus on relaxed shoulders and a proper arm swing that moves forward and backward, not across your body. "
                "This helps maintain rotational balance and overall efficiency."
            )
        
        return recommendations
    

    def _generate_side_recommendations(self):
        """Generate running form recommendations based on metrics."""
        if self.side_metrics is None:
            return []
        
        recommendations = []
        
        # Analyze foot strike pattern
        foot_strike_counts = self.side_metrics['strike_pattern'].value_counts().drop("not_applicable", errors='ignore')
        primary_foot_strike = foot_strike_counts.idxmax() if not foot_strike_counts.empty else None
        
        if primary_foot_strike == 'heel':
            recommendations.append(
                "Your running shows a predominant heel strike pattern. Consider working on a more midfoot landing "
                "to reduce impact forces and improve efficiency. Try shortening your stride slightly and increasing cadence."
            )
        elif primary_foot_strike == 'forefoot':
            recommendations.append(
                "You're landing primarily on your forefoot, which is efficient but can place more stress on your calves "
                "and Achilles. Make sure you're allowing your heel to drop slightly after initial contact for better shock absorption."
            )
        elif primary_foot_strike == 'midfoot':
            recommendations.append(
                "Your midfoot strike pattern is generally efficient and helps distribute impact forces well. "
                "Maintain this landing pattern while focusing on a quick cadence."
            )
        
        # Analyze trunk angle
        trunk_angle_mean = self.side_metrics['trunk_angle_degrees'].mean()
        if 100 - trunk_angle_mean < 85:
            recommendations.append(
                f"Your average trunk angle of {trunk_angle_mean:.1f}° indicates excessive forward lean. "
                "Work on maintaining a more upright posture by engaging your core muscles and focusing on running tall."
            )
        elif 100 - trunk_angle_mean > 95:
            recommendations.append(
                f"Your average trunk angle of {trunk_angle_mean:.1f}° indicates a backward lean. "
                "Focus on a slight forward lean from the ankles rather than the waist to improve running economy."
            )
        else:
            recommendations.append(
                f"Your average trunk angle of {trunk_angle_mean:.1f}° is within the optimal range. "
                "Continue maintaining this posture for optimal running economy."
            )
        
        # Analyze knee angles and symmetry
        if 'knee_angle_left' in self.side_metrics.columns and 'knee_angle_right' in self.side_metrics.columns:
            left_knee_mean = self.side_metrics['knee_angle_left'].mean()
            right_knee_mean = self.side_metrics['knee_angle_right'].mean()
            diff = abs(left_knee_mean - right_knee_mean)
            diff_percent = diff / ((left_knee_mean + right_knee_mean) / 2) * 100
            
            if diff_percent > 10:
                recommendations.append(
                    f"There's a {diff_percent:.1f}% difference between your left and right knee angles, which may indicate muscular imbalances. "
                    "Consider targeted strength training to address this asymmetry and reduce injury risk."
                )
        
        # Analyze arm swing
        if 'arm_swing_amplitude' in self.side_metrics.columns:
            arm_swing_mean = self.side_metrics['arm_swing_amplitude'].mean()
            if arm_swing_mean < 30:
                recommendations.append(
                    f"Your arm swing amplitude of {arm_swing_mean:.1f}° is limited. "
                    "Try to increase your arm movement with a 90° bend at the elbow, allowing your arms to swing naturally from your shoulders."
                )
        
        if 'arm_swing_symmetry' in self.side_metrics.columns:
            # arm_symmetry = self.side_metrics['arm_swing_symmetry'].mean()

            arm_swing_symmetry_counts = self.side_metrics['arm_swing_symmetry'].value_counts()['symmetrical'] if 'arm_swing_symmetry' in self.side_metrics.columns else None
            arm_symmetry = (arm_swing_symmetry_counts / len(self.side_metrics['arm_swing_symmetry'])) * 100
            if arm_symmetry < 75:
                recommendations.append(
                    f"Your arm swing shows asymmetry at {arm_symmetry:.1f}% similarity. "
                    "Focus on balanced arm movement to improve overall running efficiency and reduce rotation."
                )
        
        # Analyze foot landing position
        foot_landing_counts = self.side_metrics['foot_landing_position_category'].value_counts().drop("not_applicable", errors='ignore')
        primary_foot_landing = foot_landing_counts.idxmax() if not foot_landing_counts.empty else None
        
        if primary_foot_landing == 'ahead':
            recommendations.append(
                "Your feet are landing predominantly ahead of your center of mass, which may increase braking forces. "
                "Work on landing with your foot closer to your center of gravity by increasing cadence and focusing on pulling your foot up quickly after contact."
            )
        elif primary_foot_landing == 'behind':
            recommendations.append(
                "Your feet are landing predominantly behind your center of mass, which is unusual. "
                "This may indicate overstriding or form compensation. Focus on a more natural foot landing pattern."
            )
        
        # Analyze stride metrics
        if 'stride_frequency' in self.side_metrics.columns:
            stride_freq = self.side_metrics['stride_frequency'].mean()
            cadence = stride_freq * 2  # Convert to steps per minute
            
            if cadence < 160:
                recommendations.append(
                    f"Your current cadence of {cadence:.1f} steps/minute is relatively low. "
                    "Consider gradually increasing your cadence to 170-180 steps/minute to reduce overstriding and impact forces."
                )
            elif cadence > 200:
                recommendations.append(
                    f"Your current cadence of {cadence:.1f} steps/minute is quite high. "
                    "While this can be efficient for sprinting, for distance running you might benefit from a slightly lower cadence of 170-190 steps/minute."
                )
        
        # Analyze landing stiffness
        if 'strike_landing_stiffness' in self.side_metrics.columns:
            # stiffness = self.side_metrics['strike_landing_stiffness'].mean()
            landing_stiffness_counts = self.side_metrics['strike_landing_stiffness'].value_counts().drop("not_applicable", errors='ignore')
            primary_landing_stiffness = landing_stiffness_counts.idxmax() if not landing_stiffness_counts.empty else "N/A"
            stiffness = (landing_stiffness_counts.max() / landing_stiffness_counts.sum() * 10) if not landing_stiffness_counts.empty else 0
            
            if stiffness < 3:
                recommendations.append(
                    f"Your landing stiffness rating of {stiffness:.1f}/10 indicates a very soft landing. "
                    "While this reduces impact forces, it may reduce energy return. Consider developing more reactive strength for better running economy."
                )
            elif stiffness > 7:
                recommendations.append(
                    f"Your landing stiffness rating of {stiffness:.1f}/10 indicates a very stiff landing. "
                    "This increases impact forces and may lead to injuries. Focus on softer landings with a slightly bent knee at contact."
                )
        
        # Limit to top 5 recommendations to avoid overwhelming the runner
        return recommendations[:]

    def process_direct_video(self, video_path, output_dir=None, watch_data_path=None):
        """Process a video file directly rather than finding session files."""
        print(f"Processing video: {video_path}")
        
        # Extract session info from video path
        video_filename = os.path.basename(video_path)
        session_name = os.path.splitext(video_filename)[0]
        
        # Set output directory
        if output_dir:
            processed_dir = output_dir
        else:
            processed_dir = self.processed_dir
        
        os.makedirs(processed_dir, exist_ok=True)
        
        # Process video
        output_video = os.path.join(processed_dir, f"{session_name}_processed.mp4")
        metrics_df = self.analyzer.process_video(video_path, output_video)
        
        # Save metrics
        metrics_path = os.path.join(processed_dir, f"{session_name}_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Metrics saved to: {metrics_path}")
        
        # Store metrics for report generation
        video_angle = "SIDE" if "side" in video_filename.lower() else "REAR"
        if video_angle == "SIDE":
            self.side_metrics = metrics_df
        else:
            self.rear_metrics = metrics_df
        
        # Merge with watch data if provided
        if watch_data_path and os.path.exists(watch_data_path):
            print(f"Merging with watch data: {watch_data_path}")
            watch_df = pd.read_csv(watch_data_path)
            
            # Merge metrics with watch data
            merged_df = pd.merge_asof(
                metrics_df.sort_values('timestamp'),
                watch_df.sort_values('timestamp'),
                on='timestamp',
                direction='nearest'
            )
            
            # Save merged data
            merged_path = os.path.join(processed_dir, f"{session_name}_merged_metrics.csv")
            merged_df.to_csv(merged_path, index=False)
            print(f"Merged metrics saved to: {merged_path}")
            
            # Update metrics with merged data
            self.side_metrics = merged_df
        
        # Generate report
        self.session_id = session_name
        report_file = os.path.join(processed_dir, f"{session_name}_report.html")

        # Generate report
        if video_angle == "SIDE":
            self._generate_side_html_report(report_file) # hardcoding side for now
        else:
            self._generate_rear_html_report(report_file) # hardcoding side for now
        print(f"Report generated: {report_file}")
        
        return metrics_df


def run_analysis():
    
    # Create analyzer
    analyzer = RunAnalyzer(datetime.now().strftime("%Y%m%d"), 44)
    
    # find and process session files
    try:
        # Find session files
        files_found = analyzer.find_session_files()
        if not files_found:
            print("No session files found")
            return
        
        # Load metadata
        analyzer.load_metadata()
        
        # Process videos
        analyzer.process_videos()

        for file in [analyzer.side_video, analyzer.rear_video]:
        
            # video_angle = "SIDE" if "side" in file.lower() else "REAR" and file
            # Generate report
            if file and "side" in file.lower():
                analyzer.generate_side_report() # turning off for now while rear and side are built
            elif file and "rear" in file.lower():
                analyzer.generate_rear_report() # turning off for now while rear and side are built
            else:
                print("File Mismatch, No Report.")
            
        # assuming it makes it to here without error, delete or archive the original files
        ## This assumes a multi-user approach and not just me

        print("Analysis complete!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    return ['reports' + "/" + f"{analyzer.session_id}_rear_angle_report.html",
        'videos' + "/" + f"{analyzer.session_id}_rear_processed.mp4",
        'reports' + "/" + f"{analyzer.session_id}_side_angle_report.html",
        'videos' + "/" + f"{analyzer.session_id}_side_processed.mp4"]
    # return [f"{analyzer.session_id}_rear_angle_report.html",
    # f"{analyzer.session_id}_rear_processed.mp4",
    # f"{analyzer.session_id}_side_angle_report.html",
    # f"{analyzer.session_id}_side_processed.mp4"]


def get_latest_file(directory, keyword, extension):
    files = glob(os.path.join(directory, f"*{keyword}*.{extension}"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)