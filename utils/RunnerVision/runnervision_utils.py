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

from runnervision_utils.reports.report_generators import SideViewReportGenerator, RearViewReportGenerator

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
            self.side_video_file = f"{self.session_id}_side_processed.mp4"
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
            self.rear_video_file = f"{self.session_id}_rear_processed.mp4"
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
            side_report_generator = SideViewReportGenerator(
                metrics_df=self.side_metrics,
                session_id=self.session_id,
                reports_dir=self.reports_dir, # Pass the base reports directory
                metadata=self.metadata
                # report_file_path is handled by generate_html_file or can be passed if needed
            )
            generated_report_path = side_report_generator.generate_html_file(output_filename_base="side_angle") # Matches your previous output
            self.side_report_file = side_report_generator.report_file_name
            if generated_report_path:
                 print(f"Side view report generated: {generated_report_path}") # Or use logging
            else:
                 print(f"Failed to generate side view report for session: {self.session_id}")
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
            rear_report_generator = RearViewReportGenerator(
                metrics_df=self.rear_metrics,
                session_id=self.session_id,
                reports_dir=self.reports_dir,
                metadata=self.metadata
            )
            generated_report_path = rear_report_generator.generate_html_file(output_filename_base="rear_angle")
            self.rear_report_file = rear_report_generator.report_file_name
            if generated_report_path:
                 print(f"Rear view report generated: {generated_report_path}") # Or use logging
            else:
                 print(f"Failed to generate rear view report for session: {self.session_id}")

        elif output_format == 'pdf':
            self._generate_pdf_report(report_file)
        else:
            print(f"Unsupported output format: {output_format}")
            return
        
        print(f"Report generated: {report_file}")    


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
        
            # Generate report
            if file and "side" in file.lower():
                analyzer.generate_side_report() 
            elif file and "rear" in file.lower():
                analyzer.generate_rear_report() 
            else:
                print("File Mismatch, No Report.")

        print("Analysis complete!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    # This return is very frail
    return ['reports' + "/" + analyzer.rear_report_file if hasattr(analyzer, "rear_report_file") else "",
        'videos' + "/" + analyzer.rear_video_file if hasattr(analyzer, "rear_video_file") else "",
        'reports' + "/" + analyzer.side_report_file if hasattr(analyzer, "side_report_file") else "",
        'videos' + "/" + analyzer.side_video_file if hasattr(analyzer, "side_video_file") else ""]

def get_latest_file(directory, keyword, extension):
    files = glob(os.path.join(directory, f"*{keyword}*.{extension}"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)