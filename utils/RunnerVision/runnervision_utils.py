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
import sqlite3
from typing import Optional, Tuple
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2
from datetime import datetime
from glob import glob
import logging

from utils.db import runnervision_db_utils

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from runnervision_utils.reports.report_generators.rear_report_details.rear_report_generator import RearViewReportGenerator
from runnervision_utils.reports.report_generators.side_report_details.side_report_generator import SideViewReportGenerator
from utils.RunnerVision.RunnerVisionClass import RunnerVisionAnalyzer

class RunAnalyzer:
    def __init__(self, session_date=None, session_id=None, athlete_id = 24266563):
        """Initialize the run analyzer with session details."""
        self.base_dir = os.getcwd()

        if session_date is None:
            self.session_date = datetime.now().strftime('%Y-%m-%d')
        else:
            self.session_date = session_date
        self.user_id = str(athlete_id)
        self.session_id = self.session_date + str(athlete_id)
        
        self.videos_upload_dir = os.path.join(self.base_dir, "uploads")
        self.videos_output_dir = os.path.join(self.base_dir, 'videos')
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.processed_dir = os.path.join(self.base_dir, 'processed')
        self.reports_dir = os.path.join(self.base_dir, 'reports')
        
        for directory in [self.data_dir, self.processed_dir, self.reports_dir]:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")
            
        self.analyzer = RunnerVisionAnalyzer()
        logger.info("Initialized RunAnalyzer")

    def upload_cleanup(self, video_directory="uploads"):
        try:
            files = os.listdir(video_directory)
            for file_name in files:
                file_path = os.path.join(video_directory, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    logger.debug(f"Deleted file during upload cleanup: {file_path}")
            logger.info("Upload video cleanup successful")
        except Exception as e:
            logger.error(f"Upload video cleanup failed: {e}")

    def find_session_files(self):
        if not os.path.exists(self.videos_upload_dir):
            raise FileNotFoundError(f"Video directory not found: {self.videos_upload_dir}")
        
        all_files = os.listdir(self.videos_upload_dir)
        self.side_video = next((os.path.join(self.videos_upload_dir, f) for f in all_files if "side" in f.lower()), None)
        self.rear_video = next((os.path.join(self.videos_upload_dir, f) for f in all_files if "rear" in f.lower()), None)
        self.metadata_file = next((os.path.join(self.videos_upload_dir, f) for f in all_files if "metadata.txt" in f), None)
        
        if os.path.exists(self.data_dir):
            data_files = os.listdir(self.data_dir)
            self.watch_data = next((os.path.join(self.data_dir, f) for f in data_files 
                                   if f.startswith(self.session_id) and f.endswith('.csv')), None)
        else:
            self.watch_data = None

        logger.info(f"Found session files for session: {self.session_id}")
        logger.debug(f"Side video: {self.side_video}")
        logger.debug(f"Rear video: {self.rear_video}")
        logger.debug(f"Metadata file: {self.metadata_file}")
        logger.debug(f"Watch data: {self.watch_data}")

        return bool(self.side_video or self.rear_video)

    def load_metadata(self):
        if not self.metadata_file or not os.path.exists(self.metadata_file):
            logger.warning("No metadata file found")
            self.metadata = {}
            return
        
        self.metadata = {}
        with open(self.metadata_file, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    self.metadata[key.strip()] = value.strip()
        
        logger.info("Metadata loaded successfully")
        logger.debug(f"Metadata contents: {self.metadata}")

    def process_videos(self, conn: sqlite3.Connection):
        if self.side_video:
            logger.info(f"Processing side video: {self.side_video}")
            side_output = os.path.join(self.videos_output_dir, f"{self.session_id}_side_processed.mp4")
            self.side_metrics = self.analyzer.process_video(self.side_video, side_output)
            self.side_video_file = f"{self.session_id}_side_processed.mp4"
            side_metrics_path = os.path.join(self.data_dir, f"{self.session_id}_side_metrics.csv")
            self.side_metrics.to_csv(side_metrics_path, index=False)
            logger.info(f"Side metrics saved to: {side_metrics_path}")
            self.process_analysis_complete(
                    conn=conn,
                    user_id=self.user_id,
                    rear_df=None,
                    side_df=self.side_metrics,
                    video_duration_seconds=self.side_metrics['timestamp'].iloc[-1],
                    fps=60.0
                    )
        else:
            self.side_metrics = None
            logger.warning("No side video found")

        if self.rear_video:
            logger.info(f"Processing rear video: {self.rear_video}")
            rear_output = os.path.join(self.videos_output_dir, f"{self.session_id}_rear_processed.mp4")
            self.rear_video_file = f"{self.session_id}_rear_processed.mp4"
            rear_analyzer = RunnerVisionAnalyzer()
            self.rear_metrics = rear_analyzer.process_video(self.rear_video, rear_output)
            rear_metrics_path = os.path.join(self.data_dir, f"{self.session_id}_rear_metrics.csv")
            self.rear_metrics.to_csv(rear_metrics_path, index=False)
            logger.info(f"Rear metrics saved to: {rear_metrics_path}")
            self.process_analysis_complete(
                    conn=conn,
                    user_id=self.user_id,
                    rear_df=self.rear_metrics,
                    side_df=None,
                    video_duration_seconds=self.rear_metrics['timestamp'].iloc[-1],
                    fps=60.0
                    )
        else:
            self.rear_metrics = None
            logger.warning("No rear video found")

    def merge_with_watch_data(self):
        try:
            watch_df = pd.read_csv(self.watch_data)
            if self.side_metrics is not None:
                merged_df = pd.merge_asof(
                    self.side_metrics.sort_values('timestamp'),
                    watch_df.sort_values('timestamp'),
                    on='timestamp',
                    direction='nearest'
                )
                merged_path = os.path.join(self.data_dir, f"{self.session_id}_merged_metrics.csv")
                merged_df.to_csv(merged_path, index=False)
                self.side_metrics = merged_df
                logger.info(f"Merged metrics saved to: {merged_path}")
        except Exception as e:
            logger.error(f"Error merging with watch data: {e}")

    def generate_side_report(self, output_format='html'):
        if self.side_metrics is None or self.side_metrics.empty:
            logger.warning("No metrics available to generate side report")
            return
        
        logger.info("Generating side view report")
        report_file = os.path.join(self.reports_dir, f"{self.session_id}_side_angle_report.{output_format}")
        
        if output_format == 'html':
            side_report_generator = SideViewReportGenerator(
                metrics_df=self.side_metrics,
                session_id=self.session_id,
                reports_dir=self.reports_dir,
                metadata=self.metadata
            )
            generated_report_path = side_report_generator.generate_html_file(output_filename_base="side_angle")
            self.side_report_file = side_report_generator.report_file_name
            if generated_report_path:
                logger.info(f"Side view report generated: {generated_report_path}")
            else:
                logger.error("Failed to generate side view report")
        elif output_format == 'pdf':
            self._generate_pdf_report(report_file)
        else:
            logger.warning(f"Unsupported output format: {output_format}")

    def generate_rear_report(self, output_format='html'):
        if self.rear_metrics is None or self.rear_metrics.empty:
            logger.warning("No metrics available to generate rear report")
            return
        
        logger.info("Generating rear view report")
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
                logger.info(f"Rear view report generated: {generated_report_path}")
            else:
                logger.error("Failed to generate rear view report")
        elif output_format == 'pdf':
            self._generate_pdf_report(report_file)
        else:
            logger.warning(f"Unsupported output format: {output_format}")

    def process_analysis_complete(self,
        conn: sqlite3.Connection,
        user_id: str,
        rear_df: Optional[pd.DataFrame] = None,
        side_df: Optional[pd.DataFrame] = None,
        video_duration_seconds: float = 0.0,
        fps: float = 30.0
    ) -> str:
        """
        Complete workflow: creates session, inserts raw data, generates summaries.
        Returns session_id.
        """
        # Determine analysis type and frame count
        analysis_type = "both"
        total_frames = 0
        
        if rear_df is not None and side_df is not None:
            analysis_type = "both"
            total_frames = max(len(rear_df), len(side_df))
        elif rear_df is not None:
            analysis_type = "rear"
            total_frames = len(rear_df)
        elif side_df is not None:
            analysis_type = "side"
            total_frames = len(side_df)
        else:
            raise ValueError("At least one DataFrame (rear_df or side_df) must be provided")
        
        # Create session
        try:
            session_id = runnervision_db_utils.create_analysis_session(
                conn, user_id, video_duration_seconds, total_frames, fps, analysis_type
            )
        except Exception as e:
            logger.error(f"Error creating analysis session: {e}", exc_info=True)

        # Insert raw data
        if rear_df is not None:
            try:
                runnervision_db_utils.insert_rear_metrics_raw(conn, session_id, rear_df)
            except Exception as e:
                logger.error(f"Error inserting rear metrics: {e}", exc_info=True)
            try:
                runnervision_db_utils.generate_rear_summary(conn, session_id)
            except Exception as e:
                logger.error(f"Error inserting rear summary: {e}", exc_info=True) 
        
        if side_df is not None:
            try:
                runnervision_db_utils.insert_side_metrics_raw(conn, session_id, side_df)
            except Exception as e:
                logger.error(f"Error inserting side metrics: {e}", exc_info=True)
            try:
                runnervision_db_utils.generate_side_summary(conn, session_id)
            except Exception as e:
                logger.error(f"Error inserting side summary: {e}", exc_info=True) 

        return session_id

def run_analysis(conn: sqlite3.Connection):
    analyzer = RunAnalyzer(datetime.now().strftime("%Y%m%d"), 44)
    
    try:
        files_found = analyzer.find_session_files()
        if not files_found:
            logger.warning("No session files found")
            return

        analyzer.load_metadata()
        analyzer.process_videos(conn = conn)

        for file in [analyzer.side_video, analyzer.rear_video]:
            if file and "side" in file.lower():
                analyzer.generate_side_report()
            elif file and "rear" in file.lower():
                analyzer.generate_rear_report()
            else:
                logger.warning("File mismatch. No report generated.")
        
        logger.info("Analysis complete")

    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)

    analyzer.upload_cleanup(video_directory=analyzer.videos_upload_dir)

    return ['reports/' + analyzer.rear_report_file if hasattr(analyzer, "rear_report_file") else "",
            'videos/' + analyzer.rear_video_file if hasattr(analyzer, "rear_video_file") else "",
            'reports/' + analyzer.side_report_file if hasattr(analyzer, "side_report_file") else "",
            'videos/' + analyzer.side_video_file if hasattr(analyzer, "side_video_file") else ""]

def get_latest_file(directory, keyword, extension):
    files = glob(os.path.join(directory, f"*{keyword}*.{extension}"))
    if not files:
        logger.debug(f"No files found in {directory} matching *{keyword}*.{extension}")
        return None
    latest_file = max(files, key=os.path.getmtime)
    logger.debug(f"Latest file found: {latest_file}")
    return latest_file