# runnervision_service.py
"""
Service layer for running data analysis application.

This module contains all business logic services that handle data processing,
database interactions, and business rules while keeping the Flask routes clean.
"""

import logging

from typing import Dict, List, Any, Optional, Tuple
from dateutil.relativedelta import relativedelta

from flask import send_from_directory, abort
from werkzeug.utils import secure_filename
import os

from services.base_service import BaseService
from utils import db_utils, format_utils, exception_utils
from utils.RunnerVision import runnervision_utils as rv_utils

class RunnerVisionService:
    """Service for RunnerVision biomechanics analysis."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def get_latest_analysis(self) -> Dict[str, Any]:
        """Get latest RunnerVision analysis results."""
        try:
            base_path = 'RunnerVision/processed/2025-05-20'
            
            report_rear = rv_utils.get_latest_file(base_path, 'rear', 'html')
            report_side = rv_utils.get_latest_file(base_path, 'side', 'html')
            video_rear = rv_utils.get_latest_file(base_path, 'rear', 'mp4')
            video_side = rv_utils.get_latest_file(base_path, 'side', 'mp4')
            
            return {
                'report_rear': report_rear.replace("static/", "") if report_rear else None,
                'report_side': report_side.replace("static/", "") if report_side else None,
                'video_rear': video_rear.replace("static/", "") if video_rear else None,
                'video_side': video_side.replace("static/", "") if video_side else None
            }
        except Exception as e:
            self.logger.error(f"Error getting latest analysis: {e}")
            return {}
    
    def serve_report(self, filename: str):
        """Serve report files safely."""
        try:
            # Validate filename for security
            if '..' in filename or filename.startswith('/'):
                abort(403)
            
            return send_from_directory('reports', filename)
        except FileNotFoundError:
            abort(404)
        except Exception as e:
            self.logger.error(f"Error serving report {filename}: {e}")
            abort(500)
    
    def serve_video(self, filename: str):
        """Serve video files safely."""
        try:
            # Security check
            if '..' in filename or filename.startswith('/'):
                abort(403)
            
            filepath = os.path.join(self.config.VIDEO_FOLDER, filename)
            if not os.path.isfile(filepath):
                abort(404)
            
            return send_from_directory(
                self.config.VIDEO_FOLDER, 
                filename, 
                mimetype='video/mp4'
            )
        except Exception as e:
            self.logger.error(f"Error serving video {filename}: {e}")
            abort(500)
    
    def handle_file_upload(self, files: List) -> Dict[str, Any]:
        """Handle video file uploads."""
        if not files or all(file.filename == '' for file in files):
            return {"error": "No selected files", "success": False}
        
        saved_files = []
        errors = []
        
        for file in files:
            if file and file.filename != '' and self._allowed_file(file.filename):
                try:
                    filename = secure_filename(file.filename)
                    if 'run' not in filename.lower():
                        filename = f"run_{filename}"
                    
                    save_path = os.path.join(self.config.UPLOAD_FOLDER, filename)
                    file.save(save_path)
                    saved_files.append(filename)
                except Exception as e:
                    errors.append(f"Failed to save {file.filename}: {str(e)}")
            else:
                errors.append(f"Invalid file: {file.filename}")
        
        if errors and not saved_files:
            return {"error": ", ".join(errors), "success": False}
        
        return {
            "success": True,
            "message": f"Uploaded: {', '.join(saved_files)}",
            "files": saved_files,
            "warnings": errors if errors else None
        }
    
    def run_analysis(self) -> Dict[str, Any]:
        """Execute biomechanics analysis."""
        try:
            result = rv_utils.run_analysis()
            return {
                "status": "complete",
                "message": "Analysis finished!",
                "rear_report_path": result[0],
                "rear_video_path": result[1],
                "side_report_path": result[2],
                "side_video_path": result[3]
            }
        except Exception as e:
            self.logger.error(f"Analysis execution error: {e}")
            raise
    
    def _allowed_file(self, filename: str) -> bool:
        """Check if file extension is allowed."""
        return ('.' in filename and 
                filename.rsplit('.', 1)[1].lower() in self.config.ALLOWED_EXTENSIONS)
