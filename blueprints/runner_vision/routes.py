# blueprints/runner_vision/routes.py
"""
RunnerVision Blueprint - Biomechanics analysis functionality.
Includes: Video upload, analysis execution, and report serving.
"""

import logging
import os
from flask import Blueprint, render_template, request, jsonify
from services.runnervision_service import RunnerVisionService

logger = logging.getLogger(__name__)

# Create blueprint
template_dir = os.path.join(os.path.dirname(__file__), 'templates')
static_dir = os.path.join(os.path.dirname(__file__), 'static')

runner_vision_bp = Blueprint(
    'runner_vision',
    __name__,
    template_folder=template_dir,
    static_folder=static_dir,
    url_prefix='/vision'
)

def init_runner_vision_blueprint(config):
    """Initialize services for RunnerVision blueprint."""
    runnervision_service = RunnerVisionService(config)
    register_routes(runnervision_service)
    return runner_vision_bp

def register_routes(runnervision_service):
    """Register all RunnerVision routes."""
    
    @runner_vision_bp.route('/')
    @runner_vision_bp.route('/runnervision')
    def runnervision():
        """Display RunnerVision analysis results."""
        try:
            analysis_data = runnervision_service.get_latest_analysis()
            return render_template('runnervision.html', **analysis_data)
        except Exception as e:
            logger.error(f"Error loading RunnerVision page: {e}")
            return render_template('runnervision.html')
    
    @runner_vision_bp.route('/reports/<path:filename>')
    def serve_report(filename):
        """Serve RunnerVision report files."""
        return runnervision_service.serve_report(filename)
    
    @runner_vision_bp.route('/videos/<path:filename>')
    def serve_video(filename):
        """Serve RunnerVision video files."""
        return runnervision_service.serve_video(filename)
    
    @runner_vision_bp.route('/upload', methods=['POST'])
    def upload_files():
        """Handle video file uploads."""
        if 'videos' not in request.files:
            return jsonify({"error": "No files part in the request"}), 400
        
        files = request.files.getlist('videos')
        try:
            result = runnervision_service.handle_file_upload(files)
            return jsonify(result), 200 if result.get('success') else 400
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return jsonify({"error": "Upload failed"}), 500
    
    @runner_vision_bp.route('/run_biomechanic_analysis', methods=['POST'])
    def run_analysis():
        """Execute biomechanics analysis."""
        try:
            result = runnervision_service.run_analysis()
            return jsonify(result)
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return jsonify({"error": "Analysis failed"}), 500