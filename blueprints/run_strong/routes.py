# blueprints/run_strong/routes.py

"""
RunStrong Blueprint - Strength training functionality.
Includes: Exercise library, routine planning, workout journaling, and dashboard.
"""

import logging
import os
from flask import Blueprint, render_template, request, jsonify
from werkzeug.exceptions import BadRequest
from services.runstrong_service import RunStrongService

logger = logging.getLogger(__name__)

# Create blueprint
template_dir = os.path.join(os.path.dirname(__file__), 'templates')
static_dir = os.path.join(os.path.dirname(__file__), 'static')

run_strong_bp = Blueprint(
    'run_strong',
    __name__,
    template_folder=template_dir,
    static_folder=static_dir,
    url_prefix='/strong'
)

def _success(data=None, status_code=200):
    """Creates a standardized success JSON response."""
    response = {"status": "success"}
    if data is not None:
        response["data"] = data
    return jsonify(response), status_code

def _error(message, status_code):
    """Creates a standardized error JSON response."""
    return jsonify({"status": "error", "message": message}), status_code

def init_run_strong_blueprint(config):
    """Initialize services for RunStrong blueprint."""
    runstrong_service = RunStrongService(config.DB_PATH)
    register_routes(runstrong_service)
    return run_strong_bp

def register_routes(runstrong_service):
    """Register all RunStrong routes."""

    # --- Main Pages (HTML Rendering) ---
    @run_strong_bp.route('/')
    @run_strong_bp.route('/runstrong')
    def runstrong():
        """Display RunStrong home page."""
        return render_template('runstrong_home.html')

    @run_strong_bp.route('/api/exercises', methods=['GET'])
    def get_exercises():
        """API: Get all exercises for the planner."""
        try:
            exercises = runstrong_service.get_exercises()
            return _success(exercises)
        except Exception as e:
            logger.error(f"API error getting exercises: {e}", exc_info=True)
            return _error('Failed to load exercises.', 500)

    @run_strong_bp.route('/exercise_library')
    def exercise_library():
        """Display the exercise library page."""
        try:
            exercises = runstrong_service.get_exercises()
            return render_template('exercise_library.html', exercises=exercises)
        except Exception as e:
            logger.error(f"Error loading exercise library: {e}")
            return _error("Could not load exercise library.", 500)

    @run_strong_bp.route('/journal')
    def journal():
        """Display the workout journal page and the form for new entries."""
        try:
            sessions = runstrong_service.get_workout_journal()
            exercises = runstrong_service.get_exercises() # Fetch exercises for the form
            return render_template('journal.html', sessions=sessions, exercises=exercises)
        except Exception as e:
            logger.error(f"Error loading workout journal: {e}")
            # This should render an error page or return a JSON error
            return "Could not load workout journal.", 500

    # --- NEW API ROUTE ---
    @run_strong_bp.route('/api/journal/log', methods=['POST'])
    def log_workout_entry():
        """API endpoint to log a new workout session."""
        try:
            workout_data = request.get_json()
            if not workout_data or 'sets' not in workout_data or not workout_data['sets']:
                return _error('Invalid workout data provided.', 400)
            
            session_id = runstrong_service.log_new_workout(workout_data)
            return _success({'session_id': session_id}, status_code=201)

        except BadRequest:
            return _error('Invalid JSON format.', 400)
        except Exception as e:
            logger.error(f"API error logging workout: {e}", exc_info=True)
            return _error('Failed to log workout.', 500)

    @run_strong_bp.route('/fatigue_dashboard')
    def fatigue_dashboard():
        """Display the enhanced fatigue dashboard page."""
        try:
            # This one service call now does all the heavy lifting
            fatigue_data = runstrong_service.get_fatigue_dashboard_data()
            return render_template('fatigue_dashboard.html', data=fatigue_data)
        except Exception as e:
            logger.error(f"Error loading fatigue dashboard: {e}", exc_info=True)
            return "Could not load fatigue dashboard.", 500

    @run_strong_bp.route('/goals')
    def goals():
        """Display the goals dashboard page."""
        try:
            goals_data = runstrong_service.get_goals_with_progress()
            return render_template('goals.html', goals=goals_data)
        except Exception as e:
            logger.error(f"Error loading goals dashboard: {e}")
            return _error("Could not load goals dashboard.", 500)