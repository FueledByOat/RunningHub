# blueprints/run_strong/routes.py
"""
RunStrong Blueprint - Strength training functionality.
Includes: Exercise library, routine planning, workout journaling, and dashboard.
"""

import logging
import os
from flask import Blueprint, render_template, request, jsonify, render_template_string
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

def init_run_strong_blueprint(config):
    """Initialize services for RunStrong blueprint."""
    runstrong_service = RunStrongService(config.DB_PATH)
    register_routes(runstrong_service)
    return run_strong_bp

def register_routes(runstrong_service):
    """Register all RunStrong routes."""
    
    # Main Pages
    @run_strong_bp.route('/')
    @run_strong_bp.route('/runstrong')
    def runstrong():
        """Display RunStrong home page."""
        return render_template('runstrong_home.html')
    
    @run_strong_bp.route('/exercise-library')
    def exercise_library():
        """Display exercise library."""
        return render_template('runstrong_exercise_library.html')
    
    @run_strong_bp.route('/planner')
    def planner():
        """Display workout planner."""
        return render_template('planner.html')
    
    @run_strong_bp.route('/journal')
    def journal():
        """Display workout journal."""
        return render_template('journal.html')
    
    @run_strong_bp.route('/progress-dashboard')
    def progress_dashboard():
        """Display progress and fatigure dashboard."""
        data = runstrong_service.get_fatigue_dashboard_data()
        return render_template('progress_dashboard.html', fatigue_data=data)
        
    # Exercise Management Routes
    @run_strong_bp.route('/exercises')
    def exercises():
        """API endpoint for exercise data."""
        try:
            exercises = runstrong_service.get_exercises()
            return jsonify({'exercises': exercises})
        except Exception as e:
            logger.error(f"Error getting exercises: {e}")
            return jsonify({'error': 'Failed to load exercises'}), 500
    
    @run_strong_bp.route('/exercises/add', methods=['GET', 'POST'])
    def add_exercise():
        """Add new exercise."""
        if request.method == 'POST':
            try:
                data = request.get_json()
                runstrong_service.add_exercise(data)
                return jsonify({'message': 'Exercise added successfully!'})
            except Exception as e:
                logger.error(f"Error adding exercise: {e}")
                return jsonify({'error': 'Failed to add exercise.'}), 500
        return render_template('add_exercise.html')
    
    # Routine Management Routes
    @run_strong_bp.route('/save-routine', methods=['POST'])
    def save_routine():
        """Save workout routine."""
        try:
            data = request.get_json()
            if not data or 'name' not in data or 'routine' not in data:
                raise BadRequest("Invalid routine data")
            
            runstrong_service.save_routine(data['name'], data['routine'])
            return jsonify({'status': 'success'})
        except Exception as e:
            logger.error(f"Error saving routine: {e}")
            return jsonify({'error': 'Failed to save routine'}), 500
    
    @run_strong_bp.route('/load-routines')
    def load_routines():
        """Load all workout routines."""
        try:
            routines = runstrong_service.get_routines()
            return jsonify({'routines': routines})
        except Exception as e:
            logger.error(f"Error loading routines: {e}")
            return jsonify({'error': 'Failed to load routines'}), 500
    
    @run_strong_bp.route('/load-routine/<int:routine_id>')
    def load_routine(routine_id: int):
        """Load specific workout routine."""
        try:
            exercises = runstrong_service.get_routine_exercises(routine_id)
            return jsonify({'exercises': exercises})
        except Exception as e:
            logger.error(f"Error loading routine {routine_id}: {e}")
            return jsonify({'error': 'Failed to load routine'}), 500
    
    # API Routes for Planner
    @run_strong_bp.route('/api/exercises', methods=['GET'])
    def get_exercises():
        """Get all exercises for the planner."""
        try:
            exercises = runstrong_service.get_exercises()
            return jsonify(exercises)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @run_strong_bp.route('/api/routines', methods=['GET'])
    def get_routines():
        """Get all workout routines."""
        try:
            routines = runstrong_service.get_all_routines()
            return jsonify(routines)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @run_strong_bp.route('/api/routines', methods=['POST'])
    def create_routine():
        """Create a new workout routine with exercises."""
        try:
            data = request.get_json()
            routine_name = data.get('name')
            exercises = data.get('exercises', [])
            
            if not routine_name:
                return jsonify({'error': 'Routine name is required'}), 400
            
            # Create the routine
            routine_id = runstrong_service.create_routine(routine_name)
            
            # Add exercises to the routine
            for exercise_data in exercises:
                runstrong_service.add_exercise_to_routine(
                    routine_id=routine_id,
                    exercise_id=exercise_data['exercise']['id'],
                    sets=exercise_data['sets'],
                    reps=exercise_data['reps'],
                    load_lbs=exercise_data['load_lbs'],
                    order_index=exercise_data['order_index'],
                    notes=exercise_data.get('notes', '')
                )
            
            return jsonify({'message': 'Routine created successfully', 'routine_id': routine_id})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @run_strong_bp.route('/api/routines/<int:routine_id>/exercises', methods=['GET'])
    def get_routine_exercises_api(routine_id):
        """Get a specific routine with its exercises."""
        try:
            routine = runstrong_service.get_routine_by_id(routine_id)
            if not routine:
                return jsonify({'error': 'Routine not found'}), 404
            
            exercises = runstrong_service.get_routine_exercises(routine_id)
            
            return jsonify({
                'routine': routine,
                'exercises': exercises
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # API Routes for Journal
    @run_strong_bp.route('/api/workout-performance', methods=['POST'])
    def save_workout_performance():
        """Save workout performance data."""
        try:
            data = request.get_json()
            routine_id = data.get('routine_id')
            workout_date = data.get('workout_date')
            exercises = data.get('exercises', [])
            
            if not routine_id or not workout_date:
                return jsonify({'error': 'Routine ID and workout date are required'}), 400
            
            # Save performance data for each exercise
            for exercise_data in exercises:
                runstrong_service.save_workout_performance(
                    routine_id=exercise_data['routine_id'],
                    exercise_id=exercise_data['exercise_id'],
                    workout_date=exercise_data['workout_date'],
                    planned_sets=exercise_data['planned_sets'],
                    actual_sets=exercise_data['actual_sets'],
                    planned_reps=exercise_data['planned_reps'],
                    actual_reps=exercise_data['actual_reps'],
                    planned_load_lbs=exercise_data['planned_load_lbs'],
                    actual_load_lbs=exercise_data['actual_load_lbs'],
                    notes=exercise_data.get('notes', ''),
                    completion_status=exercise_data.get('completion_status', 'completed')
                )
            
            return jsonify({'message': 'Workout performance saved successfully'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @run_strong_bp.route('/api/workout-performance/<int:routine_id>', methods=['GET'])
    def get_workout_history(routine_id):
        """Get workout history for a specific routine."""
        try:
            history = runstrong_service.get_workout_history(routine_id)
            return jsonify(history)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    @run_strong_bp.route('/api/routines/<int:routine_id>', methods=['PUT'])
    def update_routine(routine_id):
        """Update an existing workout routine."""
        try:
            data = request.get_json()
            routine_name = data.get('name')
            exercises = data.get('exercises', [])
            
            if not routine_name:
                return jsonify({'error': 'Routine name is required'}), 400
            
            # Update the routine name
            runstrong_service.update_routine_name(routine_id, routine_name)
            
            # Remove existing exercises for this routine
            runstrong_service.clear_routine_exercises(routine_id)
            
            # Add updated exercises to the routine
            for exercise_data in exercises:
                runstrong_service.add_exercise_to_routine(
                    routine_id=routine_id,
                    exercise_id=exercise_data['exercise']['id'],
                    sets=exercise_data['sets'],
                    reps=exercise_data['reps'],
                    load_lbs=exercise_data['load_lbs'],
                    order_index=exercise_data['order_index'],
                    notes=exercise_data.get('notes', '')
                )
            
            return jsonify({'message': 'Routine updated successfully'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @run_strong_bp.route('/api/exercise-max-loads', methods=['GET'])
    def get_exercise_max_loads():
        """Get maximum load for each exercise from workout history."""
        try:
            max_loads = runstrong_service.get_exercise_max_loads()
            return jsonify(max_loads)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @run_strong_bp.route('/api/routines/<int:routine_id>', methods=['DELETE'])
    def delete_routine(routine_id):
        """Delete a workout routine."""
        try:
            runstrong_service.delete_routine(routine_id)
            return jsonify({'message': 'Routine deleted successfully'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @run_strong_bp.route('/api/fatigue-data')
    def get_fatigue_data():
        """API endpoint to get current fatigue data with optional filtering"""
        try:
            # Get muscle group filter from query parameters
            muscle_group_filter = request.args.get('muscle_group', None)
            
            # Validate filter
            valid_filters = ['all', 'upper body', 'lower body', 'core']
            if muscle_group_filter and muscle_group_filter.lower() not in valid_filters:
                muscle_group_filter = None
            
            data = runstrong_service.get_fatigue_dashboard_data(muscle_group_filter)
            return jsonify(data)
        except Exception as e:
            logger.error(f"API error: {e}")
            return jsonify({'error': 'Failed to fetch data'}), 500

    @run_strong_bp.route('/api/muscle-groups')
    def get_available_muscle_groups():
        """API endpoint to get available muscle group filters"""
        try:
            return jsonify({
                'filters': ['All', 'Upper body', 'Lower body', 'Core'],
                'default': 'All'
            })
        except Exception as e:
            logger.error(f"API error: {e}")
            return jsonify({'error': 'Failed to fetch muscle groups'}), 500
    
    @run_strong_bp.route('/api/update-fatigue')
    def update_fatigue():
        """API endpoint to trigger fatigue update"""
        try:
            data = runstrong_service.run_daily_update()
            return jsonify({'status': 'success', 'data': data})
        except Exception as e:
            logger.error(f"Update error: {e}")
            return jsonify({'error': 'Failed to update data'}), 500