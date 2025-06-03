# app.py
"""
Main Flask application for running data analysis, biomechanics, and strength training application.
Segmented into 3 parts:
RunningHub: Primary Activity and Analysis Review Center
RunnerVision: Biomechanic Analysis
RunStrong: Strength Training Support

This module serves as the entry point for the web application, handling routing
and request/response flow while delegating business logic to service layers.
"""

import logging
import uuid

from flask import Flask, render_template, request, redirect, jsonify, abort, sessions
from werkzeug.exceptions import BadRequest, NotFound

# Application modules
from config import Config
from dash_dashboard_app.layout import create_dash_dashboard_app
from dash_app import create_dash_app
from density_dash import create_density_dash
from services.activity_service import ActivityService
from services.runstrong_service import RunStrongService
from services.runnervision_service import RunnerVisionService
from services.coach_g_service import CoachGService
from services.query_service import QueryService
from services.statistics_service import StatisticsService
from services.trophy_service import TrophyService
from utils import exception_utils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FlaskAppFactory:
    """Factory class for creating and configuring Flask application."""
    
    @staticmethod
    def create_app(config: Config = None) -> Flask:
        """Create and configure Flask application with all extensions."""
        app = Flask(__name__)
        
        # Load configuration
        if config is None:
            config = Config()
        app.config.from_object(config)
        app.secret_key = config.FLASK_SECRET_KEY
        
        # Initialize Dash applications
        FlaskAppFactory._initialize_dash_apps(app, config)
        
        # Register blueprints/routes
        FlaskAppFactory._register_routes(app, config)
        
        # Register error handlers
        FlaskAppFactory._register_error_handlers(app)
        
        return app
    
    @staticmethod
    def _initialize_dash_apps(app: Flask, config: Config) -> None:
        """Initialize all Dash applications."""
        try:
            create_dash_app(app)
            create_density_dash(app, db_path=config.DB_PATH)
            create_dash_dashboard_app(app, db_path=config.DB_PATH)
            logger.info("Dash applications initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Dash apps: {e}")
            raise
    
    @staticmethod
    def _register_routes(app: Flask, config: Config) -> None:
        """Register all application routes."""
        # Initialize services
        activity_service = ActivityService(config.DB_PATH)
        runstrong_service = RunStrongService(config.DB_PATH)
        runnervision_service = RunnerVisionService(config)
        coach_g_service = CoachGService(config)
        query_service = QueryService(config.DB_PATH)
        statistics_service = StatisticsService(config.DB_PATH)
        trophy_service = TrophyService(config.DB_PATH)
        
        # Register route handlers
        HomeRoutes.register(app, activity_service)
        ActivityRoutes.register(app, activity_service)
        RunnerVisionRoutes.register(app, runnervision_service)
        RunStrongRoutes.register(app, runstrong_service)
        CoachGRoutes.register(app, coach_g_service)
        QueryRoutes.register(app, query_service)
        StatisticsRoutes.register(app, statistics_service)
        TrophyRoutes.register(app, trophy_service)
        DashRedirectRoutes.register(app)
    
    @staticmethod
    def _register_error_handlers(app: Flask) -> None:
        """Register global error handlers."""
        
        @app.errorhandler(exception_utils.DatabaseError)
        def handle_database_error(e):
            logger.error(f"Database error: {e}")
            return render_template('error.html', 
                                 error="Database error occurred"), 500
        
        @app.errorhandler(404)
        def handle_not_found(e):
            return render_template('error.html', 
                                 error="Page not found"), 404
        
        @app.errorhandler(500)
        def handle_server_error(e):
            logger.error(f"Server error: {e}")
            return render_template('error.html', 
                                 error="Internal server error"), 500


class HomeRoutes:
    """Home page route handlers."""
    
    @staticmethod
    def register(app: Flask, activity_service: ActivityService) -> None:
        """Register home routes."""
        
        @app.route("/")
        def home():
            """Render home page with latest activity."""
            try:
                latest_activity_id = activity_service.get_latest_activity_id()
                return render_template("home.html", activity=latest_activity_id)
            except Exception as e:
                logger.error(f"Error loading home page: {e}")
                return render_template("home.html", activity=None)


class ActivityRoutes:
    """Activity-related route handlers."""
    
    @staticmethod
    def register(app: Flask, activity_service: ActivityService) -> None:
        """Register activity routes."""
        
        @app.route("/activity/")
        def activity():
            """Display detailed activity information."""
            activity_id = request.args.get("id", type=int)
            units = request.args.get('units', 'mi')
            
            if not activity_id:
                abort(400, "Activity ID is required")
            
            try:
                activity_data = activity_service.get_formatted_activity_page_details(activity_id, units)
                if not activity_data:
                    abort(404, "Activity not found")
               
                return render_template("activity.html", 
                                     activity=activity_data, units=units)
            except exception_utils.DatabaseError as e:
                logger.error(f"Database error getting activity {activity_id}: {e}")
                abort(500, "Error retrieving activity data")


class RunnerVisionRoutes:
    """RunnerVision biomechanics route handlers."""
    
    @staticmethod
    def register(app: Flask, runnervision_service: RunnerVisionService) -> None:
        """Register RunnerVision routes."""
        
        @app.route('/runnervision')
        def runnervision():
            """Display RunnerVision analysis results."""
            try:
                analysis_data = runnervision_service.get_latest_analysis()
                return render_template('runnervision.html', **analysis_data)
            except Exception as e:
                logger.error(f"Error loading RunnerVision page: {e}")
                return render_template('runnervision.html')
        
        @app.route('/reports/<path:filename>')
        def serve_report(filename):
            """Serve RunnerVision report files."""
            return runnervision_service.serve_report(filename)
        
        @app.route('/videos/<path:filename>')
        def serve_video(filename):
            """Serve RunnerVision video files."""
            return runnervision_service.serve_video(filename)
        
        @app.route('/upload', methods=['POST'])
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
        
        @app.route('/run_biomechanic_analysis', methods=['POST'])
        def run_analysis():
            """Execute biomechanics analysis."""
            try:
                result = runnervision_service.run_analysis()
                return jsonify(result)
            except Exception as e:
                logger.error(f"Analysis error: {e}")
                return jsonify({"error": "Analysis failed"}), 500


class RunStrongRoutes:
    """RunStrong strength training route handlers."""
    
    @staticmethod
    def register(app: Flask, runstrong_service: RunStrongService) -> None:
        """Register RunStrong routes."""
        
        @app.route('/runstrong')
        def runstrong():
            """Display RunStrong home page."""
            return render_template('runstrong_home.html')
        
        @app.route('/exercise_library')
        def exercise_library():
            """Display RunStrong home page."""
            return render_template('runstrong_exercise_library.html')
        
        @app.route('/runstrong/exercises')
        def exercises():
            """API endpoint for exercise data."""
            try:
                exercises = runstrong_service.get_exercises()
                return jsonify({'exercises': exercises})
            except Exception as e:
                logger.error(f"Error getting exercises: {e}")
                return jsonify({'error': 'Failed to load exercises'}), 500
            
        @app.route('/runstrong/exercises/add', methods=['GET', 'POST'])
        def add_exercise():
            if request.method == 'POST':
                try:
                    data = request.get_json()
                    runstrong_service.add_exercise(data)
                    return jsonify({'message': 'Exercise added successfully!'})
                except Exception as e:
                    logger.error(f"Error adding exercise: {e}")
                    return jsonify({'error': 'Failed to add exercise.'}), 500
            return render_template('add_exercise.html')
        
        @app.route('/runstrong/save-routine', methods=['POST'])
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
        
        @app.route('/runstrong/load-routines')
        def load_routines():
            """Load all workout routines."""
            try:
                routines = runstrong_service.get_routines()
                return jsonify({'routines': routines})
            except Exception as e:
                logger.error(f"Error loading routines: {e}")
                return jsonify({'error': 'Failed to load routines'}), 500
        
        @app.route('/runstrong/load-routine/<int:routine_id>')
        def load_routine(routine_id: int):
            """Load specific workout routine."""
            try:
                exercises = runstrong_service.get_routine_exercises(routine_id)
                return jsonify({'exercises': exercises})
            except Exception as e:
                logger.error(f"Error loading routine {routine_id}: {e}")
                return jsonify({'error': 'Failed to load routine'}), 500
        
        # Template routes
        @app.route('/runstrong/planner')
        def planner():
            return render_template('planner.html')
        
        @app.route('/runstrong/journal')
        def journal():
            return render_template('journal.html')
        
        @app.route('/runstrong/dashboard')
        def dashboard():
            return render_template('dashboard.html')
        
        # late night C team
        # API Routes for Planner
        @app.route('/api/exercises', methods=['GET'])
        def get_exercises():
            """Get all exercises for the planner"""
            try:
                exercises = runstrong_service.get_exercises()
                return jsonify(exercises)
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @app.route('/api/routines', methods=['GET'])
        def get_routines():
            """Get all workout routines"""
            try:
                routines = runstrong_service.get_all_routines()
                return jsonify(routines)
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @app.route('/api/routines', methods=['POST'])
        def create_routine():
            """Create a new workout routine with exercises"""
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

        @app.route('/api/routines/<int:routine_id>/exercises', methods=['GET'])
        def get_routine_exercises(routine_id):
            """Get a specific routine with its exercises"""
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
        @app.route('/api/workout-performance', methods=['POST'])
        def save_workout_performance():
            """Save workout performance data"""
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

        @app.route('/api/workout-performance/<int:routine_id>', methods=['GET'])
        def get_workout_history(routine_id):
            """Get workout history for a specific routine"""
            try:
                history = runstrong_service.get_workout_history(routine_id)
                return jsonify(history)
            except Exception as e:
                return jsonify({'error': str(e)}), 500

class CoachGRoutes:
    """Routes for Coach G functionality."""
    
    @staticmethod
    def register(app: Flask, coach_g_service: CoachGService) -> None:
        """Register Coach G routes."""

        @app.route("/coach-g")
        def coach_g():
            """Render Coach G chat interface."""
            return render_template("coach_g.html")
        
        @app.route('/api/coach-g/', methods=['POST'])
        def coach_g_chat():
            """Handle Coach G chat interactions."""
            try:
                # Validate request
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'No data provided'}), 400
                
                user_message = data.get('message', '').strip()
                if not user_message:
                    return jsonify({'error': 'Message is required'}), 400
                
                personality_selection = data.get('personality', 'motivational')
                
                # Personality mapping
                personality_prompts = {
                    'motivational': "an energetic, encouraging running coach who inspires confidence",
                    'analytical': "a data-driven coach who focuses on metrics and structured training",
                    'supportive': "a patient, understanding coach who prioritizes runner wellbeing",
                    'challenging': "a tough but fair coach who pushes runners to exceed their limits",
                    'scientific': "an evidence-based coach who explains the science behind training",
                    'toxic' : "a foul mouthed, brash, rude, who SCREAMS and says FUCK but gets results"
                }
                
                personality = personality_prompts.get(personality_selection, personality_prompts['motivational'])
                
                # Get or create session ID
                session_id = request.cookies.get('session_id') or str(uuid.uuid4())
                
                # Handle predefined queries
                if user_message.lower() in ['whats my training status for today?', 'training status']:
                    coach_reply = coach_g_service.daily_training_summary()
                else:
                    # Generate contextual response
                    coach_reply = coach_g_service.general_reply(session_id, user_message, personality)
                
                response = jsonify({'response': coach_reply})
                
                # Set session cookie if new
                if 'session_id' not in request.cookies:
                    response.set_cookie('session_id', session_id, max_age=86400)  # 24 hours
                
                return response
                
            except Exception as e:
                logger.error(f"Error in coach_g_chat: {e}")
                return jsonify({'error': 'Internal server error'}), 500


class QueryRoutes:
    """Database query route handlers."""
    
    @staticmethod
    def register(app: Flask, query_service: QueryService) -> None:
        """Register query routes."""
        
        @app.route('/query/', methods=['GET', 'POST'])
        def query():
            """Handle database queries."""
            if request.method == 'GET':
                return render_template('query.html')
            
            sql_query = request.form.get('sql_query', '').strip()
            param_input = request.form.get('params', '{}').strip()
            
            try:
                result = query_service.execute_query(sql_query, param_input)
                return render_template('query.html', **result, sql_query=sql_query)
            
            except exception_utils.DatabaseError as e:
                logger.error(f"Query execution error: {e}")
                return render_template('query.html', 
                                     error=str(e), sql_query=sql_query)
        @app.route('/ai_query', methods=['POST'])
        def ai_query():
            user_question = request.form.get('user_question', '')
            columns = []
            rows = []
            error = None
            sql_query = ''

            try:
                query_prompt_generator = lm_utils.generate_sql_from_natural_language(user_question)
                sql_query = lm_utils.extract_sql_query(query_prompt_generator)
                # For now, use empty params for simplicity
                with db_utils.get_db_connection(Config.DB_PATH) as conn:
                    cursor = conn.cursor()
                    cursor.execute(sql_query)
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()

            except Exception as e:
                error = f"Failed to process question: {str(e)}, Query Generated: {sql_query}"

            return render_template('query.html', columns=columns, rows=rows, error=error, request=request, sql_query = sql_query)


class StatisticsRoutes:
    """Statistics page route handlers."""
    
    @staticmethod
    def register(app: Flask, statistics_service: StatisticsService) -> None:
        """Register statistics routes."""
        
        @app.route('/statistics/', methods=['GET'])
        def statistics():
            """Display running statistics."""
            period = request.args.get('period', 'week')
            units = request.args.get('units', 'mi')
            
            try:
                stats_data = statistics_service.get_statistics(period, units)
                return render_template('statistics.html', **stats_data)
            
            except Exception as e:
                logger.error(f"Error loading statistics: {e}")
                return render_template('statistics.html', 
                                     error="Failed to load statistics")


class TrophyRoutes:
    """Trophy room route handlers."""
    
    @staticmethod
    def register(app: Flask, trophy_service: TrophyService) -> None:
        """Register trophy room routes."""
        
        @app.route('/trophy_room/')
        def trophy_room():
            """Display personal records and achievements."""
            units = request.args.get('units', 'mi')
            
            try:
                records = trophy_service.get_personal_records(units)
                return render_template("trophy_room.html", 
                                     personal_records=records, units=units)
            
            except Exception as e:
                logger.error(f"Error loading trophy room: {e}")
                return render_template("trophy_room.html", 
                                     personal_records=[], units=units)


class DashRedirectRoutes:
    """Dash application redirect handlers."""
    
    @staticmethod
    def register(app: Flask) -> None:
        """Register Dash redirect routes."""
        
        @app.route("/map/")
        def dashredirect():
            activity_id = request.args.get('id')
            return redirect(f"/map/?id={activity_id}" if activity_id else "/map/")
        
        @app.route('/dashboard-redirect/')
        def dashboard_redirect():
            return redirect('/dashboard/')
        
        @app.route("/density/")
        def density_redirect():
            period = request.args.get('period', 'week')
            start_date = request.args.get("start_date", "2024-01-01")
            
            # Determine zoom level based on referrer
            zoom = 12 if request.referrer and "stat" in request.referrer else 10
            
            return redirect(f"/density_dash/?start_date={start_date}&zoom={zoom}")


# Application factory
def create_app(config: Config = None) -> Flask:
    """Create Flask application instance."""
    return FlaskAppFactory.create_app(config)


# Entry point for development server
if __name__ == '__main__':
    app = create_app()
    app.run(
        debug=True,
        port=5555,
        host='0.0.0.0'  # Allow external connections in development
    )