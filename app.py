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

from flask import Flask, render_template, request, redirect, jsonify, abort
from werkzeug.exceptions import BadRequest, NotFound

# Application modules
from config import Config
from dash_dashboard_app.layout import create_dash_dashboard_app
from dash_app import create_dash_app
from density_dash import create_density_dash
from services.activity_service import ActivityService
from services.runstrong_service import RunStrongService
from services.runnervision_service import RunnerVisionService
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
        runstrong_service = RunStrongService(config.DB_PATH_RUNSTRONG)
        runnervision_service = RunnerVisionService(config)
        query_service = QueryService(config.DB_PATH)
        statistics_service = StatisticsService(config.DB_PATH)
        trophy_service = TrophyService(config.DB_PATH)
        
        # Register route handlers
        HomeRoutes.register(app, activity_service)
        ActivityRoutes.register(app, activity_service)
        RunnerVisionRoutes.register(app, runnervision_service)
        RunStrongRoutes.register(app, runstrong_service)
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
                activity_data = activity_service.get_activity_details(activity_id, units)
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
        
        @app.route('/runstrong/exercises')
        def exercises():
            """API endpoint for exercise data."""
            try:
                exercises = runstrong_service.get_exercises()
                return jsonify({'exercises': exercises})
            except Exception as e:
                logger.error(f"Error getting exercises: {e}")
                return jsonify({'error': 'Failed to load exercises'}), 500
        
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