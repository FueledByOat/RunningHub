# app.py
"""
Main Flask application for running data analysis, biomechanics, and strength training application.
Segmented into 3 pillars:
RunningHub: Primary Activity and Analysis Review Center
RunnerVision: Biomechanic Analysis
RunStrong: Strength Training Support

This module serves as the entry point for the web application, handling routing
and request/response flow while delegating business logic to service layers.
"""

import logging
from flask import Flask, render_template, request, redirect

# Configuration and utilities
from config import Config
from dash_dashboard_app.layout import create_dash_dashboard_app
from dash_app import create_dash_app
from density_dash import create_density_dash
from utils import exception_utils

# Blueprints
from blueprints.running_hub.routes import init_running_hub_blueprint
from blueprints.runner_vision.routes import init_runner_vision_blueprint
from blueprints.run_strong.routes import init_run_strong_blueprint
from blueprints.coach_g.routes import init_coach_g_blueprint

# Services for legacy routes
from services.activity_service import ActivityService

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
        
        # Register blueprints
        FlaskAppFactory._register_blueprints(app, config)
        
        # Register legacy routes
        FlaskAppFactory._register_legacy_routes(app, config)
        
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
    def _register_blueprints(app: Flask, config: Config) -> None:
        """Register all application blueprints."""
        # Initialize and register blueprints
        running_hub_bp = init_running_hub_blueprint(config)
        runner_vision_bp = init_runner_vision_blueprint(config)
        run_strong_bp = init_run_strong_blueprint(config)
        coach_g_bp = init_coach_g_blueprint(config)
        
        app.register_blueprint(running_hub_bp)
        app.register_blueprint(runner_vision_bp)
        app.register_blueprint(run_strong_bp)
        app.register_blueprint(coach_g_bp)
        
        logger.info("All blueprints registered successfully")
    
    @staticmethod
    def _register_legacy_routes(app: Flask, config: Config) -> None:
        """Register legacy routes for backward compatibility."""
        # Root home route redirects to RunningHub
        @app.route("/")
        def home():
            """Redirect to RunningHub home."""
            return redirect("/hub/")
        
        # Legacy redirects to maintain backward compatibility
        @app.route("/activity/")
        def activity():
            """Redirect to RunningHub activity page."""
            activity_id = request.args.get('id')
            units = request.args.get('units', 'mi')
            return redirect(f"/hub/activity/?id={activity_id}&units={units}")
        
        @app.route("/runnervision")
        def runnervision():
            """Redirect to RunnerVision page."""
            return redirect("/vision/")
        
        @app.route("/runstrong")
        def runstrong():
            """Redirect to RunStrong page."""
            return redirect("/strong/")
        
        @app.route("/coach-g")
        def coach_g():
            """Redirect to Coach G page."""
            return redirect("/coach/")
        
        # Dash redirect routes
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