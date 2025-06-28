# blueprints/running_hub/routes.py
"""
RunningHub Blueprint - Core running data analysis and tracking functionality.
Includes: Home, Activity Details, Statistics, Query Tools, and Trophy Room.
"""

import logging
import os
import uuid
import json
from flask import Blueprint, render_template, request, redirect, abort, jsonify, url_for 
from werkzeug.exceptions import BadRequest, NotFound

from services.activity_service import ActivityService
from services.query_service import QueryService
from services.statistics_service import StatisticsService
from services.trophy_service import TrophyService
from services.motivation_service import MotivationService
from utils import exception_utils
from utils.db import db_utils

logger = logging.getLogger(__name__)

BLUEPRINT_PATH = os.path.dirname(__file__)
USER_PROFILE_PATH = os.path.join(BLUEPRINT_PATH, 'data', 'user_profile_data.json')

# Create blueprint with static folder for pillar-specific assets
template_dir = os.path.join(os.path.dirname(__file__), 'templates')
static_dir = os.path.join(os.path.dirname(__file__), 'static')

running_hub_bp = Blueprint(
    'running_hub', 
    __name__,
    template_folder=template_dir,
    static_folder=static_dir,
    url_prefix='/hub'
)

def load_user_profile_data():
    """Loads user profile data from the central JSON file."""
    with open(USER_PROFILE_PATH, 'r') as f:
        return json.load(f)

def init_running_hub_blueprint(config):
    """Initialize services for RunningHub blueprint."""
    activity_service = ActivityService(config.DB_PATH)
    query_service = QueryService(config.DB_PATH)
    statistics_service = StatisticsService(config.DB_PATH)
    trophy_service = TrophyService(config.DB_PATH)
    motivation_service = MotivationService(config)
    
    register_routes(activity_service, query_service, statistics_service, trophy_service, motivation_service)
    return running_hub_bp

def register_routes(activity_service, query_service, statistics_service, trophy_service, motivation_service):
    """Register all RunningHub routes."""
    
    # Home/Dashboard Routes
    @running_hub_bp.route("/")
    @running_hub_bp.route("/home")
    def home():
        """RunningHub home page with latest activity."""
        try:
            latest_activity_id = activity_service.get_latest_activity_id()
            return render_template("home.html", activity=latest_activity_id)
        except Exception as e:
            logger.error(f"Error loading RunningHub home: {e}")
            return render_template("home.html", activity=None)
    
    # Activity Routes
    @running_hub_bp.route("/activity/")
    def activity():
        """Display detailed activity information for specific activity ID."""
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
    
    # Query Routes
    @running_hub_bp.route('/query/', methods=['GET', 'POST'])
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
    
    @running_hub_bp.route('/ai_query', methods=['POST'])
    def ai_query():
        """AI-powered natural language query interface."""
        user_question = request.form.get('user_question', '')
        columns = []
        rows = []
        error = None
        sql_query = ''

        try:
            from utils import lm_utils
            from config import Config
            
            query_prompt_generator = lm_utils.generate_sql_from_natural_language(user_question)
            sql_query = lm_utils.extract_sql_query(query_prompt_generator)
            
            with db_utils.get_db_connection(Config.DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute(sql_query)
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()

        except Exception as e:
            error = f"Failed to process question: {str(e)}, Query Generated: {sql_query}"

        return render_template('query.html', 
                             columns=columns, rows=rows, error=error, 
                             request=request, sql_query=sql_query)
    
    # Statistics Routes
    @running_hub_bp.route('/statistics/', methods=['GET'])
    def statistics():
        """Display time period aggregated running statistics."""
        period = request.args.get('period', 'week')
        units = request.args.get('units', 'mi')
        
        try:
            stats_data = statistics_service.get_statistics(period, units)
            return render_template('statistics.html', **stats_data)
        
        except Exception as e:
            logger.error(f"Error loading statistics: {e}")
            return render_template('statistics.html', 
                                 error="Failed to load statistics")
    
    # Trophy Room Routes
    @running_hub_bp.route('/trophy_room/')
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
        
    @running_hub_bp.route("/motivation")
    def motivation():
        """Motivation Page, including upcoming races and inspriational LLM quote generator"""
        try:
            """Renders the main motivation page with data injected."""
            profile_data = load_user_profile_data()

            if 'races' in profile_data:
                for race in profile_data['races']:
                    if 'image' in race:
                        relative_image_path = race['image'].replace('static/', '', 1)

                        # --- THIS IS THE CORRECTED LINE ---
                        # We specify 'running_hub.static' to correctly point to the
                        # static folder defined within this blueprint.
                        race['imageUrl'] = url_for('running_hub.static', filename=relative_image_path)
                        # --- END OF CORRECTION ---
                        
                    else:
                        # Also correct the fallback URL to use the blueprint's static folder
                        race['imageUrl'] = url_for('running_hub.static', filename='images/default_placeholder.png')
            return render_template("motivation.html", profile_data=profile_data)
        
        except Exception as e:
            logger.error(f"Error loading RunningHub motivation: {e}")
            return render_template("motivation.html", profile_data = [])
        
    # Motivation Routes
    @running_hub_bp.route('/api/daily_motivation', methods=['POST'])
    def daily_motivation():
        """API endpoint to generate a daily motivational message based on selected personality."""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            personality = data.get('personality', 'motivational')

            # Use a transient session_id as we may not have a chat session cookie
            session_id = request.cookies.get('session_id', f"motivation-{uuid.uuid4()}")
            
            profile_data = load_user_profile_data()

            message = motivation_service.get_daily_motivational_message(session_id, personality, profile_data)
            
            return jsonify({'response': message})
        except Exception as e:
            logger.error(f"Error generating daily motivation: {e}", exc_info=True)
            return jsonify({'error': 'Failed to generate message due to a server error.'}), 500

    # Skill Tree Routes
    @running_hub_bp.route('/skill_tree/')
    def skill_tree():
        """Progressive skill tree for running achievement and side-quests."""
        
        try:
            return render_template("skill_tree.html")
        
        except Exception as e:
            logger.error(f"Error loading skill tree: {e}")
            return render_template("skill_tree.html")