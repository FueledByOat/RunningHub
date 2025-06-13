# blueprints/coach_g/routes.py
"""
Coach G Blueprint - AI coaching functionality.
Includes: Chat interface and contextual coaching responses.
"""

import logging
import os
import uuid
from flask import Blueprint, render_template, request, jsonify
from services.coach_g_service import CoachGService

logger = logging.getLogger(__name__)

# Create blueprint
template_dir = os.path.join(os.path.dirname(__file__), 'templates')
static_dir = os.path.join(os.path.dirname(__file__), 'static')

coach_g_bp = Blueprint(
    'coach_g',
    __name__,
    template_folder=template_dir,
    static_folder=static_dir,
    url_prefix='/coach'
)

def init_coach_g_blueprint(config):
    """Initialize services for Coach G blueprint."""
    coach_g_service = CoachGService(config)
    register_routes(coach_g_service)
    return coach_g_bp

def register_routes(coach_g_service):
    """Register all Coach G routes."""
    
    @coach_g_bp.route("/")
    @coach_g_bp.route("/coach-g")
    def coach_g():
        """Render Coach G chat interface."""
        return render_template("coach_g.html")
    
    @coach_g_bp.route('/api/chat/', methods=['POST'])
    def coach_g_chat():
        """Handle Coach G chat interactions."""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            user_message = data.get('message', '').strip()
            if not user_message:
                return jsonify({'error': 'Message is required'}), 400


            personality_selection = data.get('personality', 'motivational')
            session_id = request.cookies.get('session_id') or str(uuid.uuid4())
                
            if data.get('is_quick') == True:
                coach_reply = coach_g_service.handle_quick_query(session_id, user_message, personality_selection, data.get('quick_question_topic'))
                                                                                 
            # Simplified call to the service layer
            else:
                coach_reply = coach_g_service.handle_user_query(session_id, user_message, personality_selection)
            
            response = jsonify({'response': coach_reply})
            
            if 'session_id' not in request.cookies:
                response.set_cookie('session_id', session_id, max_age=86400)  # 24 hours
            
            return response
            
        except Exception as e:
            logger.error(f"Error in coach_g_chat: {e}")
            return jsonify({'error': 'Internal server error'}), 500