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
            exercises = runstrong_service.get_exercises_for_library()
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
            sorted_exercises = sorted(exercises, key=lambda x: x['name'])
            return render_template('journal.html', sessions=sessions, exercises=sorted_exercises)
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
        
    @run_strong_bp.route('/max-weights')
    def max_weights():
        """Display exercise max weights page."""
        try:
            exercises = runstrong_service.get_exercise_max_weights()
            return render_template('max_weights.html', exercises=exercises)
        except Exception as e:
            logger.error(f"Error displaying max weights page: {e}")
            return render_template('error.html', error='Failed to load max weights data.')

    @run_strong_bp.route('/api/exercise-max/<int:exercise_id>')
    def get_exercise_max(exercise_id):
        """API: Get maximum weight for a specific exercise."""
        try:
            max_weight = runstrong_service.get_exercise_max_for_goals(exercise_id)
            return _success({"max_weight": max_weight})
        except Exception as e:
            logger.error(f"API error getting exercise max: {e}", exc_info=True)
            return _error('Failed to get exercise max weight.', 500)
        
    # Obviously move to db or remove this

    @run_strong_bp.route('/movement-catalog')
    def movement_catalog():
        """Display movement catalog page."""
        movement = [
  {
    "name": "Dynamic Drills",
    "exercises": [
      {
        "exercise_name": "A-Skip",
        "purpose": "Knee drive & posture alignment",
        "when_to_use": "Pre-speed or tempo sessions",
        "video_url": "https://www.youtube.com/watch?v=9Z3D1gVgYY8"
      },
      {
        "exercise_name": "B-Skip",
        "purpose": "Knee drive with foot extension for stride mechanics",
        "when_to_use": "Pre-speed workouts",
        "video_url": "https://www.youtube.com/watch?v=7DMEy0cqKvw"
      },
      {
        "exercise_name": "High Knees",
        "purpose": "Fast turnover and upright posture",
        "when_to_use": "Pre-speed and threshold workouts",
        "video_url": "https://www.youtube.com/watch?v=8opcQdC-V-U"
      },
      {
        "exercise_name": "Butt Kicks",
        "purpose": "Hamstring activation and stride coordination",
        "when_to_use": "Pre-tempo or drills warmup",
        "video_url": "https://www.youtube.com/watch?v=CeZlih4DDNg"
      },
      {
        "exercise_name": "Ankling",
        "purpose": "Foot contact timing and stiffness",
        "when_to_use": "Before intervals or strides",
        "video_url": None
      },
      {
        "exercise_name": "Straight-Leg Bounds",
        "purpose": "Elasticity and glute stiffness",
        "when_to_use": "Before track work or bounding sessions",
        "video_url": None
      }
    ]
  },
  {
    "name": "Plyometrics",
    "exercises": [
      {
        "exercise_name": "Pogo Hops",
        "purpose": "Ankle stiffness and reactive contact",
        "when_to_use": "Pre-speed days or after strength",
        "video_url": "https://www.youtube.com/watch?v=sRnUeDlQLX0"
      },
      {
        "exercise_name": "Single-Leg Hops",
        "purpose": "Balance and stiffness through the foot and ankle",
        "when_to_use": "Prehab or plyo-focused days",
        "video_url": None
      },
      {
        "exercise_name": "Bounding",
        "purpose": "Stride length, coordination, and horizontal force",
        "when_to_use": "Before or during speed development cycles",
        "video_url": "https://www.youtube.com/watch?v=1Fz71FYlXv4"
      },
      {
        "exercise_name": "Broad Jumps",
        "purpose": "Explosive hip drive",
        "when_to_use": "Post-strength or in power blocks",
        "video_url": "https://www.youtube.com/watch?v=U4s4mEQ5VqU"
      },
      {
        "exercise_name": "Box Step-Up + Hop",
        "purpose": "Hip flexor strength and vertical power",
        "when_to_use": "Post strength or warmup for hill reps",
        "video_url": None
      },
      {
        "exercise_name": "Split Lunge Jumps",
        "purpose": "Hip elasticity and balance",
        "when_to_use": "Prehab or plyometric circuits",
        "video_url": "https://www.youtube.com/watch?v=FVL5vrsNhX8"
      }
    ]
  },
  {
    "name": "Band & Activation Work",
    "exercises": [
      {
        "exercise_name": "Lateral Band Walk",
        "purpose": "Glute medius activation and hip stability",
        "when_to_use": "Pre-run or gym cooldown",
        "video_url": "https://www.youtube.com/watch?v=2-fmH3C9R10"
      },
      {
        "exercise_name": "Monster Walk",
        "purpose": "Hip abduction and core control",
        "when_to_use": "Prehab or during strength circuits",
        "video_url": "https://www.youtube.com/watch?v=j3Igk5nyZE4"
      },
      {
        "exercise_name": "Banded Glute Bridge",
        "purpose": "Posterior chain activation",
        "when_to_use": "Warm-up or rehab routine",
        "video_url": "https://www.youtube.com/watch?v=U2hxU-nzEus"
      },
      {
        "exercise_name": "Banded Clamshell",
        "purpose": "Glute med & hip rotation control",
        "when_to_use": "Core or injury prevention work",
        "video_url": "https://www.youtube.com/watch?v=3z3sIFoVtRE"
      },
      {
        "exercise_name": "Fire Hydrants",
        "purpose": "Hip rotation + lateral glute firing",
        "when_to_use": "Band circuits or cooldowns",
        "video_url": None
      },
      {
        "exercise_name": "Standing Band March",
        "purpose": "Hip flexor and core patterning",
        "when_to_use": "Pre-run or gait prep",
        "video_url": None
      }
    ]
  },
  {
    "name": "Stretching & Mobility",
    "exercises": [
      {
        "exercise_name": "Couch Stretch",
        "purpose": "Hip flexor & quad release",
        "when_to_use": "Post-run or evening mobility",
        "video_url": "https://www.youtube.com/watch?v=-ZX1QMTdAC4"
      },
      {
        "exercise_name": "90/90 Stretch",
        "purpose": "Hip capsule and internal rotation",
        "when_to_use": "Mobility work or cooldown",
        "video_url": "https://www.youtube.com/watch?v=NmHJVA1ZgiQ"
      },
      {
        "exercise_name": "Pigeon Pose",
        "purpose": "Glute and deep hip mobility",
        "when_to_use": "Evening or cooldown stretch",
        "video_url": "https://www.youtube.com/watch?v=F1eZkJ65Zbo"
      },
      {
        "exercise_name": "Calf Stretch (Bent Knee)",
        "purpose": "Soleus and ankle flexibility",
        "when_to_use": "Post-run or daily maintenance",
        "video_url": None
      },
      {
        "exercise_name": "Hamstring Scoops",
        "purpose": "Hamstring dynamic mobility",
        "when_to_use": "Pre-run dynamic stretch",
        "video_url": None
      },
      {
        "exercise_name": "World's Greatest Stretch",
        "purpose": "Full-body integrated mobility",
        "when_to_use": "Recovery day or evening routine",
        "video_url": "https://www.youtube.com/watch?v=ZVnMUhxbF6s"
      }
    ]
  },
  {
    "name": "Foot & Ankle Strength",
    "exercises": [
      {
        "exercise_name": "Toe Yoga",
        "purpose": "Toe articulation & arch control",
        "when_to_use": "Barefoot drills, prehab",
        "video_url": None
      },
      {
        "exercise_name": "Short Foot / Doming",
        "purpose": "Intrinsic foot muscle activation",
        "when_to_use": "Post-run barefoot stability",
        "video_url": None
      },
      {
        "exercise_name": "Ankle ABCs",
        "purpose": "Ankle mobility + proprioception",
        "when_to_use": "Cooldown or rehab work",
        "video_url": None
      },
      {
        "exercise_name": "Single-Leg Balance (Eyes Closed)",
        "purpose": "Proprioception and stability",
        "when_to_use": "Daily or mobility circuit",
        "video_url": None
      },
      {
        "exercise_name": "Towel Curls",
        "purpose": "Arch and toe strength",
        "when_to_use": "Post-run or recovery day",
        "video_url": None
      },
      {
        "exercise_name": "Banded Inversion/Eversion",
        "purpose": "Ankle strengthening and resilience",
        "when_to_use": "Rehab or stability protocol",
        "video_url": None
      }
    ]
  }
]
        try:
            return render_template('movement_catalog.html', movement_categories = movement)
        except Exception as e:
            logger.error(f"Error displaying max weights page: {e}")
            return render_template('error.html', error='Failed to movement catalog page.')