# Flask Application Documentation  
  
## Summary  
  
- **Total Blueprints**: 4  
- **Total Routes**: 39  
- **Total Services**: 9  
- **Total Biomechanic Modules**: 24  
- **HTTP Methods**: GET(28), POST(11), PUT(1), DELETE(1)  
  
## Blueprint: coach_g  
  
**URL Prefix**: `/coach`  
**Directory**: `blueprints\coach_g`  
**Routes Count**: 2  
  
**Services Used**: CoachGService, coach_g_service  
  
### Routes  
  
- **GET** `/coach/` → `coach_g()`  
    - Render Coach G chat interface  
- **POST** `/coach/api/chat/` → `coach_g_chat()`  
    - Handle Coach G chat interactions  
  
## Blueprint: runner_vision  
  
**URL Prefix**: `/vision`  
**Directory**: `blueprints\runner_vision`  
**Routes Count**: 5  
  
**Services Used**: runnervision_service, RunnerVisionService  
  
### Routes  
  
- **GET** `/vision/` → `runnervision()`  
    - Display RunnerVision analysis results  
- **GET** `/vision/reports/<path:filename>` → `serve_report()`  
    - Serve RunnerVision report files  
    - Parameters: filename  
- **GET** `/vision/videos/<path:filename>` → `serve_video()`  
    - Serve RunnerVision video files  
    - Parameters: filename  
- **POST** `/vision/upload` → `upload_files()`  
    - Handle video file uploads  
- **POST** `/vision/run_biomechanic_analysis` → `run_analysis()`  
    - Execute biomechanics analysis  
  
## Blueprint: running_hub  
  
**URL Prefix**: `/hub`  
**Directory**: `blueprints\running_hub`  
**Routes Count**: 9  
  
**Services Used**: trophy_service, QueryService, TrophyService, StatisticsService, ActivityService, MotivationService, activity_service, statistics_service, motivation_service, query_service  
  
### Routes  
  
- **GET** `/hub/` → `home()`  
    - RunningHub home page with latest activity  
- **GET** `/hub/activity/` → `activity()`  
    - Display detailed activity information for specific activity ID  
- **GET, POST** `/hub/query/` → `query()`  
    - Handle database queries  
- **POST** `/hub/ai_query` → `ai_query()`  
    - AI-powered natural language query interface  
- **GET** `/hub/statistics/` → `statistics()`  
    - Display time period aggregated running statistics  
- **GET** `/hub/trophy_room/` → `trophy_room()`  
    - Display personal records and achievements  
- **GET** `/hub/motivation` → `motivation()`  
    - Motivation Page, including upcoming races and inspriational LLM quote generator  
- **POST** `/hub/api/daily_motivation` → `daily_motivation()`  
    - API endpoint to generate a daily motivational message based on selected personality  
- **GET** `/hub/skill_tree/` → `skill_tree()`  
    - Progressive skill tree for running achievement and side-quests  
  
## Blueprint: run_strong  
  
**URL Prefix**: `/strong`  
**Directory**: `blueprints\run_strong`  
**Routes Count**: 23  
  
**Services Used**: runstrong_service, RunStrongService  
  
### Routes  
  
- **GET** `/strong/` → `runstrong()`  
    - Display RunStrong home page  
- **GET** `/strong/exercise-library` → `exercise_library()`  
    - Display exercise library  
- **GET** `/strong/planner` → `planner()`  
    - Display workout planner  
- **GET** `/strong/journal` → `journal()`  
    - Display workout journal  
- **GET** `/strong/progress-dashboard` → `progress_dashboard()`  
    - Display progress and fatigure dashboard  
- **GET** `/strong/exercises` → `exercises()`  
    - API endpoint for exercise data  
- **GET, POST** `/strong/exercises/add` → `add_exercise()`  
    - Add new exercise  
- **POST** `/strong/save-routine` → `save_routine()`  
    - Save workout routine  
- **GET** `/strong/load-routines` → `load_routines()`  
    - Load all workout routines  
- **GET** `/strong/load-routine/<int:routine_id>` → `load_routine()`  
    - Load specific workout routine  
    - Parameters: routine_id  
- **GET** `/strong/api/exercises` → `get_exercises()`  
    - Get all exercises for the planner  
- **GET** `/strong/api/routines` → `get_routines()`  
    - Get all workout routines  
- **POST** `/strong/api/routines` → `create_routine()`  
    - Create a new workout routine with exercises  
- **GET** `/strong/api/routines/<int:routine_id>/exercises` → `get_routine_exercises_api()`  
    - Get a specific routine with its exercises  
    - Parameters: routine_id  
- **POST** `/strong/api/workout-performance` → `save_workout_performance()`  
    - Save workout performance data for an entire session  
- **GET** `/strong/api/workout-performance/<int:routine_id>` → `get_workout_history()`  
    - Get workout history for a specific routine  
    - Parameters: routine_id  
- **PUT** `/strong/api/routines/<int:routine_id>` → `update_routine()`  
    - Update an existing workout routine  
    - Parameters: routine_id  
- **GET** `/strong/api/exercise-max-loads` → `get_exercise_max_loads()`  
    - Get maximum load for each exercise from workout history  
- **DELETE** `/strong/api/routines/<int:routine_id>` → `delete_routine()`  
    - Delete a workout routine  
    - Parameters: routine_id  
- **GET** `/strong/api/fatigue-data` → `get_fatigue_data()`  
    - API endpoint to get current fatigue data with optional filtering  
- **GET** `/strong/api/muscle-groups` → `get_available_muscle_groups()`  
    - API endpoint to get available muscle group filters  
- **GET** `/strong/api/update-fatigue` → `update_fatigue()`  
    - API endpoint to trigger fatigue update  
- **POST** `/strong/api/workout-performance/freestyle` → `save_freestyle_workout()`  
    - API endpoint to save an ad-hoc/freestyle workout session  
  
