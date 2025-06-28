# Services Documentation  
  
Found 9 service files.  
  
## Service: activity_service  
  
**File**: `services\activity_service.py`  
  
### Classes  
  
#### ActivityService  
  
Service for handling activity-related operations.  
  
**Inherits from**: BaseService  
  
**Methods**:  
  
- `get_latest_activity_id()` → Optional[int]  
    - Get the ID of the most recent run activity.  
- `get_formatted_activity_page_details(activity_id, units)` → Optional[Dict[(str, Any)]]  
    - Retrieves all activity information with formatted data.  
- `_format_activity_data(activity, units)` → Dict[(str, Any)]  
    - Format activity data for display, including unit conversion for miles and kilometers.  
  
---  
  
## Service: base_service  
  
**File**: `services\base_service.py`  
  
### Classes  
  
#### BaseService  
  
Abstract base class for all services.  
  
**Inherits from**: ABC  
  
**Methods**:  
  
- `__init__(db_path)`  
- `_get_connection()` → sqlite3.Connection  
    - Get database connection with error handling.  
  
---  
  
## Service: coach_g_service  
  
**File**: `services\coach_g_service.py`  
  
### Classes  
  
#### CoachGService  
  
Service for handling Coach G interactions.  
  
**Inherits from**: BaseService  
  
**Methods**:  
  
- `__init__(config)`  
- `_initialize_language_model()`  
    - Initialize the language model.  
- `handle_quick_query(session_id, user_query, personality, topic)` → str  
    - Handles the user's quick query by routing to the appropriate function.  
- `handle_user_query(session_id, user_query, personality)` → str  
    - Handles the user's query by routing to the appropriate function based on keywords.  
- `_create_text_summary_for_history(metrics)` → str  
    - Creates a simple, text-only summary for the LLM's context history.  
- `_create_summary_for_strength_history(strength_metrics)` → tuple  
    - Creates a simple, text-only summary for the LLM's context history on strength training fatigue.  
- `_get_daily_training_summary(conn)` → str  
    - Fetches, formats, and converts the latest daily training metrics to HTML.  
- `_get_strength_training_summary()` → tuple  
    - Fetches, formats, and converts the latest strength training metrics to HTML.  
- `_sanitize_user_input(user_query)` → str  
    - Basic sanitization of user input.  
- `_save_message(session_id, role, message)`  
    - Save a message to the conversation history.  
- `_get_recent_messages(session_id, max_tokens)` → List[Dict]  
    - Retrieve recent messages for context.  
- `_get_weekly_running_summary()` → tuple[(str, str)]  
    - Fetches and formats a summary of the last 7 days of running.  
- `_get_training_trend_summary()` → tuple[(str, str)]  
    - Fetches and formats a summary of the last 28-day training trend.  
- `get_daily_motivational_message(session_id, personality, profile_data)` → str  
  
---  
  
## Service: motivation_service  
  
**File**: `services\motivation_service.py`  
  
### Classes  
  
#### MotivationService  
  
Service for managing personal records and achievements.  
  
**Inherits from**: CoachGService  
  
---  
  
## Service: query_service  
  
**File**: `services\query_service.py`  
  
### Classes  
  
#### QueryService  
  
Service for handling database queries.  
  
**Inherits from**: BaseService  
  
**Methods**:  
  
- `execute_query(sql_query, param_input)` → Dict[(str, Any)]  
    - Execute a database query with parameters.  
  
---  
  
## Service: runnervision_service  
  
**File**: `services\runnervision_service.py`  
  
### Classes  
  
#### RunnerVisionService  
  
Service for RunnerVision biomechanics analysis.  
  
**Methods**:  
  
- `__init__(config)`  
- `get_latest_analysis()` → Dict[(str, Any)]  
    - Get latest RunnerVision analysis results.  
- `serve_report(filename)`  
    - Serve report files safely.  
- `serve_video(filename)`  
    - Serve video files safely.  
- `handle_file_upload(files)` → Dict[(str, Any)]  
    - Handle video file uploads.  
- `run_analysis()` → Dict[(str, Any)]  
    - Execute biomechanics analysis.  
- `_allowed_file(filename)` → bool  
    - Check if file extension is allowed.  
  
---  
  
## Service: runstrong_service  
  
**File**: `services\runstrong_service.py`  
  
### Classes  
  
#### RunStrongService  
  
Service for RunStrong strength training operations.  
  
**Inherits from**: BaseService  
  
**Methods**:  
  
- `add_exercise(data)` → None  
    - Add single exercise to db.  
- `save_routine(routine_name, routine_exercises)` → None  
    - Save a workout routine.  
- `get_routines()` → List[Tuple[(int, str)]]  
    - Get all workout routines.  
- `get_exercises()` → List[Tuple[(int, str)]]  
    - Get all available exercises.  
- `get_exercise_by_id(exercise_id)` → Optional[Tuple[(int, str)]]  
    - Get exercise by ID.  
- `get_all_routines()` → List[Tuple[(int, str)]]  
    - Get all available routines.  
- `get_routine_by_id(routine_id)` → Optional[Tuple[(int, str)]]  
    - Get routine by ID.  
- `create_routine(name)` → Optional[int]  
    - Create new routine and return routine ID.  
- `add_exercise_to_routine(routine_id, exercise_id, sets, reps, load_lbs, order_index, notes)` → bool  
    - Add exercise to routine.  
- `get_routine_exercises(routine_id)` → List[Dict]  
    - Get all exercises for a specific routine  
- `delete_routine(routine_id)` → bool  
    - Delete a routine and all associated exercises  
- `save_workout_performance_bulk(routine_id, workout_date, exercises)`  
    - Service layer method to save a complete workout performance log.  
- `get_workout_history(routine_id)` → List[Dict]  
    - Get workout history for a specific routine  
- `get_workout_performance_by_date(routine_id, workout_date)` → List[Dict]  
    - Get workout performance for a specific routine and date  
- `get_exercise_progress(exercise_id, limit)` → List[Dict]  
    - Get progress history for a specific exercise  
- `get_recent_workouts(limit)` → List[Dict]  
    - Get recent workout sessions  
- `get_workout_stats(routine_id)` → Dict  
    - Get workout statistics  
- `initialize_runstrong_database()` → bool  
    - Initialize the database with all required tables  
- `update_routine_name(routine_id, name)`  
    - Update routine name.  
- `clear_routine_exercises(routine_id)`  
    - Remove all exercises from a routine.  
- `get_exercise_max_loads()` → dict  
    - Get maximum load for each exercise from workout performance history.  
- `get_routine_name_datecreated()` → list  
    - Get all workout routines.  
- `run_daily_update()`  
- `get_fatigue_dashboard_data(muscle_group_filter)` → Dict  
    - Get fatigue dashboard data with optional muscle group filtering  
- `update_weekly_summary(week_start)`  
- `get_recommendation(overall_fatigue)`  
- `get_least_used_muscle_groups(muscle_fatigue, days_threshold)`  
- `save_freestyle_workout(workout_date, exercises)`  
    - Saves an ad-hoc (freestyle) workout session.  
  
---  
  
## Service: statistics_service  
  
**File**: `services\statistics_service.py`  
  
### Classes  
  
#### StatisticsService  
  
Service for generating running statistics.  
  
**Inherits from**: BaseService  
  
**Methods**:  
  
- `get_statistics(period, units)` → Dict[(str, Any)]  
    - Get comprehensive running statistics for a time period.  
- `_get_period_start_date(now, period)` → str  
    - Calculate start date based on period.  
- `_get_date_range_label(period)` → str  
    - Get human-readable date range label.  
- `_get_summary_statistics(start_date, units)` → Dict[(str, Any)]  
    - Get summary statistics for the period.  
- `_get_weekly_distances(now)` → List[float]  
    - Get distances for the last 7 days.  
- `_get_pace_trends(units)` → Dict[(str, List)]  
    - Get pace trends for the last 10 activities.  
- `_get_shoe_usage(start_date, units)` → List[Dict[(str, Any)]]  
    - Get shoe usage statistics.  
- `_get_recent_activities(units)` → List[Dict[(str, Any)]]  
    - Get recent activities with formatted data.  
  
---  
  
## Service: trophy_service  
  
**File**: `services\trophy_service.py`  
  
### Classes  
  
#### TrophyService  
  
Service for managing personal records and achievements.  
  
**Inherits from**: BaseService  
  
**Methods**:  
  
- `get_personal_records(units)` → List[Dict[(str, Any)]]  
    - Get all personal records for standard race distances.  
- `_format_record(activity, distance_name, units)` → Dict[(str, Any)]  
    - Format a personal record for display.  
  
---  
  
