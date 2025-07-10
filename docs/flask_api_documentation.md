# Flask Application Documentation  
  
## Summary  
  
- **Total Blueprints**: 4  
- **Total Routes**: 23  
- **Total Services**: 9  
- **Total Biomechanic Modules**: 24  
- **HTTP Methods**: GET(17), POST(7)  
  
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
  
**Services Used**: StatisticsService, TrophyService, motivation_service, statistics_service, query_service, ActivityService, MotivationService, trophy_service, activity_service, QueryService  
  
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
**Routes Count**: 7  
  
**Services Used**: runstrong_service, RunStrongService  
  
### Routes  
  
- **GET** `/strong/` → `runstrong()`  
    - Display RunStrong home page  
- **GET** `/strong/api/exercises` → `get_exercises()`  
    - API: Get all exercises for the planner  
- **GET** `/strong/exercise_library` → `exercise_library()`  
    - Display the exercise library page  
- **GET** `/strong/journal` → `journal()`  
    - Display the workout journal page and the form for new entries  
- **POST** `/strong/api/journal/log` → `log_workout_entry()`  
    - API endpoint to log a new workout session  
- **GET** `/strong/fatigue_dashboard` → `fatigue_dashboard()`  
    - Display the enhanced fatigue dashboard page  
- **GET** `/strong/goals` → `goals()`  
    - Display the goals dashboard page  
  
