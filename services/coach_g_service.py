# services/coach_g_service.py
"""
Service layer for running Coach G Language Model.
"""
import logging
import re
from typing import List, Dict

import markdown  # Import the markdown library
from services.base_service import BaseService
from utils import language_model_utils, exception_utils
from config import LanguageModelConfig
from utils.db import language_db_utils, runstrong_db_utils

class CoachGService(BaseService):
    """Service for handling Coach G interactions."""
    
    def __init__(self, config):
        super().__init__(config.DB_PATH)
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.lm_config = LanguageModelConfig()
        self._initialize_language_model()

    def _initialize_language_model(self):
        """Initialize the language model."""
        if self.lm_config.LANGUAGE_MODEL_ACTIVE:
            try:
                self.coach_g = language_model_utils.LanguageModel()
                self.tokenizer = self.coach_g.tokenizer
                self.logger.info("CoachGService initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize language model in service: {e}")
                raise
        else:
            self.logger.info("Language Model activation set to False")



    def handle_quick_query(self, session_id: str, user_query: str, personality: str, topic: str) -> str:
        """
        Handles the user's quick query by routing to the appropriate function.
        """
        if topic == 'running_status':
            response_for_user = self._get_daily_training_summary()
            
            # If a summary was successfully generated, create and save a text version for history
            if "<p>" in response_for_user: # A simple check to see if we have data
                with self._get_connection() as conn:
                    latest_metrics = language_db_utils.get_latest_daily_training_metrics(conn=conn)
                if latest_metrics:
                    text_for_history = self._create_text_summary_for_history(latest_metrics[0])
                    self._save_message(session_id, "coach", text_for_history)
            
            return response_for_user
        
        elif topic == 'strength_status':
            text_for_history, response_for_user = self._get_strength_training_summary()
            self._save_message(session_id, "coach", text_for_history)
            return response_for_user

        elif topic == 'weekly_summary':
            text_for_history, response_for_user = self._get_weekly_running_summary()
            self._save_message(session_id, "coach", text_for_history)
            return response_for_user
            
        elif topic == 'training_trend':
            text_for_history, response_for_user = self._get_training_trend_summary()
            self._save_message(session_id, "coach", text_for_history)
            return response_for_user

        else:
            # Fallback for unknown topics
            self.logger.warning(f"Unknown quick query topic received: {topic}")
            return self.handle_user_query(session_id, user_query, personality)

    def handle_user_query(self, session_id: str, user_query: str, personality: str) -> str:
        """
        Handles the user's query by routing to the appropriate function.
        """
        sanitized_query = self._sanitize_user_input(user_query)
        self._save_message(session_id, "user", sanitized_query)

        daily_metric_keywords = ['atl', 'ctl', 'fatigue', 'freshness', 'training status']

        try:
            is_data_query = any(keyword in sanitized_query.lower() for keyword in daily_metric_keywords)

            if is_data_query:
                # Get the HTML response for the user
                response_for_user = self._get_daily_training_summary()
                
                # If a summary was successfully generated, create and save a text version for history
                if "<p>" in response_for_user: # A simple check to see if we have data
                    with self._get_connection() as conn:
                        latest_metrics = language_db_utils.get_latest_daily_training_metrics(conn=conn)
                    if latest_metrics:
                        text_for_history = self._create_text_summary_for_history(latest_metrics[0])
                        self._save_message(session_id, "coach", text_for_history)
                
                return response_for_user
            else:
                history = self._get_recent_messages(session_id, max_tokens=self.lm_config.MAX_CONTEXT_TOKENS)
                response = self.coach_g.generate_general_coach_g_reply(sanitized_query, personality, history)
                self._save_message(session_id, "coach", response)
                return response

        except Exception as e:
            self.logger.error(f"Error handling user query: {e}", exc_info=True)
            return "<p>I'm having a bit of trouble connecting right now. Let's try again in a moment.</p>"

    def _create_text_summary_for_history(self, metrics: Dict) -> str:
        """Creates a simple, text-only summary for the LLM's context history."""
        tsb = metrics.get('tsb', 0)
        return (
            f"Here is the user's training status for {metrics['date']}: "
            f"CTL (Fitness) is {metrics.get('ctl', 0):.1f}, "
            f"ATL (Fatigue) is {metrics.get('atl', 0):.1f}, "
            f"and TSB (Freshness) is {tsb:.1f}."
        )
    def _create_summary_for_strength_history(self, strength_metrics: Dict) -> tuple:
        """Creates a simple, text-only summary for the LLM's context history on strength training fatigue."""
        overall_fatigue = strength_metrics.get("overall_fatigue", 0)
        
        # Identify top fatigued muscle groups (fatigue_level > 70)
        fatigued_muscles = [
            m["muscle_group"] for m in strength_metrics.get("muscle_fatigue", [])
            if m["fatigue_level"] >= 70
        ]
        top_fatigued = ", ".join(fatigued_muscles[:4]) + ("..." if len(fatigued_muscles) > 4 else "")
        
        # Recent training summary
        recent_training_days = [
            d for d in strength_metrics.get("daily_training", []) if d["hasTraining"]
        ]
        recent_training_summary = ", ".join(
            f"{d['day']} ({d['volume']:.0f} units, {d['intensity']}% intensity)"
            for d in recent_training_days[::-1]
        )

        if overall_fatigue < 50:
            coach_g_reply = 'Your overall fatigue is not too high, you can hit the gym pretty fresh!'
        elif overall_fatigue < 75: 
            coach_g_reply = 'Your overall fatigue is moderate, if you do go to the gym, be cognizant of your most fatigued muscles.'
        else:
            coach_g_reply = "You've been putting in work at the gym, give yourself a day or two to rebuild."

        text_summary = (
            f"Strength training status summary:\n"
            f"- Overall fatigue score: {overall_fatigue} out of 100.\n"
            f"- Heavily fatigued muscles: {top_fatigued or 'None'}.\n"
            f"- Recent training days include: {recent_training_summary or 'None'}."
            f"- Coach G's Advice**: {coach_g_reply}"
        )

        markdown_summary = (
            f"### Strength Training Status Summary:\n\n"
            f"**Overall fatigue score**: {overall_fatigue} out of 100.\n\n"
            f"**Heavily fatigued muscles**: {top_fatigued or 'None'}.\n\n"
            f"**Recent training days include**: {recent_training_summary or 'None'}.\n\n"
            f"**Coach G's Advice**: {coach_g_reply}"
        )
        return text_summary, markdown_summary
    
    def _get_daily_training_summary(self) -> str:
        """
        Fetches, formats, and converts the latest daily training metrics to HTML.
        """
        try:
            with self._get_connection() as conn:
                latest_metrics = language_db_utils.get_latest_daily_training_metrics(conn=conn)
            
            if not latest_metrics:
                return "<p>I couldn't find any recent training data to give you a summary.</p>"
            
            markdown_summary = self.coach_g.format_daily_training_summary(latest_metrics[0])
            html_summary = markdown.markdown(markdown_summary)
            
            return html_summary

        except Exception as e:
            self.logger.error(f"Error getting daily training summary: {e}")
            return "<p>I was unable to retrieve your latest training summary.</p>"
        
    def _get_strength_training_summary(self) -> tuple:
        """
        Fetches, formats, and converts the latest strength training metrics to HTML.
        """
        try:
            with self._get_connection() as conn:
                latest_metrics = runstrong_db_utils.get_fatigue_dashboard_data(conn=conn)
            
            if not latest_metrics:
                return "<p>I couldn't find any recent training data to give you a summary.</p>"
            
            text_summary, markdown_summary = self._create_summary_for_strength_history(latest_metrics)
            html_summary = markdown.markdown(markdown_summary)
            
            return text_summary, html_summary

        except Exception as e:
            self.logger.error(f"Error getting daily training summary: {e}")
            return "<p>I was unable to retrieve your latest training summary.</p>"

    def _sanitize_user_input(self, user_query: str) -> str:
        """Basic sanitization of user input."""
        return re.sub(r'[^\w\s\.\!\?\,\-\'\"]+', '', user_query).strip()[:500]

    def _save_message(self, session_id: str, role: str, message: str):
        """Save a message to the conversation history."""
        try:
            with self._get_connection() as conn:
                language_db_utils.save_message(conn, session_id, role, message)
        except Exception as e:
            self.logger.error(f"Error saving message: {e}")
            raise exception_utils.DatabaseError(f"Failed to save message: {e}")

    def _get_recent_messages(self, session_id: str, max_tokens: int = 512) -> List[Dict]:
        """Retrieve recent messages for context."""
        try:
            with self._get_connection() as conn:
                return language_db_utils.get_recent_messages(conn, session_id, max_tokens, self.tokenizer)
        except Exception as e:
            self.logger.error(f"Error retrieving messages: {e}")
            return []
        
    def _get_weekly_running_summary(self) -> tuple[str, str]:
        """
        Fetches and formats a summary of the last 7 days of running.
        Returns a text summary for history and a Markdown summary for the user.
        """
        try:
            with self._get_connection() as conn:
                weekly_summary = language_db_utils.get_running_summary_for_last_n_days(conn, days=7)
                metrics_data = language_db_utils.get_daily_metrics_for_last_n_days(conn, days=7)

            if not weekly_summary or not weekly_summary.get('num_runs'):
                no_data_message = "<p>It looks like you haven't logged any runs in the last 7 days. Time to lace up!</p>"
                return "No running data for the last 7 days.", no_data_message
            
            # --- Data Processing ---
            total_distance_m = weekly_summary.get('total_distance', 0)
            total_time_s = weekly_summary.get('total_moving_time', 0)
            num_runs = weekly_summary.get('num_runs', 0)
            total_elevation_m = weekly_summary.get('total_elevation', 0)

            # Convert to preferred units (e.g., miles, hours/minutes)
            total_dist_miles = total_distance_m * 0.000621371
            hours, remainder = divmod(total_time_s, 3600)
            minutes, _ = divmod(remainder, 60)
            total_elev_ft = total_elevation_m * 3.28084

            start_ctl = metrics_data[0].get('ctl', 0) if metrics_data else 0
            end_ctl = metrics_data[-1].get('ctl', 0) if metrics_data else 0
            ctl_change = end_ctl - start_ctl

            avg_tsb = sum(m.get('tsb', 0) for m in metrics_data) / len(metrics_data) if metrics_data else 0

            # --- Insight Generation ---
            if ctl_change > 2:
                insight = f"Fantastic work! You've significantly built your fitness (CTL) by {ctl_change:.1f} points. This was a productive week."
            elif ctl_change > 0:
                insight = f"Great consistency! You've maintained your training and nudged your fitness up by {ctl_change:.1f} points."
            else:
                insight = f"This was a lighter week, which can be great for recovery. Your fitness (CTL) changed by {ctl_change:.1f} points, allowing your body to absorb previous training."

            # --- Create Summaries ---
            text_summary = (
                f"User's weekly running summary: {num_runs} runs, {total_dist_miles:.1f} miles, "
                f"{int(hours)}h {int(minutes)}m, {total_elev_ft:.0f} ft elevation. "
                f"Fitness (CTL) changed by {ctl_change:.1f}. Average Form (TSB) was {avg_tsb:.1f}. "
                f"Coach's Insight: {insight}"
            )

            markdown_summary = (
    f"### Weekly Running Summary (Last 7 Days)\n\n"
    f"Here is a look at your training week:\n\n"
    f"- **Total Runs:** {num_runs}\n"
    f"- **Total Distance:** {total_dist_miles:.1f} miles\n"
    f"- **Total Time:** {int(hours)}h {int(minutes)}m\n"
    f"- **Total Elevation Gain:** {total_elev_ft:.0f} ft\n\n"
    f"#### Training Load Analysis:\n"
    f"- Your **Fitness (CTL)** started at `{start_ctl:.1f}` and ended at `{end_ctl:.1f}`, a change of **{ctl_change:+.1f}** points.\n"
    f"- Your average **Form (TSB)** for the week was **{avg_tsb:.1f}**.\n\n"
    f"**Coach G's Insight:** {insight}"
)
            html_summary = markdown.markdown(markdown_summary)
            return text_summary, html_summary

        except Exception as e:
            self.logger.error(f"Error getting weekly running summary: {e}")
            return "Error fetching weekly summary.", "<p>I was unable to retrieve your weekly running summary.</p>"


    def _get_training_trend_summary(self) -> tuple[str, str]:
        """
        Fetches and formats a summary of the last 28-day training trend.
        Returns a text summary for history and a Markdown summary for the user.
        """
        try:
            with self._get_connection() as conn:
                # Look at a 4-week trend
                metrics_data = language_db_utils.get_daily_metrics_for_last_n_days(conn, days=28)

            if not metrics_data or len(metrics_data) < 7: # Need at least a week of data for a trend
                no_data_message = "<p>I don't have enough data from the last month to analyze your training trend. Keep logging those activities!</p>"
                return "Not enough data for a training trend.", no_data_message

            # --- Data Processing ---
            start_ctl = metrics_data[0].get('ctl', 0)
            current_ctl = metrics_data[-1].get('ctl', 0)
            current_tsb = metrics_data[-1].get('tsb', 0)
            ctl_change = current_ctl - start_ctl
            

            # --- Insight Generation ---
            trend_description = ""
            if ctl_change > 4:
                trend_description = "a strong **Productive** phase"
            elif ctl_change > 0.5:
                trend_description = "a **Maintaining/Building** phase"
            elif ctl_change < -4:
                trend_description = "a **Detraining or Tapering** phase"
            else:
                trend_description = "a **Maintaining/Recovery** phase"

            insight = (f"Over the last 4 weeks, you've been in {trend_description}. Your fitness (CTL) has "
                    f"shifted from `{start_ctl:.1f}` to `{current_ctl:.1f}`. Your current form (TSB) is `{current_tsb:.1f}`. ")

            if current_tsb > 5:
                insight += "You should be feeling fresh and ready for your next key workout."
            elif current_tsb < -15:
                insight += "You're carrying some significant fatigue, which is expected during a build. Be sure to focus on recovery."
            else:
                insight += "Your form is neutral, indicating a good balance between fitness and fatigue."

            # --- Create Summaries ---
            text_summary = (
                f"User's 28-day training trend: In a {trend_description}. "
                f"CTL changed from {start_ctl:.1f} to {current_ctl:.1f}. "
                f"Current TSB is {current_tsb:.1f}. "
                f"Coach's Insight: {insight}"
            )

            markdown_summary = (
    f"### Your 4-Week Training Trend\n\n"
    f"Here's an analysis of your training trend over the last 28 days:\n\n"
    f"- **Fitness Trend:** You are in {trend_description}.\n"
    f"- **Fitness (CTL) Change:** Went from **{start_ctl:.1f} to {current_ctl:.1f}** ({ctl_change:+.1f} points).\n"
    f"- **Current Form (TSB):** Your current freshness score is **{current_tsb:.1f}**.\n\n"
    f"**Coach G's Insight:** {insight}"
)
            html_summary = markdown.markdown(markdown_summary)
            return text_summary, html_summary

        except Exception as e:
            self.logger.error(f"Error getting training trend summary: {e}")
            return "Error fetching training trend.", "<p>I was unable to retrieve your training trend summary.</p>"