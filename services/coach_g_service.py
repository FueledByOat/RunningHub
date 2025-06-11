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
from utils.db import language_db_utils

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

    def _create_text_summary_for_history(self, metrics: Dict) -> str:
        """Creates a simple, text-only summary for the LLM's context history."""
        tsb = metrics.get('tsb', 0)
        return (
            f"Here is the user's training status for {metrics['date']}: "
            f"CTL (Fitness) is {metrics.get('ctl', 0):.1f}, "
            f"ATL (Fatigue) is {metrics.get('atl', 0):.1f}, "
            f"and TSB (Freshness) is {tsb:.1f}."
        )

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