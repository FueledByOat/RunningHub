# coach_g_service.py
"""
Service layer for running Coach G Language Model.

This module contains all business logic services that handle data processing,
database interactions, and business rules while keeping the Flask routes clean.
"""
import logging
import re

import json
from typing import Dict, Any, Optional, List

from services.base_service import BaseService
from utils import format_utils, exception_utils, language_model_utils
from config import LanguageModelConfig, Config
from utils.db import db_utils
from utils.db import language_db_utils

class CoachGService(BaseService):
    """Service for handling Coach G interactions."""
    
    def __init__(self, config):
        # Extract db_path from config object and pass to parent
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

    def handle_user_query(self, session_id: str, user_query: str, personality: str) -> str:
        """
        Handles the user's query by routing to the appropriate function.
        """
        sanitized_query = self._sanitize_user_input(user_query)
        self._save_message(session_id, "user", sanitized_query)

        # Keywords to trigger the daily training metrics summary
        daily_metric_keywords = ['atl', 'ctl', 'fatigue', 'freshness', 'training status']

        try:
            if any(keyword in sanitized_query.lower() for keyword in daily_metric_keywords):
                response = self._get_daily_training_summary()
            else:
                history = self._get_recent_messages(session_id, max_tokens=self.lm_config.MAX_CONTEXT_TOKENS)
                response = self.coach_g.generate_general_coach_g_reply(sanitized_query, personality, history)

            self._save_message(session_id, "coach", response)
            return response
        except Exception as e:
            self.logger.error(f"Error handling user query: {e}", exc_info=True)
            return "I'm having a bit of trouble connecting right now. Let's try again in a moment."

    def _get_daily_training_summary(self) -> str:
        """
        Fetches and formats the latest daily training metrics.
        """
        try:
            with self._get_connection() as conn:
                latest_metrics = language_db_utils.get_latest_daily_training_metrics(conn=conn)
            
            if not latest_metrics:
                return "I couldn't find any recent training data to give you a summary."
            
            # The database utility returns a list of dictionaries
            return self.coach_g.format_daily_training_summary(latest_metrics[0])

        except Exception as e:
            self.logger.error(f"Error getting daily training summary: {e}")
            return "I was unable to retrieve your latest training summary."

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