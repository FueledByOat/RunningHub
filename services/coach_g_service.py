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
        """Initialize the language model with error handling."""
        if self.lm_config.LANGUAGE_MODEL_ACTIVE == True:
            try:
                self.coach_g = language_model_utils.LanguageModel()
                self.tokenizer = self.coach_g.tokenizer
                self.logger.info("CoachGService initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize language model in service: {e}")
                raise
        else:
            self.logger.info(f"Language Model activation set to False")

    def daily_training_summary(self) -> str:
        """Get daily training summary."""
        try:
            return self.coach_g.generate_daily_training_summary()
        except Exception as e:
            self.logger.error(f"Error generating daily training summary: {e}")
            return "Unable to generate training summary at this time."

    def general_reply(self, session_id: str, user_query: str, personality: str) -> str:
        try:
            sanitized_user_query = self._sanitize_user_input(user_query)
            self._save_message(session_id, "user", sanitized_user_query)
            
            history = self._get_recent_messages(
                session_id, 
                max_tokens=self.lm_config.MAX_CONTEXT_TOKENS
            )
            
            # *** NEW LOGIC FLOW ***
            intent = self._classify_intent(sanitized_user_query)

            if intent == "DAILY_SUMMARY":
                response = self.daily_training_summary()
            elif intent == "DATA_QUERY":
                # Only run the expensive data pipeline if needed
                response = self._handle_data_query(sanitized_user_query, personality, history)
            else: # GENERAL_CHAT
                # Go directly to the general-purpose response generator
                response = self._generate_with_quality_check(sanitized_user_query, personality, history)
            
            self._save_message(session_id, "coach", response)
            return response
                
        except Exception as e:
            self.logger.error(f"Critical error in general_reply: {e}", exc_info=True)
            fallback_msg = self._get_fallback_response(user_query)
            self._save_message(session_id, "coach", fallback_msg)
            return fallback_msg

    def _handle_data_query(self, user_query: str, personality: str, history: List[Dict]) -> str:
        """The original logic for handling text-to-sql, now isolated."""
        try:
            raw_sql_suggestion = self.coach_g.generate_sql_from_natural_language(user_query)
            if raw_sql_suggestion:
                extracted_sql = self.coach_g.extract_sql_query(raw_sql_suggestion)
                if extracted_sql:
                    query_results = self._execute_generated_query(extracted_sql)
                    if query_results and query_results.get('error') is None:
                        data_driven_response = self.coach_g.generate_response_from_data(
                            user_query, extracted_sql, query_results, personality, history
                        )
                        if self._is_response_acceptable(data_driven_response, is_data_driven=True):
                            return data_driven_response
        except Exception as e:
            self.logger.warning(f"Data query handler failed: {e}. Falling back.")
        
        # Fallback if any step in the data query process fails
        self.logger.info(f"Falling back to general reply for data query: '{user_query[:50]}...'")
        return self._generate_with_quality_check(user_query, personality, history)
    
    def _sanitize_user_input(self, user_query: str) -> str:
        """Sanitize user input to prevent prompt injection."""
        # Remove potential prompt injection patterns
        sanitized = re.sub(r'(?i)(coach g|user):\s*', '', user_query)
        sanitized = re.sub(r'[^\w\s\.\!\?\,\-\'\"]+', '', sanitized)
        return sanitized.strip()[:500]  # Limit length
    
    def _generate_with_quality_check(self, user_query: str, personality: str, history: List[Dict]) -> str:
        """Generate response with quality validation."""
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                response = self.coach_g.generate_general_coach_g_reply(
                    user_query, personality, history=history
                )
                
                # Quality check
                if self._is_response_acceptable(response):
                    return response
                
                self.logger.warning(f"Response quality check failed, attempt {attempt + 1}")
                
            except Exception as e:
                self.logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
        
        # Fallback after all attempts
        return self._get_fallback_response(user_query)
    
    def _is_response_acceptable(self, response: str, is_data_driven: bool = False) -> bool: # Added is_data_driven flag
        """Check if response meets quality standards."""
        if not response or len(response.strip()) < 10: # Min char length
            self.logger.debug(f"Response failed length check (empty or <10 chars). Data_driven: {is_data_driven}")
            return False
        
        # --- Adjust sentence count limits based on the flag and LanguageModelConfig ---
        min_s = self.lm_config.RESPONSE_MIN_SENTENCES_DATA_DRIVEN if is_data_driven else self.lm_config.RESPONSE_MIN_SENTENCES
        max_s = self.lm_config.RESPONSE_MAX_SENTENCES_DATA_DRIVEN if is_data_driven else self.lm_config.RESPONSE_MAX_SENTENCES
        # --- End Adjust sentence count ---

        sentence_count = len([s for s in re.split(r'[.!?]+', response) if s.strip()])
        if sentence_count < min_s or sentence_count > max_s:
            self.logger.debug(f"Response failed sentence count: {sentence_count} (min:{min_s}, max:{max_s}, data_driven:{is_data_driven}). Response: '{response[:100]}...'")
            return False
        
        # Relaxed repetition check (e.g., 35% unique words minimum)
        words = response.lower().split()
        if len(words) > 5: # Only check for non-trivial responses
            unique_word_ratio = len(set(words)) / len(words) if len(words) > 0 else 0
            if unique_word_ratio < 0.35: # Allow more repetition for potentially complex data explanations
                self.logger.debug(f"Response failed repetition check (ratio: {unique_word_ratio:.2f}, data_driven:{is_data_driven}). Response: '{response[:100]}...'")
                return False
        
        # Check for obvious generation artifacts
        # Ensure these artifacts are compared in lowercase as well.
        forbidden_artifacts = ['<|end|>', '[end]', 'user:', 'coach g:'] # Add more if you see them
        response_lower = response.lower()
        if any(artifact in response_lower for artifact in forbidden_artifacts):
            self.logger.debug(f"Response failed artifact check (data_driven:{is_data_driven}). Response: '{response[:100]}...'")
            return False
        
        return True
    
    def _get_fallback_response(self, user_query: str) -> str:
        """Provide a contextual fallback response."""
        fallbacks = [
            "That's a great question about your training. Let me suggest focusing on consistency and gradual progression.",
            "Thanks for asking! Based on your running goals, I'd recommend starting with your current fitness level in mind.",
            "I appreciate you sharing that with me. Every runner's journey is unique, so let's work with what feels right for you."
        ]
        
        # Simple keyword matching for context
        query_lower = user_query.lower()
        if any(word in query_lower for word in ['pace', 'speed', 'fast']):
            return "Pacing is crucial in running. Start conservatively and build your speed gradually as your fitness improves."
        elif any(word in query_lower for word in ['distance', 'long', 'marathon']):
            return "Building distance takes time and patience. Focus on increasing your weekly mileage by no more than 10% each week."
        
        return fallbacks[hash(user_query) % len(fallbacks)]
   
    def _save_message(self, session_id: str, role: str, message: str):
        """Save message to conversations table."""
        try:
            with self._get_connection() as conn:
                language_db_utils.save_message(conn, session_id, role, message)
                self.logger.debug(f"Saved {role} message for session {session_id}")
        except Exception as e:
            self.logger.error(f"Error saving conversation message: {e}")
            raise exception_utils.DatabaseError(f"Failed to save message: {e}") from e

    def _get_recent_messages(self, session_id: str, max_tokens: int = 512) -> List[Dict]:
        """Get recent conversation messages within token limit."""
        try:
            with self._get_connection() as conn:
                history = language_db_utils.get_recent_messages(
                    conn, session_id, max_tokens, self.tokenizer
                )
                self.logger.debug(f"Retrieved {len(history)} messages for session {session_id}")
                return history
        except Exception as e:
            self.logger.error(f"Error retrieving conversation messages: {e}")
            return []  # Return empty list to allow conversation to continue

    

    def latest_training_metrics(self, sql_query: str, param_input: str = '{}') -> Dict[str, Any]:
        """Execute a database query with parameters."""
        if not sql_query.strip().lower().startswith("select"):
            raise exception_utils.DatabaseError("Only SELECT queries allowed")
        
        try:
            # Parse parameters
            param_input = param_input.strip() or '{}'
            params = json.loads(param_input)
            
            if not isinstance(params, dict):
                raise ValueError("Parameters must be a JSON object")
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql_query, params)
                
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                
                return {
                    'columns': columns,
                    'rows': rows,
                    'error': None
                }
                
        except json.JSONDecodeError:
            raise exception_utils.DatabaseError("Invalid JSON format in parameters")
        except Exception as e:
            self.logger.error(f"Query execution error: {e}")
            raise exception_utils.DatabaseError(f"Query failed: {e}")

    def _execute_generated_query(self, sql_query: str) -> Dict[str, Any]:
        """Executes a validated SELECT query generated by the LLM."""
        if not sql_query.strip().lower().startswith("select"):
            self.logger.warning(f"Attempt to execute non-SELECT query: {sql_query}")
            raise exception_utils.DatabaseError("Only SELECT queries are allowed.")

        # Potentially add more validation/parsing here if needed (e.g., allowed tables)
        # For now, the SELECT check is a primary safeguard.

        try:
            with self._get_connection() as conn:
                rows, columns = language_db_utils.execute_generated_query(conn, sql_query)
                return {'columns': columns, 'rows': rows, 'error': None}
        except Exception as e: # Catch other unexpected errors
            self.logger.error(f"Unexpected error executing generated SQL query '{sql_query}': {e}")
            return {'columns': [], 'rows': [], 'error': f"Unexpected error: {str(e)}"}
        
    def _classify_intent(self, user_query: str) -> str:
        """Classifies user intent to decide the response strategy."""
        query_lower = user_query.lower().strip()
        
        # --- Predefined commands ---
        if query_lower in ['whats my training status for today?', 'training status']:
            return "DAILY_SUMMARY"

        # --- Simple keyword-based classification for data queries ---
        data_keywords = [
            'what was', 'how many', 'how far', 'show me', 'what is my',
            'average', 'total', 'longest', 'shortest', 'last week', 
            'this month', 'compare', 'ctl', 'tsb', 'atl', 'fatigue', 'fitness'
        ]
        if any(keyword in query_lower for keyword in data_keywords):
            self.logger.info("Intent classified as: DATA_QUERY")
            return "DATA_QUERY"
        
        # --- Default to general conversation ---
        self.logger.info("Intent classified as: GENERAL_CHAT")
        return "GENERAL_CHAT"