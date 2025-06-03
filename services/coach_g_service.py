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
from utils import db_utils, format_utils, exception_utils, language_model_utils
from config import LanguageModelConfig, Config

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
        try:
            self.coach_g = language_model_utils.LanguageModel()
            self.tokenizer = self.coach_g.tokenizer
            self.logger.info("CoachGService initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize language model in service: {e}")
            raise

    def daily_training_summary(self) -> str:
        """Get daily training summary."""
        try:
            return self.coach_g.generate_daily_training_summary()
        except Exception as e:
            self.logger.error(f"Error generating daily training summary: {e}")
            return "Unable to generate training summary at this time."

    def general_reply(self, session_id: str, user_query: str, personality: str) -> str:
        """Generate a controlled coaching response with validation."""
        try:
            # Input validation
            if not session_id or not user_query.strip():
                raise ValueError("Session ID and user query are required")
            
            # Sanitize input
            user_query = self._sanitize_user_input(user_query)
            
            # Save user message
            self._save_message(session_id, "user", user_query)
            
            # Get conversation history
            history = self._get_recent_messages(
                session_id, 
                max_tokens=self.lm_config.MAX_CONTEXT_TOKENS
            )
            
            # Generate response with retries for quality
            response = self._generate_with_quality_check(user_query, personality, history)
            
            # Save assistant's response
            self._save_message(session_id, "coach", response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error in general_reply: {e}")
            return self._get_fallback_response(user_query)
    
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
    
    def _is_response_acceptable(self, response: str) -> bool:
        """Check if response meets quality standards."""
        if not response or len(response.strip()) < 5:
            return False
        
        # More lenient sentence count (1-8 sentences)
        sentence_count = len([s for s in re.split(r'[.!?]+', response) if s.strip()])
        if sentence_count < 1 or sentence_count > 8:
            return False
        
        # Relaxed repetition check (40% unique words minimum)
        words = response.lower().split()
        if len(words) > 5 and len(set(words)) < len(words) * 0.4:
            return False
        
        # Check for obvious generation artifacts
        if any(artifact in response.lower() for artifact in ['<|end|>', '[end]', 'user:', 'coach g:']):
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
                db_utils.save_message(conn, session_id, role, message)
                self.logger.debug(f"Saved {role} message for session {session_id}")
        except Exception as e:
            self.logger.error(f"Error saving conversation message: {e}")
            raise exception_utils.DatabaseError(f"Failed to save message: {e}") from e

    def _get_recent_messages(self, session_id: str, max_tokens: int = 512) -> List[Dict]:
        """Get recent conversation messages within token limit."""
        try:
            with self._get_connection() as conn:
                history = db_utils.get_recent_messages(
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
