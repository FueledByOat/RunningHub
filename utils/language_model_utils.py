# language_model_utils.py
"""Utilities for natural language to SQL query generation using Hugging Face models."""

import logging
import re
import torch
from typing import Optional, List, Dict
from functools import lru_cache

from transformers import pipeline, AutoTokenizer
from huggingface_hub import login

from utils import exception_utils, db_utils
from config import Config, LanguageModelConfig

logger = logging.getLogger(__name__)

class LanguageModel:
    """Natural language to SQL query generator using Hugging Face models."""
   
    def __init__(self):
        self._pipe = None
        self.tokenizer = None
        self.config = LanguageModelConfig()
        self._initialize_model()
   
    def _initialize_model(self):
        """Initialize the Hugging Face model pipeline."""
        try:
            # Authenticate with Hugging Face
            login(token=Config.HF_TOKEN, add_to_git_credential=False)
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_NAME)
           
            # Initialize text generation pipeline
            self._pipe = pipeline(
                "text-generation",
                model=self.config.MODEL_NAME,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="cpu",
                return_full_text=False
            )
            logger.info("Language model initialized successfully")
           
        except Exception as e:
            logger.error(f"Failed to initialize language model: {e}")
            raise exception_utils.LanguageModelError(f"Model initialization failed: {e}") from e
    
    @property
    @lru_cache(maxsize=1)
    def _schema_prompt(self) -> str:
        """Database schema prompt template."""
        return """Given the following SQL Database schema of a user's running and biking data:

activities table:
- id INTEGER PRIMARY KEY
- distance REAL -- in meters
- moving_time INTEGER -- in seconds
- elapsed_time INTEGER -- in seconds
- total_elevation_gain REAL -- in meters
- type TEXT -- can be 'Run' or 'Ride'
- workout_type INTEGER
- start_date_local TEXT
- kudos_count INTEGER
- gear_id TEXT -- foreign key to gear table
- average_speed REAL
- max_speed REAL
- average_cadence REAL
- average_watts REAL
- max_watts INTEGER
- weighted_average_watts INTEGER
- device_watts BOOLEAN
- kilojoules REAL
- average_heartrate REAL
- max_heartrate REAL
- elev_high REAL
- elev_low REAL
- import_date TEXT

gear table (contains shoe and bike data):
- gear_id TEXT PRIMARY KEY
- nickname TEXT
- resource_state INTEGER
- retired BOOLEAN
- distance INTEGER
- brand_name TEXT
- model_name TEXT
- description TEXT

Write a SQL query to return the relevant columns to answer the question:"""
    
    def generate_sql_from_natural_language(self, user_input: str, max_tokens: int = 256) -> str:
        """Generate SQL query from natural language input.
        
        Args:
            user_input: Natural language question about the data
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated SQL query response
            
        Raises:
            LanguageModelError: If generation fails
        """
        if not user_input or not user_input.strip():
            raise exception_utils.LanguageModelError("User input cannot be empty")
        
        if not self._pipe:
            raise exception_utils.LanguageModelError("Model not initialized")
        
        try:
            # Construct prompt
            full_prompt = f"{self._schema_prompt}\n\n{user_input.strip()}"
            
            messages = [{
                "role": "user", 
                "content": f"You are a SQLite Expert. {full_prompt}"
            }]
            
            # Generate response
            outputs = self._pipe(
                messages, 
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.1,  # Lower temperature for more consistent SQL
                pad_token_id=self._pipe.tokenizer.eos_token_id
            )
            
            if not outputs or not outputs[0].get("generated_text"):
                raise exception_utils.LanguageModelError("Model returned empty response")
            
            # Extract assistant response
            generated_text = outputs[0]["generated_text"]
            if isinstance(generated_text, list) and len(generated_text) > 0:
                assistant_response = generated_text[-1].get("content", "").strip()
            else:
                assistant_response = str(generated_text).strip()
            
            if not assistant_response:
                raise exception_utils.LanguageModelError("Generated response is empty")
            
            logger.info(f"Generated SQL response for query: {user_input[:50]}...")
            return assistant_response
            
        except Exception as e:
            logger.error(f"SQL generation failed for input '{user_input[:50]}...': {e}")
            raise exception_utils.LanguageModelError(f"Failed to generate SQL: {e}") from e
    
    def extract_sql_query(self, assistant_response: str) -> Optional[str]:
        """Extract SQL query from model response.
        
        Args:
            assistant_response: Raw response from language model
            
        Returns:
            Extracted SQL query or None if not found
            
        Raises:
            LanguageModelError: If extraction fails
        """
        if not assistant_response or not assistant_response.strip():
            raise exception_utils.LanguageModelError("Assistant response cannot be empty")
        
        try:
            # Try to extract SQL from code blocks
            sql_patterns = [
                r"```sql\s+(.*?)\s*```",  # SQL code blocks
                r"```\s*(SELECT.*?);?\s*```",  # Generic code blocks with SELECT
                r"(SELECT.*?);",  # Direct SQL statements
            ]
            
            for pattern in sql_patterns:
                match = re.search(pattern, assistant_response, re.DOTALL | re.IGNORECASE)
                if match:
                    query = match.group(1).strip()
                    # Clean up the query
                    query = re.sub(r'\s+', ' ', query)  # Normalize whitespace
                    if not query.endswith(';'):
                        query += ';'
                    
                    logger.info("Successfully extracted SQL query")
                    return query
            
            logger.warning(f"No SQL query found in response: {assistant_response[:100]}...")
            return None
            
        except Exception as e:
            logger.error(f"SQL extraction failed: {e}")
            raise exception_utils.LanguageModelError(f"Failed to extract SQL: {e}") from e

    def generate_daily_training_summary(self) -> str:
        """Generate a daily training summary based on latest metrics."""
        try:
            # Get training data
            with db_utils.get_db_connection(Config.DB_PATH) as conn:
                training_metrics = db_utils.get_latest_daily_training_metrics(conn=conn)
            
            if not training_metrics:
                logger.warning("No training metrics found for daily summary")
                return "No training data available for today."
            
            # Build prompt with training data
            metrics = training_metrics[0]
            prompt = self._build_summary_prompt(metrics)
            
            # Generate response
            response = self._pipe(prompt, max_new_tokens=100)[0]["generated_text"]
            logger.info("Daily training summary generated successfully")
            return response.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate training summary: {e}")
            raise exception_utils.LanguageModelError(f"Training summary generation failed: {e}") from e
    
    def _build_summary_prompt(self, metrics: Dict) -> str:
        """Build the prompt for training summary generation."""
        return f"""
            You are a running coach assistant, Coach G. Summarize today's training load.

            Date: {metrics['date']}
            Total TSS: {round(metrics['total_tss'], 1)}
            CTL (Chronic Training Load): {round(metrics['ctl'], 1)}
            ATL (Acute Training Load): {round(metrics['atl'], 1)}
            TSB (Training Stress Balance): {round(metrics['tsb'], 1)}

            Guidelines:
            - If TSS > 100, mention heavy training day
            - If ATL >> CTL, caution about building fatigue
            - If TSB < 0, suggest user may be accumulating fatigue
            - If TSB > 0, highlight good day for hard training/racing
            - Use concise, motivational tone

            Respond with 2-4 sentences.
            """


    def generate_general_coach_g_reply(self, user_query: str, personality: str, history: List[Dict] = None) -> str:
        """Generate a controlled coaching response."""
        try:
            # Build conversation context
            context = self._build_conversation_context(history or [])
            
            # Get personality prompt
            personality_prompt = self.config.PERSONALITY_TEMPLATES.get(
                personality, self.config.PERSONALITY_TEMPLATES['motivational']
            )
            
            # Create structured prompt
            prompt = self._build_chat_prompt(personality_prompt, context, user_query)
            
            # Validate prompt is a string
            if not isinstance(prompt, str):
                logger.error(f"Prompt is not a string: {type(prompt)}")
                raise ValueError(f"Prompt must be string, got {type(prompt)}")
            
            # Log the prompt for debugging
            logger.debug(f"Generated prompt: {prompt[:200]}...")
            
            # Generate with better parameters
            generation_params = {
                'max_new_tokens': self.config.MAX_NEW_TOKENS,
                'temperature': self.config.TEMPERATURE,
                'top_p': self.config.TOP_P,
                'repetition_penalty': self.config.REPETITION_PENALTY,
                'do_sample': True,
                'pad_token_id': self.tokenizer.eos_token_id,
            }
            
            # Safe pipeline call
            try:
                pipeline_output = self._pipe(prompt, **generation_params)
                if not pipeline_output or len(pipeline_output) == 0:
                    raise ValueError("Pipeline returned empty output")
                
                raw_output = pipeline_output[0]["generated_text"].strip()
                
            except Exception as pipe_error:
                logger.error(f"Pipeline error: {pipe_error}")
                logger.error(f"Prompt type: {type(prompt)}")
                logger.error(f"Prompt content: {repr(prompt[:500])}")
                raise
            
            # Clean and format response
            response = self._clean_and_format_response(raw_output)
            
            logger.info(f"Generated response for query: {user_query[:50]}...")
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate coach reply: {e}")
            raise exception_utils.LanguageModelError(f"Coach reply generation failed: {e}") from e
    
    def _build_chat_prompt(self, personality_prompt: str, context: str, user_query: str) -> str:
        """Build a structured chat prompt."""
        
        # Ensure all inputs are strings
        personality_prompt = str(personality_prompt) if personality_prompt else ""
        context = str(context) if context else ""
        user_query = str(user_query) if user_query else ""
        
        prompt_parts = [
            f"You are Coach G {personality_prompt}",
            "",
            "Previous conversation:" if context else "",
            context if context else "",
            "",
            f"User: {user_query}",
            f"Coach G (respond in {self.config.RESPONSE_MIN_SENTENCES}-{self.config.RESPONSE_MAX_SENTENCES} complete sentences):"
        ]
        
        # Remove empty parts and join
        filtered_parts = [part for part in prompt_parts if part.strip()]
        final_prompt = "\n".join(filtered_parts)
        
        # Validate final prompt
        if not isinstance(final_prompt, str):
            raise ValueError(f"Final prompt is not a string: {type(final_prompt)}")
        
        return final_prompt
    
    def _clean_and_format_response(self, raw_output: str) -> str:
        """Clean and format the model's response."""
        # Extract coach response
        response = self._extract_coach_reply(raw_output)
        
        # Clean up common issues
        response = self._clean_response_text(response)
        
        # Ensure reasonable length
        response = self._enforce_response_bounds(response)
        
        return response
    
    def _extract_coach_reply(self, generated_text: str) -> str:
        """Extract Coach G's response, handling various output formats."""
        # Remove prompt echo if present
        if "Coach G:" in generated_text:
            parts = generated_text.split("Coach G:", 1)
            if len(parts) > 1:
                generated_text = parts[1]
        
        # Stop at next speaker or conversation markers
        stop_patterns = [
            r"\n(?:User|Coach G):",
            r"\n\n(?:User|Coach G):",
            r"\n---",
            r"\n\*\*\*",
            # Common model artifacts
            r"\n\[END\]",
            r"\n<\|end\|>",
            r"\n\[/INST\]"
        ]
        
        for pattern in stop_patterns:
            match = re.search(pattern, generated_text, re.IGNORECASE)
            if match:
                generated_text = generated_text[:match.start()]
                break
        
        return generated_text.strip()
    
    def _clean_response_text(self, text: str) -> str:
        """Clean up common text generation artifacts."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove repeated phrases (simple detection)
        sentences = text.split('. ')
        seen = set()
        cleaned_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence.lower() not in seen:
                seen.add(sentence.lower())
                cleaned_sentences.append(sentence)
        
        # Rejoin sentences
        cleaned = '. '.join(cleaned_sentences)
        
        # Ensure proper sentence ending
        if cleaned and not cleaned.endswith(('.', '!', '?')):
            cleaned += '.'
        
        return cleaned
    
    def _enforce_response_bounds(self, response: str) -> str:
        """Ensure response meets length requirements."""
        sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
        
        # Too few sentences - this is often a sign of truncation
        if len(sentences) < self.config.RESPONSE_MIN_SENTENCES:
            # Try to encourage complete thoughts
            if response and not response.endswith(('.', '!', '?')):
                # Truncated mid-sentence, add fallback
                response += " Let me know if you'd like more specific guidance!"
        
        # Too many sentences - trim excess
        elif len(sentences) > self.config.RESPONSE_MAX_SENTENCES:
            kept_sentences = sentences[:self.config.RESPONSE_MAX_SENTENCES]
            response = '. '.join(kept_sentences)
            if not response.endswith(('.', '!', '?')):
                response += '.'
        
        return response
    
    def _build_conversation_context(self, history: List[Dict]) -> str:
        """Build conversation context with token awareness."""
        if not history:
            return ""
        
        context_lines = []
        total_tokens = 0
        max_tokens = self.config.MAX_CONTEXT_TOKENS // 2
        
        recent_messages = history[-self.config.MAX_CONTEXT_MESSAGES:]
        
        for msg in recent_messages:
            # Safely extract message content
            message_content = msg.get('message', '')
            
            # Ensure message_content is a string
            if not isinstance(message_content, str):
                message_content = str(message_content) if message_content else ''
            
            # Clean the message content
            message_content = message_content.strip()
            if not message_content:
                continue
                
            role_label = "User" if msg.get("role") == "user" else "Coach G"
            line = f"{role_label}: {message_content}"
            
            # Safe token counting with proper string input
            try:
                line_tokens = len(self.tokenizer.encode(line))  # Pass string directly
            except Exception as e:
                self.logger.warning(f"Tokenizer failed for line: {repr(line)}, error: {e}")
                line_tokens = len(line.split()) * 1.3  # Fallback estimate
            
            if total_tokens + line_tokens > max_tokens:
                break
                
            context_lines.append(line)
            total_tokens += line_tokens
        
        return "\n".join(context_lines)

# Global instance
_sql_generator = None

def get_sql_generator() -> LanguageModel:
    """Get or create global SQL generator instance."""
    global _sql_generator
    if _sql_generator is None:
        _sql_generator = LanguageModel()
    return _sql_generator

def generate_sql_from_natural_language(user_input: str) -> str:
    """Generate SQL query from natural language input.
    
    Args:
        user_input: Natural language question
        
    Returns:
        Generated SQL response
    """
    generator = get_sql_generator()
    return generator.generate_sql_from_natural_language(user_input)

def extract_sql_query(assistant_response: str) -> Optional[str]:
    """Extract SQL query from assistant response.
    
    Args:
        assistant_response: Raw model response
        
    Returns:
        Extracted SQL query or None
    """
    generator = get_sql_generator()
    return generator.extract_sql_query(assistant_response)