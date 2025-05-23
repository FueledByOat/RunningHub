# language_model_utils.py
"""Utilities for natural language to SQL query generation using Hugging Face models."""

import logging
import re
import torch
from typing import Optional
from functools import lru_cache

from transformers import pipeline
from huggingface_hub import login

from utils import exception_utils
from config import Config

logger = logging.getLogger(__name__)

class SQLGenerator:
    """Natural language to SQL query generator using Hugging Face models."""
    
    def __init__(self):
        self._pipe = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Hugging Face model pipeline."""
        try:
            # Authenticate with Hugging Face
            login(token=Config.HF_TOKEN, add_to_git_credential=False)
            
            # Initialize text generation pipeline
            self._pipe = pipeline(
                "text-generation",
                model="google/gemma-2-2b-it",
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="cpu",
                return_full_text=False  # Only return generated text
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


# Global instance
_sql_generator = None

def get_sql_generator() -> SQLGenerator:
    """Get or create global SQL generator instance."""
    global _sql_generator
    if _sql_generator is None:
        _sql_generator = SQLGenerator()
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