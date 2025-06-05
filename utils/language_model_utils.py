# language_model_utils.py
"""Utilities for natural language to SQL query generation using Hugging Face models."""

import logging
import re
import torch
from typing import Optional, List, Dict, Any
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
           
            # --- GPU Configuration ---
            device_to_use = "cpu" # Default
            if torch.cuda.is_available():
                device_to_use = "cuda" # For NVIDIA GPUs
                logger.info("CUDA (NVIDIA GPU) is available. Setting device_map to 'cuda'.")
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                # For Apple Silicon (M1/M2/M3 Macs)
                # Note: MPS support can sometimes be less stable or performant than CUDA for certain models/ops.
                # It also might not support all dtypes like bfloat16 as well as CUDA.
                # You might need to use torch.float16 or torch.float32 if bfloat16 causes issues on MPS.
                device_to_use = "mps"
                logger.info("MPS (Apple Silicon GPU) is available. Setting device_map to 'mps'.")
            else:
                logger.info("No CUDA or MPS GPU detected. Using CPU.")
            # --- End GPU Configuration ---

            model_dtype = torch.bfloat16
            # Enable TF32 tensor cores for matrix multiplications
            torch.set_float32_matmul_precision('high')
            # Initialize text generation pipeline
            self._pipe = pipeline(
                "text-generation",
                model=self.config.MODEL_NAME,
                model_kwargs={"torch_dtype": model_dtype}, # bfloat16 is good for Ampere+ NVIDIA GPUs. float16 for broader compatibility.
                device_map="auto", # Was "cpu"
                return_full_text=False
            )
            logger.info("Language model initialized successfully on device: {device_to_use}")
           
        except Exception as e:
            logger.error(f"Failed to initialize language model: {e}")
            raise exception_utils.LanguageModelError(f"Model initialization failed: {e}") from e
    
    @property
    @lru_cache(maxsize=1)
    def _schema_prompt(self) -> str:
        """Database schema prompt template, enhanced with descriptions and examples."""
        # --- IMPORTANT ASSUMPTION ---
        # The following definition for 'daily_training_metrics' is an example.
        # You MUST ensure this matches your actual table structure if it exists,
        # or OMIT/MODIFY it if your CTL/TSB data is handled differently (e.g., not directly queryable via SQL by the LLM).
        # If this table is NOT directly queryable by the LLM for arbitrary dates/conditions,
        # then questions like "What was my CTL on March 15th?" cannot be answered by SQL generation
        # and the LLM should be guided not to attempt such queries.
        #
        # For example, if CTL/TSB is only available for "today" via a specific function call
        # (like your existing generate_daily_training_summary), you should note that here
        # and remove examples that query daily_training_metrics for arbitrary dates.
        #
        # For this example, we are ASSUMING 'daily_training_metrics' IS queryable as defined below.
        # --- END IMPORTANT ASSUMPTION ---

        return """Given the following SQL Database schema of a user's running and biking data:

**Understanding Key Metrics for SQL Generation:**
-   When the user asks about "fitness", "form", or "endurance", consider querying the 'ctl' (Chronic Training Load) column from the 'daily_training_metrics' table.
-   When the user mentions "fatigue", "tiredness", or "recovery", consider querying the 'tsb' (Training Stress Balance) column from the 'daily_training_metrics' table. 'atl' (Acute Training Load) can also be relevant for recent fatigue.
-   "Intensity" might relate to 'average_watts', 'average_heartrate', or workout descriptions if available.
-   "Distance" or "mileage" usually refers to the 'distance' column in the 'activities' table.
-   "Time" usually refers to 'moving_time' in the 'activities' table.
-   Dates are usually in 'YYYY-MM-DD' format. For date comparisons or functions like "last week" or "this month", you will need to generate appropriate SQLite date functions (e.g., `DATE('now', '-7 days')`, `STRFTIME('%Y-%m', start_date_local)`). Remember that `start_date_local` in the `activities` table is a DATETIME string. Use `DATE(start_date_local)` or `SUBSTR(start_date_local, 1, 10)` to get the date part for comparisons with 'YYYY-MM-DD' strings.

**Database Tables:**

**activities table:** (Contains data for individual run sessions)
-   id INTEGER PRIMARY KEY
-   distance REAL -- Total distance of the activity in meters.
-   moving_time INTEGER -- Moving time in seconds (e.g., for calculating pace).
-   elapsed_time INTEGER -- Total elapsed time in seconds.
-   total_elevation_gain REAL -- Elevation gain in meters.
-   type TEXT -- Activity type,Query as `type = 'Run'`.
-   workout_type INTEGER -- Specific workout type (e.g., 0 for default run, 1 for race, 2 for long run, 3 for workout).
-   start_date_local TEXT -- Local start date and time of the activity (e.g., '2024-05-28T10:00:00Z'). For date-only operations, use `DATE(start_date_local)`.
-   kudos_count INTEGER -- Number of kudos received.
-   gear_id TEXT -- Foreign key to the gear table (links to shoes, bikes).
-   average_speed REAL -- Average speed in meters/second.
-   max_speed REAL -- Maximum speed in meters/second.
-   average_cadence REAL -- Average cadence (steps or revolutions per minute).
-   average_watts REAL -- Average power in watts (primarily for cycling or running with a power meter).
-   max_watts INTEGER -- Maximum power in watts.
-   weighted_average_watts INTEGER -- Weighted average power (cycling).
-   device_watts BOOLEAN -- True if power data came from a device.
-   kilojoules REAL -- Energy expenditure.
-   average_heartrate REAL -- Average heart rate in beats per minute.
-   max_heartrate REAL -- Maximum heart rate in beats per minute.
-   elev_high REAL -- Maximum elevation during the activity in meters.
-   elev_low REAL -- Minimum elevation during the activity in meters.
-   import_date TEXT -- Date the activity was imported.

**gear table:** (Contains details about shoes and bikes)
-   gear_id TEXT PRIMARY KEY
-   nickname TEXT -- User-given nickname for the gear (e.g., "My Favorite Runners").
-   distance INTEGER -- Total distance covered with this gear in meters.
-   brand_name TEXT
-   model_name TEXT
-   description TEXT
-   retired BOOLEAN -- Whether the gear is retired (0 for false, 1 for true).

**daily_training_metrics table:** (Contains daily aggregated training load metrics)
-   date TEXT PRIMARY KEY -- Date of the metrics (format 'YYYY-MM-DD').
-   ctl REAL -- Chronic Training Load; a rolling daily average of training stress, an indicator of fitness.
-   atl REAL -- Acute Training Load; a shorter-term rolling daily average of training stress, an indicator of recent fatigue.
-   tsb REAL -- Training Stress Balance (CTL - ATL); an indicator of form/freshness. Negative values typically mean fatigue, positive values suggest freshness.
-   total_tss REAL -- Total Training Stress Score for the day from all activities.

**Few-Shot Examples (User Question to SQL Query):**

1.  User Question: "What was my total running distance last week?"
    SQL Query: SELECT SUM(distance) / 1000.0 AS total_distance_km FROM activities WHERE type = 'Run' AND DATE(start_date_local) BETWEEN DATE('now', 'weekday 0', '-13 days') AND DATE('now', 'weekday 0', '-7 days');
    (Note: A simpler "last 7 days" SQL: `SELECT SUM(distance) / 1000.0 AS total_distance_km FROM activities WHERE type = 'Run' AND DATE(start_date_local) >= DATE('now', '-7 days');`)

2.  User Question: "Show me my 3 longest runs ever by distance."
    SQL Query: SELECT DATE(start_date_local) AS run_date, distance / 1000.0 AS distance_km, moving_time / 60.0 AS duration_minutes FROM activities WHERE type = 'Run' ORDER BY distance DESC LIMIT 3;

3.  User Question: "What shoes have I used for more than 200km and are not retired?"
    SQL Query: SELECT nickname, distance / 1000 AS distance_km FROM gear WHERE (distance / 1000.0) > 200 AND retired = 0;

4.  User Question: "What's my current fitness level according to CTL?"
    SQL Query: SELECT ctl FROM daily_training_metrics ORDER BY date DESC LIMIT 1;

5.  User Question: "Am I likely to be fatigued today based on TSB?"
    SQL Query: SELECT tsb FROM daily_training_metrics ORDER BY date DESC LIMIT 1;

6.  User Question: "What was my average heart rate for runs in May 2024?"
    SQL Query: SELECT AVG(average_heartrate) AS avg_hr_may_2024 FROM activities WHERE type = 'Run' AND STRFTIME('%Y-%m', start_date_local) = '2024-05';

You are a SQLite Expert. Based on the schema, the metric explanations, and the examples, write a concise and accurate SQL query to return the relevant columns to answer the user's question.
Only output the SQL query. Do not add any explanations or surrounding text.
The user's question is:""" # The user_input (actual question) will be appended after this by the calling function.
    
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
    
    def _clean_and_format_response(self, raw_output: str, is_data_driven: bool = False) -> str: # Added is_data_driven flag
        """Clean and format the model's response."""
        response = self._extract_coach_reply(raw_output) # _extract_coach_reply likely remains the same
        response = self._clean_response_text(response)   # _clean_response_text likely remains the same
        
        # --- Pass is_data_driven to _enforce_response_bounds ---
        response = self._enforce_response_bounds(response, is_data_driven=is_data_driven)
        # --- End Pass ---
        
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
    
    def _enforce_response_bounds(self, response: str, is_data_driven: bool = False) -> str: # Added is_data_driven flag
        """Ensure response meets length requirements."""
        sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
        
        # --- Use different sentence limits based on is_data_driven ---
        min_sentences = self.config.RESPONSE_MIN_SENTENCES_DATA_DRIVEN if is_data_driven else self.config.RESPONSE_MIN_SENTENCES
        max_sentences = self.config.RESPONSE_MAX_SENTENCES_DATA_DRIVEN if is_data_driven else self.config.RESPONSE_MAX_SENTENCES
        # --- End Use different limits ---

        # Too few sentences - this is often a sign of truncation or a very brief (maybe unhelpful) answer
        if len(sentences) < min_sentences:
            logger.debug(f"Response too short ({len(sentences)} sentences, min: {min_sentences}, data_driven: {is_data_driven}). Original: '{response[:100]}...'")
            # Try to encourage complete thoughts, especially if it seems cut off
            if response and not response.endswith(('.', '!', '?')): # Simple check for truncation
                response += " Let me know if you'd like more specific guidance!"
            # If it's short but grammatically complete, it might be okay, but log it.
        
        # Too many sentences - trim excess
        elif len(sentences) > max_sentences:
            logger.debug(f"Response too long ({len(sentences)} sentences, max: {max_sentences}, data_driven: {is_data_driven}). Trimming. Original: '{response[:100]}...'")
            kept_sentences = sentences[:max_sentences]
            response = '. '.join(kept_sentences) # Rejoin with period
            if not response.endswith(('.', '!', '?')) and response: # Ensure it ends with punctuation if not empty
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
                logger.warning(f"Tokenizer failed for line: {repr(line)}, error: {e}")
                line_tokens = len(line.split()) * 1.3  # Fallback estimate
            
            if total_tokens + line_tokens > max_tokens:
                break
                
            context_lines.append(line)
            total_tokens += line_tokens
        
        return "\n".join(context_lines)
    
    def generate_response_from_data(self, user_query: str, sql_query: str, query_results: Dict[str, Any], personality: str, history: List[Dict] = None) -> str:
        """
        Generates a natural language response based on the user's query,
        the SQL query run, and the data returned by that query.
        """
        try:
            if not self._pipe:
                raise exception_utils.LanguageModelError("Model not initialized for data response generation")

            context_str = self._build_conversation_context(history or [])
            personality_prompt_str = LanguageModelConfig.PERSONALITY_TEMPLATES.get(personality, "motivational")

            # --- Use the new helper for data_summary ---
            data_summary = self._format_query_results_for_llm(query_results)
            # --- End Use the new helper ---

            prompt = f"""You are Coach G, {personality_prompt_str}.
                Your role is to interpret data and provide coaching advice.
                The user is interacting with you via a chat interface.

                Previous conversation:
                {context_str}

                User asked: "{user_query.strip()}"

                To help answer this, I ran the following SQL query against their activity database:
                `{sql_query.strip()}`

                The query returned the following data:
                {data_summary}

                Instructions for Coach G:
                1.  Analyze the user's question and the provided data.
                2.  If the data directly answers the question, explain the data clearly and concisely. Refer to specific numbers or details from the data (e.g., "Your average heart rate was 150 bpm," or "You completed 5 runs last week for a total of 35 km.").
                3.  Provide coaching insights or context based on the data. For example:
                    -   If 'ctl' (Chronic Training Load) data is present: A higher CTL generally indicates better long-term fitness. You can comment on its current value or trend if data provides it.
                    -   If 'tsb' (Training Stress Balance) data is present: This is a key indicator of form/freshness.
                        - TSB > +15 to +25: Very fresh, good for peak performance or key workouts.
                        - TSB +5 to +15: Fresh, ready for hard training.
                        - TSB -10 to +5: Neutral zone, can train normally.
                        - TSB -10 to -25: Likely fatigued from recent training. Consider easier days or rest.
                        - TSB < -25: Highly fatigued, recovery is strongly recommended.
                    -   If 'atl' (Acute Training Load) is present: This reflects recent training stress. A rapid increase in ATL, especially if it significantly exceeds CTL, can lead to high fatigue (negative TSB).
                    -   If the user asks about "fitness" and CTL is available, use that in your response.
                    -   If the user asks about "fatigue" and TSB (or ATL/CTL relationship) is available, use that.
                4.  If the data is empty (e.g., "No specific data rows were found...") or if an error occurred fetching it (indicated in the data summary), clearly acknowledge this first (e.g., "I couldn't find specific data for that period," or "There was an issue retrieving the details."). Then, if appropriate, offer general advice related to the user's question or ask for clarification. Do not apologize excessively.
                5.  Do NOT invent data or metrics if they are not present in the "Data found" section. If the data is insufficient to answer, state that.
                6.  Maintain a helpful, encouraging, and conversational tone consistent with your chosen personality.
                7.  Avoid simply repeating the SQL query or the raw data table in your response unless you are quoting a specific value as part of your explanation.
                8.  Keep the response focused and avoid overly long paragraphs. Aim for clarity and actionable insights where possible.

                Respond directly to the user.
                Coach G:"""

            logger.debug(f"Prompt for generate_response_from_data (first 300 chars): {prompt[:300]}...")
            logger.debug(f"Data summary passed to generate_response_from_data: {data_summary}")


            generation_params = {
                'max_new_tokens': self.config.MAX_NEW_TOKENS + 50, # Allow slightly more tokens for data interpretation
                'temperature': self.config.TEMPERATURE,
                'top_p': self.config.TOP_P,
                'repetition_penalty': self.config.REPETITION_PENALTY,
                'do_sample': True,
                'pad_token_id': self.tokenizer.eos_token_id if self.tokenizer else self._pipe.tokenizer.eos_token_id,
            }
            
            pipeline_output = self._pipe(prompt, **generation_params)
            if not pipeline_output or not pipeline_output[0].get("generated_text"):
                raise exception_utils.LanguageModelError("Model returned empty response for data-driven query")
            
            raw_output = pipeline_output[0]["generated_text"].strip()
            
            # --- Pass is_data_driven=True to cleaning ---
            response = self._clean_and_format_response(raw_output, is_data_driven=True)
            # --- End Pass is_data_driven=True ---

            logger.info(f"Generated data-driven response for query: {user_query[:50]}...")
            return response

        except Exception as e:
            logger.error(f"Failed to generate response from data: {e}", exc_info=True)
            # Fallback response in case of error during data-to-text generation
            return f"I had a bit of trouble interpreting the data for your query about '{user_query[:30]}...'. Could you try rephrasing or asking something else?"

    def _format_query_results_for_llm(self, query_results: Dict[str, Any], max_rows_to_show: int = 5) -> str:
        """
        Formats SQL query results into a string suitable for an LLM prompt.
        Handles cases with errors, no data, or data present.
        """
        if not query_results: # Should not happen if _execute_generated_query returns a dict
            return "No query results were provided."

        if query_results.get('error'):
            return f"An error occurred while fetching data: {query_results['error']}"
        
        columns = query_results.get('columns')
        rows = query_results.get('rows')

        if not columns:
            return "The query returned no columns, so no data could be displayed."
        
        if not rows: # Explicitly check if rows is empty, not just None
            return "No specific data rows were found for your query."

        # Ensure rows are lists of dictionaries as expected by dict_factory
        # If rows are not dicts, this formatting will need adjustment or error handling
        if not all(isinstance(row, dict) for row in rows):
            # This might happen if dict_factory wasn't used or somehow bypassed.
            # For simplicity, we'll assume rows are dicts. Add logging if this assumption is violated.
            self.logger.warning("Query results rows are not all dictionaries. Formatting might be incorrect.")
            # Attempt a more generic formatting for non-dict rows if necessary, or return an error.
            # For now, proceed assuming they are dicts or will gracefully degrade.

        formatted_string = "Data found:\n"
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"
        formatted_string += header + "\n"
        formatted_string += separator + "\n"

        for i, row_dict in enumerate(rows):
            if i >= max_rows_to_show:
                formatted_string += f"... (and {len(rows) - max_rows_to_show} more row(s))\n"
                break
            
            # Handle if a row isn't a dict (though dict_factory should ensure it)
            if isinstance(row_dict, dict):
                row_values = [str(row_dict.get(col, 'N/A')) for col in columns]
            else: # Fallback for unexpected row format
                row_values = [str(val) for val in row_dict][:len(columns)] 
                row_values.extend(['N/A'] * (len(columns) - len(row_values)))

            formatted_string += "| " + " | ".join(row_values) + " |\n"
        
        return formatted_string.strip()

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