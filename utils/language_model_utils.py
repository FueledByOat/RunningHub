# language_model_utils.py
"""Utilities for natural language to SQL query generation using Hugging Face models."""

import logging
import re
import torch
from typing import Optional, List, Dict, Any
from functools import lru_cache

from transformers import pipeline, AutoTokenizer
from huggingface_hub import login, InferenceClient

from utils import exception_utils
from config import Config, LanguageModelConfig
from utils.db import db_utils
from utils.db import language_db_utils

logger = logging.getLogger(__name__)

class LanguageModel:
    """Handles interactions with the Hugging Face language model."""
    
    def __init__(self):
        self.config = LanguageModelConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.LOCAL_MODEL_NAME)
        self._pipe = None
        if not self.config.USE_REMOTE_MODEL:
            self._initialize_local_model()

    def _initialize_local_model(self):
        """Initializes the local Hugging Face model pipeline."""
        try:
            login(token=Config.HF_TOKEN, add_to_git_credential=False)
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Initializing model on device: {device}")
            
            self._pipe = pipeline(
                "text-generation",
                model=self.config.LOCAL_MODEL_NAME,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="cpu",
                return_full_text=False
            )
            logger.info("Language model initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize language model: {e}")
            raise exception_utils.LanguageModelError(f"Model initialization failed: {e}")

    def generate_general_coach_g_reply(self, user_query: str, personality: str, history: List[Dict]) -> str:
        """
        Generates a general, long-form reply from Coach G.
        """
        context = self._build_conversation_context(history)
        personality_prompt = self.config.PERSONALITY_TEMPLATES.get(personality, self.config.PERSONALITY_TEMPLATES['motivational'])

        # --- REFINED PROMPT ---
        # Provides clearer, more explicit instructions to the model.
        prompt = f"""You are Coach G, a {personality_prompt} running coach. Your task is to provide a supportive and insightful response to the user.
Follow these rules strictly:
1.  Your reply must be a thoughtful, long-form response of 2-4 complete sentences.
2.  Do NOT mention that you are an AI, a language model, or a bot.
3.  Directly address the user's message in your reply.
4.  Do not output anything other than Coach G's response.

Here is the conversation history:
{context}
User: {user_query}

Now, provide only Coach G's response.
Coach G:"""

        try:
            if self.config.USE_REMOTE_MODEL:
                pass 
            
            if not self._pipe:
                raise exception_utils.LanguageModelError("Local model is not initialized.")

            outputs = self._pipe(
                prompt,
                max_new_tokens=self.config.MAX_NEW_TOKENS,
                do_sample=True,
                temperature=self.config.TEMPERATURE,
                top_p=self.config.TOP_P,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            raw_response = outputs[0]['generated_text']
            # Use the new, more robust cleaning function
            return self._clean_response(raw_response)

        except Exception as e:
            logger.error(f"Failed to generate general reply: {e}")
            raise exception_utils.LanguageModelError(f"Failed to generate reply: {e}")

    def format_daily_training_summary(self, metrics: Dict) -> str:
        """
        Formats the daily training metrics into a markdown-ready string.
        """
        tsb = metrics.get('tsb', 0)
        interpretation = ""
        if tsb > 5:
            interpretation = "You should be feeling fresh and ready for a quality workout today. It's a great day to push a little harder!"
        elif -10 <= tsb <= 5:
            interpretation = "You're in a good training zone, carrying some fitness without too much fatigue. A solid effort today is a great plan."
        else:
            interpretation = "You're likely carrying some fatigue from your recent training. It might be a good day to focus on recovery or a lighter session."

        return (
            f"### Your Training Status for {metrics['date']}\n\n"
            f"**CTL (Fitness):** {metrics.get('ctl', 0):.1f}\n\n"
            f"**ATL (Fatigue):** {metrics.get('atl', 0):.1f}\n\n"
            f"**TSB (Freshness):** {tsb:.1f}\n\n"
            f"**Coach G's Advice:** {interpretation}"
        )

    def _build_conversation_context(self, history: List[Dict]) -> str:
        """Builds a string of the recent conversation history."""
        if not history:
            return "(No history yet)"
        
        context_lines = []
        for msg in reversed(history[-self.config.MAX_CONTEXT_MESSAGES:]):
            role = "User" if msg.get("role") == "user" else "Coach G"
            context_lines.append(f"{role}: {msg.get('message', '')}")
        
        return "\n".join(reversed(context_lines))

    def _clean_response(self, text: str) -> str:
        """
        Cleans the raw model output to ensure it's a coherent and safe reply from Coach G.
        """
        # Stop the response at the next conversational turn or end-of-text token.
        stop_phrases = ['\nUser:', '\nCoach G:', '<|endoftext|>', '[INST]']
        for phrase in stop_phrases:
            stop_index = text.find(phrase)
            if stop_index != -1:
                text = text[:stop_index]

        # Remove JSON/code block artifacts
        text = re.sub(r'```(json|sql|)?', '', text)
        text = text.replace('```', '')

        # Standardize whitespace and remove leading/trailing junk
        text = re.sub(r'\s+', ' ', text).strip()

        # If the response is empty or too short after cleaning, return a safe default.
        if not text or len(text) < 20:
            logger.warning(f"Cleaned response was too short. Returning a default. Original: '{text}'")
            return "That's a great question. Consistency is key in training. Keep up the great work and let me know what else is on your mind!"

        return text
        
    def _generate_report_summary_response(self, prompt: str, max_tokens: int = 400) -> str:
        """
        Generates a detailed response specifically for data-driven report summaries,
        using a higher token limit.

        Args:
            prompt (str): The detailed prompt containing the runner's data.
            max_tokens (int): The maximum number of new tokens to generate. Defaults to 512.

        Returns:
            str: The raw, potentially multi-line markdown string from the model.
        """
        try:
            # KEY CHANGE: This function uses its own 'max_tokens' parameter with a much
            # higher default (512), ignoring the more restrictive global config.
            if self.config.USE_REMOTE_MODEL:
                # Assuming your remote generation function can accept a max_tokens override
                raw_output = self._generate_remote_response(prompt, "", max_tokens)
            else:
                if not self._pipe:
                    raise exception_utils.LanguageModelError("Local model not initialized.")

                generation_params = {
                    'max_new_tokens': max_tokens, # Use the new, larger token limit
                    'temperature': self.config.TEMPERATURE,
                    'top_p': self.config.TOP_P,
                    'repetition_penalty': self.config.REPETITION_PENALTY,
                    'do_sample': True,
                    'pad_token_id': self.tokenizer.eos_token_id,
                }

                pipeline_output = self._pipe(prompt, **generation_params)
                if not pipeline_output or not pipeline_output[0].get("generated_text"):
                    raise exception_utils.LanguageModelError("Local model returned empty response.")

                # KEY CHANGE: We want the full text, including the prompt, to parse it cleanly.
                full_output = pipeline_output[0]["generated_text"]
                
                # The model often includes the prompt in its response. We need to remove it.
                # This ensures we only return the newly generated text.
                if full_output.startswith(prompt):
                    raw_output = full_output[len(prompt):].strip()
                else:
                    raw_output = full_output.strip()


            # KEY CHANGE: We return the raw markdown here. The conversion to HTML
            # should happen in the text_generation.py file, which is responsible
            # for the final formatting of the report content.
            return raw_output

        except Exception as e:
            logger.error(f"Report summary generation failed for prompt '{prompt[:100]}...': {e}", exc_info=True)
            raise exception_utils.LanguageModelError(f"Report summary generation failed: {e}") from e

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