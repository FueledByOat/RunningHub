# language_model_utils.py
"""Utilities for natural language to SQL query generation using Hugging Face models."""

import logging
import os # Import os to get the API key
import torch
import re
from typing import Optional, List, Dict, Any, Union

import litellm
from transformers import pipeline, AutoTokenizer
from huggingface_hub import login


from utils import exception_utils
from config import Config, LanguageModelConfig 

logger = logging.getLogger(__name__)

os.environ["TOGETHER_API_KEY"] = Config.TOGETHER_API_KEY or ""

class LanguageModel:
    """
    Handles interactions with language models.
    This class acts as a factory and wrapper, initializing and providing a
    unified interface for either a local Hugging Face model or a remote API-based model.
    """
    
    def __init__(self):
        self.config = LanguageModelConfig()
        try:
            logger.info(f"Loading tokenizer: {self.config.LOCAL_MODEL_NAME}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.LOCAL_MODEL_NAME)
        except Exception as e:
            logger.error(f"FATAL: Could not load tokenizer {self.config.LOCAL_MODEL_NAME}. Error: {e}")
            raise

        if self.config.USE_REMOTE_MODEL:
            # --- REFINED: No client initialization needed for litellm ---
            logger.info(f"Remote mode enabled. Will use litellm to call {self.config.REMOTE_MODEL_NAME}")
        else:
            self._initialize_local_model()

    def _initialize_local_model(self):
        """Initializes the local Hugging Face model pipeline."""
        logger.info("Initializing LOCAL language model.")
        try:
            login(token=Config.HF_TOKEN, add_to_git_credential=False)
            
            logger.info(f"Loading local tokenizer for: {self.config.LOCAL_MODEL_NAME}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.LOCAL_MODEL_NAME)
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Initializing model on device: {device}")
            
            self.model_client = pipeline(
                "text-generation",
                model=self.config.LOCAL_MODEL_NAME,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
                return_full_text=False
            )
            logger.info("Local language model initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize local language model: {e}")
            raise exception_utils.LanguageModelError(f"Local model initialization failed: {e}")

    def generate(self, prompt_or_messages: Union[str, List[Dict]], **kwargs) -> str:
        """
        Generates and cleans a response using the initialized model (local or remote).
        """
        raw_response = ""
        try:
            if self.config.USE_REMOTE_MODEL:
                # --- REFINED: Direct call to litellm.completion ---
                if not isinstance(prompt_or_messages, list):
                    raise TypeError(f"Remote model expects a List[Dict], but received {type(prompt_or_messages)}.")
                
                logger.debug(f"Calling litellm with model '{self.config.REMOTE_MODEL_NAME}'")
                response = litellm.completion(
                    model=self.config.REMOTE_MODEL_NAME,
                    messages=prompt_or_messages,
                    temperature=self.config.TEMPERATURE,
                    max_tokens=self.config.MAX_NEW_TOKENS,
                    **kwargs
                )
                # Extract the message content from the response
                raw_response = response.choices[0].message.content

            else:
                # Local model path remains unchanged
                if self.model_client is None:
                     raise exception_utils.LanguageModelError("Local model is not initialized.")
                if not isinstance(prompt_or_messages, str):
                    raise TypeError(f"Local model expects a string prompt, but received {type(prompt_or_messages)}.")
                
                # ... (local model generation logic is unchanged)
                generation_params = {
                    'max_new_tokens': kwargs.get('max_new_tokens', self.config.MAX_NEW_TOKENS),
                    'temperature': self.config.TEMPERATURE,
                    'top_p': self.config.TOP_P,
                    'eos_token_id': [self.tokenizer.eos_token_id] + self.tokenizer.convert_tokens_to_ids(self.config.STOP_TOKENS),
                    'do_sample': True,
                }
                outputs = self.model_client(prompt_or_messages, **generation_params)
                raw_response = outputs[0]['generated_text']

            return self._clean_response(raw_response)

        except Exception as e:
            logger.error(f"Error during model generation: {e}", exc_info=True)
            # Try to get more info from litellm exceptions
            if hasattr(e, 'response'):
                logger.error(f"LiteLLM API Response Error: {e.response.text}")
            return "Sorry, I encountered an error while generating a response."

    # --- This function is now perfect. It prepares the data correctly. ---
    def generate_general_coach_g_reply(self, user_query: str, personality: str, history: List[Dict]) -> str:
        """
        Generates a standard conversational reply, preparing the data
        in the correct format for either a local or remote model.
        """
        personality_prompt = self.config.PERSONALITY_TEMPLATES.get(personality, self.config.PERSONALITY_TEMPLATES['motivational'])
        system_message = f"You are Coach G, a {personality_prompt} running coach..." # (abbreviated)

        if self.config.USE_REMOTE_MODEL:
            # REMOTE PATH: Build a list of standard dictionaries. litellm handles this perfectly.
            messages = [{'role': 'system', 'content': system_message}]
            for msg in history:
                messages.append({'role': msg['role'], 'content': msg.get('message', '')})
            messages.append({'role': 'user', 'content': user_query})
            return self.generate(messages)
        else:
            # LOCAL PATH: Build a single formatted string.
            context = self._build_conversation_context(history)
            prompt = f"""{system_message}\n\nHere is the conversation history (oldest to newest):\n{context}\n\nUser: {user_query}\n\nNow, provide only Coach G's response.\nCoach G:"""
            return self.generate(prompt)

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
        
    def generate_report_summary_response(self, prompt: str, max_tokens: int = 400) -> str:
        """
        Generates a detailed response for data-driven reports, preparing the
        prompt in the correct format for either local or remote models.
        """
        logger.info(f"Generating report summary with max_tokens={max_tokens}.")
        
        if self.config.USE_REMOTE_MODEL:
            # --- REMOTE PATH: Wrap the prompt string in a message list ---
            # For a direct instruction like this, we treat it as a single 'user' message.
            messages = [
                {'role': 'user', 'content': prompt}
            ]
            # Call the base generate() method with the correctly formatted list
            return self.generate(messages, max_new_tokens=max_tokens)
        else:
            # --- LOCAL PATH: Pass the prompt string directly ---
            # The local pipeline expects a single string, so this is correct.
            return self.generate(prompt, max_new_tokens=max_tokens)