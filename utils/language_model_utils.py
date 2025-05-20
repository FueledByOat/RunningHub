# language_model_utils.py
"""Storing functions relating to querying a language model"""

import torch
# from accelerate import disk_offload # trying to manage memory
from transformers import pipeline
from huggingface_hub import login
import os
import re

# load_dotenv('secrets.env') # it's magically working now? 
hf_secret = os.getenv("HF_TOKEN") # access a token registered on huggingface to allow use of gated model
login(token = hf_secret, add_to_git_credential = False) # performs cli login using token above

# Defines the model and pipeline to be used for text generation
pipe = pipeline(
    "text-generation",
    model="google/gemma-2-2b-it",
    model_kwargs={"torch_dtype": torch.bfloat16,},
    device_map="cpu", # other options exist here but cpu seems to work and avoids GPU per our goals
)
def generate_sql_from_natural_language(user_input):
    manual_prompt_template = """Given the following SQL Database schema of a user's running and biking data:
            activities table:
            id INTEGER PRIMARY KEY,
            distance REAL, -- in meters
            moving_time INTEGER, -- in seconds
            elapsed_time INTEGER, -- in seconds
            total_elevation_gain REAL, -- in meters
            type TEXT, -- can be Run or Ride
            workout_type INTEGER,
            start_date_local TEXT,
            kudos_count INTEGER,
            gear_id TEXT, -- foreign key to gear table
            average_speed REAL,
            max_speed REAL,
            average_cadence REAL,
            average_watts REAL,
            max_watts INTEGER,
            weighted_average_watts INTEGER,
            device_watts BOOLEAN,
            kilojoules REAL,
            average_heartrate REAL,
            max_heartrate REAL,
            elev_high REAL,
            elev_low REAL,
            import_date TEXT

            gear table (contains shoe and bike data):
            gear_id TEXT PRIMARY KEY,
            nickname TEXT,
            resource_state INTEGER,
            retired BOOLEAN,
            distance INTEGER,
            brand_name TEXT,
            model_name TEXT,      
            description TEXT,
            
            Write a SQL query to return the relevant columns to answer the question: """
    sql_query = manual_prompt_template + str(user_input)

    messages = [
    {"role": "user", "content": "You are SQLite Expert. " + sql_query},
    ]   
    outputs = pipe(messages, max_new_tokens=256)
    assistant_response = outputs[0]["generated_text"][-1]["content"].strip()

    return assistant_response

def extract_sql_query(assistant_response):
    match = re.search(r"```sql\s+(.*?;)", assistant_response, re.DOTALL)
    query = match.group(1)
    return query