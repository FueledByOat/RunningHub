# config.py (project root)
import logging
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="secrets.env")

class Config:
    DB_PATH = os.getenv('DATABASE', 'default.db')
    CACHE_TTL = int(os.getenv('CACHE_TTL', '300'))
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    VIDEO_FOLDER = os.getenv('VIDEO_FOLDER', 'videos')
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
    ALLOWED_EXTENSIONS = os.getenv('ALLOWED_EXTENSIONS', 'mp4')
    FLASK_SECRET_KEY = os.getenv('FLASK_SECRET_KEY', "broken_key")
    HF_TOKEN = os.getenv('HF_TOKEN', "broken_hf_key")

# Set up logging configuration
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='running_hub.log', 
    filemode='a')