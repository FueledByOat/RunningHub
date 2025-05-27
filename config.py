# config.py (project root)
import logging
import os
from typing import Optional
from dotenv import load_dotenv
load_dotenv(dotenv_path="secrets.env")

class Config:
    """Configuration class for environment variables and settings."""
    DB_PATH: Optional[str] = os.getenv('DATABASE', 'strava_data.db')
    DB_PATH_RUNSTRONG: Optional[str] = os.getenv('RUNSTRONG_DATABASE', 'runstrong.db')
    CACHE_TTL = int(os.getenv('CACHE_TTL', '300'))
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    VIDEO_FOLDER = os.getenv('VIDEO_FOLDER', 'videos')
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
    ALLOWED_EXTENSIONS = os.getenv('ALLOWED_EXTENSIONS', 'mp4')
    FLASK_SECRET_KEY = os.getenv('FLASK_SECRET_KEY', "broken_key")
    # Huggingface token
    HF_TOKEN = os.getenv('HF_TOKEN', "broken_hf_key")
    
    # Strava API credentials
    CLIENT_ID: Optional[str] = os.getenv("CLIENT_ID")
    CLIENT_SECRET: Optional[str] = os.getenv("CLIENT_SECRET")
    REFRESH_TOKEN: Optional[str] = os.getenv("REFRESH_TOKEN")
    ACCESS_TOKEN: Optional[str] = os.getenv("ACCESS_TOKEN")
    # API rate limiting settings
    DEFAULT_RATE_LIMIT_DELAY: int = 1  # seconds between API calls
    MAX_ACTIVITIES_PER_REQUEST: int = 30

    @classmethod
    def validate_required_config(cls) -> bool:
        """
        Validate that all required configuration is present.
        
        Returns:
            bool: True if all required config is present, False otherwise
        """
        required_vars = [
            'DATABASE', 'CLIENT_ID', 'CLIENT_SECRET', 'REFRESH_TOKEN'
        ]
        
        missing_vars = [
            var for var in required_vars 
            if not getattr(cls, var) or getattr(cls, var) is None
        ]
        
        if missing_vars:
            print(f"Missing required environment variables: {missing_vars}")
            return False
            
        return True

# Set up logging configuration
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='logs/running_hub.log', 
    filemode='a')