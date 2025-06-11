# strava_data_pipeline.py
"""
Main execution file for Strava API data pipeline.
Handles token refresh, activity fetching, and data storage.
"""

import os
import logging
import requests
import time
import importlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dotenv import load_dotenv
import utils.strava_utils as strava_utils
import config


class StravaDataPipeline:
    """Main pipeline class for managing Strava data extraction and storage."""
    
    def __init__(self):
        """Initialize the pipeline with configuration and logging."""
        self._setup_logging()
        self._load_config()
        
    def _setup_logging(self) -> None:
        """Configure logging with proper formatting and file handling."""
        logging.basicConfig(
            level=logging.INFO,  # Changed from DEBUG to INFO for cleaner logs
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            # handlers=[
            #     logging.FileHandler('strava_pipeline.log', mode='a'),
            #     logging.StreamHandler()  # Also log to console
            # ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _load_config(self) -> None:
        """Load environment variables and validate configuration."""
        load_dotenv(dotenv_path="secrets.env", override=True)
        
        # Validate required environment variables
        required_vars = ['CLIENT_ID', 'CLIENT_SECRET', 'REFRESH_TOKEN', 'DB_PATH']
        missing_vars = [var for var in required_vars if not getattr(config.Config, var)]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
            
        self.client_id = config.Config.CLIENT_ID
        self.client_secret = config.Config.CLIENT_SECRET
        self.refresh_token = config.Config.REFRESH_TOKEN
        self.db_path = config.Config.DB_PATH
        self.access_token = None
        
    def refresh_access_token(self) -> bool:
        """
        Refresh the Strava access token.
        
        Returns:
            bool: True if refresh successful, False otherwise
        """
        try:
            self.logger.info("Refreshing access token...")
            strava_utils.refresh_access_token(
                self.client_id,
                self.client_secret,
                self.refresh_token
            )
            # Reload config after token refresh
            load_dotenv(dotenv_path="secrets.env", override=True)
            importlib.reload(config)
            self._load_config()
            self.access_token = config.Config.ACCESS_TOKEN
            self.logger.info("Access token refreshed successfully")
            return True
            
        except Exception as e:
            self.logger.critical(f"Access token refresh failed: {e}")
            return False
    
    def fetch_and_store_activities(self) -> List[Dict]:
        """
        Fetch new activities from Strava API and store them in DB_PATH.
        
        Returns:
            List[Dict]: List of fetched activities
        """
        try:
            # Get the timestamp of the latest imported activity
            latest_ts = strava_utils.latest_activity_import_date(self.db_path)
            # latest_ts = False
            self.logger.info(f"Latest activity import timestamp: {latest_ts}")
            
            # Calculate date range for fetching (last 4 days from latest import)
            if latest_ts:
                after_ts = latest_ts
                before_ts = latest_ts + (86400 * 4)  # 4 days in seconds
            else:
                # If no previous imports, fetch last 2 days
                after_ts = int((datetime.now() - timedelta(days=2)).timestamp())
                before_ts = int(datetime.now().timestamp())

            # Fetch activities from Strava API
            activities = strava_utils.get_activities(
                self.access_token,
                after=after_ts,
                before=before_ts,
                per_page=30
            )
            
            if not activities:
                self.logger.info("No new activities found")
                return []
            
            self.logger.info(f"Fetched {len(activities)} activities")
            
            # Store activities in DB_PATH
            strava_utils.insert_activities_batch(activities, self.db_path)
            self.logger.info(f"Successfully stored {len(activities)} activities in DB_PATH")
            
            return activities
            
        except Exception as e:
            self.logger.error(f"Error fetching and storing activities: {e}")
            return []
    
    def fetch_and_store_streams(self, activities: List[Dict]) -> None:
        """
        Fetch stream data for activities and store in DB_PATH.
        
        Args:
            activities: List of activity dictionaries
        """
        if not activities:
            self.logger.info("No activities provided for stream fetching")
            return
            
        self.logger.info(f"Starting stream data fetch for {len(activities)} activities")
        successful_streams = 0
        
        for i, activity in enumerate(activities):
            activity_id = activity.get('id')
            if not activity_id:
                self.logger.warning("Activity missing ID, skipping stream fetch")
                continue
                
            try:
                # Fetch stream data from Strava API
                stream = strava_utils.get_streams(self.access_token, int(activity_id))
                
                if not stream:
                    self.logger.warning(f"No stream data available for activity {activity_id}")
                    continue
                
                # Store stream data in DB_PATH
                strava_utils.insert_stream_data(int(activity_id), stream, self.db_path)
                successful_streams += 1
                
                self.logger.debug(f"Successfully processed streams for activity {activity_id}")
                
            except requests.exceptions.HTTPError as e:
                if "429" in str(e):
                    self.logger.warning("Rate limit reached, stopping stream fetch")
                    break
                elif "404" in str(e):
                    self.logger.warning(f"Stream data not found for activity {activity_id}")
                    continue
                else:
                    self.logger.error(f"HTTP error for activity {activity_id}: {e}")
                    continue
                    
            except Exception as e:
                self.logger.error(f"Unexpected error processing activity {activity_id}: {e}")
                continue
            
            # Rate limiting - be respectful to Strava's API
            time.sleep(1)
            
            # Progress logging for long operations
            if (i + 1) % 10 == 0:
                self.logger.info(f"Processed {i + 1}/{len(activities)} activities for streams")
        
        self.logger.info(f"Stream fetch complete: {successful_streams}/{len(activities)} successful")
    
    def fetch_and_store_gear(self, activities: List[Dict]) -> None:
        """
        Fetch gear information for activities and store in DB_PATH.
        
        Args:
            activities: List of activity dictionaries
        """
        if not activities:
            self.logger.info("No activities provided for gear fetching")
            return
            
        # Get unique gear IDs to avoid duplicate API calls
        gear_ids = set()
        for activity in activities:
            gear_id = activity.get('gear_id')
            if gear_id:
                gear_ids.add(gear_id)
        
        if not gear_ids:
            self.logger.info("No gear IDs found in activities")
            return
            
        self.logger.info(f"Starting gear data fetch for {len(gear_ids)} unique gear items")
        successful_gear = 0
        
        for i, gear_id in enumerate(gear_ids):
            try:
                # Fetch gear data from Strava API
                gear_data = strava_utils.get_gear(self.access_token, gear_id)
                
                if not gear_data:
                    self.logger.warning(f"No gear data available for gear ID {gear_id}")
                    continue
                
                # Store gear data in DB_PATH
                strava_utils.insert_single_gear(gear_data, self.db_path)
                successful_gear += 1
                
                self.logger.debug(f"Successfully processed gear {gear_id}")
                
            except requests.exceptions.HTTPError as e:
                if "429" in str(e):
                    self.logger.warning("Rate limit reached, stopping gear fetch")
                    break
                elif "404" in str(e):
                    self.logger.warning(f"Gear data not found for gear ID {gear_id}")
                    continue
                else:
                    self.logger.error(f"HTTP error for gear {gear_id}: {e}")
                    continue
                    
            except Exception as e:
                self.logger.error(f"Unexpected error processing gear {gear_id}: {e}")
                continue
            
            # Rate limiting
            time.sleep(1)
        
        self.logger.info(f"Gear fetch complete: {successful_gear}/{len(gear_ids)} successful")
    
    def update_daily_dashboard_metrics(self) -> None:
        """
        After any potential activity import, call to db to calculate
        metrics like ctl, atl, tsb, tss, etc.
        """
        try:
            strava_utils.update_daily_dashboard_metrics()
        except Exception as e:
            self.logger.warning("Daily Dashboard Metrics database update failed!")

        self.logger.info(f"Daily Dashboard Metrics Update is successful")

    def run_pipeline(self) -> bool:
        """
        Execute the complete data pipeline.
        
        Returns:
            bool: True if pipeline completed successfully, False otherwise
        """
        try:
            self.logger.info("=== Starting Strava Data Pipeline ===")
            
            # Step 1: Refresh access token
            if not self.refresh_access_token():
                return False
            
            # Step 2: Fetch and store activities
            activities = self.fetch_and_store_activities()
            
            # Step 3: Fetch and store stream data
            self.fetch_and_store_streams(activities)
            
            # Step 4: Fetch and store gear data
            self.fetch_and_store_gear(activities)

            # Step 5: Update daily dashboard metrics
            self.update_daily_dashboard_metrics()
            
            self.logger.info("=== Pipeline completed successfully ===")
            return True
            
        except Exception as e:
            self.logger.critical(f"Pipeline failed with error: {e}")
            return False


def main():
    """Main execution function."""
    pipeline = StravaDataPipeline()
    success = pipeline.run_pipeline()
    
    if not success:
        exit(1)


if __name__ == "__main__":
    main()