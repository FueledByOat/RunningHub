import sqlite3
import requests
import json
import os
from datetime import datetime
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv('secrets.env')

def create_weather_table(cursor):
    """Create the weather table if it doesn't exist"""
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS weather (
            id INTEGER PRIMARY KEY,
            activity_id INTEGER,
            location_name TEXT,
            location_region TEXT,
            location_country TEXT,
            time_epoch INTEGER,
            time TEXT,
            temp_c REAL,
            temp_f REAL,
            is_day INTEGER,
            condition_text TEXT,
            condition_icon TEXT,
            condition_code INTEGER,
            wind_mph REAL,
            wind_kph REAL,
            wind_degree INTEGER,
            wind_dir TEXT,
            pressure_mb REAL,
            pressure_in REAL,
            precip_mm REAL,
            precip_in REAL,
            snow_cm REAL,
            humidity INTEGER,
            cloud INTEGER,
            feelslike_c REAL,
            feelslike_f REAL,
            windchill_c REAL,
            windchill_f REAL,
            heatindex_c REAL,
            heatindex_f REAL,
            dewpoint_c REAL,
            dewpoint_f REAL,
            will_it_rain INTEGER,
            chance_of_rain INTEGER,
            will_it_snow INTEGER,
            chance_of_snow INTEGER,
            vis_km REAL,
            vis_miles REAL,
            gust_mph REAL,
            gust_kph REAL,
            uv REAL,
            FOREIGN KEY (activity_id) REFERENCES activities (id)
        )
    ''')

def parse_latlng(latlng_str):
    """Parse the lat/lng string format [lat, lng] into separate values"""
    if not latlng_str or latlng_str == 'NULL':
        return None, None
    
    # Remove brackets and split
    coords = latlng_str.strip('[]').split(',')
    if len(coords) != 2:
        return None, None
    
    try:
        lat = float(coords[0].strip())
        lng = float(coords[1].strip())
        return lat, lng
    except ValueError:
        return None, None

def get_weather_data(lat, lng, date_str, hour, api_key):
    """Fetch weather data from WeatherAPI"""
    url = f"https://api.weatherapi.com/v1/history.json"
    params = {
        'q': f"{lat},{lng}",
        'dt': date_str,
        'hour': hour,
        'key': api_key
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def extract_hour_from_datetime(weather_date_str):
    """Extract hour from weather_date string"""
    try:
        # Parse the datetime string and extract hour
        dt = datetime.strptime(weather_date_str, '%Y-%m-%d %H:%M:%S')
        return dt.hour, dt.strftime('%Y-%m-%d')
    except ValueError:
        # Try alternative format
        try:
            dt = datetime.strptime(weather_date_str, '%m/%d/%Y %H:%M')
            return dt.hour, dt.strftime('%Y-%m-%d')
        except ValueError:
            print(f"Could not parse datetime: {weather_date_str}")
            return None, None

def insert_weather_data(cursor, activity_id, weather_response):
    """Insert weather data into the database"""
    if not weather_response:
        return
    
    location = weather_response.get('location', {})
    forecast = weather_response.get('forecast', {})
    
    if not forecast.get('forecastday'):
        return
    
    # Get the hourly data
    hourly_data = forecast['forecastday'][0].get('hour', [])
    
    # Usually we want the first hour entry, but you can modify this logic
    if hourly_data:
        hour_data = hourly_data[0]  # Take the first hour entry
        
        cursor.execute('''
            INSERT INTO weather (
                activity_id, location_name, location_region, location_country,
                time_epoch, time, temp_c, temp_f, is_day,
                condition_text, condition_icon, condition_code,
                wind_mph, wind_kph, wind_degree, wind_dir,
                pressure_mb, pressure_in, precip_mm, precip_in, snow_cm,
                humidity, cloud, feelslike_c, feelslike_f,
                windchill_c, windchill_f, heatindex_c, heatindex_f,
                dewpoint_c, dewpoint_f, will_it_rain, chance_of_rain,
                will_it_snow, chance_of_snow, vis_km, vis_miles,
                gust_mph, gust_kph, uv
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            activity_id,
            location.get('name'),
            location.get('region'),
            location.get('country'),
            hour_data.get('time_epoch'),
            hour_data.get('time'),
            hour_data.get('temp_c'),
            hour_data.get('temp_f'),
            hour_data.get('is_day'),
            hour_data.get('condition', {}).get('text'),
            hour_data.get('condition', {}).get('icon'),
            hour_data.get('condition', {}).get('code'),
            hour_data.get('wind_mph'),
            hour_data.get('wind_kph'),
            hour_data.get('wind_degree'),
            hour_data.get('wind_dir'),
            hour_data.get('pressure_mb'),
            hour_data.get('pressure_in'),
            hour_data.get('precip_mm'),
            hour_data.get('precip_in'),
            hour_data.get('snow_cm'),
            hour_data.get('humidity'),
            hour_data.get('cloud'),
            hour_data.get('feelslike_c'),
            hour_data.get('feelslike_f'),
            hour_data.get('windchill_c'),
            hour_data.get('windchill_f'),
            hour_data.get('heatindex_c'),
            hour_data.get('heatindex_f'),
            hour_data.get('dewpoint_c'),
            hour_data.get('dewpoint_f'),
            hour_data.get('will_it_rain'),
            hour_data.get('chance_of_rain'),
            hour_data.get('will_it_snow'),
            hour_data.get('chance_of_snow'),
            hour_data.get('vis_km'),
            hour_data.get('vis_miles'),
            hour_data.get('gust_mph'),
            hour_data.get('gust_kph'),
            hour_data.get('uv')
        ))

def main():
    # Get API key from environment
    api_key = os.getenv('WEATHER_API_KEY')
    if not api_key:
        print("Error: WEATHER_API_KEY not found in environment variables")
        return
    
    # Connect to database
    conn = sqlite3.connect('strava_data.db')  # Adjust database path as needed
    cursor = conn.cursor()
    
    # Create weather table
    create_weather_table(cursor)
    
    # Get activities from database
    cursor.execute('''
        SELECT
            id,
            start_date_local,
            datetime(start_date_local, '+' || (elapsed_time / 2) || ' seconds') AS weather_date,
            start_latlng
        FROM
            activities
        WHERE
            start_date_local > '2025-07-02 06:00'
        ORDER BY
            start_date_local DESC
    ''')
    
    activities = cursor.fetchall()
    print(f"Found {len(activities)} activities to process")
    
    # Check which activities already have weather data
    cursor.execute('SELECT DISTINCT activity_id FROM weather')
    existing_weather_ids = set(row[0] for row in cursor.fetchall())
    
    processed = 0
    skipped = 0
    
    for activity_id, start_date, weather_date, start_latlng in activities:
        # Skip if weather data already exists
        if activity_id in existing_weather_ids:
            skipped += 1
            continue
        
        # Parse coordinates
        lat, lng = parse_latlng(start_latlng)
        if lat is None or lng is None:
            print(f"Skipping activity {activity_id}: Invalid coordinates")
            continue
        
        # Extract hour and date from weather_date
        hour, date_str = extract_hour_from_datetime(weather_date)
        if hour is None:
            print(f"Skipping activity {activity_id}: Could not parse date")
            continue
        
        print(f"Processing activity {activity_id}: {lat}, {lng} at {date_str} {hour}:00")
        
        # Fetch weather data
        weather_data = get_weather_data(lat, lng, date_str, hour, api_key)
        
        if weather_data:
            insert_weather_data(cursor, activity_id, weather_data)
            processed += 1
            
            # Commit every 10 records
            if processed % 10 == 0:
                conn.commit()
                print(f"Processed {processed} activities...")
        
        # Rate limiting - WeatherAPI has limits
        time.sleep(0.1)  # Adjust as needed based on your API plan
    
    # Final commit
    conn.commit()
    conn.close()
    
    print(f"Completed! Processed: {processed}, Skipped: {skipped}")

if __name__ == "__main__":
    main()