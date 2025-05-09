from flask import Flask, render_template
import sqlite3
from dash_app import create_dash_app
import pandas as pd
import datetime
import json
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for
from dateutil.relativedelta import relativedelta
import utils.db_utils as db_utils # custom utils
import utils.format_utils as format_utils # custom utils

# -------------------------------------
# App setup and initialization
# -------------------------------------

app = Flask(__name__) # Flask app

activity_id = db_utils.get_latest_activity() # load latest activity_id as default
dash_app = create_dash_app(app) # Initialize Dash app

@app.route("/")
def home():
    return render_template("home.html", activity = activity_id)

@app.route("/activity/")
def activity():
    activity_id = request.args.get("id", type=int)
    conn = sqlite3.connect(db_utils.DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM activities ORDER BY start_date DESC LIMIT 1")
    # for testing
    cur.execute(f"SELECT * FROM activities WHERE id = {activity_id}")
    activity = dict(cur.fetchone())
    # print(activity.distance)
    
    activity['distance'] = round(activity['distance'] / 1609, 2)
    activity['average_pace'] = format_utils.format_pace(activity['distance'], activity['moving_time'])
    activity['moving_time'] = format_utils.format_time(activity['moving_time'])
    activity['moving_time_minutes'] = activity['moving_time']
    activity['average_cadence'] = int(round(activity['average_cadence'] * 2, 0))
    
    activity['average_speed'] = round(activity['average_speed'], 1)
    activity['max_speed'] = round(activity['max_speed'], 1)
    activity['max_heartrate'] = round(activity['max_heartrate'])
    activity['average_heartrate'] = round(activity['average_heartrate'])
    activity['kilojoules'] = round(activity['kilojoules'])        

    return render_template("activity.html", activity=activity)

@app.route('/statistics/')
def statistics():
    """Render the statistics page with data from the database"""
    period = request.args.get('period', 'month')  # Default to month view
    
    # Get current date
    now = datetime.now()
    
    # Determine date range based on period
    if period == 'week':
        start_date = (now - timedelta(days=7)).strftime('%Y-%m-%d')
        date_range = f"Last 7 days"
    elif period == 'month':
        start_date = (now - timedelta(days=30)).strftime('%Y-%m-%d')
        date_range = f"Last 30 days"
    elif period == 'year':
        start_date = (now - relativedelta(years=1)).strftime('%Y-%m-%d')
        date_range = f"Last 12 months"
    else:  # 'all'
        start_date = '2000-01-01'  # A date far in the past
        date_range = "All time"
    
    conn = sqlite3.connect(db_utils.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.row_factory = db_utils.dict_factory
    
    # Count total activities in the period
    total_activities = conn.execute(
        'SELECT COUNT(*) as count FROM activities WHERE start_date >= ?', 
        (start_date,)
    ).fetchone()['count']
    
    # Calculate total distance in km
    total_distance_result = conn.execute(
        'SELECT SUM(distance) as total FROM activities WHERE start_date >= ?', 
        (start_date,)
    ).fetchone()
    total_distance = round(total_distance_result['total'] / 1000, 2) if total_distance_result['total'] else 0
    
    # Calculate total moving time
    total_time_result = conn.execute(
        'SELECT SUM(moving_time) as total FROM activities WHERE start_date >= ?', 
        (start_date,)
    ).fetchone()
    
    total_seconds = total_time_result['total'] if total_time_result['total'] else 0
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    total_time = f"{hours}h {minutes}m"
    
    # Estimate total calories (kilojoules / 4.184 is approximately calories)
    total_calories_result = conn.execute(
        'SELECT SUM(kilojoules) as total FROM activities WHERE start_date >= ?', 
        (start_date,)
    ).fetchone()
    total_calories = round(total_calories_result['total'] / 4.184) if total_calories_result['total'] else 0
    
    # Get weekly distances for the chart (last 7 days)
    weekly_distances = [0] * 7  # Initialize with zeros for each day
    
    # Calculate the date for 7 days ago
    seven_days_ago = now - timedelta(days=7)
    
    # Query for activities in the last 7 days
    weekly_activities = conn.execute(
        '''SELECT start_date_local, distance 
           FROM activities 
           WHERE start_date_local >= ? 
           ORDER BY start_date_local''',
        (seven_days_ago.strftime('%Y-%m-%d'),)
    ).fetchall()
    
    # Process the weekly activities
    for activity in weekly_activities:
        try:
            activity_date = datetime.strptime(activity['start_date_local'].split('T')[0], '%Y-%m-%d')
            day_diff = (now - activity_date).days
            if 0 <= day_diff < 7:
                index = 6 - day_diff  # Index 6 is today, 0 is 6 days ago
                weekly_distances[index] += round(activity['distance'] / 1000, 2)  # Convert meters to km
        except (ValueError, IndexError):
            continue
    
    # Get pace trends for the chart (last 10 activities)
    pace_activities = conn.execute(
        '''SELECT start_date_local, distance, moving_time 
           FROM activities 
           WHERE type = 'Run' AND distance > 0 AND moving_time > 0
           ORDER BY start_date_local DESC 
           LIMIT 10'''
    ).fetchall()
    
    pace_dates = []
    pace_values = []
    
    # Calculate pace for each activity (minutes per kilometer)
    for activity in reversed(pace_activities):  # Reverse to get chronological order
        try:
            # Format date for display
            date_str = datetime.strptime(activity['start_date_local'].split('T')[0], '%Y-%m-%d').strftime('%d %b')
            pace_dates.append(date_str)
            
            # Calculate pace: minutes per kilometer
            pace_minutes = (activity['moving_time'] / 60) / (activity['distance'] / 1000)
            pace_values.append(round(pace_minutes, 2))
        except (ValueError, ZeroDivisionError):
            continue
    
    # Get personal records
    # For simplicity, we'll use fastest 5K, 10K, and longest run as examples
    personal_records = []
    
    # Fastest 5K (approximately 5000m)
    fastest_5k = conn.execute(
        '''SELECT id, name, distance, moving_time, start_date_local
           FROM activities
           WHERE type = 'Run' AND distance BETWEEN 4500 AND 5500
           ORDER BY moving_time ASC
           LIMIT 1'''
    ).fetchone()
    
    if fastest_5k:
        date_str = datetime.strptime(fastest_5k['start_date_local'].split('T')[0], '%Y-%m-%d').strftime('%d %b %Y')
        minutes = fastest_5k['moving_time'] // 60
        seconds = fastest_5k['moving_time'] % 60
        time_str = f"{minutes}:{seconds:02d}"
        pace = round((fastest_5k['moving_time'] / 60) / (fastest_5k['distance'] / 1000), 2)
        
        personal_records.append({
            'distance': '5K',
            'time': time_str,
            'pace': pace,
            'date': date_str
        })
    
    # Fastest 10K (approximately 10000m)
    fastest_10k = conn.execute(
        '''SELECT id, name, distance, moving_time, start_date_local
           FROM activities
           WHERE type = 'Run' AND distance BETWEEN 9500 AND 10500
           ORDER BY moving_time ASC
           LIMIT 1'''
    ).fetchone()
    
    if fastest_10k:
        date_str = datetime.strptime(fastest_10k['start_date_local'].split('T')[0], '%Y-%m-%d').strftime('%d %b %Y')
        minutes = fastest_10k['moving_time'] // 60
        seconds = fastest_10k['moving_time'] % 60
        time_str = f"{minutes}:{seconds:02d}"
        pace = round((fastest_10k['moving_time'] / 60) / (fastest_10k['distance'] / 1000), 2)
        
        personal_records.append({
            'distance': '10K',
            'time': time_str,
            'pace': pace,
            'date': date_str
        })
    
    # Longest run
    longest_run = conn.execute(
        '''SELECT id, name, distance, moving_time, start_date_local
           FROM activities
           WHERE type = 'Run'
           ORDER BY distance DESC
           LIMIT 1'''
    ).fetchone()
    
    if longest_run:
        date_str = datetime.strptime(longest_run['start_date_local'].split('T')[0], '%Y-%m-%d').strftime('%d %b %Y')
        minutes = longest_run['moving_time'] // 60
        seconds = longest_run['moving_time'] % 60
        time_str = f"{minutes}:{seconds:02d}"
        distance_km = round(longest_run['distance'] / 1000, 2)
        pace = round((longest_run['moving_time'] / 60) / (longest_run['distance'] / 1000), 2)
        
        personal_records.append({
            'distance': f"{distance_km} km",
            'time': time_str,
            'pace': pace,
            'date': date_str
        })
    
    # Get shoe usage data
    shoes = conn.execute(
        '''SELECT gear_id, COUNT(*) as activities,
           SUM(distance) as total_distance,
           MAX(start_date_local) as last_used
           FROM activities
           WHERE gear_id IS NOT NULL AND gear_id != ''
           GROUP BY gear_id
           ORDER BY last_used DESC'''
    ).fetchall()
    
    shoe_data = []
    for shoe in shoes:
        if shoe['total_distance'] and shoe['last_used']:
            last_used_date = datetime.strptime(shoe['last_used'].split('T')[0], '%Y-%m-%d').strftime('%d %b %Y')
            
            shoe_data.append({
                'name': shoe['gear_id'],  # Ideally, you'd have a mapping of gear_id to shoe names
                'distance': round(shoe['total_distance'] / 1000, 2),  # Convert to km
                'activities': shoe['activities'],
                'last_used': last_used_date
            })
    
    # Get recent activities
    recent_activities = conn.execute(
        '''SELECT id, name, distance, moving_time, start_date_local
           FROM activities
           ORDER BY start_date_local DESC
           LIMIT 5'''
    ).fetchall()
    
    activities_list = []
    for activity in recent_activities:
        date_str = datetime.strptime(activity['start_date_local'].split('T')[0], '%Y-%m-%d').strftime('%d %b')
        distance_km = round(activity['distance'] / 1000, 2)
        
        minutes = activity['moving_time'] // 60
        seconds = activity['moving_time'] % 60
        time_str = f"{minutes}:{seconds:02d}"
        
        pace = round((activity['moving_time'] / 60) / (activity['distance'] / 1000), 2) if activity['distance'] > 0 else 0
        
        activities_list.append({
            'id': activity['id'],
            'name': activity['name'],
            'date': date_str,
            'distance': distance_km,
            'time': time_str,
            'pace': pace
        })
    
    # Summary stats for the cards
    stats = {
        'total_distance': total_distance,
        'total_time': total_time,
        'total_calories': total_calories,
        'total_activities': total_activities
    }
    
    conn.close()
    
    # Render the template with all the data
    return render_template(
        'statistics.html',
        period=period,
        date_range=date_range,
        total_activities=total_activities,
        total_distance=total_distance,
        stats=stats,
        weekly_distances=json.dumps(weekly_distances),
        pace_dates=pace_dates,
        # pace_dates=json.dumps(pace_dates),
        pace_values=json.dumps(pace_values),
        personal_records=personal_records,
        shoes=shoe_data,
        recent_activities=activities_list
    )

@app.route("/map/")
def dashredirect():
    return redirect(f"/map/?id={request.args.get('id')}")

if __name__ == '__main__':
    app.run(debug=True, port=5555)