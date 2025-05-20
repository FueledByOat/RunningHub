from flask import Flask, render_template
import sqlite3
from dash_dashboard_app.layout import create_dash_dashboard_app
from dash_app import create_dash_app
from density_dash import create_density_dash
import pandas as pd
import datetime
import json
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for
from dateutil.relativedelta import relativedelta
import utils.db_utils as db_utils # custom utils
import utils.format_utils as format_utils # custom utils
import utils.language_model_utils as lm_utils # custom utils

# -------------------------------------
# Flask and Dash App setup and initialization
# -------------------------------------

app = Flask(__name__) # Flask app


dash_app = create_dash_app(app) # Initialize Dash activities page app
create_density_dash(app, db_path=db_utils.DB_PATH)
create_dash_dashboard_app(app, db_path=db_utils.DB_PATH)

# -------------------------------------
# Home Page Endpoints
# -------------------------------------


@app.route("/")
def home():
    LATEST_ACTIVITY_ID = db_utils.get_latest_activity() # load latest activity_id as default
    return render_template("home.html", activity = LATEST_ACTIVITY_ID)

@app.route("/runnervision/")
def runnervision_home():
    LATEST_ACTIVITY_ID = db_utils.get_latest_activity() # load latest activity_id as default
    return render_template("home.html", activity = LATEST_ACTIVITY_ID)

# -------------------------------------
# Simple Web Page Endpoints
# -------------------------------------

@app.route('/query/', methods=['GET', 'POST'])
def query():
    error = None
    columns = []
    rows = []
    sql_query = ''
    param_input = ''

    if request.method == 'POST':
        sql_query = request.form.get('sql_query', '')
        param_input = request.form.get('params', '{}')
        

        try:
            if not sql_query.strip().lower().startswith("select"):
                raise Exception("Only SELECT queries allowed in dev mode.")

            # Try to parse parameters as JSON
            param_input = param_input.strip() or '{}'
            params = json.loads(param_input)
            assert isinstance(params, dict)

            with sqlite3.connect(db_utils.DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute(sql_query, params)

                if sql_query.strip().lower().startswith('select'):
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()
                else:
                    conn.commit()
        except json.JSONDecodeError:
            error = "Invalid JSON format in parameters."
        except Exception as e:
            error = str(e)

    return render_template('query.html',
                           columns=columns,
                           rows=rows,
                           error=error,
                           request=request,
                           sql_query = sql_query)

@app.route('/ai_query', methods=['POST'])
def run_ai_query():
    user_question = request.form.get('user_question', '')
    columns = []
    rows = []
    error = None
    sql_query = ''

    try:
        query_prompt_generator = lm_utils.generate_sql_from_natural_language(user_question)
        sql_query = lm_utils.extract_sql_query(query_prompt_generator)
        # For now, use empty params for simplicity
        with sqlite3.connect(db_utils.DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(sql_query)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
        print(columns)
        print(rows)

    except Exception as e:
        error = f"Failed to process question: {str(e)}, Query Generated: {sql_query}"

    return render_template('query.html', columns=columns, rows=rows, error=error, request=request, sql_query = sql_query)


@app.route('/runnervision')
def runnervision():
    return render_template('runnervision.html')

@app.route('/run_biomechanic_analysis', methods=['POST'])
def run_analysis():
    # Simulate or run long analysis process
    result = perform_video_analysis()  # Your logic here
    return jsonify({"status": "complete", "message": "Analysis finished!"})

@app.route("/activity/")
def activity():
    activity_id = request.args.get("id", type=int)
    units = request.args.get('units', 'mi')  # Default to miles
    conn = sqlite3.connect(db_utils.DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""SELECT a.*,  coalesce(concat(g.model_name, " ", g.nickname), a.gear_id) as gear_name
                FROM activities as a
                LEFT JOIN gear as g 
                on a.gear_id = g.gear_id
                WHERE a.id = ?
                AND a.gear_id IS NOT NULL AND a.gear_id != ''
                AND a.type in ('Run', 'Ride')
                ORDER BY a.start_date 
                DESC LIMIT 1""", (activity_id,)
    )
    activity = dict(cur.fetchone())
    # print(activity.distance)
    
    activity['distance'] = round(activity['distance'] / 1609, 2) if units == 'mi' else round(activity['distance'] * .001, 2)
    activity['average_pace'] = format_utils.format_pace(activity['distance'], activity['moving_time'], units = units)
    activity['moving_time'] = format_utils.format_time(activity['moving_time'])
    activity['moving_time_minutes'] = activity['moving_time']
    
    # Remove cadence for Rides
    activity['average_cadence'] = 0 if activity['type'] == 'Ride' else activity['average_cadence']
    activity['average_cadence'] = int(round(activity['average_cadence'] * 2, 0))
    activity['average_speed'] = round(activity['average_speed'] * 2.237, 1)  if units == 'mi' else round(activity['average_speed'] * 3.6, 1) 
    activity['max_speed'] = round(activity['max_speed'] * 2.237, 1) if units == 'mi' else round(activity['max_speed'] * 3.6, 1)
    activity['max_heartrate'] = round(activity['max_heartrate'])
    activity['average_heartrate'] = round(activity['average_heartrate'])
    activity['kilojoules'] = round(activity['kilojoules'])
    activity['start_date'], activity['start_time'] = format_utils.date_format(activity['start_date'])

    return render_template("activity.html", activity=activity, units=units)

@app.route('/trophy_room/')
def trophy_room():
    units = request.args.get('units', 'mi')  # Default to miles
    conn = sqlite3.connect(db_utils.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.row_factory = db_utils.dict_factory
    # Get personal records
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
        time_str = format_utils.format_time(fastest_5k['moving_time'])
        distance = round(fastest_5k['distance'] / 1000, 2) if units == 'km' else round(fastest_5k['distance'] / 1609, 2)
        pace = format_utils.format_pace(distance, fastest_5k['moving_time'], units = units)
        
        personal_records.append({
            'distance': '5K',
            'time': time_str,
            'pace': pace,
            'date': date_str
        })
    
    # Fastest 8K (approximately 80000m)
    fastest_8k = conn.execute(
        '''SELECT id, name, distance, moving_time, start_date_local
           FROM activities
           WHERE type = 'Run' AND distance BETWEEN 7600 AND 8400
           ORDER BY moving_time ASC
           LIMIT 1'''
    ).fetchone()
    
    if fastest_8k:
        date_str = datetime.strptime(fastest_8k['start_date_local'].split('T')[0], '%Y-%m-%d').strftime('%d %b %Y')
        time_str = format_utils.format_time(fastest_8k['moving_time'])
        distance = round(fastest_8k['distance'] / 1000, 2) if units == 'km' else round(fastest_8k['distance'] / 1609, 2)
        pace = format_utils.format_pace(distance, fastest_8k['moving_time'], units = units)
        
        personal_records.append({
            'distance': '8K',
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
        time_str = format_utils.format_time(fastest_10k['moving_time'])
        distance = round(fastest_10k['distance'] / 1000, 2) if units == 'km' else round(fastest_10k['distance'] / 1609, 2)
        pace = format_utils.format_pace(distance, fastest_10k['moving_time'], units = units)
        
        personal_records.append({
            'distance': '10K',
            'time': time_str,
            'pace': pace,
            'date': date_str
        })
    
    # Fastest HM Half Marathon (approximately 21,097m)
    fastest_HM = conn.execute(
        '''SELECT id, name, distance, moving_time, start_date_local
           FROM activities
           WHERE type = 'Run' AND distance BETWEEN 20750 AND 22000
           ORDER BY moving_time ASC
           LIMIT 1'''
    ).fetchone()
    
    if fastest_HM:
        date_str = datetime.strptime(fastest_HM['start_date_local'].split('T')[0], '%Y-%m-%d').strftime('%d %b %Y')
        time_str = format_utils.format_time(fastest_HM['moving_time'])
        distance = round(fastest_HM['distance'] / 1000, 2) if units == 'km' else round(fastest_HM['distance'] / 1609, 2)
        pace = format_utils.format_pace(distance, fastest_HM['moving_time'], units = units)
        
        personal_records.append({
            'distance': 'Half Marathon',
            'time': time_str,
            'pace': pace,
            'date': date_str
        })
    
    # Fastest FM Full Marathon (approximately 42195m)
    fastest_FM = conn.execute(
        '''SELECT id, name, distance, moving_time, start_date_local
           FROM activities
           WHERE type = 'Run' AND distance BETWEEN 41800 AND 43050
           ORDER BY moving_time ASC
           LIMIT 1'''
    ).fetchone()
    
    if fastest_FM:
        date_str = datetime.strptime(fastest_FM['start_date_local'].split('T')[0], '%Y-%m-%d').strftime('%d %b %Y')
        time_str = format_utils.format_time(fastest_FM['moving_time'])
        distance = round(fastest_FM['distance'] / 1000, 2) if units == 'km' else round(fastest_FM['distance'] / 1609, 2)
        pace = format_utils.format_pace(distance, fastest_FM['moving_time'], units = units)
        
        personal_records.append({
            'distance': 'Full Marathon',
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
        time_str = format_utils.format_time(longest_run['moving_time'])
        distance = round(longest_run['distance'] / 1000, 2) if units == 'km' else round(longest_run['distance'] / 1609, 2)
        pace = format_utils.format_pace(distance, longest_run['moving_time'], units = units)
        
        personal_records.append({
            'distance': f"Longest Run: {distance}" + units,
            'time': time_str,
            'pace': pace,
            'date': date_str
        })
    return render_template("trophy_room.html", personal_records=personal_records, units = units)


@app.route('/statistics/', methods=['GET'])
def statistics():
    """Render the statistics page with data from the database"""
    period = request.args.get('period', 'week')  # Default to week view
    units = request.args.get('units', 'mi')  # Default to miles
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
        "SELECT COUNT(*) as count FROM activities WHERE start_date >= ? AND type = 'Run'", 
        (start_date,)
    ).fetchone()['count']

    total_elevation = conn.execute(
        "SELECT SUM(total_elevation_gain) as count FROM activities WHERE start_date >= ? AND type = 'Run'", 
        (start_date,)
    ).fetchone()['count']
    
    # Calculate total distance in km
    total_distance_result = conn.execute(
        "SELECT SUM(distance) as total FROM activities WHERE start_date >= ? AND type = 'Run'", 
        (start_date,)
    ).fetchone()
    total_distance = round(total_distance_result['total'] / 1000, 2) if units == 'km' else round(total_distance_result['total'] / 1609, 2)
    
    # Calculate total moving time
    total_time_result = conn.execute(
        "SELECT SUM(moving_time) as total FROM activities WHERE start_date >= ? AND type = 'Run'", 
        (start_date,)
    ).fetchone()
    
    total_seconds = total_time_result['total'] if total_time_result['total'] else 0
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    total_time = f"{hours}h {minutes}m"
    
    # Estimate total calories (kilojoules / 4.184 is approximately calories)
    total_calories_result = conn.execute(
        "SELECT SUM(kilojoules) as total FROM activities WHERE start_date >= ? AND type = 'Run'", 
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
           AND type = 'Run'
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

    # Get shoe usage data
    shoes = conn.execute(
        '''SELECT coalesce(concat(g.model_name, " ", g.nickname), a.gear_id) as gear_id, 
            COUNT(*) as activities,
           SUM(a.distance) as total_distance,
           MAX(start_date_local) as last_used
           FROM activities as a
           LEFT JOIN gear as g 
           on a.gear_id = g.gear_id
           WHERE a.gear_id IS NOT NULL AND a.gear_id != ''
           GROUP BY a.gear_id
           HAVING MAX(start_date_local) >= ?
           ORDER BY last_used DESC
           ''', (start_date,)
    ).fetchall()
    
    shoe_data = []
    for shoe in shoes:
        if shoe['total_distance'] and shoe['last_used']:
            last_used_date = datetime.strptime(shoe['last_used'].split('T')[0], '%Y-%m-%d').strftime('%d %b %Y')
            
            shoe_data.append({
                'name': shoe['gear_id'],  # Ideally, you'd have a mapping of gear_id to shoe names
                'distance': round(shoe['total_distance'] / 1000, 2) if units == 'km' else round(shoe['total_distance'] / 1609, 2), 
                'activities': shoe['activities'],
                'last_used': last_used_date
            })
    
    # Get recent activities
    recent_activities = conn.execute(
        '''SELECT id, name, distance, moving_time, start_date_local
           FROM activities
           WHERE type in ("Run")
           ORDER BY start_date_local DESC
           LIMIT 5'''
    ).fetchall()
    



    activities_list = []
    for activity in recent_activities:
        date_str = datetime.strptime(activity['start_date_local'].split('T')[0], '%Y-%m-%d').strftime('%d %b')
        distance_recent_activities = round(activity['distance'] / 1000, 2) if units == 'km' else round(activity['distance'] / 1609, 2)

        time_str = format_utils.format_time(activity['moving_time'])
        
        pace = format_utils.format_pace(distance_recent_activities, activity['moving_time'], units=units) if activity['distance'] > 0 else 0
        
        activities_list.append({
            'id': activity['id'],
            'name': activity['name'],
            'date': date_str,
            'distance': distance_recent_activities,
            'time': time_str,
            'pace': pace
        })
    
    # Summary stats for the cards
    stats = {
        'total_distance': total_distance,
        'total_time': total_time,
        'total_calories': total_calories,
        'total_activities': total_activities,
        'total_elevation' : total_elevation
    }
    
    conn.close()
    
    # Render the template with all the data
    return render_template(
        'statistics.html',
        period=period,
        units=units,
        date_range=date_range,
        total_activities=total_activities,
        total_distance=total_distance,
        stats=stats,
        weekly_distances=json.dumps(weekly_distances),
        pace_dates=pace_dates,
        # pace_dates=json.dumps(pace_dates),
        pace_values=json.dumps(pace_values),
        shoes=shoe_data,
        recent_activities=activities_list,
        start_date = start_date
    )
# -------------------------------------
# Dash Pages and Redirect Endpoints
# -------------------------------------


@app.route("/map/")
def dashredirect():
    return redirect(f"/map/?id={request.args.get('id')}")

@app.route("/density/")
def density_redirect():
    # trying to route this and stats together for jinja changes
    period = request.args.get('period', 'week')  # Default to week view
    # Get current date
    now = datetime.now()
    print(request.referrer)
    zoom = 12 if "stat" in request.referrer else 10
    # Determine date range based on period
    if period == 'week':
        start_date = (now - timedelta(days=7)).strftime('%Y-%m-%d')
    elif period == 'month':
        start_date = (now - timedelta(days=30)).strftime('%Y-%m-%d')
    elif period == 'year':
        start_date = (now - relativedelta(years=1)).strftime('%Y-%m-%d')
    else:  # 'all'
        start_date = '2000-01-01'  # A date far in the past

    start_date = request.args.get("start_date", "2024-01-01")
    print(zoom)
    return redirect(f"/density_dash/?start_date={start_date}&zoom={zoom}")

@app.route('/dashboard-redirect/')
def dashboard_redirect():
    return redirect('/dashboard/')

if __name__ == '__main__':
    app.run(debug=True, port=5555)