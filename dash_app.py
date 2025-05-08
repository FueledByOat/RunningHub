# dash_app.py

import dash
from dash import html, dcc
import dash_leaflet as dl
import polyline
import json
import sqlite3
import numpy as np
import plotly.graph_objs as go

def decode_polyline(polyline_str):
    try:
        return polyline.decode(polyline_str)  # List of (lat, lon)
    except Exception as e:
        print(f"Error decoding polyline: {e}")
        return []
    
def get_streams_data(activity_id, db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT distance_data, heartrate_data, altitude_data FROM streams WHERE activity_id = ?", (activity_id,))
    row = cur.fetchone()
    conn.close()

    if row:
        try:
            distance = json.loads(row[0]) if row[0] else []
            heartrate = json.loads(row[1]) if row[1] else []
            altitude = json.loads(row[2]) if row[2] else []
            return distance, heartrate, altitude
        except Exception as e:
            print("Failed to load stream data:", e)
    return [], [], []

def interpolate_to_common_x(x_ref, y_raw, x_raw):
    """Interpolates y_raw over x_ref, assuming x_raw and y_raw are aligned."""
    if len(x_raw) < 2 or len(x_ref) < 2:
        return [None] * len(x_ref)
    return np.interp(x_ref, x_raw, y_raw).tolist()

def create_dash_app(flask_app, polyline_str, activity_id=None, db_path=None):
    dash_app = dash.Dash(__name__, server=flask_app, url_base_pathname='/map/')
    decoded = decode_polyline(polyline_str)
    lat_lng = [{'lat': lat, 'lon': lon} for lat, lon in decoded]

    # Get stream data
    distance, heartrate, altitude = get_streams_data(activity_id, db_path) if activity_id and db_path else ([], [], [])

    # Interpolate both to a common X axis (distance in meters)
    x_ref = np.linspace(0, max(distance) if distance else 1, num=500)  # Common X-axis
    hr_interp = interpolate_to_common_x(x_ref, heartrate, distance)
    alt_interp = interpolate_to_common_x(x_ref, altitude, distance)

    # Dash layout
    dash_app.layout = html.Div([
        html.H3("Route & Metrics Overview"),
        dl.Map(center=[lat_lng[0]['lat'], lat_lng[0]['lon']], zoom=13,
               style={'width': '100%', 'height': '500px'}, children=[
                   dl.TileLayer(),
                   dl.Polyline(positions=[[p['lat'], p['lon']] for p in lat_lng], color='blue')
               ]),
        html.Div([
            dcc.Graph(
                figure=go.Figure(
                    data=[go.Scatter(x=x_ref, y=hr_interp, mode='lines', name='Heart Rate')],
                    layout=go.Layout(title='Heart Rate vs Distance', xaxis_title='Distance (m)', yaxis_title='BPM')
                )
            ),
            dcc.Graph(
                figure=go.Figure(
                    data=[go.Scatter(x=x_ref, y=alt_interp, mode='lines', name='Altitude')],
                    layout=go.Layout(title='Altitude vs Distance', xaxis_title='Distance (m)', yaxis_title='Meters')
                )
            ),
        ], style={'marginTop': '30px'})
    ])

    return dash_app
