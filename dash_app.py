# dash_app.py

import dash
from dash import dcc, html, Input, Output
import dash_leaflet as dl
import sqlite3
import numpy as np
from urllib.parse import parse_qs
import utils.dash_utils as dash_utils
import plotly.graph_objs as go
import utils.db_utils as db_utils

external_stylesheets = ["/static/css/styles.css"]

def create_dash_app(flask_app):
    """satisfying pylint for now"""
    dash_app = dash.Dash(__name__, server=flask_app, url_base_pathname='/map/', suppress_callback_exceptions=True, external_stylesheets=external_stylesheets)

    dash_app.layout = html.Div([
        dcc.Location(id='url', refresh=False),
        html.H3("Route & Metrics Overview"),
        html.Div(id='map-container'),
        html.Div(id='graphs-container', style={'marginTop': '30px'})
    ])

    @dash_app.callback(
        [Output('map-container', 'children'),
         Output('graphs-container', 'children')],
        Input('url', 'search')
    )
    def update_content(search):

        params = parse_qs(search.lstrip('?'))
        activity_id = int(params.get('id', [None])[0]) if 'id' in params else None

        if not activity_id:
            return html.Div("No activity ID provided."), html.Div()

        # Fetch data
        polyline_str = db_utils.get_latest_polyline(activity_id)
        decoded = dash_utils.decode_polyline(polyline_str)
        lat_lng = [{'lat': lat, 'lon': lon} for lat, lon in decoded]

        distance, heartrate, altitude, power, time = dash_utils.get_streams_data(activity_id)
        distance = [i / 1609 for i in distance] # now in units of second to parts of a mile
        x_ref = np.linspace(0, max(distance) if distance else 1, num=500)
        hr_interp = dash_utils.interpolate_to_common_x(x_ref, heartrate, distance)
        alt_interp = dash_utils.interpolate_to_common_x(x_ref, altitude, distance)
        power_interp = dash_utils.interpolate_to_common_x(x_ref, power, distance)

        # pace section 
        # Ensure inputs are numpy arrays
        distance_miles = np.array(distance)
        time_sec = np.array(time)  # This is the original list of time in seconds

        # Compute deltas
        delta_distance = np.diff(distance_miles)      # miles per second
        delta_time = np.diff(time_sec)                # usually 1 sec, but safe to use diff

        # Avoid divide-by-zero
        with np.errstate(divide='ignore', invalid='ignore'):
            pace_per_mile_sec = delta_time / delta_distance  # seconds per mile
            pace_per_mile_min = pace_per_mile_sec / 60       # minutes per mile
            pace_per_mile_min = np.clip(pace_per_mile_min, 2, 12)  # reasonable range, clip anything absurd     

        # Midpoints for pace values (since they are between distance[i] and distance[i+1])
        x_mid = (distance_miles[1:] + distance_miles[:-1]) / 2

        def moving_average(y, window_size=10):
            return np.convolve(y, np.ones(window_size)/window_size, mode='same')

        # Use your interpolation function
        pace_interp = dash_utils.interpolate_to_common_x(x_ref, pace_per_mile_min, x_mid)

        pace_interp_smooth = moving_average(np.array(pace_interp), window_size=10)

        map_component = dl.Map(center=[lat_lng[0]['lat'], lat_lng[0]['lon']], zoom=14,
                                style={'width': '100%', 'height': '400px'}, children=[
                                    dl.TileLayer(),
                                    dl.Polyline(positions=[[p['lat'], p['lon']] for p in lat_lng], color='blue')
                                ])

        graphs_component = html.Div([
            dcc.Graph(
                figure=go.Figure(
                    data=[go.Scatter(x=x_ref, y=hr_interp, mode='lines', name='Heart Rate')],
                    layout=go.Layout(title='Heart Rate vs Distance', xaxis_title='Distance (mi)', yaxis_title='BPM')
                )
            ),
            dcc.Graph(
                figure=go.Figure(
                    data=[go.Scatter(x=x_ref, y=alt_interp, mode='lines', name='Altitude')],
                    layout=go.Layout(title='Altitude vs Distance', xaxis_title='Distance (mi)', yaxis_title='Meters')
                )
            ),
            dcc.Graph(
                figure=go.Figure(
                    data=[go.Scatter(x=x_ref, y=power_interp, mode='lines', name='Power')],
                    layout=go.Layout(title='Power (Watts) vs Distance', xaxis_title='Distance (mi)', yaxis_title='Power (Watts)')
                )
            ),
            dcc.Graph(
                figure=go.Figure(
                    data=[go.Scatter(x=x_ref, y=pace_interp_smooth, mode='lines', name='Pace Chart')],
                    layout=go.Layout(title='Pace Chart', xaxis_title='Distance (mi)', yaxis=dict(autorange='reversed'), yaxis_title='Pace')
                )
            ),
        ])

        return map_component, graphs_component

    return dash_app
