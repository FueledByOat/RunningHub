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
        polyline_str = db_utils.get_activity_polyline(activity_id)
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

        # Calculate deltas
        delta_distance = np.diff(distance_miles)
        delta_time = np.diff(time_sec)

        # Avoid divide-by-near-zero errors
        min_valid_distance = 0.00001  # around 0.05 feet
        valid = delta_distance > min_valid_distance

        # Apply valid mask
        delta_distance = delta_distance[valid]
        delta_time = delta_time[valid]

        # Midpoints between GPS points (for x values of pace)
        x_mid = (distance_miles[1:] + distance_miles[:-1]) / 2
        x_mid = x_mid[valid]

        pace_sec_per_mile = delta_time / delta_distance
        pace_min_per_mile = pace_sec_per_mile / 60

        # Clip unrealistic pace values
        pace_min_per_mile = np.clip(pace_min_per_mile, 3, 20)

        def adaptive_moving_average(data, window_size=10):
            """Applies moving average with adaptive window size at edges."""
            data = np.array(data, dtype=np.float32)
            nan_mask = np.isnan(data)
            data[nan_mask] = 0
            
            smoothed = np.zeros_like(data)
            
            for i in range(len(data)):
                # Calculate adaptive window boundaries
                half_window = window_size // 2
                start_idx = max(0, i - half_window)
                end_idx = min(len(data), i + half_window + 1)
                
                # Extract window data
                window_data = data[start_idx:end_idx]
                window_nan_mask = nan_mask[start_idx:end_idx]
                
                # Count valid values in window
                valid_count = np.sum(~window_nan_mask)
                
                if valid_count > 0:
                    # Calculate average of valid values only
                    smoothed[i] = np.sum(window_data[~window_nan_mask]) / valid_count
                else:
                    smoothed[i] = np.nan
            
            # Reapply NaNs where original data was missing
            smoothed[nan_mask] = np.nan
            
            return smoothed.tolist()

        # Use your interpolation function
        pace_interp = dash_utils.interpolate_to_common_x(x_ref, pace_min_per_mile, x_mid)

        pace_smoothed = adaptive_moving_average(np.array(pace_interp), window_size=10)

        map_component = dl.Map(center=[lat_lng[0]['lat'], lat_lng[0]['lon']], zoom=13,
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
                    data=[go.Scatter(x=x_ref, y=pace_smoothed, mode='lines', name='Pace Chart')],
                    layout=go.Layout(title='Pace Chart', xaxis_title='Distance (mi)', yaxis=dict(autorange='reversed'), yaxis_title='Pace')
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
        ])

        return map_component, graphs_component

    return dash_app
