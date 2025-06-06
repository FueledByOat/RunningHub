# dash_apps/analytics/callbacks.py
"""
Callback functions for the analytics Dash application.
"""
import dash
from dash import Input, Output, html
import dash_leaflet as dl
import numpy as np
from urllib.parse import parse_qs
import plotly.graph_objs as go

import utils.dash_utils as dash_utils
import utils.db.db_utils as db_utils

def register_analytics_callbacks(dash_app, config):
    """Register all callbacks for the analytics dashboard."""
    
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
        distance = [i / 1609 for i in distance]  # Convert to miles
        x_ref = np.linspace(0, max(distance) if distance else 1, num=500)
        hr_interp = dash_utils.interpolate_to_common_x(x_ref, heartrate, distance)
        alt_interp = dash_utils.interpolate_to_common_x(x_ref, altitude, distance)
        power_interp = dash_utils.interpolate_to_common_x(x_ref, power, distance)

        # Pace calculations
        distance_miles = np.array(distance)
        time_sec = np.array(time)

        delta_distance = np.diff(distance_miles)
        delta_time = np.diff(time_sec)

        min_valid_distance = 0.00001
        valid = delta_distance > min_valid_distance

        delta_distance = delta_distance[valid]
        delta_time = delta_time[valid]

        x_mid = (distance_miles[1:] + distance_miles[:-1]) / 2
        x_mid = x_mid[valid]

        pace_sec_per_mile = delta_time / delta_distance
        pace_min_per_mile = pace_sec_per_mile / 60
        pace_min_per_mile = np.clip(pace_min_per_mile, 3, 20)

        pace_interp = dash_utils.interpolate_to_common_x(x_ref, pace_min_per_mile, x_mid)
        pace_smoothed = _adaptive_moving_average(np.array(pace_interp), window_size=10)

        # Create map component
        map_component = dl.Map(
            center=[lat_lng[0]['lat'], lat_lng[0]['lon']], 
            zoom=13,
            style={'width': '100%', 'height': '400px'}, 
            children=[
                dl.TileLayer(),
                dl.Polyline(positions=[[p['lat'], p['lon']] for p in lat_lng], color='blue')
            ]
        )

        # Create graphs component
        graphs_component = html.Div([
            _create_graph('Heart Rate vs Distance', x_ref, hr_interp, 'Distance (mi)', 'BPM'),
            _create_pace_graph('Pace Chart', x_ref, pace_smoothed),
            _create_graph('Altitude vs Distance', x_ref, alt_interp, 'Distance (mi)', 'Meters'),
            _create_graph('Power (Watts) vs Distance', x_ref, power_interp, 'Distance (mi)', 'Power (Watts)'),
        ])

        return map_component, graphs_component

def _adaptive_moving_average(data, window_size=10):
    """Apply moving average with adaptive window size at edges."""
    data = np.array(data, dtype=np.float32)
    nan_mask = np.isnan(data)
    data[nan_mask] = 0
    
    smoothed = np.zeros_like(data)
    
    for i in range(len(data)):
        half_window = window_size // 2
        start_idx = max(0, i - half_window)
        end_idx = min(len(data), i + half_window + 1)
        
        window_data = data[start_idx:end_idx]
        window_nan_mask = nan_mask[start_idx:end_idx]
        
        valid_count = np.sum(~window_nan_mask)
        
        if valid_count > 0:
            smoothed[i] = np.sum(window_data[~window_nan_mask]) / valid_count
        else:
            smoothed[i] = np.nan
    
    smoothed[nan_mask] = np.nan
    return smoothed.tolist()

def _create_graph(title, x_data, y_data, x_title, y_title):
    """Create a standard graph component."""
    return dash.dcc.Graph(
        figure=go.Figure(
            data=[go.Scatter(x=x_data, y=y_data, mode='lines', name=title.split()[0])],
            layout=go.Layout(title=title, xaxis_title=x_title, yaxis_title=y_title)
        )
    )

def _create_pace_graph(title, x_data, y_data):
    """Create a pace graph with reversed y-axis."""
    return dash.dcc.Graph(
        figure=go.Figure(
            data=[go.Scatter(x=x_data, y=y_data, mode='lines', name='Pace Chart')],
            layout=go.Layout(
                title=title, 
                xaxis_title='Distance (mi)', 
                yaxis=dict(autorange='reversed'), 
                yaxis_title='Pace'
            )
        )
    )