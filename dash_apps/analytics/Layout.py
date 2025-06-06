# dash_apps/analytics/layout.py
"""
Layout components for the analytics Dash application.
"""
from dash import dcc, html

def create_analytics_layout():
    """Create the main layout for the analytics dashboard."""
    return html.Div([
        dcc.Location(id='url', refresh=False),
        html.H3("Route & Metrics Overview"),
        html.Div(id='map-container'),
        html.Div(id='graphs-container', style={'marginTop': '30px'})
    ])