import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import sqlite3
import pandas as pd
import utils.db_utils as db_utils


def create_dash_dashboard_app(server, db_path):

    def create_sparkline(data, color):
        return dcc.Graph(
            figure=go.Figure(
                data=[go.Scatter(
                    y=data,
                    mode='lines',
                    line=dict(color=color, width=2),
                    hoverinfo='none',
                )],
                layout=go.Layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=60,
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
            ),
            config={'displayModeBar': False}
        )

    def status_badge_acr(value, thresholds, labels):
        if value < thresholds[0]:
            return dbc.Badge(labels[0], color="warning")
        elif thresholds[0] <= value <= thresholds[1]:
            return dbc.Badge(labels[1], color="success")
        else:
            return dbc.Badge(labels[2], color="danger")
    

    def acwr_card(acwr_value, trend_data):
        badge = status_badge_acr(acwr_value, [0.8, 1.3], ["High Risk", "Optimal", "Caution"])
        return dbc.Card([
            dbc.CardHeader("ACWR Trend"),
            dbc.CardBody([
                html.H3(f"{acwr_value:.2f}", className="card-title"),
                badge,
                html.P("Ratio of short (7d) vs long (28d) load.\nTarget: 0.8 - 1.3"),
                create_sparkline(trend_data, "#0c1559")
            ])
        ], color="light", className="mb-3")

    def status_badge_hrd(value, thresholds, labels):
        if value < thresholds[0]:
            return dbc.Badge(labels[0], color="success")
        elif thresholds[0] <= value <= thresholds[1]:
            return dbc.Badge(labels[1], color="warning")
        else:
            return dbc.Badge(labels[2], color="danger")

    def hrd_card(hrd_value, trend_data):
        badge = status_badge_hrd(hrd_value, [5, 10], ["High Drift", "Elevated", "Good"])
        return dbc.Card([
            dbc.CardHeader("Heart Rate Drift"),
            dbc.CardBody([
                html.H3(f"{hrd_value:.1f}%", className="card-title"),
                badge,
                html.P("Percent rise in HR over session.\nTarget: < 5%"),
                create_sparkline(trend_data, "#0c1559")
            ])
        ], color="light", className="mb-3")

    def status_badge_cad(value, thresholds, labels):
        if value < thresholds[0]:
            return dbc.Badge(labels[0], color="success")
        elif thresholds[0] <= value <= thresholds[1]:
            return dbc.Badge(labels[1], color="warning")
        else:
            return dbc.Badge(labels[2], color="danger")
        
    def cadence_card(cv_value, trend_data):
        badge = status_badge_cad(cv_value, [4, 6], ["Unstable", "Slightly Unstable", "Stable"]) # Changed thresholds
        return dbc.Card([
            dbc.CardHeader("Cadence Stability"),
            dbc.CardBody([
                html.H3(f"{cv_value:.1f}%", className="card-title"),
                badge,
                html.P("Coefficient of variation vs pace.\nTarget: < 4%"),
                create_sparkline(trend_data, "#0c1559")
            ])
        ], color="light", className="mb-3")

    def build_dashboard_cards(acwr_value, acwr_trend, hrd_value, hrd_trend, cv_value, cv_trend):
        return dbc.Row([
            dbc.Col(acwr_card(acwr_value, acwr_trend), md=4),
            dbc.Col(hrd_card(hrd_value, hrd_trend), md=4),
            dbc.Col(cadence_card(cv_value, cv_trend), md=4),
        ], className="mb-4")

    # --- Data Retrieval Functions ---
    def get_acwr_data():
        try:
            df = db_utils.get_acwr_data()
            return df
        except sqlite3.Error as e:
            print(f"Database error in get_acwr_data: {e}")
            return pd.DataFrame()  # Return an empty DataFrame in case of an error

    def get_hr_drift_data():
        try:
            df = db_utils.get_hr_drift_data()
            return df
        except sqlite3.Error as e:
            print(f"Database error in get_hr_drift_data: {e}")
            return pd.DataFrame()

    def get_cadence_stability_data():
        try:
            df = db_utils.get_cadence_stability_data()
            return df
        except sqlite3.Error as e:
            print(f"Database error in get_cadence_stability_data: {e}")
            return pd.DataFrame()

    # --- End Data Retrieval Functions ---

    # Load data once for display (consider caching)
    df_acwr = get_acwr_data()
    df_hr_drift = get_hr_drift_data()
    df_cadence = get_cadence_stability_data()

    latest_acwr = df_acwr['acwr'].iloc[0] if not df_acwr.empty else None
    acwr_trend = df_acwr['acwr'].head(7).tolist()[::-1] if not df_acwr.empty else []

    latest_hr_drift = df_hr_drift['hr_drift_pct'].iloc[0] if not df_hr_drift.empty else None
    hr_drift_trend = df_hr_drift['hr_drift_pct'].head(7).tolist()[::-1] if not df_hr_drift.empty else []

    latest_cadence_cv = df_cadence['cadence_cv'].iloc[0] if not df_cadence.empty else None
    cadence_cv_trend = df_cadence['cadence_cv'].head(7).tolist()[::-1] if not df_cadence.empty else []

    print(latest_acwr)
    print(latest_hr_drift)
    print(latest_cadence_cv)
    print(cadence_cv_trend)

    dashboard_cards = build_dashboard_cards(
        latest_acwr,
        acwr_trend,
        latest_hr_drift,
        hr_drift_trend,
        latest_cadence_cv,
        cadence_cv_trend
    )

    app_layout = dbc.Container(
        [
            html.H1("Athlete Performance Dashboard", className="mb-4"),
            dashboard_cards,
            # Add more components here for detailed analysis/graphs
        ],
        fluid=True,
    )

    # App Layout
    dash_app = dash.Dash(__name__, server=server, url_base_pathname='/dashboard/', external_stylesheets=[dbc.themes.BOOTSTRAP])
    dash_app.title = "Running Trends Dashboard"
    dash_app.layout = app_layout
    # dash_app.layout = dbc.Container([
    # html.H2("Runner Health Dashboard", className="my-4"),
    # build_dashboard_cards(latest_acwr, acwr_trend, latest_hr_drift, hr_drift_trend, latest_cadence_cv, cadence_cv_trend)
    #     ], fluid=True)

    return dash_app
