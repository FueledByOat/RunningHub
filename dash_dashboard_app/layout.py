import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import sqlite3
import pandas as pd
import utils.db_utils as db_utils


def create_dash_dashboard_app(server, db_path):

    def create_sparkline(data, color, reference_line=None):
        """Create a sparkline with optional reference value line"""
        figure = go.Figure()
        
        # Add the main data line
        figure.add_trace(go.Scatter(
            y=data,
            mode='lines',
            line=dict(color=color, width=2),
            hoverinfo='y',
            name='Value'
        ))
        
        # Add reference line if provided
        if reference_line is not None:
            figure.add_shape(
                type="line",
                x0=0,
                x1=len(data)-1,
                y0=reference_line,
                y1=reference_line,
                line=dict(color="#888888", width=1, dash="dot"),
            )
        
        return dcc.Graph(
            figure=figure.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                height=60,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                hovermode='closest'
            ),
            config={'displayModeBar': False}
        )

    def status_badge(value, thresholds, labels, colors):
        """
        Generic status badge function
        
        Parameters:
        - value: The metric value to evaluate
        - thresholds: List of threshold values [low, high]
        - labels: List of status labels ["low status", "optimal status", "high status"]
        - colors: List of colors ["low color", "optimal color", "high color"]
        """
        if value < thresholds[0]:
            return dbc.Badge(labels[0], color=colors[0])
        elif thresholds[0] <= value <= thresholds[1]:
            return dbc.Badge(labels[1], color=colors[1])
        else:
            return dbc.Badge(labels[2], color=colors[2])

    def metric_card(title, value, badge, description, trend_data, color="#0c1559", reference_line=None):
        """Generic metric card for any running metric"""
        return dbc.Card([
            dbc.CardHeader(title),
            dbc.CardBody([
                html.H3(value, className="card-title"),
                badge,
                html.P(description),
                create_sparkline(trend_data, color, reference_line)
            ])
        ], color="light", className="mb-3")

    def acwr_card(acwr_value, trend_data):
        """
        ACWR (Acute:Chronic Workload Ratio) card
        
        Clinical insight: The 0.8-1.3 range is supported by research as the optimal training zone
        to minimize injury risk. Values below 0.8 indicate undertraining relative to baseline,
        while values above 1.3 indicate excessive acute workload.
        """
        badge = status_badge(
            acwr_value, 
            [0.8, 1.3], 
            ["Undertraining", "Optimal", "Overreaching"],
            ["warning", "success", "danger"]
        )
        
        value_text = f"{acwr_value:.2f}" if acwr_value is not None else "No data"
        description = "Ratio of 7-day vs 28-day load.\nTarget: 0.8 - 1.3"
        
        return metric_card(
            "ACWR Trend", 
            value_text, 
            badge, 
            description, 
            trend_data,
            reference_line=1.0  # Ideal balance between acute and chronic load
        )

    def hrd_card(hrd_value, trend_data):
        """
        Heart Rate Drift card
        
        Clinical insight: HR drift below 5% indicates good aerobic efficiency. 
        Values between 5-10% suggest moderate fitness, while values above 10% 
        indicate potential aerobic limitation or insufficient base training.
        """
        badge = status_badge(
            hrd_value, 
            [5, 10], 
            ["Optimal", "Moderate", "Concerning"],
            ["success", "warning", "danger"]
        )
        
        value_text = f"{hrd_value:.1f}%" if hrd_value is not None else "No data"
        description = "Percent rise in HR over session.\nTarget: < 5%"
        
        return metric_card(
            "Heart Rate Drift", 
            value_text, 
            badge, 
            description, 
            trend_data,
            reference_line=5.0  # Reference line at the optimal threshold
        )

    def cadence_card(cv_value, trend_data):
        """
        Cadence Stability card
        
        Clinical insight: Lower CV values indicate better running economy.
        Values below 4% suggest strong neuromuscular control and consistent form.
        Higher values (>6%) may indicate fatigue or form breakdown.
        """
        badge = status_badge(
            cv_value, 
            [4, 6], 
            ["Stable", "Moderate", "Unstable"],
            ["success", "warning", "danger"]
        )
        
        value_text = f"{cv_value:.1f}%" if cv_value is not None else "No data"
        description = "Coefficient of variation vs pace.\nTarget: < 4%"
        
        return metric_card(
            "Cadence Stability", 
            value_text, 
            badge, 
            description, 
            trend_data,
            reference_line=4.0  # Reference line at the optimal threshold
        )

    def build_dashboard_cards(acwr_value, acwr_trend, hrd_value, hrd_trend, cv_value, cv_trend):
        """Build the complete dashboard row with all three metric cards"""
        return dbc.Row([
            dbc.Col(acwr_card(acwr_value, acwr_trend), md=4),
            dbc.Col(hrd_card(hrd_value, hrd_trend), md=4),
            dbc.Col(cadence_card(cv_value, cv_trend), md=4),
        ], className="mb-4")
 
    # --- Load and Process Data ---

    def load_dashboard_data(db_path=db_path):
        """Load all required data for the dashboard"""
        # Load data
        df_acwr = db_utils.get_acwr_data(db_path)
        df_hr_drift = db_utils.get_hr_drift_data(db_path)
        df_cadence = db_utils.get_cadence_stability_data(db_path)
        
        # Process ACWR data
        latest_acwr = df_acwr['acwr'].iloc[0] if not df_acwr.empty and 'acwr' in df_acwr.columns else None
        acwr_trend = df_acwr['acwr'].head(14).tolist()[::-1] if not df_acwr.empty and 'acwr' in df_acwr.columns else []
        
        # Process HR drift data
        latest_hr_drift = df_hr_drift['hr_drift_pct'].iloc[0] if not df_hr_drift.empty and 'hr_drift_pct' in df_hr_drift.columns else None
        hr_drift_trend = df_hr_drift['hr_drift_pct'].head(14).tolist()[::-1] if not df_hr_drift.empty and 'hr_drift_pct' in df_hr_drift.columns else []
        
        # Process cadence data
        latest_cadence_cv = df_cadence['cadence_cv'].iloc[0] if not df_cadence.empty and 'cadence_cv' in df_cadence.columns else None
        cadence_cv_trend = df_cadence['cadence_cv'].head(14).tolist()[::-1] if not df_cadence.empty and 'cadence_cv' in df_cadence.columns else []
        
        return {
            'acwr': {
                'latest': latest_acwr,
                'trend': acwr_trend,
                'dataframe': df_acwr
            },
            'hr_drift': {
                'latest': latest_hr_drift,
                'trend': hr_drift_trend,
                'dataframe': df_hr_drift
            },
            'cadence': {
                'latest': latest_cadence_cv,
                'trend': cadence_cv_trend,
                'dataframe': df_cadence
            }
        }

    # --- Dashboard Component Building ---

    def build_dashboard_layout():
        """Create the complete dashboard layout"""
        # Load data
        dashboard_data = load_dashboard_data()
        
        # Extract data for cards
        acwr_value = dashboard_data['acwr']['latest'] or 0
        acwr_trend = dashboard_data['acwr']['trend'] or [0] * 7
        
        hrd_value = dashboard_data['hr_drift']['latest'] or 0
        hrd_trend = dashboard_data['hr_drift']['trend'] or [0] * 7
        
        cv_value = dashboard_data['cadence']['latest'] or 0
        cv_trend = dashboard_data['cadence']['trend'] or [0] * 7
        
        # Build dashboard cards
        dashboard_cards = build_dashboard_cards(acwr_value, acwr_trend, hrd_value, hrd_trend, cv_value, cv_trend)

        return dashboard_cards
    
    app_layout = dbc.Container(
        [
            html.H1("Athlete Performance Dashboard", className="mb-4"),
            build_dashboard_layout(),
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
