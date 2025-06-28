import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import sqlite3
import pandas as pd

import utils.db.db_utils as db_utils
import utils.db.dash_db_utils as dash_db_utils
import plotly.io as pio


brand_template = dict(
    layout=dict(
        font=dict(family="Arial, sans-serif", color="var(--text)"),
        paper_bgcolor="var(--background)",
        plot_bgcolor="var(--white)",
        colorway=["var(--primary)", "var(--accent)", "var(--primary-light)", "var(--accent-light)"],
        title=dict(font=dict(size=20, color="var(--primary)")),
        xaxis=dict(gridcolor="var(--gray-light)", zerolinecolor="var(--gray-light)"),
        yaxis=dict(gridcolor="var(--gray-light)", zerolinecolor="var(--gray-light)"),
        legend=dict(font=dict(color="var(--text)"), bgcolor="var(--white)", bordercolor="var(--gray-light)")
    )
)
pio.templates["brand"] = brand_template
pio.templates.default = "brand"

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

        'ACWR': {
        "Safest": "0.8-1.0",
        "Progressive": "1.0-1.3",
        "Caution": "1.3-1.5",
        "High Risk": ">1.5"
    }
        """
        badge = status_badge(
            acwr_value, 
            [0.5, 0.8, 1.3, 1.5], 
            ["Undertraining", "Safe Loading", "Optimal Loading", "High Risk", "Very High Risk"],
            ["warning", "success", "primary", "warning", "danger"]
        )
        
        value_text = f"{acwr_value:.2f}" if acwr_value is not None else "No data"
        description = """Acute:Chronic Workload Ratio compares your last 7 days of training to your previous 28 days. The optimal range (0.8-1.3) indicates safe progression. Values above 1.5 significantly increase injury risk."""
        
        return metric_card(
            "Acute:Chronic Workload Ratio (ACWR) Trend", 
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

        'Heart_Rate_Drift': {
        "Elite": "<3%",
        "Trained": "3-5%",
        "Recreational": "5-8%", 
        "Beginner": ">8%"
    },
    },
        """
        badge = status_badge(
            hrd_value, 
            [3, 5, 8, 10],  # Percentage values 
            ["Excellent", "Good", "Moderate", "Poor", "Very Poor"],
            ["success", "primary", "warning", "danger", "dark"]
        )
        
        value_text = f"{hrd_value:.1f}%" if hrd_value is not None else "No data"
        description = """Heart Rate Drift measures the % increase in HR from first to second half of steady efforts. Low drift (<5%) indicates good aerobic efficiency. High drift (>8%) suggests fatigue or detraining."""
        
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

        'Cadence_Stability': {
        "Elite": "<3%",
        "Trained": "3-6%",
        "Recreational": "6-10%",
        "Beginner": ">10%"
    },
        """
        badge = status_badge(
            cv_value, 
            [2, 5, 8, 12],  # Percentage coefficient of variation 
            ["Highly Stable", "Stable", "Moderate Variation", "Variable", "Highly Variable"],
            ["success", "primary", "warning", "danger", "dark"]
        )
        
        value_text = f"{cv_value:.1f}%" if cv_value is not None else "No data"
        description = """Cadence Stability assesses consistency of your stride rate across different paces. Lower variation (<5%) indicates efficient running mechanics. Higher values may suggest fatigue or biomechanical inefficiency."""
        
        return metric_card(
            "Cadence Stability", 
            value_text, 
            badge, 
            description, 
            trend_data,
            reference_line=4.0  # Reference line at the optimal threshold
        )
    
    def ctl_card(ctl_value, trend_data):
        """
            Generate a Fitness, Fatigue, or Form card based on CTL, ATL, or TSB.
        
                Metrics:
            - CTL (Chronic Training Load): 28-day exponentially weighted training load.
                Calculation:
            - Training Load is proxied by 'kilojoules' for now (can be refined).
            - CTL = 28-day EMA of training load.
            Thresholds:
            - CTL: [50, 100] (Fitness Load)
            Parameters:
            - value: Latest value for the metric.
            - trend_data: Historical trend data (list).
            """
        badge = status_badge(
            ctl_value, 
            [30, 50, 80, 100],  # Fitness levels 
            ["Untrained", "Base Fitness", "Good Fitness", "High Fitness", "Elite"],
            ["danger", "warning", "primary", "success", "info"],
        )
        
        value_text = f"{ctl_value:.1f}" if ctl_value is not None else "No data"
        description = """Chronic Training Load (CTL) represents your fitness level. Higher values indicate better fitness, but increases should be gradual (5-8 points/month)"""
        
        return metric_card(
            "CTL", 
            value_text, 
            badge, 
            description, 
            trend_data,
            reference_line=80.0  # Reference line at the optimal threshold
        )
    
    def atl_card(atl_value, trend_data):
            """
                Generate a Fitness, Fatigue, or Form card based on CTL, ATL, or TSB.
            
                """
            badge = status_badge(
                atl_value, 
                [25, 40, 70, 90],   # Fatigue levels 
                ["Minimal Fatigue", "Low Fatigue", "Moderate Fatigue", "High Fatigue", "Severe Fatigue"],
                ["success", "primary", "warning", "danger", "dark"]
            )
            
            value_text = f"{atl_value:.1f}" if atl_value is not None else "No data"
            description = f"""Acute Training Load (ATL) reflects recent training stress and fatigue. Spikes indicate higher short-term fatigue."""
            
            return metric_card(
                "ATL", 
                value_text, 
                badge, 
                description, 
                trend_data,
                reference_line=70.0  # Reference line at the optimal threshold
            )
    
    def tsb_card(tsb_value, trend_data):
        """
            Generate a Fitness, Fatigue, or Form card based on CTL, ATL, or TSB.
        
            """
        badge = status_badge(
            tsb_value, 
            [-25, -10, 5, 15],  # Form/balance levels 
            ["Overreached", "Fatigued", "Balanced", "Fresh", "Very Fresh"],
            ["danger", "warning", "success", "primary", "info"],
        )
        
        value_text = f"{tsb_value:.1f}" if tsb_value is not None else "No data"
        description = """Training Stress Balance (TSB) shows your form. Negative values indicate fatigue, positive values indicate freshness. Target +5 to +15 for peak performance."""
        
        return metric_card(
            "TSB", 
            value_text, 
            badge, 
            description, 
            trend_data,
            reference_line=0.0  # Reference line at the optimal threshold
        )
    
    def tss_card(tss_value, trend_data):
        """
            Generate a Fitness, Fatigue, or Form card based on CTL, ATL, or TSB.
        
            """
        badge = status_badge(
            tss_value, 
            [50, 80, 120, 150],  # Daily training load
            ["Easy", "Moderate", "Hard", "Very Hard", "Extreme"],
            ["success", "primary", "warning", "danger", "dark"]
        )
        
        value_text = f"{tss_value:.1f}" if tss_value is not None else "No data"
        description = f"""Training Stress Score (TSS) measures workout intensity. Daily scores reflect workout difficulty, while weekly TSS targets should align with training goals."""
        
        return metric_card(
            "TSS", 
            value_text, 
            badge, 
            description, 
            trend_data,
            reference_line=100.0  # Reference line at the optimal threshold
        )
    
    def effidiency_index_card(efficiency_index_value, trend_data):
        """
            Generate a Fitness, Fatigue, or Form card based on CTL, ATL, or TSB.

            reference_values = {
            'efficiency_factor': {
            "Elite": ">0.40",
            "Advanced": "0.35-0.40",
            "Intermediate": "0.30-0.35",
            "Beginner": "0.25-0.30",
            "Untrained": "<0.25"
    }
        
            """
        badge = status_badge(
            efficiency_index_value, 
            [0.25, 0.30, 0.35, 0.40],
            ["Poor", "Fair", "Good", "Very Good", "Excellent"],
            ["danger", "warning", "primary", "success", "info"]
        )
        
        value_text = f"{efficiency_index_value:.1f}" if efficiency_index_value is not None else "No data"
        description = f"""Efficiency Index normalizes EF for pace, allowing comparison across different workout intensities."""
        
        return metric_card(
            "Efficiency Index", 
            value_text, 
            badge, 
            description, 
            trend_data,
            reference_line=0.30  # Reference line at the optimal threshold
        )
    
    def ef_7day_card(ef_7day_value, trend_data):
        """
            Generate a Fitness, Fatigue, or Form card based on CTL, ATL, or TSB.

            reference_values = {
            'efficiency_factor': {
            "Elite": ">0.40",
            "Advanced": "0.35-0.40",
            "Intermediate": "0.30-0.35",
            "Beginner": "0.25-0.30",
            "Untrained": "<0.25"
    }
        
            """
        badge = status_badge(
            ef_7day_value, 
            [0.25, 0.30, 0.35, 0.40],
            ["Poor", "Fair", "Good", "Very Good", "Excellent"],
            ["danger", "warning", "primary", "success", "info"]
        )
        
        value_text = f"{ef_7day_value:.1f}" if ef_7day_value is not None else "No data"
        description = f"""7-day average Efficiency Factor shows your recent aerobic efficiency trend."""
        
        return metric_card(
            "7 Day Efficiency Factor", 
            value_text, 
            badge, 
            description, 
            trend_data,
            reference_line=0.30  # Reference line at the optimal threshold
        )

    def ef_90day_card(ef_90day_value, trend_data):
        """
            Generate a Fitness, Fatigue, or Form card based on CTL, ATL, or TSB.

            reference_values = {
            'efficiency_factor': {
            "Elite": ">0.40",
            "Advanced": "0.35-0.40",
            "Intermediate": "0.30-0.35",
            "Beginner": "0.25-0.30",
            "Untrained": "<0.25"
    }
        
            """
        badge = status_badge(
            ef_90day_value, 
            [0.25, 0.30, 0.35, 0.40],
            ["Poor", "Fair", "Good", "Very Good", "Excellent"],
            ["danger", "warning", "primary", "success", "info"]
        )
        
        value_text = f"{ef_90day_value:.1f}" if ef_90day_value is not None else "No data"
        description = f"""90-day average Efficiency Factor shows your long-term aerobic efficiency progress."""
        
        return metric_card(
            "90 Day Efficiency Factor", 
            value_text, 
            badge, 
            description, 
            trend_data,
            reference_line=0.30 # Reference line at the optimal threshold
        )
    
    def create_training_shape_chart(conn: sqlite3.Connection):
        """
        Create a Plotly Dash chart showing training shape over time
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Get the data
        df = dash_db_utils.get_enhanced_training_shape_data(conn)

        if df.empty:
            # Return empty figure if no data
            fig = go.Figure()
            fig.update_layout(title="No training data available")
            return fig
        
        # Create subplots with 2 rows
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Training Shape", "Weekly Distance (km)"),
            row_heights=[0.7, 0.3]
        )
        
        # Add Training Shape line
        fig.add_trace(
            go.Scatter(
                x=df['start_date'],
                y=df['training_shape'],
                mode='lines',
                name='Training Shape',
                line=dict(color='rgb(26, 118, 255)', width=3),
                hovertemplate='%{y:.1f}',
            ),
            row=1, col=1
        )

        # Add Freshness Score line
        fig.add_trace(
            go.Scatter(
                x=df['start_date'],
                y=df['freshness_score'],
                mode='lines',
                name='Freshness Score',
                line=dict(color='rgb(0, 64, 0)', width=2),
                hovertemplate='%{y:.1f}',
                visible='legendonly'
            ),
            row=1, col=1
        )

        
        # # Add fitness_trend line
        # fig.add_trace(
        #     go.Scatter(
        #         x=df['start_date'],
        #         y=df['fitness_trend'],
        #         mode='lines',
        #         name='fitness_trend Score',
        #         line=dict(color='rgb(0, 64, 100)', width=2, dash='dashdot'),
        #         hovertemplate='%{y:.1f}',
        #         visible='legendonly'
        #     ),
        #     row=1, col=1
        # )

        
        # # Add form_trend line
        # fig.add_trace(
        #     go.Scatter(
        #         x=df['start_date'],
        #         y=df['form_trend'],
        #         mode='lines',
        #         name='form_trend Score',
        #         line=dict(color='rgb(100, 64, 0)', width=2, dash='dashdot'),
        #         hovertemplate='%{y:.1f}',
        #         visible='legendonly'
        #     ),
        #     row=1, col=1
        # )
        
        # Add weekly distance bars
        fig.add_trace(
            go.Bar(
                x=df['start_date'],
                y=df['weekly_distance_km'],
                name='Weekly Distance',
                marker_color='rgba(58, 71, 80, 0.6)',
                hovertemplate='%{y:.1f} km',
            ),
            row=2, col=1
        )
        
        # Add threshold lines for Training Shape
        fig.add_shape(
            type="line",
            x0=df['start_date'].min(),
            y0=80,
            x1=df['start_date'].max(),
            y1=80,
            line=dict(
                color="green",
                width=2,
                dash="dash",
            ),
            row=1, col=1
        )
        
        fig.add_shape(
            type="line",
            x0=df['start_date'].min(),
            y0=50,
            x1=df['start_date'].max(),
            y1=50,
            line=dict(
                color="orange",
                width=2,
                dash="dash",
            ),
            row=1, col=1
        )
        
        # Add annotations for the threshold lines
        fig.add_annotation(
            x=df['start_date'].max(),
            y=80,
            text="Excellent",
            showarrow=False,
            xshift=10,
            font=dict(color="green"),
            row=1, col=1
        )
        
        fig.add_annotation(
            x=df['start_date'].max(),
            y=50,
            text="Good",
            showarrow=False,
            xshift=10,
            font=dict(color="orange"),
            row=1, col=1
        )
        
        # Update layout
        fig.update_layout(
            title="Athlete Training Shape Over Time",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='white',
            height=800,
        )
        
        # Update y-axes
        fig.update_yaxes(
            title_text="Training Shape (0-100)",
            range=[0, 100],
            gridcolor='lightgray',
            row=1, col=1
        )
        
        fig.update_yaxes(
            title_text="Distance (km)",
            gridcolor='lightgray',
            row=2, col=1
        )
        
        # Update x-axis
        fig.update_xaxes(
            title_text="Date",
            gridcolor='lightgray',
            row=2, col=1,
            rangeslider_visible=False
        )
        
        return fig
    
    def create_cumulative_training_shape_chart(conn: sqlite3.Connection):
        """
        Create a Plotly Dash chart showing training shape over time
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Get the data
        df = dash_db_utils.get_cumulative_training_shape_data(conn)

        if df.empty:
            # Return empty figure if no data
            fig = go.Figure()
            fig.update_layout(title="No training data available")
            return fig
        
        # Create subplots with 2 rows
        fig = make_subplots(
            rows=1, cols=1,
            vertical_spacing=0.1,
            subplot_titles=("Cumulative Training Shape"),
            row_heights=[1]
        )
        
        # Add Training Shape line
        fig.add_trace(
            go.Scatter(
                x=df['start_date'],
                y=df['fitness_score'],
                mode='lines',
                name='Cumulative Fitness Score',
                line=dict(color='rgb(26, 118, 255)', width=3),
                hovertemplate='%{y:.1f}',
            ),
            row=1, col=1
        )

        return fig
    
    def build_fitness_chart():
        with db_utils.get_db_connection(db_path) as conn:
            training_shape_figure = create_training_shape_chart(conn)
        
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Training Shape - Fitness Trajectory"),
                    dbc.CardBody([
                        dcc.Graph(
                            id='training-shape-chart',
                            figure=training_shape_figure
                        )
                    ])
                ])
            ])
        ])

    def build_cumulative_fitness_chart():
        with db_utils.get_db_connection(db_path) as conn:
            cumulative_figure = create_cumulative_training_shape_chart(conn)

        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Cumulative Fitness Bank"),
                    dbc.CardBody([
                        dcc.Graph(
                            id='cumulative-shape-chart',
                            figure=cumulative_figure
                        )
                    ])
                ])
            ])
        ])

    def build_dashboard_cards(acwr_value, acwr_trend, hrd_value, hrd_trend, cv_value, cv_trend, ctl_value, ctl_trend, atl_value, atl_trend, tsb_value, tsb_trend, tss_value, tss_trend):
        """Build the complete dashboard row with all metric cards"""
        return dbc.Row([
            dbc.Col(acwr_card(acwr_value, acwr_trend), md=4),
            dbc.Col(hrd_card(hrd_value, hrd_trend), md=4),
            dbc.Col(cadence_card(cv_value, cv_trend), md=4),
            dbc.Col(ctl_card(ctl_value, ctl_trend), md=4),
            dbc.Col(atl_card(atl_value, atl_trend), md=4),
            dbc.Col(tsb_card(tsb_value, tsb_trend), md=4),
            dbc.Col(tss_card(tss_value, tss_trend), md=4),
        ], className="mb-4")
    
    def build_dashboard_efficiency_cards(efficiency_index_value, efficiency_index_trend, ef_7day_value, ef_7day_trend, ef_90day_value, ef_90day_trend):
        """Build the efficiency dashboard row with all three metric cards"""
        return dbc.Row([
            dbc.Col(effidiency_index_card(efficiency_index_value, efficiency_index_trend), md=4),
            dbc.Col(ef_7day_card(ef_7day_value, ef_7day_trend), md=4),
            dbc.Col(ef_90day_card(ef_90day_value, ef_90day_trend), md=4),
        ], className="mb-4")
 
    # --- Load and Process Data ---

    def load_dashboard_data(db_path):
        """Load all required data for the dashboard"""
        # Load data
        with db_utils.get_db_connection(db_path) as conn:
            # Pass the connection object to each db utility function
            df_acwr = dash_db_utils.get_acwr_data(conn)
            df_hr_drift = dash_db_utils.get_hr_drift_data(conn)
            df_cadence = dash_db_utils.get_cadence_stability_data(conn)
            
            # This function now BOTH calculates and SAVES the data to the database
            df_ctl_atl_tsb_tss = dash_db_utils.get_ctl_atl_tsb_tss_data(conn)
        
        # Process ACWR data
        latest_acwr = df_acwr['acwr'].iloc[0] if not df_acwr.empty and 'acwr' in df_acwr.columns else None
        acwr_trend = df_acwr['acwr'].head(90).tolist()[::-1] if not df_acwr.empty and 'acwr' in df_acwr.columns else []
        
        # Process HR drift data
        latest_hr_drift = df_hr_drift['hr_drift_pct'].head(90).mean() if not df_hr_drift.empty and 'hr_drift_pct' in df_hr_drift.columns else None
        hr_drift_trend = df_hr_drift['hr_drift_pct'].head(90).tolist()[::-1] if not df_hr_drift.empty and 'hr_drift_pct' in df_hr_drift.columns else []
        
        # Process cadence data
        latest_cadence_cv = df_cadence['cadence_cv'].head(90).mean() if not df_cadence.empty and 'cadence_cv' in df_cadence.columns else None
        cadence_cv_trend = df_cadence['cadence_cv'].head(90).tolist()[::-1] if not df_cadence.empty and 'cadence_cv' in df_cadence.columns else []
        
        # Process CTL data
        latest_ctl = df_ctl_atl_tsb_tss['CTL'].iloc[-1] if not df_ctl_atl_tsb_tss.empty and 'CTL' in df_ctl_atl_tsb_tss.columns else None
        ctl_trend = df_ctl_atl_tsb_tss['CTL'].tail(90).tolist()[::1] if not df_ctl_atl_tsb_tss.empty and 'CTL' in df_ctl_atl_tsb_tss.columns else []

        # Process ATL data
        latest_atl = df_ctl_atl_tsb_tss['ATL'].iloc[-1] if not df_ctl_atl_tsb_tss.empty and 'ATL' in df_ctl_atl_tsb_tss.columns else None
        atl_trend = df_ctl_atl_tsb_tss['ATL'].tail(90).tolist()[::1] if not df_ctl_atl_tsb_tss.empty and 'ATL' in df_ctl_atl_tsb_tss.columns else []

        # Process TSB data
        latest_tsb = df_ctl_atl_tsb_tss['TSB'].iloc[-1] if not df_ctl_atl_tsb_tss.empty and 'TSB' in df_ctl_atl_tsb_tss.columns else None
        tsb_trend = df_ctl_atl_tsb_tss['TSB'].tail(90).tolist()[::1] if not df_ctl_atl_tsb_tss.empty and 'TSB' in df_ctl_atl_tsb_tss.columns else []

        # Process TSS data
        latest_tss = df_ctl_atl_tsb_tss['tss'].iloc[-1] if not df_ctl_atl_tsb_tss.empty and 'tss' in df_ctl_atl_tsb_tss.columns else None
        tss_trend = df_ctl_atl_tsb_tss['tss'].tail(90).tolist()[::1] if not df_ctl_atl_tsb_tss.empty and 'tss' in df_ctl_atl_tsb_tss.columns else []
        
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
            },
            'CTL': {
                'latest': latest_ctl,
                'trend': ctl_trend,
                'dataframe': df_ctl_atl_tsb_tss  
            },
            'ATL': {
                'latest': latest_atl,
                'trend': atl_trend,
                'dataframe': df_ctl_atl_tsb_tss  
            },
            'TSB': {
                'latest': latest_tsb,
                'trend': tsb_trend,
                'dataframe': df_ctl_atl_tsb_tss  
            },
            'tss': {
                'latest': latest_tss,
                'trend': tss_trend,
                'dataframe': df_ctl_atl_tsb_tss  
            }
        }
        
    def load_dashboard_efficiency_data(db_path):
        """Load all required data for the efficiency metrics"""
        with db_utils.get_db_connection(db_path) as conn:
            efficiency = dash_db_utils.get_efficiency_index(conn)

        # Process  data
        latest_efficiency_index = efficiency['flat_efficiency_factor'].iloc[-1] if not efficiency.empty and 'efficiency_index' in efficiency.columns else None
        efficiency_index_trend = efficiency['flat_efficiency_factor'].tolist()[::1] if not efficiency.empty and 'efficiency_index' in efficiency.columns else []
    
        _7day_ef = efficiency.dropna(subset=['ef_7day'])
        latest_ef_7day = _7day_ef['ef_7day'].iloc[-1] if not _7day_ef.empty and 'ef_7day' in _7day_ef.columns else None
        ef_7day_trend = _7day_ef['ef_7day'].tail(7).tolist()[::1] if not _7day_ef.empty and 'ef_7day' in _7day_ef.columns else []
    
        _90day_ef = efficiency.dropna(subset=['ef_90day'])
        latest_ef_90day = _90day_ef['ef_90day'].iloc[-1] if not _90day_ef.empty and 'ef_90day' in _90day_ef.columns else None
        ef_90day_trend = _90day_ef['ef_90day'].tail(90).tolist()[::1] if not _90day_ef.empty and 'ef_90day' in _90day_ef.columns else []
    
    
        return {
                'efficiency_index': {
                    'latest': latest_efficiency_index,
                    'trend': efficiency_index_trend,
                    'dataframe': efficiency
                },
                'ef_7day': {
                    'latest': latest_ef_7day,
                    'trend': ef_7day_trend,
                    'dataframe': _7day_ef
                },
                'ef_90day': {
                    'latest': latest_ef_90day,
                    'trend': ef_90day_trend,
                    'dataframe': _90day_ef
                }
        }


    # --- Dashboard Component Building ---

    def build_dashboard_layout():
        """Create the complete dashboard layout"""
        # Load data
        dashboard_data = load_dashboard_data(db_path)
        
        # Extract data for cards
        acwr_value = dashboard_data['acwr']['latest'] or 0
        acwr_trend = dashboard_data['acwr']['trend'] or [0] * 7
        
        hrd_value = dashboard_data['hr_drift']['latest'] or 0
        hrd_trend = dashboard_data['hr_drift']['trend'] or [0] * 7
        
        cv_value = dashboard_data['cadence']['latest'] or 0
        cv_trend = dashboard_data['cadence']['trend'] or [0] * 7

        ctl_value = dashboard_data['CTL']['latest'] or 0
        ctl_trend = dashboard_data['CTL']['trend'] or [0] * 7

        atl_value = dashboard_data['ATL']['latest'] or 0
        atl_trend = dashboard_data['ATL']['trend'] or [0] * 7

        tsb_value = dashboard_data['TSB']['latest'] or 0
        tsb_trend = dashboard_data['TSB']['trend'] or [0] * 7

        tss_value = dashboard_data['tss']['latest'] or 0
        tss_trend = dashboard_data['tss']['trend'] or [0] * 7
        
        # Build dashboard cards
        dashboard_cards = build_dashboard_cards(acwr_value, acwr_trend, hrd_value, hrd_trend, cv_value, cv_trend, ctl_value, ctl_trend, atl_value, atl_trend, tsb_value, tsb_trend, tss_value, tss_trend)

        return dashboard_cards
    
    def build_dashboard_efficiency_layout():
        """Create the complete dashboard layout"""
        # Load data
        dashboard_data = load_dashboard_efficiency_data(db_path)
        
        # Extract data for cards
        efficiency_index_value = dashboard_data['efficiency_index']['latest'] or 0
        efficiency_index_trend = dashboard_data['efficiency_index']['trend'] or [0] * 7
        
        ef_7day_value = dashboard_data['ef_7day']['latest'] or 0
        ef_7day_trend = dashboard_data['ef_7day']['trend'] or [0] * 7
        ef_90day_value = dashboard_data['ef_90day']['latest'] or 0
        ef_90day_trend = dashboard_data['ef_90day']['trend'] or [0] * 7
        
        # Build dashboard efficiency cards
        dashboard_cards = build_dashboard_efficiency_cards(efficiency_index_value, efficiency_index_trend, ef_7day_value, ef_7day_trend, ef_90day_value, ef_90day_trend)

        return dashboard_cards
    
    app_layout = dbc.Container(
     [
        # Styled Header
        html.Header(
            dbc.Container(
        [
            html.A(
                [
                    html.H1(
                        ["RUNNING", html.Span("HUB")],
                        className="logo"
                    ),
                    html.P(
                        "Track, analyze, and improve your running performance",
                        className="tagline"
                    )
                ],
                href="/"  # <-- Link to your homepage
            )
        ],
        className="container"
    ),
    className="site-header"
),

        # Dashboard Content
 html.Main(
    [
        html.H2(
            "Athlete Performance Dashboard - 90 Days",
            className="display-4 mb-4",
            style={'textAlign': 'center'}
        ),
        html.P(
            "These are general metrics where the displayed value is a 90-day average, and the sparklines show point-in-time calculations for a single activity or day.",
            className="lead text-muted mb-5",
            style={'textAlign': 'center', 'maxWidth': '800px', 'margin': '0 auto'}
        ),

        build_dashboard_layout(),

        html.H3(
            "Fitness Trends",
            className="mb-3 mt-5",
            style={'textAlign': 'center'}
        ),
        build_fitness_chart(),

        html.H4(
            "Training Shape",
            className="mb-2 mt-4",
            style={'textAlign': 'center'}
        ),
        html.P(
            "Training Shape is a comprehensive score (0–100) that reflects how well your training is balanced across fitness, speed, efficiency, and recovery. It adapts to your personal heart rate data and workload history to help you gauge your readiness to perform. A higher score suggests you're not just fit — you're fresh, efficient, and fast.",
            className="text-muted mb-4",
            style={'textAlign': 'center', 'maxWidth': '800px', 'margin': '0 auto'}
        ),

        html.H4(
            "Freshness Score",
            className="mb-2 mt-4",
            style={'textAlign': 'center'}
        ),
        html.P(
            "Freshness Score shows how well you've recovered from recent training stress, scaled from 0 to 100. It’s derived from the balance between your short-term and long-term training load — high freshness usually means you're ready to train or race, while low scores may indicate lingering fatigue.",
            className="text-muted mb-5",
            style={'textAlign': 'center', 'maxWidth': '800px', 'margin': '0 auto'}
        ),

        html.H3(
            "Athlete Efficiency Metrics",
            className="mb-3 mt-5",
            style={'textAlign': 'center'}
        ),
        build_dashboard_efficiency_layout(),
    ],
    className="dashboard-content"
),
dbc.Button(
    "🔝", 
    href="/", 
    className="back-button", 
    title="Back to Dashboard", 
    color="link"
),],
    fluid=True
)

    # App Layout
    external_stylesheets = [
    dbc.themes.BOOTSTRAP,          # Bootstrap theme (can be CYBORG, FLATLY, etc.)
    "/static/css/styles.css"        # Your custom CSS from Flask's static folder
]
    dash_app = dash.Dash(__name__, server=server, url_base_pathname='/dashboard/', external_stylesheets=external_stylesheets)
    dash_app.title = "Running Trends Dashboard"
    dash_app.layout = app_layout

    return dash_app
