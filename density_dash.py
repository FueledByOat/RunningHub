# density_dash.py

from dash import Dash, html, dcc, Input, Output
import dash_leaflet as dl
import sqlite3
import polyline
import urllib.parse

def decode_polyline(p):
    try:
        return polyline.decode(p)
    except Exception:
        return []

def get_polylines_since(start_date, db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT map_summary_polyline FROM activities WHERE start_date >= ?", (start_date,))
    rows = cur.fetchall()
    conn.close()
    return [decode_polyline(row[0]) for row in rows if row[0]]

def get_latest_starting_coords(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""SELECT start_latlng FROM activities 
                WHERE start_latlng is not null
                
                order by start_date desc""")
    rows = cur.fetchall()[0][0].strip("[]")
    lat, lon = map(float, rows.split(",")) 
    conn.close()
    return (lat, lon)

def create_density_dash(flask_app, db_path):
    app = Dash(__name__, server=flask_app, url_base_pathname="/density_dash/")
    
    app.layout = html.Div([
        dcc.Location(id="url"),
        dl.Map(id="map", center=get_latest_starting_coords(db_path), 
               zoom=10, style={'width': '100%', 'height': '100vh'})
    ])

    @app.callback(
        Output("map", "children"),
        Output("map", "zoom"),
        Input("url", "search")
    )
    def update_map(query_string):
        query = urllib.parse.parse_qs(query_string.lstrip("?"))
        start_date = query.get("start_date", ["2024-01-01"])[0]
        zoom_str = query.get("zoom", ["10"])[0]
        try:
            zoom = int(zoom_str)
        except ValueError:
            zoom = 10

        decoded_routes = get_polylines_since(start_date, db_path)

        polylines = []
        for coords in decoded_routes:
            polylines.append(
                dl.Polyline(positions=coords, color="blue", opacity=0.1, weight=3)
            )

        return [dl.TileLayer()] + polylines, zoom

    return app