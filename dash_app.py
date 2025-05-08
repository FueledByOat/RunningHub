# dash_app.py

from dash import Dash, html
import dash_leaflet as dl
import polyline


def decode_polyline(polyline_str):
    try:
        print(polyline.decode(polyline_str))
        return polyline.decode(polyline_str)  # List of (lat, lon)
    except Exception as e:
        print(f"Error decoding polyline: {e}")
        return []


def create_dash_app(server, polyline_str):
    coords = decode_polyline(polyline_str)

    dash_app = Dash(
        __name__,
        server=server,
        url_base_pathname="/map/"
    )

    if coords:
        polyline_layer = dl.Polyline(positions=coords, color="#0c1559")
        map_center = coords[len(coords) // 2]
    else:
        polyline_layer = dl.Polyline(positions=[], color="#0c1559")
        map_center = [0, 0]

    dash_app.layout = html.Div([
        dl.Map(center=map_center, zoom=14,
               style={'width': '100%', 'height': '500px'},
               children=[
                   dl.TileLayer(),
                   polyline_layer
               ])
    ])

    return dash_app
