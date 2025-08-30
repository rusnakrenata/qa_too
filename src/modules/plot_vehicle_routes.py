import folium
from folium import FeatureGroup
import geopandas as gpd
import pandas as pd
import logging
from shapely.geometry import LineString, MultiLineString

logger = logging.getLogger(__name__)

def plot_vehicle_routes(
    edges_gdf: gpd.GeoDataFrame,
    nodes_gdf: gpd.GeoDataFrame,
    vehicle_routes_df: pd.DataFrame,
    vehicles_df: pd.DataFrame | None = None,
    show_route_nodes: bool = True,
    show_route_edges: bool = True,
    show_basemap_edges: bool = False,
    zoom_start: int = 25,
    color_by_vehicle: bool = False,
    selected_vehicle_ids: list[int] | None = None  # NEW: highlight selected vehicles
) -> folium.Map | None:
    """
    Folium map with optional route selection highlighting.
    """
    # Validate inputs
    if edges_gdf is None or edges_gdf.empty or not {"edge_id", "geometry"}.issubset(edges_gdf.columns):
        logger.warning("edges_gdf must include ['edge_id','geometry'].")
        return None
    if nodes_gdf is None or nodes_gdf.empty or not {"node_id", "geometry"}.issubset(nodes_gdf.columns):
        logger.warning("nodes_gdf must include ['node_id','geometry'].")
        return None
    if vehicle_routes_df is None or vehicle_routes_df.empty:
        logger.warning("vehicle_routes_df is empty.")
        return None

    # Ensure CRS
    try:
        edges = edges_gdf.to_crs(epsg=4326) if edges_gdf.crs != "EPSG:4326" else edges_gdf.copy()
        nodes = nodes_gdf.to_crs(epsg=4326) if nodes_gdf.crs != "EPSG:4326" else nodes_gdf.copy()
    except Exception:
        edges = edges_gdf.copy()
        nodes = nodes_gdf.copy()

    # Init map
    minx, miny, maxx, maxy = edges.total_bounds
    center_lat = (miny + maxy) / 2
    center_lon = (minx + maxx) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, tiles="cartodbpositron")
    m.fit_bounds([[miny, minx], [maxy, maxx]])

    # Optional: all base edges
    if show_basemap_edges:
        layer_edges = FeatureGroup(name="All City Edges", show=True)
        for _, row in edges.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            geoms = [geom] if isinstance(geom, LineString) else getattr(geom, "geoms", [geom])
            for line in geoms:
                if not hasattr(line, "coords"):
                    continue
                coords = [(lat, lon) for lon, lat in line.coords]
                folium.PolyLine(coords, color="#c9c9c9", weight=1, opacity=0.6).add_to(layer_edges)
        layer_edges.add_to(m)

    # Build lookups
    edge_geom_map = edges.set_index("edge_id")["geometry"].to_dict()
    node_geom_map = nodes.set_index("node_id")["geometry"].to_dict()

    # Color palette
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf"
    ]

    # Optional O/D markers
    if vehicles_df is not None and not vehicles_df.empty:
        required = {"vehicle_id","origin_lat","origin_lon","destination_lat","destination_lon"}
        if required.issubset(vehicles_df.columns):
            layer_origins = FeatureGroup(name="Origins", show=True)
            layer_destinations = FeatureGroup(name="Destinations", show=True)
            for _, v in vehicles_df.iterrows():
                folium.CircleMarker(
                    location=[float(v["origin_lat"]), float(v["origin_lon"])],
                    radius=2, color="#185075", fill=True, fill_opacity=0.9,
                    popup=f"Vehicle {v['vehicle_id']} Origin", tooltip=f"Veh {v['vehicle_id']} O"
                ).add_to(layer_origins)
                folium.CircleMarker(
                    location=[float(v["destination_lat"]), float(v["destination_lon"])],
                    radius=2, color="#146122", fill=True, fill_opacity=0.9,
                    popup=f"Vehicle {v['vehicle_id']} Destination", tooltip=f"Veh {v['vehicle_id']} D"
                ).add_to(layer_destinations)
            layer_origins.add_to(m)
            layer_destinations.add_to(m)
        else:
            logger.warning(f"vehicles_df missing columns: {required - set(vehicles_df.columns)}")

    # Draw routes
    for i, (_, r) in enumerate(vehicle_routes_df.iterrows()):
        vid = r.get("vehicle_id")
        is_selected = selected_vehicle_ids and vid in selected_vehicle_ids
        color = palette[i % len(palette)] if color_by_vehicle else "#A8A9AC"

        route_layer = FeatureGroup(name=f"Route – Veh {vid}", show=show_route_edges)

        if show_route_edges:
            path_edges = r.get("path_edge_ids") or []
            route_lines = []
            for eid in path_edges:
                geom = edge_geom_map.get(int(eid))
                if geom is None or geom.is_empty:
                    continue
                if isinstance(geom, LineString):
                    route_lines.append(geom)
                elif isinstance(geom, MultiLineString):
                    route_lines.extend(list(geom.geoms))

            for line in route_lines:
                coords = [(lat, lon) for lon, lat in line.coords]
                folium.PolyLine(
                    coords,
                    color=color,
                    weight=1 if is_selected else 1,
                    opacity=1.0 if is_selected else 0.6,
                    tooltip=f"Vehicle {vid} – {len(path_edges)} edges"
                ).add_to(route_layer)

        if show_route_nodes:
            nodes_layer = FeatureGroup(name=f"Route Nodes – Veh {vid}", show=True)
            path_nodes = r.get("path_node_ids") or []
            for nid in path_nodes:
                ng = node_geom_map.get(int(nid))
                if ng is None or ng.is_empty:
                    continue
                lat, lon = float(ng.y), float(ng.x)
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=1 if is_selected else 1,
                    color="#EB203B" if is_selected else "#555855",
                    fill=True,
                    fill_opacity=0.5 if is_selected else 1,
                    tooltip=f"Veh {vid} Node {nid}"
                ).add_to(nodes_layer)
            nodes_layer.add_to(m)

        route_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m
