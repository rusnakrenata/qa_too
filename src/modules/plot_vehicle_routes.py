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
    vehicles_df: pd.DataFrame | None = None,   # optional: to plot O/D dots
    show_route_nodes: bool = False,            # show path_node_ids as dots
    zoom_start: int = 14,
    color_by_vehicle: bool = True              # simple per-vehicle color rotation
) -> folium.Map | None:
    """
    Folium map:
      - Basemap edges (light gray)
      - For each vehicle in vehicle_routes_df:
          * draw polyline using path_edge_ids (concatenated from edges_gdf geometries)
          * optional: draw nodes along route (path_node_ids) as small dots
      - Optional: also plot vehicle origins/destinations from vehicles_df

    Args:
        edges_gdf: GeoDataFrame with ['edge_id','geometry'] (LineString). Prefer EPSG:4326.
        nodes_gdf: GeoDataFrame with ['node_id','geometry'] (Point). Prefer EPSG:4326.
        vehicle_routes_df: DataFrame with columns:
            ['vehicle_id','route_id','path_edge_ids','path_node_ids', ...]
            (path_edge_ids and path_node_ids are lists)
        vehicles_df: Optional DataFrame with columns:
            ['vehicle_id','origin_lat','origin_lon','destination_lat','destination_lon']
        show_route_nodes: If True, plot nodes along the path (green dots).
        zoom_start: Initial zoom level.
        color_by_vehicle: Cycle colors per vehicle for route lines.

    Returns:
        folium.Map or None
    """
    # Basic checks
    need_edges_cols = {"edge_id", "geometry"}
    if edges_gdf is None or edges_gdf.empty or not need_edges_cols.issubset(edges_gdf.columns):
        logger.warning("edges_gdf must include ['edge_id','geometry'].")
        return None
    need_nodes_cols = {"node_id", "geometry"}
    if nodes_gdf is None or nodes_gdf.empty or not need_nodes_cols.issubset(nodes_gdf.columns):
        logger.warning("nodes_gdf must include ['node_id','geometry'].")
        return None
    if vehicle_routes_df is None or vehicle_routes_df.empty:
        logger.warning("vehicle_routes_df is empty.")
        return None

    # Ensure WGS84 (don’t crash if CRS is None)
    try:
        edges = edges_gdf.to_crs(epsg=4326) if edges_gdf.crs != "EPSG:4326" else edges_gdf.copy()
    except Exception:
        edges = edges_gdf.copy()
    try:
        nodes = nodes_gdf.to_crs(epsg=4326) if nodes_gdf.crs != "EPSG:4326" else nodes_gdf.copy()
    except Exception:
        nodes = nodes_gdf.copy()

    # Map init and bounds
    minx, miny, maxx, maxy = edges.total_bounds
    center_lat = (miny + maxy) / 2
    center_lon = (minx + maxx) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, tiles="cartodbpositron")
    m.fit_bounds([[miny, minx], [maxy, maxx]])

    # Basemap edges
    layer_edges = FeatureGroup(name="Edges", show=True)
    for _, row in edges.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        # support LineString/MultiLineString
        geoms = [geom] if isinstance(geom, LineString) else getattr(geom, "geoms", [geom])
        for line in geoms:
            if not hasattr(line, "coords"):
                continue
            coords = [(lat, lon) for lon, lat in line.coords]
            folium.PolyLine(coords, color="#c9c9c9", weight=2, opacity=0.7).add_to(layer_edges)
    layer_edges.add_to(m)

    # Build quick lookups
    edge_geom_map = edges.set_index("edge_id")["geometry"].to_dict()
    node_geom_map = nodes.set_index("node_id")["geometry"].to_dict()

    # Color rotation (simple palette)
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf"
    ]

    # Optional O/D layers
    layer_origins = FeatureGroup(name="Vehicle Origins", show=vehicles_df is not None)
    layer_destinations = FeatureGroup(name="Vehicle Destinations", show=vehicles_df is not None)
    if vehicles_df is not None and not vehicles_df.empty:
        required = {"vehicle_id","origin_lat","origin_lon","destination_lat","destination_lon"}
        if required.issubset(vehicles_df.columns):
            for _, v in vehicles_df.iterrows():
                folium.CircleMarker(
                    location=[float(v["origin_lat"]), float(v["origin_lon"])],
                    radius=3, color="#1f77b4", fill=True, fill_opacity=0.9,
                    popup=f"Vehicle {v['vehicle_id']} (Origin)", tooltip=f"Veh {v['vehicle_id']} O"
                ).add_to(layer_origins)
                folium.CircleMarker(
                    location=[float(v["destination_lat"]), float(v["destination_lon"])],
                    radius=3, color="#2ca02c", fill=True, fill_opacity=0.9,
                    popup=f"Vehicle {v['vehicle_id']} (Destination)", tooltip=f"Veh {v['vehicle_id']} D"
                ).add_to(layer_destinations)
            layer_origins.add_to(m)
            layer_destinations.add_to(m)
        else:
            logger.warning(f"vehicles_df missing columns: {required - set(vehicles_df.columns)}")

    # Draw each vehicle’s route (+ optional nodes)
    for i, (_, r) in enumerate(vehicle_routes_df.iterrows()):
        vid = r.get("vehicle_id")
        route_layer = FeatureGroup(name=f"Route – Veh {vid}", show=True)

        # color choice
        color = palette[i % len(palette)] if color_by_vehicle else "#3366ff"

        # Build a combined LineString or MultiLineString from path_edge_ids
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
            else:
                continue

        if route_lines:
            # If edges are contiguous, you could stitch them; here we just draw them all.
            for line in route_lines:
                coords = [(lat, lon) for lon, lat in line.coords]
                folium.PolyLine(
                    coords, color=color, weight=4, opacity=0.9,
                    tooltip=f"Vehicle {vid} – {len(path_edges)} edges"
                ).add_to(route_layer)

        # Optional: plot nodes along the route
        if show_route_nodes:
            nodes_layer = FeatureGroup(name=f"Route Nodes – Veh {vid}", show=False)
            path_nodes = r.get("path_node_ids") or []
            for nid in path_nodes:
                ng = node_geom_map.get(int(nid))
                if ng is None or ng.is_empty:
                    continue
                lat, lon = float(ng.y), float(ng.x)
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=2,
                    color="#2ca02c",  # green
                    fill=True,
                    fill_opacity=0.8,
                    tooltip=f"Veh {vid} Node {nid}"
                ).add_to(nodes_layer)
            nodes_layer.add_to(m)

        route_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m
