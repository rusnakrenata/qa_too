import folium
from folium import FeatureGroup
import geopandas as gpd
import logging
import pandas as pd
from shapely.geometry import LineString

logger = logging.getLogger(__name__)

def plot_vehicles(
    edges_gdf: gpd.GeoDataFrame,
    vehicles_df: pd.DataFrame | None = None,
    show_od_lines: bool = False,
    zoom_start: int = 14
) -> folium.Map | None:
    """
    Interactive Folium map:
      - Draws city edges (light gray)
      - Plots vehicle origins (blue) & destinations (green) as dots
      - Optional O→D lines per vehicle

    Args:
        edges_gdf: GeoDataFrame with 'geometry' (LineString/MultiLineString). CRS EPSG:4326 preferred.
        vehicles_df: DataFrame with columns:
            ['vehicle_id','origin_lat','origin_lon','destination_lat','destination_lon']
        show_od_lines: Whether to draw faint lines between origin and destination.
        zoom_start: Initial zoom level.

    Returns:
        folium.Map or None
    """
    if edges_gdf is None or edges_gdf.empty or "geometry" not in edges_gdf.columns:
        logger.warning("edges_gdf is empty or missing geometry.")
        return None

    # Ensure WGS84
    try:
        edges = edges_gdf.to_crs(epsg=4326) if edges_gdf.crs != "EPSG:4326" else edges_gdf.copy()
    except Exception:
        edges = edges_gdf.copy()

    # Map init (fit to edges bounds)
    minx, miny, maxx, maxy = edges.total_bounds
    center_lat = (miny + maxy) / 2
    center_lon = (minx + maxx) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, tiles="cartodbpositron")
    m.fit_bounds([[miny, minx], [maxy, maxx]])

    # Layers
    layer_edges = FeatureGroup(name="Edges", show=True)
    layer_origins = FeatureGroup(name="Vehicle Origins", show=True if vehicles_df is not None else False)
    layer_destinations = FeatureGroup(name="Vehicle Destinations", show=True if vehicles_df is not None else False)
    layer_od_lines = FeatureGroup(name="O→D Lines", show=show_od_lines)

    # Draw edges (baseline)
    for _, row in edges.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        # Handle LineString and MultiLineString
        geoms = [geom] if isinstance(geom, LineString) else getattr(geom, "geoms", [])
        if not geoms:
            geoms = [geom]

        for line in geoms:
            if not hasattr(line, "coords"):  # guard for odd geometries
                continue
            coords = [(lat, lon) for lon, lat in line.coords]  # WKT is (lon, lat)
            folium.PolyLine(coords, color="#c9c9c9", weight=2, opacity=0.7).add_to(layer_edges)

    # Plot vehicles
    if vehicles_df is not None and not vehicles_df.empty:
        required = {"vehicle_id", "origin_lat", "origin_lon", "destination_lat", "destination_lon"}
        missing = required - set(vehicles_df.columns)
        if missing:
            logger.warning(f"vehicles_df missing columns: {missing}")
        else:
            # Origins
            for _, v in vehicles_df.iterrows():
                folium.CircleMarker(
                    location=[float(v["origin_lat"]), float(v["origin_lon"])],
                    radius=3,
                    color="#1f77b4",  # blue
                    fill=True,
                    fill_opacity=0.9,
                    popup=f"Vehicle {v['vehicle_id']} (Origin)",
                    tooltip=f"Veh {v['vehicle_id']} O"
                ).add_to(layer_origins)

            # Destinations
            for _, v in vehicles_df.iterrows():
                folium.CircleMarker(
                    location=[float(v["destination_lat"]), float(v["destination_lon"])],
                    radius=3,
                    color="#2ca02c",  # green
                    fill=True,
                    fill_opacity=0.9,
                    popup=f"Vehicle {v['vehicle_id']} (Destination)",
                    tooltip=f"Veh {v['vehicle_id']} D"
                ).add_to(layer_destinations)

            # Optional O→D lines
            if show_od_lines:
                for _, v in vehicles_df.iterrows():
                    folium.PolyLine(
                        locations=[
                            [float(v["origin_lat"]), float(v["origin_lon"])],
                            [float(v["destination_lat"]), float(v["destination_lon"])]
                        ],
                        color="#aaaaaa",
                        weight=1,
                        opacity=0.6
                    ).add_to(layer_od_lines)

    # Add layers and control
    layer_edges.add_to(m)
    if vehicles_df is not None and not vehicles_df.empty:
        layer_origins.add_to(m)
        layer_destinations.add_to(m)
        if show_od_lines:
            layer_od_lines.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m
