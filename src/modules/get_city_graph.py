import osmnx as ox
import geopandas as gpd
import pandas as pd
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

def get_city_graph(
    city_name: str,
    center_coords: Optional[Tuple[float, float]] = None,
    radius_km: Optional[float] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download and return the city road network as node and edge DataFrames.

    Args:
        city_name: Name of the city (e.g., 'Košice, Slovakia').
        center_coords: Optional (lat, lon) tuple for the subset center.
        radius_km: Optional radius around the center point in kilometers.

    Returns:
        Tuple containing:
            - nodes: DataFrame of nodes.
            - edges: DataFrame of edges.
    """
    # Filter for relevant road types suitable for vehicle routing
    custom_filter = (
        '["highway"~"motorway|trunk|primary|secondary|tertiary|residential|unclassified|service|living_street"]'
    )

    try:
        if center_coords and radius_km:
            lat, lon = center_coords
            radius_meters = radius_km * 1000

            logger.info(f"Downloading city subset for '{city_name}' at ({lat}, {lon}) within {radius_km} km radius.")
            G = ox.graph_from_point(
                center_coords,
                dist=radius_meters,
                network_type='drive',
                custom_filter=custom_filter
            )
            G.graph['crs'] = 'epsg:4326'

            nodes, edges = ox.graph_to_gdfs(G)
            nodes.reset_index(inplace=True)
            edges.reset_index(inplace=True)

            logger.info(f"City subset downloaded: {len(nodes)} nodes, {len(edges)} edges.")

        else:
            logger.info(f"Downloading entire city graph for '{city_name}'.")
            G = ox.graph_from_place(
                city_name,
                network_type='drive',
                custom_filter=custom_filter
            )
            G.graph['crs'] = 'epsg:4326'

            nodes, edges = ox.graph_to_gdfs(G)
            nodes.reset_index(inplace=True)
            edges.reset_index(inplace=True)

            logger.info(f"Entire city graph downloaded: {len(nodes)} nodes, {len(edges)} edges.")

        return nodes, edges

    except Exception as e:
        logger.error(f"Error downloading city graph for '{city_name}': {e}", exc_info=True)
        return pd.DataFrame(), pd.DataFrame()