from sqlalchemy import text
import pandas as pd
import geopandas as gpd
from shapely import wkt
import logging
from typing import Any, Tuple

logger = logging.getLogger(__name__)

def get_city_data_from_db(
    session: Any,
    city_id: int
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Fetch city nodes and edges from the database and return as GeoDataFrames.

    Args:
        session: SQLAlchemy session
        city_id: City ID

    Returns:
        nodes_gdf: GeoDataFrame of nodes
        edges_gdf: GeoDataFrame of edges
    """
    try:
        nodes_query = session.execute(
            text(f"SELECT node_id, geometry AS geometry FROM nodes WHERE city_id = {city_id}")
        )
        edges_query = session.execute(
            text(f"SELECT edge_id, geometry AS geometry FROM edges WHERE city_id = {city_id}")
        )
        nodes_df = pd.DataFrame(nodes_query.fetchall(), columns=["node_id", "geometry"]) #type: ignore
        edges_df = pd.DataFrame(edges_query.fetchall(), columns=["edge_id", "geometry"]) #type: ignore
        nodes_df['geometry'] = nodes_df['geometry'].apply(wkt.loads)
        edges_df['geometry'] = edges_df['geometry'].apply(wkt.loads)
        nodes_gdf = gpd.GeoDataFrame(nodes_df, geometry='geometry', crs='EPSG:4326')
        edges_gdf = gpd.GeoDataFrame(edges_df, geometry='geometry', crs='EPSG:4326')
        logger.info(f"Fetched city data for city_id={city_id}: {len(nodes_gdf)} nodes, {len(edges_gdf)} edges.")
        return nodes_gdf, edges_gdf
    
    except Exception as e:
        logger.error(f"Error fetching city data from DB: {e}", exc_info=True)
        return gpd.GeoDataFrame(), gpd.GeoDataFrame()
    
    
