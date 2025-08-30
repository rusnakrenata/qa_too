from datetime import datetime
import pandas as pd
import logging
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

def store_city_to_db(
    session: Any,
    city_name: str,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    City: Any,
    Node: Any,
    Edge: Any,
    center_coords: Optional[Tuple[float, float]] = None,
    radius_km: Optional[float] = None
) -> Any:
    """
    Store city, nodes, and edges into the database, including attraction points and route alternatives.

    Args:
        session: SQLAlchemy session.
        city_name: Name of the city.
        nodes: DataFrame of nodes.
        edges: DataFrame of edges.
        City, Node, Edge: SQLAlchemy models.
        center_coords: Optional (lat, lon) for city subset.
        radius_km: Optional radius for city subset.


    Returns:
        city: The created City object.
    """
    try:
        node_count = len(nodes)
        edge_count = len(edges)

        is_subset = center_coords is not None and radius_km is not None
        
        center_lat = center_coords[0] if center_coords else None
        center_lon = center_coords[1] if center_coords else None

  
        city = City(
            name=city_name,
            node_count=node_count,
            edge_count=edge_count,
            center_lat=center_lat,
            center_lon=center_lon,
            radius_km=radius_km,
            is_subset=is_subset,
            created_at=datetime.utcnow()
        )
        session.add(city)
        session.commit()

        for _, node in nodes.iterrows():
            node_data = {
                'city_id': city.city_id,
                'osmid': node.get('osmid'),
                'x': float(node['x']) if pd.notna(node['x']) else None,
                'y': float(node['y']) if pd.notna(node['y']) else None,
                'geometry': str(node['geometry']) if node['geometry'] is not None and pd.notna(node['geometry']) else None
            }
            session.add(Node(**node_data))
        session.commit()

        for _, edge in edges.iterrows():
            edge_data = {
                'city_id': city.city_id,
                'u': edge.get('u'),
                'v': edge.get('v'),
                'length': str(edge['length']) if pd.notna(edge.get('length')) else None,
                'geometry': str(edge['geometry']) if edge['geometry'] is not None and pd.notna(edge['geometry']) else None
            }
            session.add(Edge(**edge_data))
        session.commit()

        if is_subset:
            logger.info(f"City subset '{city_name}' stored with {node_count} nodes, {edge_count} edges.")
        else:
            logger.info(f"City '{city_name}' stored with {node_count} nodes, {edge_count} edges.")

        return city

    except Exception as e:
        logger.error(f"Error storing city to DB: {e}", exc_info=True)
        session.rollback()
        return None

