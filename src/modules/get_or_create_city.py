from typing import Any
from models import City, Node, Edge
from get_city_graph import get_city_graph
from store_city_to_db import store_city_to_db
from get_city_data_from_db import get_city_data_from_db

def get_or_create_city(
    session,
    city_name: str,
    center_coords: tuple | None = None,
    radius_km: float | None = None,
    attraction_point: tuple | None = None,
    d_alternatives: int | None = None,
) -> Any:
    """
    Get or create a city (full or subset) in the database.
    Returns: (City, edges_df)
    """

    # Build filters only for provided values
    is_subset = any(v is not None for v in (center_coords, attraction_point, radius_km, d_alternatives))
    filters = {"name": city_name, "is_subset": is_subset}

    if center_coords is not None:
        filters["center_lat"] = center_coords[0]
        filters["center_lon"] = center_coords[1]
    if radius_km is not None:
        filters["radius_km"] = radius_km
    if attraction_point is not None:
        filters["attraction_lat"] = attraction_point[0]
        filters["attraction_lon"] = attraction_point[1]
    if d_alternatives is not None:
        filters["d_alternatives"] = d_alternatives

    # Query for existing city
    city = session.query(City).filter_by(**filters).first()

    # If not found, build and store
    if city is None:
        nodes, edges_df = get_city_graph(city_name, center_coords=center_coords, radius_km=radius_km)
        city = store_city_to_db(
            session=session,
            city_name=city_name,
            nodes=nodes,
            edges=edges_df,
            City=City,
            Node=Node,
            Edge=Edge,
            center_coords=center_coords,
            radius_km=radius_km,
            attraction_point=attraction_point,
            d_alternatives=d_alternatives,
        )
        #city = session.query(City).filter_by(**filters).first()

    # Now it's safe to load from DB using city.city_id

    city = session.query(City).filter_by(city_id=city.city_id).first()
    nodes_df, edges_df = get_city_data_from_db(session, city.city_id)
    city_id = city.city_id

    # Don't close the session here; let the caller manage it.
    return city_id, edges_df
