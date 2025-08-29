import random
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from geopy.distance import geodesic
from typing import Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def generate_vehicles_attraction(
    session: Any,
    run_config_id: int,
    iteration_id: int,
    Vehicle: Any,
    edges_gdf: gpd.GeoDataFrame,
    n_vehicles: int,
    min_length: float,
    max_length: float,
    attraction_point: Tuple[float, float],  # (lat, lon)
    d_alternatives: int = 3
) -> gpd.GeoDataFrame:
    """
    Generate vehicles with random origin edges and destination edge randomly selected
    from the d_alternatives nearest edges to a point of attraction.
    Vehicles are stored in DB and returned as a GeoDataFrame.

    Args:
        session: SQLAlchemy session
        Vehicle: SQLAlchemy Vehicle model
        run_config_id: Run configuration ID
        iteration_id: Iteration ID
        edges_gdf: GeoDataFrame of edges
        n_vehicles: Number of vehicles to generate
        min_length: Minimum allowed trip length (meters)
        max_length: Maximum allowed trip length (meters)
        attraction_point: Optional (lat, lon) of point of attraction
        d_alternatives: Number of nearest destination edges to consider

    Returns:
        vehicles_gdf: GeoDataFrame of generated vehicles
    """

    # 1. Default to St. Elisabeth Cathedral if no attraction point provided
    if attraction_point is None:
        raise ValueError("Attraction point must be set!")

    point_of_attraction = Point(attraction_point[1], attraction_point[0])  # lon, lat

    # 2. Reproject to UTM (EPSG:32634) for accurate distance measurements
    projected_gdf = edges_gdf.to_crs(epsg=32634)
    projected_attraction = gpd.GeoSeries([point_of_attraction], crs='EPSG:4326').to_crs(epsg=32634).iloc[0]

    # 3. Compute distances
    projected_gdf['distance_to_attraction'] = projected_gdf.geometry.distance(projected_attraction)
    edges_gdf['distance_to_attraction'] = projected_gdf['distance_to_attraction']

    # 4. Select d_alternatives nearest edges
    nearest_edges = edges_gdf.nsmallest(d_alternatives, 'distance_to_attraction')

    print("Selected closest edges to point of attraction:")
    for _, row in nearest_edges.iterrows():
        print(f" - edge_id={row['edge_id']}, distance={row['distance_to_attraction']:.2f} m")

    # 5. Generate vehicles
    vehicles = []
    vehicle_records = []
    vehicle_id = 0

    for _ in range(n_vehicles):
        valid_vehicle = False
        retries = 0
        max_retries = 100

        while not valid_vehicle and retries < max_retries:
            retries += 1

            # Random origin edge
            origin_edge = edges_gdf.sample(n=1).iloc[0]
            origin_position_on_edge = random.random()
            origin_line = origin_edge['geometry']
            origin_point = origin_line.interpolate(origin_position_on_edge, normalized=True)

            # Random destination from selected nearest edges
            dest_edge = nearest_edges.sample(n=1).iloc[0]
            destination_line = dest_edge['geometry']
            destination_edge_id = dest_edge['edge_id']
            destination_position_on_edge = random.random()
            destination_point = destination_line.interpolate(destination_position_on_edge, normalized=True)

            # Check trip distance
            distance = geodesic((origin_point.y, origin_point.x), (destination_point.y, destination_point.x)).meters

            if min_length <= distance <= max_length:
                valid_vehicle = True
                vehicle_id += 1

                vehicle = {
                    'vehicle_id': vehicle_id,
                    'origin_edge_id': origin_edge['edge_id'],
                    'origin_position_on_edge': origin_position_on_edge,
                    'origin_geometry': origin_point,
                    'destination_edge_id': destination_edge_id,
                    'destination_position_on_edge': destination_position_on_edge,
                    'destination_geometry': destination_point
                }
                vehicles.append(vehicle)

                vehicle_record = Vehicle(
                    vehicle_id=vehicle_id,
                    run_configs_id=run_config_id,
                    iteration_id=iteration_id,
                    origin_edge_id=origin_edge['edge_id'],
                    origin_position_on_edge=origin_position_on_edge,
                    origin_geometry=origin_point,
                    destination_edge_id=destination_edge_id,
                    destination_position_on_edge=destination_position_on_edge,
                    destination_geometry=destination_point
                )
                vehicle_records.append(vehicle_record)

        if not valid_vehicle:
            logger.warning(f"Skipped a vehicle after {max_retries} retries due to distance constraints.")

    # 6. Save to DB
    try:
        session.bulk_save_objects(vehicle_records)
        session.commit()
    except Exception as e:
        logger.error(f"Error saving vehicles to DB: {e}", exc_info=True)
        session.rollback()
    finally:
        session.close()

    # 7. Return as GeoDataFrame
    vehicles_df = pd.DataFrame(vehicles)
    if not vehicles_df.empty:
        vehicles_df['origin_geometry'] = vehicles_df['origin_geometry'].apply(lambda x: Point(x.x, x.y))
        vehicles_df['destination_geometry'] = vehicles_df['destination_geometry'].apply(lambda x: Point(x.x, x.y))
        vehicles_gdf = gpd.GeoDataFrame(vehicles_df, geometry='origin_geometry', crs='EPSG:4326')
    else:
        empty_data = {
            'vehicle_id': [], 'origin_edge_id': [], 'origin_position_on_edge': [], 'origin_geometry': [],
            'destination_edge_id': [], 'destination_position_on_edge': [], 'destination_geometry': []
        }
        empty_df = pd.DataFrame(empty_data)
        vehicles_gdf = gpd.GeoDataFrame(empty_df, geometry='origin_geometry', crs='EPSG:4326')

    logger.info(f"Generated {len(vehicles_gdf)} vehicles for run_config_id={run_config_id}, iteration_id={iteration_id}.")
    return vehicles_gdf
