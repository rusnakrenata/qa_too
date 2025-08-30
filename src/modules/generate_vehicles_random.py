import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from geopy.distance import geodesic
from typing import Any
import logging

logger = logging.getLogger(__name__)

def generate_vehicles(
    session: Any,
    run_config_id: int,
    iteration_id: int,
    Vehicle: Any,
    nodes_gdf: gpd.GeoDataFrame,
    n_vehicles: int,
    min_length: float,
    max_length: float
) -> pd.DataFrame:
    """
    Generate vehicles by sampling ORIGIN and DESTINATION *nodes* only,
    and store lat/lon along with node ids for easy mapping.

    Expects nodes_gdf to include: ['node_id','geometry'] where geometry is Point(x=lon, y=lat).
    """
    required_cols = {'node_id', 'geometry'}
    if nodes_gdf.empty or not required_cols.issubset(nodes_gdf.columns):
        logger.error("nodes_gdf must have columns ['node_id','geometry'].")
        return pd.DataFrame(columns=[
            'vehicle_id','origin_node_id','destination_node_id',
            'origin_lat','origin_lon','destination_lat','destination_lon','distance_m'
        ])

    node_ids = nodes_gdf['node_id'].to_numpy()
    geom_series = nodes_gdf['geometry']

    vehicles_out = []
    vehicle_records = []
    vehicle_id = 0
    n_nodes = len(node_ids)
    max_retries = 100

    for _ in range(n_vehicles):
        valid = False
        retries = 0
        while not valid and retries < max_retries:
            retries += 1

            # choose two distinct nodes when possible
            if n_nodes >= 2:
                idx_o, idx_d = np.random.choice(n_nodes, size=2, replace=False)
            else:
                idx_o = idx_d = 0

            o_id = int(node_ids[idx_o])
            d_id = int(node_ids[idx_d])
            if o_id == d_id:
                continue

            o_pt = geom_series.iloc[idx_o]
            d_pt = geom_series.iloc[idx_d]
            if not isinstance(o_pt, Point) or not isinstance(d_pt, Point):
                continue

            # geometry is in EPSG:4326, Point(lon, lat)
            o_lon, o_lat = float(o_pt.x), float(o_pt.y)
            d_lon, d_lat = float(d_pt.x), float(d_pt.y)

            dist_m = geodesic((o_lat, o_lon), (d_lat, d_lon)).meters
            if min_length <= dist_m <= max_length:
                vehicle_id += 1

                # persist
                rec = Vehicle(
                    vehicle_id=vehicle_id,
                    run_configs_id=run_config_id,
                    iteration_id=iteration_id,
                    origin_node_id=o_id,
                    destination_node_id=d_id,
                    origin_lat=o_lat,
                    origin_lon=o_lon,
                    destination_lat=d_lat,
                    destination_lon=d_lon,
                )
                vehicle_records.append(rec)

                # for immediate use / plotting
                vehicles_out.append({
                    'vehicle_id': vehicle_id,
                    'origin_node_id': o_id,
                    'destination_node_id': d_id,
                    'origin_lat': o_lat,
                    'origin_lon': o_lon,
                    'destination_lat': d_lat,
                    'destination_lon': d_lon,
                    'distance_m': dist_m
                })
                valid = True

        if not valid:
            logger.warning(f"Skipped a vehicle after {max_retries} retries due to distance constraints.")

    # Save
    try:
        if vehicle_records:
            session.bulk_save_objects(vehicle_records)
            session.commit()
    except Exception as e:
        logger.error(f"Error saving vehicles to DB: {e}", exc_info=True)
        session.rollback()
    finally:
        session.close()

    df = pd.DataFrame(
        vehicles_out,
        columns=[
            'vehicle_id','origin_node_id','destination_node_id',
            'origin_lat','origin_lon','destination_lat','destination_lon','distance_m'
        ]
    )
    logger.info(f"Generated {len(df)} vehicles (with lat/lon) for run_config_id={run_config_id}, iteration_id={iteration_id}.")
    return df
