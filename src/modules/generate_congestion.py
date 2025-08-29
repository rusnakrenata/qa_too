import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import text as sa_text
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import logging
from typing import Any

logger = logging.getLogger(__name__)

def haversine_np(lat1, lon1, lat2, lon2):
    """Calculate distance in meters between two coordinates using the haversine formula."""
    R = 6371000  # Earth radius in meters
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def process_group(group_df: pd.DataFrame, time_step: int, distance_factor: float) -> pd.DataFrame:
    """
    Process a group of vehicle-route points to calculate congestion scores.

    Args:
        group_df: DataFrame containing grouped route points.
        time_step: Time interval in seconds.
        distance_factor: Factor influencing congestion distance sensitivity.

    Returns:
        DataFrame with congestion scores.
    """
    if len(group_df) < 2:
        return pd.DataFrame()

    vehicle_ids = group_df['vehicle_id'].values
    route_ids = group_df['route_id'].values
    lats = group_df['lat'].values
    lons = group_df['lon'].values
    speeds = group_df['speed'].values
    edge_id = group_df['edge_id'].iloc[0]

    cardinal = group_df['cardinal'].iloc[0].upper()
    cardinal_map = {
        'N': np.array([0, 1]), 'S': np.array([0, -1]), 'E': np.array([1, 0]), 'W': np.array([-1, 0]),
        'NE': np.array([1, 1]), 'NW': np.array([-1, 1]), 'SE': np.array([1, -1]), 'SW': np.array([-1, -1])
    }

    if cardinal not in cardinal_map:
        return pd.DataFrame()

    edge_unit_vec = cardinal_map[cardinal] / np.linalg.norm(cardinal_map[cardinal])

    positions = np.stack([lons, lats], axis=1)
    projections = positions @ edge_unit_vec

    results = []
    for i in range(len(group_df)):
        for j in range(len(group_df)):
            if i == j or projections[i] <= projections[j]:
                continue

            distance = haversine_np(lats[i], lons[i], lats[j], lons[j])
            avg_speed = (speeds[i] + speeds[j]) / 2.0

            score = np.maximum((avg_speed - distance / distance_factor) / avg_speed, 0)
            max_score = score * time_step

            if max_score > 0:
                results.append({
                    'edge_id': edge_id,
                    'vehicle_id_a': vehicle_ids[i],
                    'route_id_a': route_ids[i],
                    'vehicle_id_b': vehicle_ids[j],
                    'route_id_b': route_ids[j],
                    'congestion_score': max_score
                })

    return pd.DataFrame(results)

def generate_congestion(
    session: Any,
    run_config_id: int,
    iteration_id: int,
    time_step: int = 10,
    distance_factor: float = 4.0
) -> pd.DataFrame:
    """
    Compute pairwise congestion scores and insert results into the database.

    Args:
        session: SQLAlchemy database session.
        run_config_id: ID for the current run configuration.
        iteration_id: ID of the current iteration.
        time_step: Time interval for congestion calculation (seconds).
        distance_factor: Factor for distance sensitivity in congestion.

    Returns:
        DataFrame containing congestion results.
    """
    try:
        logger.info("Loading route_points from DB at: %s", datetime.now())
        start = datetime.now()

        query = sa_text("""
            SELECT edge_id, vehicle_id, route_id, lat, lon, speed, time, cardinal
            FROM trafficOptimization.route_points
            WHERE run_configs_id = :run_config_id AND iteration_id = :iteration_id
        """)
        df = pd.read_sql_query(query, session.bind, params={
            'run_config_id': run_config_id,
            'iteration_id': iteration_id
        })

        logger.info("Bucketing route_points for spatial-temporal filtering at: %s", datetime.now())
        df['bucket'] = (
            df['edge_id'].astype(str) + "_" +
            df['cardinal'] + "_" +
            (df['lat'] * 100).astype(int).astype(str) + "_" +
            (df['lon'] * 100).astype(int).astype(str) + "_" +
            (df['time'] // time_step).astype(int).astype(str)
        )

        group_list = [group for _, group in df.groupby('bucket')]

        logger.info(f"Starting parallel processing of {len(group_list)} buckets at: {datetime.now()}")
        results = []

        with ProcessPoolExecutor(max_workers=min(16, multiprocessing.cpu_count())) as executor:
            futures = [executor.submit(process_group, group.copy(), time_step, distance_factor) for group in group_list]
            for future in as_completed(futures):
                result = future.result()
                if not result.empty:
                    results.append(result)

        if not results:
            logger.warning("No congestion pairs detected.")
            return pd.DataFrame(columns=['edge_id', 'vehicle1', 'vehicle1_route', 'vehicle2', 'vehicle2_route', 'congestion_score'])

        logger.info("Aggregating results at: %s", datetime.now())
        all_congestion = pd.concat(results, ignore_index=True)

        grouped = all_congestion.groupby(
            ['edge_id', 'vehicle_id_a', 'vehicle_id_b', 'route_id_a', 'route_id_b']
        )['congestion_score'].sum().reset_index().rename(columns={
            'vehicle_id_a': 'vehicle1', 'route_id_a': 'vehicle1_route',
            'vehicle_id_b': 'vehicle2', 'route_id_b': 'vehicle2_route'
        })

        grouped['run_configs_id'] = run_config_id
        grouped['iteration_id'] = iteration_id
        grouped['created_at'] = datetime.now()

        logger.info("Inserting congestion results into DB at: %s", datetime.now())
        grouped.to_sql('congestion_map', session.bind, if_exists='append', index=False, method='multi', chunksize=20000)
        session.commit()

        logger.info("Congestion calculation completed successfully in %s", datetime.now() - start)
        return grouped

    except Exception as e:
        session.rollback()
        logger.error(f"Error in generate_congestion: {e}", exc_info=True)
        return pd.DataFrame(columns=['edge_id', 'vehicle1', 'vehicle1_route', 'vehicle2', 'vehicle2_route', 'congestion_score'])
    
    finally:
        session.close()