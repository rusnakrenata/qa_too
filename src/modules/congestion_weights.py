import logging
import pandas as pd
import numpy as np
from typing import List, Any, Tuple, Set

logger = logging.getLogger(__name__)

def get_invalid_vehicle_route_pairs(
    vehicle_routes_df: pd.DataFrame, 
    route_alternatives: int
) -> Set[Tuple[Any, int]]:
    """
    Identify invalid vehicle-route pairs.

    Args:
        vehicle_routes_df: DataFrame with ['vehicle_id', 'route_id'] columns.
        route_alternatives: Number of route alternatives per vehicle.

    Returns:
        Set of invalid (vehicle_id, route_id) pairs.
    """
    vehicle_ids = vehicle_routes_df['vehicle_id'].unique()
    invalid_pairs = set()
    for vid in vehicle_ids:
        routes = set(vehicle_routes_df[vehicle_routes_df['vehicle_id'] == vid]['route_id'])
        for k in range(1, route_alternatives + 1):
            if k not in routes:
                invalid_pairs.add((vid, k))
    return invalid_pairs


def congestion_weights(
    weights_df: pd.DataFrame,
    n_vehicles: int,
    route_alternatives: int,
    vehicle_ids: List[Any],
    vehicle_routes_df: pd.DataFrame
) -> Tuple[List[List[List[List[float]]]], float]:
    """
    Constructs a 4D congestion weight matrix with penalization for non-existent routes.

    Args:
        weights_df: DataFrame containing columns ['vehicle1', 'vehicle2', 'vehicle1_route', 'vehicle2_route', 'weighted_congestion_score'].
        n_vehicles: Number of vehicles.
        route_alternatives: Number of route alternatives per vehicle.
        vehicle_ids: List of vehicle IDs.
        vehicle_routes_df: DataFrame with columns ['vehicle_id', 'route_id'].

    Returns:
        Tuple containing:
            - 4D list representing congestion weights.
            - Maximum congestion weight value (w_max).
    """
    # Identify invalid (vehicle, route) pairs
    invalid_pairs = get_invalid_vehicle_route_pairs(vehicle_routes_df, route_alternatives)

    # Filter weights DataFrame for relevant vehicle IDs
    vehicle_ids_set = set(int(v) for v in vehicle_ids)
    weights_df = weights_df[
        weights_df['vehicle1'].apply(lambda x: int(x) in vehicle_ids_set) &
        weights_df['vehicle2'].apply(lambda x: int(x) in vehicle_ids_set)
    ]

    # Determine max weight from the given DataFrame
    w_max = float(weights_df['weighted_congestion_score'].max()) if not weights_df.empty else 1.0


    # Create a lookup dictionary for congestion scores
    weights_lookup = {}
    for _, row in weights_df.iterrows():
        i = vehicle_ids.index(int(row['vehicle1']))
        j = vehicle_ids.index(int(row['vehicle2']))
        k1 = int(row['vehicle1_route']) - 1
        k2 = int(row['vehicle2_route']) - 1
        weights_lookup[(i, j, k1, k2)] = row['weighted_congestion_score']

    # Set of valid vehicle-route pairs
    valid_pairs = set(zip(vehicle_routes_df['vehicle_id'], vehicle_routes_df['route_id']))

    # Initialize the congestion weights matrix
    w = np.zeros((n_vehicles, n_vehicles, route_alternatives, route_alternatives), dtype=np.float64)

    # Populate the congestion weights matrix
    for i, vi in enumerate(vehicle_ids):
        for j, vj in enumerate(vehicle_ids):
            for k1 in range(route_alternatives):
                for k2 in range(route_alternatives):
                    key = (i, j, k1, k2)
                    pair1 = (vi, k1 + 1)
                    pair2 = (vj, k2 + 1)

                    if key in weights_lookup:
                        w[i, j, k1, k2] = weights_lookup[key]
                    elif (pair1 in valid_pairs) and (pair2 in valid_pairs):
                        w[i, j, k1, k2] = 0.0
                    elif (pair1 in invalid_pairs) or (pair2 in invalid_pairs):
                        w[i, j, k1, k2] = 0.0

    return w.tolist(), w_max