from collections import defaultdict
from typing import List, Dict, Tuple, Any
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import time
from congestion_weights import congestion_weights
from models import QuboRunStats


logger = logging.getLogger(__name__)

def qubo_matrix(
    session: Any,
    run_configs_id: int,
    iteration_id: int,
    cluster_id: int,
    cluster_resolution: int,
    n_vehicles: int,
    route_alternatives: int,
    weights_df: pd.DataFrame,
    duration_penalty_df: pd.DataFrame,
    vehicle_ids_filtered: List[int],
    vehicle_routes_df: pd.DataFrame,
    comp_type: str = "hybrid",
    qubo_output_dir: Path = Path("output/qubo_matrices"),
) -> Tuple[Dict[Tuple[int, int], float], int, float]:
    """
    Construct QUBO matrix to minimize congestion.

    Args:
        session: SQLAlchemy session
        run_configs_id: Run configuration ID
        iteration_id: Current iteration ID
        cluster_id: ID of the cluster
        route_alternatives: Number of route alternatives per vehicle
        weights_df: Pairwise congestion weights
        duration_penalty_df: Duration penalties per vehicle-route
        vehicle_ids_filtered: List of vehicle IDs
        vehicle_routes_df: Valid vehicle routes
        comp_type: Solver type (e.g., 'hybrid', 'hybrid_cqm')
        qubo_output_dir: Directory for QUBO output files

    Returns:
        Q: QUBO dictionary
        route_alternatives: Echoed input
        lambda_penalty: Penalty strength for constraints
    """
    start_time = time.time()

    logger.info("Starting QUBO vehicle filtering...")
    n_filtered = len(vehicle_ids_filtered)
    logger.info(f"{n_filtered} vehicles selected. Time elapsed: {time.time() - start_time:.2f}s")

    logger.info("Computing congestion weights...")
    congestion_w, max_w = congestion_weights(weights_df, n_filtered, route_alternatives, vehicle_ids_filtered, vehicle_routes_df)
    logger.info(f"Max congestion weight: {max_w:.4f}")

    duration_penalty_df = duration_penalty_df[duration_penalty_df['vehicle'].isin(vehicle_ids_filtered)]
    vehicle_id_to_idx = {int(v): i for i, v in enumerate(vehicle_ids_filtered)}
    route_ids = sorted(vehicle_routes_df['route_id'].unique())
    route_id_to_idx = {int(r): k for k, r in enumerate(route_ids)}

    penalty_matrix = np.zeros((n_filtered, len(route_ids)))
    for _, row in duration_penalty_df.iterrows():
        i = vehicle_id_to_idx[int(row['vehicle'])]
        k = route_id_to_idx[int(row['route'])]
        penalty_matrix[i, k] = row['penalty']

    penalties = [penalty_matrix[i, k] for i in range(n_filtered) for k in range(route_alternatives)]

    Q = defaultdict(float)
    for i in range(n_filtered):
        for j in range(i + 1, n_filtered):
            for k1 in range(route_alternatives):
                for k2 in range(route_alternatives):
                    q1 = i * route_alternatives + k1
                    q2 = j * route_alternatives + k2
                    Q[(q1, q2)] += congestion_w[i][j][k1][k2]

    valid_pairs = set(zip(vehicle_routes_df['vehicle_id'], vehicle_routes_df['route_id']))
    dynamic_penalties, q_indices, not_real_routes_indices = [], [], []

    for i, vehicle_id in enumerate(vehicle_ids_filtered):
        for route_num in range(1, route_alternatives + 1):
            q = i * route_alternatives + (route_num - 1)
            q_indices.append(q)
            if (vehicle_id, route_num) in valid_pairs:
                row_values = [Q.get((q, j), 0) for j in range(n_filtered * route_alternatives)]
                col_values = [Q.get((j, q), 0) for j in range(n_filtered * route_alternatives)]
                row_sum = sum(max(row_values[x], row_values[x+1]) for x in range(0, len(row_values)-1, 2))
                col_sum = sum(max(col_values[x], col_values[x+1]) for x in range(0, len(col_values)-1, 2))
                dynamic_penalties.append(row_sum + col_sum)
            else:
                not_real_routes_indices.append(q)

    lambda_penalty = max(dynamic_penalties) if dynamic_penalties else 1.0
    #lambda_penalty = lambda_penalty * 1.2 # Adjusted penalty for better performance``
    logger.info(f"Dynamic penalties calculated: {len(dynamic_penalties)} values, max penalty: {max(dynamic_penalties) if dynamic_penalties else 0.0}")   

    if comp_type != "hybrid_cqm":
        logger.info("Applying one-hot constraints...")
        for q in q_indices:
            if q in not_real_routes_indices:
                Q[(q, q)] += lambda_penalty
            else:
                Q[(q, q)] += -lambda_penalty + penalties[q]

        for i in range(n_filtered):
            for k1 in range(route_alternatives):
                q1 = i * route_alternatives + k1
                for k2 in range(route_alternatives):
                    if k1 != k2:
                        q2 = i * route_alternatives + k2
                        Q[(q1, q2)] += lambda_penalty
    else:
        logger.info("Hybrid CQM mode: skipping one-hot constraints.")
        for q in q_indices:
            if q not in not_real_routes_indices:
                Q[(q, q)] += penalties[q]

    logger.info(f"QUBO matrix built with {len(Q)} terms for {n_filtered} vehicles.")

    Q_matrix = np.zeros((max(idx for pair in Q for idx in pair) + 1,) * 2)
    for (q1, q2), value in Q.items():
        Q_matrix[q1, q2] = value

    Q_df = pd.DataFrame(Q_matrix)
    Q_df.columns = [f"{i:9g}" for i in range(Q_df.shape[1])]
    Q_df.index = [f"{i:9g}" for i in range(Q_df.shape[0])]

    Q_df.to_csv(qubo_output_dir / f"qubo_matrix_{cluster_id}.csv", index=True, header=True, float_format='%9g')

    stats = QuboRunStats(
        run_configs_id=run_configs_id,
        iteration_id=iteration_id,
        filtering_percentage=n_filtered / n_vehicles,
        cluster_resolution=cluster_resolution,
        cluster_id=cluster_id,
        n_vehicles=n_vehicles,
        n_filtered_vehicles=n_filtered,
        max_weight = max_w
    )
    session.add(stats)
    session.commit()

    return dict(Q), route_alternatives, lambda_penalty
