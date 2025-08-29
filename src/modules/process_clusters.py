from typing import Any, List, Tuple
import logging
from qa_testing import qa_testing
from gurobi_testing import gurobi_testing
from qubo_matrix import qubo_matrix
from pathlib import Path
from utils import check_bqm_against_solver_limits

logger = logging.getLogger(__name__)

def process_clusters(    
    session: Any,
    run_config: Any,
    iteration_id: int,
    n_vehicles: int, #all vehicles from config
    route_alternatives: int,
    vehicle_routes_df: Any,
    weights_df: Any,
    duration_penalty_df: Any,
    clusters: List[Tuple[List[int], Any, float, int]],
    cluster_resolution: int,
    comp_type: str = "hybrid",
    qubo_output_dir: Path = Path("output/qubo_matrices")
) -> Tuple[List[int], List[int], List[List[int]], List[Any]]:
    """
    Process each vehicle cluster: build Q_matrixUBO, run Q_matrixA and Gurobi, collect results.

    Args:
        clusters: List of clusters, each as (vehicle_ids, affected_edges_df, total_congestion, size)
        session: SQ_matrixLAlchemy session
        run_config: RunConfig object
        iteration_id: Current iteration ID
        vehicle_routes_df: DataFrame with vehicle-route mappings
        weights_df: Congestion weights
        duration_penalty_df: Penalties for route durations

    Returns:
        Tuple of:
            - qa_assignments: List of Q_matrixA route assignments
            - gurobi_assignments: List of Gurobi route assignments
            - all_vehicle_ids: List of filtered vehicle ID groups (per cluster)
            - all_affected_edges: List of affected edge DataFrames (per cluster)
    """
    qa_assignments = []
    gurobi_assignments = []
    all_vehicle_ids = []
    all_affected_edges = []

    for cluster_idx, (vehicle_ids, affected_edges_df, total_congestion, cluster_size) in enumerate(clusters):
        logger.info(f"Processing cluster {cluster_idx + 1}/{len(clusters)}: "
                    f"{cluster_size} vehicles, congestion: {total_congestion}")

        # Track affected edges
        if affected_edges_df is not None and not affected_edges_df.empty:
            affected_edges_df = affected_edges_df.copy()
            affected_edges_df["cluster_id"] = cluster_idx
            all_affected_edges.append(affected_edges_df)

        
        # Build Q_matrixUBO
        Q_matrix, Q_route_alt, lambda_penalty = qubo_matrix(
            session=session,
            run_configs_id=run_config.run_configs_id,
            iteration_id=iteration_id,
            cluster_id=cluster_idx,
            cluster_resolution=cluster_resolution,
            n_vehicles=n_vehicles,
            route_alternatives=route_alternatives,
            weights_df=weights_df,
            duration_penalty_df=duration_penalty_df,
            vehicle_ids_filtered=vehicle_ids,
            vehicle_routes_df=vehicle_routes_df,
            comp_type=comp_type,
            qubo_output_dir=qubo_output_dir
        )
        
        
        check_bqm_against_solver_limits(Q_matrix)

        # Q_matrixuantum Annealing
        qa_result = qa_testing(
            session=session,
            run_configs_id=run_config.run_configs_id,
            iteration_id=iteration_id,
            Q=Q_matrix,
            n_vehicles=n_vehicles,
            route_alternatives=Q_route_alt,
            vehicle_ids=vehicle_ids,
            lambda_value=lambda_penalty,
            comp_type=comp_type,
            num_reads=10,
            vehicle_routes_df=vehicle_routes_df,
            cluster_id=cluster_idx
        )
        qa_assignments.extend(qa_result["assignment"])

        # Gurobi
        gurobi_result, _ = gurobi_testing(
            session=session,
            run_configs_id=run_config.run_configs_id,
            iteration_id=iteration_id,
            Q=Q_matrix,
            n_vehicles=n_vehicles,
            route_alternatives=Q_route_alt,
            comp_type=comp_type,
            time_limit_seconds=qa_result["solver_time"],
            cluster_id=cluster_idx
        )
        sorted_gurobi = [v for k, v in sorted(gurobi_result.items(), key=lambda i: int(i[0].split("_")[1]))]
        gurobi_assignments.extend(sorted_gurobi)

        # Save processed vehicle IDs
        all_vehicle_ids.append(vehicle_ids)

    return qa_assignments, gurobi_assignments, all_vehicle_ids, all_affected_edges
