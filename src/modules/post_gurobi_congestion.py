import pandas as pd
from sqlalchemy import text as sa_text
import logging
from typing import Any, List, Union
from models import GurobiRoute

logger = logging.getLogger(__name__)

def post_gurobi_congestion(
    session: Any,
    run_configs_id: int,
    iteration_id: int,
    all_vehicle_ids: List[Any],
    optimized_vehicle_ids: List[Any],
    gurobi_assignment: List[Any],
    route_alternatives: int,
    method: str = "duration"
) :
    """
    Recomputes congestion based on the Gurobi-selected vehicle-route assignments and shortest routes for non-optimized vehicles.
    Args:
        session: SQLAlchemy session
        run_config_id: ID of the run configuration
        iteration_id: Iteration number
        all_vehicle_ids: List of all vehicle IDs in the simulation
        optimized_vehicle_ids: List of vehicle IDs used in QUBO/Gurobi
        gurobi_assignment: List of (var_name, value) tuples or dict {var_name: value}
        route_alternatives: Number of route alternatives per vehicle
        method: 'distance' or 'duration' for non-optimized vehicles
    Returns:
        DataFrame with columns ['edge_id', 'congestion_score']
    """
    try:
        vehicle_route_pairs = []

        # For optimized vehicles, use Gurobi assignment
        for idx, vehicle_id in enumerate(optimized_vehicle_ids):
            # Each vehicle has t variables: x_{q}, where q = idx * t + k
            assigned_route = None
            for k in range(route_alternatives):
                if gurobi_assignment[idx * route_alternatives + k] == 1:
                    assigned_route = k + 1
                    break

            if assigned_route is None:
                # fallback: assign shortest route
                sql = sa_text(f'''
                    SELECT route_id FROM vehicle_routes
                    WHERE vehicle_id = :vehicle_id AND run_configs_id = :run_configs_id AND iteration_id = :iteration_id
                    ORDER BY {method} ASC LIMIT 1
                ''')
                result = session.execute(sql, {
                    'vehicle_id': vehicle_id,
                    'run_configs_id': run_configs_id,
                    'iteration_id': iteration_id
                })
                row = result.fetchone()
                if row is not None:
                    assigned_route = row.route_id
                    #logger.warning(f"Invalid Gurobi assignment for vehicle {vehicle_id}: not one-hot. Assigned shortest route {assigned_route}.")
                else:
                    assigned_route = 1  # fallback if no route found
                    #logger.warning(f"Invalid Gurobi assignment for vehicle {vehicle_id}: not one-hot. Assigned first route {assigned_route}.")
            vehicle_route_pairs.append((vehicle_id, assigned_route))
        logger.info(f"Number of optimized vehicles: {len(optimized_vehicle_ids)}")

        # For non-optimized vehicles, assign by shortest (distance/duration)
        non_optimized = set(all_vehicle_ids) - set(optimized_vehicle_ids)
        logger.info(f"Number of non-optimized vehicles: {len(non_optimized)}")
        if non_optimized:
            sql = sa_text(f"""
                WITH shortest_routes AS (
                    SELECT vehicle_id, MIN({method}) AS min_value
                    FROM vehicle_routes
                    WHERE run_configs_id = :run_configs_id AND iteration_id = :iteration_id
                    GROUP BY vehicle_id
                ),
                gurobi_routes AS (
                    SELECT vr.vehicle_id, max(vr.route_id) as route_id 
                    FROM vehicle_routes vr
                    JOIN shortest_routes sr
                    ON vr.vehicle_id = sr.vehicle_id AND vr.{method} = sr.min_value
                    WHERE vr.run_configs_id = :run_configs_id AND vr.iteration_id = :iteration_id
                    GROUP BY vr.vehicle_id
                )
                SELECT vehicle_id, route_id FROM gurobi_routes
            """)
            result = session.execute(sql, {
                'run_configs_id': run_configs_id,
                'iteration_id': iteration_id
            })
            for row in result.fetchall():
                if row.vehicle_id in non_optimized:
                    vehicle_route_pairs.append((row.vehicle_id, row.route_id))

        # Clear previous data for this run_config and iteration
        session.query(GurobiRoute).filter(
            GurobiRoute.run_configs_id == run_configs_id,
            GurobiRoute.iteration_id == iteration_id
        ).delete()
        
        # Insert new selected routes using the model
        gurobi_routes = [
            GurobiRoute(
                run_configs_id=run_configs_id,
                iteration_id=iteration_id,
                vehicle_id=vehicle_id,
                route_id=route_id
            )
            for vehicle_id, route_id in vehicle_route_pairs
        ]
        session.add_all(gurobi_routes)
        session.commit()

        # After building vehicle_route_pairs, check for completeness and uniqueness
        vehicle_id_counts = {}
        for vehicle_id, route_id in vehicle_route_pairs:
            vehicle_id_counts[vehicle_id] = vehicle_id_counts.get(vehicle_id, 0) + 1
        missing_vehicles = set(all_vehicle_ids) - set(vehicle_id_counts.keys())
        duplicated_vehicles = [vid for vid, count in vehicle_id_counts.items() if count > 1]
        if missing_vehicles:
            logger.warning(f"Missing vehicles in vehicle_route_pairs: {missing_vehicles}")
        if duplicated_vehicles:
            logger.warning(f"Duplicated vehicles in vehicle_route_pairs: {duplicated_vehicles}")
        if not missing_vehicles and not duplicated_vehicles:
            logger.info(f"All {len(all_vehicle_ids)} vehicles have exactly one route assigned.")

        # Now run your main query using session
        result = session.execute(sa_text(f"""
            WITH 
            filtered_routes AS (
                SELECT vehicle_id, route_id
                FROM trafficOptimization.gurobi_routes
                WHERE run_configs_id = :run_configs_id 
                AND iteration_id = :iteration_id
            ),
            filtered_congestion AS (
                SELECT edge_id, vehicle1, vehicle1_route, vehicle2, vehicle2_route, congestion_score
                FROM trafficOptimization.congestion_map
                WHERE run_configs_id = :run_configs_id 
                AND iteration_id = :iteration_id
            )
            SELECT 
                cm.edge_id, 
                SUM(cm.congestion_score) AS congestion_score
            FROM filtered_congestion cm
            JOIN filtered_routes sr1 
                ON sr1.vehicle_id = cm.vehicle1 AND sr1.route_id = cm.vehicle1_route
            JOIN filtered_routes sr2 
                ON sr2.vehicle_id = cm.vehicle2 AND sr2.route_id = cm.vehicle2_route
            GROUP BY cm.edge_id;
        """), {'run_configs_id': run_configs_id, 'iteration_id': iteration_id})
        rows = list(result.fetchall())
        session.commit()

        logger.info(f"Recomputed Gurobi congestion for run_configs_id={run_configs_id}, iteration_id={iteration_id}.")
        return pd.DataFrame(rows, columns=pd.Index(['edge_id', 'congestion_score'])), gurobi_routes
    
    except Exception as e:
        session.rollback()
        logger.error(f"Error in post_gurobi_congestion: {e}", exc_info=True)
        return pd.DataFrame(), None
    
    finally:
        session.close()
