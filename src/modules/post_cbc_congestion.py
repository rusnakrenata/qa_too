import pandas as pd
from sqlalchemy import text as sa_text
import logging
from typing import Any, List
from models import CbcRoute

logger = logging.getLogger(__name__)

def post_cbc_congestion(
    session: Any,
    run_configs_id: int,
    iteration_id: int,
    all_vehicle_ids: List[Any],
    optimized_vehicle_ids: List[Any],
    cbc_assignment: List[int],
    route_alternatives: int,
    method: str = "duration"
):
    """
    Recomputes congestion based on CBC assignment and shortest routes for non-optimized vehicles.

    Args:
        session: SQLAlchemy session
        run_configs_id: Run config ID
        iteration_id: Iteration ID
        all_vehicle_ids: List of all vehicle IDs
        optimized_vehicle_ids: Vehicles used in CBC optimization
        cbc_assignment: Flat list of binary route choices
        route_alternatives: Number of route alternatives per vehicle
        method: 'distance' or 'duration' for fallback assignments

    Returns:
        Tuple: (DataFrame of congestion scores, list of inserted CbcRoute rows)
    """
    try:
        vehicle_route_pairs = []

        for idx, vehicle_id in enumerate(optimized_vehicle_ids):
            assignment = cbc_assignment[idx * route_alternatives : (idx + 1) * route_alternatives]
            if assignment.count(1) == 1:
                route_id = assignment.index(1) + 1
            else:
                sql = sa_text(f'''
                    SELECT route_id FROM vehicle_routes
                    WHERE vehicle_id = :vehicle_id AND run_configs_id = :run_configs_id AND iteration_id = :iteration_id
                    ORDER BY {method} ASC LIMIT 1
                ''')
                result = session.execute(sql, {
                    'vehicle_id': vehicle_id,
                    'run_configs_id': run_configs_id,
                    'iteration_id': iteration_id
                }).fetchone()
                route_id = result.route_id if result else 1
            vehicle_route_pairs.append((vehicle_id, route_id))

        non_optimized = set(all_vehicle_ids) - set(optimized_vehicle_ids)
        if non_optimized:
            sql = sa_text(f"""
                WITH shortest_routes AS (
                    SELECT vehicle_id, MIN({method}) AS min_value
                    FROM vehicle_routes
                    WHERE run_configs_id = :run_configs_id AND iteration_id = :iteration_id
                    GROUP BY vehicle_id
                ),
                fallback_routes AS (
                    SELECT vr.vehicle_id, MAX(vr.route_id) as route_id
                    FROM vehicle_routes vr
                    JOIN shortest_routes sr
                    ON vr.vehicle_id = sr.vehicle_id AND vr.{method} = sr.min_value
                    WHERE vr.run_configs_id = :run_configs_id AND vr.iteration_id = :iteration_id
                    GROUP BY vr.vehicle_id
                )
                SELECT vehicle_id, route_id FROM fallback_routes
            """)
            result = session.execute(sql, {
                'run_configs_id': run_configs_id,
                'iteration_id': iteration_id
            })
            for row in result.fetchall():
                if row.vehicle_id in non_optimized:
                    vehicle_route_pairs.append((row.vehicle_id, row.route_id))

        session.query(CbcRoute).filter(
            CbcRoute.run_configs_id == run_configs_id,
            CbcRoute.iteration_id == iteration_id
        ).delete()

        cbc_routes = [
            CbcRoute(
                run_configs_id=run_configs_id,
                iteration_id=iteration_id,
                vehicle_id=vehicle_id,
                route_id=route_id
            )
            for vehicle_id, route_id in vehicle_route_pairs
        ]
        session.add_all(cbc_routes)
        session.commit()

        result = session.execute(sa_text(f"""
            WITH 
            filtered_routes AS (
                SELECT vehicle_id, route_id
                FROM trafficOptimization.cbc_routes
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

        logger.info(f"Recomputed CBC congestion for run_configs_id={run_configs_id}, iteration_id={iteration_id}.")
        return pd.DataFrame(rows, columns=["edge_id", "congestion_score"]), cbc_routes

    except Exception as e:
        session.rollback()
        logger.error(f"Error in post_cbc_congestion: {e}", exc_info=True)
        return pd.DataFrame(), None
    
    finally:
        session.close()
