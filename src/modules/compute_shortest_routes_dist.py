import pandas as pd
from sqlalchemy import text as sa_text
import logging
from typing import Any, Tuple, List, Type
from datetime import datetime
from models import ShortestRouteDur, ShortestRouteDis  # Assume you create this second model

logger = logging.getLogger(__name__)

def compute_shortest_routes_dist(
    session: Any,
    run_configs_id: int,
    iteration_id: int
) -> Tuple[pd.DataFrame, List[Any]]:
    """
    Compute congestion for shortest routes and return:
    - congestion_df: DataFrame with ['edge_id', 'congestion_score']
    - route_objs: List of ORM route objects stored in shortest_routes_distance table
    """
   

    try:
        # Step 1: Select (vehicle_id, route_id) pairs
        route_sql = sa_text(f"""
            WITH 
            filtered_vehicle_routes AS (
                SELECT vehicle_id, route_id, distance
                FROM vehicle_routes
                WHERE run_configs_id = :run_configs_id 
                AND iteration_id = :iteration_id
            ),
            shortest_routes AS (
                SELECT vehicle_id, MIN(distance) AS min_value
                FROM filtered_vehicle_routes
                GROUP BY vehicle_id
            ),
            shortest_selected_routes AS (
                SELECT fvr.vehicle_id, MAX(fvr.route_id) AS route_id
                FROM filtered_vehicle_routes fvr
                JOIN shortest_routes sr
                ON fvr.vehicle_id = sr.vehicle_id AND fvr.distance = sr.min_value
                GROUP BY fvr.vehicle_id
            )
            SELECT vehicle_id, route_id
            FROM shortest_selected_routes;
        """)
        route_result = session.execute(route_sql, {
            'run_configs_id': run_configs_id,
            'iteration_id': iteration_id
        })
        route_pairs = route_result.fetchall()

        logger.info(f"Selected shortest-distance routes for {len(route_pairs)} vehicles.")

        # Step 2: Delete old records and insert new ones
        session.query(ShortestRouteDis).filter(
            ShortestRouteDis.run_configs_id == run_configs_id,
            ShortestRouteDis.iteration_id == iteration_id
        ).delete()

        route_objs = [
            ShortestRouteDis(
                run_configs_id=run_configs_id,
                iteration_id=iteration_id,
                vehicle_id=row.vehicle_id,
                route_id=row.route_id,
                created_at=datetime.utcnow()
            )
            for row in route_pairs
        ]
        session.add_all(route_objs)
        session.commit()

        # Step 3: Calculate congestion
        congestion_sql = sa_text(f"""
            SELECT 
                cm.edge_id, 
                SUM(cm.congestion_score) AS congestion_score
            FROM congestion_map cm
            JOIN shortest_routes_distance sr1 
                ON sr1.vehicle_id = cm.vehicle1 AND sr1.route_id = cm.vehicle1_route
                AND sr1.run_configs_id = :run_configs_id AND sr1.iteration_id = :iteration_id                                 
            JOIN shortest_routes_distance sr2 
                ON sr2.vehicle_id = cm.vehicle2 AND sr2.route_id = cm.vehicle2_route
                AND sr2.run_configs_id = :run_configs_id AND sr2.iteration_id = :iteration_id
            WHERE cm.run_configs_id = :run_configs_id 
            AND cm.iteration_id = :iteration_id
            GROUP BY cm.edge_id;
        """)
        congestion_result = session.execute(congestion_sql, {
            'run_configs_id': run_configs_id,
            'iteration_id': iteration_id
        })
        congestion_df = pd.DataFrame(congestion_result.fetchall(), columns=["edge_id", "congestion_score"])
        session.close()

        return congestion_df, route_objs
    
    except Exception as e:
        logger.error(f"Error computing shortest-distance routes: {e}", exc_info=True)
        return pd.DataFrame(columns=["edge_id", "congestion_score"]), []
