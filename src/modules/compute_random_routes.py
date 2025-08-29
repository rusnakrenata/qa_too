import pandas as pd
import numpy as np
import random
from sqlalchemy import text as sa_text
import logging
from typing import Any
from models import RandomRoute
from datetime import datetime

logger = logging.getLogger(__name__)

def compute_random_routes(
    session: Any,
    run_configs_id: int,
    iteration_id: int
):
    """
    For each vehicle, select a random route and compute congestion.
    Returns a DataFrame with columns ['edge_id', 'congestion_score'].
    """
    # 1. Select a random route for each vehicle using a window function
    sql = sa_text(f'''
        WITH random_routes AS (
            SELECT vehicle_id, route_id,
                   ROW_NUMBER() OVER (PARTITION BY vehicle_id ORDER BY RAND()) as rn
            FROM vehicle_routes
            WHERE run_configs_id = :run_configs_id AND iteration_id = :iteration_id
        )
        SELECT vehicle_id, route_id FROM random_routes WHERE rn = 1
    ''')
    result = session.execute(sql, {'run_configs_id': run_configs_id, 'iteration_id': iteration_id})
    vehicle_route_pairs = [(row.vehicle_id, row.route_id) for row in result.fetchall()]
    logger.info(f"Selected random routes for {len(vehicle_route_pairs)} vehicles.")

    # Clear previous data for this run_config and iteration
    session.query(RandomRoute).filter(
        RandomRoute.run_configs_id == run_configs_id,
        RandomRoute.iteration_id == iteration_id
    ).delete()

# Insert new random routes
    random_route_objs = [
        RandomRoute(
            run_configs_id=run_configs_id,
            iteration_id=iteration_id,
            vehicle_id=vehicle_id,
            route_id=route_id,
            created_at=datetime.utcnow()
        )
        for vehicle_id, route_id in vehicle_route_pairs
    ]
    session.add_all(random_route_objs)
    session.commit()
    

    # 3. Calculate congestion
    result = session.execute(sa_text(f"""
        SELECT 
            cm.edge_id, 
            SUM(cm.congestion_score) AS congestion_score
        FROM congestion_map cm
        JOIN random_routes rr1 
            ON rr1.vehicle_id = cm.vehicle1 AND rr1.route_id = cm.vehicle1_route
            AND rr1.run_configs_id = :run_configs_id AND rr1.iteration_id = :iteration_id
        JOIN random_routes rr2 
            ON rr2.vehicle_id = cm.vehicle2 AND rr2.route_id = cm.vehicle2_route
            AND rr2.run_configs_id = :run_configs_id AND rr2.iteration_id = :iteration_id
        WHERE cm.run_configs_id = :run_configs_id 
        AND cm.iteration_id = :iteration_id
        GROUP BY cm.edge_id
    """), {"run_configs_id": run_configs_id, "iteration_id": iteration_id})

    rows = list(result.fetchall())
    session.close()
    logger.info(f"Computed congestion for {len(rows)} edges using random routes.")

    return pd.DataFrame(rows, columns=["edge_id", "congestion_score"]), random_route_objs # type: ignore

