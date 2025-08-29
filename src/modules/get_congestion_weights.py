from queue import Empty
import pandas as pd
import sqlalchemy as sa
import logging
from typing import Any
import itertools
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from datetime import datetime

logger = logging.getLogger(__name__)

def get_congestion_weights(
    session: Any,
    run_configs_id: int,
    iteration_id: int
    ):
    """
    Fetches pairwise vehicle congestion weights from SQL query and returns as a DataFrame.

    Args:
        session: SQLAlchemy session
        run_configs_id: ID of the run config
        iteration_id: Iteration number

    Returns:
        weights_df: DataFrame with columns:
            ['vehicle1', 'vehicle2', 'vehicle1_route', 'vehicle2_route', 'weighted_congestion_score']
    """
    try:
        # Set parameters
        session.execute(sa.text("SET @iteration := :iteration_id"), {'iteration_id': iteration_id})
        session.execute(sa.text("SET @run_configs_id := :run_configs_id"), {'run_configs_id': run_configs_id})
        # Query
        sql = sa.text("""


            SELECT
                vehicle1,
                vehicle2,
                vehicle1_route,
                vehicle2_route,
                sum(congestion_score) as weighted_congestion_score
            FROM trafficOptimization.congestion_map            
                WHERE iteration_id = @iteration
                AND run_configs_id = @run_configs_id
        group by vehicle1, vehicle2, vehicle1_route, vehicle2_route
        """)
        result = session.execute(sql)
        weights_df = pd.DataFrame(result.fetchall(), columns=pd.Index([
            'vehicle1', 'vehicle2', 'vehicle1_route', 'vehicle2_route', 'weighted_congestion_score'
        ]))
        logger.info(f"Fetched {len(weights_df)} congestion weight records for run_config_id={run_configs_id}, iteration_id={iteration_id}.")


        sql2 = sa.text("""

        WITH routes_with_min AS (
                SELECT vehicle_id,
                    MIN(duration)  AS min_duration
                FROM vehicle_routes
                WHERE run_configs_id = @run_configs_id AND iteration_id = @iteration
                group by vehicle_id
            )
        Select
            a.vehicle_id as vehicle,
            a.route_id as route,
            duration - min_duration as penalty
        FROM vehicle_routes as a
            join routes_with_min as b on a.vehicle_id=b.vehicle_id
        WHERE run_configs_id = @run_configs_id AND iteration_id = @iteration

        """
        )
        result = session.execute(sql2)
        duration_penalty_df = pd.DataFrame(result.fetchall(), columns=pd.Index([
            'vehicle', 'route', 'penalty'
        ]))


        return weights_df,duration_penalty_df
    
    except Exception as e:
        session.rollback()
        logger.error(f"Error fetching congestion weights: {e}", exc_info=True)
        return pd.DataFrame({col: pd.Series(dtype='float64') for col in ['vehicle1', 'vehicle2', 'vehicle1_route', 'vehicle2_route', 'weighted_congestion_score']})
    
    finally:
        session.close()
