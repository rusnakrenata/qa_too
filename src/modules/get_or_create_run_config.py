import logging
from typing import Any

logger = logging.getLogger(__name__)

def get_or_create_run_config(
    session: Any,
    city_id: int,
    config_class: Any,
    n_vehicles: int,
    route_alternatives: int,
    min_length: int,
    max_length: int
) -> Any:
    """
    Get or create a run configuration in the database, including distance_factor.

    Args:
        session: SQLAlchemy session.
        city_id: City ID.
        config_class: SQLAlchemy RunConfig model.
        n_vehicles: Number of vehicles.
        route_alternatives: Number of route alternatives.
        min_length: Minimum route length.
        max_length: Maximum route length.
        time_step: Time step for congestion calculation.
        time_window: Time window for congestion calculation.
        distance_factor: Factor influencing congestion distance sensitivity.

    Returns:
        The existing or newly created RunConfig object.
    """
    try:
        existing_run = session.query(config_class).filter_by(
            city_id=city_id,
            n_vehicles=n_vehicles,
            k_alternatives=route_alternatives,
            min_length=min_length,
            max_length=max_length
        ).first()

        if existing_run:
            logger.info(f"Run config already exists (run_id={existing_run.run_configs_id}), skipping insertion.")
            return existing_run
        else:
            run_config = config_class(
                city_id=city_id,
                n_vehicles=n_vehicles,
                k_alternatives=route_alternatives,
                min_length=min_length,
                max_length=max_length
            )
            session.add(run_config)
            session.commit()
            logger.info(f"Run configuration saved (run_id={run_config.run_configs_id}).")
            return run_config

    except Exception as e:
        logger.error(f"Error in get_or_create_run_config: {e}", exc_info=True)
        session.rollback()
        return None
    
    finally:
        session.close()
