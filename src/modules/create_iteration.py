import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

def create_iteration(
    session: Any,
    run_config_id: int,
    provided_iteration_id: Optional[int],
    iteration_class: Any
) -> Optional[int]:
    """
    Create a new simulation iteration for a given run configuration.

    Args:
        session: SQLAlchemy session
        run_config_id: ID of the run configuration
        provided_iteration_id: If provided, checks for existence
        iteration_class: SQLAlchemy Iteration model

    Returns:
        new_iteration_id: The created iteration ID, or None if not created
    """
    try:
        if provided_iteration_id is not None:
            existing = session.query(iteration_class).filter_by(id=provided_iteration_id, run_configs_id=run_config_id).first()
            if existing:
                logger.warning(f"Run configuration for iteration {provided_iteration_id} already exists. Stopping further execution.")
                return None
            else:
                logger.warning("Do not provide iteration_id. It will be generated automatically.")
                return None
        else:
            count = session.query(iteration_class).filter_by(run_configs_id=run_config_id).count()
            new_iteration_id = count + 1
            iteration = iteration_class(
                iteration_id=new_iteration_id,
                run_configs_id=run_config_id
            )
            session.add(iteration)
            session.commit()
            logger.info(f"Iteration created (iteration_id={new_iteration_id}) for run_config_id={run_config_id}.")
            return new_iteration_id
        
    except Exception as e:
        logger.error(f"Error creating iteration: {e}", exc_info=True)
        session.rollback()
        return None
    
    finally:
        session.close()
