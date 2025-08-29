from sqlalchemy import create_engine, MetaData, text
import logging

logger = logging.getLogger(__name__)

# Import engine from db_config
from db_config import engine


def drop_all_tables() -> None:
    """
    Drops all tables in the connected MariaDB database for the traffic optimization system.

    Usage:
    - This script disables foreign key checks, drops all tables, then re-enables checks.
    - Use with caution! This will delete all data and schema.
    - Always use context managers (with statements) for DB connections to ensure proper cleanup.

    Example:
        from db_config import get_session
        with get_session() as session:
            ... # ORM operations
    """
    try:
        with engine.connect() as conn:
            logger.info("Connected. Disabling foreign key checks...")
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))

            meta = MetaData()
            meta.reflect(bind=engine)

            logger.info(f"Dropping {len(meta.tables)} tables...")
            for table in meta.sorted_tables:
                logger.info(f"Dropping table {table.name}...")
                conn.execute(text(f"DROP TABLE IF EXISTS `{table.name}`"))

            logger.info("Re-enabling foreign key checks...")
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))

            logger.info("All tables dropped successfully.")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)

if __name__ == "__main__":
    drop_all_tables()
