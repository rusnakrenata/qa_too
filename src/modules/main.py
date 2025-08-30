import logging
from pathlib import Path
from datetime import datetime
from typing import Any
import pandas as pd
import geopandas as gpd
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text as sa_text

# ---------- CONFIG & MODELS ----------
from config import *
from models import *

# ---------- MODULES ----------
from get_or_create_city import get_or_create_city
from get_or_create_run_config import get_or_create_run_config
from create_iteration import create_iteration
from generate_vehicles_random import generate_vehicles
from generate_vehicle_routes import generate_vehicle_routes

from plot_vehicles import plot_vehicles
from plot_vehicle_routes import plot_vehicle_routes


# ---------- CONSTANTS ----------
OFFSET_DEG = 0.0000025
MAPS_OUTPUT_DIR = Path("files_html")
MATRIX_OUTPUT_DIR = Path("files_csv")

# ---------- LOGGER ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def create_db_session() -> Any:
    Session = sessionmaker(bind=engine, autocommit=False)
    return Session()


def main():
    start = datetime.now()
    try:
        with create_db_session() as session:
            logger.info("Start workflow: %s", start)

            city_id, edges_df, nodes_df = get_or_create_city(session, city_name=CITY_NAME,
                                                center_coords=CENTER_COORDS,
                                                radius_km=RADIUS_KM)

            run_config = get_or_create_run_config(session, city_id=city_id,
                                        config_class=RunConfig,
                                        n_vehicles=N_VEHICLES,
                                        route_alternatives=K_ALTERNATIVES,
                                        min_length=MIN_LENGTH,
                                        max_length=MAX_LENGTH)
            iteration_id = create_iteration(session, run_config_id=run_config.run_configs_id,
                                provided_iteration_id=None, iteration_class=Iteration)
            
            edges_gdf = gpd.GeoDataFrame(edges_df)
            nodes_gdf = gpd.GeoDataFrame(nodes_df)

            # when you enable generation:
            vehicles_df = generate_vehicles(
                session=session,
                run_config_id=run_config.run_configs_id,
                iteration_id=iteration_id,
                Vehicle=Vehicle,
                nodes_gdf=nodes_gdf,
                n_vehicles=N_VEHICLES,
                min_length=MIN_LENGTH,
                max_length=MAX_LENGTH
)
            m = plot_vehicles(
            edges_gdf=edges_gdf,
            vehicles_df=vehicles_df,   # DataFrame with lat/lon columns
            show_od_lines=False
            )
            m.save("city_vehicles.html")

            vehicle_routes_df = generate_vehicle_routes(
                session=session,
                run_config_id=run_config.run_configs_id,
                iteration_id=iteration_id,
                route_class=VehicleRoute,         # from models.py
                vehicles_df=vehicles_df,          # has lat/lon columns
                edges_gdf=edges_gdf,              # edge_id, geometry (EPSG:4326)
                nodes_gdf=nodes_gdf,              # node_id, geometry (EPSG:4326)
                time_step=TIME_STEP,
                time_window=TIME_WINDOW,
                max_concurrent=20

            )
           
            mp = plot_vehicle_routes(
                edges_gdf=edges_gdf,
                nodes_gdf=nodes_gdf,
                vehicle_routes_df=vehicle_routes_df,
                vehicles_df=vehicles_df,       # pass None to skip O/D dots
                show_route_nodes=True,         # turn on to see node dots
                zoom_start=14
            )
            mp.save("vehicle_routes.html")
            logger.info("Workflow finished!")

    except Exception as e:
        logger.error("Workflow error: %s", str(e), exc_info=True)
    finally:
        logger.info("Total duration: %s", datetime.now() - start)


if __name__ == "__main__":
    main()
