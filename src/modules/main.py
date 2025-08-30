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
from compute_weights import compute_qubo_overlap_weights
from qubo_matrix import qubo_matrix
from qa_testing import qa_testing
from gurobi_testing import gurobi_testing


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

            # Ensure output directory exists
            MAPS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            city_id, edges_df, nodes_df = get_or_create_city(
                session,
                city_name=CITY_NAME,
                center_coords=CENTER_COORDS,
                radius_km=RADIUS_KM
            )

            run_config = get_or_create_run_config(
                session,
                city_id=city_id,
                config_class=RunConfig,
                n_vehicles=N_VEHICLES,
                route_alternatives=K_ALTERNATIVES,
                min_length=MIN_LENGTH,
                max_length=MAX_LENGTH
            )

            iteration_id = create_iteration(
                session,
                run_config_id=run_config.run_configs_id,
                provided_iteration_id=None,
                iteration_class=Iteration
            )

            edges_gdf = gpd.GeoDataFrame(edges_df)
            nodes_gdf = gpd.GeoDataFrame(nodes_df)

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
                vehicles_df=vehicles_df,
                show_od_lines=False
            )
            m.save(MAPS_OUTPUT_DIR / f"city_vehicles_{run_config.run_configs_id}_{iteration_id}.html")

            vehicle_routes_df = generate_vehicle_routes(
                session=session,
                run_config_id=run_config.run_configs_id,
                iteration_id=iteration_id,
                route_class=VehicleRoute,
                vehicles_df=vehicles_df,
                edges_gdf=edges_gdf,
                nodes_gdf=nodes_gdf,
                time_step=TIME_STEP,
                time_window=TIME_WINDOW,
                max_concurrent=20
            )

            mp = plot_vehicle_routes(
                edges_gdf=edges_gdf,
                nodes_gdf=nodes_gdf,
                vehicle_routes_df=vehicle_routes_df,
                vehicles_df=vehicles_df,
                show_route_nodes=True,
                zoom_start=14
            )
            mp.save(MAPS_OUTPUT_DIR / f"vehicle_routes_{run_config.run_configs_id}_{iteration_id}.html")

            diag, offdiag, max_weight_cvw, max_weight_cvv, n_nodes_distinct, overall_overlap = compute_qubo_overlap_weights(
                vehicle_routes_df, node_column="path_node_ids"
            )

            lambda_value = 1
            lambda_values = [0.5]

            for lambda_penalty in lambda_values:
                Q, lambda_penalty = qubo_matrix(
                    session=session,
                    run_configs_id=run_config.run_configs_id,
                    iteration_id=iteration_id,
                    diag=diag,
                    offdiag=offdiag,
                    max_weight_cvv=max_weight_cvv,
                    max_weight_cvw=max_weight_cvw,
                    n_nodes_distinct=n_nodes_distinct,
                    overall_overlap=overall_overlap,
                    lambda_penalty=None # calculated inside function if None
                )
                print("lambda penalty:", lambda_penalty)

                qa_results = qa_testing(
                    session=session,
                    run_configs_id=run_config.run_configs_id,
                    iteration_id=iteration_id,
                    Q=Q,
                    vehicle_routes_df=vehicle_routes_df,
                    lambda_value=lambda_penalty,
                    comp_type=COMP_TYPE,
                    num_reads=10
                )

                time_limit = qa_results["solver_time"]

                gurobi_vehicles, objective_value = gurobi_testing(
                    session=session,
                    run_configs_id=run_config.run_configs_id,
                    iteration_id=iteration_id,
                    Q=Q,
                    vehicle_routes_df=vehicle_routes_df,
                    lambda_value=lambda_penalty,
                    time_limit_seconds=time_limit
                )

                suffix = f"{run_config.run_configs_id}_{iteration_id}_lambda{lambda_penalty:.2f}"

                mp_qa = plot_vehicle_routes(
                    edges_gdf=edges_gdf,
                    nodes_gdf=nodes_gdf,
                    vehicle_routes_df=vehicle_routes_df,
                    vehicles_df=vehicles_df,
                    show_route_nodes=True,
                    show_route_edges=True,
                    zoom_start=5,
                    selected_vehicle_ids=qa_results["selected_vehicle_ids"]
                )
                mp_qa.save(MAPS_OUTPUT_DIR / f"vehicle_routes_qa_{suffix}.html")

                mp_gurobi = plot_vehicle_routes(
                    edges_gdf=edges_gdf,
                    nodes_gdf=nodes_gdf,
                    vehicle_routes_df=vehicle_routes_df,
                    vehicles_df=vehicles_df,
                    show_route_nodes=True,
                    show_route_edges=True,
                    zoom_start=5,
                    selected_vehicle_ids=gurobi_vehicles
                )
                mp_gurobi.save(MAPS_OUTPUT_DIR / f"vehicle_routes_gurobi_{suffix}.html")

            logger.info("Workflow finished!")

    except Exception as e:
        logger.error("Workflow error: %s", str(e), exc_info=True)
    finally:
        logger.info("Total duration: %s", datetime.now() - start)


if __name__ == "__main__": 
    main()