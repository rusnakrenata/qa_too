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
from generate_vehicles_attraction import generate_vehicles_attraction
from generate_vehicle_routes import generate_vehicle_routes
from generate_congestion import generate_congestion
from get_congestion_weights import get_congestion_weights
from filter_routes_for_qubo import get_clusters_by_connectivity
from process_clusters import process_clusters
from process_clusters_full import process_clusters_full
from post_qa_congestion import post_qa_congestion
from post_gurobi_congestion import post_gurobi_congestion
from post_sa_congestion import post_sa_congestion
from post_tabu_congestion import post_tabu_congestion   
from post_cbc_congestion import post_cbc_congestion
from compute_random_routes import compute_random_routes
from compute_shortest_routes_dist import compute_shortest_routes_dist
from compute_shortest_routes_dur import compute_shortest_routes_dur
from visualize_and_save_congestion_heatmap import visualize_and_save_congestion_heatmap
from save_congestion_summary import save_congestion_summary
from run_sql_query import run_sql_query


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


def generate_and_store_vehicles(session, run_configs, iteration_id, edges_gdf: gpd.GeoDataFrame,
                                attraction_point=None, d_alternatives=None):
    if attraction_point and d_alternatives:
        logger.info("Using attraction-based vehicle generation.")
        return generate_vehicles_attraction(
            session=session,
            run_config_id=run_configs.run_configs_id,
            iteration_id=iteration_id,
            Vehicle=Vehicle,
            edges_gdf=edges_gdf,
            n_vehicles=N_VEHICLES,
            min_length=MIN_LENGTH,
            max_length=MAX_LENGTH,
            attraction_point=attraction_point,
            d_alternatives=d_alternatives
        )
    logger.info("Using random vehicle generation.")
    return generate_vehicles(
        session=session,
        run_config_id=run_configs.run_configs_id,
        iteration_id=iteration_id,
        Vehicle=Vehicle,
        edges_gdf=edges_gdf,
        n_vehicles=N_VEHICLES,
        min_length=MIN_LENGTH,
        max_length=MAX_LENGTH
    )


def get_k_alternatives(session, run_configs_id, iteration_id):
    sql = sa_text("""
        SELECT MAX(route_id) as max_route_id
        FROM vehicle_routes
        WHERE run_configs_id = :run_configs_id AND iteration_id = :iteration_id
    """)
    result = session.execute(sql, {'run_configs_id': run_configs_id, 'iteration_id': iteration_id})
    return int(result.scalar() or 1)


def main():
    start = datetime.now()
    try:
        with create_db_session() as session:
            logger.info("Start workflow: %s", start)

            city_id, edges_df = get_or_create_city(session, city_name=CITY_NAME,
                                                center_coords=CENTER_COORDS,
                                                radius_km=RADIUS_KM,
                                                attraction_point=ATTRACTION_POINT,
                                                d_alternatives=D_ALTERNATIVES)
            if edges_df is None:
                logger.error("No edges_df found for the city. Exiting workflow.")
                return

            edges_gdf = gpd.GeoDataFrame(edges_df)
            run_config = get_or_create_run_config(session, city_id=city_id,
                                                  config_class=RunConfig,
                                                  n_vehicles=N_VEHICLES,
                                                  route_alternatives=K_ALTERNATIVES,
                                                  min_length=MIN_LENGTH,
                                                  max_length=MAX_LENGTH,
                                                  time_step=TIME_STEP,
                                                  time_window=TIME_WINDOW,
                                                  distance_factor=DISTANCE_FACTOR)

            iteration_id = create_iteration(session, run_config_id=run_config.run_configs_id,
                                            provided_iteration_id=None, iteration_class=Iteration)

            vehicles_gdf = generate_and_store_vehicles(session, run_configs=run_config,
                                                       iteration_id=iteration_id, edges_gdf=edges_gdf,
                                                       attraction_point=ATTRACTION_POINT,
                                                       d_alternatives=D_ALTERNATIVES)

            routes_df = generate_vehicle_routes(session, run_config_id=run_config.run_configs_id,
                                                iteration_id=iteration_id, route_class=VehicleRoute,
                                                route_point_class=RoutePoint, vehicles_gdf=vehicles_gdf,
                                                edges_gdf=edges_gdf, k_alternatives=K_ALTERNATIVES,
                                                time_step=TIME_STEP, time_window=TIME_WINDOW)

            route_alternatives = get_k_alternatives(session, run_configs_id=run_config.run_configs_id, iteration_id=iteration_id)
            congestion_df = generate_congestion(session, run_config_id=run_config.run_configs_id,
                                                iteration_id=iteration_id, time_step=TIME_STEP,
                                                distance_factor=DISTANCE_FACTOR)

            weights_df, duration_penalty_df = get_congestion_weights(session, run_configs_id=run_config.run_configs_id,
                                                                      iteration_id=iteration_id)

            all_vehicle_ids = routes_df["vehicle_id"].unique().tolist()
            print(f"Total vehicles: {len(all_vehicle_ids)}")
            clusters = get_clusters_by_connectivity(congestion_df, resolution=CLUSTER_RESOLUTION, min_cluster_size=MIN_CLUSTER_SIZE)
            if not clusters:
                clusters = [(all_vehicle_ids, congestion_df.groupby('edge_id', as_index=False).agg({'congestion_score': 'sum'}), 0.0, len(all_vehicle_ids))]

            clusters = clusters[:MAX_CLUSTERS] if MAX_CLUSTERS else clusters


            if FULL:
                (
                    qa_assignment,
                    gurobi_assignment,
                    sa_assignment,
                    tabu_assignment,
                    cbc_assignments,
                    all_filtered_ids,
                    affected_edges_df,
                ) = process_clusters_full(
                    session=session,
                    run_config=run_config,
                    iteration_id=iteration_id,
                    n_vehicles=N_VEHICLES,
                    route_alternatives=route_alternatives,
                    vehicle_routes_df=routes_df,
                    weights_df=weights_df,
                    duration_penalty_df=duration_penalty_df,
                    clusters=clusters,
                    cluster_resolution=CLUSTER_RESOLUTION,
                    comp_type=COMP_TYPE,
                    qubo_output_dir=Path(MATRIX_OUTPUT_DIR),
                )
            else:
                (
                    qa_assignment,
                    gurobi_assignment,
                    all_filtered_ids,
                    affected_edges_df,
                ) = process_clusters(
                    session=session,
                    run_config=run_config,
                    iteration_id=iteration_id,
                    n_vehicles=N_VEHICLES,
                    route_alternatives=route_alternatives,
                    vehicle_routes_df=routes_df,
                    weights_df=weights_df,
                    duration_penalty_df=duration_penalty_df,
                    clusters=clusters,
                    cluster_resolution=CLUSTER_RESOLUTION,
                    comp_type=COMP_TYPE,
                    qubo_output_dir=Path(MATRIX_OUTPUT_DIR),
                )
                # placeholders so later code can branch cleanly
                sa_assignment = None
                tabu_assignment = None
                cbc_assignments = None

            # affected_edges_df can be list[DataFrame] or a single DataFrame; normalize
            if isinstance(affected_edges_df, list) and affected_edges_df:
                affected_df = pd.concat(affected_edges_df, ignore_index=True)
            elif isinstance(affected_edges_df, pd.DataFrame):
                affected_df = affected_edges_df.copy()
            else:
                affected_df = pd.DataFrame(columns=["edge_id", "congestion_score", "cluster_id"])

            # Flatten optimized vehicle ids robustly
            if isinstance(all_filtered_ids, list) and all_filtered_ids and isinstance(all_filtered_ids[0], (list, tuple, set)):
                optimized_vehicle_ids = [vid for group in all_filtered_ids for vid in group]
            else:
                optimized_vehicle_ids = list(all_filtered_ids or [])

            # Compute post-* congestion DataFrames
            post_qa_df, _ = post_qa_congestion(
                session,
                run_configs_id=run_config.run_configs_id,
                iteration_id=iteration_id,
                all_vehicle_ids=all_vehicle_ids,
                optimized_vehicle_ids=optimized_vehicle_ids,
                qa_assignment=qa_assignment,
                method=ROUTE_METHOD,
            )

            post_gurobi_df, _ = post_gurobi_congestion(
                session,
                run_configs_id=run_config.run_configs_id,
                iteration_id=iteration_id,
                all_vehicle_ids=all_vehicle_ids,
                optimized_vehicle_ids=optimized_vehicle_ids,
                gurobi_assignment=gurobi_assignment,
                route_alternatives=route_alternatives,
                method=ROUTE_METHOD,
            )

            sa_df = tabu_df = cbc_df = None
            if FULL:
                if sa_assignment is not None:
                    sa_df, _ = post_sa_congestion(
                        session,
                        run_configs_id=run_config.run_configs_id,
                        iteration_id=iteration_id,
                        all_vehicle_ids=all_vehicle_ids,
                        optimized_vehicle_ids=optimized_vehicle_ids,
                        sa_assignement=sa_assignment,
                        method=ROUTE_METHOD,
                    )
                if tabu_assignment is not None:
                    tabu_df, _ = post_tabu_congestion(
                        session,
                        run_configs_id=run_config.run_configs_id,
                        iteration_id=iteration_id,
                        all_vehicle_ids=all_vehicle_ids,
                        optimized_vehicle_ids=optimized_vehicle_ids,
                        tabu_assignment=tabu_assignment,
                        method=ROUTE_METHOD,
                    )
                if cbc_assignments is not None:
                    cbc_df, _ = post_cbc_congestion(
                        session,
                        run_configs_id=run_config.run_configs_id,
                        iteration_id=iteration_id,
                        all_vehicle_ids=all_vehicle_ids,
                        optimized_vehicle_ids=optimized_vehicle_ids,
                        cbc_assignment=cbc_assignments,  # keep your function’s param name in sync
                        route_alternatives=route_alternatives,
                        method=ROUTE_METHOD,
                    )

            # Baselines
            random_df, _ = compute_random_routes(session, run_configs_id=run_config.run_configs_id, iteration_id=iteration_id)
            shortest_dur_df, _ = compute_shortest_routes_dur(session, run_configs_id=run_config.run_configs_id, iteration_id=iteration_id)
            shortest_dis_df, _ = compute_shortest_routes_dist(session, run_configs_id=run_config.run_configs_id, iteration_id=iteration_id)

            prefix = f"{run_config.run_configs_id}_{iteration_id}"

            # Collect all available score series for consistent vmin/vmax
            score_sources = [
                shortest_dur_df.get("congestion_score"),
                shortest_dis_df.get("congestion_score"),
                post_qa_df.get("congestion_score"),
                random_df.get("congestion_score"),
                post_gurobi_df.get("congestion_score"),
                congestion_df.get("congestion_score"),
            ]
            for extra_df in (sa_df, tabu_df, cbc_df):
                if extra_df is not None and "congestion_score" in extra_df:
                    score_sources.append(extra_df["congestion_score"])

            # Filter out None, concat, and compute bounds safely
            valid_scores = [s for s in score_sources if s is not None and len(s) > 0]
            if valid_scores:
                all_scores = pd.concat(valid_scores, ignore_index=True)
                vmin, vmax = float(all_scores.min()), float(all_scores.max())
            else:
                vmin, vmax = 0.0, 1.0  # sensible default to avoid errors

            # Build heatmap inputs dynamically
            heatmap_inputs = [
                (congestion_df, "congestion_heatmap"),
                (affected_df, "affected_edges_df_heatmap"),
                (shortest_dur_df, "shortest_routes_dur_congestion_heatmap"),
                (shortest_dis_df, "shortest_routes_dis_congestion_heatmap"),
                (post_qa_df, "post_qa_congestion_heatmap"),
                (random_df, "random_routes_congestion_heatmap"),
                (post_gurobi_df, "post_gurobi_congestion_heatmap"),
            ]
            if FULL:
                if sa_df is not None:
                    heatmap_inputs.append((sa_df, "post_sa_congestion_heatmap"))
                if tabu_df is not None:
                    heatmap_inputs.append((tabu_df, "post_tabu_congestion_heatmap"))
                if cbc_df is not None:
                    heatmap_inputs.append((cbc_df, "post_cbc_congestion_heatmap"))

            # Render maps
            for df, name in heatmap_inputs:
                if df is not None and not df.empty:
                    visualize_and_save_congestion_heatmap(
                        edges_gdf=edges_gdf,
                        congestion_df=df,
                        output_filename=MAPS_OUTPUT_DIR / f"{prefix}_{name}.html",
                        vmin=vmin,
                        vmax=vmax,
                        offset_deg=OFFSET_DEG,
                    )

            # Persist summary (keep signature unchanged)
            save_congestion_summary(
                session,
                run_config_id=run_config.run_configs_id,
                iteration_id=iteration_id,
                edges_df=edges_df,
                congestion_df=congestion_df,
                post_qa_congestion_df=post_qa_df,
                shortest_routes_dur_df=shortest_dur_df,
                shortest_routes_dis_df=shortest_dis_df,
                random_routes_df=random_df,
                post_gurobi_df=post_gurobi_df,
                sa_df=sa_df,
                tabu_df=tabu_df,
                cbc_df=cbc_df
            )

            if FULL:
              run_sql_query(session, run_configs_id=run_config.run_configs_id, iteration_id=iteration_id, sql_name="congestion_results_full.sql")
            else:
              run_sql_query(session, run_configs_id=run_config.run_configs_id, iteration_id=iteration_id, sql_name="congestion_results.sql")


            logger.info("Workflow finished!")

    except Exception as e:
        logger.error("Workflow error: %s", str(e), exc_info=True)
    finally:
        logger.info("Total duration: %s", datetime.now() - start)


if __name__ == "__main__":
    main()
