# ---------- IMPORT MODULES ----------
import pandas as pd
import numpy as np
import logging
import os
import geopandas as gpd
from pathlib import Path
from typing import Any, List, Tuple, Optional
from sqlalchemy import text as sa_text
from sqlalchemy.orm import sessionmaker



# ---------- MODULES ----------
from get_city_graph import get_city_graph
from get_city_data_from_db import get_city_data_from_db
from store_city_to_db import store_city_to_db
from get_or_create_run_config import get_or_create_run_config
from create_iteration import create_iteration
from generate_vehicles_random import generate_vehicles
from generate_vehicles_attraction import generate_vehicles_attraction
from generate_vehicle_routes import generate_vehicle_routes
from generate_congestion import generate_congestion
from plot_congestion_heatmap import plot_congestion_heatmap_interactive
from get_congestion_weights import get_congestion_weights
from filter_routes_for_qubo import get_clusters_by_connectivity
from qubo_matrix import qubo_matrix
from compute_shortest_routes_dist import compute_shortest_routes_dist
from compute_shortest_routes_dur import compute_shortest_routes_dur 
from compute_random_routes import compute_random_routes
from post_qa_congestion import post_qa_congestion
from qa_testing import qa_testing
from sa_testing import sa_testing
from post_sa_congestion import post_sa_congestion
from gurobi_testing import gurobi_testing
from post_gurobi_congestion import post_gurobi_congestion
from tabu_testing import tabu_testing   
from post_tabu_congestion import post_tabu_congestion
from cbc_testing import cbc_testing
from post_cbc_congestion import post_cbc_congestion
from datetime import datetime



# ---------- CONFIGURATION ----------
from models import * #City, Node, Edge, RunConfig, Iteration, Vehicle, VehicleRoute, CongestionMap, RoutePoint  # adjust to your actual model imports
from config import *

# Named constants
OFFSET_DEG = 0.0000025
QUBO_OUTPUT_DIR = Path("files_csv")
MAPS_OUTPUT_DIR = Path("files_html")
CONGESTION_WEIGHTS_FILENAME = QUBO_OUTPUT_DIR / "congestion_weights.csv"
# Removed static HTML filenames

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def create_db_session() -> Any:
    """Create a new SQLAlchemy session."""
    Session = sessionmaker(bind=engine, autocommit=False)
    return Session()


def get_or_create_city(session) -> Any:
    """Get or create the city in the database."""
    # Check if we're looking for a city subset or full city
    if CENTER_COORDS is not None:
        # Look for existing city subset with matching coordinates
        lat, lon = CENTER_COORDS  # type: ignore
        city = session.query(City).filter_by(
            name=CITY_NAME,
            center_lat=lat,
            center_lon=lon,
            radius_km=RADIUS_KM,
            is_subset=True
        ).first()
        
        if not city:
            # Create new city subset
            nodes, edges = get_city_graph(CITY_NAME, center_coords=CENTER_COORDS, radius_km =RADIUS_KM)
            city = store_city_to_db(
                session, CITY_NAME, nodes, edges, City, Node, Edge,
                center_coords=CENTER_COORDS, radius_km=RADIUS_KM
            )
    else:
        # Look for existing full city
        city = session.query(City).filter_by(
            name=CITY_NAME,
            is_subset=False
        ).first()
        
        if not city:
            # Create new full city
            nodes, edges = get_city_graph(CITY_NAME)
            city = store_city_to_db(session, CITY_NAME, nodes, edges, City, Node, Edge)
    
    return city


def get_or_create_run_config_for_city(session, city) -> Any:
    """Get or create a run configuration for the city."""
    return get_or_create_run_config(
        session, city.city_id, RunConfig, N_VEHICLES, K_ALTERNATIVES,
        MIN_LENGTH, MAX_LENGTH, TIME_STEP, TIME_WINDOW
    )


def create_simulation_iteration(session, run_configs_id) -> Optional[int]:
    """Create a new simulation iteration."""
    iteration_id = create_iteration(session, run_configs_id, None, Iteration)
    if iteration_id is None:
        logger.warning("No new iteration created. Exiting workflow.")
    return iteration_id





def generate_and_store_vehicles(
    session,
    run_configs,
    iteration_id,
    attraction_point=None,        # (lat, lon) tuple or None
    d_alternatives=None           # int or None
) -> Any:
    """Generate vehicles and store them in the database."""
    logger.info("Generate vehicles at: %s", datetime.now())
    
    # Load edges GeoDataFrame
    _, edges_gdf = get_city_data_from_db(session, run_configs.city_id)
    
    if attraction_point is not None and d_alternatives is not None:
        # Use attraction-aware version
        logger.info("Using attraction-based vehicle generation.")
        return generate_vehicles_attraction(
            session=session,
            Vehicle=Vehicle,
            run_config_id=run_configs.run_configs_id,
            iteration_id=iteration_id,
            edges_gdf=edges_gdf,
            nr_vehicles=N_VEHICLES,
            min_length=MIN_LENGTH,
            max_length=MAX_LENGTH,
            attraction_point=attraction_point,
            d_alternatives=d_alternatives
        )
    else:
        # Use default version
        logger.info("Using random vehicle generation.")
        return generate_vehicles(
            session=session,
            Vehicle=Vehicle,
            run_config_id=run_configs.run_configs_id,
            iteration_id=iteration_id,
            edges_gdf=edges_gdf,
            nr_vehicles=N_VEHICLES,
            min_length=MIN_LENGTH,
            max_length=MAX_LENGTH
        )



def generate_and_store_routes(session, run_configs_id, iteration_id, vehicles_gdf, edges) -> pd.DataFrame:
    """Generate vehicle routes and store them in the database."""
    logger.info("Generate vehicle routes at: %s", datetime.now())
    vehicle_routes_df = generate_vehicle_routes(
        session, VehicleRoute, RoutePoint,
        run_configs_id, iteration_id,
        vehicles_gdf, edges, K_ALTERNATIVES, TIME_STEP, TIME_WINDOW
    )
    return vehicle_routes_df


def compute_and_store_congestion(session, run_configs_id, iteration_id) -> pd.DataFrame:
    """Compute congestion and store in the database."""
    logger.info("Compute congestion at: %s", datetime.now())
    congestion_df = generate_congestion(
        session,
        run_configs_id, iteration_id,TIME_STEP
    )
    return congestion_df  # Do not groupby h


def get_k_alternatives(session, run_configs_id, iteration_id):
    sql = sa_text("""
        SELECT MAX(route_id) as max_route_id
        FROM vehicle_routes
        WHERE run_configs_id = :run_configs_id AND iteration_id = :iteration_id
    """)
    result = session.execute(sql, {'run_configs_id': run_configs_id, 'iteration_id': iteration_id})
    max_route_id = result.scalar()
    return int(max_route_id) if max_route_id is not None else 1


def build_and_save_qubo_matrix(
    vehicle_routes_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    duration_penalty_df: pd.DataFrame,
    session: Any,
    run_configs_id: int,
    iteration_id: int,
    t: int,
    lambda_strategy: str,
    cluster_id: int,
    vehicle_ids_filtered: List
) -> Tuple[Any, List[Any], pd.DataFrame, float]:
    """Build QUBO matrix and save run stats."""
    Q,  t_Q, lambda_penalty = qubo_matrix(
        t,  weights_df, duration_penalty_df, 
        vehicle_ids_filtered,
        vehicle_routes_df,
        lambda_strategy=lambda_strategy,
    )
    # Find the matrix size
    max_index = 0
    if Q:
        max_index = max(max(k[0], k[1]) for k in Q.keys()) + 1

    # Create the matrix and fill it
    Q_matrix = np.zeros((max_index, max_index))
    for (q1, q2), value in Q.items():
        Q_matrix[q1, q2] = value

    # Create DataFrame with proper column and row labels
    Q_df = pd.DataFrame(Q_matrix)
    
    # Add column numbers as headers (formatted to 9 characters)
    Q_df.columns = [f'{0:18g}'] + [f'{i:9g}' for i in range(1, max_index)]
    Q_df.index = [f'{i:9g}' for i in range(max_index)]
    
   
    # Save with custom formatting - 9 characters per number

    Q_df.to_csv( QUBO_OUTPUT_DIR / "qubo_matrix_{}.csv".format(cluster_id), index=True, header=True, float_format='%9g')
    N_FILTERED = len(vehicle_ids_filtered)
    logger.info("Filtered vehicles number: %d", N_FILTERED)
    stats = QuboRunStats(
        run_configs_id=run_configs_id,
        iteration_id=iteration_id,
        filtering_percentage=N_FILTERED/N_VEHICLES,
        cluster_id=cluster_id,
        n_vehicles=N_VEHICLES,
        n_filtered_vehicles=N_FILTERED
    )
    session.add(stats)
    session.commit()
    return Q,  t_Q, lambda_penalty


def visualize_and_save_congestion(
    edges: gpd.GeoDataFrame,
    congestion_df: pd.DataFrame,
    affected_edges_df: pd.DataFrame,
    shortest_routes_dur_df: pd.DataFrame,
    shortest_routes_dis_df: pd.DataFrame,
    post_qa_congestion_df: pd.DataFrame,
    post_gurobi_congestion_df: pd.DataFrame,
    random_routes_df: pd.DataFrame,
    congestion_heatmap_filename: Path,
    affected_edges_heatmap_filename: Path,
    shortest_dur_heatmap_filename: Path,
    shortest_dis_heatmap_filename: Path,
    post_qa_heatmap_filename: Path,
    random_routes_heatmap_filename: Path,
    post_gurobi_heatmap_file: Path
) -> None:
    """Visualize congestion and save heatmaps."""
    all_scores = pd.concat([
        shortest_routes_dur_df['congestion_score'],
        shortest_routes_dis_df['congestion_score'],
        post_qa_congestion_df['congestion_score'],
        random_routes_df['congestion_score'],
        post_gurobi_congestion_df['congestion_score']
    ])
    vmin = float(all_scores.min())
    vmax = float(all_scores.max())
    plot_map = plot_congestion_heatmap_interactive(edges, congestion_df, offset_deg=OFFSET_DEG)
    if plot_map is not None:
        plot_map.save(congestion_heatmap_filename)
    plot_map_affected_edges = plot_congestion_heatmap_interactive(edges, affected_edges_df, offset_deg=OFFSET_DEG)
    if plot_map_affected_edges is not None:
        plot_map_affected_edges.save(affected_edges_heatmap_filename)
    plot_map_dur = plot_congestion_heatmap_interactive(edges, shortest_routes_dur_df, offset_deg=OFFSET_DEG, vmin=vmin, vmax=vmax)
    if plot_map_dur is not None:
        plot_map_dur.save(shortest_dur_heatmap_filename)
    plot_map_dis = plot_congestion_heatmap_interactive(edges, shortest_routes_dis_df, offset_deg=OFFSET_DEG, vmin=vmin, vmax=vmax)
    if plot_map_dis is not None:
        plot_map_dis.save(shortest_dis_heatmap_filename)
    plot_map_post_qa = plot_congestion_heatmap_interactive(edges, post_qa_congestion_df, offset_deg=OFFSET_DEG, vmin=vmin, vmax=vmax)
    if plot_map_post_qa is not None:
        plot_map_post_qa.save(post_qa_heatmap_filename)
    plot_map_random = plot_congestion_heatmap_interactive(edges, random_routes_df, offset_deg=OFFSET_DEG, vmin=vmin, vmax=vmax)
    if plot_map_random is not None:
        plot_map_random.save(random_routes_heatmap_filename)
    plot_map_post_gurobi = plot_congestion_heatmap_interactive(edges, post_gurobi_congestion_df, offset_deg=OFFSET_DEG, vmin=vmin, vmax=vmax)
    if plot_map_post_gurobi is not None:
        plot_map_post_gurobi.save(post_gurobi_heatmap_file)


def save_congestion_summary(
    session: Any,
    edges: pd.DataFrame,
    congestion_df: pd.DataFrame,
    post_qa_congestion_df: pd.DataFrame,
    post_sa_congestion_df: pd.DataFrame,
    post_tabu_congestion_df: pd.DataFrame,
    shortest_routes_dur_df: pd.DataFrame,
    shortest_routes_dis_df: pd.DataFrame,
    random_routes_df: pd.DataFrame,
    post_gurobi_df: pd.DataFrame,
    post_cbc_congestion: pd.DataFrame,
    run_config: RunConfig,
    iteration_id: int
) -> None:
    """Save congestion summary to the database."""
    congestion_df_grouped = congestion_df.groupby('edge_id', as_index=False).agg({'congestion_score': 'sum'})
    merged = pd.DataFrame({'edge_id': edges.drop_duplicates(subset='edge_id')['edge_id']})
    merged = merged.merge(congestion_df_grouped[['edge_id', 'congestion_score']].rename(columns={'congestion_score': 'congestion_all'}), on='edge_id', how='left')  # type: ignore
    merged = merged.merge(post_qa_congestion_df[['edge_id', 'congestion_score']].rename(columns={'congestion_score': 'congestion_post_qa'}), on='edge_id', how='left')  # type: ignore
    merged = merged.merge(post_sa_congestion_df[['edge_id', 'congestion_score']].rename(columns={'congestion_score': 'congestion_post_sa'}), on='edge_id', how='left')  # type: ignore
    merged = merged.merge(post_tabu_congestion_df[['edge_id', 'congestion_score']].rename(columns={'congestion_score': 'congestion_post_tabu'}), on='edge_id', how='left')  # type: ignore
    merged = merged.merge(shortest_routes_dur_df[['edge_id', 'congestion_score']].rename(columns={'congestion_score': 'congestion_shortest_dur'}), on='edge_id', how='left')  # type: ignore
    merged = merged.merge(shortest_routes_dis_df[['edge_id', 'congestion_score']].rename(columns={'congestion_score': 'congestion_shortest_dis'}), on='edge_id', how='left')  # type: ignore
    merged = merged.merge(random_routes_df[['edge_id', 'congestion_score']].rename(columns={'congestion_score': 'congestion_random'}), on='edge_id', how='left')  # type: ignore
    merged = merged.merge(post_gurobi_df[['edge_id', 'congestion_score']].rename(columns={'congestion_score': 'congestion_post_gurobi'}), on='edge_id', how='left')  # type: ignore
    merged = merged.merge(post_cbc_congestion[['edge_id', 'congestion_score']].rename(columns={'congestion_score': 'congestion_post_cbc'}), on='edge_id', how='left')  # type: ignore
    merged = merged.fillna(0)
    records = [
        CongestionSummary(
            run_configs_id=run_config.run_configs_id,
            iteration_id=iteration_id,
            edge_id=int(row['edge_id']),
            congestion_all=float(row['congestion_all']),
            congestion_post_qa=float(row['congestion_post_qa']),
            congestion_post_sa=float(row['congestion_post_sa']),
            congestion_post_tabu=float(row['congestion_post_tabu']),
            congestion_shortest_dur=float(row['congestion_shortest_dur']),
            congestion_shortest_dis=float(row['congestion_shortest_dis']),
            congestion_random=float(row['congestion_random']),
            congestion_post_gurobi=float(row['congestion_post_gurobi']),
            congestion_post_cbc=float(row['congestion_post_cbc'])
        )
        for _, row in merged.iterrows()
    ]
    session.add_all(records)
    session.commit()


def save_dist_dur_summary(session, run_configs_id, iteration_id):
    """Save distance and duration summary to the database using dist_dur_results.sql."""
    import os
    from models import DistDurSummary
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sql_file = os.path.join(base_dir, 'sql', 'dist_dur_results.sql')
    with open(sql_file, 'r') as f:
        sql = f.read()
    # Replace %s placeholders with :run_configs_id and :iteration_id for SQLAlchemy
    sql = sql.replace('%s', ':run_configs_id', 1)
    sql = sql.replace('%s', ':iteration_id', 1)
    result = session.execute(sa_text(sql), {'run_configs_id': run_configs_id, 'iteration_id': iteration_id})
    row = result.fetchone()
    if row is not None:
        summary = DistDurSummary(
            run_configs_id=run_configs_id,
            iteration_id=iteration_id,
            shortest_dist=row[0],
            shortest_dur=row[1],
            post_qa_dist=row[2],
            post_qa_dur=row[3],
            rnd_dist=row[4],
            rnd_dur=row[5],
            post_gurobi_dist=row[6],
            post_gurobi_dur=row[7]

        )
        session.add(summary)
        session.commit()
    else:
        logger.warning("No distance/duration summary results found for run_configs_id=%s, iteration_id=%s", run_configs_id, iteration_id)


def run_congestion_results_sql(session, run_configs_id, iteration_id):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sql_file = os.path.join(base_dir, 'sql', 'congestion_results.sql')
    with open(sql_file, 'r') as f:
        sql = f.read()
    # Split queries (naive split on ';')
    queries = [q.strip() for q in sql.split(';') if q.strip()]
    params = (run_configs_id, iteration_id)
    for i, query in enumerate(queries):
        print(f'--- Query {i+1} ---')
        try:
            df = pd.read_sql(query, session.bind, params=params)
            print(df)
        except Exception as e:
            print(f'Error running query {i+1}: {e}')


def check_bqm_against_solver_limits(Q):
    import dimod
    from dwave.system import LeapHybridBQMSampler, LeapHybridCQMSampler

    dwave_constraints_check= True
    bqm = dimod.BQM.from_qubo(Q)
    num_variables = len(bqm.variables)
    num_linear = len(bqm.linear)
    num_quadratic = len(bqm.quadratic)
    num_biases = num_linear + num_quadratic

    sampler = LeapHybridCQMSampler()
    max_vars_cqm = sampler.properties["maximum_number_of_variables"]
    max_biases_cqm = sampler.properties["maximum_number_of_biases"]

    sampler = LeapHybridBQMSampler()
    max_vars_bqm = sampler.properties["maximum_number_of_variables"]
    max_biases_bqm = sampler.properties["maximum_number_of_biases"]
    print("Number of variables:", num_variables)
    print("Number of linear biases:", num_linear)
    print("Number of quadratic biases:", num_quadratic)
    print("Total number of biases:", num_biases)
    print("BQM Solver maximum_number_of_variables:", max_vars_bqm)
    print("BQM Solver maximum_number_of_biases:", max_biases_bqm)
    print("CQM Solver maximum_number_of_variables:", max_vars_cqm)
    print("CQM Solver maximum_number_of_biases:", max_biases_cqm)
    if num_variables > max(max_vars_bqm, max_vars_cqm):
        dwave_constraints_check = False
        logger.warning("Too many variables for this solver!")
    if num_biases > max(max_biases_bqm, max_biases_cqm):
        dwave_constraints_check = False
        logger.warning("Too many biases for this solver!")
    return dwave_constraints_check

def process_clusters(clusters, session, run_config, iteration_id, vehicle_routes_df, weights_df, duration_penalty_df, all_vehicle_ids):
    qa_assignment, sa_assignement, tabu_assignement, gurobi_assignement, cbc_assignement, all_filtered_vehicle_ids_list, all_affected_edges = [], [], [], [], [], [], []

    for idx, (filtered_ids, affected_edges_df, total_congestion, size) in enumerate(clusters):
        logger.info(f"Processing cluster {idx + 1}/{len(clusters)}: {size} vehicles, congestion: {total_congestion}")

        if affected_edges_df is not None and not affected_edges_df.empty:
            affected_edges_df = affected_edges_df.copy()
            affected_edges_df["cluster_id"] = idx
            all_affected_edges.append(affected_edges_df)

        t = get_k_alternatives(session, run_config.run_configs_id, iteration_id)
        Q, t_Q, lambda_penalty = build_and_save_qubo_matrix(vehicle_routes_df, weights_df, duration_penalty_df, session,
            run_config.run_configs_id, iteration_id, t,  idx, filtered_ids)

        qa_result = qa_testing(
            Q=Q,
            run_configs_id=run_config.run_configs_id,
            iteration_id=iteration_id,
            session=session,
            n=len(filtered_ids),
            t=t,
            vehicle_ids=filtered_ids,
            lambda_value=lambda_penalty,
            comp_type=COMP_TYPE,
            num_reads=10,
            vehicle_routes_df=vehicle_routes_df,
            cluster_id=idx
        )
        qa_assignment += qa_result['assignment']


        sa_result = sa_testing(
            Q=Q,
            run_configs_id=run_config.run_configs_id,
            iteration_id=iteration_id,
            session=session,
            n=len(filtered_ids),
            t=t,
            vehicle_ids=filtered_ids,
            num_reads=10,
            vehicle_routes_df=vehicle_routes_df,
            cluster_id=idx
        )
        sa_assignement += sa_result['assignment']

        tabu_result = tabu_testing(
            Q=Q,
            run_configs_id=run_config.run_configs_id,
            iteration_id=iteration_id,
            session=session,
            n=len(filtered_ids),
            t=t,
            vehicle_ids=filtered_ids,
            num_reads=10,
            vehicle_routes_df=vehicle_routes_df,
            cluster_id=idx
        )
        tabu_assignement += tabu_result['assignment']

        cbc_result = cbc_testing(Q, run_config.run_configs_id, iteration_id, session,
                                    time_limit_seconds=qa_result['solver_time'], cluster_id=idx)

        sorted_cbc = [v for k, v in sorted(cbc_result.items(), key=lambda i: int(i[0].split('_')[1]))]
        cbc_assignement += sorted_cbc

        gurobi_result, _ = gurobi_testing(Q, t_Q, run_config.run_configs_id, iteration_id, session,
                                                  time_limit_seconds=qa_result['solver_time'], cluster_id=idx)
        sorted_gurobi = [v for k, v in sorted(gurobi_result.items(), key=lambda i: int(i[0].split('_')[1]))]
        gurobi_assignement += sorted_gurobi


        all_filtered_vehicle_ids_list.append(filtered_ids)

    return qa_assignment, sa_assignement, tabu_assignement, gurobi_assignement, cbc_assignement, all_filtered_vehicle_ids_list, all_affected_edges

def main():
    start = datetime.now()
    try:
        with create_db_session() as session:
            logger.info("Start workflow: %s", start)
            city = get_or_create_city(session)
            _, edges = get_city_data_from_db(session, city.city_id)
            edges = gpd.GeoDataFrame(edges) if not isinstance(edges, gpd.GeoDataFrame) else edges
            run_config = get_or_create_run_config_for_city(session, city)
            iteration_id = create_simulation_iteration(session, run_config.run_configs_id)
            if not iteration_id:
                return

            vehicles = generate_and_store_vehicles(session, run_config, iteration_id, attraction_point=ATTRACTION_POINT, d_alternatives=D_ALTERNATIVES)
            routes_df = generate_and_store_routes(session, run_config.run_configs_id, iteration_id, vehicles, edges)
            congestion_df = compute_and_store_congestion(session, run_config.run_configs_id, iteration_id)
            weights_df, duration_penalty_df = get_congestion_weights(session, run_config.run_configs_id, iteration_id)
            weights_df.to_csv(CONGESTION_WEIGHTS_FILENAME, index=False)

            all_vehicle_ids = vehicles["vehicle_id"].tolist()
            clusters = get_clusters_by_connectivity(congestion_df, resolution=1.2, min_cluster_size=MIN_CLUSTER_SIZE) or [
                (all_vehicle_ids, congestion_df.groupby('edge_id', as_index=False).agg({'congestion_score': 'sum'}), 0.0, len(all_vehicle_ids))
            ]

            clusters = clusters[:MAX_CLUSTERS] if MAX_CLUSTERS else clusters
            qa_assignments, sa_assignement, tabu_assignement, gurobi_assignement, cbc_assignement, all_filtered_ids, affected_edges = process_clusters(
                clusters, session, run_config, iteration_id, routes_df, weights_df, duration_penalty_df, all_vehicle_ids
            )
            session.commit()

            affected_df = pd.concat(affected_edges, ignore_index=True) if affected_edges else pd.DataFrame(columns=['edge_id', 'congestion_score', 'cluster_id'])
            all_filtered = [vid for sublist in all_filtered_ids for vid in sublist]

            post_qa_df, _ = post_qa_congestion(session, run_config.run_configs_id, iteration_id, all_vehicle_ids, all_filtered, qa_assignments, ROUTE_METHOD)
            post_sa_df, _ = post_sa_congestion(session, run_config.run_configs_id, iteration_id, all_vehicle_ids, all_filtered, sa_assignement, ROUTE_METHOD)
            post_tabu_df, _ = post_tabu_congestion(session, run_config.run_configs_id, iteration_id, all_vehicle_ids, all_filtered, tabu_assignement, ROUTE_METHOD)
            t = get_k_alternatives(session, run_config.run_configs_id, iteration_id)
            post_gurobi_df, _ = post_gurobi_congestion(session, run_config.run_configs_id, iteration_id, all_vehicle_ids, all_filtered, gurobi_assignement, t, ROUTE_METHOD)
            post_cbc_df, _ = post_cbc_congestion(session, run_config.run_configs_id, iteration_id, all_vehicle_ids, all_filtered, cbc_assignement, t, ROUTE_METHOD)

            post_qa_df = post_qa_df.groupby('edge_id', as_index=False).agg({'congestion_score': 'sum'})
            post_sa_df = post_sa_df.groupby('edge_id', as_index=False).agg({'congestion_score': 'sum'})
            post_tabu_df = post_tabu_df.groupby('edge_id', as_index=False).agg({'congestion_score': 'sum'})
            post_gurobi_df = post_gurobi_df.groupby('edge_id', as_index=False).agg({'congestion_score': 'sum'})
            post_cbc_df = post_cbc_df.groupby('edge_id', as_index=False).agg({'congestion_score': 'sum'})

            random_df, _ = compute_random_routes(session, run_config.run_configs_id, iteration_id)
            shortest_dur_df, _ = compute_shortest_routes_dur(session, run_config.run_configs_id, iteration_id)
            shortest_dis_df, _ = compute_shortest_routes_dist(session, run_config.run_configs_id, iteration_id)

            prefix = f"{run_config.run_configs_id}_{iteration_id}"
            visualize_and_save_congestion(
                edges, congestion_df, affected_df, shortest_dur_df, shortest_dis_df,
                post_qa_df, post_gurobi_df, random_df,
                MAPS_OUTPUT_DIR / f"{prefix}_congestion_heatmap.html",
                MAPS_OUTPUT_DIR / f"{prefix}_affected_edges_heatmap.html",
                MAPS_OUTPUT_DIR / f"{prefix}_shortest_routes_dur_congestion_heatmap.html",
                MAPS_OUTPUT_DIR / f"{prefix}_shortest_routes_dis_congestion_heatmap.html",
                MAPS_OUTPUT_DIR / f"{prefix}_post_qa_congestion_heatmap.html",
                MAPS_OUTPUT_DIR / f"{prefix}_random_routes_congestion_heatmap.html",
                MAPS_OUTPUT_DIR / f"{prefix}_post_gurobi_congestion_heatmap.html"
            )

            save_congestion_summary(session, edges, congestion_df, post_qa_df, post_sa_df, post_tabu_df, shortest_dur_df, shortest_dis_df, random_df, post_gurobi_df, post_cbc_df, run_config, iteration_id)
            save_dist_dur_summary(session, run_config.run_configs_id, iteration_id)
            run_congestion_results_sql(session, run_config.run_configs_id, iteration_id)

            logger.info("Workflow finished!")
    except Exception as e:
        logger.error("Workflow error: %s", str(e), exc_info=True)
    finally:
        logger.info("Total duration: %s", datetime.now() - start)


if __name__ == "__main__":
    main()
