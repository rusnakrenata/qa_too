from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, Float, JSON, DateTime, BigInteger, Numeric, Boolean, desc, text, Numeric, Index, PrimaryKeyConstraint
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Import engine and config from db_config
from db_config import engine

# Test connection and create declarative base
try:
    with engine.connect() as connection:
        Base = declarative_base()
except Exception as e:
    logger.error(f"Error connecting to MariaDB: {e}")

####### --TABLES-- #######

class City(Base):
    """City table: stores city metadata."""
    __tablename__ = 'cities'
    city_id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    node_count = Column(Integer)
    edge_count = Column(Integer)
    center_lat = Column(Numeric(9, 6), nullable=True)  # Center latitude for city subset
    center_lon = Column(Numeric(9, 6), nullable=True)  # Center longitude for city subset
    radius_km = Column(Float, nullable=True)  # Radius in km for city subset
    is_subset = Column(Boolean, default=False)  # Whether this is a city subset
    attraction_lat = Column(Numeric(9, 6), nullable=True)  # Attraction point latitude
    attraction_lon = Column(Numeric(9, 6), nullable=True)  # Attraction point longitude
    d_alternatives = Column(Integer, nullable=True)  # Number of route alternatives
    created_at = Column(DateTime, default=datetime.utcnow)

class Node(Base):
    """Node table: stores graph nodes."""
    __tablename__ = 'nodes'
    node_id = Column(Integer, primary_key=True)
    city_id = Column(Integer,  nullable=False)
    osmid = Column(String(255))
    x = Column(Numeric(9,6))
    y = Column(Numeric(9,6))
    geometry = Column(String(255), nullable=True)  # Store as WKT or GeoJSON
    created_at = Column(DateTime, default=datetime.utcnow)

class Edge(Base):
    """Edge table: stores graph edges."""
    __tablename__ = 'edges'
    edge_id = Column(Integer, primary_key=True)
    city_id = Column(Integer,  nullable=False)
    u = Column(String(255))
    v = Column(String(255))
    length = Column(String(255), nullable=True)
    geometry = Column(String(10000), nullable=True)  # Store as GeoJSON or WKT format for simplicity
    created_at = Column(DateTime, default=datetime.utcnow)

class Iteration(Base):
    """Iteration table: stores simulation iterations."""
    __tablename__ = 'iterations'
    iteration_id = Column(Integer, nullable=False)
    run_configs_id = Column(Integer, nullable=False)  # Link to RunConfig
    created_at = Column(DateTime, default=datetime.utcnow)
    __table_args__ = (
        PrimaryKeyConstraint('run_configs_id', 'iteration_id'),
        Index('idx_run_iter', 'run_configs_id', 'iteration_id')
    )

class RunConfig(Base):
    """RunConfig table: stores simulation configuration."""
    __tablename__ = 'run_configs'
    run_configs_id = Column(Integer, primary_key=True)
    city_id = Column(Integer,  nullable=False)
    n_vehicles = Column(Integer)
    k_alternatives = Column(Integer)
    min_length = Column(Integer)
    max_length = Column(Integer)
    time_step = Column(Integer, nullable=False)
    time_window = Column(Integer, nullable=False)
    distance_factor = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Vehicle(Base):
    """Vehicle table: stores vehicle metadata."""
    __tablename__ = 'vehicles'
    vehicle_id = Column(BigInteger, nullable=False) 
    run_configs_id = Column(Integer,  nullable=False)  # Link to RunConfig
    iteration_id = Column(Integer, nullable=False) 
    origin_edge_id = Column(Integer, nullable=False)
    origin_position_on_edge = Column(Float)
    origin_geometry = Column(String(255), nullable=True)
    destination_edge_id = Column(Integer, nullable=False)
    destination_position_on_edge = Column(Float)
    destination_geometry = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    __table_args__ = (
        PrimaryKeyConstraint('vehicle_id', 'run_configs_id', 'iteration_id'),
        Index('idx_run_iter_vehicle', 'run_configs_id', 'iteration_id', 'vehicle_id')
    )

class VehicleRoute(Base):
    """VehicleRoute table: stores route alternatives for each vehicle."""
    __tablename__ = 'vehicle_routes'
    vehicle_id = Column(Integer, nullable=False)
    run_configs_id = Column(Integer,  nullable=False)
    iteration_id = Column(Integer, nullable=False) 
    route_id = Column(Integer, nullable=False) 
    duration = Column(Integer)
    distance = Column(Integer)
    duration_in_traffic = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    __table_args__ = (
        PrimaryKeyConstraint('run_configs_id', 'iteration_id', 'vehicle_id', 'route_id'),
        Index('idx_run_iter_vehicle_method', 'run_configs_id', 'iteration_id', 'vehicle_id')
    )

class RoutePoint(Base):
    """RoutePoint table: stores points along each vehicle route."""
    __tablename__ = 'route_points'
    vehicle_id = Column(Integer, nullable=False)
    run_configs_id = Column(Integer, nullable=False)
    iteration_id = Column(Integer, nullable=False)
    route_id = Column(Integer, nullable=False)
    point_id = Column(Integer, nullable=False)
    edge_id = Column(Integer, nullable=False)
    cardinal = Column(String(255), nullable=True)
    speed = Column(Float)
    lat = Column(Numeric(9, 6))
    lon = Column(Numeric(9, 6))
    time = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    __table_args__ = (
        PrimaryKeyConstraint(
            'run_configs_id', 'iteration_id', 
            'edge_id', 'vehicle_id', 'route_id', 'point_id', 'time' ),
        Index('edge_id', 'time', 'cardinal', 'vehicle_id','lat','lon')
    )

class CongestionMap(Base):
    """CongestionMap table: stores pairwise congestion scores."""
    __tablename__ = 'congestion_map'
    congestion_map_id = Column(Integer, primary_key=True)
    run_configs_id = Column(Integer,  nullable=False)
    iteration_id = Column(Integer, nullable=False)
    edge_id = Column(Integer,  nullable=False)
    vehicle1 = Column(Integer,  nullable=False)
    vehicle1_route = Column(Integer, nullable=False)
    vehicle2 = Column(Integer,  nullable=False)
    vehicle2_route = Column(Integer, nullable=False)
    congestion_score = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    __table_args__ = (
        Index('idx_cm_cfg_iter_v1', 'run_configs_id', 'iteration_id', 'vehicle1', 'vehicle1_route'),
        Index('idx_cm_cfg_iter_v2', 'run_configs_id', 'iteration_id', 'vehicle2', 'vehicle2_route'),
        Index('idx_cm_cfg_iter_edge', 'run_configs_id', 'iteration_id', 'edge_id'),
    )

class QAResult(Base):
    """QAResult table: stores results of QUBO/QA optimization runs."""
    __tablename__ = 'qa_results'
    qa_result_id = Column(Integer, primary_key=True)
    run_configs_id = Column(Integer, nullable=False)
    iteration_id = Column(Integer, nullable=False)
    lambda_value = Column(Float)
    comp_type = Column(String(50))
    num_reads = Column(Integer)
    n_vehicles = Column(Integer)
    k_alternatives = Column(Integer)
    vehicle_ids = Column(JSON)
    assignment_valid = Column(Integer)
    assignment = Column(JSON)
    energy = Column(Float)
    duration = Column(Float)
    solver_time = Column(Float)
    qubo_path = Column(String(255))
    qubo_size = Column(Integer)
    qubo_density = Column(Float)
    cluster_id = Column(Integer)
    invalid_assignment_vehicles = Column(String(2000), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class QuboRunStats(Base):
    __tablename__ = 'qubo_run_stats'
    qubo_run_stats_id = Column(Integer, primary_key=True, autoincrement=True)
    run_configs_id = Column(Integer, nullable=False)
    iteration_id = Column(Integer, nullable=False)
    cluster_id = Column(Integer, nullable=False)
    filtering_percentage = Column(Float, nullable=True)
    cluster_resolution = Column(Float, nullable=True)  # Resolution of clustering
    n_vehicles = Column(Integer, nullable=False)
    n_filtered_vehicles = Column(Integer, nullable=False)
    max_weight = Column(Float, nullable=True)  # Maximum weight of the QUBO matrix
    created_at = Column(DateTime, default=datetime.utcnow)

class CongestionSummary(Base):
    """Stores per-edge congestion results for all, post-QA, shortest-duration, and shortest-distance congestion."""
    __tablename__ = 'congestion_summary'
    congestion_summary_id = Column(Integer, primary_key=True, autoincrement=True)
    run_configs_id = Column(Integer, nullable=False)
    iteration_id = Column(Integer, nullable=False)
    edge_id = Column(Integer, nullable=False)
    congestion_all = Column(Float, nullable=True)
    congestion_post_qa = Column(Float, nullable=True)
    congestion_post_sa = Column(Float, nullable=True)
    congestion_post_tabu = Column(Float, nullable=True)
    congestion_shortest_dur = Column(Float, nullable=True)
    congestion_shortest_dis = Column(Float, nullable=True)
    congestion_random = Column(Float, nullable=True)
    congestion_post_gurobi = Column(Float, nullable=True)
    congestion_post_cbc = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class QASelectedRoute(Base):
    """SelectedRoute table: stores the routes selected by QA optimization for each vehicle."""
    __tablename__ = 'qa_selected_routes'
    run_configs_id = Column(Integer, nullable=False)
    iteration_id = Column(Integer, nullable=False)
    vehicle_id = Column(Integer, nullable=False)
    route_id = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    __table_args__ = (
        PrimaryKeyConstraint('run_configs_id', 'iteration_id', 'vehicle_id', 'route_id'),
        Index('idx_run_iter_qa_selected_routes', 'run_configs_id', 'iteration_id', 'vehicle_id')
    )

class RandomRoute(Base):
    """SelectedRoute table: stores the routes selected by QA optimization for each vehicle."""
    __tablename__ = 'random_routes'
    run_configs_id = Column(Integer, nullable=False)
    iteration_id = Column(Integer, nullable=False)
    vehicle_id = Column(Integer, nullable=False)
    route_id = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    __table_args__ = (
        PrimaryKeyConstraint('run_configs_id', 'iteration_id', 'vehicle_id', 'route_id'),
        Index('idx_run_iter_random_routes', 'run_configs_id', 'iteration_id', 'vehicle_id')
    )

class GurobiResult(Base):
    __tablename__ = 'gurobi_results'
    gurobi_result_id = Column(Integer, primary_key=True)
    run_configs_id = Column(Integer, nullable=False)
    iteration_id = Column(Integer, nullable=False)
    assignment = Column(JSON)  # Store variable assignment as JSON
    objective_value = Column(Float)
    duration = Column(Float)
    solver_time = Column(Float)
    best_bound = Column(Float)       
    gap = Column(Float)     
    cluster_id = Column(Integer)
    time_limit_seconds = Column(Integer, nullable=True)  # Time limit in seconds
    created_at = Column(DateTime, default=datetime.utcnow)

class GurobiRoute(Base):
    """SelectedRoute table: stores the routes selected by QA optimization for each vehicle."""
    __tablename__ = 'gurobi_routes'
    run_configs_id = Column(Integer, nullable=False)
    iteration_id = Column(Integer, nullable=False)
    vehicle_id = Column(Integer, nullable=False)
    route_id = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    __table_args__ = (
        PrimaryKeyConstraint('run_configs_id', 'iteration_id', 'vehicle_id', 'route_id'),
        Index('idx_run_iter_gurobi_routes', 'run_configs_id', 'iteration_id', 'vehicle_id')
    )


class ShortestRouteDur(Base):
    __tablename__ = "shortest_routes_duration"
    run_configs_id = Column(Integer, nullable=False)
    iteration_id = Column(Integer, nullable=False)
    vehicle_id = Column(Integer, nullable=False)
    route_id = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    __table_args__ = (
        PrimaryKeyConstraint('run_configs_id', 'iteration_id', 'vehicle_id', 'route_id'),
        Index('idx_run_iter_shortest_routes_dur', 'run_configs_id', 'iteration_id', 'vehicle_id')
    )



class ShortestRouteDis(Base):
    __tablename__ = "shortest_routes_distance"
    run_configs_id = Column(Integer, nullable=False)
    iteration_id = Column(Integer, nullable=False)
    vehicle_id = Column(Integer, nullable=False)
    route_id = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    __table_args__ = (
        PrimaryKeyConstraint('run_configs_id', 'iteration_id', 'vehicle_id', 'route_id'),
        Index('idx_run_iter_shortest_routes_dis', 'run_configs_id', 'iteration_id', 'vehicle_id')
    )


class ObjectiveValue(Base):
    __tablename__ = "objective_values"
    run_configs_id = Column(Integer, nullable=False)
    iteration_id = Column(Integer, nullable=False)
    method = Column(String(32), nullable=False)  # 'qa', 'gurobi', 'random', etc.
    objective_value = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    __table_args__ = (
        PrimaryKeyConstraint('run_configs_id', 'iteration_id', 'method'),
    )


class SaResult(Base):
    __tablename__ = 'sa_results'
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_configs_id = Column(Integer, nullable=False)
    iteration_id = Column(Integer, nullable=False)
    num_reads = Column(Integer, nullable=False)
    n_vehicles = Column(Integer, nullable=False)
    k_alternatives = Column(Integer, nullable=False)
    vehicle_ids = Column(JSON, nullable=False)
    assignment_valid = Column(Boolean, nullable=False)
    assignment = Column(JSON, nullable=False)
    energy = Column(Float, nullable=False)
    duration = Column(Float, nullable=False)
    solver_time = Column(Float, nullable=False)
    invalid_assignment_vehicles = Column(String(2000), nullable=True)
    cluster_id = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
# Create all tables if they do not exist



class SaSelectedRoute(Base):
    __tablename__ = 'sa_selected_routes'
    run_configs_id = Column(Integer, nullable=False)
    iteration_id = Column(Integer, nullable=False)
    vehicle_id = Column(Integer, nullable=False)
    route_id = Column(Integer, nullable=False)
    __table_args__ = (
        PrimaryKeyConstraint('run_configs_id', 'iteration_id', 'vehicle_id', 'route_id'),
        Index('idx_run_iter_sa_selected_routes', 'run_configs_id', 'iteration_id', 'vehicle_id')
    )

class TabuResult(Base):
    __tablename__ = 'tabu_results'
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_configs_id = Column(Integer, nullable=False)
    iteration_id = Column(Integer, nullable=False)
    num_reads = Column(Integer, nullable=False)
    n_vehicles = Column(Integer, nullable=False)
    k_alternatives = Column(Integer, nullable=False)
    vehicle_ids = Column(JSON, nullable=False)
    assignment_valid = Column(Boolean, nullable=False)
    assignment = Column(JSON, nullable=False)
    energy = Column(Float, nullable=False)
    duration = Column(Float, nullable=False)
    solver_time = Column(Float, nullable=False)
    invalid_assignment_vehicles = Column(String(2000), nullable=True)
    cluster_id = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)


class TabuSelectedRoute(Base):
    __tablename__ = 'tabu_selected_routes'
    run_configs_id = Column(Integer, nullable=False)
    iteration_id = Column(Integer, nullable=False)
    vehicle_id = Column(Integer, nullable=False)
    route_id = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    __table_args__ = (
        PrimaryKeyConstraint('run_configs_id', 'iteration_id', 'vehicle_id', 'route_id'),
        Index('idx_run_iter_sa_selected_routes', 'run_configs_id', 'iteration_id', 'vehicle_id')
    )

class CbcResult(Base):
    __tablename__ = 'cbc_results'

    id = Column(Integer, primary_key=True)
    run_configs_id = Column(Integer, nullable=False)
    iteration_id = Column(Integer, nullable=False)
    assignment = Column(JSON, nullable=False)
    objective_value = Column(Float)
    duration = Column(Float)
    solver_time = Column(Float)
    status = Column(String(50), nullable=True)  # Status of the CBC solver
    cluster_id = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

class CbcRoute(Base):
    __tablename__ = 'cbc_routes'
    run_configs_id = Column(Integer, nullable=False)
    iteration_id = Column(Integer, nullable=False)
    vehicle_id = Column(Integer, nullable=False)
    route_id = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    __table_args__ = (
        PrimaryKeyConstraint('run_configs_id', 'iteration_id', 'vehicle_id', 'route_id'),
        Index('idx_run_iter_cbc_routes', 'run_configs_id', 'iteration_id', 'vehicle_id')
    )

Base.metadata.create_all(engine)
"""
Database models for the traffic optimization system.

Schema overview:
- City, Node, Edge: Graph structure of the city
- RunConfig, Iteration: Simulation configuration and runs
- Vehicle, VehicleRoute, RoutePoint: Vehicles and their possible routes
- CongestionMap: Pairwise congestion scores between vehicles/routes
- QAResult: Results of QUBO/QA optimization

Usage:
- Import models and Base for ORM operations
- Use `from db_config import get_session` for session management
- Always use context managers (with statements) for sessions to ensure proper cleanup

Example:
    from db_config import get_session
    with get_session() as session:
        ... # ORM operations
"""