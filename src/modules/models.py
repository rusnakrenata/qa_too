from sqlalchemy import create_engine, Column, Integer, String, Numeric, Text, ForeignKey, Float, JSON, DateTime, BigInteger, Numeric, Boolean, desc, text, Numeric, Index, PrimaryKeyConstraint
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from datetime import datetime
from decimal import Decimal
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
    created_at = Column(DateTime, default=datetime.utcnow)


class Vehicle(Base):
    """Vehicle table: vehicles starting/ending at nodes."""
    __tablename__ = 'vehicles'
    vehicle_id = Column(BigInteger, nullable=False)
    run_configs_id = Column(Integer, nullable=False)  # Link to RunConfig
    iteration_id = Column(Integer, nullable=False)
    origin_node_id = Column(Integer, nullable=False)
    destination_node_id = Column(Integer, nullable=False)
    origin_lat = Column(Numeric(9, 6), nullable=False)
    origin_lon = Column(Numeric(9, 6), nullable=False)
    destination_lat = Column(Numeric(9, 6), nullable=False)
    destination_lon = Column(Numeric(9, 6), nullable=False)
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
       # NEW: full path as IDs for later joins
    path_edge_ids = Column(JSON, nullable=True)   # list[int]
    path_node_ids = Column(JSON, nullable=True)   # list[int]

    created_at = Column(DateTime, default=datetime.utcnow)
    __table_args__ = (
        PrimaryKeyConstraint('run_configs_id', 'iteration_id', 'vehicle_id', 'route_id'),
        Index('idx_run_iter_vehicle_method', 'run_configs_id', 'iteration_id', 'vehicle_id')
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
    assignment = Column(JSON)
    n_nodes_distinct = Column(Integer, nullable=True)  # Number of distinct nodes
    overall_overlap = Column(Integer, nullable=True)  # Overall overlap
    energy = Column(Float)
    duration = Column(Float)
    solver_time = Column(Float)
    n_vehicles_selected = Column(Integer, nullable=True)  # Number of selected vehicles
    selected_vehicle_ids = Column(JSON, nullable=True)  # List of selected vehicle IDs
    created_at = Column(DateTime, default=datetime.utcnow)

class QuboRunStats(Base):
    __tablename__ = 'qubo_run_stats'
    qubo_run_stats_id = Column(Integer, primary_key=True, autoincrement=True)
    run_configs_id = Column(Integer, nullable=False)
    iteration_id = Column(Integer, nullable=False)
    n_vehicles = Column(Integer, nullable=False)
    lambda_penalty = Column(Float, nullable=False)
    max_weight_cvv = Column(Float, nullable=True)  # Maximum weight of the QUBO matrix
    max_weight_cvw = Column(Float, nullable=True)  # Maximum weight of the QUBO matrix
    n_nodes_distinct = Column(Integer, nullable=True)  # Number of distinct nodes
    overall_overlap = Column(Integer, nullable=True)  # Overall overlap
    created_at = Column(DateTime, default=datetime.utcnow)



class GurobiResult(Base):
    __tablename__ = 'gurobi_results'
    gurobi_result_id = Column(Integer, primary_key=True)
    run_configs_id = Column(Integer, nullable=False)
    iteration_id = Column(Integer, nullable=False)
    lambda_value = Column(Float)
    assignment = Column(JSON)  # Store variable assignment as JSON
    n_nodes_distinct = Column(Integer, nullable=True)  # Number of distinct nodes
    overall_overlap = Column(Integer, nullable=True)  # Overall overlap 
    objective_value = Column(Float)
    duration = Column(Float)
    solver_time = Column(Float)
    best_bound = Column(Float)       
    gap = Column(Float)     
    time_limit_seconds = Column(Integer, nullable=True)  # Time limit in seconds
    n_vehicles_selected = Column(Integer, nullable=True)  # Number of selected vehicles
    selected_vehicle_ids = Column(JSON, nullable=True)  # List of selected vehicle IDs
    created_at = Column(DateTime, default=datetime.utcnow)




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