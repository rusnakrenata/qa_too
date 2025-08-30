import os
import time
from pathlib import Path
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import Model, GRB, quicksum
from models import GurobiResult
from compute_weights import compute_qubo_overlap_weights
import logging

logger = logging.getLogger(__name__)

# Set Gurobi license path
project_root = Path(__file__).resolve().parents[2]
license_path = project_root / "gurobi.lic"
os.environ["GRB_LICENSE_FILE"] = str(license_path)


def gurobi_testing(
    session,
    run_configs_id: int,
    iteration_id: int,
    Q: dict,
    vehicle_routes_df: pd.DataFrame,
    lambda_value: float,
    time_limit_seconds: int = 300,
 ) -> tuple:
    """
    Solves a QUBO problem using Gurobi and stores results in the database.


    Returns:
        Tuple containing:
            - result: Variable assignments.
            - objective_value: Optimal objective value.
    """
    # Ensure time limit between 1 and 10 minutes
    time_limit_seconds = min(time_limit_seconds, 600)

    start_time_model = time.perf_counter()

    # Initialize Gurobi model
    model = Model("QUBO_Traffic_Overlap")
    model.setParam("TimeLimit", time_limit_seconds)
    model.setParam("OutputFlag", 0)

    # Define binary variables
    variables = {}
    for i, j in Q:
        if i not in variables:
            variables[i] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}")
        if j not in variables:
            variables[j] = model.addVar(vtype=GRB.BINARY, name=f"x_{j}")

    model.update()

    # Set QUBO objective
    obj = quicksum(Q[i, j] * variables[i] * variables[j] for i, j in Q)
    model.setObjective(obj, GRB.MINIMIZE)


    # Solve model
    start_time_solver = time.perf_counter()
    model.optimize()

    # Measure durations
    model_duration = time.perf_counter() - start_time_model
    solver_duration = time.perf_counter() - start_time_solver

    # Extract solution
    #result = {v.VarName: int(v.X) for v in model.getVars()}
    objective_value = model.ObjVal if model.SolCount > 0 else None
    best_bound = model.ObjBound if model.SolCount > 0 else None
    gap = model.MIPGap if model.SolCount > 0 else None

    var_indices = sorted(int(v.VarName.split("_")[1]) for v in model.getVars())
    assignment = [int(model.getVarByName(f"x_{i}").X) for i in var_indices]
    print("Assignment Gurobi:", assignment)

    # === Selected vehicles
    n_vehicles_selected = sum(assignment)
    selected_vehicle_ids = vehicle_routes_df.loc[
        np.array(assignment).astype(bool), "vehicle_id"
    ].tolist()

    diag, offdiag, max_weight_cvw, max_weight_cvv, n_nodes_distinct, overall_overlap = compute_qubo_overlap_weights(vehicle_routes_df, node_column="path_node_ids", assignment=assignment)


    # Store results in the database
    gurobi_result = GurobiResult(
        run_configs_id=run_configs_id,
        iteration_id=iteration_id,
        lambda_value=lambda_value,
        assignment=assignment,
        n_nodes_distinct=n_nodes_distinct,
        overall_overlap=overall_overlap,    
        objective_value=objective_value,
        duration=model_duration,
        solver_time=solver_duration,
        best_bound=best_bound,
        gap=gap,
        time_limit_seconds=time_limit_seconds,
        n_vehicles_selected=n_vehicles_selected,
        selected_vehicle_ids=selected_vehicle_ids,  
    )
    session.add(gurobi_result)
    session.commit()
    session.close()

    logger.info(f"Gurobi testing complete:  energy={objective_value}, duration={model_duration:.2f}s, solver_time={solver_duration:.2f}s")

    return selected_vehicle_ids, objective_value