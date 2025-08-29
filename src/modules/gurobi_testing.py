import os
import time
from pathlib import Path
import gurobipy as gp
from gurobipy import Model, GRB, quicksum
from models import GurobiResult

# Set Gurobi license path
project_root = Path(__file__).resolve().parents[2]
license_path = project_root / "gurobi.lic"
os.environ["GRB_LICENSE_FILE"] = str(license_path)


def gurobi_testing(
    session,
    run_configs_id: int,
    iteration_id: int,
    Q: dict,
    n_vehicles: int,
    route_alternatives: int,
    comp_type: str,
    time_limit_seconds: int = 300,
    cluster_id: int = 0
) -> tuple:
    """
    Solves a QUBO problem using Gurobi and stores results in the database.

    Args:
        session: SQLAlchemy session object.
        run_configs_id: Run configuration identifier.
        iteration_id: Iteration identifier.
        Q: QUBO matrix as a dictionary with keys (i, j).
        n_vehicles: Number of vehicles.
        route_alternatives: Number of route alternatives per vehicle.
        comp_type: Optimization mode ('hybrid_cqm' or others).
        time_limit_seconds: Solver time limit in seconds (default 300s).
        cluster_id: Identifier for cluster grouping (default 0).

    Returns:
        Tuple containing:
            - result: Variable assignments.
            - objective_value: Optimal objective value.
    """
    # Ensure time limit between 1 and 10 minutes
    time_limit_seconds = min(time_limit_seconds, 600)

    start_time_model = time.perf_counter()

    # Initialize Gurobi model
    model = Model("QUBO_Traffic")
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

    # Add one-hot constraints if using hybrid_cqm
    if comp_type == "hybrid_cqm":
        for i in range(n_vehicles):
            terms = [variables[i * route_alternatives + k] for k in range(route_alternatives)]
            model.addConstr(quicksum(terms) == 1, name=f"one_hot_vehicle_{i}")

    # Solve model
    start_time_solver = time.perf_counter()
    model.optimize()

    # Measure durations
    model_duration = time.perf_counter() - start_time_model
    solver_duration = time.perf_counter() - start_time_solver

    # Extract solution
    result = {v.VarName: int(v.X) for v in model.getVars()}
    objective_value = model.ObjVal if model.SolCount > 0 else None
    best_bound = model.ObjBound if model.SolCount > 0 else None
    gap = model.MIPGap if model.SolCount > 0 else None

    assignment = list(result.items())

    # Store results in the database
    gurobi_result = GurobiResult(
        run_configs_id=run_configs_id,
        iteration_id=iteration_id,
        assignment=assignment,
        objective_value=objective_value,
        duration=model_duration,
        solver_time=solver_duration,
        best_bound=best_bound,
        gap=gap,
        time_limit_seconds=time_limit_seconds,
        cluster_id=cluster_id
    )
    session.add(gurobi_result)
    session.commit()
    session.close()

    return result, objective_value