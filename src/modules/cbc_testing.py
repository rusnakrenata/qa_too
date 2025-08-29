import time
from pulp import (
    LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD,
    LpBinary, LpStatusOptimal, LpStatus
)
from models import CbcResult
import logging

logger = logging.getLogger(__name__)

def cbc_testing(
    session,
    run_configs_id,
    iteration_id,
    Q: dict,
    time_limit_seconds: int = 300,
    cluster_id: int = 0
):
    """
    Solves a QUBO problem using CBC by linearizing it into a MIP formulation.

    Args:
        session: SQLAlchemy session
        run_configs_id: Run config ID
        iteration_id: Iteration ID
        Q: QUBO matrix as a dict {(i, j): value}
        time_limit_seconds: Max solver time (in seconds), default is 300s (limited to 60s)
        cluster_id: Optional clustering info for grouping assignments

    Returns:
        dict: Assignment results as {variable_name: value}
    """
    try:
        start_time_model = time.perf_counter()

        # Extract all variable indices
        all_vars = sorted(set(i for i, _ in Q) | set(j for _, j in Q))

        # Create MIP model
        model = LpProblem("QUBO_Linearized", LpMinimize)
        x = {i: LpVariable(f"x_{i}", cat=LpBinary) for i in all_vars}

        # Linearize quadratic terms: introduce z_ij
        z = {}
        for (i, j), value in Q.items():
            if i != j:
                key = tuple(sorted((i, j)))
                if key not in z:
                    z[key] = LpVariable(f"z_{key[0]}_{key[1]}", cat=LpBinary)
                    # Constraints to linearize x_i * x_j
                    model += z[key] <= x[i]
                    model += z[key] <= x[j]
                    model += z[key] >= x[i] + x[j] - 1

        # Objective function: linear + quadratic terms
        linear_terms = [Q[i, i] * x[i] for (i, j) in Q if i == j]
        quad_terms = [Q[i, j] * z[tuple(sorted((i, j)))] for (i, j) in Q if i != j and tuple(sorted((i, j))) in z]
        model += lpSum(linear_terms + quad_terms)

        # Solve with time limit (max 60s)
        time_limit_seconds = min(time_limit_seconds, 600)
        solver = PULP_CBC_CMD(msg=False, timeLimit=time_limit_seconds)

        start_time_solver = time.perf_counter()
        status = model.solve(solver)
        model_duration = time.perf_counter() - start_time_model
        solver_duration = time.perf_counter() - start_time_solver

        result = {}
        for v in model.variables():
            if v.name.startswith("x_"):
                val = v.varValue
                result[v.name] = int(round(val)) if val is not None else 0

        try:
            objective_value = model.objective.value()
        except Exception:
            objective_value = None

        assignment = list(result.items())

        # Store in database
        cbc_result = CbcResult(
            run_configs_id=run_configs_id,
            iteration_id=iteration_id,
            assignment=assignment,
            objective_value=objective_value,
            duration=model_duration,
            solver_time=solver_duration,
            status=LpStatus[status],
            cluster_id=cluster_id
        )
        session.add(cbc_result)
        session.commit()
        session.close()

        return result, objective_value

    except Exception as e:
        logger.error(f"Error in CBC linearized QUBO: {e}", exc_info=True)
        return {}
