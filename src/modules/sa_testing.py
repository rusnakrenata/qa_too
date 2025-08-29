import time
import datetime
import logging
from typing import Any, Dict, List
from dimod import BinaryQuadraticModel, SimulatedAnnealingSampler
from models import SaResult

logger = logging.getLogger(__name__)


def sa_testing(
    session: Any,
    run_configs_id: int,
    iteration_id: int,
    Q: dict,
    n_vehicles: int,
    route_alternatives: int,
    vehicle_ids: List[int],
    vehicle_routes_df: Any,
    num_reads: int = 1,
    cluster_id: int = None
) -> Dict[str, Any]:
    """
    Solve a QUBO formulation using Simulated Annealing and store results.

    Args:
        session: SQLAlchemy session object.
        run_configs_id: Run configuration ID.
        iteration_id: Iteration number.
        Q: QUBO matrix as dictionary {(q1, q2): value}.
        n_vehicles: Number of vehicles.
        route_alternatives: Number of routes per vehicle.
        vehicle_ids: List of vehicle IDs.
        vehicle_routes_df: DataFrame of valid vehicle routes.
        num_reads: Number of SA reads (default is 1).
        cluster_id: Optional cluster identifier.

    Returns:
        Dictionary containing results: assignment validity, assignment details, energy, and solver duration.
    """
    start_time_model = time.perf_counter()
    logger.info("Starting Simulated Annealing for QUBO solving.")

    # Run Simulated Annealing
    bqm = BinaryQuadraticModel.from_qubo(Q)
    sampler = SimulatedAnnealingSampler()
    
    start_time_solver = time.perf_counter()
    response = sampler.sample(bqm, num_reads=num_reads)
    
    model_duration = time.perf_counter() - start_time_model
    solver_duration = time.perf_counter() - start_time_solver

    # Extract solution
    best_sample, energy = response.first.sample, response.first.energy
    assignment = [int(x) for x in best_sample.values()]

    # Validate assignment
    invalid_assignment_vehicles = []
    for i, vehicle_id in enumerate(vehicle_ids):
        assignment_slice = assignment[i * route_alternatives: (i + 1) * route_alternatives]
        if assignment_slice.count(1) != 1:
            invalid_assignment_vehicles.append(vehicle_id)

    assignment_valid = len(invalid_assignment_vehicles) == 0
    invalid_vehicles_str = ",".join(map(str, invalid_assignment_vehicles))

    # Store results in DB
    result_record = SaResult(
        run_configs_id=run_configs_id,
        iteration_id=iteration_id,
        num_reads=num_reads,
        n_vehicles=n_vehicles,
        k_alternatives=route_alternatives,
        vehicle_ids=vehicle_ids,
        assignment_valid=assignment_valid,
        assignment=assignment,
        energy=energy,
        duration=model_duration,
        solver_time=solver_duration,
        invalid_assignment_vehicles=invalid_vehicles_str,
        cluster_id=cluster_id,
        created_at=datetime.datetime.utcnow()
    )

    session.add(result_record)
    session.commit()
    session.close()

    logger.info(
        f"SA result stored: assignment_valid={assignment_valid}, "
        f"energy={energy}, duration={model_duration:.2f}s, solver_time={solver_duration:.2f}s"
    )

    return {
        'comp_type': 'sa',
        'num_reads': num_reads,
        'n_vehicles': n_vehicles,
        'k_alternatives': route_alternatives,
        'assignment_valid': assignment_valid,
        'assignment': assignment,
        'energy': energy,
        'duration': model_duration,
        'vehicle_ids': vehicle_ids
    }