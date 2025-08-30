import time
import datetime
from matplotlib.pylab import sample
import numpy as np
import pandas as pd
import os
from pathlib import Path
from collections import defaultdict
from models import QAResult
import logging
from typing import Any, Dict
from compute_weights import compute_qubo_overlap_weights


logger = logging.getLogger(__name__)

def get_api_token() -> str:
    """
    Retrieve the API token securely from environment variable or fallback.
    """
    return os.environ.get('QA_API_TOKEN', 'notoken')


def authenticate_with_token(token: str) -> bool:
    from dwave.system import DWaveSampler
    try:
        sampler = DWaveSampler(token=token)
        logger.info(f"Connected to D-Wave. Solver: {sampler.solver.name}")
        return True
    except Exception as e:
        logger.error(f"Failed to authenticate with D-Wave: {e}")
        return False



def qa_testing(    
    session: Any,
    run_configs_id: int,
    iteration_id: int,
    Q: dict,
    vehicle_routes_df: pd.DataFrame,
    lambda_value=None,
    comp_type: str = 'sa',
    num_reads: int = 1,
) -> Dict[str, Any]:
    """
    Run QUBO formulation for the car-to-trase assignment using a specified quantum/classical sampler.

    Returns:
        dict: Results including assignment, energy, overlap diagnostics, selected vehicle info, etc.
    """
    api_token = get_api_token()
    start_time = time.perf_counter()
    logger.info("Starting QA testing with comp_type: %s", comp_type)

    # === Sampler selection
    from dimod import BinaryQuadraticModel
    if comp_type == 'sa':
        from dimod import SimulatedAnnealingSampler
        bqm = BinaryQuadraticModel.from_qubo(Q)
        sampler = SimulatedAnnealingSampler()
        response = sampler.sample(bqm, num_reads=num_reads)
        total_annealing_time_s = time.perf_counter() - start_time

    elif comp_type == 'hybrid':
        from dwave.system import LeapHybridSampler
        bqm = BinaryQuadraticModel.from_qubo(Q)
        sampler = LeapHybridSampler(connection_close=True, token=api_token)
        response = sampler.sample(bqm, label="Traffic Optimization hybrid BQM")
        total_annealing_time_s = response.info.get('run_time', 0) / 1_000_000

    elif comp_type == 'qpu':
        from dwave.system import DWaveSampler, EmbeddingComposite
        bqm = BinaryQuadraticModel.from_qubo(Q)
        sampler = EmbeddingComposite(DWaveSampler(token=api_token))
        response = sampler.sample(bqm, num_reads=num_reads, label="Traffic Optimization QPU")
        annealing_time_us = response.info.get("timing", {}).get("qpu_access_time", 0)
        total_annealing_time_s = (annealing_time_us * num_reads) / 1_000_000

    else:
        logger.error(f"Unknown comp_type: {comp_type}")
        raise ValueError(f"Unknown comp_type: {comp_type}")

    model_duration = time.perf_counter() - start_time

    # === Process Results
    record = response.first
    best_sample, energy = record[:2]
    assignment = [int(x) for x in best_sample.values()]
    logger.info(f"Assignment: {assignment}")

    # === Compute overlap diagnostics
    diag, offdiag, max_weight_cvw, max_weight_cvv, n_nodes_distinct, overall_overlap = compute_qubo_overlap_weights(
        vehicle_routes_df, node_column="path_node_ids", assignment=assignment
    )

    # === Selected vehicles
    n_vehicles_selected = sum(assignment)
    selected_vehicle_ids = vehicle_routes_df.loc[
        np.array(assignment).astype(bool), "vehicle_id"
    ].tolist()

    # === Store in DB
    result_record = QAResult(
        run_configs_id=run_configs_id,
        iteration_id=iteration_id,
        lambda_value=lambda_value,
        comp_type=comp_type,
        num_reads=num_reads,
        assignment=assignment,
        n_nodes_distinct=n_nodes_distinct,
        overall_overlap=overall_overlap,
        energy=energy,
        duration=model_duration,
        solver_time=total_annealing_time_s,
        n_vehicles_selected=n_vehicles_selected,
        selected_vehicle_ids=selected_vehicle_ids,
        created_at=datetime.datetime.utcnow()
    )
    session.add(result_record)
    session.commit()

    logger.info(f"QA testing complete: energy={energy:.3f}, duration={model_duration:.2f}s, vehicles_selected={n_vehicles_selected}")

    return {
        'comp_type': comp_type,
        'num_reads': num_reads,
        'assignment': assignment,
        'energy': energy,
        'duration': model_duration,
        'solver_time': total_annealing_time_s,
        'lambda_value': lambda_value,
        'n_vehicles_selected': n_vehicles_selected,
        'selected_vehicle_ids': selected_vehicle_ids
    }
