import numpy as np
import pandas as pd
from pathlib import Path
from models import QuboRunStats
from typing import Dict, Tuple, Any
from collections import defaultdict
import logging
import time

logger = logging.getLogger(__name__)


def _build_index(diag: Dict[Tuple[Any, Any], int]):
    """Return variables list (sorted) and mapping {(veh, route) -> 0-based index}."""
    variables = sorted(diag.keys())
    index_of = {var: i for i, var in enumerate(variables)}
    return variables, index_of


def qubo_matrix(
    session,
    run_configs_id: int,
    iteration_id: int,
    diag: Dict[Tuple[Any, Any], int],
    offdiag: Dict[Tuple[Tuple[Any, Any], Tuple[Any, Any]], int],
    max_weight_cvv: float,
    max_weight_cvw: float,
    n_nodes_distinct: int,
    overall_overlap: int,
    lambda_penalty: float,
    qubo_output_dir: Path = Path("output/qubo_matrices"),
):
    """
    Build an N x N **upper-triangular** matrix Q with 0-based indices:
      - Diagonal term (i,i): -lambda_penalty * c_{v,v}
      - Upper-right off-diagonal (i,j) for i<j: c_{v,w}
      - Lower triangle is kept at 0
    Inputs:
      diag[(veh,route)] = c_{v,v}
      offdiag[((veh,route),(veh,route))] = c_{v,w}  (unordered pairs; function will place into i<j)
    Returns:
      Q (np.ndarray), variables (ordered list of (veh,route)), index_of (mapping)
    """
    start_time = time.time()
    variables, index_of = _build_index(diag)
    n = len(variables)
    logger.info(f"Building QUBO matrix for {n} vehicle-route variables, λ={lambda_penalty}")

    Q = defaultdict(float)

    # Diagonal terms
    for v, c_vv in diag.items():
        i = index_of[v]
        Q[(i, i)] = -lambda_penalty * float(c_vv)

    # Off-diagonal terms (upper triangle only)
    for (v, w), c_vw in offdiag.items():
        i, j = index_of[v], index_of[w]
        if i == j:
            continue
        if i < j:
            Q[(i, j)] = float(c_vw)
        else:
            Q[(j, i)] = float(c_vw)

    # Save full matrix with headers/indices
    Q_matrix = np.zeros((max(idx for pair in Q for idx in pair) + 1,) * 2)
    for (i, j), value in Q.items():
        Q_matrix[i, j] = value

    Q_df = pd.DataFrame(Q_matrix)
    Q_df.columns = [f"{i:9g}" for i in range(Q_df.shape[1])]
    Q_df.index = [f"{i:9g}" for i in range(Q_df.shape[0])]
    Q_df.to_csv(qubo_output_dir / f"qubo_matrix_{lambda_penalty}.csv", index=True, header=True, float_format='%9g')

    elapsed = time.time() - start_time
    logger.info(f"QUBO matrix built and saved to {qubo_output_dir / f'qubo_matrix_{lambda_penalty}.csv'} in {elapsed:.2f}s. , λ={lambda_penalty}")

    # Save run stats
    stats = QuboRunStats(
        run_configs_id=run_configs_id,
        iteration_id=iteration_id,
        n_vehicles=n,
        lambda_penalty=lambda_penalty,
        max_weight_cvv=max_weight_cvv,
        max_weight_cvw=max_weight_cvw,
        n_nodes_distinct=n_nodes_distinct,
        overall_overlap=overall_overlap
    )
    session.add(stats)
    session.commit()

    return dict(Q), variables, index_of


