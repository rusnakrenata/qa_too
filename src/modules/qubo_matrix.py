import numpy as np
import pandas as pd
from pathlib import Path
from models import QuboRunStats
from typing import Dict, Tuple, Any, Optional  # CHANGED (added Optional)
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
    lambda_penalty: Optional[float] = None,   # CHANGED: default None -> auto-compute
    qubo_output_dir: Path = Path("output/qubo_matrices"),
):
    """
    Build an N x N **upper-triangular** matrix Q with 0-based indices:
      - Diagonal term (i,i): -lambda * c_{v,v}
      - Upper-right off-diagonal (i,j) for i<j: c_{v,w}
      - Lower triangle is kept at 0

    If lambda_penalty is None, compute it via the One-shot baseline:
        lambda = median(s_v) / max(1, median(u_v))
      where u_v = c_{v,v} and s_v = sum_w c_{v,w}.
    """
    start_time = time.time()

    # --- NEW: ensure output dir exists
    qubo_output_dir.mkdir(parents=True, exist_ok=True)  # NEW

    variables, index_of = _build_index(diag)
    n = len(variables)

    # --- NEW: One-shot lambda computation (if not provided)
    if (lambda_penalty is None) or (isinstance(lambda_penalty, float) and np.isnan(lambda_penalty)):
        # u_v in variables order
        u = np.array([float(diag[v]) for v in variables], dtype=float)

        # s_v = sum of overlaps per variable
        s_map = defaultdict(float)
        for (v, w), c_vw in offdiag.items():
            s_map[v] += float(c_vw)
            s_map[w] += float(c_vw)
        s = np.array([float(s_map[v]) for v in variables], dtype=float)

        # guard against no positive uniques
        mask = u > 0
        if not np.any(mask):
            lambda_penalty = 1.0
            logger.warning("All u_v (diag) are zero; defaulting lambda_penalty to 1.0")
        else:
            med_u = np.median(u[mask])
            med_s = np.median(s[mask]) if np.any(mask) else np.median(s)
            lambda_penalty = float(med_s) / max(1.0, float(med_u))
        logger.info(f"Computed lambda_penalty (one-shot baseline): λ={lambda_penalty:.6g}")  # NEW
    else:
        logger.info(f"Using provided lambda_penalty: λ={lambda_penalty:.6g}")  # NEW

    logger.info(f"Building QUBO matrix for {n} vehicle-route variables, λ={lambda_penalty:.6g}")

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
    max_idx = max((idx for pair in Q for idx in pair), default=-1)  # NEW (handles empty Q)
    Q_matrix = np.zeros((max_idx + 1, max_idx + 1)) if max_idx >= 0 else np.zeros((n, n))  # NEW robust sizing
    for (i, j), value in Q.items():
        Q_matrix[i, j] = value

    Q_df = pd.DataFrame(Q_matrix)
    Q_df.columns = [f"{i:9g}" for i in range(Q_df.shape[1])]
    Q_df.index = [f"{i:9g}" for i in range(Q_df.shape[0])]

    out_path = qubo_output_dir / f"qubo_matrix_{lambda_penalty:.6g}.csv"  # CHANGED: tidy filename
    Q_df.to_csv(out_path, index=True, header=True, float_format='%9g')

    elapsed = time.time() - start_time
    logger.info(f"QUBO matrix built and saved to {out_path} in {elapsed:.2f}s. , λ={lambda_penalty:.6g}")

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

    return dict(Q), lambda_penalty
