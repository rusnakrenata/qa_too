from typing import Dict, List, Tuple, Union, Any, Optional
import ast
from collections import defaultdict
from itertools import combinations
import logging
import time
import pandas as pd

logger = logging.getLogger(__name__)


def _parse_nodes(val: Any) -> List[Union[int, str]]:
    if val is None:
        return []
    if isinstance(val, (list, tuple, set)):
        return list(val)
    if isinstance(val, str):
        s = val.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")) or (s.startswith("{") and s.endswith("}")):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple, set)):
                    return list(parsed)
            except Exception:
                pass
        if "," in s:
            return [p.strip() for p in s.split(",") if p.strip() != ""]
        return [s]
    return [str(val)]


def compute_qubo_overlap_weights(
    df: pd.DataFrame,
    node_column: str = "path_node_ids",
    vehicle_col: str = "vehicle_id",
    route_col: str = "route_id",
    deduplicate_nodes: bool = True,
    assignment: Optional[List[int]] = None,
) -> Tuple[
    Dict[Tuple[Any, Any], int],
    Dict[Tuple[Tuple[Any, Any], Tuple[Any, Any]], int],
    int,  # max_weight_cvw
    int,  # max_weight_cvv
    int,  # n_intersection_nodes_distinct
    int,  # overall_overlap (sum of all c_vw)
]:
    """
    Compute diagonal (c_vv) and off-diagonal (c_vw) weights from route nodes.

    Filtering:
        If 'assignment' is provided, it must be a 0/1 list whose length equals the number
        of variables sorted by (vehicle_id, route_id). Only variables with 1 are kept.

    Returns:
        diag: {(veh,route): c_vv}
        offdiag: {((veh,route),(veh,route)): c_vw} for v<w
        max_weight_cvw: max off-diagonal overlap
        max_weight_cvv: max diagonal node count
        n_intersection_nodes_distinct: number of distinct node IDs shared by >=2 selected routes
        overall_overlap: sum of all c_vw over v<w
    """
    start_time = time.time()

    required = {vehicle_col, route_col, node_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    # Build full variable list in a stable order
    df_vars = df[[vehicle_col, route_col]].drop_duplicates()
    variables_sorted = sorted([tuple(x) for x in df_vars.to_records(index=False)])

    # If an assignment mask is provided, filter df down to chosen variables
    if assignment is not None:
        if len(assignment) != len(variables_sorted):
            raise ValueError(
                f"Assignment length {len(assignment)} does not match number of variables {len(variables_sorted)}."
            )
        keep_set = {var for var, bit in zip(variables_sorted, assignment) if int(bit) == 1}
        if not keep_set:
            # Nothing selected; return zeros
            logger.info("Assignment mask selected zero routes; returning empty weights.")
            return {}, {}, 0, 0, 0, 0
        mask = df.apply(lambda r: (r[vehicle_col], r[route_col]) in keep_set, axis=1)
        df = df.loc[mask].reset_index(drop=True)

    # Map each variable to its node list
    var_nodes: Dict[Tuple[Any, Any], List[Union[int, str]]] = {}
    for _, row in df.iterrows():
        var = (row[vehicle_col], row[route_col])
        nodes = _parse_nodes(row[node_column])
        if deduplicate_nodes:
            nodes = list(dict.fromkeys(nodes))
        var_nodes[var] = nodes

    # Diagonal weights: number of nodes per route
    diag: Dict[Tuple[Any, Any], int] = {var: len(nodes) for var, nodes in var_nodes.items()}

    # Inverted index: node -> variables that contain it
    index: Dict[Any, List[Tuple[Any, Any]]] = defaultdict(list)
    for var, nodes in var_nodes.items():
        for n in nodes:
            index[n].append(var)

    # Distinct intersection nodes = nodes shared by >= 2 routes
    n_intersection_nodes_distinct = sum(1 for vars_with_node in index.values() if len(vars_with_node) >= 2)

    all_nodes = set()
    for nodes in var_nodes.values():
        all_nodes.update(nodes)
    n_nodes_distinct = len(all_nodes)

    # Pairwise overlaps
    from collections import defaultdict as dd
    offdiag = dd(int)
    for vars_with_node in index.values():
        if len(vars_with_node) < 2:
            continue
        for v, w in combinations(sorted(vars_with_node), 2):
            offdiag[(v, w)] += 1

    max_weight_cvw = max(offdiag.values()) if offdiag else 0
    max_weight_cvv = max(diag.values()) if diag else 0
    overall_overlap = int(sum(offdiag.values())) if offdiag else 0

    elapsed = time.time() - start_time
    logger.info(
        "Computed QUBO weights in %.2fs. Routes=%d, pairs=%d, max c_vw=%d, max c_vv=%d, distinct intersection nodes=%d, overall overlap=%d",
        elapsed,
        len(var_nodes),
        len(offdiag),
        max_weight_cvw,
        max_weight_cvv,
        n_nodes_distinct,
        overall_overlap,
    )

    return diag, dict(offdiag), max_weight_cvw, max_weight_cvv, n_nodes_distinct, overall_overlap
