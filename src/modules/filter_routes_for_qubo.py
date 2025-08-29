import pandas as pd
from igraph import Graph
import leidenalg
import logging
from collections import defaultdict


logger = logging.getLogger(__name__)


def get_clusters_by_connectivity(
    congestion_df: pd.DataFrame,
    resolution: float = 4.0,
    min_cluster_size: int = 100,
) -> list:
    """
    Get clusters ordered by connectivity (total congestion), starting with the most connected.
    Ensures clusters are at least min_cluster_size by merging small ones.

    Args:
        congestion_df: DataFrame with columns ['edge_id', 'vehicle1', 'vehicle2', 'congestion_score']
        resolution: Resolution parameter for Leiden algorithm (default 0.7)
        min_cluster_size: Minimum number of vehicles in a cluster

    Returns:
        List of tuples: [(vehicle_ids_list, affected_edges_df, total_congestion, cluster_size), ...]
    """

    # Prepare nodes and edges
    edges = list(zip(congestion_df['vehicle1'], congestion_df['vehicle2']))
    weights = list(congestion_df['congestion_score'])
    nodes = set(congestion_df['vehicle1']).union(congestion_df['vehicle2'])
    node_to_idx = {n: i for i, n in enumerate(sorted(nodes))}
    idx_to_node = {i: n for n, i in node_to_idx.items()}

    g = Graph()
    g.add_vertices(len(nodes))
    g.add_edges([(node_to_idx[u], node_to_idx[v]) for u, v in edges])
    g.es['weight'] = weights



    # Leiden clustering
    part = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights='weight',
        resolution_parameter = resolution
    )

    initial_clusters = [list(community) for community in part]

    # Merge small clusters
    merged_clusters = merge_small_clusters(g, initial_clusters, min_cluster_size) #initial_clusters

    # Process clusters
    clusters_info = []
    for community in merged_clusters:
        cluster_size = len(community)
        subgraph = g.subgraph(community)
        total_congestion = sum(subgraph.es['weight']) if subgraph.ecount() > 0 else 0.0
        vehicle_ids = [idx_to_node[i] for i in community]

        affected_edges_df = congestion_df[
            congestion_df['vehicle1'].isin(vehicle_ids) & congestion_df['vehicle2'].isin(vehicle_ids)
        ][['edge_id', 'congestion_score']]

        if isinstance(affected_edges_df, pd.DataFrame) and not affected_edges_df.empty:
            affected_edges_df = affected_edges_df.groupby('edge_id', as_index=False)['congestion_score'].sum()

        clusters_info.append((vehicle_ids, affected_edges_df, total_congestion, cluster_size))

    # Sort by total congestion
    clusters_info.sort(reverse=True, key=lambda x: x[2])

    logger.info(
        f"Found {len(clusters_info)} clusters with at least {min_cluster_size} vehicles"
        + f", most connected has {clusters_info[0][3] if clusters_info else 0} vehicles"
    )

    return clusters_info


def merge_small_clusters(g: Graph, clusters: list[list[int]], min_cluster_size: int) -> list[list[int]]:
    """
    Ensures that all clusters are at least min_cluster_size.
    1. Tries to merge small clusters into larger neighboring clusters.
    2. Remaining small clusters are grouped together until they reach min_cluster_size.

    Args:
        g: igraph.Graph object.
        clusters: List of clusters (each cluster is a list of vertex indices).
        min_cluster_size: Minimum size a cluster should have.

    Returns:
        List of merged clusters (list of vertex index lists).
    """

    node_to_cluster = {}
    cluster_members = {}
    for cid, cluster in enumerate(clusters):
        cluster_members[cid] = set(cluster)
        for node in cluster:
            node_to_cluster[node] = cid

    cluster_sizes = {cid: len(members) for cid, members in cluster_members.items()}
    active_clusters = set(cluster_members.keys())
    merged_clusters = {}
    merged_ids = set()

    # Inter-cluster edge weights
    inter_weights = defaultdict(lambda: defaultdict(float))
    for edge in g.es:
        u, v = edge.tuple
        cu = node_to_cluster[u]
        cv = node_to_cluster[v]
        if cu != cv:
            w = edge["weight"]
            inter_weights[cu][cv] += w
            inter_weights[cv][cu] += w

    # Step 1: Merge small clusters into larger neighbors
    for scid in list(active_clusters):
        if scid in merged_ids or cluster_sizes[scid] >= min_cluster_size:
            continue

        neighbors = inter_weights[scid]
        sorted_neighbors = sorted(neighbors.items(), key=lambda x: -x[1])

        for neighbor_cid, _ in sorted_neighbors:
            if neighbor_cid in merged_ids or neighbor_cid == scid:
                continue
            if cluster_sizes[neighbor_cid] >= min_cluster_size:
                new_id = max(cluster_members.keys()) + 1
                new_members = cluster_members[scid] | cluster_members[neighbor_cid]
                cluster_members[new_id] = new_members
                cluster_sizes[new_id] = len(new_members)

                # Update mappings
                for n in new_members:
                    node_to_cluster[n] = new_id

                # Remove old clusters
                merged_ids.update({scid, neighbor_cid})
                active_clusters.discard(scid)
                active_clusters.discard(neighbor_cid)
                active_clusters.add(new_id)
                break

    # Step 2: Collect remaining small clusters
    remaining_small = [
        cid for cid in active_clusters
        if cluster_sizes[cid] < min_cluster_size and cid not in merged_ids
    ]

    # Group remaining smalls together into batches
    batched_clusters = []
    current_batch = set()
    current_size = 0

    for cid in remaining_small:
        current_batch.update(cluster_members[cid])
        current_size += cluster_sizes[cid]

        if current_size >= min_cluster_size:
            batched_clusters.append(list(current_batch))
            current_batch = set()
            current_size = 0

    if current_batch:
        batched_clusters.append(list(current_batch))  # Add leftovers even if below threshold

    # Step 3: Return all valid clusters
    final_clusters = [
        list(cluster_members[cid])
        for cid in active_clusters
        if cid not in merged_ids and cluster_sizes[cid] >= min_cluster_size
    ] + batched_clusters

    return final_clusters




