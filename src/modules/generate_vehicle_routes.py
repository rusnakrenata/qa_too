import time
import asyncio
import aiohttp
import nest_asyncio
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString
from shapely.strtree import STRtree
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform
from pyproj import Transformer
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import logging
from typing import Any, List, Tuple, Dict
import multiprocessing

from utils import (
    # do NOT use utils.create_linestring_from_polyline (it can crash on 1-point inputs)
    get_point_on_line,
    convert_valhalla_leg_to_google_like_steps,
    async_get_routes_from_valhalla,
    find_closest_osm_edge,
)

logger = logging.getLogger(__name__)

# =========================
# Safe polyline utilities
# =========================

def _normalize_polyline_points(raw: Any) -> List[Tuple[float, float]]:
    """
    Normalize to list[(lon, lat)].
    Accepts:
      - [(lon, lat)] or [(lat, lon)] -> auto-detect & swap if needed
      - [{'lon': x, 'lat': y}] or [{'lat': y, 'lon': x}]
      - [[lon, lat]] / [[lat, lon]]
    """
    pts: List[Tuple[float, float]] = []
    if raw is None:
        return pts

    # single numeric pair
    if isinstance(raw, (tuple, list)) and len(raw) == 2 and all(isinstance(v, (int, float)) for v in raw):
        raw = [raw]

    if isinstance(raw, (list, tuple)):
        for p in raw:
            if p is None:
                continue
            if isinstance(p, dict):
                lon = p.get("lon", p.get("x"))
                lat = p.get("lat", p.get("y"))
                if lon is None or lat is None:
                    continue
                try:
                    pts.append((float(lon), float(lat)))
                except Exception:
                    continue
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                try:
                    a = float(p[0]); b = float(p[1])
                    pts.append((a, b))
                except Exception:
                    continue

    # Heuristic: if first coord looks like lat (<=90) and second often >90 (lon), swap
    if pts:
        xs = [abs(x) for x, _ in pts]
        ys = [abs(y) for _, y in pts]
        lat_like_first = sum(1 for x in xs if x <= 90.0)
        lon_like_second = sum(1 for y in ys if y > 90.0)  # many longitudes > 90 in EU/UK
        if lat_like_first >= 0.6 * len(pts) and lon_like_second >= 0.2 * len(pts):
            pts = [(y, x) for (x, y) in pts]  # swap to (lon, lat)

    # drop consecutive duplicates
    cleaned: List[Tuple[float, float]] = []
    last = None
    for x, y in pts:
        if last is None or (x != last[0] or y != last[1]):
            cleaned.append((x, y))
            last = (x, y)

    return cleaned

def _safe_linestring_from_polyline(raw: Any) -> LineString:
    """
    Returns a LineString; returns EMPTY LineString if < 2 coords.
    Never raises for 0 or 1-point inputs.
    """
    pts = _normalize_polyline_points(raw)
    if len(pts) < 2:
        return LineString([])
    return LineString(pts)

# =========================
# Helpers
# =========================

def _dedupe_consecutive(seq: List[Any]) -> List[Any]:
    out = []
    prev = object()
    for x in seq:
        if x != prev:
            out.append(x)
            prev = x
    return out

def get_points_in_time_window(steps, time_step, time_window):
    """
    Samples points along the Valhalla polyline by time; SAFE against degenerate step polylines.
    """
    cumulative_times = []
    time_acc = 0
    for step in steps:
        duration = step['duration']['value']
        cumulative_times.append((time_acc, time_acc + duration))
        time_acc += duration

    points = []
    if not cumulative_times:
        return points

    step_idx = 0
    step_start, step_end = cumulative_times[step_idx]

    for step_time in range(time_step, time_window, time_step):
        while step_time >= step_end and step_idx + 1 < len(cumulative_times):
            step_idx += 1
            step_start, step_end = cumulative_times[step_idx]

        step = steps[step_idx]
        duration = step['duration']['value']
        if duration == 0:
            continue

        ls = _safe_linestring_from_polyline(step['polyline']['points'])
        if ls.is_empty:
            continue

        fraction = (step_time - step_start) / duration
        point_on_line = get_point_on_line(ls, fraction)

        points.append({
            'location': point_on_line,
            'time': step_time,
            'speed': step['distance']['value'] / max(duration, 1e-9)
        })

    return points

def _ordered_nodes_from_edges(edge_seq: List[int], edges_lookup: pd.DataFrame) -> List[str]:
    """
    Given ordered edge_id sequence and lookup ['edge_id','u','v'],
    reconstruct ordered OSM node ids; includes continuity repair.
    """
    if not edge_seq:
        return []
    uv = edges_lookup.set_index("edge_id")[["u","v"]].to_dict("index")

    path_osm_nodes: List[str] = []
    first_u = uv[edge_seq[0]]["u"]
    first_v = uv[edge_seq[0]]["v"]
    path_osm_nodes.extend([first_u, first_v])

    for prev_e, cur_e in zip(edge_seq[:-1], edge_seq[1:]):
        cu, cv = uv[cur_e]["u"], uv[cur_e]["v"]
        last_node = path_osm_nodes[-1]
        if cu == last_node:
            path_osm_nodes.append(cv)
        elif cv == last_node:
            path_osm_nodes.append(cu)
        else:
            path_osm_nodes.extend([cu, cv])

    return _dedupe_consecutive(path_osm_nodes)

def _build_route_line_from_steps(steps) -> LineString:
    """
    Concatenate Valhalla step polylines into one LineString (EPSG:4326), safely.
    """
    coords: List[Tuple[float, float]] = []
    first = True
    for st in steps:
        g = _safe_linestring_from_polyline(st["polyline"]["points"])
        if g.is_empty or len(g.coords) < 2:
            continue
        gcoords = list(g.coords)
        if first:
            coords.extend(gcoords)
            first = False
        else:
            coords.extend(gcoords[1:] if coords and gcoords and coords[-1] == gcoords[0] else gcoords)
    return LineString(coords) if len(coords) >= 2 else LineString([])

def _edges_from_node_sequence(
    osm_node_seq: List[str],
    edges_gdf: gpd.GeoDataFrame,        # needs ['edge_id','u','v','geometry'] in EPSG:4326
    path_geom_wgs84: LineString,        # built from steps (EPSG:4326)
    edges_proj: gpd.GeoDataFrame        # same edges in metric CRS (3857), same row order as edges_gdf
) -> List[int]:
    """
    For each consecutive node pair, pick the connecting edge.
    If multiple candidates exist (parallel edges), choose the one with
    greatest overlap (in meters) with the route polyline.
    """
    if not osm_node_seq or len(osm_node_seq) < 2:
        return []

    transformer_to_proj = Transformer.from_crs("EPSG:4326", edges_proj.crs, always_xy=True)
    path_proj = transform(lambda x, y, z=None: transformer_to_proj.transform(x, y), path_geom_wgs84)

    edges = edges_gdf[['edge_id','u','v','geometry']].reset_index(drop=True)
    edges_proj = edges_proj.reset_index(drop=True)

    final_edge_ids: List[int] = []
    for a, b in zip(osm_node_seq[:-1], osm_node_seq[1:]):
        cand = edges[((edges['u'] == a) & (edges['v'] == b)) | ((edges['u'] == b) & (edges['v'] == a))]
        if cand.empty:
            continue
        if len(cand) == 1 or path_proj.is_empty:
            final_edge_ids.append(int(cand.iloc[0]['edge_id']))
            continue

        best_eid = None
        best_len = -1.0
        for idx in cand.index:
            g_proj = edges_proj.loc[idx, 'geometry']
            inter = g_proj.intersection(path_proj)
            score = inter.length if not inter.is_empty else 0.0
            if score > best_len:
                best_len = score
                best_eid = int(edges.loc[idx, 'edge_id'])
        final_edge_ids.append(best_eid if best_eid is not None else int(cand.iloc[0]['edge_id']))

    return _dedupe_consecutive(final_edge_ids)

# =========================
# Worker
# =========================

def _process_one_vehicle(
    vehicle_row: pd.Series,
    route_list: List[Dict],
    edges_proj_dict: Dict[str, Any],
    edges_gdf: gpd.GeoDataFrame,
    nodes_gdf: gpd.GeoDataFrame,
    time_step: int,
    time_window: int
) -> Dict[str, Any] | None:

    if not route_list:
        return None

    edges_proj = edges_proj_dict['edges']
    edge_geometries = [geom if isinstance(geom, BaseGeometry) else geom.__geo_interface__
                       for geom in edges_proj['geometry'].values.tolist()]
    edge_tree = STRtree(edge_geometries)
    transformer = Transformer.from_crs("EPSG:4326", edges_proj_dict['crs'], always_xy=True)

    route = route_list[0]
    summary = route.get('summary', {})
    duration_sec = int(summary.get('time', 0))
    distance_m = float(summary.get('length', 0) * 1000)

    steps = convert_valhalla_leg_to_google_like_steps(route['leg'])

    # A) time-sampled points (safe) -> sampled edges
    points = get_points_in_time_window(steps, time_step, time_window)
    if not points:
        return None

    sampled_edge_seq: List[int] = []
    for point in points:
        loc = point['location']
        lat, lon = float(loc.y), float(loc.x)
        edge_info = find_closest_osm_edge(lon, lat, edges_proj, edge_tree, transformer=transformer)
        sampled_edge_seq.append(int(edge_info['id']))
    sampled_edge_seq = _dedupe_consecutive(sampled_edge_seq)

    # B) nodes from sampled edges (with continuity repair)
    need_cols = {'edge_id', 'u', 'v'}
    if not need_cols.issubset(edges_gdf.columns):
        logger.error("edges_gdf must include columns ['edge_id','u','v'] to compute node path.")
        osm_node_seq: List[str] = []
    else:
        edges_lookup = edges_gdf[['edge_id','u','v']].copy()
        osm_node_seq = _ordered_nodes_from_edges(sampled_edge_seq, edges_lookup)

    # C) build a continuous route line (safe) for overlap disambiguation
    path_geom = _build_route_line_from_steps(steps)

    # D) final edges reconstructed from nodes (no skipped connectors)
    edge_seq = _edges_from_node_sequence(
        osm_node_seq=osm_node_seq,
        edges_gdf=edges_gdf,
        path_geom_wgs84=path_geom,
        edges_proj=edges_proj
    )

    # E) map OSM node ids -> DB node_id
    path_node_ids: List[int] = []
    if {'node_id','osmid'}.issubset(nodes_gdf.columns):
        osmid_to_nodeid = nodes_gdf[['osmid','node_id']].drop_duplicates().set_index('osmid')['node_id']
        for osm_id in osm_node_seq:
            if osm_id in osmid_to_nodeid.index:
                path_node_ids.append(int(osmid_to_nodeid.loc[osm_id]))
    else:
        logger.error("nodes_gdf must include ['node_id','osmid'] to map path nodes to DB IDs.")

    return {
        "vehicle_id": int(vehicle_row['vehicle_id']),
        "route_id": 1,
        "duration": duration_sec,
        "distance": distance_m,
        "path_edge_ids": edge_seq,
        "path_node_ids": path_node_ids
    }

# top-level star helper (picklable)
def _process_one_vehicle_star(args):
    return _process_one_vehicle(*args)

# =========================
# Main
# =========================

def generate_vehicle_routes(
    session: Any,
    run_config_id: int,
    iteration_id: int,
    route_class: Any,                 # models.VehicleRoute with JSON columns path_edge_ids, path_node_ids
    vehicles_df: pd.DataFrame,        # must have: vehicle_id, origin_lat, origin_lon, destination_lat, destination_lon
    edges_gdf: gpd.GeoDataFrame,      # requires: edge_id, u, v, geometry (EPSG:4326)
    nodes_gdf: gpd.GeoDataFrame,      # requires: node_id, osmid, geometry (EPSG:4326)
    time_step: int,
    time_window: int,
    max_concurrent: int = 20,
    use_threads_on_windows: bool = True   # default True to avoid ProcessPool pickling issues
) -> pd.DataFrame:
    """
    - EXACTLY one route per vehicle
    - NO route_points table
    - Stores path_edge_ids and path_node_ids (DB IDs) per vehicle
    """
    try:
        # validate
        vcols = {"vehicle_id","origin_lat","origin_lon","destination_lat","destination_lon"}
        if vehicles_df is None or vehicles_df.empty or not vcols.issubset(vehicles_df.columns):
            logger.error(f"vehicles_df must include {vcols}.")
            return pd.DataFrame()

        ecols = {"edge_id","u","v","geometry"}
        if edges_gdf is None or edges_gdf.empty or not ecols.issubset(edges_gdf.columns):
            logger.error("edges_gdf must include ['edge_id','u','v','geometry'] in EPSG:4326.")
            return pd.DataFrame()

        ncols = {"node_id","osmid","geometry"}
        if nodes_gdf is None or nodes_gdf.empty or not ncols.issubset(nodes_gdf.columns):
            logger.error("nodes_gdf must include ['node_id','osmid','geometry'] in EPSG:4326.")
            return pd.DataFrame()

        # fetch routes
        nest_asyncio.apply()

        async def batched_valhalla_fetch(vdf, max_concurrent=max_concurrent):
            sem = asyncio.Semaphore(max_concurrent)
            async with aiohttp.ClientSession() as http_session:
                async def fetch(row):
                    async with sem:
                        origin = (float(row["origin_lon"]), float(row["origin_lat"]))  # (lon, lat)
                        dest   = (float(row["destination_lon"]), float(row["destination_lat"]))
                        return await async_get_routes_from_valhalla(http_session, origin, dest, 1)
                tasks = [fetch(row) for _, row in vdf.iterrows()]
                return await asyncio.gather(*tasks)

        loop = asyncio.get_event_loop()
        logger.info("Fetching routes from Valhalla (1 per vehicle)...")
        t0 = time.time()
        routes_data_list = loop.run_until_complete(batched_valhalla_fetch(vehicles_df))
        logger.info(f"Valhalla fetch completed in {time.time() - t0:.2f} seconds")

        # projected edges for snapping/overlap
        edges_proj = edges_gdf.to_crs(epsg=3857)
        edges_proj_dict = {'edges': edges_proj, 'crs': edges_proj.crs}

        vehicle_data_list = [
            (row, routes_data_list[i], edges_proj_dict, edges_gdf, nodes_gdf, time_step, time_window)
            for i, (_, row) in enumerate(vehicles_df.iterrows())
            if routes_data_list[i]
        ]

        logger.info("Processing vehicle routes (edge/node paths)...")
        t1 = time.time()

        if use_threads_on_windows:
            with ThreadPoolExecutor(max_workers=min(16, multiprocessing.cpu_count())) as executor:
                results = list(executor.map(_process_one_vehicle_star, vehicle_data_list))
        else:
            with ProcessPoolExecutor(max_workers=min(16, multiprocessing.cpu_count())) as executor:
                results = list(executor.map(_process_one_vehicle_star, vehicle_data_list))

        logger.info(f"Processing completed in {time.time() - t1:.2f} seconds")

        rows = [r for r in results if r is not None]
        if not rows:
            logger.warning("No routes processed successfully.")
            return pd.DataFrame()

        route_objs = [
            route_class(
                vehicle_id=row['vehicle_id'],
                run_configs_id=run_config_id,
                iteration_id=iteration_id,
                route_id=row['route_id'],
                duration=row['duration'],
                distance=row['distance'],
                duration_in_traffic=None,
                path_edge_ids=row['path_edge_ids'],
                path_node_ids=row['path_node_ids']
            ) for row in rows
        ]

        session.bulk_save_objects(route_objs)
        session.commit()
        logger.info(f"Stored {len(route_objs)} vehicle routes with edge/node paths.")

        return pd.DataFrame(rows, columns=['vehicle_id','route_id','duration','distance','path_edge_ids','path_node_ids'])

    except Exception as e:
        logger.error(f"Error in generate_vehicle_routes: {e}", exc_info=True)
        session.rollback()
        return pd.DataFrame()

    finally:
        session.close()
