import time
import asyncio
import aiohttp
import nest_asyncio
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from shapely.strtree import STRtree
from pyproj import Transformer
from shapely.geometry.base import BaseGeometry
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime
from sqlalchemy import insert, text
import logging
from typing import Any
import multiprocessing

from utils import (
    create_linestring_from_polyline,
    get_point_on_line,
    convert_valhalla_leg_to_google_like_steps,
    async_get_routes_from_valhalla,
    find_closest_osm_edge,
    calculate_initial_bearing,
    bearing_to_cardinal
)

logger = logging.getLogger(__name__)

def insert_in_batches(session, model_class, obj_list, batch_size=5000):
    # Use all actual DB columns (no 'id' filter since you confirmed it's not used)
    table_columns = [col.name for col in model_class.__table__.columns]

    def to_row(obj):
        return {col: getattr(obj, col) for col in table_columns}

    rows = [to_row(obj) for obj in obj_list]
    if not rows:
        return

    stmt = insert(model_class.__table__)
    
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        session.execute(stmt, batch)

def get_points_in_time_window(steps, time_step, time_window):
    cumulative_times = []
    time_acc = 0
    for step in steps:
        duration = step['duration']['value']
        cumulative_times.append((time_acc, time_acc + duration))
        time_acc += duration

    points = []
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

        polyline_points = step['polyline']['points']
        gdf_line = create_linestring_from_polyline(polyline_points)
        fraction = (step_time - step_start) / duration
        point_on_line = get_point_on_line(gdf_line.geometry.iloc[0], fraction)

        points.append({
            'location': point_on_line,
            'time': step_time,
            'speed': step['distance']['value'] / duration
        })

    return points

def process_vehicle_route(vehicle_data):
    vehicle, vehicle_idx, route_data, edges_proj_dict, time_step, time_window = vehicle_data
    route_points_records = []
    vehicle_routes_records = []

    edges_proj = edges_proj_dict['edges']
    edge_geometries = [geom if isinstance(geom, BaseGeometry) else geom.__geo_interface__ for geom in edges_proj['geometry'].values.tolist()]
    edge_tree = STRtree(edge_geometries)
    transformer = Transformer.from_crs("EPSG:4326", edges_proj_dict['crs'], always_xy=True)

    origin = (vehicle['origin_geometry'].x, vehicle['origin_geometry'].y)

    for route_id, route in enumerate(route_data, start=1):
        summary = route['summary']

        vehicle_routes_records.append({
            "vehicle_id": vehicle['vehicle_id'],
            "route_id": route_id,
            "duration": summary['time'],
            "distance": summary['length'] * 1000
        })

        steps = convert_valhalla_leg_to_google_like_steps(route['leg'])
        points = get_points_in_time_window(steps, time_step, time_window)

        for point_id, point in enumerate(points, start=1):
            lat = point['location'].y
            lon = point['location'].x
            time_val = point['time']
            speed = point['speed']
            previous_location = points[point_id - 2]['location'] if point_id > 1 else Point(origin[0], origin[1])

            edge = find_closest_osm_edge(lon, lat, edges_proj, edge_tree, transformer=transformer)
            bearing = calculate_initial_bearing(previous_location.y, previous_location.x, lat, lon)
            cardinal = bearing_to_cardinal(bearing)

            route_points_records.append({
                "vehicle_id": vehicle['vehicle_id'],
                "route_id": route_id,
                "point_id": point_id,
                "edge_id": edge['id'],
                "cardinal": cardinal,
                "speed": speed,
                "lat": lat,
                "lon": lon,
                "time": time_val
            })

    return vehicle_routes_records, route_points_records

def generate_vehicle_routes(
    session: Any,
    run_config_id: int,
    iteration_id: int,
    route_class: Any,
    route_point_class: Any,
    vehicles_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
    k_alternatives: int,
    time_step: int,
    time_window: int
) -> pd.DataFrame:
    """
    Generate and store vehicle routes and route points using Valhalla API and multiprocessing.

    Args:
        session: SQLAlchemy session
        route_class: SQLAlchemy route_class model
        route_point_class: SQLAlchemy route_point_class model
        run_config_id: Run configuration ID
        iteration_id: Iteration ID
        vehicles_df: DataFrame of vehicles_df
        edges_gdf: GeoDataFrame of edges
        max_nr_of_alternative_routes: Number of alternative routes per vehicle
        time_step: Time step for route points
        time_window: Time window for route points
    """
    try:
        nest_asyncio.apply()
        async def batched_valhalla_fetch(vehicles_gdf, max_concurrent=20):
            semaphore = asyncio.Semaphore(max_concurrent)
            async with aiohttp.ClientSession() as http_session:
                async def fetch(vehicle):
                    async with semaphore:
                        origin = (vehicle['origin_geometry'].x, vehicle['origin_geometry'].y)
                        dest = (vehicle['destination_geometry'].x, vehicle['destination_geometry'].y)
                        return await async_get_routes_from_valhalla(http_session, origin, dest, k_alternatives)
                tasks = [fetch(vehicle) for _, vehicle in vehicles_gdf.iterrows()]
                return await asyncio.gather(*tasks)
        loop = asyncio.get_event_loop()
        logger.info("Fetching routes from Valhalla...")
        t0 = time.time()
        routes_data_list = loop.run_until_complete(batched_valhalla_fetch(vehicles_gdf))
        logger.info(f"Valhalla fetch completed in {time.time() - t0:.2f} seconds")
        edges_proj = edges_gdf.to_crs(epsg=3857)
        edges_proj_dict = {'edges': edges_proj, 'crs': edges_proj.crs}
        vehicle_data_list = [
            (vehicle, idx, routes_data_list[i], edges_proj_dict, time_step, time_window)
            for i, (idx, vehicle) in enumerate(vehicles_gdf.iterrows())
            if routes_data_list[i]
        ]
        logger.info("Processing vehicle routes...")
        t1 = time.time()
        with ProcessPoolExecutor(max_workers=min(16, multiprocessing.cpu_count())) as executor:
            results = list(executor.map(process_vehicle_route, vehicle_data_list))
        logger.info(f"Processing completed in {time.time() - t1:.2f} seconds")
        all_vehicle_routes = []
        all_route_points = []
        for vehicle_routes, route_points in results:
            all_vehicle_routes.extend(vehicle_routes)
            all_route_points.extend(route_points)
        logger.info("Writing to database...")
        t2 = time.time()
        vehicle_objs = [
            route_class(
                vehicle_id=rec['vehicle_id'],
                run_configs_id=run_config_id,
                iteration_id=iteration_id,
                route_id=rec['route_id'],
                duration=rec['duration'],
                distance=rec['distance'],
                duration_in_traffic=None
            ) for rec in all_vehicle_routes
        ]
        route_objs = [
            route_point_class(
                vehicle_id=rec['vehicle_id'],
                run_configs_id=run_config_id,
                iteration_id=iteration_id,
                route_id=rec['route_id'],
                point_id=rec['point_id'],
                edge_id=int(rec['edge_id']),
                cardinal=rec['cardinal'],
                speed=rec['speed'],
                lat=rec['lat'],
                lon=rec['lon'],
                time=rec['time']
            ) for rec in all_route_points
        ]
        session.bulk_save_objects(vehicle_objs)
        session.bulk_save_objects(route_objs)
        session.commit()
        logger.info(f"Routes and route points written to DB in {time.time() - t2:.2f} seconds.")
        vehicle_routes_df = pd.DataFrame(all_vehicle_routes)
        #print("vehicle_routes_df: ", vehicle_routes_df)
        return vehicle_routes_df
    
    except Exception as e:
        logger.error(f"Error in generate_vehicle_routes: {e}", exc_info=True)
        session.rollback()
        return pd.DataFrame()
    
    finally:
        session.close()
