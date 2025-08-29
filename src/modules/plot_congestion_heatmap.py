import matplotlib.pyplot as plt
import folium
import branca.colormap as cm
from branca.colormap import LinearColormap, linear
from shapely.geometry import LineString
import geopandas as gpd
import logging
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)

def plot_congestion_heatmap_interactive(
    edges_gdf: gpd.GeoDataFrame,
    congestion_df: Optional[pd.DataFrame],
    offset_deg: float = 0.00005,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None
) -> Optional[folium.Map]:
    """
    Plot interactive congestion heatmap using Folium with green-to-red color scale and tooltips.
    Aggregates congestion score per edge_id before merging.

    Args:
        edges_gdf: GeoDataFrame of road network edges with geometry.
        congestion_df: DataFrame with columns ['edge_id', 'congestion_score'].
        offset_deg: Offset for parallel lines in degrees.
        vmin: Minimum value for colormap (optional, for consistent scaling).
        vmax: Maximum value for colormap (optional, for consistent scaling).

    Returns:
        Folium Map object or None if no data.
    """
    if congestion_df is None or congestion_df.empty:
        logger.warning("No congestion map data to plot.")
        return None

    # Aggregate congestion scores by edge_id
    congestion_agg = congestion_df.groupby('edge_id', as_index=False)['congestion_score'].sum()
    congestion_agg = pd.DataFrame(congestion_agg)

    # Merge scores with edges
    merged = edges_gdf.merge(congestion_agg, left_on='edge_id', right_on='edge_id', how='left')
    merged['congestion_score'] = merged['congestion_score'].fillna(0)
    merged['edge_id'] = merged['edge_id']

    # Convert to WGS84 for mapping
    merged = merged.to_crs(epsg=4326)

    # Setup folium map
    minx, miny, maxx, maxy = merged.total_bounds
    center_lat = (miny + maxy) / 2
    center_lon = (minx + maxx) / 2
    map = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles='cartodbpositron')
    map.fit_bounds([[miny, minx], [maxy, maxx]])

    # Colormap
    if vmin is None:
        vmin = float(merged['congestion_score'].min())
    if vmax is None:
        vmax = float(merged['congestion_score'].max())
    colormap = LinearColormap(['silver', 'yellow', 'red', 'purple'], vmin=vmin, vmax=vmax, caption='Congestion Score')
    colormap.add_to(map)

    # Plot edges
    for _, row in merged.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty or geom.length == 0:
            continue
        coords = list(geom.coords)
        x0, y0 = coords[0]
        x1, y1 = coords[-1]
        side = 'left' if ((y1 < y0) and (x1 < x0)) or ((y1 > y0) and (x1 > x0)) else 'right'
        try:
            offset_geom = geom.parallel_offset(offset_deg, side=side, join_style=2)
            lines = [offset_geom] if isinstance(offset_geom, LineString) else list(offset_geom.geoms)
            for line in lines:
                coords = [(lat, lon) for lon, lat in line.coords]
                folium.PolyLine(
                    coords,
                    color=colormap(row['congestion_score']),
                    weight=3,
                    opacity=0.7,
                    tooltip=f"Edge ID: {row['edge_id']}<br>Score: {row['congestion_score']:.2f}"
                ).add_to(map)
        except Exception as e:
            logger.warning(f"Skipping edge {row['edge_id']} due to offset error: {e}")
    return map


def plot_congestion_heatmap(
    edges_gdf: gpd.GeoDataFrame,
    congestion_df: Optional[pd.DataFrame],
    cmap: str = 'Reds',
    figsize: tuple = (12, 12)
) -> None:
    """
    Aggregates congestion data over edges and plots a heatmap.

    Args:
        edges_gdf: GeoDataFrame containing road network edges with geometry.
        congestion_df: DataFrame containing edge_id and congestion_score.
        cmap: Color map used for the heatmap (default: 'Reds').
        figsize: Tuple defining the size of the plot (default: (12, 12)).
    """
    if congestion_df is None or congestion_df.empty:
        logger.warning("No congestion map data to plot.")
        return
    # Merge and clean
    merged_gdf = edges_gdf.merge(congestion_df, left_on='edge_id', right_on='edge_id', how='left')
    merged_gdf['congestion_score'] = merged_gdf['congestion_score'].fillna(0)
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    merged_gdf.plot(
        column='congestion_score',
        cmap=cmap,
        linewidth=2,
        ax=ax,
        legend=True,
        legend_kwds={'label': "Congestion Score", 'shrink': 0.5}
    )
    ax.set_title('Traffic Congestion Heatmap')
    ax.set_axis_off()
    plt.show()


