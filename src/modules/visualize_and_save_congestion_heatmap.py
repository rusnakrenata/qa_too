from pathlib import Path
import geopandas as gpd
import pandas as pd
import logging
from plot_congestion_heatmap import plot_congestion_heatmap_interactive

logger = logging.getLogger(__name__)

def visualize_and_save_congestion_heatmap(
    edges_gdf: gpd.GeoDataFrame,
    congestion_df: pd.DataFrame,
    output_filename: Path,
    vmin: float = None,
    vmax: float = None,
    offset_deg: float = 0.0001  # or use global OFFSET_DEG if defined elsewhere
) -> None:
    """
    Visualize and save a single congestion heatmap.

    Args:
        edges: GeoDataFrame of edges.
        congestion_df: DataFrame with 'congestion_score' and geometry.
        output_filename: Path to save the heatmap HTML file.
        vmin: Minimum congestion value for consistent color scaling.
        vmax: Maximum congestion value for consistent color scaling.
        offset_deg: Small offset in degrees for rendering distinction.
    """
    try:
        if congestion_df.empty:
            logger.warning(f"Empty congestion DataFrame. Skipping heatmap: {output_filename}")
            return

        plot_map = plot_congestion_heatmap_interactive(
            edges_gdf, congestion_df, offset_deg=offset_deg, vmin=vmin, vmax=vmax
        )

        if plot_map is not None:
            plot_map.save(output_filename)
            logger.info(f"Heatmap saved: {output_filename}")
        else:
            logger.warning(f"Heatmap not generated for {output_filename}")

    except Exception as e:
        logger.error(f"Error generating heatmap for {output_filename}: {e}", exc_info=True)
