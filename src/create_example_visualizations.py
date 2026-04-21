#!/usr/bin/env python3
"""
Generate example visualizations for README.

Creates elevation change and clustering visualizations with:
- WGS84 (EPSG:4326) coordinate labels for lat/lon
- Satellite imagery basemap (ESRI World Imagery)
"""

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import contextily as ctx
from pathlib import Path


def create_elevation_change_map(data_path, output_path, subset_size=15000):
    """
    Create elevation change visualization with satellite basemap.

    Parameters
    ----------
    data_path : str or Path
        Path to areas.parquet file
    output_path : str or Path
        Path to save output PNG
    subset_size : int
        Size of subset area in meters (centered)
    """
    print("Loading data for elevation change map...")
    df = gpd.read_parquet(data_path)

    # Select a representative subset for visualization
    bbox = df.total_bounds
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2

    # Create subset around center
    half_size = subset_size / 2
    subset = df.cx[center_x-half_size:center_x+half_size,
                   center_y-half_size:center_y+half_size]

    # If subset is too small, use a larger area
    if len(subset) < 100:
        subset = df.cx[center_x-15000:center_x+15000,
                       center_y-15000:center_y+15000]

    print(f"Visualizing {len(subset)} polygons")

    # Reproject to Web Mercator for basemap compatibility
    subset_wm = subset.to_crs(epsg=3857)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12), dpi=150)

    # Classify polygons by elevation change
    subset_wm['change_class'] = pd.cut(
        subset_wm['diff_mean'],
        bins=[-np.inf, -3, -1.5, 1.5, 3, np.inf],
        labels=['Severe Erosion (< -3m)', 'Erosion (-3m to -1.5m)',
                'Moderate Change (-1.5m to 1.5m)', 'Deposition (1.5m to 3m)',
                'Severe Deposition (> 3m)']
    )

    # Color scheme: red for erosion, blue for deposition
    colors = {
        'Severe Erosion (< -3m)': '#8B0000',
        'Erosion (-3m to -1.5m)': '#FF6B6B',
        'Moderate Change (-1.5m to 1.5m)': '#FFD93D',
        'Deposition (1.5m to 3m)': '#6BCF7F',
        'Severe Deposition (> 3m)': '#0066CC'
    }

    # Plot each class
    for change_class, color in colors.items():
        data = subset_wm[subset_wm['change_class'] == change_class]
        if len(data) > 0:
            data.plot(ax=ax, color=color, edgecolor='black', linewidth=0.3, alpha=0.8)

    # Add satellite basemap
    try:
        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, alpha=0.6)
    except Exception as e:
        print(f"Warning: Could not add basemap: {e}")
        ax.set_facecolor('#f0f0f0')

    # Convert axis labels to WGS84 (lat/lon)
    # Get current bounds in Web Mercator
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Create a GeoDataFrame with the bounds to convert coordinates
    from shapely.geometry import Point

    # Sample points along axes for conversion
    n_ticks = 6
    x_ticks_wm = np.linspace(xlim[0], xlim[1], n_ticks)
    y_ticks_wm = np.linspace(ylim[0], ylim[1], n_ticks)

    # Convert x-axis ticks (longitude)
    x_points = [Point(x, ylim[0]) for x in x_ticks_wm]
    x_gdf = gpd.GeoDataFrame(geometry=x_points, crs='EPSG:3857')
    x_gdf_wgs84 = x_gdf.to_crs('EPSG:4326')
    x_labels = [f"{p.x:.4f}°E" for p in x_gdf_wgs84.geometry]

    # Convert y-axis ticks (latitude)
    y_points = [Point(xlim[0], y) for y in y_ticks_wm]
    y_gdf = gpd.GeoDataFrame(geometry=y_points, crs='EPSG:3857')
    y_gdf_wgs84 = y_gdf.to_crs('EPSG:4326')
    y_labels = [f"{p.y:.4f}°S" for p in y_gdf_wgs84.geometry]

    ax.set_xticks(x_ticks_wm)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_yticks(y_ticks_wm)
    ax.set_yticklabels(y_labels)

    # Add legend
    legend_elements = [Patch(facecolor=color, edgecolor='black', label=label)
                       for label, color in colors.items()]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
              title='Elevation Change', framealpha=0.95)

    # Styling
    ax.set_title('Erosion and Deposition Patterns in Gisborne Region\n(2018-2020 to 2023)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude', fontsize=11, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=11, fontweight='bold')

    # Add statistics text
    erosion_count = len(subset[subset['diff_mean'] < -1.5])
    deposition_count = len(subset[subset['diff_mean'] > 1.5])
    stats_text = f'Erosion areas: {erosion_count}\nDeposition areas: {deposition_count}'
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved elevation change map to {output_path}")
    plt.close()


def create_clustering_map(data_path, output_path, subset_size=15000):
    """
    Create clustering visualization with satellite basemap.

    Parameters
    ----------
    data_path : str or Path
        Path to clusters.parquet file
    output_path : str or Path
        Path to save output PNG
    subset_size : int
        Size of subset area in meters (centered)
    """
    print("Loading clustered data...")
    df = gpd.read_parquet(data_path)

    # Select a representative subset for visualization
    bbox = df.total_bounds
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2

    # Create subset around center
    half_size = subset_size / 2
    subset = df.cx[center_x-half_size:center_x+half_size,
                   center_y-half_size:center_y+half_size]

    # If subset is too small, use a larger area
    if len(subset) < 100:
        subset = df.cx[center_x-15000:center_x+15000,
                       center_y-15000:center_y+15000]

    print(f"Visualizing {len(subset)} polygons across {subset['cluster'].nunique()} clusters")

    # Reproject to Web Mercator for basemap compatibility
    subset_wm = subset.to_crs(epsg=3857)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12), dpi=150)

    # Create a colormap with distinct colors for clusters
    n_clusters = df['cluster'].nunique()
    cmap = plt.colormaps.get_cmap('tab20')

    # Plot clusters
    subset_wm.plot(
        column='cluster',
        ax=ax,
        cmap=cmap,
        edgecolor='black',
        linewidth=0.3,
        alpha=0.85,
        legend=True,
        legend_kwds={
            'label': 'Cluster ID',
            'orientation': 'vertical',
            'shrink': 0.8,
            'pad': 0.05
        }
    )

    # Add satellite basemap
    try:
        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, alpha=0.6)
    except Exception as e:
        print(f"Warning: Could not add basemap: {e}")
        ax.set_facecolor('#f0f0f0')

    # Convert axis labels to WGS84 (lat/lon)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    from shapely.geometry import Point

    # Sample points along axes for conversion
    n_ticks = 6
    x_ticks_wm = np.linspace(xlim[0], xlim[1], n_ticks)
    y_ticks_wm = np.linspace(ylim[0], ylim[1], n_ticks)

    # Convert x-axis ticks (longitude)
    x_points = [Point(x, ylim[0]) for x in x_ticks_wm]
    x_gdf = gpd.GeoDataFrame(geometry=x_points, crs='EPSG:3857')
    x_gdf_wgs84 = x_gdf.to_crs('EPSG:4326')
    x_labels = [f"{p.x:.4f}°E" for p in x_gdf_wgs84.geometry]

    # Convert y-axis ticks (latitude)
    y_points = [Point(xlim[0], y) for y in y_ticks_wm]
    y_gdf = gpd.GeoDataFrame(geometry=y_points, crs='EPSG:3857')
    y_gdf_wgs84 = y_gdf.to_crs('EPSG:4326')
    y_labels = [f"{p.y:.4f}°S" for p in y_gdf_wgs84.geometry]

    ax.set_xticks(x_ticks_wm)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_yticks(y_ticks_wm)
    ax.set_yticklabels(y_labels)

    # Styling
    ax.set_title('K-Means Clustering Results (k=25)\nSimilar Erosion/Deposition Patterns',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude', fontsize=11, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=11, fontweight='bold')

    # Add statistics text
    n_polygons = len(subset)
    clusters_in_view = subset['cluster'].nunique()
    stats_text = f'Total polygons: {n_polygons}\nClusters shown: {clusters_in_view}/{n_clusters}'
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved clustering map to {output_path}")
    plt.close()


def main():
    """Generate both example visualizations."""
    # Setup paths
    repo_root = Path(__file__).parent.parent
    data_dir = repo_root / "data"

    # Generate elevation change map
    create_elevation_change_map(
        data_path=data_dir / "areas.parquet",
        output_path=repo_root / "elevation_change_example.png"
    )

    # Generate clustering map
    create_clustering_map(
        data_path=data_dir / "clusters.parquet",
        output_path=repo_root / "clustering_example.png"
    )

    print("\nDone! Generated example visualizations.")


if __name__ == "__main__":
    main()
