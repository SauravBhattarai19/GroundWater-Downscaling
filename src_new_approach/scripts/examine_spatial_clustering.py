#!/usr/bin/env python3
"""
Examine Spatial Clustering in Detail

Investigates why there are no buffer conflicts and visualizes
the actual spatial clustering results.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# Add project path
sys.path.insert(0, 'src_new_approach')

# Import our CV implementation
from spatiotemporal_cv import BlockedSpatiotemporalCV, prepare_metadata_for_cv

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,                
    'axes.labelsize': 14,           
    'axes.titlesize': 16,           
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'lines.linewidth': 2,           
})


def examine_spatial_clustering():
    """Examine the spatial clustering in detail."""
    print("🔍 Examining spatial clustering structure...")
    
    # Load feature stack
    feature_path = "data/processed_coarse_to_fine/feature_stack_55km.nc"
    ds = xr.open_dataset(feature_path)
    
    lat_values = ds.lat.values
    lon_values = ds.lon.values
    time_values = pd.to_datetime(ds.time.values)
    
    print(f"   Study area: {lat_values.min():.2f}°N to {lat_values.max():.2f}°N")
    print(f"               {lon_values.max():.2f}°W to {lon_values.min():.2f}°W")
    print(f"   Spatial extent: {lat_values.max() - lat_values.min():.1f}° lat × {lon_values.min() - lon_values.max():.1f}° lon")
    
    # Create sample data
    n_times = len(time_values)
    n_spatial = len(lat_values) * len(lon_values)
    n_samples = n_times * n_spatial
    
    X_dummy = np.random.randn(n_samples, 10)
    y_dummy = np.random.randn(n_samples)
    
    common_dates = [t.strftime('%Y-%m') for t in time_values]
    spatial_shape = (len(lat_values), len(lon_values))
    
    metadata = prepare_metadata_for_cv(
        X_dummy, y_dummy, common_dates, spatial_shape, feature_path
    )
    
    # Initialize CV splitter
    cv_splitter = BlockedSpatiotemporalCV(
        n_spatial_blocks=5,
        n_temporal_blocks=4, 
        spatial_buffer_deg=0.5,
        temporal_buffer_months=1,
        random_state=42
    )
    
    # Get spatial blocks
    spatial_coords = metadata['spatial_coords']
    spatial_blocks = cv_splitter.create_spatial_blocks(spatial_coords)
    cluster_centers = cv_splitter.cluster_centers_
    
    print(f"   Created {len(cluster_centers)} spatial clusters")
    
    # Analyze cluster centers and distances
    print(f"\n📍 Cluster Centers:")
    for i, center in enumerate(cluster_centers):
        print(f"   Block {i}: {center[0]:.2f}°N, {center[1]:.2f}°W")
    
    # Calculate distances between cluster centers
    center_distances = cdist(cluster_centers, cluster_centers, metric='euclidean')
    
    print(f"\n📏 Distances between cluster centers (degrees):")
    for i in range(len(cluster_centers)):
        for j in range(i+1, len(cluster_centers)):
            dist_deg = center_distances[i, j]
            dist_km = dist_deg * 111  # Approximate conversion
            print(f"   Block {i} ↔ Block {j}: {dist_deg:.2f}° ({dist_km:.0f} km)")
    
    # Find minimum distance between centers
    min_distance = np.min(center_distances[center_distances > 0])
    min_distance_km = min_distance * 111
    
    print(f"\n🎯 Minimum distance between cluster centers: {min_distance:.2f}° ({min_distance_km:.0f} km)")
    print(f"   Current buffer size: {cv_splitter.spatial_buffer:.2f}° ({cv_splitter.spatial_buffer * 111:.0f} km)")
    
    if min_distance > cv_splitter.spatial_buffer * 2:
        print(f"   ✅ No buffer conflicts: clusters are well-separated")
        print(f"      Buffer zone ({cv_splitter.spatial_buffer * 111:.0f} km) < half minimum distance ({min_distance_km/2:.0f} km)")
    else:
        print(f"   ⚠️ Potential buffer conflicts: clusters may be too close")
        print(f"      Buffer zone ({cv_splitter.spatial_buffer * 111:.0f} km) ≥ half minimum distance ({min_distance_km/2:.0f} km)")
    
    return {
        'lat_values': lat_values,
        'lon_values': lon_values,
        'spatial_coords': spatial_coords,
        'spatial_blocks': spatial_blocks,
        'cluster_centers': cluster_centers,
        'center_distances': center_distances,
        'cv_splitter': cv_splitter
    }


def visualize_spatial_clustering(cluster_data, output_dir):
    """Create detailed visualization of spatial clustering."""
    print("🎨 Creating spatial clustering visualization...")
    
    lat_values = cluster_data['lat_values']
    lon_values = cluster_data['lon_values']
    spatial_coords = cluster_data['spatial_coords']
    spatial_blocks = cluster_data['spatial_blocks']
    cluster_centers = cluster_data['cluster_centers']
    center_distances = cluster_data['center_distances']
    cv_splitter = cluster_data['cv_splitter']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel 1: Spatial blocks with grid
    ax1 = axes[0, 0]
    
    # Create spatial grid for visualization
    lat_grid, lon_grid = np.meshgrid(lat_values, lon_values, indexing='ij')
    
    # Plot spatial blocks
    unique_blocks = np.unique(spatial_blocks)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_blocks)))
    
    for i, block_id in enumerate(unique_blocks):
        mask_2d = np.zeros_like(lat_grid, dtype=bool)
        
        # Find samples in this spatial block (using first time step)
        block_samples = np.where((spatial_blocks == block_id) & 
                                (np.arange(len(spatial_blocks)) < len(lat_values) * len(lon_values)))[0]
        
        for idx in block_samples:
            coord = spatial_coords[idx]
            lat_idx = np.argmin(np.abs(lat_values - coord[0]))
            lon_idx = np.argmin(np.abs(lon_values - coord[1]))
            mask_2d[lat_idx, lon_idx] = True
        
        ax1.contourf(lon_grid, lat_grid, mask_2d, levels=[0.5, 1.5], 
                    colors=[colors[i]], alpha=0.7)
        ax1.contour(lon_grid, lat_grid, mask_2d, levels=[0.5], 
                   colors='black', linewidths=1, alpha=0.8)
    
    # Plot cluster centers
    for i, center in enumerate(cluster_centers):
        ax1.plot(center[1], center[0], 'ko', markersize=12, markeredgecolor='white', markeredgewidth=2)
        ax1.text(center[1], center[0], str(i), ha='center', va='center', 
                color='white', fontweight='bold', fontsize=10)
        
        # Draw buffer circles around centers
        circle = plt.Circle((center[1], center[0]), cv_splitter.spatial_buffer, 
                           fill=False, edgecolor='red', linewidth=2, linestyle='--', alpha=0.7)
        ax1.add_patch(circle)
    
    ax1.set_xlabel('Longitude (°W)', fontweight='bold')
    ax1.set_ylabel('Latitude (°N)', fontweight='bold')
    ax1.set_title('(A) Spatial Blocks with Buffer Zones', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Panel 2: Distance matrix between cluster centers
    ax2 = axes[0, 1]
    
    im = ax2.imshow(center_distances, cmap='viridis', aspect='equal')
    ax2.set_xticks(range(len(cluster_centers)))
    ax2.set_yticks(range(len(cluster_centers)))
    ax2.set_xticklabels([f'Block {i}' for i in range(len(cluster_centers))])
    ax2.set_yticklabels([f'Block {i}' for i in range(len(cluster_centers))])
    
    # Add text annotations
    for i in range(len(cluster_centers)):
        for j in range(len(cluster_centers)):
            dist = center_distances[i, j]
            color = 'white' if dist < center_distances.max()/2 else 'black'
            ax2.text(j, i, f'{dist:.2f}°', ha='center', va='center', 
                    color=color, fontweight='bold', fontsize=8)
    
    plt.colorbar(im, ax=ax2, label='Distance (degrees)')
    ax2.set_title('(B) Distance Matrix Between Centers', fontweight='bold')
    
    # Panel 3: Buffer analysis
    ax3 = axes[1, 0]
    
    # Calculate actual buffer overlaps for each cluster
    buffer_sizes = np.arange(0.1, 2.1, 0.1)
    overlap_counts = []
    
    for buffer_size in buffer_sizes:
        overlaps = 0
        for i in range(len(cluster_centers)):
            for j in range(len(cluster_centers)):
                if i != j and center_distances[i, j] < buffer_size * 2:
                    overlaps += 1
        overlap_counts.append(overlaps)
    
    ax3.plot(buffer_sizes, overlap_counts, 'b-', linewidth=3, marker='o', markersize=6)
    ax3.axvline(cv_splitter.spatial_buffer, color='red', linestyle='--', linewidth=2, 
               label=f'Current buffer ({cv_splitter.spatial_buffer:.1f}°)')
    
    ax3.set_xlabel('Buffer Size (degrees)', fontweight='bold')
    ax3.set_ylabel('Number of Potential Overlaps', fontweight='bold')
    ax3.set_title('(C) Buffer Overlap Analysis', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Panel 4: Statistics and recommendations
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    min_distance = np.min(center_distances[center_distances > 0])
    max_safe_buffer = min_distance / 2
    current_buffer = cv_splitter.spatial_buffer
    
    stats_text = f"""
SPATIAL CLUSTERING ANALYSIS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Study Area: {lat_values.min():.1f}°N to {lat_values.max():.1f}°N
            {lon_values.max():.1f}°W to {lon_values.min():.1f}°W

Spatial Extent: {lat_values.max() - lat_values.min():.1f}° × {abs(lon_values.min() - lon_values.max()):.1f}°

CLUSTER SEPARATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Number of Clusters: {len(cluster_centers)}
Min Distance: {min_distance:.2f}° ({min_distance * 111:.0f} km)
Max Distance: {center_distances.max():.2f}° ({center_distances.max() * 111:.0f} km)
Mean Distance: {center_distances[center_distances > 0].mean():.2f}° ({center_distances[center_distances > 0].mean() * 111:.0f} km)

BUFFER ANALYSIS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Current Buffer: {current_buffer:.2f}° ({current_buffer * 111:.0f} km)
Max Safe Buffer: {max_safe_buffer:.2f}° ({max_safe_buffer * 111:.0f} km)
Buffer Status: {'✅ SAFE' if current_buffer <= max_safe_buffer else '⚠️ RISKY'}

RECOMMENDATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    if current_buffer <= max_safe_buffer:
        stats_text += f"✅ Current buffer size is appropriate.\n   No spatial buffer conflicts expected."
    else:
        stats_text += f"⚠️ Current buffer may be too large.\n   Consider reducing to ≤{max_safe_buffer:.2f}° ({max_safe_buffer * 111:.0f} km)"
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    ax4.set_title('(D) Analysis Summary', fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save the visualization
    output_path = output_dir / "spatial_clustering_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✅ Saved spatial clustering analysis: {output_path}")
    
    plt.close()
    
    return output_path


def main():
    """Main execution function."""
    print("🔬 Spatial Clustering Analysis")
    print("=" * 50)
    
    # Setup output directory
    output_dir = Path("figures_coarse_to_fine/cv_clustering_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Examine clustering
        cluster_data = examine_spatial_clustering()
        
        # Create visualization
        visualization_fig = visualize_spatial_clustering(cluster_data, output_dir)
        
        print("\n" + "=" * 50)
        print("✅ Spatial Clustering Analysis Complete!")
        print(f"📁 Output saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())