"""
Comprehensive Cross-Validation Structure Visualization
Shows the spatial and temporal organization of the CV folds
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
import xarray as xr
from sklearn.cluster import KMeans
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_study_data():
    """Load the study data to get spatial coordinates and temporal structure"""
    
    # Load a sample dataset to get coordinates
    # Using the processed data as it has the complete spatiotemporal structure
    data_file = "data/processed/coarse_data_Aquifer_daily.nc"
    
    try:
        ds = xr.open_dataset(data_file)
        print(f"   📊 Loaded data: {ds.dims}")
        
        # Get spatial coordinates
        lats = ds.latitude.values
        lons = ds.longitude.values
        times = pd.to_datetime(ds.time.values)
        
        # Create coordinate grid
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        spatial_coords = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
        
        return spatial_coords, lats, lons, times, ds
        
    except FileNotFoundError:
        print("   ⚠️ Main data file not found, using CHIRPS extent")
        # Fallback: create coordinates from CHIRPS extent
        # Study area: -113.95 to -77.84°W, 28.84 to 51.20°N
        lats = np.linspace(28.84, 51.20, 50)
        lons = np.linspace(-113.95, -77.84, 80)
        
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        spatial_coords = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
        
        # Create time series from 2003 to 2024
        times = pd.date_range('2003-01-01', '2024-11-01', freq='M')
        
        return spatial_coords, lats, lons, times, None

def create_spatial_blocks(spatial_coords, n_blocks=5):
    """Create spatial blocks using KMeans clustering"""
    
    # Remove duplicates for clustering
    unique_coords, inverse_indices = np.unique(spatial_coords, axis=0, return_inverse=True)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_blocks, random_state=42, n_init=10)
    unique_blocks = kmeans.fit_predict(unique_coords)
    
    # Map back to all samples
    spatial_blocks = unique_blocks[inverse_indices]
    
    return spatial_blocks, kmeans.cluster_centers_

def create_temporal_blocks(times, n_blocks=4, buffer_months=1):
    """Create temporal blocks with buffer periods"""
    
    unique_times = pd.to_datetime(times).unique()
    n_times = len(unique_times)
    
    # Calculate block size
    block_size = n_times // n_blocks
    
    temporal_blocks = {}
    block_periods = {}
    
    for i in range(n_blocks):
        start_idx = i * block_size
        end_idx = (i + 1) * block_size if i < n_blocks - 1 else n_times
        
        # Add buffer gap between blocks
        if i > 0:
            start_idx += buffer_months
        if i < n_blocks - 1:
            end_idx -= buffer_months
            
        if start_idx < end_idx:
            block_times = unique_times[start_idx:end_idx]
            temporal_blocks[i] = block_times
            block_periods[i] = (block_times.min(), block_times.max())
        else:
            temporal_blocks[i] = np.array([])
            block_periods[i] = (None, None)
    
    return temporal_blocks, block_periods

def plot_us_context_with_study_area():
    """Panel A: US map with study area highlighted"""
    
    # Create the map projection
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), subplot_kw={'projection': proj})
    
    # Set extent to cover continental US
    ax.set_extent([-130, -65, 20, 55], crs=proj)
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=1)
    ax.add_feature(cfeature.BORDERS, linewidth=1)
    ax.add_feature(cfeature.STATES, linewidth=0.5, alpha=0.5)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
    
    # Study area bounds from CHIRPS
    study_west, study_east = -113.95, -77.84
    study_south, study_north = 28.84, 51.20
    
    # Add study area rectangle
    study_rect = Rectangle(
        (study_west, study_south), 
        study_east - study_west, 
        study_north - study_south,
        linewidth=4, 
        edgecolor='red', 
        facecolor='red', 
        alpha=0.3,
        transform=proj
    )
    ax.add_patch(study_rect)
    
    # Add title and labels
    ax.set_title('(A) Study Area Context', fontsize=24, fontweight='bold')
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.xlabel_style = {'size': 14}
    gl.ylabel_style = {'size': 14}
    
    # Add text annotation for study area
    ax.text(-95, 45, 'GRACE Downscaling\nStudy Area', 
            fontsize=16, fontweight='bold', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
            transform=proj, ha='center')
    
    return fig, ax

def plot_spatial_blocks(spatial_coords, lats, lons, spatial_blocks, cluster_centers):
    """Panel B: Spatial blocks from K-means clustering"""
    
    # Reshape spatial blocks to grid
    spatial_blocks_grid = spatial_blocks.reshape(len(lats), len(lons))
    
    # Create the plot
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), subplot_kw={'projection': proj})
    
    # Set extent to study area
    ax.set_extent([-114, -77, 28, 52], crs=proj)
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=1)
    ax.add_feature(cfeature.BORDERS, linewidth=1)
    ax.add_feature(cfeature.STATES, linewidth=0.8, alpha=0.7)
    
    # Plot spatial blocks
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Create meshgrid for plotting
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Plot each block with different colors
    for block_id in range(5):
        mask = spatial_blocks_grid == block_id
        if np.any(mask):
            # Create masked data
            block_data = np.where(mask, block_id, np.nan)
            
            # Plot the block
            im = ax.contourf(lon_grid, lat_grid, block_data, 
                           levels=[block_id-0.5, block_id+0.5],
                           colors=[colors[block_id]], alpha=0.6, 
                           transform=proj)
            
            # Add block label (large number instead of star)
            center_lat, center_lon = cluster_centers[block_id]
            ax.text(center_lon, center_lat, f'{block_id}', 
                   fontsize=32, fontweight='bold', ha='center', va='center',
                   color='white', 
                   bbox=dict(boxstyle="circle,pad=0.3", facecolor='black', alpha=0.8),
                   transform=proj, zorder=11)
    
    # Add title
    ax.set_title('(B) Spatial Blocks (K-Means Clustering)', fontsize=24, fontweight='bold')
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.xlabel_style = {'size': 14}
    gl.ylabel_style = {'size': 14}
    
    return fig, ax

def plot_temporal_blocks(times, temporal_blocks, block_periods):
    """Panel C: Temporal division timeline"""
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    
    # Colors for temporal blocks
    colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']
    
    # Plot timeline
    y_pos = 0.5
    timeline_height = 0.3
    
    # Full timeline background
    full_start = pd.to_datetime(times).min()
    full_end = pd.to_datetime(times).max()
    
    # Draw full timeline background
    ax.barh(y_pos, (full_end - full_start).days, left=full_start, 
           height=timeline_height, color='lightgray', alpha=0.3, 
           edgecolor='black', linewidth=2)
    
    # Draw each temporal block
    for block_id, (start_time, end_time) in block_periods.items():
        if start_time is not None and end_time is not None:
            # Convert to matplotlib dates for plotting
            duration = (end_time - start_time).days
            
            # Draw the block
            ax.barh(y_pos, duration, left=start_time, 
                   height=timeline_height, color=colors[block_id], 
                   alpha=0.8, edgecolor='white', linewidth=2)
            
            # Add block label
            mid_time = start_time + (end_time - start_time) / 2
            ax.text(mid_time, y_pos, f'T{block_id}', 
                   fontsize=24, fontweight='bold', ha='center', va='center',
                   color='white' if block_id in [0, 3] else 'black')
            
            # Add time period text below
            ax.text(mid_time, y_pos - 0.2, 
                   f'{start_time.strftime("%Y-%m")}\nto\n{end_time.strftime("%Y-%m")}', 
                   fontsize=18, ha='center', va='top', fontweight='bold')
    
    # Formatting
    ax.set_xlim(full_start - pd.DateOffset(months=6), full_end + pd.DateOffset(months=6))
    ax.set_ylim(0, 1)
    ax.set_ylabel('')
    ax.set_yticks([])
    
    # Format x-axis
    ax.set_xlabel('Time Period', fontsize=22, fontweight='bold')
    
    # Add title
    ax.set_title('(C) Temporal Blocks with Buffer Periods', fontsize=28, fontweight='bold')
    
    # Add grid
    ax.grid(True, axis='x', alpha=0.3)
    
    # Format dates on x-axis
    years = pd.date_range(start='2003', end='2025', freq='2Y')
    ax.set_xticks(years)
    ax.set_xticklabels([year.strftime('%Y') for year in years], fontsize=18, fontweight='bold')
    
    return fig, ax

def plot_cv_fold_matrix():
    """Panel D: Complete CV fold structure matrix"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Create the 5x4 fold matrix
    spatial_blocks = 5
    temporal_blocks = 4
    fold_matrix = np.arange(1, spatial_blocks * temporal_blocks + 1).reshape(spatial_blocks, temporal_blocks)
    
    # Create heatmap
    im = ax.imshow(fold_matrix, cmap='tab20', aspect='auto')
    
    # Add fold numbers and details
    for i in range(spatial_blocks):
        for j in range(temporal_blocks):
            fold_num = fold_matrix[i, j]
            
            # Main fold number only
            ax.text(j, i, f'Fold {fold_num}', 
                   fontsize=20, fontweight='bold', 
                   ha='center', va='center', color='white',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
    
    # Formatting
    ax.set_xticks(range(temporal_blocks))
    ax.set_xticklabels([f'Temporal\nBlock {i}' for i in range(temporal_blocks)], fontsize=18, fontweight='bold')
    
    ax.set_yticks(range(spatial_blocks))
    ax.set_yticklabels([f'Spatial\nBlock {i}' for i in range(spatial_blocks)], fontsize=18, fontweight='bold')
    
    ax.set_xlabel('Temporal Blocks', fontsize=22, fontweight='bold')
    ax.set_ylabel('Spatial Blocks', fontsize=22, fontweight='bold')
    
    # Add title
    ax.set_title('(D) Complete CV Fold Structure Matrix', fontsize=24, fontweight='bold')
    
    # Add grid
    ax.set_xticks(np.arange(-0.5, temporal_blocks, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, spatial_blocks, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)
    
    # Add total folds annotation
    ax.text(1.5, -0.8, f'Total: {spatial_blocks} × {temporal_blocks} = {spatial_blocks * temporal_blocks} CV Folds', 
           fontsize=16, fontweight='bold', ha='center', 
           bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.7),
           transform=ax.transData)
    
    return fig, ax

def create_combined_cv_visualization():
    """Create the complete 4-panel CV structure visualization"""
    
    print("🗺️ Creating Comprehensive CV Structure Visualization")
    print("=" * 60)
    
    # Load data
    print("📊 Loading study data...")
    spatial_coords, lats, lons, times, ds = load_study_data()
    
    # Create spatial blocks
    print("🔍 Creating spatial blocks with K-means...")
    spatial_blocks, cluster_centers = create_spatial_blocks(spatial_coords, n_blocks=5)
    print(f"   ✅ Created {len(np.unique(spatial_blocks))} spatial blocks")
    
    # Create temporal blocks  
    print("📅 Creating temporal blocks...")
    temporal_blocks, block_periods = create_temporal_blocks(times, n_blocks=4, buffer_months=1)
    print(f"   ✅ Created {len(temporal_blocks)} temporal blocks")
    
    # Set up the figure with large, presentation-ready text
    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 18,
        'axes.titlesize': 24,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 12,
        'figure.titlesize': 28
    })
    
    print("🎨 Creating visualizations...")
    
    # Create individual panels
    print("   Panel A: US context map...")
    fig_a, ax_a = plot_us_context_with_study_area()
    
    print("   Panel B: Spatial blocks...")
    fig_b, ax_b = plot_spatial_blocks(spatial_coords, lats, lons, spatial_blocks, cluster_centers)
    
    print("   Panel C: Temporal blocks...")
    fig_c, ax_c = plot_temporal_blocks(times, temporal_blocks, block_periods)
    
    print("   Panel D: CV fold matrix...")
    fig_d, ax_d = plot_cv_fold_matrix()
    
    # Save individual panels
    output_dir = Path("figures_coarse_to_fine/cv_structure")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 Saving visualizations to {output_dir}...")
    
    fig_a.savefig(output_dir / "panel_a_us_context.png", dpi=300, bbox_inches='tight')
    fig_b.savefig(output_dir / "panel_b_spatial_blocks.png", dpi=300, bbox_inches='tight')
    fig_c.savefig(output_dir / "panel_c_temporal_blocks.png", dpi=300, bbox_inches='tight')
    fig_d.savefig(output_dir / "panel_d_cv_matrix.png", dpi=300, bbox_inches='tight')
    
    # Create combined figure using a simpler approach
    print("🎭 Creating combined 4-panel figure...")
    
    # Load the saved individual panels and combine them
    import matplotlib.image as mpimg
    
    # Read the saved panel images
    img_a = mpimg.imread(output_dir / "panel_a_us_context.png")
    img_b = mpimg.imread(output_dir / "panel_b_spatial_blocks.png")
    img_c = mpimg.imread(output_dir / "panel_c_temporal_blocks.png")
    img_d = mpimg.imread(output_dir / "panel_d_cv_matrix.png")
    
    # Create combined figure
    fig, axes = plt.subplots(2, 2, figsize=(24, 18))
    
    # Display each panel
    axes[0, 0].imshow(img_a)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_b)
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(img_c)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(img_d)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Save combined figure
    combined_path = output_dir / "comprehensive_cv_structure.png"
    fig.savefig(combined_path, dpi=300, bbox_inches='tight')
    
    plt.close(fig)  # Close to free memory
    
    print(f"✅ Comprehensive CV Structure Visualization Complete!")
    print(f"📁 Saved to: {output_dir}")
    print(f"🖼️ Combined figure: {combined_path}")
    
    # Print summary
    print("\n📊 CV STRUCTURE SUMMARY:")
    print(f"   • Study Area: {lons.min():.1f}°W to {lons.max():.1f}°W, {lats.min():.1f}°N to {lats.max():.1f}°N")
    print(f"   • Spatial Blocks: {len(np.unique(spatial_blocks))} (K-means clustering)")
    print(f"   • Temporal Blocks: {len(temporal_blocks)} (with 1-month buffers)")
    print(f"   • Total CV Folds: {len(np.unique(spatial_blocks)) * len(temporal_blocks)}")
    print(f"   • Time Period: {pd.to_datetime(times).min().strftime('%Y-%m')} to {pd.to_datetime(times).max().strftime('%Y-%m')}")
    
    return fig, output_dir

# Keep the old function name for compatibility
create_comprehensive_cv_visualization = create_combined_cv_visualization

if __name__ == "__main__":
    # Create the comprehensive visualization
    fig, output_dir = create_comprehensive_cv_visualization()
    plt.show()