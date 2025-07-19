#!/usr/bin/env python3
"""
Create publication-ready study area figure for AGU Geophysical Research Letters
Mississippi River Basin GRACE Downscaling Study

Using the same data handling approach as the main pipeline.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import Rectangle
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.plot import show
import xarray as xr
import rioxarray as rxr
from shapely.geometry import box, Polygon
import warnings
warnings.filterwarnings('ignore')

# Add src to path (same as pipeline.py)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Try to import utils for consistent handling
try:
    from utils import parse_grace_months
except ImportError:
    print("Warning: Could not import utils")

# Set publication-quality defaults with larger text
plt.rcParams['font.family'] = 'DejaVu Sans'  # Use DejaVu Sans instead of Arial
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.dpi'] = 300

# Define Mississippi River Basin bounds from config/data_loader.py
MISSISSIPPI_BOUNDS = {
    'lon_min': -113.94,
    'lat_min': 28.84,
    'lon_max': -77.84,
    'lat_max': 49.74
}

def add_north_arrow(ax, x=0.95, y=0.95, arrow_length=0.1, color='black'):
    """Add north arrow to map."""
    ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
                ha='center', va='bottom', fontsize=14, fontweight='bold',
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', lw=2, color=color),
                color=color, transform=ax.transAxes)

def add_scale_bar(ax, length_km=200):
    """Add scale bar to map."""
    try:
        from matplotlib_scalebar.scalebar import ScaleBar
        scalebar = ScaleBar(111000, units='m', length_fraction=0.25, 
                           location='lower left', pad=0.5, color='black',
                           box_alpha=0.8, font_properties={'size': 9})
        ax.add_artist(scalebar)
    except ImportError:
        # Simple scale bar alternative
        ax.text(0.02, 0.02, f'{length_km} km', transform=ax.transAxes,
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3",
                facecolor="white", alpha=0.8))

def get_reference_raster(data_dir="data/raw/chirps"):
    """Get reference raster from CHIRPS (same as features.py)."""
    chirps_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.tif')])
    if chirps_files:
        chirps_path = os.path.join(data_dir, chirps_files[0])
        reference_raster = rxr.open_rasterio(chirps_path, masked=True).squeeze()
        # Ensure CRS is set
        if reference_raster.rio.crs is None:
            reference_raster = reference_raster.rio.write_crs('EPSG:4326')
        return reference_raster
    return None

def load_and_clip_raster_to_basin(raster_path, basin_gdf, reference_raster=None):
    """Load raster and clip to basin shapefile geometry."""
    try:
        print(f"    🔄 Loading raster: {os.path.basename(raster_path)}")
        
        # Open raster
        raster = rxr.open_rasterio(raster_path, masked=True).squeeze()
        
        # Set CRS if missing
        if raster.rio.crs is None:
            raster = raster.rio.write_crs('EPSG:4326')
        
        # Reproject to match reference if provided
        if reference_raster is not None:
            raster = raster.rio.reproject_match(reference_raster)
        
        print(f"    📏 Original raster shape: {raster.shape}")
        print(f"    📏 Original raster bounds: {raster.rio.bounds()}")
        
        # Clip to basin shapefile - this is the key fix
        if not basin_gdf.empty and basin_gdf.is_valid.all():
            # Ensure basin is in same CRS as raster
            basin_proj = basin_gdf.to_crs(raster.rio.crs)
            
            # Use rio.clip with shapefile geometry - this should actually clip the data
            raster_clipped = raster.rio.clip(basin_proj.geometry.values, 
                                           basin_proj.crs, 
                                           drop=True,  # This will drop areas outside the polygon
                                           invert=False)
            
            print(f"    ✂️ Clipped raster shape: {raster_clipped.shape}")
            print(f"    ✂️ Clipped raster bounds: {raster_clipped.rio.bounds()}")
            
            # Additional check - make sure clipping actually worked
            if (raster_clipped.rio.bounds() == raster.rio.bounds()):
                print("    ⚠️ Warning: Clipping may not have worked properly")
            
            return raster_clipped
        else:
            print("  ⚠️ Invalid basin geometry, using original raster")
            return raster
        
    except Exception as e:
        print(f"  ⚠️ Error loading/clipping raster {raster_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_and_reproject_raster(raster_path, reference_raster=None, clip_bounds=None):
    """Load raster and optionally reproject to match reference (like features.py)."""
    try:
        # Open raster
        raster = rxr.open_rasterio(raster_path, masked=True).squeeze()
        
        # Set CRS if missing
        if raster.rio.crs is None:
            raster = raster.rio.write_crs('EPSG:4326')
        
        # Reproject to match reference if provided
        if reference_raster is not None:
            raster = raster.rio.reproject_match(reference_raster)
        
        # Clip to bounds if provided
        if clip_bounds is not None:
            minx, miny, maxx, maxy = clip_bounds
            raster = raster.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
        
        return raster
        
    except Exception as e:
        print(f"  ⚠️ Error loading raster {raster_path}: {e}")
        return None

def create_basin_geometry():
    """Create basin geometry from bounds."""
    bounds = MISSISSIPPI_BOUNDS
    basin_polygon = Polygon([
        [bounds['lon_min'], bounds['lat_min']],
        [bounds['lon_min'], bounds['lat_max']],
        [bounds['lon_max'], bounds['lat_max']],
        [bounds['lon_max'], bounds['lat_min']],
        [bounds['lon_min'], bounds['lat_min']]
    ])
    return gpd.GeoDataFrame([1], geometry=[basin_polygon], crs='EPSG:4326')

def load_basin_shapefile():
    """Load basin shapefile with fallback to bounds."""
    # Try to load shapefile
    basin_paths = [
        "data/shapefiles/processed/mississippi_river_basin.shp",
        "data/shapefiles/mississippi_river_basin.shp",
        "data/raw/shapefiles/mississippi_river_basin.shp"
    ]
    
    for basin_path in basin_paths:
        if os.path.exists(basin_path):
            try:
                print(f"📂 Loading basin shapefile: {basin_path}")
                basin_gdf = gpd.read_file(basin_path)
                
                # Ensure it's in WGS84
                if basin_gdf.crs is None:
                    print("  ⚠️ No CRS found, assuming EPSG:4326")
                    basin_gdf = basin_gdf.set_crs('EPSG:4326')
                else:
                    basin_gdf = basin_gdf.to_crs('EPSG:4326')
                
                # Skip geometry validation for now and just use it
                print(f"  ✅ Loaded {len(basin_gdf)} features")
                
                # Check bounds are reasonable
                bounds = basin_gdf.total_bounds
                print(f"  ✅ Basin bounds: [{bounds[0]:.2f}, {bounds[1]:.2f}, {bounds[2]:.2f}, {bounds[3]:.2f}]")
                
                # Return the loaded shapefile
                return basin_gdf
                
            except Exception as e:
                print(f"  ⚠️ Error reading shapefile: {e}")
                continue
    
    # Fallback to creating from bounds
    print("  📍 Using predefined Mississippi River Basin bounds")
    return create_basin_geometry()

def safe_plot_basin_boundary(basin_gdf, ax, **kwargs):
    """Safely plot basin boundary, handling invalid geometries."""
    try:
        if not basin_gdf.empty and basin_gdf.is_valid.all():
            # Check bounds are finite
            bounds = basin_gdf.total_bounds
            if np.all(np.isfinite(bounds)):
                basin_gdf.boundary.plot(ax=ax, **kwargs)
                return True
            else:
                print("  ⚠️ Basin has infinite bounds, skipping boundary plot")
        else:
            print("  ⚠️ Invalid basin geometry, skipping boundary plot")
    except Exception as e:
        print(f"  ⚠️ Error plotting basin boundary: {e}")
    return False

def set_basin_axis_limits(ax, basin_gdf, padding=0.1):
    """Set axis limits based on basin bounds with padding."""
    try:
        bounds = basin_gdf.total_bounds
        if np.all(np.isfinite(bounds)):
            # Add padding
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            pad_x = width * padding
            pad_y = height * padding
            
            ax.set_xlim(bounds[0] - pad_x, bounds[2] + pad_x)
            ax.set_ylim(bounds[1] - pad_y, bounds[3] + pad_y)
        else:
            # Fallback to predefined bounds
            bounds = MISSISSIPPI_BOUNDS
            ax.set_xlim(bounds['lon_min'], bounds['lon_max'])
            ax.set_ylim(bounds['lat_min'], bounds['lat_max'])
    except Exception as e:
        print(f"  ⚠️ Error setting axis limits: {e}")
        # Fallback to predefined bounds
        bounds = MISSISSIPPI_BOUNDS
        ax.set_xlim(bounds['lon_min'], bounds['lon_max'])
        ax.set_ylim(bounds['lat_min'], bounds['lat_max'])

def create_study_area_figure():
    """Create comprehensive study area figure using pipeline approach."""
    
    print("🗺️ Creating Mississippi River Basin Study Area Figure")
    print("="*60)
    
    # Load basin geometry
    basin_gdf = load_basin_shapefile()
    basin_bounds = basin_gdf.total_bounds  # [minx, miny, maxx, maxy]
    
    # Get reference raster for consistent alignment
    print("\n📐 Loading reference raster for alignment...")
    reference_raster = get_reference_raster()
    
    # Create figure with 4 panels - research paper quality
    fig = plt.figure(figsize=(14, 16))
    
    # Define grid with minimal spacing for attractive layout
    gs = fig.add_gridspec(2, 2, hspace=0.08, wspace=0.05, 
                         left=0.05, right=0.95, top=0.90, bottom=0.05)
    
    # Panel labels
    panel_labels = ['A', 'B', 'C', 'D']
    
    # ========== PANEL A: Topography and Geographic Context ==========
    print("\n📍 Panel A: Topography and Geographic Context")
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Load DEM using rioxarray (same as pipeline)
    dem_path = "data/raw/usgs_dem/srtm_dem.tif"
    dem_plotted = False
    
    if os.path.exists(dem_path):
        print(f"  📂 Loading DEM: {dem_path}")
        try:
            # Load and clip DEM to basin shapefile
            dem_clipped = load_and_clip_raster_to_basin(dem_path, basin_gdf, reference_raster)
            
            if dem_clipped is not None:
                # Get data and extent
                dem_data = dem_clipped.values
                dem_extent = [float(dem_clipped.x.min()), float(dem_clipped.x.max()),
                             float(dem_clipped.y.min()), float(dem_clipped.y.max())]
                
                # Plot DEM
                if not np.all(np.isnan(dem_data)):
                    # Create hillshade
                    from matplotlib.colors import LightSource
                ls = LightSource(azdeg=315, altdeg=45)
                
                # Plot DEM with terrain colormap
                vmin = np.nanpercentile(dem_data, 2)
                vmax = np.nanpercentile(dem_data, 98)
                
                im1 = ax1.imshow(dem_data, extent=dem_extent, cmap='terrain',
                                alpha=0.8, vmin=vmin, vmax=vmax, origin='upper')
                
                # Add hillshade
                hillshade = ls.hillshade(dem_data, vert_exag=0.1)
                ax1.imshow(hillshade, extent=dem_extent, cmap='gray', 
                          alpha=0.3, origin='upper')
                
                # Colorbar
                cbar1 = plt.colorbar(im1, ax=ax1, orientation='horizontal', 
                                   pad=0.05, shrink=0.8, aspect=30)
                cbar1.set_label('Elevation (m)', fontsize=10)
                cbar1.ax.tick_params(labelsize=9)
                dem_plotted = True
                print("  ✅ DEM plotted successfully")
                
        except Exception as e:
            print(f"  ⚠️ Could not process DEM: {e}")
    
    if not dem_plotted:
        ax1.set_facecolor('#f0f0f0')
    
    # Add state boundaries - skip for now due to geopandas dataset deprecation
    try:
        # Alternative: Use Natural Earth data if available locally
        ne_states_path = "data/shapefiles/ne_110m_admin_1_states_provinces.shp"
        if os.path.exists(ne_states_path):
            states = gpd.read_file(ne_states_path)
            states = states.to_crs('EPSG:4326')
            
            # Clip to region
            states_clip = states.cx[basin_bounds[0]:basin_bounds[2], 
                                   basin_bounds[1]:basin_bounds[3]]
            states_clip.boundary.plot(ax=ax1, color='gray', linewidth=1, alpha=0.5)
            print("  ✅ State boundaries added")
        else:
            print("  ⚠️ State boundaries data not available")
    except Exception as e:
        print(f"  ⚠️ Could not load state boundaries: {e}")
    
    # Add basin outline
    safe_plot_basin_boundary(basin_gdf, ax1, color='black', linewidth=2)
    
    # Add cartographic elements
    add_north_arrow(ax1)
    add_scale_bar(ax1)
    
    # Labels and formatting - clean for research paper
    set_basin_axis_limits(ax1, basin_gdf)
    ax1.set_title('Topography and Geographic Context', fontsize=14, pad=15, fontweight='bold')
    ax1.text(0.02, 0.98, panel_labels[0], transform=ax1.transAxes, 
            fontsize=16, fontweight='bold', va='top', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax1.axis('off')  # Remove all axis elements for clean look
    
    # ========== PANEL B: GRACE Grids and USGS Wells ==========
    print("\n📍 Panel B: GRACE Grids and USGS Monitoring Wells")
    ax2 = fig.add_subplot(gs[0, 1])
    wells_clip = None  # Initialize
    
    # Load GRACE data (same approach as pipeline)
    grace_dir = "data/raw/grace"
    grace_files = sorted([f for f in os.listdir(grace_dir) if f.endswith('.tif')])
    
    if grace_files:
        print(f"  📂 Found {len(grace_files)} GRACE files")
        grace_path = os.path.join(grace_dir, grace_files[0])
        
        try:
            # Load and clip GRACE to basin shapefile
            grace_clipped = load_and_clip_raster_to_basin(grace_path, basin_gdf, reference_raster)
            
            if grace_clipped is not None:
                # Get data and extent
                grace_data = grace_clipped.values
                grace_extent = [float(grace_clipped.x.min()), float(grace_clipped.x.max()),
                              float(grace_clipped.y.min()), float(grace_clipped.y.max())]
                
                # Plot GRACE (already clipped to basin shape)
                im2 = ax2.imshow(grace_data, extent=grace_extent, cmap='RdBu_r',
                               alpha=0.8, vmin=-50, vmax=50, origin='upper')
                
                # Colorbar
                cbar2 = plt.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.05,
                                   shrink=0.8, aspect=30)
                cbar2.set_label('GRACE TWS Anomaly (cm)', fontsize=10)
                cbar2.ax.tick_params(labelsize=9)
                
                # Add resolution text
                res_x = abs(float(grace_clipped.x[1] - grace_clipped.x[0]))
                res_y = abs(float(grace_clipped.y[1] - grace_clipped.y[0]))
                ax2.text(0.02, 0.02, f'GRACE Resolution: ~{res_x:.1f}° × {res_y:.1f}°',
                    transform=ax2.transAxes, fontsize=9, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            print("  ✅ GRACE data plotted")
            
        except Exception as e:
            print(f"  ⚠️ Error loading GRACE: {e}")
    
    # Load and plot USGS wells
    wells_path = "data/raw/usgs_well_data/well_metadata.csv"
    if os.path.exists(wells_path):
        print(f"  📂 Loading wells: {wells_path}")
        try:
            wells_df = pd.read_csv(wells_path)
            print(f"  📊 Loaded {len(wells_df)} wells")
            
            # Filter wells to basin geometry using spatial intersection
            wells_gdf = gpd.GeoDataFrame(
                wells_df, 
                geometry=gpd.points_from_xy(wells_df['lon'], wells_df['lat']),
                crs='EPSG:4326'
            )
            
            # Find wells within basin polygon
            wells_in_basin = gpd.sjoin(wells_gdf, basin_gdf, how='inner', predicate='within')
            wells_clip = wells_in_basin  # Store for caption
            print(f"  📍 {len(wells_in_basin)} wells within basin geometry")
            
            # Plot wells
            if len(wells_in_basin) > 0:
                # Scale by data availability if available
                if 'n_months_total' in wells_in_basin.columns:
                    sizes = np.clip(wells_in_basin['n_months_total'].values / 10, 2, 30)
                else:
                    sizes = 5
                
                ax2.scatter(wells_in_basin['lon'], wells_in_basin['lat'],
                          c='red', s=sizes, alpha=0.6, edgecolors='darkred',
                          linewidth=0.5, label='USGS Wells')
                
                # Add density contours for dense areas
                if len(wells_in_basin) > 50:
                    from scipy.stats import gaussian_kde
                    try:
                        xy = np.vstack([wells_in_basin['lon'], wells_in_basin['lat']])
                        z = gaussian_kde(xy)(xy)
                        
                        # Create contours
                        from matplotlib.tri import Triangulation
                        tri = Triangulation(wells_in_basin['lon'], wells_in_basin['lat'])
                        levels = np.percentile(z[z > 0], [50, 75, 90])
                        ax2.tricontour(tri, z, levels=levels, colors='darkred',
                                     linewidths=1, alpha=0.5)
                        print("  ✅ Well density contours added")
                    except Exception as e:
                        print(f"  ⚠️ Could not create density contours: {e}")
                
        except Exception as e:
            print(f"  ⚠️ Error loading wells: {e}")
    
    # Add basin outline
    safe_plot_basin_boundary(basin_gdf, ax2, color='black', linewidth=2)
    
    # Formatting - clean for research paper
    set_basin_axis_limits(ax2, basin_gdf)
    ax2.set_title('GRACE TWS Anomaly and USGS Monitoring Wells', fontsize=14, pad=15, fontweight='bold')
    ax2.text(0.02, 0.98, panel_labels[1], transform=ax2.transAxes,
            fontsize=16, fontweight='bold', va='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax2.axis('off')  # Remove all axis elements for clean look
    
    # ========== PANEL C: Land Cover and Major Aquifers ==========
    print("\n📍 Panel C: Land Cover and Major Aquifers")
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Load MODIS land cover
    modis_dir = "data/raw/modis_land_cover"
    modis_files = sorted([f for f in os.listdir(modis_dir) if f.endswith('.tif')])
    
    if modis_files:
        # Use the latest file
        modis_path = os.path.join(modis_dir, modis_files[-1])
        print(f"  📂 Using MODIS: {os.path.basename(modis_path)}")
        
        try:
            # Load and clip MODIS to basin shapefile
            modis_clipped = load_and_clip_raster_to_basin(modis_path, basin_gdf, reference_raster)
            
            if modis_clipped is not None:
                modis_data = modis_clipped.values
                modis_extent = [float(modis_clipped.x.min()), float(modis_clipped.x.max()),
                              float(modis_clipped.y.min()), float(modis_clipped.y.max())]
                
                # MODIS IGBP land cover colors
                from matplotlib.colors import ListedColormap, BoundaryNorm
                
                lc_colors = [
                    '#05450a',  # 1: Evergreen Needleleaf Forest
                    '#086a10',  # 2: Evergreen Broadleaf Forest
                    '#54a708',  # 3: Deciduous Needleleaf Forest
                    '#78d203',  # 4: Deciduous Broadleaf Forest
                '#009900',  # 5: Mixed Forest
                '#c6b044',  # 6: Closed Shrublands
                '#dcd159',  # 7: Open Shrublands
                '#dade48',  # 8: Woody Savannas
                '#fbff13',  # 9: Savannas
                '#b6ff05',  # 10: Grasslands
                '#27ff87',  # 11: Permanent Wetlands
                '#c24f44',  # 12: Croplands
                '#a5a5a5',  # 13: Urban and Built-Up
                '#ff6d4c',  # 14: Cropland/Natural Vegetation Mosaic
                '#69fff8',  # 15: Snow and Ice
                '#f9ffa4',  # 16: Barren or Sparsely Vegetated
                '#1c0dff'   # 17: Water Bodies
            ]
            
            cmap_lc = ListedColormap(lc_colors)
            bounds_lc = list(range(1, 19))
            norm_lc = BoundaryNorm(bounds_lc, cmap_lc.N)
            
            # Plot land cover
            im3 = ax3.imshow(modis_data, extent=modis_extent, cmap=cmap_lc,
                           norm=norm_lc, interpolation='nearest', origin='upper')
            
            print("  ✅ MODIS land cover plotted")
            
        except Exception as e:
            print(f"  ⚠️ Error loading MODIS: {e}")
    
    # Load Mississippi Basin specific aquifers
    aquifer_dir = "data/shapefiles/processed/aquifers_mississippi"
    aquifer_files = [
        "Coastal_lowlands_aquifer_system_mississippi.shp",
        "High_Plains_aquifer_mississippi.shp", 
        "Mississippi_River_Valley_alluvial_aquifer_mississippi.shp",
        "Mississippi_embayment_aquifer_system_mississippi.shp"
    ]
    
    aquifer_colors = ['#8B008B', '#9370DB', '#4B0082', '#9400D3']  # Violet shades for each aquifer
    aquifer_names = ['Coastal Lowlands', 'High Plains', 'Mississippi Valley', 'Mississippi Embayment']
    
    if os.path.exists(aquifer_dir):
        print(f"  📂 Loading Mississippi Basin aquifers from: {aquifer_dir}")
        
        for i, aquifer_file in enumerate(aquifer_files):
            aquifer_path = os.path.join(aquifer_dir, aquifer_file)
            
            if os.path.exists(aquifer_path):
                try:
                    aquifers = gpd.read_file(aquifer_path)
                    aquifers = aquifers.to_crs('EPSG:4326')
                    
                    # Clip to basin - check for valid geometry first
                    if not basin_gdf.empty and basin_gdf.is_valid.all():
                        aquifers_clip = gpd.clip(aquifers, basin_gdf)
                        
                        # Only plot if we have valid clipped data
                        if not aquifers_clip.empty:
                            aquifers_clip.boundary.plot(ax=ax3, color=aquifer_colors[i], 
                                                       linewidth=2, alpha=0.9, linestyle='-')
                            print(f"  ✅ {aquifer_names[i]} aquifer added")
                        else:
                            print(f"  ⚠️ No {aquifer_names[i]} aquifer found within basin")
                    else:
                        print("  ⚠️ Invalid basin geometry, skipping aquifer clipping")
                        
                except Exception as e:
                    print(f"  ⚠️ Error loading {aquifer_names[i]} aquifer: {e}")
            else:
                print(f"  ⚠️ Aquifer file not found: {aquifer_file}")
    else:
        print(f"  ⚠️ Aquifer directory not found: {aquifer_dir}")
    
    # Add basin outline
    safe_plot_basin_boundary(basin_gdf, ax3, color='black', linewidth=2)
    
    # Create simplified legend with aquifers
    legend_elements = [
        mpatches.Patch(color='#78d203', label='Forest'),
        mpatches.Patch(color='#b6ff05', label='Grassland'),
        mpatches.Patch(color='#c24f44', label='Cropland'),
        mpatches.Patch(color='#a5a5a5', label='Urban'),
        mpatches.Patch(color='#1c0dff', label='Water'),
        mlines.Line2D([0], [0], color='#8B008B', linewidth=2, label='Coastal Lowlands'),
        mlines.Line2D([0], [0], color='#9370DB', linewidth=2, label='High Plains'),
        mlines.Line2D([0], [0], color='#4B0082', linewidth=2, label='Mississippi Valley'),
        mlines.Line2D([0], [0], color='#9400D3', linewidth=2, label='Mississippi Embayment')
    ]
    ax3.legend(handles=legend_elements, loc='lower right', framealpha=0.9,
              fontsize=9, ncol=3)
    
    # Formatting - clean for research paper
    set_basin_axis_limits(ax3, basin_gdf)
    ax3.set_title('Land Cover and Major Aquifers', fontsize=14, pad=15, fontweight='bold')
    ax3.text(0.02, 0.98, panel_labels[2], transform=ax3.transAxes,
            fontsize=16, fontweight='bold', va='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax3.axis('off')  # Remove all axis elements for clean look
    
    # ========== PANEL D: Precipitation and Population ==========
    print("\n📍 Panel D: Mean Annual Precipitation and Population Density")
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Load CHIRPS precipitation
    chirps_dir = "data/raw/chirps"
    chirps_files = sorted([f for f in os.listdir(chirps_dir) if f.endswith('.tif')])
    
    if chirps_files:
        print(f"  📂 Processing {len(chirps_files)} CHIRPS files")
        
        try:
            # Calculate annual mean from monthly files
            monthly_sum = None
            n_months = min(12, len(chirps_files))  # Use one year
            chirps_extent = None
            
            for i in range(n_months):
                chirps_path = os.path.join(chirps_dir, chirps_files[i])
                
                # Load and clip each monthly file to basin
                chirps_clipped = load_and_clip_raster_to_basin(chirps_path, basin_gdf, reference_raster)
                
                if chirps_clipped is not None:
                    if monthly_sum is None:
                        monthly_sum = chirps_clipped.values
                        chirps_extent = [float(chirps_clipped.x.min()), 
                                       float(chirps_clipped.x.max()),
                                       float(chirps_clipped.y.min()), 
                                       float(chirps_clipped.y.max())]
                    else:
                        monthly_sum += chirps_clipped.values
            
            # Annual total
            annual_precip = monthly_sum
            
            # Plot precipitation
            vmin = np.nanpercentile(annual_precip[~np.isnan(annual_precip)], 5)
            vmax = np.nanpercentile(annual_precip[~np.isnan(annual_precip)], 95)
            
            im4 = ax4.imshow(annual_precip, extent=chirps_extent, cmap='Blues',
                           alpha=0.8, vmin=vmin, vmax=vmax, origin='upper')
            
            # Colorbar
            cbar4 = plt.colorbar(im4, ax=ax4, orientation='horizontal', pad=0.12,
                               shrink=0.8, aspect=30)
            cbar4.set_label('Annual Precipitation (mm)', fontsize=10)
            cbar4.ax.tick_params(labelsize=9)
            
            print("  ✅ Precipitation plotted")
            
        except Exception as e:
            print(f"  ⚠️ Error processing CHIRPS: {e}")
    
    # Load population data
    landscan_dir = "data/raw/landscan"
    landscan_files = sorted([f for f in os.listdir(landscan_dir) if f.endswith('.tif')])
    
    if landscan_files:
        landscan_path = os.path.join(landscan_dir, landscan_files[-1])
        print(f"  📂 Loading population: {os.path.basename(landscan_path)}")
        
        try:
            # Load and clip population data to basin shapefile
            pop_clipped = load_and_clip_raster_to_basin(landscan_path, basin_gdf, reference_raster)
            
            if pop_clipped is not None:
                pop_data = pop_clipped.values
            pop_extent = [float(pop_clipped.x.min()), float(pop_clipped.x.max()),
                        float(pop_clipped.y.min()), float(pop_clipped.y.max())]
            
            # Log transform for visualization
            pop_log = np.log10(pop_data + 1)
            
            # Urban areas (>200 people/km²)
            urban_threshold = np.log10(200)
            urban_mask = pop_log > urban_threshold
            
            # Plot urban areas
            from matplotlib.colors import ListedColormap
            urban_cmap = ListedColormap(['none', 'red'])
            ax4.imshow(urban_mask.astype(int), extent=pop_extent, cmap=urban_cmap,
                     alpha=0.5, interpolation='nearest', origin='upper')
            
            print("  ✅ Urban areas (>200 people/km²) plotted")
            
            print("  ✅ Population density plotted")
            
        except Exception as e:
            print(f"  ⚠️ Error loading population: {e}")
    
    # Add basin outline
    safe_plot_basin_boundary(basin_gdf, ax4, color='black', linewidth=2)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='red', alpha=0.5, label='Urban Areas (>200 people/km²)'),
        mpatches.Patch(color='blue', alpha=0.6, label='Precipitation Gradient')
    ]
    ax4.legend(handles=legend_elements, loc='lower left', framealpha=0.9, fontsize=10)
    
    # Formatting - clean for research paper
    set_basin_axis_limits(ax4, basin_gdf)
    ax4.set_title('Annual Precipitation and Population Centers', fontsize=14, pad=15, fontweight='bold')
    ax4.text(0.02, 0.98, panel_labels[3], transform=ax4.transAxes,
            fontsize=16, fontweight='bold', va='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax4.axis('off')  # Remove all axis elements for clean look
    
    # Save figure - clean research paper style
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "study_area_figure_AGU.png")
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Figure saved to: {output_path}")
    
    # Also save as PDF
    pdf_path = os.path.join(output_dir, "study_area_figure_AGU.pdf")
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white', format='pdf')
    print(f"✅ PDF version saved to: {pdf_path}")
    
    plt.close()
    
    # Create figure caption
    n_wells_text = str(len(wells_clip)) if wells_clip is not None else 'N/A'
    
    caption = f"""
Figure 1. Mississippi River Basin study area showing integrated datasets for GRACE downscaling analysis. 
(A) Topographic relief from SRTM DEM with state boundaries and basin outline. 
(B) GRACE TWS anomaly data with USGS groundwater monitoring wells (n={n_wells_text}); 
    red shading indicates well density. 
(C) MODIS land cover classification and Mississippi Basin aquifer systems (Coastal Lowlands, High Plains, Mississippi Valley, and Mississippi Embayment). 
(D) Mean annual precipitation from CHIRPS with urban areas (>200 people/km²) from LandScan population data.
All data clipped to Mississippi River Basin boundary. Scale bar and north arrow shown in panel A apply to all panels.
"""
    
    caption_path = os.path.join(output_dir, "study_area_figure_caption.txt")
    with open(caption_path, 'w') as f:
        f.write(caption)
    print(f"✅ Caption saved to: {caption_path}")
    
    print("\n🎉 Study area figure creation complete!")
    print("="*60)

if __name__ == "__main__":
    create_study_area_figure()