#!/usr/bin/env python3
"""
Monthly TWS Visualization Script for Downscaled GRACE Data

Creates a publication-quality 3x4 subplot layout showing average TWS for each month (JAN-DEC)
with NetCDF data values precisely clipped to Mississippi River Basin boundary.

Usage:
    python plot_monthly_tws.py
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
# import matplotlib as mpl  # Not used
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)
sns.set_palette("husl")

def load_mrb_boundary():
    """Load Mississippi River Basin boundary."""
    mrb_file = Path("/home/sauravbhattarai/Documents/ORISE/GroundWater-Downscaling/data/shapefiles/MRB.geojson")
    
    if mrb_file.exists():
        print(f"✅ Loading MRB boundary from: {mrb_file}")
        gdf = gpd.read_file(mrb_file)
        print(f"   Features: {len(gdf)}")
        print(f"   Columns: {list(gdf.columns)}")
        
        # If multiple features, combine into single geometry
        if len(gdf) > 1:
            combined_geometry = unary_union(gdf.geometry)
            print(f"   Combined {len(gdf)} features into single MRB boundary")
            return combined_geometry
        else:
            return gdf.geometry.iloc[0]
    else:
        print("⚠️ MRB boundary file not found, creating approximate boundary")
        mrb_coords = [
            (-89.0, 47.8), (-95.2, 48.9), (-103.0, 45.8), (-106.8, 40.5),
            (-108.0, 37.0), (-106.0, 32.0), (-100.0, 29.5), (-94.0, 28.8),
            (-89.2, 29.1), (-85.0, 30.2), (-82.5, 32.0), (-81.0, 35.5),
            (-80.5, 39.0), (-79.8, 40.8), (-80.2, 42.0), (-82.8, 43.5),
            (-84.5, 45.2), (-87.0, 46.8), (-89.0, 47.8)
        ]
        return Polygon(mrb_coords)

# Removed unused function get_mrb_states

def create_precise_mask(lats, lons, mrb_polygon):
    """Create a precise mask for MRB boundary with no leakage."""
    print("🎯 Creating precise MRB mask...")
    
    # Create coordinate arrays
    lon_2d, lat_2d = np.meshgrid(lons, lats)
    
    # Vectorized point-in-polygon test with high precision
    mask = np.zeros_like(lon_2d, dtype=bool)
    
    # Process in chunks for better performance
    chunk_size = 1000
    total_points = lon_2d.size
    processed = 0
    
    for i in range(0, len(lats), chunk_size):
        for j in range(0, len(lons), chunk_size):
            i_end = min(i + chunk_size, len(lats))
            j_end = min(j + chunk_size, len(lons))
            
            for ii in range(i, i_end):
                for jj in range(j, j_end):
                    point = Point(lon_2d[ii, jj], lat_2d[ii, jj])
                    # Use more precise containment check
                    mask[ii, jj] = mrb_polygon.contains(point) or mrb_polygon.touches(point)
                    processed += 1
        
        if processed % 5000 == 0:
            print(f"   Progress: {processed:,}/{total_points:,} ({100*processed/total_points:.1f}%)")
    
    print(f"   Points inside MRB: {mask.sum():,} / {mask.size:,} ({100*mask.sum()/mask.size:.1f}%)")
    return mask

def mask_data_with_mrb(ds, mrb_polygon):
    """Precisely mask NetCDF data with MRB boundary - no color leakage."""
    print("🗺️ Precisely masking data with MRB boundary...")
    
    # Get coordinates
    lats = ds.lat.values
    lons = ds.lon.values
    
    # Create precise mask
    mask = create_precise_mask(lats, lons, mrb_polygon)
    
    # Apply mask to dataset
    ds_masked = ds.copy()
    for var_name in ds.data_vars:
        data = ds[var_name].values.copy()
        
        if data.ndim == 3:  # (time, lat, lon)
            # Apply mask to all time steps
            for t in range(data.shape[0]):
                data[t, ~mask] = np.nan
        elif data.ndim == 2:  # (lat, lon)
            data[~mask] = np.nan
        
        # Use masked array for better handling
        ds_masked[var_name].values = np.ma.masked_invalid(data)
    
    return ds_masked, mask

def create_publication_quality_plot():
    """Create publication-quality monthly TWS visualization."""
    
    print("📊 Creating publication-quality monthly TWS visualization...")
    
    # Load data
    grace_path = "results_coarse_to_fine/grace_downscaled_5km.nc"
    print(f"📂 Loading data from: {grace_path}")
    ds = xr.open_dataset(grace_path)
    
    # Get TWS variable
    tws_var = None
    for var_name in ['tws_anomaly', 'grace_downscaled', 'downscaled_grace', 'tws', 'grace']:
        if var_name in ds:
            tws_var = var_name
            break
    
    if tws_var is None:
        tws_var = list(ds.data_vars)[0]
        print(f"⚠️ Using variable: {tws_var}")
    else:
        print(f"✅ Using TWS variable: {tws_var}")
    
    # Load MRB boundary and mask data
    mrb_polygon = load_mrb_boundary()
    ds_masked, mask = mask_data_with_mrb(ds, mrb_polygon)
    
    # Calculate monthly climatology
    ds_monthly = ds_masked.groupby(ds_masked.time.dt.month).mean()
    tws_monthly = ds_monthly[tws_var]
    
    # Get time info
    times = pd.to_datetime(ds.time.values)
    print(f"📊 Data shape: {tws_monthly.shape}")
    print(f"📅 Time range: {times[0].strftime('%Y-%m')} to {times[-1].strftime('%Y-%m')}")
    
    # Set up publication-quality figure
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor('white')
    
    # Define projection
    proj = ccrs.PlateCarree()
    
    # Calculate extent from MRB bounds
    mrb_bounds = mrb_polygon.bounds
    lon_pad = (mrb_bounds[2] - mrb_bounds[0]) * 0.05
    lat_pad = (mrb_bounds[3] - mrb_bounds[1]) * 0.05
    extent = [mrb_bounds[0] - lon_pad, mrb_bounds[2] + lon_pad, 
              mrb_bounds[1] - lat_pad, mrb_bounds[3] + lat_pad]
    
    # Calculate colorbar limits from valid data only
    valid_data = []
    for i in range(12):
        month_data = tws_monthly.isel(month=i).values
        valid_month = month_data[~np.isnan(month_data)]
        valid_data.extend(valid_month)
    
    valid_data = np.array(valid_data)
    vmin = np.percentile(valid_data, 2)
    vmax = np.percentile(valid_data, 98)
    
    # Make symmetric around zero for better interpretation
    vmax_abs = max(abs(vmin), abs(vmax))
    vmin, vmax = -vmax_abs, vmax_abs
    
    print(f"🎨 Color range: {vmin:.1f} to {vmax:.1f} cm (symmetric, MRB only)")
    
    # Use better colormap: blue for positive (more water), red for negative (less water)
    cmap = plt.cm.RdBu  # Already correctly oriented: red=negative, blue=positive
    
    # Month names
    month_names = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                   'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    
    # Create subplots with minimal spacing
    for i in range(12):
        ax = plt.subplot(3, 4, i + 1, projection=proj)
        
        # Set extent
        ax.set_extent(extent, crs=proj)
        
        # Clean axis appearance for publication quality
        # Remove tick marks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        
        # For cartopy GeoAxes, set clean appearance
        if hasattr(ax, 'patch'):
            ax.patch.set_visible(False)  # Remove background
        
        # Try to remove spines if they exist
        try:
            for spine_name in ['top', 'right', 'bottom', 'left']:
                if spine_name in ax.spines:
                    ax.spines[spine_name].set_visible(False)
        except:
            pass
        
        # Add only MRB states (clipped to MRB region)
        states_feature = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lakes',
            scale='50m',
            facecolor='none',
            edgecolor='gray',
            linewidth=0.5,
            alpha=0.6
        )
        ax.add_feature(states_feature)
        
        # Add subtle geographic features
        ax.add_feature(cfeature.RIVERS, linewidth=0.3, alpha=0.4, color='steelblue')
        ax.add_feature(cfeature.LAKES, linewidth=0.2, alpha=0.3, facecolor='lightblue', edgecolor='steelblue')
        
        # Get month data and apply mask more precisely
        month_data = tws_monthly.isel(month=i).values.copy()
        month_data[~mask] = np.nan
        
        # Create masked array to prevent color leakage
        month_data_masked = np.ma.masked_invalid(month_data)
        
        # Create coordinate grids
        lons, lats = np.meshgrid(ds.lon, ds.lat)
        
        # Plot with precise masking
        im = ax.pcolormesh(lons, lats, month_data_masked,
                          transform=proj, 
                          cmap=cmap,
                          vmin=vmin, vmax=vmax,
                          shading='auto',
                          alpha=0.9,
                          rasterized=True)  # Better for PDF output
        
        # Add MRB boundary with clean line
        if hasattr(mrb_polygon, 'exterior'):
            coords = list(mrb_polygon.exterior.coords)
            lons_boundary = [coord[0] for coord in coords]
            lats_boundary = [coord[1] for coord in coords]
            ax.plot(lons_boundary, lats_boundary, 'k-', linewidth=1.5, 
                   transform=proj, alpha=0.8, zorder=10)
        
        # Add month label with clean styling
        ax.text(0.03, 0.03, month_names[i], 
               transform=ax.transAxes, 
               fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        alpha=0.9, edgecolor='none'))
    
    # Add publication-quality colorbar with minimal spacing
    cbar_ax = fig.add_axes([0.2, 0.04, 0.6, 0.04])  # Much smaller bottom margin
    cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal', extend='both')
    
    # Enhanced colorbar styling
    cbar.set_label('Total Water Storage Anomaly (cm)', fontsize=16, fontweight='bold', labelpad=10)
    cbar.ax.tick_params(labelsize=14, width=1.5, length=6)
    
    # Add colorbar ticks for better readability
    tick_values = np.linspace(vmin, vmax, 9)
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f'{val:.1f}' for val in tick_values])
    
    # Adjust layout for publication quality - minimal spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.96, bottom=0.10, left=0.02, right=0.98, 
                       hspace=-0.1, wspace=0.02)  # Reduced hspace from 0.05 to 0.02
    
    # Save with high quality
    output_path = Path("figures_coarse_to_fine/monthly_tws_mrb_publication.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white',
                edgecolor='none', transparent=False, format='png')
    print(f"💾 Saved publication-quality figure: {output_path}")
    
    # Save as high-quality PDF
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white',
                edgecolor='none', transparent=False, format='pdf')
    print(f"💾 Saved publication-quality PDF: {pdf_path}")
    
    return output_path

def main():
    """Main function."""
    print("🎨 Creating publication-quality Monthly TWS Visualization...")
    output_path = create_publication_quality_plot()
    print(f"✅ Publication-quality visualization complete!")
    print(f"📄 Ready for research paper: {output_path}")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())