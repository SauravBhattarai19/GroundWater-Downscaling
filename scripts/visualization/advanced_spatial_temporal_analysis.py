#!/usr/bin/env python3
"""
Advanced Spatial-Temporal Analysis and Visualization for GRACE Groundwater Downscaling (ENHANCED)
===============================================================================================

This script creates comprehensive publication-quality figures including:
1. Spatial trend maps with statistical significance (hatching)
2. Multi-component analysis (GWS, TWS, precipitation, snow, soil moisture)
3. Regional aggregation using shapefiles
4. Timing of extremes analysis
5. Comprehensive trend analysis at multiple scales

ENHANCED: Incorporates robust visualization techniques from simple_robust_visualization.py

Author: GRACE Analysis Pipeline
Date: 2024
"""

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
import geopandas as gpd
from shapely.geometry import mapping
import rioxarray
from scipy import stats
from tqdm import tqdm
import warnings
from pathlib import Path
from datetime import datetime
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm, ListedColormap
from matplotlib import cm
try:
    import regionmask
except ImportError:
    regionmask = None
    warnings.warn("⚠️ regionmask not installed; shapefile-based masking will be disabled")

warnings.filterwarnings('ignore')

# Set style for publication quality
plt.style.use('default')
sns.set_palette("husl")

# Publication settings
FIGURE_DPI = 300
FONT_SIZE = 12


class AdvancedGRACEVisualizer:
    """Enhanced visualization class for GRACE groundwater analysis with robust plotting."""
    
    def __init__(self, base_dir=".", shapefile_path=None):
        """
        Initialize the visualizer.
        
        Parameters:
        -----------
        base_dir : str
            Base directory of the project
        shapefile_path : str
            Path to Mississippi River Basin shapefile
        """
        self.base_dir = Path(base_dir)
        self.figures_dir = self.base_dir / "figures" / "advanced_enhanced"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.subdirs = {
            'trends': self.figures_dir / 'trends',
            'regional': self.figures_dir / 'regional',
            'extremes': self.figures_dir / 'extremes',
            'components': self.figures_dir / 'components',
            'seasonal': self.figures_dir / 'seasonal',
            'overview': self.figures_dir / 'overview'
        }
        
        for subdir in self.subdirs.values():
            subdir.mkdir(exist_ok=True)
        
        # Load shapefile if provided
        self.shapefile = None
        self.shapefile_bounds = None
        self.region_mask = None
        
        if shapefile_path is None:
            # Try default shapefile path
            default_shapefile = self.base_dir / "data/shapefiles/processed/mississippi_river_basin.shp"
            if default_shapefile.exists():
                shapefile_path = str(default_shapefile)
        
        if shapefile_path and os.path.exists(shapefile_path):
            print(f"📍 Loading shapefile: {shapefile_path}")
            self.shapefile = gpd.read_file(shapefile_path)
            
            # Store original bounds before any transformation
            original_bounds = self.shapefile.total_bounds
            print(f"   Original bounds: {original_bounds}")
            print(f"   Original CRS: {self.shapefile.crs}")
            
            # Ensure CRS is WGS84
            if self.shapefile.crs != 'EPSG:4326':
                print(f"   Converting from {self.shapefile.crs} to EPSG:4326...")
                try:
                    # Try conversion with error checking
                    self.shapefile = self.shapefile.to_crs('EPSG:4326')
                    
                    # Validate conversion worked
                    test_bounds = self.shapefile.total_bounds
                    if not np.all(np.isfinite(test_bounds)):
                        print("   ⚠️ CRS conversion resulted in invalid bounds, keeping original CRS")
                        # Reload with original CRS
                        self.shapefile = gpd.read_file(shapefile_path)
                        print(f"   Keeping original CRS: {self.shapefile.crs}")
                    else:
                        print(f"   ✅ CRS conversion successful")
                        
                except Exception as e:
                    print(f"   ❌ CRS conversion failed: {e}, keeping original CRS")
                    # Reload with original CRS
                    self.shapefile = gpd.read_file(shapefile_path)
            
            # Calculate bounds safely with multiple fallback methods
            try:
                # Method 1: Use total_bounds
                bounds = self.shapefile.total_bounds
                
                # Check if bounds are valid
                if not np.all(np.isfinite(bounds)):
                    print("   ⚠️ Invalid bounds from total_bounds, trying alternative method...")
                    
                    # Method 2: Calculate from individual geometry bounds
                    try:
                        bounds_df = self.shapefile.geometry.bounds
                        minx = bounds_df.minx.min()
                        miny = bounds_df.miny.min()
                        maxx = bounds_df.maxx.max()
                        maxy = bounds_df.maxy.max()
                        bounds = np.array([minx, miny, maxx, maxy])
                        
                        # Check if these bounds are valid
                        if not np.all(np.isfinite(bounds)):
                            print("   ⚠️ Alternative bounds method also failed, trying envelope method...")
                            
                            # Method 3: Use envelope of all geometries
                            envelope = self.shapefile.geometry.unary_union.envelope
                            minx, miny, maxx, maxy = envelope.bounds
                            bounds = np.array([minx, miny, maxx, maxy])
                            
                            if not np.all(np.isfinite(bounds)):
                                print("   ⚠️ All bounds methods failed, using original bounds before conversion")
                                bounds = original_bounds
                                
                    except Exception as e2:
                        print(f"   ⚠️ Alternative bounds calculation failed: {e2}, using original bounds")
                        bounds = original_bounds
                
                # Validate final bounds
                if np.all(np.isfinite(bounds)) and bounds[2] > bounds[0] and bounds[3] > bounds[1]:
                    self.shapefile_bounds = bounds
                    print(f"   ✅ Final bounds: {self.shapefile_bounds}")
                else:
                    print("   ❌ Final bounds validation failed, using original bounds")
                    self.shapefile_bounds = original_bounds
                
            except Exception as e:
                print(f"   ❌ Error calculating bounds: {e}, using original bounds")
                self.shapefile_bounds = original_bounds
                
        else:
            print("⚠️ No shapefile found, will use full domain")
        
        # Load all datasets
        self._load_all_data()
    
    def _get_safe_extent(self, data_array, buffer=0.5):
        """Get a safe extent for plotting, handling shapefile bounds issues."""
        # Default to data extent
        data_extent = [
            float(data_array.lon.min()), 
            float(data_array.lon.max()),
            float(data_array.lat.min()), 
            float(data_array.lat.max())
        ]
        
        if self.shapefile is not None and self.shapefile_bounds is not None:
            # Check if shapefile bounds are valid
            if np.all(np.isfinite(self.shapefile_bounds)):
                # Use shapefile bounds with buffer
                return [
                    self.shapefile_bounds[0] - buffer,
                    self.shapefile_bounds[2] + buffer,
                    self.shapefile_bounds[1] - buffer,
                    self.shapefile_bounds[3] + buffer
                ]
        
        # Fallback to data extent
        return data_extent
    
    def _add_map_features(self, ax, add_shapefile=True):
        """Add consistent map features to an axis (robust method from simple script)."""
        # Add basic features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.STATES, linewidth=0.5, alpha=0.7)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.7)
        
        # Add shapefile boundary if available and requested
        if add_shapefile and self.shapefile is not None:
            try:
                # Try different methods to plot shapefile boundary
                
                # Method 1: Direct boundary plot
                try:
                    self.shapefile.boundary.plot(ax=ax, color='black', linewidth=1.5,
                                               transform=ccrs.PlateCarree())
                except Exception as e1:
                    print(f"    ⚠️ Direct boundary plot failed: {e1}")
                    
                    # Method 2: Manual geometry plotting
                    try:
                        for idx, row in self.shapefile.iterrows():
                            geom = row.geometry
                            if geom.is_valid:
                                if geom.type == 'Polygon':
                                    x, y = geom.exterior.xy
                                    ax.plot(x, y, 'k-', linewidth=1.5, transform=ccrs.PlateCarree())
                                elif geom.type == 'MultiPolygon':
                                    for poly in geom.geoms:
                                        if poly.is_valid:
                                            x, y = poly.exterior.xy
                                            ax.plot(x, y, 'k-', linewidth=1.5, transform=ccrs.PlateCarree())
                    except Exception as e2:
                        print(f"    ⚠️ Manual geometry plot failed: {e2}")
                        
                        # Method 3: Skip shapefile plotting
                        print(f"    ⚠️ All shapefile plotting methods failed, continuing without boundary")
                
            except Exception as e:
                print(f"    ⚠️ Could not plot shapefile boundary: {e}")
        
        # Add gridlines
        try:
            gl = ax.gridlines(draw_labels=True, alpha=0.5)
            gl.top_labels = False
            gl.right_labels = False
        except Exception as e:
            print(f"    ⚠️ Could not add gridlines: {e}")
            # Add basic gridlines without labels
            gl = ax.gridlines(alpha=0.5)
        
        return gl
    
    def _get_robust_colorscale(self, data, percentiles=(5, 95), symmetric=False):
        """Get robust color scale using percentiles (from simple script)."""
        valid_data = data[~np.isnan(data)]
        if len(valid_data) == 0:
            return 0, 1
        
        if symmetric:
            # For symmetric data (like anomalies)
            vmax = np.percentile(np.abs(valid_data), percentiles[1])
            if vmax == 0:
                vmax = 1
            return -vmax, vmax
        else:
            # For regular data
            vmin, vmax = np.percentile(valid_data, percentiles)
            if vmin == vmax:
                vmin, vmax = vmin - 1, vmax + 1
            return vmin, vmax
    
    def _load_all_data(self):
        """Load all required datasets."""
        print("\n📦 Loading all datasets...")
        
        # 1. Load groundwater data
        gws_files = [
            "results/groundwater_storage_anomalies.nc",
            "results/groundwater_storage_anomalies_corrected.nc",
            "results/groundwater_storage_anomalies_enhanced.nc"
        ]
        
        for gws_file in gws_files:
            if (self.base_dir / gws_file).exists():
                print(f"  ✅ Loading groundwater: {gws_file}")
                self.gws_ds = xr.open_dataset(self.base_dir / gws_file)
                break
        else:
            raise FileNotFoundError("No groundwater storage file found!")
        
        # Print data info for debugging
        print(f"     Dataset shape: {self.gws_ds.groundwater.shape}")
        print(f"     Valid data points: {np.sum(~np.isnan(self.gws_ds.groundwater.values))}")
        print(f"     Data range: {float(np.nanmin(self.gws_ds.groundwater.values)):.2f} to {float(np.nanmax(self.gws_ds.groundwater.values)):.2f}")
        
        # 2. Load feature stack for additional components
        feature_file = "data/processed/feature_stack.nc"
        if (self.base_dir / feature_file).exists():
            print(f"  ✅ Loading features: {feature_file}")
            self.features_ds = xr.open_dataset(self.base_dir / feature_file)
        else:
            print("  ⚠️ Feature stack not found")
            self.features_ds = None
        
        # 3. Create region mask if shapefile is loaded
        if self.shapefile is not None:
            self._create_region_mask()
    
    def _create_region_mask(self):
        """Create a mask for the shapefile region."""
        print("  🎭 Creating region mask from shapefile...")
        
        # Print some debugging info
        print(f"    Shapefile bounds: {self.shapefile_bounds}")
        print(f"    Shapefile CRS: {self.shapefile.crs}")
        print(f"    Data lon range: {float(self.gws_ds.lon.min())} to {float(self.gws_ds.lon.max())}")
        print(f"    Data lat range: {float(self.gws_ds.lat.min())} to {float(self.gws_ds.lat.max())}")
        
        if regionmask is None:
            print("    ⚠️ regionmask not available, trying basic mask creation")
            self.region_mask = self._create_basic_mask()
            return
        
        try:
            # Validate shapefile geometry first
            if not self.shapefile.is_valid.all():
                print("    ⚠️ Some geometries are invalid, attempting to fix...")
                self.shapefile['geometry'] = self.shapefile['geometry'].buffer(0)
            
            # Create a mask using regionmask
            if len(self.shapefile) == 1:
                # Single polygon
                geom = self.shapefile.geometry.iloc[0]
            else:
                # Multiple polygons - union them
                try:
                    # Try union_all() for newer geopandas versions
                    geom = self.shapefile.union_all()
                except AttributeError:
                    # Fall back to unary_union for older versions
                    geom = self.shapefile.geometry.unary_union
            
            # Check if geometry is valid
            if not geom.is_valid:
                print("    ⚠️ Geometry is invalid, attempting to fix...")
                geom = geom.buffer(0)
            
            region = regionmask.Regions([geom], names=['Mississippi_Basin'])
            
            # Create mask for the groundwater grid
            self.region_mask = region.mask(self.gws_ds.lon, self.gws_ds.lat)
            
            # Count valid pixels
            n_valid = np.sum(~np.isnan(self.region_mask))
            print(f"    ✅ regionmask created: {n_valid} valid pixels")
            
            # If regionmask failed to create valid pixels, try basic mask
            if n_valid == 0:
                print("    ⚠️ regionmask created 0 valid pixels, trying basic mask...")
                self.region_mask = self._create_basic_mask()
            
        except Exception as e:
            print(f"    ⚠️ Error creating regionmask: {e}, trying basic mask...")
            self.region_mask = self._create_basic_mask()
    
    def _create_basic_mask(self):
        """Create a basic mask using point-in-polygon testing."""
        try:
            from shapely.geometry import Point
            
            print("    🔧 Creating basic mask using point-in-polygon...")
            
            # Get data coordinates
            lon_1d = self.gws_ds.lon.values
            lat_1d = self.gws_ds.lat.values
            
            # Create coordinate grids (lon x lat)
            lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d)
            
            # Union all geometries
            if len(self.shapefile) == 1:
                union_geom = self.shapefile.geometry.iloc[0]
            else:
                try:
                    union_geom = self.shapefile.union_all()
                except AttributeError:
                    union_geom = self.shapefile.geometry.unary_union
            
            # Check if geometry is valid
            if not union_geom.is_valid:
                union_geom = union_geom.buffer(0)
            
            # Initialize mask
            mask = np.full(lon_grid.shape, np.nan)
            
            # Test each grid point (with progress for large grids)
            n_lat, n_lon = len(lat_1d), len(lon_1d)
            n_inside = 0
            
            for i in range(n_lat):
                for j in range(n_lon):
                    point = Point(lon_grid[i, j], lat_grid[i, j])
                    if union_geom.contains(point) or union_geom.touches(point):
                        mask[i, j] = 0.0  # Valid pixel
                        n_inside += 1
            
            print(f"    ✅ Basic mask created: {n_inside} valid pixels")
            return mask
            
        except Exception as e:
            print(f"    ❌ Basic mask creation failed: {e}")
            return None
    
    def calculate_pixel_trends(self, data_array, min_years=5):
        """
        Calculate linear trends for each pixel with statistical significance.
        
        Parameters:
        -----------
        data_array : xr.DataArray
            3D array (time, lat, lon)
        min_years : int
            Minimum years of data required for trend calculation
        
        Returns:
        --------
        dict with trend, p_value, std_error arrays
        """
        print(f"  📈 Calculating pixel-wise trends...")
        
        # Get dimensions
        n_time = len(data_array.time)
        n_lat = len(data_array.lat)
        n_lon = len(data_array.lon)
        
        # Initialize output arrays
        trend = np.full((n_lat, n_lon), np.nan)
        p_value = np.full((n_lat, n_lon), np.nan)
        std_error = np.full((n_lat, n_lon), np.nan)
        n_valid = np.full((n_lat, n_lon), 0)
        
        # Time array for regression (months since start)
        time_numeric = np.arange(n_time)
        
        # Progress bar for large grids
        total_pixels = n_lat * n_lon
        with tqdm(total=total_pixels, desc="Calculating trends") as pbar:
            for i in range(n_lat):
                for j in range(n_lon):
                    # Extract time series for this pixel
                    ts = data_array[:, i, j].values
                    
                    # Check for sufficient valid data
                    valid_mask = ~np.isnan(ts)
                    n_valid_points = np.sum(valid_mask)
                    
                    if n_valid_points >= min_years * 12:  # Monthly data
                        # Perform linear regression
                        try:
                            slope, intercept, r_value, p_val, std_err = stats.linregress(
                                time_numeric[valid_mask], ts[valid_mask]
                            )
                            
                            # Convert to annual trend (slope is per month)
                            trend[i, j] = slope * 12
                            p_value[i, j] = p_val
                            std_error[i, j] = std_err * 12
                            n_valid[i, j] = n_valid_points
                            
                        except Exception:
                            pass
                    
                    pbar.update(1)
        
        return {
            'trend': trend,
            'p_value': p_value,
            'std_error': std_error,
            'n_valid': n_valid
        }
    
    def create_trend_map_with_significance(self, data_array, variable_name, 
                                         units='cm/year', clip_to_shapefile=True):
        """
        Create trend map with statistical significance hatching.
        
        This is the implementation of Figure 2 mentioned in the brainstorming.
        """
        print(f"\n🎨 Creating trend map for {variable_name}")
        
        # Calculate trends
        trend_results = self.calculate_pixel_trends(data_array)
        
        # Apply shapefile mask if requested
        if clip_to_shapefile and self.region_mask is not None:
            for key in trend_results:
                trend_results[key] = np.where(
                    ~np.isnan(self.region_mask), 
                    trend_results[key], 
                    np.nan
                )
        
        # Create figure
        fig = plt.figure(figsize=(12, 10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Set extent using safe method
        extent = self._get_safe_extent(data_array)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        
        # Plot trend using robust color scaling
        vmin, vmax = self._get_robust_colorscale(trend_results['trend'], 
                                                percentiles=(5, 95), symmetric=True)
        
        print(f"    Trend color scale: {vmin:.3f} to {vmax:.3f} {units}")
        
        im = ax.pcolormesh(data_array.lon, data_array.lat, 
                          trend_results['trend'],
                          cmap='RdBu_r', vmin=vmin, vmax=vmax,
                          transform=ccrs.PlateCarree())
        
        # Add significance hatching
        # Create masks for different significance levels
        sig_01 = trend_results['p_value'] < 0.01
        sig_05 = (trend_results['p_value'] >= 0.01) & (trend_results['p_value'] < 0.05)
        sig_10 = (trend_results['p_value'] >= 0.05) & (trend_results['p_value'] < 0.10)
        
        # Add hatching for each significance level
        lon_mesh, lat_mesh = np.meshgrid(data_array.lon, data_array.lat)
        
        # Dense hatching for p < 0.01
        ax.contourf(lon_mesh, lat_mesh, sig_01.astype(float), 
                   levels=[0.5, 1.5], colors='none', 
                   hatches=['///'], transform=ccrs.PlateCarree())
        
        # Medium hatching for p < 0.05
        ax.contourf(lon_mesh, lat_mesh, sig_05.astype(float), 
                   levels=[0.5, 1.5], colors='none', 
                   hatches=[r'\\'], transform=ccrs.PlateCarree())
        
        # Light hatching for p < 0.10
        ax.contourf(lon_mesh, lat_mesh, sig_10.astype(float), 
                   levels=[0.5, 1.5], colors='none', 
                   hatches=['...'], transform=ccrs.PlateCarree())
        
        # Add features using robust method
        gl = self._add_map_features(ax, add_shapefile=clip_to_shapefile)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                           pad=0.05, shrink=0.8)
        cbar.set_label(f'{variable_name} Trend ({units})', fontsize=FONT_SIZE)
        
        # Title
        time_range = f"{str(data_array.time.values[0])[:4]}-{str(data_array.time.values[-1])[:4]}"
        plt.title(f'{variable_name} Linear Trend ({time_range})\n' + 
                 r'Hatching: /// p<0.01, \\\ p<0.05, ... p<0.10', 
                 fontsize=FONT_SIZE+2, pad=20)
        
        # Add legend for hatching
        legend_elements = [
            mpatches.Patch(facecolor='white', edgecolor='black', 
                          hatch='///', label='p < 0.01'),
            mpatches.Patch(facecolor='white', edgecolor='black', 
                          hatch=r'\\', label='p < 0.05'),
            mpatches.Patch(facecolor='white', edgecolor='black', 
                          hatch='...', label='p < 0.10'),
            mpatches.Patch(facecolor='white', edgecolor='black', 
                          label='Not significant')
        ]
        ax.legend(handles=legend_elements, loc='lower left', 
                 bbox_to_anchor=(0, -0.2), ncol=4, frameon=True)
        
        # Save figure
        filename = f"{variable_name.lower().replace(' ', '_')}_trend_significance.png"
        save_path = self.subdirs['trends'] / filename
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        print(f"  💾 Saved: {filename}")
        
        # Also create supplementary figures
        self._create_trend_supplementary_figures(trend_results, variable_name, 
                                               data_array, clip_to_shapefile)
        
        return trend_results
    
    def create_robust_overview_maps(self):
        """Create robust overview maps similar to simple_robust_visualization.py"""
        print("\n🎨 Creating robust overview maps...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), 
                               subplot_kw={'projection': ccrs.PlateCarree()})
        axes = axes.flatten()
        
        # Get data extent
        extent = self._get_safe_extent(self.gws_ds.groundwater)
        
        # Time slices to show
        time_indices = [0, len(self.gws_ds.time)//4, len(self.gws_ds.time)//2, 
                       3*len(self.gws_ds.time)//4, len(self.gws_ds.time)-1]
        
        # Plot mean first
        ax = axes[0]
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        
        mean_gws = self.gws_ds.groundwater.mean(dim='time')
        
        # Apply region mask if available
        if self.region_mask is not None:
            mean_gws = mean_gws.where(~np.isnan(self.region_mask))
        
        # Use robust color scale
        vmin, vmax = self._get_robust_colorscale(mean_gws.values, symmetric=True)
        
        im = ax.pcolormesh(self.gws_ds.lon, self.gws_ds.lat, mean_gws, 
                          cmap='RdBu_r', vmin=vmin, vmax=vmax,
                          transform=ccrs.PlateCarree())
        
        # Add features
        gl = self._add_map_features(ax, add_shapefile=True)
        
        ax.set_title('Mean GWS Anomaly (2003-2022)', fontsize=FONT_SIZE, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                           pad=0.05, shrink=0.8)
        cbar.set_label('GWS Anomaly (cm)', fontsize=FONT_SIZE)
        
        # Plot time slices
        for i, time_idx in enumerate(time_indices):
            ax = axes[i+1]
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            
            gws_slice = self.gws_ds.groundwater.isel(time=time_idx)
            
            # Apply region mask if available
            if self.region_mask is not None:
                gws_slice = gws_slice.where(~np.isnan(self.region_mask))
            
            time_str = str(self.gws_ds.time.values[time_idx])[:7]
            
            im = ax.pcolormesh(self.gws_ds.lon, self.gws_ds.lat, gws_slice, 
                              cmap='RdBu_r', vmin=vmin, vmax=vmax,
                              transform=ccrs.PlateCarree())
            
            # Add features
            gl = self._add_map_features(ax, add_shapefile=True)
            
            ax.set_title(f'GWS Anomaly ({time_str})', fontsize=FONT_SIZE, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                               pad=0.05, shrink=0.8)
            cbar.set_label('GWS Anomaly (cm)', fontsize=FONT_SIZE)
        
        plt.suptitle('Groundwater Storage Anomaly Overview', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        output_path = self.subdirs['overview'] / 'groundwater_overview_enhanced.png'
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved: {output_path}")
    
    def create_robust_spatial_statistics(self):
        """Create spatial statistics visualization with robust methods."""
        print("\n🗺️ Creating robust spatial statistics...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), 
                               subplot_kw={'projection': ccrs.PlateCarree()})
        axes = axes.flatten()
        
        # Get extent
        extent = self._get_safe_extent(self.gws_ds.groundwater)
        
        # Statistics to plot
        stats_to_plot = [
            ('Mean', self.gws_ds.groundwater.mean(dim='time'), 'RdBu_r', True),
            ('Std Dev', self.gws_ds.groundwater.std(dim='time'), 'viridis', False),
            ('Min', self.gws_ds.groundwater.min(dim='time'), 'Blues_r', False),
            ('Max', self.gws_ds.groundwater.max(dim='time'), 'Reds', False)
        ]
        
        for i, (title, data, cmap, symmetric) in enumerate(stats_to_plot):
            ax = axes[i]
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            
            # Apply region mask if available
            if self.region_mask is not None:
                data = data.where(~np.isnan(self.region_mask))
            
            # Use robust color scale
            vmin, vmax = self._get_robust_colorscale(data.values, symmetric=symmetric)
            
            im = ax.pcolormesh(self.gws_ds.lon, self.gws_ds.lat, data, 
                             cmap=cmap, vmin=vmin, vmax=vmax,
                             transform=ccrs.PlateCarree())
            
            # Add features
            gl = self._add_map_features(ax, add_shapefile=True)
            
            ax.set_title(f'GWS {title}', fontsize=FONT_SIZE, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                               pad=0.05, shrink=0.8)
            cbar.set_label(f'{title} (cm)', fontsize=FONT_SIZE)
        
        plt.suptitle('Spatial Statistics of Groundwater Storage', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        output_path = self.subdirs['overview'] / 'spatial_statistics_enhanced.png'
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved: {output_path}")
    
    def create_robust_time_series_analysis(self):
        """Create time series analysis with robust methods."""
        print("\n📈 Creating robust time series analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Calculate regional average (with mask if available)
        if self.region_mask is not None:
            regional_avg = self.gws_ds.groundwater.where(
                ~np.isnan(self.region_mask)
            ).mean(dim=['lat', 'lon'])
        else:
            regional_avg = self.gws_ds.groundwater.mean(dim=['lat', 'lon'])
        
        time_index = pd.to_datetime(self.gws_ds.time.values)
        
        # 1. Full time series
        ax1 = axes[0, 0]
        ax1.plot(time_index, regional_avg.values, 'b-', linewidth=2, alpha=0.7)
        
        # Add 12-month rolling mean
        rolling_mean = pd.Series(regional_avg.values, index=time_index).rolling(12, center=True).mean()
        ax1.plot(time_index, rolling_mean.values, 'r-', linewidth=3, label='12-month rolling mean')
        
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Regional GWS Anomaly (cm)')
        ax1.set_title('Regional Groundwater Storage Time Series')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Seasonal cycle (with proper datetime handling)
        ax2 = axes[0, 1]
        monthly_clim = []
        for month in range(1, 13):
            # Use pandas datetime for month selection
            month_mask = time_index.month == month
            if np.any(month_mask):
                month_data = regional_avg.values[month_mask]
                monthly_clim.append(np.nanmean(month_data))
            else:
                monthly_clim.append(0.0)
        
        months = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
        bars = ax2.bar(range(1, 13), monthly_clim, color='skyblue', alpha=0.7, edgecolor='black')
        ax2.set_xticks(range(1, 13))
        ax2.set_xticklabels(months)
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Mean GWS Anomaly (cm)')
        ax2.set_title('Seasonal Cycle')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 3. Annual means
        ax3 = axes[1, 0]
        annual_means = []
        years = []
        for year in range(2003, 2023):
            # Use pandas datetime for year selection
            year_mask = time_index.year == year
            if np.any(year_mask):
                year_data = regional_avg.values[year_mask]
                annual_means.append(np.nanmean(year_data))
                years.append(year)
        
        bars = ax3.bar(years, annual_means, color='lightcoral', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Year')
        ax3.set_ylabel('Annual Mean GWS Anomaly (cm)')
        ax3.set_title('Annual Mean Groundwater Storage')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Fix x-axis to show proper integer years
        if years:
            year_ticks = list(range(min(years), max(years) + 1, 2))  # Every 2 years
            ax3.set_xticks(year_ticks)
            ax3.set_xticklabels([str(y) for y in year_ticks])
        
        # Add trend line
        if len(years) > 2:
            z = np.polyfit(years, annual_means, 1)
            p = np.poly1d(z)
            ax3.plot(years, p(years), 'r--', linewidth=2, 
                    label=f'Trend: {z[0]:.3f} cm/year')
            ax3.legend()
        
        # 4. Distribution
        ax4 = axes[1, 1]
        valid_data = regional_avg.values[~np.isnan(regional_avg.values)]
        ax4.hist(valid_data, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        ax4.axvline(np.mean(valid_data), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(valid_data):.2f} cm')
        ax4.axvline(0, color='black', linestyle='-', alpha=0.5)
        ax4.set_xlabel('GWS Anomaly (cm)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Regional GWS Values')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Groundwater Storage Time Series Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        output_path = self.subdirs['overview'] / 'time_series_analysis_enhanced.png'
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved: {output_path}")
    
    def _create_trend_supplementary_figures(self, trend_results, variable_name, 
                                          data_array, clip_to_shapefile):
        """Create supplementary figures for trend analysis."""
        
        # 1. Histogram of trends by significance
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Get valid trends
        valid_trends = trend_results['trend'][~np.isnan(trend_results['trend'])]
        p_values = trend_results['p_value'][~np.isnan(trend_results['p_value'])]
        
        if len(valid_trends) > 0:
            # Separate by significance
            sig_mask = p_values < 0.05
            sig_trends = valid_trends[sig_mask]
            nonsig_trends = valid_trends[~sig_mask]
            
            # Plot histograms
            bins = np.linspace(np.percentile(valid_trends, 1), 
                              np.percentile(valid_trends, 99), 50)
            
            ax1.hist(sig_trends, bins=bins, alpha=0.7, label='Significant (p<0.05)', 
                    color='blue', edgecolor='black')
            ax1.hist(nonsig_trends, bins=bins, alpha=0.5, label='Not significant', 
                    color='gray', edgecolor='black')
            ax1.axvline(0, color='red', linestyle='--', linewidth=2)
            ax1.set_xlabel(f'Trend ({variable_name} per year)')
            ax1.set_ylabel('Number of pixels')
            ax1.set_title('Distribution of Trends')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Trend uncertainty map
        ax2 = plt.axes(projection=ccrs.PlateCarree())
        
        # Set extent using safe method
        extent = self._get_safe_extent(data_array)
        ax2.set_extent(extent, crs=ccrs.PlateCarree())
        
        # Plot standard error
        std_err_masked = trend_results['std_error'].copy()
        if clip_to_shapefile and self.region_mask is not None:
            std_err_masked = np.where(~np.isnan(self.region_mask), 
                                    std_err_masked, np.nan)
        
        im2 = ax2.pcolormesh(data_array.lon, data_array.lat, 
                            std_err_masked, cmap='viridis', 
                            transform=ccrs.PlateCarree())
        
        # Add features using robust method
        gl = self._add_map_features(ax2, add_shapefile=clip_to_shapefile)
        
        plt.colorbar(im2, ax=ax2, label='Standard Error')
        ax2.set_title('Trend Uncertainty (Standard Error)')
        
        plt.tight_layout()
        filename = f"{variable_name.lower().replace(' ', '_')}_trend_supplementary.png"
        plt.savefig(self.subdirs['trends'] / filename, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
    
    def create_combined_trend_maps(self, clip_to_shapefile=True):
        """
        Create combined trend maps with GWS large on left and other components on right.
        Layout: GWS (large, left) | Precipitation (top right), Soil Moisture (middle right), Snow (bottom right)
        """
        print("\n🎨 Creating combined trend maps...")
        
        # Collect all available data components
        components = []
        
        # 1. Groundwater Storage (always first/main)
        if hasattr(self, 'gws_ds'):
            components.append({
                'data': self.gws_ds.groundwater,
                'name': 'Groundwater Storage',
                'units': 'cm/year',
                'short_name': 'GWS'
            })
        
        # 2. Precipitation
        if self.features_ds is not None:
            precip_indices = []
            for i, feat in enumerate(self.features_ds.feature.values):
                if 'pr' in str(feat).lower() or 'chirps' in str(feat).lower():
                    precip_indices.append(i)
            
            if precip_indices:
                components.append({
                    'data': self.features_ds.features[:, precip_indices[0], :, :],
                    'name': 'Precipitation',
                    'units': 'mm/year',
                    'short_name': 'Precip'
                })
        
        # 3. Soil Moisture
        if hasattr(self, 'gws_ds') and 'soil_moisture_anomaly' in self.gws_ds:
            components.append({
                'data': self.gws_ds.soil_moisture_anomaly,
                'name': 'Soil Moisture',
                'units': 'cm/year',
                'short_name': 'SM'
            })
        
        # 4. Snow Water Equivalent
        if hasattr(self, 'gws_ds') and 'swe_anomaly' in self.gws_ds:
            components.append({
                'data': self.gws_ds.swe_anomaly,
                'name': 'Snow Water Equivalent',
                'units': 'cm/year',
                'short_name': 'SWE'
            })
        
        if len(components) == 0:
            print("  ⚠️ No data components found for trend analysis")
            return
        
        # Calculate trends for all components
        print(f"  📊 Calculating trends for {len(components)} components...")
        trend_data = {}
        
        for comp in components:
            print(f"    Processing {comp['name']}...")
            trend_results = self.calculate_pixel_trends(comp['data'])
            
            # Apply shapefile mask if requested
            if clip_to_shapefile and self.region_mask is not None:
                for key in trend_results:
                    trend_results[key] = np.where(
                        ~np.isnan(self.region_mask), 
                        trend_results[key], 
                        np.nan
                    )
            
            trend_data[comp['short_name']] = {
                'trend_results': trend_results,
                'data_array': comp['data'],
                'name': comp['name'],
                'units': comp['units']
            }
        
        # Calculate common color scale for all trend data (same units: cm/year)
        all_trends = []
        for comp_name, comp_data in trend_data.items():
            if comp_data['units'] == 'cm/year':  # Only for comparable units
                trend_values = comp_data['trend_results']['trend']
                valid_trends = trend_values[~np.isnan(trend_values)]
                if len(valid_trends) > 0:
                    all_trends.extend(valid_trends)
        
        if len(all_trends) > 0:
            # Common color scale for cm/year units
            common_vmin, common_vmax = self._get_robust_colorscale(
                np.array(all_trends), percentiles=(5, 95), symmetric=True)
        else:
            common_vmin, common_vmax = -1, 1
        
        # Create figure with balanced layout (adjusted for vertical colorbars)
        fig = plt.figure(figsize=(20, 12))
        
        # Create complex grid: 
        # - 4 rows, 4 columns
        # - Left: GWS plot (3 rows) + colorbar space (1 row)  
        # - Middle: 3 component plots (1 row each) + legend space (1 row)
        # - Right: vertical colorbars for each component (3 rows) + empty (1 row)
        gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 0.1], 
                             width_ratios=[2.2, 1, 0.06, 0.2], 
                             hspace=0.1, wspace=0.08)
        
        # Get common extent for all maps
        extent = self._get_safe_extent(list(trend_data.values())[0]['data_array'])
        
        # Plot GWS (large, left side - spans first 3 rows)
        gws_plotted = False
        if 'GWS' in trend_data:
            ax_gws = fig.add_subplot(gs[0:3, 0], projection=ccrs.PlateCarree())
            ax_gws.set_extent(extent, crs=ccrs.PlateCarree())
            
            gws_data = trend_data['GWS']
            trend_array = gws_data['trend_results']['trend']
            p_array = gws_data['trend_results']['p_value']
            
            # Plot trend using common color scale
            im_gws = ax_gws.pcolormesh(gws_data['data_array'].lon, 
                                      gws_data['data_array'].lat, 
                                      trend_array, cmap='RdBu_r', 
                                      vmin=common_vmin, vmax=common_vmax,
                                      transform=ccrs.PlateCarree())
            
            # Add significance hatching (only p < 0.10)
            sig_10 = p_array < 0.10
            lon_mesh, lat_mesh = np.meshgrid(gws_data['data_array'].lon, 
                                           gws_data['data_array'].lat)
            
            ax_gws.contourf(lon_mesh, lat_mesh, sig_10.astype(float), 
                           levels=[0.5, 1.5], colors='none', 
                           hatches=['///'], transform=ccrs.PlateCarree())
            
            # Add features
            self._add_map_features(ax_gws, add_shapefile=clip_to_shapefile)
            
            # Calculate significance statistics for GWS
            valid_gws_trends = trend_array[~np.isnan(trend_array)]
            valid_gws_pvals = p_array[~np.isnan(p_array)]
            
            if len(valid_gws_trends) > 0 and len(valid_gws_pvals) > 0:
                # Calculate percentages
                n_total = len(valid_gws_pvals)
                n_significant = np.sum(valid_gws_pvals < 0.10)
                pct_significant = (n_significant / n_total) * 100
                
                # Among significant pixels, count increasing and decreasing
                sig_mask = valid_gws_pvals < 0.10
                if np.any(sig_mask):
                    sig_trends = valid_gws_trends[sig_mask]
                    n_increasing = np.sum(sig_trends > 0)
                    n_decreasing = np.sum(sig_trends < 0)
                    pct_increasing = (n_increasing / n_significant) * 100 if n_significant > 0 else 0
                    pct_decreasing = (n_decreasing / n_significant) * 100 if n_significant > 0 else 0
                    
                    # Create statistics text with symbols
                    gws_stats_text = f"S:{pct_significant:.0f}% ↑{pct_increasing:.0f}% ↓{pct_decreasing:.0f}%"
                else:
                    gws_stats_text = "S:0%"
            else:
                gws_stats_text = "No data"
            
            # Title with statistics
            ax_gws.set_title(f'{gws_data["name"]} Trend\n{gws_stats_text}', 
                            fontsize=14, fontweight='bold')
            
            gws_plotted = True
        
        # Plot other components on right side (3 rows, 1 column each)
        other_components = [name for name in ['Precip', 'SM', 'SWE'] if name in trend_data]
        
        for i, comp_name in enumerate(other_components):
            if i >= 3:  # Only plot first 3 in right column
                break
                
            ax = fig.add_subplot(gs[i, 1], projection=ccrs.PlateCarree())
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            
            comp_data = trend_data[comp_name]
            trend_array = comp_data['trend_results']['trend']
            p_array = comp_data['trend_results']['p_value']
            
            # Choose color scale based on units and component
            if comp_data['units'] == 'cm/year':
                # Use common scale for cm/year, but check for snow data specifically
                if comp_name == 'SWE':
                    # For snow, use a tighter scale since range is often smaller
                    vmin, vmax = self._get_robust_colorscale(trend_array, 
                                                           percentiles=(10, 90), symmetric=True)
                    # Ensure minimum range for visibility
                    if abs(vmax - vmin) < 0.1:
                        vmax = max(abs(np.nanpercentile(trend_array, 90)), 0.05)
                        vmin = -vmax
                else:
                    vmin, vmax = common_vmin, common_vmax
            else:
                # Use individual scale for different units (like mm/year)
                vmin, vmax = self._get_robust_colorscale(trend_array, 
                                                       percentiles=(5, 95), symmetric=True)
            
            # Plot trend
            im = ax.pcolormesh(comp_data['data_array'].lon, 
                              comp_data['data_array'].lat, 
                              trend_array, cmap='RdBu_r', 
                              vmin=vmin, vmax=vmax,
                              transform=ccrs.PlateCarree())
            

            
            # Add significance hatching (only p < 0.10)
            sig_10 = p_array < 0.10
            lon_mesh, lat_mesh = np.meshgrid(comp_data['data_array'].lon, 
                                           comp_data['data_array'].lat)
            
            ax.contourf(lon_mesh, lat_mesh, sig_10.astype(float), 
                       levels=[0.5, 1.5], colors='none', 
                       hatches=['///'], transform=ccrs.PlateCarree())
            
            # Add features (more compact for smaller plots)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.STATES, linewidth=0.3, alpha=0.7)
            
            # Add shapefile outline if available
            if clip_to_shapefile and self.shapefile is not None:
                try:
                    self.shapefile.boundary.plot(ax=ax, color='black', linewidth=1,
                                               transform=ccrs.PlateCarree())
                except:
                    pass
            
            # Calculate significance statistics
            valid_trends = trend_array[~np.isnan(trend_array)]
            valid_pvals = p_array[~np.isnan(p_array)]
            
            if len(valid_trends) > 0 and len(valid_pvals) > 0:
                # Calculate percentages
                n_total = len(valid_pvals)
                n_significant = np.sum(valid_pvals < 0.10)
                pct_significant = (n_significant / n_total) * 100
                
                # Among significant pixels, count increasing and decreasing
                sig_mask = valid_pvals < 0.10
                if np.any(sig_mask):
                    sig_trends = valid_trends[sig_mask]
                    n_increasing = np.sum(sig_trends > 0)
                    n_decreasing = np.sum(sig_trends < 0)
                    pct_increasing = (n_increasing / n_significant) * 100 if n_significant > 0 else 0
                    pct_decreasing = (n_decreasing / n_significant) * 100 if n_significant > 0 else 0
                    
                    # Create statistics text with symbols
                    stats_text = f"S:{pct_significant:.0f}% ↑{pct_increasing:.0f}% ↓{pct_decreasing:.0f}%"
                else:
                    stats_text = "S:0%"
            else:
                stats_text = "No data"
            
            # Title with statistics
            ax.set_title(f'{comp_data["name"]}\n{stats_text}', fontsize=11, fontweight='bold')
            
            # Add individual vertical colorbar for each component
            cbar_ax = fig.add_subplot(gs[i, 2])
            cbar = plt.colorbar(im, cax=cbar_ax, orientation='vertical')
            cbar.set_label(f'Trend ({comp_data["units"]})', fontsize=9, fontweight='bold')
            cbar.ax.tick_params(labelsize=8)
        
        # Add horizontal colorbar for GWS at bottom left (reduced height)
        if gws_plotted:
            # Create colorbar axis in the bottom row, left column
            cbar_ax = fig.add_subplot(gs[3, 0])
            cbar = plt.colorbar(im_gws, cax=cbar_ax, orientation='horizontal')
            cbar.set_label('Groundwater Storage Trend (cm/year)', fontsize=12, fontweight='bold')
            cbar.ax.tick_params(labelsize=10)
        
        # Overall title
        time_range = f"{str(list(trend_data.values())[0]['data_array'].time.values[0])[:4]}-{str(list(trend_data.values())[0]['data_array'].time.values[-1])[:4]}"
        plt.suptitle(f'Water Storage Component Trends ({time_range})\n' + 
                    'Hatching indicates statistical significance (p < 0.10)', 
                    fontsize=15, fontweight='bold', y=0.95)
        
        # Add legend for hatching and symbols in the bottom right area
        legend_elements = [
            mpatches.Patch(facecolor='white', edgecolor='black', 
                          hatch='///', label='/// p < 0.10'),
            mpatches.Patch(facecolor='white', edgecolor='black', 
                          label='No hatching = NS'),
            mpatches.Patch(facecolor='none', edgecolor='none', 
                          label='S=Significant%, ↑=Increasing%, ↓=Decreasing%')
        ]
        
        # Position legend in bottom right area
        legend_ax = fig.add_subplot(gs[3, 1:])  # Span across remaining bottom columns
        legend_ax.axis('off')
        legend_ax.legend(handles=legend_elements, loc='center', ncol=1, frameon=True,
                       fontsize=9)
        
        # Save figure
        plt.tight_layout()
        filename = "combined_trend_maps_with_significance.png"
        save_path = self.subdirs['trends'] / filename
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        print(f"  💾 Saved combined trend maps: {filename}")
        return trend_data
    
    def create_combined_uncertainty_maps(self, clip_to_shapefile=True):
        """
        Create combined uncertainty maps in 2x2 layout showing trend uncertainty.
        """
        print("\n🎨 Creating combined uncertainty maps...")
        
        # Collect all available data components
        components = []
        
        # 1. Groundwater Storage
        if hasattr(self, 'gws_ds'):
            components.append({
                'data': self.gws_ds.groundwater,
                'name': 'Groundwater Storage',
                'units': 'cm/year',
                'short_name': 'GWS'
            })
        
        # 2. Precipitation
        if self.features_ds is not None:
            precip_indices = []
            for i, feat in enumerate(self.features_ds.feature.values):
                if 'pr' in str(feat).lower() or 'chirps' in str(feat).lower():
                    precip_indices.append(i)
            
            if precip_indices:
                components.append({
                    'data': self.features_ds.features[:, precip_indices[0], :, :],
                    'name': 'Precipitation',
                    'units': 'mm/year',
                    'short_name': 'Precip'
                })
        
        # 3. Soil Moisture
        if hasattr(self, 'gws_ds') and 'soil_moisture_anomaly' in self.gws_ds:
            components.append({
                'data': self.gws_ds.soil_moisture_anomaly,
                'name': 'Soil Moisture',
                'units': 'cm/year',
                'short_name': 'SM'
            })
        
        # 4. Snow Water Equivalent
        if hasattr(self, 'gws_ds') and 'swe_anomaly' in self.gws_ds:
            components.append({
                'data': self.gws_ds.swe_anomaly,
                'name': 'Snow Water Equivalent',
                'units': 'cm/year',
                'short_name': 'SWE'
            })
        
        if len(components) == 0:
            print("  ⚠️ No data components found for uncertainty analysis")
            return
        
        # Calculate trends for all components (reuse if already calculated)
        print(f"  📊 Calculating trend uncertainties for {len(components)} components...")
        uncertainty_data = {}
        
        for comp in components:
            print(f"    Processing {comp['name']}...")
            trend_results = self.calculate_pixel_trends(comp['data'])
            
            # Apply shapefile mask if requested
            if clip_to_shapefile and self.region_mask is not None:
                for key in trend_results:
                    trend_results[key] = np.where(
                        ~np.isnan(self.region_mask), 
                        trend_results[key], 
                        np.nan
                    )
            
            uncertainty_data[comp['short_name']] = {
                'trend_results': trend_results,
                'data_array': comp['data'],
                'name': comp['name'],
                'units': comp['units']
            }
        
        # Create 2x2 figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), 
                                subplot_kw={'projection': ccrs.PlateCarree()})
        axes = axes.flatten()
        
        # Get common extent for all maps
        extent = self._get_safe_extent(list(uncertainty_data.values())[0]['data_array'])
        
        # Plot up to 4 components
        component_names = list(uncertainty_data.keys())[:4]
        
        for i, comp_name in enumerate(component_names):
            ax = axes[i]
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            
            comp_data = uncertainty_data[comp_name]
            std_error = comp_data['trend_results']['std_error']
            
            # Use robust color scaling for uncertainty
            vmin, vmax = self._get_robust_colorscale(std_error, 
                                                   percentiles=(5, 95), symmetric=False)
            
            # Plot standard error
            im = ax.pcolormesh(comp_data['data_array'].lon, 
                              comp_data['data_array'].lat, 
                              std_error, cmap='viridis', 
                              vmin=vmin, vmax=vmax,
                              transform=ccrs.PlateCarree())
            
            # Add features
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.STATES, linewidth=0.5, alpha=0.7)
            
            # Add shapefile outline if available
            if clip_to_shapefile and self.shapefile is not None:
                try:
                    self.shapefile.boundary.plot(ax=ax, color='black', linewidth=1.5,
                                               transform=ccrs.PlateCarree())
                except:
                    pass
            
            # Add gridlines
            gl = ax.gridlines(draw_labels=True, alpha=0.5)
            gl.top_labels = False
            gl.right_labels = False
            
            # Title
            ax.set_title(f'{comp_data["name"]}\nTrend Uncertainty', 
                        fontsize=12, fontweight='bold')
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                               pad=0.05, shrink=0.8)
            cbar.set_label(f'Standard Error ({comp_data["units"]})', fontsize=10)
        
        # Hide unused subplots
        for i in range(len(component_names), 4):
            axes[i].set_visible(False)
        
        # Overall title
        time_range = f"{str(list(uncertainty_data.values())[0]['data_array'].time.values[0])[:4]}-{str(list(uncertainty_data.values())[0]['data_array'].time.values[-1])[:4]}"
        plt.suptitle(f'Trend Uncertainty Analysis ({time_range})\n' + 
                    'Higher values indicate less reliable trend estimates', 
                    fontsize=16, fontweight='bold')
        
        # Save figure
        plt.tight_layout()
        filename = "combined_uncertainty_maps.png"
        save_path = self.subdirs['trends'] / filename
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        print(f"  💾 Saved combined uncertainty maps: {filename}")
        return uncertainty_data

    def analyze_all_components(self):
        """Analyze trends for all water storage components."""
        print("\n🔄 ANALYZING ALL WATER STORAGE COMPONENTS")
        print("="*50)
        
        # Create combined trend maps
        trend_data = self.create_combined_trend_maps(clip_to_shapefile=True)
        
        # Create combined uncertainty maps
        uncertainty_data = self.create_combined_uncertainty_maps(clip_to_shapefile=True)
        
        return trend_data, uncertainty_data
    
    def create_extreme_timing_maps(self):
        """Create maps showing when extremes occurred."""
        print("\n⏰ CREATING EXTREME TIMING MAPS")
        print("="*50)
        
        variables = []
        if hasattr(self, 'gws_ds'):
            variables.append(('groundwater', self.gws_ds.groundwater, 'Groundwater Storage'))
        if hasattr(self, 'gws_ds') and 'tws' in self.gws_ds:
            variables.append(('tws', self.gws_ds.tws, 'Total Water Storage'))
        
        for var_name, data_array, display_name in variables:
            print(f"\n📊 Processing {display_name}")
            
            # Find timing of minimum and maximum for each pixel
            n_lat, n_lon = len(data_array.lat), len(data_array.lon)
            
            min_time_index = np.full((n_lat, n_lon), np.nan)
            max_time_index = np.full((n_lat, n_lon), np.nan)
            min_month = np.full((n_lat, n_lon), np.nan)
            max_month = np.full((n_lat, n_lon), np.nan)
            
            # Convert time coordinate to pandas datetime for easier handling
            time_as_datetime = pd.to_datetime(data_array.time.values)
            
            for i in range(n_lat):
                for j in range(n_lon):
                    ts = data_array[:, i, j].values
                    if not np.all(np.isnan(ts)):
                        # Find minimum
                        min_idx = np.nanargmin(ts)
                        min_time_index[i, j] = min_idx
                        min_month[i, j] = time_as_datetime[min_idx].month
                        
                        # Find maximum
                        max_idx = np.nanargmax(ts)
                        max_time_index[i, j] = max_idx
                        max_month[i, j] = time_as_datetime[max_idx].month
            
            # Apply mask if using shapefile
            if self.region_mask is not None:
                min_time_index = np.where(~np.isnan(self.region_mask), 
                                        min_time_index, np.nan)
                max_time_index = np.where(~np.isnan(self.region_mask), 
                                        max_time_index, np.nan)
                min_month = np.where(~np.isnan(self.region_mask), 
                                   min_month, np.nan)
                max_month = np.where(~np.isnan(self.region_mask), 
                                   max_month, np.nan)
            
            # Create figure with 4 panels
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12),
                                                        subplot_kw={'projection': ccrs.PlateCarree()})
            
            # Common extent
            extent = self._get_safe_extent(data_array)
            
            # 1. Year of minimum
            ax1.set_extent(extent, crs=ccrs.PlateCarree())
            
            # Convert time index to year
            min_year = np.full_like(min_time_index, np.nan)
            valid = ~np.isnan(min_time_index)
            for i in range(n_lat):
                for j in range(n_lon):
                    if valid[i, j]:
                        idx = int(min_time_index[i, j])
                        min_year[i, j] = time_as_datetime[idx].year
            
            # Create discrete year ticks for better display
            year_min = int(np.nanmin(min_year))
            year_max = int(np.nanmax(min_year))
            year_ticks = list(range(year_min, year_max + 1, 2))  # Every 2 years
            
            im1 = ax1.pcolormesh(data_array.lon, data_array.lat, min_year,
                               cmap='viridis', vmin=year_min, vmax=year_max,
                               transform=ccrs.PlateCarree())
            ax1.add_feature(cfeature.STATES, linewidth=0.5)
            ax1.add_feature(cfeature.COASTLINE)
            
            # Add shapefile outline
            if self.shapefile is not None:
                try:
                    for idx, row in self.shapefile.iterrows():
                        geom = row.geometry
                        if geom.type == 'Polygon':
                            x, y = geom.exterior.xy
                            ax1.plot(x, y, 'k-', linewidth=2, transform=ccrs.PlateCarree())
                        elif geom.type == 'MultiPolygon':
                            for poly in geom.geoms:
                                x, y = poly.exterior.xy
                                ax1.plot(x, y, 'k-', linewidth=2, transform=ccrs.PlateCarree())
                except:
                    pass
            
            cbar1 = plt.colorbar(im1, ax=ax1, label='Year', ticks=year_ticks)
            cbar1.ax.set_yticklabels([str(y) for y in year_ticks])
            ax1.set_title(f'Year of Minimum {display_name}')
            
            # 2. Month of minimum (seasonality)
            ax2.set_extent(extent, crs=ccrs.PlateCarree())
            
            # Use discrete colormap for months with better gradual transition
            from matplotlib.colors import ListedColormap
            month_colors = plt.cm.tab20c(np.linspace(0, 1, 12))
            cmap_months = ListedColormap(month_colors)
            
            im2 = ax2.pcolormesh(data_array.lon, data_array.lat, min_month,
                               cmap=cmap_months, vmin=0.5, vmax=12.5,
                               transform=ccrs.PlateCarree())
            ax2.add_feature(cfeature.STATES, linewidth=0.5)
            ax2.add_feature(cfeature.COASTLINE)
            
            # Add shapefile outline
            if self.shapefile is not None:
                try:
                    for idx, row in self.shapefile.iterrows():
                        geom = row.geometry
                        if geom.type == 'Polygon':
                            x, y = geom.exterior.xy
                            ax2.plot(x, y, 'k-', linewidth=2, transform=ccrs.PlateCarree())
                        elif geom.type == 'MultiPolygon':
                            for poly in geom.geoms:
                                x, y = poly.exterior.xy
                                ax2.plot(x, y, 'k-', linewidth=2, transform=ccrs.PlateCarree())
                except:
                    pass
            
            cbar2 = plt.colorbar(im2, ax=ax2, label='Month', ticks=range(1, 13))
            cbar2.ax.set_yticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
            ax2.set_title(f'Month of Typical Minimum {display_name}')
            
            # 3. Minimum value magnitude
            ax3.set_extent(extent, crs=ccrs.PlateCarree())
            
            min_values = np.full((n_lat, n_lon), np.nan)
            for i in range(n_lat):
                for j in range(n_lon):
                    ts = data_array[:, i, j].values
                    if not np.all(np.isnan(ts)):
                        min_values[i, j] = np.nanmin(ts)
            
            if self.region_mask is not None:
                min_values = np.where(~np.isnan(self.region_mask), 
                                    min_values, np.nan)
            
            vmax = np.nanpercentile(np.abs(min_values), 95)
            if vmax == 0:
                vmax = 1.0
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            
            im3 = ax3.pcolormesh(data_array.lon, data_array.lat, min_values,
                               cmap='RdBu_r', norm=norm,
                               transform=ccrs.PlateCarree())
            ax3.add_feature(cfeature.STATES, linewidth=0.5)
            ax3.add_feature(cfeature.COASTLINE)
            
            # Add shapefile outline
            if self.shapefile is not None:
                try:
                    for idx, row in self.shapefile.iterrows():
                        geom = row.geometry
                        if geom.type == 'Polygon':
                            x, y = geom.exterior.xy
                            ax3.plot(x, y, 'k-', linewidth=2, transform=ccrs.PlateCarree())
                        elif geom.type == 'MultiPolygon':
                            for poly in geom.geoms:
                                x, y = poly.exterior.xy
                                ax3.plot(x, y, 'k-', linewidth=2, transform=ccrs.PlateCarree())
                except:
                    pass
            
            plt.colorbar(im3, ax=ax3, label='Minimum Value (cm)')
            ax3.set_title(f'Magnitude of Minimum {display_name}')
            
            # 4. Histogram of extreme years
            ax4.remove()  # Remove map projection
            ax4 = fig.add_subplot(2, 2, 4)
            
            # Count occurrences by year
            years = []
            valid = ~np.isnan(min_year)
            for y in min_year[valid]:
                years.append(int(y))
            
            if years:
                year_counts = pd.Series(years).value_counts().sort_index()
                
                ax4.bar(year_counts.index, year_counts.values, color='darkred', alpha=0.7)
                ax4.set_xlabel('Year')
                ax4.set_ylabel('Number of Pixels with Annual Minimum')
                ax4.set_title(f'Distribution of {display_name} Minima by Year')
                ax4.grid(True, alpha=0.3)
                
                # Fix x-axis to show integer years
                year_min = min(year_counts.index)
                year_max = max(year_counts.index)
                year_ticks = list(range(year_min, year_max + 1, 2))  # Every 2 years
                ax4.set_xticks(year_ticks)
                ax4.set_xticklabels([str(y) for y in year_ticks])
                
                # Rotate x labels
                plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
            
            plt.suptitle(f'Timing and Magnitude of {display_name} Extremes', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            filename = f"{var_name}_extreme_timing.png"
            plt.savefig(self.subdirs['extremes'] / filename, dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close()
            print(f"  💾 Saved: {filename}")
    
    def create_regional_analysis(self, region_shapefiles=None, process_all_geometries=False):
        """
        Create regional analysis using different shapefiles.
        
        Parameters:
        -----------
        region_shapefiles : dict or str
            Dictionary of region_name: shapefile_path OR
            Single shapefile path to process all geometries within
        process_all_geometries : bool
            If True and single shapefile provided, process each geometry separately
        """
        print("\n🌍 CREATING REGIONAL ANALYSIS")
        print("="*50)
        
        # Handle different input types
        if isinstance(region_shapefiles, str):
            # Single shapefile provided
            shapefile_path = region_shapefiles
            print(f"📂 Processing single shapefile: {shapefile_path}")
            
            if os.path.exists(shapefile_path):
                gdf = gpd.read_file(shapefile_path)
                if gdf.crs != 'EPSG:4326':
                    gdf = gdf.to_crs('EPSG:4326')
                
                if process_all_geometries:
                    # Process each geometry separately
                    print(f"  ✅ Found {len(gdf)} geometries to process individually")
                    
                    # Find name column
                    name_col = None
                    for col in ['name', 'NAME', 'Name', 'STATE_NAME', 'AQ_NAME']:
                        if col in gdf.columns:
                            name_col = col
                            break
                    
                    if not name_col:
                        # Use index if no name column
                        gdf['name'] = [f"Region_{i}" for i in range(len(gdf))]
                        name_col = 'name'
                    
                    # Convert to dictionary format
                    region_shapefiles = {}
                    for idx, row in gdf.iterrows():
                        region_name = str(row[name_col]).replace(' ', '_')
                        # Create temporary single-geometry GeoDataFrame
                        single_geom = gpd.GeoDataFrame([row], geometry='geometry', crs=gdf.crs)
                        region_shapefiles[region_name] = single_geom
                else:
                    # Process as single region
                    region_name = os.path.splitext(os.path.basename(shapefile_path))[0]
                    region_shapefiles = {region_name: gdf}
            else:
                print(f"  ❌ Shapefile not found: {shapefile_path}")
                return {}
        
        elif region_shapefiles is None:
            # Try to auto-detect processed shapefiles
            processed_dir = Path("data/shapefiles/processed")
            region_shapefiles = {}
            
            # Check for main basin
            basin_path = processed_dir / "mississippi_river_basin.shp"
            if basin_path.exists():
                region_shapefiles['Mississippi_Basin'] = str(basin_path)
            
            # Check for subregions
            subregions_huc = processed_dir / "subregions_huc"
            if subregions_huc.exists():
                for shp in subregions_huc.glob("*.shp"):
                    region_name = shp.stem
                    region_shapefiles[region_name] = str(shp)
            
            if not region_shapefiles:
                print("  ⚠️ No shapefiles found automatically")
                return {}
        
        # Now process all regions
        regional_results = {}
        
        for region_name, shapefile in region_shapefiles.items():
            if shapefile is None:
                continue
            
            print(f"\n📍 Processing region: {region_name}")
            
            # Load shapefile if needed
            if isinstance(shapefile, str):
                if os.path.exists(shapefile):
                    gdf = gpd.read_file(shapefile)
                    if gdf.crs != 'EPSG:4326':
                        gdf = gdf.to_crs('EPSG:4326')
                else:
                    print(f"  ❌ Shapefile not found: {shapefile}")
                    continue
            else:
                gdf = shapefile
            
            # Create mask
            try:
                if len(gdf) > 1:
                    # Multiple geometries - union them
                    try:
                        # Try union_all() for newer geopandas versions
                        region_geom = gdf.union_all()
                    except AttributeError:
                        # Fall back to unary_union for older versions
                        region_geom = gdf.geometry.unary_union
                else:
                    # Single geometry
                    region_geom = gdf.geometry.iloc[0]
                
                if regionmask is not None:
                    region_obj = regionmask.Regions([region_geom], names=[region_name])
                    mask = region_obj.mask(self.gws_ds.lon, self.gws_ds.lat)
                    
                    # Check if mask is valid
                    if np.all(np.isnan(mask)):
                        print(f"  ⚠️ Region {region_name} does not overlap with data domain")
                        continue
                else:
                    print(f"  ⚠️ regionmask not available, skipping {region_name}")
                    continue
                
                # Calculate regional averages
                regional_ts = {}
                
                # Groundwater
                if hasattr(self, 'gws_ds'):
                    gws_masked = self.gws_ds.groundwater.where(~np.isnan(mask))
                    regional_ts['groundwater'] = gws_masked.mean(dim=['lat', 'lon'])
                
                # TWS
                if hasattr(self, 'gws_ds') and 'tws' in self.gws_ds:
                    tws_masked = self.gws_ds.tws.where(~np.isnan(mask))
                    regional_ts['tws'] = tws_masked.mean(dim=['lat', 'lon'])
                
                # Soil moisture
                if hasattr(self, 'gws_ds') and 'soil_moisture_anomaly' in self.gws_ds:
                    sm_masked = self.gws_ds.soil_moisture_anomaly.where(~np.isnan(mask))
                    regional_ts['soil_moisture'] = sm_masked.mean(dim=['lat', 'lon'])
                
                # Snow
                if hasattr(self, 'gws_ds') and 'swe_anomaly' in self.gws_ds:
                    swe_masked = self.gws_ds.swe_anomaly.where(~np.isnan(mask))
                    regional_ts['swe'] = swe_masked.mean(dim=['lat', 'lon'])
                
                regional_results[region_name] = regional_ts
                
                # Create regional time series plot
                self._plot_regional_timeseries(regional_ts, region_name, gdf)
                
            except Exception as e:
                print(f"  ❌ Error processing {region_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Create comparison plot if multiple regions
        if len(regional_results) > 1:
            self._plot_regional_comparison(regional_results)
        
        return regional_results
    
    def _plot_regional_timeseries(self, regional_ts, region_name, region_gdf):
        """Plot time series for a specific region."""
        fig = plt.figure(figsize=(16, 10))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], 
                            width_ratios=[2, 1], hspace=0.3, wspace=0.2)
        
        # Convert time to pandas datetime for better plotting
        time_index = pd.to_datetime(self.gws_ds.time.values)
        
        # 1. Multi-component time series
        ax1 = fig.add_subplot(gs[0, 0])
        
        colors = {'groundwater': 'blue', 'tws': 'black', 
                 'soil_moisture': 'brown', 'swe': 'cyan'}
        
        for var_name, ts_data in regional_ts.items():
            if var_name in colors:
                ax1.plot(time_index, ts_data.values, label=var_name.replace('_', ' ').title(),
                        color=colors[var_name], linewidth=2, alpha=0.8)
        
        ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax1.set_ylabel('Storage Anomaly (cm)')
        ax1.set_title(f'{region_name}: Water Storage Components Time Series')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 2. Map showing region
        ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
        
        # Plot context
        bounds = region_gdf.total_bounds
        # Check if bounds are valid
        if np.all(np.isfinite(bounds)):
            buffer = 2.0  # degrees
            ax2.set_extent([bounds[0]-buffer, bounds[2]+buffer,
                           bounds[1]-buffer, bounds[3]+buffer],
                          crs=ccrs.PlateCarree())
        else:
            ax2.set_global()  # Use global extent as fallback
        
        ax2.add_feature(cfeature.STATES, linewidth=0.5)
        ax2.add_feature(cfeature.COASTLINE)
        ax2.add_feature(cfeature.BORDERS, linestyle='--')
        ax2.add_feature(cfeature.RIVERS, alpha=0.5)
        
        # Highlight region
        region_gdf.plot(ax=ax2, facecolor='lightblue', edgecolor='red',
                       linewidth=2, alpha=0.5, transform=ccrs.PlateCarree())
        
        ax2.set_title(f'{region_name} Location')
        
        # 3. Seasonal cycle
        ax3 = fig.add_subplot(gs[1, :])
        
        # Use the original time coordinate for datetime operations
        original_time = self.gws_ds.time
        time_dt = pd.to_datetime(original_time.values)
        
        for var_name, ts_data in regional_ts.items():
            if var_name in colors:
                # Calculate monthly climatology using original time coordinate
                monthly_clim = []
                for month in range(1, 13):
                    # Create month mask using original time coordinate
                    month_mask = time_dt.month == month
                    if np.any(month_mask):
                        month_data = ts_data.values[month_mask]
                        monthly_clim.append(np.nanmean(month_data))
                    else:
                        monthly_clim.append(0.0)
                
                months = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
                ax3.plot(range(1, 13), monthly_clim, 'o-', 
                        label=var_name.replace('_', ' ').title(),
                        color=colors[var_name], linewidth=2, markersize=8)
        
        ax3.set_xticks(range(1, 13))
        ax3.set_xticklabels(months)
        ax3.set_xlabel('Month')
        ax3.set_ylabel('Mean Anomaly (cm)')
        ax3.set_title('Seasonal Cycle')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        
        # 4. Trend analysis
        ax4 = fig.add_subplot(gs[2, :])
        
        trend_results = []
        
        for var_name, ts_data in regional_ts.items():
            if var_name in colors:
                # Calculate trend
                time_numeric = np.arange(len(ts_data))
                valid = ~np.isnan(ts_data.values)
                
                if np.sum(valid) > 24:  # At least 2 years
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        time_numeric[valid], ts_data.values[valid]
                    )
                    
                    # Convert to annual trend
                    annual_trend = slope * 12
                    
                    trend_results.append({
                        'Variable': var_name.replace('_', ' ').title(),
                        'Trend (cm/year)': annual_trend,
                        'p-value': p_value,
                        'R²': r_value**2
                    })
                    
                    # Plot data with trend line
                    ax4.scatter(time_index[valid], ts_data.values[valid], 
                              alpha=0.5, s=20, color=colors[var_name])
                    trend_line = slope * time_numeric + intercept
                    ax4.plot(time_index, trend_line, '--', color=colors[var_name],
                            linewidth=2, label=f'{var_name}: {annual_trend:.3f} cm/yr')
        
        ax4.set_xlabel('Year')
        ax4.set_ylabel('Storage Anomaly (cm)')
        ax4.set_title('Linear Trends')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add trend table
        if trend_results:
            trend_df = pd.DataFrame(trend_results)
            table_text = trend_df.to_string(index=False, float_format='%.3f')
            ax4.text(1.02, 0.5, table_text, transform=ax4.transAxes,
                    fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'{region_name} Regional Water Storage Analysis', 
                    fontsize=16, fontweight='bold')
        
        filename = f"{region_name.lower().replace(' ', '_')}_regional_analysis.png"
        plt.savefig(self.subdirs['regional'] / filename, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  💾 Saved: {filename}")
    
    def _plot_regional_comparison(self, regional_results):
        """Compare multiple regions."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Variables to compare
        variables = ['groundwater', 'tws', 'soil_moisture', 'swe']
        colors = plt.cm.Set3(np.linspace(0, 1, len(regional_results)))
        
        for i, var in enumerate(variables):
            ax = axes[i]
            
            for j, (region_name, regional_ts) in enumerate(regional_results.items()):
                if var in regional_ts:
                    time_index = pd.to_datetime(self.gws_ds.time.values)
                    ax.plot(time_index, regional_ts[var].values, 
                           label=region_name, color=colors[j], 
                           linewidth=2, alpha=0.8)
            
            ax.set_title(var.replace('_', ' ').title())
            ax.set_ylabel('Anomaly (cm)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            
            if i >= 2:
                ax.set_xlabel('Year')
        
        plt.suptitle('Regional Comparison of Water Storage Components', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plt.savefig(self.subdirs['regional'] / 'regional_comparison.png', 
                   dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  💾 Saved: regional_comparison.png")
    
    def create_comprehensive_report(self):
        """Generate a comprehensive summary report."""
        print("\n📄 GENERATING COMPREHENSIVE REPORT")
        print("="*50)
        
        report_lines = [
            "GRACE GROUNDWATER DOWNSCALING - ADVANCED ANALYSIS REPORT",
            "="*60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "ANALYSIS SUMMARY:",
            "-"*30
        ]
        
        # Add information about figures created
        figure_counts = {}
        for subdir_name, subdir_path in self.subdirs.items():
            png_files = list(subdir_path.glob('*.png'))
            figure_counts[subdir_name] = len(png_files)
            
            report_lines.extend([
                f"\n{subdir_name.upper()} FIGURES ({len(png_files)} files):",
                "-"*20
            ])
            
            for fig_file in sorted(png_files):
                report_lines.append(f"  • {fig_file.name}")
        
        # Add data summary
        report_lines.extend([
            "\n\nDATA COVERAGE:",
            "-"*30,
            f"Spatial domain: {float(self.gws_ds.lat.min()):.2f}°N to {float(self.gws_ds.lat.max()):.2f}°N, "
            f"{float(self.gws_ds.lon.min()):.2f}°E to {float(self.gws_ds.lon.max()):.2f}°E",
            f"Temporal coverage: {str(self.gws_ds.time.values[0])[:10]} to {str(self.gws_ds.time.values[-1])[:10]}",
            f"Number of time steps: {len(self.gws_ds.time)}",
            f"Spatial resolution: {len(self.gws_ds.lat)} x {len(self.gws_ds.lon)} pixels"
        ])
        
        if self.shapefile is not None:
            report_lines.extend([
                f"\nShapefile used for clipping: Yes",
                f"Region area: Analysis limited to shapefile boundaries"
            ])
        
        # Save report
        report_path = self.figures_dir / "analysis_report.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"  💾 Report saved to: {report_path}")
        
        # Print summary
        print(f"\n📊 ANALYSIS COMPLETE!")
        print(f"  Total figures created: {sum(figure_counts.values())}")
        for category, count in figure_counts.items():
            print(f"    {category}: {count} figures")
        print(f"  Output directory: {self.figures_dir}")


def main():
    """Main function to run all analyses."""
    print("🚀 GRACE ADVANCED SPATIAL-TEMPORAL ANALYSIS (ENHANCED)")
    print("="*70)
    
    # Initialize visualizer
    # Option 1: Use the created Mississippi River Basin shapefile
    shapefile_path = "data/shapefiles/processed/mississippi_river_basin.shp"
    
    # Check if processed shapefiles exist
    if not os.path.exists(shapefile_path):
        print("⚠️ Mississippi Basin shapefile not found!")
        print("📌 Run this first: python scripts/create_mississippi_basin_from_huc.py")
        shapefile_path = None
    
    visualizer = AdvancedGRACEVisualizer(
        base_dir=".",
        shapefile_path=shapefile_path
    )
    
    # Run all analyses
    
    # 0. Create robust overview visualizations first
    visualizer.create_robust_overview_maps()
    visualizer.create_robust_spatial_statistics()
    visualizer.create_robust_time_series_analysis()
    
    # 1. Analyze all water storage components (creates trend maps with significance)
    visualizer.analyze_all_components()
    
    # 2. Create extreme timing maps
    visualizer.create_extreme_timing_maps()
    
    # 3. Regional analysis
    # Option A: Use specific regions
    if os.path.exists("data/shapefiles/processed/subregions_huc"):
        print("\n📊 Processing HUC-based subregions...")
        regions = {}
        for shp in Path("data/shapefiles/processed/subregions_huc").glob("*.shp"):
            regions[shp.stem] = str(shp)
        if regions:
            visualizer.create_regional_analysis(regions)
    
    # Option B: Process individual states
    if os.path.exists("data/shapefiles/processed/individual_states"):
        print("\n📊 Processing individual states (first 5)...")
        state_files = list(Path("data/shapefiles/processed/individual_states").glob("*.shp"))[:5]
        if state_files:
            state_regions = {shp.stem: str(shp) for shp in state_files}
            visualizer.create_regional_analysis(state_regions)
    
    # 4. Generate comprehensive report
    visualizer.create_comprehensive_report()
    
    print("\n✅ All analyses complete!")
    print(f"📁 Check results in: {visualizer.figures_dir}")
    
    return visualizer


if __name__ == "__main__":
    main() 