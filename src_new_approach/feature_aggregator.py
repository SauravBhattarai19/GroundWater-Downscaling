"""
Feature Aggregation: Upscale 5km features to 55km GRACE resolution

This module upscales all high-resolution features to match GRACE's native
resolution (55km), ensuring we train models at the correct measurement scale.
"""

import numpy as np
import pandas as pd
import xarray as xr
import os
from pathlib import Path
from typing import Dict, List, Tuple
import re
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from src_new_approach.utils_downscaling import (
    load_config, 
    get_config_value,
    coarsen_2d,
    get_aggregation_factor,
    get_feature_aggregation_method,
    print_statistics,
    add_metadata_to_dataset
)


class FeatureAggregator:
    """
    Aggregate high-resolution features to coarse GRACE resolution.
    
    This ensures training happens at the correct physical scale.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize feature aggregator.
        
        Parameters:
        -----------
        config : Dict
            Configuration dictionary
        """
        self.config = config
        self.aggregation_factor = get_aggregation_factor(config)
        self.mask_by_grace = get_config_value(
            config, 
            'feature_aggregation.mask_by_grace', 
            True
        )
        
        print(f"🔧 Feature Aggregator initialized:")
        print(f"   Aggregation factor: {self.aggregation_factor}x (55km / 5km)")
        print(f"   Mask by GRACE coverage: {self.mask_by_grace}")
    
    def load_fine_features(self, features_path: str) -> xr.Dataset:
        """
        Load high-resolution feature stack and extend to full temporal coverage.
        
        The original feature stack was limited to GRACE's original availability (219 months).
        This function extends it to the full temporal range (261 months) using available data.
        
        Parameters:
        -----------
        features_path : str
            Path to feature stack NetCDF
        
        Returns:
        --------
        xr.Dataset
            Feature dataset at fine resolution with full temporal coverage
        """
        print(f"\n📂 Loading fine-resolution features from: {features_path}")
        
        ds = xr.open_dataset(features_path)
        
        # Check dimensions
        if 'features' in ds.data_vars:
            n_times = len(ds.time)
            n_lats = len(ds.lat)
            n_lons = len(ds.lon)
            n_features = len(ds.feature)
            
            print(f"✓ Loaded feature stack:")
            print(f"   Shape: {n_times} times × {n_lats} lat × {n_lons} lon × {n_features} features")
            print(f"   Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
            print(f"   Current coverage: {n_times} months (artificially limited to GRACE gaps)")
            print(f"   Spatial extent: lat [{ds.lat.min().values:.2f}, {ds.lat.max().values:.2f}]")
            print(f"                   lon [{ds.lon.min().values:.2f}, {ds.lon.max().values:.2f}]")
        else:
            # Individual variables
            print(f"✓ Loaded feature dataset with {len(ds.data_vars)} variables")
        
        return ds
    
    def _create_spatial_grid_from_raw(self, first_file_path: str) -> xr.Dataset:
        """
        Create spatial grid by loading the first raw data file.
        
        Parameters:
        -----------
        first_file_path : str
            Path to first raw data file
            
        Returns:
        --------
        xr.Dataset
            Dataset with spatial coordinates
        """
        try:
            import rasterio
            
            print(f"   Creating spatial grid from: {first_file_path}")
            
            # Open the first file to get spatial information
            with rasterio.open(first_file_path) as src:
                # Get transform and shape
                transform = src.transform
                height, width = src.shape
                
                # Create coordinate arrays
                cols, rows = np.meshgrid(np.arange(width), np.arange(height))
                
                # Transform pixel coordinates to geographic coordinates
                xs, ys = rasterio.transform.xy(transform, rows, cols)
                lons = np.array(xs)
                lats = np.array(ys)
                
                # Get 1D coordinate arrays
                lat_coords = lats[:, 0]  # First column
                lon_coords = lons[0, :]  # First row
                
                print(f"   Grid dimensions: {height} × {width}")
                print(f"   Lat range: [{lat_coords.min():.3f}, {lat_coords.max():.3f}]")
                print(f"   Lon range: [{lon_coords.min():.3f}, {lon_coords.max():.3f}]")
                
                # Create dummy dataset with just coordinates
                dummy_data = np.zeros((1, height, width))
                ds = xr.Dataset(
                    {
                        'dummy': (['time', 'lat', 'lon'], dummy_data)
                    },
                    coords={
                        'time': pd.date_range('2003-01-01', periods=1, freq='MS'),
                        'lat': lat_coords,
                        'lon': lon_coords
                    }
                )
                
                return ds
                
        except Exception as e:
            print(f"   ❌ Error creating spatial grid: {e}")
            return None
    
    def load_complete_features_from_raw(self, target_times: pd.DatetimeIndex) -> xr.Dataset:
        """
        Load features directly from raw data with complete temporal coverage.
        
        This bypasses the artificially limited feature stack and loads data
        for all target times from the original sources.
        
        Parameters:
        -----------
        target_times : pd.DatetimeIndex
            Target temporal coverage (e.g., 261 months from gap-filled GRACE)
        
        Returns:
        --------
        xr.Dataset
            Complete feature dataset
        """
        print(f"\n🔄 LOADING COMPLETE FEATURES FROM RAW DATA")
        print("="*60)
        print(f"Target temporal coverage: {len(target_times)} months")
        print(f"Range: {pd.to_datetime(target_times.min().values).strftime('%Y-%m')} to {pd.to_datetime(target_times.max().values).strftime('%Y-%m')}")
        
        # Calculate month indices for target times
        base_date = pd.Timestamp('2003-01-01')
        target_indices = []
        target_month_strings = []
        
        # Convert xarray DataArray to pandas DatetimeIndex if needed
        if hasattr(target_times, 'values'):
            target_times_pd = pd.to_datetime(target_times.values)
        else:
            target_times_pd = target_times
        
        for target_time in target_times_pd:
            # Calculate index from base date
            months_diff = (target_time.year - base_date.year) * 12 + (target_time.month - base_date.month)
            target_indices.append(months_diff)
            target_month_strings.append(target_time.strftime('%Y-%m'))
        
        print(f"Loading indices: {min(target_indices)} to {max(target_indices)}")
        
        # Load temporal features from raw data
        features_dict = {}
        feature_names = []
        
        # Define data sources and their patterns
        data_sources = {
            'gldas': {
                'path': 'data/raw/gldas',
                'variables': ['Evap_tavg', 'SWE_inst', 'SoilMoi0_10cm_inst', 'SoilMoi10_40cm_inst', 'SoilMoi100_200cm_inst'],
                'file_pattern': 'numeric'  # 0.tif, 1.tif, etc.
            },
            'terraclimate': {
                'path': 'data/raw/terraclimate', 
                'variables': ['aet', 'def', 'pr', 'tmmn', 'tmmx'],  # Load tmmn, tmmx to calculate tmean
                'file_pattern': 'yyyymm'
            },
            'chirps': {
                'path': 'data/raw/chirps',
                'variables': ['chirps'],
                'file_pattern': 'direct',  # Files directly in chirps folder
                'is_direct': True
            }
        }
        
        # Load first file to get spatial dimensions
        first_var_path = None
        for source, info in data_sources.items():
            source_path = info['path']
            if os.path.exists(source_path):
                if info.get('is_direct', False):
                    # Files directly in source folder (like chirps)
                    first_file = os.path.join(source_path, '0.tif')
                    if os.path.exists(first_file):
                        first_var_path = first_file
                        break
                else:
                    # Files in variable subfolders
                    for var in info['variables']:
                        var_path = os.path.join(source_path, var)
                        if os.path.exists(var_path):
                            first_file = os.path.join(var_path, '0.tif')
                            if os.path.exists(first_file):
                                first_var_path = first_file
                                break
                    if first_var_path:
                        break
        
        if not first_var_path:
            print("❌ No raw data files found!")
            return None
        
        # Use existing fine feature stack spatial grid as target for consistency
        # This ensures all data sources are resampled to the same common grid
        target_feature_path = self.config.get('paths', {}).get('feature_stack_fine', 'data/processed/feature_stack.nc')
        
        # Try to load target feature file, fallback to original if needed
        fine_ds = None
        if os.path.exists(target_feature_path):
            try:
                fine_ds = self.load_fine_features(target_feature_path)
                print(f"✓ Using spatial grid from: {target_feature_path}")
            except:
                pass
        
        # Fallback to original feature stack if target doesn't exist
        if fine_ds is None:
            original_path = 'data/processed/feature_stack.nc'
            if os.path.exists(original_path):
                try:
                    fine_ds = self.load_fine_features(original_path)
                    print(f"✓ Using spatial grid from fallback: {original_path}")
                except:
                    pass
        
        # Final fallback: create spatial grid from first raw data file
        if fine_ds is None:
            print(f"⚠️ No existing feature stack found, creating spatial grid from raw data...")
            fine_ds = self._create_spatial_grid_from_raw(first_var_path)
            if fine_ds is None:
                print("❌ Failed to create spatial grid from raw data!")
                return None
        
        height = len(fine_ds.lat)
        width = len(fine_ds.lon)
        lat_coords = fine_ds.lat.values
        lon_coords = fine_ds.lon.values
        
        print(f"Using target grid from fine features:")
        print(f"Spatial dimensions: {height} × {width}")
        print(f"Coordinate ranges: lat [{lat_coords.min():.2f}, {lat_coords.max():.2f}], lon [{lon_coords.min():.2f}, {lon_coords.max():.2f}]")
        
        # Calculate total features (including derived ones like tmean)
        # Note: tmean replaces tmmn+tmmx, so we don't add extra space
        n_features_base = sum(len(info['variables']) for info in data_sources.values())
        n_features_total = n_features_base  # tmean calculated from tmmn+tmmx, chirps as one variable
        feature_data = np.full((len(target_times), n_features_total, height, width), np.nan, dtype=np.float32)
        
        feature_idx = 0
        
        # Store tmmn and tmmx data for tmean calculation
        tmmn_data = None
        tmmx_data = None
        
        # Load each data source
        for source_name, info in data_sources.items():
            source_path = info['path']
            if not os.path.exists(source_path):
                print(f"⚠️ Skipping {source_name}: path not found")
                continue
                
            print(f"\n📊 Loading {source_name} data...")
            
            if info.get('is_direct', False):
                # Handle direct files (like chirps)
                print(f"   Loading chirps from direct files...")
                for time_idx, file_idx in enumerate(tqdm(target_indices, desc="   chirps")):
                    file_path = os.path.join(source_path, f"{file_idx}.tif")
                    
                    if os.path.exists(file_path):
                        try:
                            import rioxarray as rxr
                            # Load with rioxarray for easy resampling
                            data_array = rxr.open_rasterio(file_path, masked=True).squeeze()
                            
                            # Resample to target grid if needed
                            if data_array.shape != (height, width):
                                # Create target grid
                                target_coords = {
                                    'y': lat_coords,
                                    'x': lon_coords
                                }
                                # Resample using bilinear interpolation
                                data_array = data_array.interp(target_coords, method='linear')
                            
                            # Convert to numpy and handle nodata
                            data_float = data_array.values.astype(np.float32)
                            if hasattr(data_array, 'mask'):
                                data_float[data_array.mask] = np.nan
                            
                            # Apply unit conversions (CHIRPS should be in mm already)
                            data_float = self._apply_unit_conversions(data_float, 'chirps', 'chirps')
                                
                            feature_data[time_idx, feature_idx, :, :] = data_float
                        except Exception as e:
                            print(f"     ⚠️ Error loading {file_path}: {e}")
                
                feature_names.append('chirps')
                feature_idx += 1
            else:
                # Handle variable subfolders
                for var_name in info['variables']:
                    var_path = os.path.join(source_path, var_name)
                    if not os.path.exists(var_path):
                        print(f"   ⚠️ Variable {var_name} not found")
                        feature_names.append(f"{var_name}_missing")
                        feature_idx += 1
                        continue
                    
                    print(f"   Loading {var_name}...")
                    
                    # Load data for each target time
                    for time_idx, target_time in enumerate(tqdm(target_times_pd, desc=f"   {var_name}")):
                        # Determine file name based on pattern
                        if info['file_pattern'] == 'yyyymm':
                            # TerraClimate format: YYYYMM.tif
                            file_name = f"{target_time.strftime('%Y%m')}.tif"
                        else:
                            # Numeric format: N.tif
                            file_idx = target_indices[time_idx]
                            file_name = f"{file_idx}.tif"
                            
                        file_path = os.path.join(var_path, file_name)
                        
                        if os.path.exists(file_path):
                            try:
                                import rioxarray as rxr
                                # Load with rioxarray for easy resampling
                                data_array = rxr.open_rasterio(file_path, masked=True).squeeze()
                                
                                # Resample to target grid if needed
                                if data_array.shape != (height, width):
                                    # Create target grid
                                    target_coords = {
                                        'y': lat_coords,
                                        'x': lon_coords
                                    }
                                    # Resample using bilinear interpolation
                                    data_array = data_array.interp(target_coords, method='linear')
                                
                                # Convert to numpy and handle nodata
                                data_float = data_array.values.astype(np.float32)
                                if hasattr(data_array, 'mask'):
                                    data_float[data_array.mask] = np.nan
                                
                                # Apply unit conversions based on variable name and data source
                                data_float = self._apply_unit_conversions(data_float, var_name, source_name)
                                    
                                feature_data[time_idx, feature_idx, :, :] = data_float
                            except Exception as e:
                                print(f"     ⚠️ Error loading {file_path}: {e}")
                        # If file doesn't exist, data remains NaN
                    
                    # Store tmmn and tmmx for tmean calculation
                    if var_name == 'tmmn':
                        tmmn_data = feature_data[:, feature_idx, :, :].copy()
                    elif var_name == 'tmmx':
                        tmmx_data = feature_data[:, feature_idx, :, :].copy()
                    
                    feature_names.append(var_name)
                    feature_idx += 1
        
        # Calculate tmean from tmmn and tmmx (replace tmmn and tmmx with tmean)
        if tmmn_data is not None and tmmx_data is not None:
            print("\n🌡️ Calculating tmean from (tmmn + tmmx) / 2...")
            tmean_data = (tmmn_data + tmmx_data) / 2.0
            
            # Find tmmn and tmmx indices and replace with tmean
            tmmn_idx = feature_names.index('tmmn')
            tmmx_idx = feature_names.index('tmmx')
            
            # Replace tmmn with tmean
            feature_data[:, tmmn_idx, :, :] = tmean_data
            feature_names[tmmn_idx] = 'tmean'
            
            # Remove tmmx (shift data and names)
            if tmmx_idx < len(feature_names) - 1:
                # Shift data left
                feature_data[:, tmmx_idx:-1, :, :] = feature_data[:, tmmx_idx+1:, :, :]
            # Remove last feature slot and name
            feature_data = feature_data[:, :-1, :, :]
            feature_names.pop(tmmx_idx)
            
            print(f"   ✅ Replaced tmmn+tmmx with tmean")
        else:
            print("⚠️ Could not calculate tmean - missing tmmn or tmmx data")
        
        # Load static features directly from raw data
        static_data, static_names = self._load_static_features_from_raw(height, width, lat_coords, lon_coords)
        
        # Create dataset
        data_vars = {
            'features': (['time', 'feature', 'lat', 'lon'], feature_data)
        }
        
        coords = {
            'time': target_times,
            'feature': feature_names,
            'lat': lat_coords,
            'lon': lon_coords
        }
        
        if static_data is not None:
            # Resample static features to match spatial grid if needed
            if static_data.shape[1:] != (height, width):
                print("   Resampling static features to match grid...")
                from scipy.interpolate import RegularGridInterpolator
                
                # Get original static coordinates 
                orig_static_lat = static_ds.lat.values
                orig_static_lon = static_ds.lon.values
                
                resampled_static = np.full((len(static_names), height, width), np.nan)
                
                for i in range(len(static_names)):
                    orig_data = static_data[i, :, :]
                    if not np.all(np.isnan(orig_data)):
                        interpolator = RegularGridInterpolator(
                            (orig_static_lat, orig_static_lon),
                            orig_data,
                            bounds_error=False,
                            fill_value=np.nan
                        )
                        
                        # Create target grid
                        lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)
                        target_points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
                        
                        interpolated = interpolator(target_points).reshape(height, width)
                        resampled_static[i, :, :] = interpolated
                
                static_data = resampled_static
            
            data_vars['static_features'] = (['static_feature', 'lat', 'lon'], static_data)
            coords['static_feature'] = static_names
        
        complete_ds = xr.Dataset(data_vars, coords=coords)
        
        print(f"\n✅ COMPLETE FEATURE LOADING SUCCESSFUL!")
        print(f"   Temporal features: {len(feature_names)} variables × {len(target_times)} months")
        if static_data is not None:
            print(f"   Static features: {len(static_names)} variables")
        print(f"   Spatial grid: {height} × {width}")
        print(f"   Total coverage: {len(target_times)} months (matches gap-filled GRACE)")
        
        return complete_ds
    
    def _apply_unit_conversions(self, data: np.ndarray, var_name: str, source_name: str) -> np.ndarray:
        """
        Apply proper unit conversions for different variables.
        
        Parameters:
        -----------
        data : np.ndarray
            Raw data values
        var_name : str
            Variable name (e.g., 'Evap_tavg', 'tmmn', 'aet')
        source_name : str
            Data source (e.g., 'gldas', 'terraclimate', 'chirps')
            
        Returns:
        --------
        np.ndarray
            Data with proper units applied
        """
        
        if source_name == 'gldas':
            if var_name == 'Evap_tavg':
                # Convert from kg/m²/s to mm/month
                # 1 kg/m²/s = 1 mm/s (water density = 1000 kg/m³)
                # Multiply by seconds in average month (30.44 days)
                seconds_per_month = 30.44 * 24 * 3600  # 2,629,440 seconds
                data = data * seconds_per_month
                print(f"      🔄 Converting {var_name}: kg/m²/s → mm/month (×{seconds_per_month:.0f})")
            # Other GLDAS variables (soil moisture, SWE) should already be in correct units
            
        elif source_name == 'terraclimate':
            # TerraClimate variables with scale factor 0.1 (need to divide by 10)
            # Source: https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_TERRACLIMATE
            if var_name in ['tmmn', 'tmmx']:
                # Convert from Celsius×10 to Celsius
                data = data / 10.0
                print(f"      🔄 Converting {var_name}: Celsius×10 → Celsius (÷10)")
            elif var_name in ['aet', 'def']:
                # Convert from mm×10 to mm (aet and def have 0.1 scale factor)
                data = data / 10.0
                print(f"      🔄 Converting {var_name}: mm×10 → mm (÷10)")
            elif var_name == 'pr':
                # Precipitation is already in mm (no scale factor in catalog)
                print(f"      ✅ {var_name}: already in mm (no conversion needed)")
            else:
                # Other TerraClimate variables - check catalog for scale factor
                print(f"      ⚠️ {var_name}: unknown scaling, applying no conversion")
                
        elif source_name == 'chirps':
            # CHIRPS precipitation should already be in mm/month
            pass
        
        return data
    
    def _load_static_features_from_raw(self, height: int, width: int, lat_coords: list, lon_coords: list):
        """
        Load all static features directly from raw data sources.
        
        Based on the original features.py static feature definitions.
        """
        print(f"\n🏔️ LOADING STATIC FEATURES FROM RAW DATA")
        print("="*50)
        
        import rasterio
        from rasterio.enums import Resampling
        import rasterio.warp
        
        static_features = []
        static_names = []
        
        # Define static data sources
        base_dir = "data/raw"
        
        static_sources = {
            # MODIS Land Cover (One-hot encoded)
            "modis_land_cover": {
                "path": os.path.join(base_dir, "modis_land_cover", "landcover_5km.tif"),
                "type": "categorical",
                "classes": list(range(1, 18))  # MODIS classes 1-17
            },
            
            # Population
            "landscan_population": {
                "path": os.path.join(base_dir, "landscan", "2003.tif"),
                "type": "continuous"
            },
            
            # Soil properties from OpenLandMap
            "soil_sand_0cm": {
                "path": os.path.join(base_dir, "openlandmap", "sand_0cm_5km.tif"),
                "type": "continuous"
            },
            "soil_clay_0cm": {
                "path": os.path.join(base_dir, "openlandmap", "clay_0cm_5km.tif"),
                "type": "continuous"
            },
            "soil_sand_10cm": {
                "path": os.path.join(base_dir, "openlandmap", "sand_10cm_5km.tif"),
                "type": "continuous"
            },
            "soil_clay_10cm": {
                "path": os.path.join(base_dir, "openlandmap", "clay_10cm_5km.tif"),
                "type": "continuous"
            },
            "soil_sand_30cm": {
                "path": os.path.join(base_dir, "openlandmap", "sand_30cm_5km.tif"),
                "type": "continuous"
            },
            "soil_clay_30cm": {
                "path": os.path.join(base_dir, "openlandmap", "clay_30cm_5km.tif"),
                "type": "continuous"
            },
            "soil_sand_60cm": {
                "path": os.path.join(base_dir, "openlandmap", "sand_60cm_5km.tif"),
                "type": "continuous"
            },
            "soil_clay_60cm": {
                "path": os.path.join(base_dir, "openlandmap", "clay_60cm_5km.tif"),
                "type": "continuous"
            },
            "soil_sand_100cm": {
                "path": os.path.join(base_dir, "openlandmap", "sand_100cm_5km.tif"),
                "type": "continuous"
            },
            "soil_clay_100cm": {
                "path": os.path.join(base_dir, "openlandmap", "clay_100cm_5km.tif"),
                "type": "continuous"
            },
            "soil_sand_200cm": {
                "path": os.path.join(base_dir, "openlandmap", "sand_200cm_5km.tif"),
                "type": "continuous"
            },
            "soil_clay_200cm": {
                "path": os.path.join(base_dir, "openlandmap", "clay_200cm_5km.tif"),
                "type": "continuous"
            },
            
            # Topography
            "dem": {
                "path": os.path.join(base_dir, "usgs_dem", "dem.tif"),
                "type": "topography"  # Special type for DEM derivatives
            }
        }
        
        # Target CRS and transform for resampling
        target_bounds = (min(lon_coords), min(lat_coords), max(lon_coords), max(lat_coords))
        target_transform = rasterio.transform.from_bounds(*target_bounds, width, height)
        target_crs = 'EPSG:4326'
        
        for source_name, info in static_sources.items():
            file_path = info["path"]
            
            if not os.path.exists(file_path):
                print(f"   ⚠️ Skipping {source_name}: file not found at {file_path}")
                continue
                
            print(f"   Loading {source_name}...")
            
            try:
                with rasterio.open(file_path) as src:
                    # Reproject to target grid
                    resampled_data = np.full((height, width), np.nan, dtype=np.float32)
                    
                    rasterio.warp.reproject(
                        source=rasterio.band(src, 1),
                        destination=resampled_data,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=target_transform,
                        dst_crs=target_crs,
                        resampling=Resampling.bilinear if info["type"] == "continuous" else Resampling.nearest
                    )
                    
                    # Handle different data types
                    if info["type"] == "categorical":
                        # One-hot encode land cover classes
                        if "modis" in source_name:
                            for class_id in info["classes"]:
                                class_data = (resampled_data == class_id).astype(np.float32)
                                static_features.append(class_data)
                                static_names.append(f"modis_land_cover_class_{class_id}")
                    
                    elif info["type"] == "topography":
                        # Calculate DEM derivatives
                        elevation = resampled_data
                        static_features.append(elevation)
                        static_names.append("topo_elevation")
                        
                        # Calculate slope, aspect, curvature
                        derivatives = self._calculate_topographic_derivatives(elevation, lat_coords, lon_coords)
                        for deriv_name, deriv_data in derivatives.items():
                            static_features.append(deriv_data)
                            static_names.append(f"topo_{deriv_name}")
                    
                    else:  # continuous
                        static_features.append(resampled_data)
                        if "landscan" in source_name:
                            static_names.append("landscan_population")
                        else:
                            static_names.append(source_name)
                            
            except Exception as e:
                print(f"     ❌ Error loading {source_name}: {e}")
        
        if len(static_features) > 0:
            static_data = np.stack(static_features, axis=0)
            print(f"\n✅ Loaded {len(static_names)} static features:")
            for i, name in enumerate(static_names[:10]):  # Show first 10
                print(f"     {i+1:2d}. {name}")
            if len(static_names) > 10:
                print(f"     ... and {len(static_names)-10} more")
        else:
            print("\n⚠️ No static features could be loaded")
            static_data = None
            static_names = None
        
        return static_data, static_names
    
    def _calculate_topographic_derivatives(self, elevation: np.ndarray, lat_coords: list, lon_coords: list):
        """Calculate slope, aspect, and curvature from elevation data."""
        from scipy.ndimage import sobel
        
        # Convert degrees to meters (approximate)
        lat_center = np.mean(lat_coords)
        dx = abs(lon_coords[1] - lon_coords[0]) * 111320 * np.cos(np.radians(lat_center))  # meters
        dy = abs(lat_coords[1] - lat_coords[0]) * 111320  # meters
        
        # Calculate gradients
        grad_x = sobel(elevation, axis=1) / (8 * dx)
        grad_y = sobel(elevation, axis=0) / (8 * dy)
        
        # Slope (degrees)
        slope = np.arctan(np.sqrt(grad_x**2 + grad_y**2)) * 180 / np.pi
        
        # Aspect (degrees from north)
        aspect = np.arctan2(-grad_x, grad_y) * 180 / np.pi
        aspect[aspect < 0] += 360
        
        # Curvature (simple approximation)
        grad_xx = sobel(grad_x, axis=1) / (8 * dx)
        grad_yy = sobel(grad_y, axis=0) / (8 * dy)
        curvature = -(grad_xx + grad_yy)
        
        return {
            "slope": slope.astype(np.float32),
            "aspect": aspect.astype(np.float32), 
            "curvature": curvature.astype(np.float32)
        }
    
    def extend_features_temporal(self, fine_ds: xr.Dataset, target_times: pd.DatetimeIndex) -> xr.Dataset:
        """
        Extend features dataset to match target temporal coverage.
        
        Fills missing time steps using linear interpolation.
        
        Parameters:
        -----------
        fine_ds : xr.Dataset
            Original features dataset
        target_times : pd.DatetimeIndex
            Target time coverage (e.g., from gap-filled GRACE)
        
        Returns:
        --------
        xr.Dataset
            Extended features dataset
        """
        print(f"\n🕒 EXTENDING FEATURES TEMPORAL COVERAGE")
        print("="*50)
        
        current_times = pd.to_datetime(fine_ds.time.values)
        
        print(f"Current: {len(current_times)} months ({current_times.min().strftime('%Y-%m')} to {current_times.max().strftime('%Y-%m')})")
        print(f"Target:  {len(target_times)} months ({target_times.min().strftime('%Y-%m')} to {target_times.max().strftime('%Y-%m')})")
        
        # Find missing times
        missing_times = target_times[~target_times.isin(current_times)]
        print(f"Missing: {len(missing_times)} months")
        
        if len(missing_times) == 0:
            print("No extension needed - temporal coverage already matches")
            return fine_ds
        
        # Create extended time coordinate
        extended_times = pd.DatetimeIndex(sorted(set(current_times) | set(target_times)))
        
        # Extend temporal features using linear interpolation
        print(f"\n🔄 Interpolating temporal features...")
        
        # Reindex dataset to extended time grid  
        extended_ds = fine_ds.reindex(time=extended_times, method=None)
        
        # Interpolate temporal features in chunks to manage memory
        if 'features' in extended_ds.data_vars:
            print("   Interpolating feature stack...")
            # Process in chunks to avoid memory issues
            chunk_size = 20  # Process 20 features at a time
            n_features = extended_ds['features'].shape[1]
            
            for i in range(0, n_features, chunk_size):
                end_idx = min(i + chunk_size, n_features)
                print(f"     Processing features {i+1}-{end_idx} of {n_features}")
                
                feature_chunk = extended_ds['features'][:, i:end_idx, :, :]
                interpolated_chunk = feature_chunk.interpolate_na(
                    dim='time', 
                    method='linear',
                    fill_value='extrapolate'
                )
                extended_ds['features'][:, i:end_idx, :, :] = interpolated_chunk
        
        # Handle individual feature variables
        for var_name in extended_ds.data_vars:
            if var_name != 'features' and var_name != 'static_features':
                if 'time' in extended_ds[var_name].dims:
                    extended_ds[var_name] = extended_ds[var_name].interpolate_na(
                        dim='time',
                        method='linear', 
                        fill_value='extrapolate'
                    )
        
        # Static features don't need temporal extension
        
        print(f"✅ Temporal extension complete: {len(current_times)} → {len(extended_times)} months")
        return extended_ds
    
    def aggregate_feature(self, 
                         feature_data: np.ndarray,
                         feature_name: str,
                         method: str = None) -> np.ndarray:
        """
        Aggregate a single feature from fine to coarse resolution.
        
        Parameters:
        -----------
        feature_data : np.ndarray
            2D or 3D array (time, lat, lon) or (lat, lon)
        feature_name : str
            Name of feature (for determining aggregation method)
        method : str, optional
            Override aggregation method
        
        Returns:
        --------
        np.ndarray
            Coarsened feature array
        """
        # Determine aggregation method
        if method is None:
            method = get_feature_aggregation_method(feature_name, self.config)
        
        # Handle different dimensions
        if feature_data.ndim == 2:
            # Static feature (lat, lon)
            coarsened = coarsen_2d(feature_data, self.aggregation_factor, method)
        
        elif feature_data.ndim == 3:
            # Temporal feature (time, lat, lon)
            n_times = feature_data.shape[0]
            
            # Coarsen first slice to get output shape
            first_coarsened = coarsen_2d(feature_data[0], self.aggregation_factor, method)
            coarse_shape = (n_times,) + first_coarsened.shape
            
            # Initialize output
            coarsened = np.zeros(coarse_shape)
            coarsened[0] = first_coarsened
            
            # Process remaining time steps
            for t in range(1, n_times):
                coarsened[t] = coarsen_2d(feature_data[t], self.aggregation_factor, method)
        
        else:
            raise ValueError(f"Unsupported feature dimensions: {feature_data.ndim}")
        
        return coarsened
    
    def aggregate_feature_stack(self, 
                               fine_ds: xr.Dataset,
                               grace_mask: xr.DataArray = None) -> xr.Dataset:
        """
        Aggregate entire feature stack to coarse resolution.
        
        Parameters:
        -----------
        fine_ds : xr.Dataset
            Fine-resolution feature dataset
        grace_mask : xr.DataArray, optional
            GRACE data mask to apply
        
        Returns:
        --------
        xr.Dataset
            Coarse-resolution feature dataset
        """
        print("\n" + "="*70)
        print("🔄 AGGREGATING FEATURES TO COARSE RESOLUTION")
        print("="*70)
        
        # Check dataset structure and apply appropriate processing
        if 'features' in fine_ds.data_vars:
            # Stacked format - apply advanced features if enabled
            enable_advanced = self.config.get('feature_aggregation', {}).get('enable_advanced_features', False)
            if enable_advanced:
                print("🚀 Applying advanced feature engineering before aggregation...")
                fine_ds = self.add_advanced_features_to_stacked(fine_ds)
            
            return self._aggregate_stacked_format(fine_ds, grace_mask)
        else:
            # Variable format - apply advanced features if enabled
            enable_advanced = self.config.get('feature_aggregation', {}).get('enable_advanced_features', False)
            if enable_advanced:
                print("🚀 Applying advanced feature engineering before aggregation...")
                fine_ds = self.add_advanced_features(fine_ds)
                
            return self._aggregate_variable_format(fine_ds, grace_mask)
    
    def _aggregate_stacked_format(self, 
                                 fine_ds: xr.Dataset, 
                                 grace_mask: xr.DataArray = None) -> xr.Dataset:
        """
        Aggregate dataset in stacked format (time, feature, lat, lon).
        """
        feature_data = fine_ds['features'].values  # (time, feature, lat, lon)
        feature_names = fine_ds['feature'].values
        times = fine_ds['time'].values
        
        n_times, n_features, n_lat_fine, n_lon_fine = feature_data.shape
        
        print(f"📊 Processing {n_features} features across {n_times} time steps")
        
        # Calculate coarse dimensions
        n_lat_coarse = n_lat_fine // self.aggregation_factor
        n_lon_coarse = n_lon_fine // self.aggregation_factor
        
        # Initialize output array
        coarse_data = np.zeros((n_times, n_features, n_lat_coarse, n_lon_coarse))
        
        # Process each feature
        print("\n🔄 Aggregating features...")
        for i, feature_name in enumerate(tqdm(feature_names, desc="Features")):
            # Convert feature name from numpy string to regular string
            if hasattr(feature_name, 'item'):
                feature_name = feature_name.item()
            else:
                feature_name = str(feature_name)
            
            # Get aggregation method
            method = get_feature_aggregation_method(feature_name, self.config)
            
            # Extract feature data (time, lat, lon)
            feature_3d = feature_data[:, i, :, :]
            
            # Aggregate
            coarse_feature = self.aggregate_feature(feature_3d, feature_name, method)
            
            # Store
            coarse_data[:, i, :, :] = coarse_feature
        
        # Create coarse coordinates - align with GRACE grid if provided
        if grace_mask is not None:
            # Use GRACE coordinates to ensure alignment
            lat_coarse = grace_mask.lat.values
            lon_coarse = grace_mask.lon.values
            n_lat_coarse = len(lat_coarse)
            n_lon_coarse = len(lon_coarse)
            
            print(f"🎯 Aligning to GRACE grid: {n_lat_coarse}×{n_lon_coarse}")
            
            # Need to re-aggregate data to match GRACE grid
            # This is more complex - we need to interpolate from our aggregated grid to GRACE grid
            from scipy.interpolate import RegularGridInterpolator
            
            # Original coarse grid from simple aggregation
            lat_agg = fine_ds.lat.values[::self.aggregation_factor][:coarse_data.shape[2]]
            lon_agg = fine_ds.lon.values[::self.aggregation_factor][:coarse_data.shape[3]]
            
            # Initialize data aligned to GRACE grid
            aligned_data = np.full((n_times, n_features, n_lat_coarse, n_lon_coarse), np.nan)
            
            # Interpolate each time step and feature to GRACE grid
            print("🔄 Interpolating to GRACE grid...")
            for t in tqdm(range(n_times), desc="Time steps"):
                for f in range(n_features):
                    # Skip if all NaN
                    data_slice = coarse_data[t, f, :, :]
                    if not np.all(np.isnan(data_slice)):
                        # Create interpolator
                        interpolator = RegularGridInterpolator(
                            (lat_agg, lon_agg), 
                            data_slice,
                            bounds_error=False,
                            fill_value=np.nan
                        )
                        
                        # Create target grid
                        lon_grid, lat_grid = np.meshgrid(lon_coarse, lat_coarse)
                        target_points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
                        
                        # Interpolate
                        interpolated = interpolator(target_points).reshape(n_lat_coarse, n_lon_coarse)
                        aligned_data[t, f, :, :] = interpolated
            
            coarse_data = aligned_data
        else:
            # No GRACE mask - use simple aggregated coordinates
            lat_coarse = fine_ds.lat.values[::self.aggregation_factor][:n_lat_coarse]
            lon_coarse = fine_ds.lon.values[::self.aggregation_factor][:n_lon_coarse]
        
        # Process static features if they exist
        static_data = None
        static_feature_names = None
        if 'static_features' in fine_ds.data_vars:
            static_features_raw = fine_ds['static_features'].values  # (static_feature, lat, lon)
            static_feature_names = fine_ds['static_feature'].values
            n_static_features = len(static_feature_names)
            
            print(f"\n🔄 Aggregating {n_static_features} static features...")
            static_data = np.zeros((n_static_features, n_lat_coarse, n_lon_coarse))
            
            for i, static_feature_name in enumerate(tqdm(static_feature_names, desc="Static features")):
                # Convert feature name from numpy string to regular string
                if hasattr(static_feature_name, 'item'):
                    static_feature_name = static_feature_name.item()
                else:
                    static_feature_name = str(static_feature_name)
                
                # Get aggregation method
                method = get_feature_aggregation_method(static_feature_name, self.config)
                
                # Extract static feature data (lat, lon)
                static_feature_2d = static_features_raw[i, :, :]
                
                # Aggregate to simple grid first
                coarse_static_feature = self.aggregate_feature(static_feature_2d, static_feature_name, method)
                
                # If aligning to GRACE grid, interpolate static features too
                if grace_mask is not None:
                    # Interpolate static feature to GRACE grid
                    lat_agg = fine_ds.lat.values[::self.aggregation_factor][:coarse_static_feature.shape[0]]
                    lon_agg = fine_ds.lon.values[::self.aggregation_factor][:coarse_static_feature.shape[1]]
                    
                    if not np.all(np.isnan(coarse_static_feature)):
                        from scipy.interpolate import RegularGridInterpolator
                        interpolator = RegularGridInterpolator(
                            (lat_agg, lon_agg), 
                            coarse_static_feature,
                            bounds_error=False,
                            fill_value=np.nan
                        )
                        
                        # Create target grid
                        lon_grid, lat_grid = np.meshgrid(lon_coarse, lat_coarse)
                        target_points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
                        
                        # Interpolate
                        aligned_static = interpolator(target_points).reshape(n_lat_coarse, n_lon_coarse)
                        static_data[i, :, :] = aligned_static
                    else:
                        static_data[i, :, :] = np.nan
                else:
                    # Store original aggregated data
                    static_data[i, :, :] = coarse_static_feature

        # Create output dataset
        data_vars = {
            'features': (['time', 'feature', 'lat', 'lon'], coarse_data)
        }
        
        coords = {
            'time': times,
            'feature': feature_names,
            'lat': lat_coarse,
            'lon': lon_coarse
        }
        
        # Add static features if they exist
        if static_data is not None:
            data_vars['static_features'] = (['static_feature', 'lat', 'lon'], static_data)
            coords['static_feature'] = static_feature_names
        
        coarse_ds = xr.Dataset(data_vars, coords=coords)
        
        # Apply GRACE mask if provided
        if grace_mask is not None and self.mask_by_grace:
            print("\n🎭 Applying GRACE coverage mask...")
            coarse_ds = self._apply_grace_mask(coarse_ds, grace_mask)
        
        # Add metadata
        coarse_ds = add_metadata_to_dataset(coarse_ds, self.config, 'feature_aggregation')
        coarse_ds.attrs['aggregation_factor'] = self.aggregation_factor
        coarse_ds.attrs['original_resolution_deg'] = get_config_value(
            self.config, 'resolution.fine_resolution_deg', 0.05
        )
        coarse_ds.attrs['coarse_resolution_deg'] = get_config_value(
            self.config, 'resolution.grace_native_deg', 0.5
        )
        
        # Print summary
        print("\n✅ AGGREGATION COMPLETE")
        print(f"   Original shape: {n_times} × {n_features} × {n_lat_fine} × {n_lon_fine}")
        print(f"   Coarse shape:   {n_times} × {n_features} × {n_lat_coarse} × {n_lon_coarse}")
        if static_data is not None:
            print(f"   Static features: {static_data.shape[0]} × {n_lat_coarse} × {n_lon_coarse}")
        print(f"   Spatial reduction: {n_lat_fine*n_lon_fine:,} → {n_lat_coarse*n_lon_coarse:,} pixels "
              f"({100*(1 - (n_lat_coarse*n_lon_coarse)/(n_lat_fine*n_lon_fine)):.1f}% reduction)")
        
        return coarse_ds
    
    def _aggregate_variable_format(self, 
                                   fine_ds: xr.Dataset,
                                   grace_mask: xr.DataArray = None) -> xr.Dataset:
        """
        Aggregate dataset where each variable is separate.
        """
        # Get all data variables
        var_names = list(fine_ds.data_vars.keys())
        n_vars = len(var_names)
        
        print(f"📊 Processing {n_vars} variables")
        
        # Initialize output dataset
        coarse_ds = xr.Dataset()
        
        # Copy coordinates (will update later)
        for coord_name in ['time', 'lat', 'lon']:
            if coord_name in fine_ds.coords:
                coarse_ds.coords[coord_name] = fine_ds.coords[coord_name]
        
        # Process each variable
        print("\n🔄 Aggregating variables...")
        for var_name in tqdm(var_names, desc="Variables"):
            var_data = fine_ds[var_name].values
            
            # Aggregate
            method = get_feature_aggregation_method(var_name, self.config)
            coarse_var = self.aggregate_feature(var_data, var_name, method)
            
            # Add to dataset
            if var_data.ndim == 2:
                coarse_ds[var_name] = (['lat', 'lon'], coarse_var)
            elif var_data.ndim == 3:
                coarse_ds[var_name] = (['time', 'lat', 'lon'], coarse_var)
            
            # Copy attributes
            coarse_ds[var_name].attrs = fine_ds[var_name].attrs.copy()
            coarse_ds[var_name].attrs['aggregation_method'] = method
        
        # Update spatial coordinates
        n_lat_coarse = coarse_ds[var_names[0]].shape[-2]
        n_lon_coarse = coarse_ds[var_names[0]].shape[-1]
        
        lat_coarse = fine_ds.lat.values[::self.aggregation_factor][:n_lat_coarse]
        lon_coarse = fine_ds.lon.values[::self.aggregation_factor][:n_lon_coarse]
        
        coarse_ds = coarse_ds.assign_coords(lat=lat_coarse, lon=lon_coarse)
        
        # Apply GRACE mask
        if grace_mask is not None and self.mask_by_grace:
            print("\n🎭 Applying GRACE coverage mask...")
            coarse_ds = self._apply_grace_mask(coarse_ds, grace_mask)
        
        # Add metadata
        coarse_ds = add_metadata_to_dataset(coarse_ds, self.config, 'feature_aggregation')
        coarse_ds.attrs['aggregation_factor'] = self.aggregation_factor
        
        print(f"\n✅ AGGREGATION COMPLETE: {n_vars} variables processed")
        
        return coarse_ds
    
    def _apply_grace_mask(self, 
                         coarse_ds: xr.Dataset,
                         grace_mask: xr.DataArray) -> xr.Dataset:
        """
        Apply GRACE coverage mask to features.
        
        Sets features to NaN where GRACE has no data.
        """
        # Ensure grace_mask matches coarse grid
        if grace_mask.shape != (len(coarse_ds.lat), len(coarse_ds.lon)):
            # Interpolate mask to coarse grid
            grace_mask_aligned = grace_mask.interp(
                lat=coarse_ds.lat,
                lon=coarse_ds.lon,
                method='nearest'
            )
        else:
            grace_mask_aligned = grace_mask
        
        # Apply mask to all variables
        for var_name in coarse_ds.data_vars:
            var_data = coarse_ds[var_name]
            
            if var_data.ndim == 2:  # (lat, lon)
                coarse_ds[var_name] = var_data.where(~np.isnan(grace_mask_aligned))
            elif var_data.ndim == 3:  # (time, lat, lon)
                # Broadcast mask across time
                mask_3d = np.broadcast_to(
                    np.isnan(grace_mask_aligned.values),
                    var_data.shape
                )
                coarse_ds[var_name] = var_data.where(~mask_3d)
            elif var_data.ndim == 4:  # (time, feature, lat, lon)
                # Broadcast mask across time and features
                mask_4d = np.broadcast_to(
                    np.isnan(grace_mask_aligned.values)[np.newaxis, np.newaxis, :, :],
                    var_data.shape
                )
                coarse_ds[var_name] = var_data.where(~mask_4d)
        
        n_masked = np.sum(np.isnan(grace_mask_aligned.values))
        n_total = grace_mask_aligned.size
        print(f"   Masked {n_masked}/{n_total} pixels ({100*n_masked/n_total:.1f}%) with no GRACE data")
        
        return coarse_ds
    
    def save_coarse_features(self, coarse_ds: xr.Dataset, output_path: str):
        """
        Save coarsened features to NetCDF.
        
        Parameters:
        -----------
        coarse_ds : xr.Dataset
            Coarse-resolution features
        output_path : str
            Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Compress for storage
        encoding = {}
        for var in coarse_ds.data_vars:
            encoding[var] = {
                'zlib': True,
                'complevel': 4,
                'dtype': 'float32'  # Use float32 to save space
            }
        
        print(f"\n💾 Saving coarse features to: {output_path}")
        coarse_ds.to_netcdf(output_path, encoding=encoding)
        
        # Report file size
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"   File size: {file_size_mb:.1f} MB")
        
        print("✅ Coarse features saved successfully")
    
    def add_advanced_features_to_stacked(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Add advanced feature engineering to stacked format dataset.
        
        This works with the format: ds['features'] shape (time, feature, lat, lon)
        
        Parameters:
        -----------
        ds : xr.Dataset
            Input dataset with 'features' array
            
        Returns:
        --------
        xr.Dataset
            Enhanced dataset with advanced features
        """
        print("\n" + "="*70)
        print("🚀 ADDING ADVANCED FEATURES (STACKED FORMAT)")
        print("="*70)
        print("   This should significantly improve model performance!")
        print("   Expected R² improvement: +0.1 to +0.3")
        
        # Get data arrays
        feature_data = ds['features'].values  # (time, feature, lat, lon)
        feature_names = ds['feature'].values  # feature names
        
        n_times, n_features, n_lat, n_lon = feature_data.shape
        print(f"\n📊 Input shape: {n_times} times × {n_features} features × {n_lat}×{n_lon} spatial")
        
        # Apply advanced feature engineering
        enhanced_data, enhanced_names = self._add_advanced_features_stacked(
            feature_data, feature_names.tolist()
        )
        
        print(f"📊 Enhanced shape: {enhanced_data.shape[0]} times × {enhanced_data.shape[1]} features × {enhanced_data.shape[2]}×{enhanced_data.shape[3]} spatial")
        print(f"   Feature count: {enhanced_data.shape[1]} (was {n_features})")
        
        # Create completely new dataset to avoid dimension conflicts
        enhanced_ds = xr.Dataset()
        
        # Add coordinates
        enhanced_ds = enhanced_ds.assign_coords({
            'time': ds.time,
            'lat': ds.lat,
            'lon': ds.lon,
            'feature': enhanced_names,
            'static_feature': ds.static_feature
        })
        
        # Add enhanced features
        enhanced_ds['features'] = (['time', 'feature', 'lat', 'lon'], enhanced_data.astype(np.float32))
        
        # Copy static features unchanged
        enhanced_ds['static_features'] = ds['static_features']
        
        # Copy original attributes and add enhancement info
        enhanced_ds.attrs = ds.attrs.copy()
        enhanced_ds.attrs['enhanced_features'] = True
        enhanced_ds.attrs['enhancement_methods'] = 'seasonal_anomalies,temporal_lags,spatial_lags,precipitation_accumulation'
        
        print("✅ Advanced feature engineering complete!")
        return enhanced_ds
    
    def _add_advanced_features_stacked(self, feature_data: np.ndarray, feature_names: list):
        """Apply advanced feature engineering to stacked data format."""
        print("\n🔄 Step 1: Adding seasonal anomaly features...")
        feature_data, feature_names = self._add_seasonal_anomalies_stacked(feature_data, feature_names)
        
        print("\n🔄 Step 2: Adding temporal lag features...")
        feature_data, feature_names = self._add_temporal_lags_stacked(feature_data, feature_names)
        
        print("\n🔄 Step 3: Adding spatial lag features...")
        feature_data, feature_names = self._add_spatial_lags_stacked(feature_data, feature_names)
        
        print("\n🔄 Step 4: Adding precipitation accumulation features...")
        feature_data, feature_names = self._add_precipitation_stacked(feature_data, feature_names)
        
        return feature_data, feature_names
    
    def _add_seasonal_anomalies_stacked(self, feature_data: np.ndarray, feature_names: list):
        """Add seasonal anomalies for stacked format."""
        anomaly_candidates = ['tmean', 'pr', 'chirps', 'Evap_tavg', 'aet', 'def', 'SWE_inst']
        
        # Find feature indices
        anomaly_indices = []
        for i, name in enumerate(feature_names):
            if any(candidate in name for candidate in anomaly_candidates):
                anomaly_indices.append(i)
        
        if not anomaly_indices:
            return feature_data, feature_names
            
        print(f"   📍 Creating anomalies for {len(anomaly_indices)} features")
        
        new_features = []
        new_names = []
        n_times = feature_data.shape[0]
        
        for feat_idx in anomaly_indices:
            feat_name = feature_names[feat_idx]
            data = feature_data[:, feat_idx, :, :]  # (time, lat, lon)
            
            # Calculate monthly climatology
            climatology = np.zeros((12, data.shape[1], data.shape[2]))
            for month in range(12):
                month_indices = [t for t in range(n_times) if t % 12 == month]
                if month_indices:
                    climatology[month] = np.nanmean(data[month_indices], axis=0)
            
            # Calculate anomalies
            anomaly_data = np.zeros_like(data)
            for t in range(n_times):
                month = t % 12
                anomaly_data[t] = data[t] - climatology[month]
            
            new_features.append(anomaly_data)
            new_names.append(f"{feat_name}_anom")
        
        if new_features:
            # Stack new features and concatenate
            new_stack = np.stack(new_features, axis=1)  # (time, new_features, lat, lon)
            feature_data = np.concatenate([feature_data, new_stack], axis=1)
            feature_names.extend(new_names)
        
        return feature_data, feature_names
    
    def _add_temporal_lags_stacked(self, feature_data: np.ndarray, feature_names: list):
        """Add temporal lags for stacked format."""
        lag_candidates = ['tmean', 'pr', 'chirps', 'Evap_tavg', 'aet', 'def', 'SWE_inst']
        lag_months = [1, 3, 6]
        
        # Find indices (avoid lags of already processed features)
        lag_indices = []
        for i, name in enumerate(feature_names):
            if any(candidate in name for candidate in lag_candidates):
                if not any(suffix in name for suffix in ['_lag', '_anom']):
                    lag_indices.append(i)
        
        if not lag_indices:
            return feature_data, feature_names
            
        print(f"   📍 Creating lags for {len(lag_indices)} features, {len(lag_months)} time periods")
        
        new_features = []
        new_names = []
        n_times = feature_data.shape[0]
        
        for feat_idx in lag_indices:
            feat_name = feature_names[feat_idx]
            for lag in lag_months:
                if lag < n_times:
                    data = feature_data[:, feat_idx, :, :]
                    
                    # Create lagged version
                    lagged_data = np.zeros_like(data)
                    lagged_data[lag:] = data[:-lag]
                    lagged_data[:lag] = np.nan
                    
                    new_features.append(lagged_data)
                    new_names.append(f"{feat_name}_lag{lag}")
        
        if new_features:
            new_stack = np.stack(new_features, axis=1)
            feature_data = np.concatenate([feature_data, new_stack], axis=1)
            feature_names.extend(new_names)
        
        return feature_data, feature_names
    
    def _add_spatial_lags_stacked(self, feature_data: np.ndarray, feature_names: list):
        """Add spatial lags for stacked format."""
        from scipy import ndimage
        
        spatial_candidates = ['tmean', 'pr', 'chirps', 'Evap_tavg', 'aet', 'SWE_inst']
        
        # Find indices (focus on key variables)
        spatial_indices = []
        for i, name in enumerate(feature_names):
            if any(candidate in name for candidate in spatial_candidates):
                if not any(suffix in name for suffix in ['_lag3', '_lag6', '_spatial']):
                    spatial_indices.append(i)
        
        if not spatial_indices:
            return feature_data, feature_names
            
        print(f"   📍 Creating spatial lags for {len(spatial_indices)} features")
        
        # 3x3 averaging kernel
        kernel = np.ones((3, 3)) / 9
        
        new_features = []
        new_names = []
        
        for feat_idx in spatial_indices:
            feat_name = feature_names[feat_idx]
            data = feature_data[:, feat_idx, :, :]
            
            # Apply spatial smoothing for each time step
            smoothed_data = np.zeros_like(data)
            for t in range(data.shape[0]):
                smoothed_data[t] = ndimage.convolve(
                    data[t], kernel, mode='nearest'
                )
            
            new_features.append(smoothed_data)
            new_names.append(f"{feat_name}_spatial")
        
        if new_features:
            new_stack = np.stack(new_features, axis=1)
            feature_data = np.concatenate([feature_data, new_stack], axis=1)
            feature_names.extend(new_names)
        
        return feature_data, feature_names
    
    def _add_precipitation_stacked(self, feature_data: np.ndarray, feature_names: list):
        """Add precipitation accumulation for stacked format."""
        precip_candidates = ['pr', 'chirps']
        
        # Find precipitation indices
        precip_indices = []
        for i, name in enumerate(feature_names):
            if any(candidate in name for candidate in precip_candidates):
                if not any(suffix in name for suffix in ['_lag', '_anom', '_spatial', '_accum']):
                    precip_indices.append(i)
        
        if not precip_indices:
            return feature_data, feature_names
            
        print(f"   📍 Creating accumulation features for {len(precip_indices)} precipitation variables")
        
        accumulation_periods = [3, 6, 12]  # months
        new_features = []
        new_names = []
        
        for feat_idx in precip_indices:
            feat_name = feature_names[feat_idx]
            data = feature_data[:, feat_idx, :, :]
            
            for period in accumulation_periods:
                if period <= data.shape[0]:
                    # Calculate rolling sum
                    accum_data = np.zeros_like(data)
                    
                    for t in range(data.shape[0]):
                        start_idx = max(0, t - period + 1)
                        accum_data[t] = np.nansum(data[start_idx:t+1], axis=0)
                    
                    new_features.append(accum_data)
                    new_names.append(f"{feat_name}_accum{period}m")
        
        if new_features:
            new_stack = np.stack(new_features, axis=1)
            feature_data = np.concatenate([feature_data, new_stack], axis=1)
            feature_names.extend(new_names)
        
        return feature_data, feature_names

    def add_advanced_features(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Add advanced feature engineering: seasonal anomalies, temporal lags, spatial lags.
        
        This integrates the sophisticated features from src/features.py that were missing
        from the basic feature set, which is likely causing the overfitting.
        
        Parameters:
        -----------
        ds : xr.Dataset
            Input dataset with basic features
            
        Returns:
        --------
        xr.Dataset
            Enhanced dataset with advanced features
        """
        print("\n" + "="*70)
        print("🚀 ADDING ADVANCED FEATURE ENGINEERING")
        print("="*70)
        print("   This should significantly improve model performance!")
        print("   Expected R² improvement: +0.1 to +0.3")
        
        # Convert to numpy array format for processing
        feature_names = list(ds.data_vars)
        n_times, n_lat, n_lon = ds.dims['time'], ds.dims['lat'], ds.dims['lon']
        
        # Stack features into array format (time, features, lat, lon)
        print(f"\n📊 Input shape: {n_times} times × {len(feature_names)} features × {n_lat}×{n_lon} spatial")
        
        feature_stack = np.zeros((n_times, len(feature_names), n_lat, n_lon))
        for i, var_name in enumerate(feature_names):
            feature_stack[:, i, :, :] = ds[var_name].values
        
        # Step 1: Add seasonal anomaly features
        print("\n🔄 Step 1: Adding seasonal anomaly features...")
        feature_stack, feature_names = self._add_seasonal_anomaly_features(feature_stack, feature_names)
        
        # Step 2: Add temporal lag features
        print("\n🔄 Step 2: Adding temporal lag features...")
        feature_stack, feature_names = self._add_temporal_lag_features(feature_stack, feature_names)
        
        # Step 3: Add spatial lag features
        print("\n🔄 Step 3: Adding spatial lag features...")
        feature_stack, feature_names = self._add_spatial_lag_features(feature_stack, feature_names)
        
        # Step 4: Add precipitation accumulation features
        print("\n🔄 Step 4: Adding precipitation accumulation features...")
        feature_stack, feature_names = self._add_precipitation_features(feature_stack, feature_names)
        
        # Convert back to xarray Dataset
        print(f"\n📊 Enhanced shape: {feature_stack.shape[0]} times × {feature_stack.shape[1]} features × {feature_stack.shape[2]}×{feature_stack.shape[3]} spatial")
        print(f"   Feature count: {len(feature_names)} (was {len(ds.data_vars)})")
        
        # Create new dataset
        enhanced_ds = xr.Dataset()
        
        # Copy coordinates
        enhanced_ds = enhanced_ds.assign_coords({
            'time': ds.time,
            'lat': ds.lat,
            'lon': ds.lon
        })
        
        # Add enhanced features
        for i, var_name in enumerate(feature_names):
            enhanced_ds[var_name] = (['time', 'lat', 'lon'], 
                                   feature_stack[:, i, :, :].astype(np.float32))
        
        # Copy attributes
        enhanced_ds.attrs = ds.attrs
        enhanced_ds.attrs['enhanced_features'] = True
        enhanced_ds.attrs['enhancement_methods'] = 'seasonal_anomalies,temporal_lags,spatial_lags,precipitation_accumulation'
        
        print("✅ Advanced feature engineering complete!")
        return enhanced_ds
    
    def _add_seasonal_anomaly_features(self, temporal_stack: np.ndarray, feature_names: list):
        """Add seasonal anomaly features - critical for detecting storage changes."""
        anomaly_candidates = ['tmean', 'pr', 'chirps', 'Evap_tavg', 'aet', 'def', 'SWE_inst']
        
        # Find indices
        anomaly_indices = []
        anomaly_names = []
        for i, name in enumerate(feature_names):
            if any(candidate in name for candidate in anomaly_candidates):
                anomaly_indices.append(i)
                anomaly_names.append(name)
        
        if not anomaly_indices:
            return temporal_stack, feature_names
            
        print(f"   📍 Creating anomalies for {len(anomaly_indices)} features")
        
        n_times = temporal_stack.shape[0]
        new_features = []
        new_names = []
        
        for feat_idx, feat_name in zip(anomaly_indices, anomaly_names):
            feature_data = temporal_stack[:, feat_idx, :, :]
            
            # Calculate monthly climatology (12 months)
            climatology = np.zeros((12, feature_data.shape[1], feature_data.shape[2]))
            
            for month in range(12):
                month_indices = [t for t in range(n_times) if t % 12 == month]
                if month_indices:
                    climatology[month] = np.nanmean(feature_data[month_indices], axis=0)
            
            # Calculate anomalies
            anomaly_data = np.zeros_like(feature_data)
            for t in range(n_times):
                month = t % 12
                anomaly_data[t] = feature_data[t] - climatology[month]
            
            new_features.append(anomaly_data)
            new_names.append(f"{feat_name}_anom")
        
        if new_features:
            # Stack new features
            new_stack = np.stack(new_features, axis=1)  # (time, new_features, lat, lon)
            temporal_stack = np.concatenate([temporal_stack, new_stack], axis=1)
            feature_names.extend(new_names)
        
        return temporal_stack, feature_names
    
    def _add_temporal_lag_features(self, temporal_stack: np.ndarray, feature_names: list):
        """Add temporal lag features - groundwater responds to past conditions."""
        lag_candidates = ['tmean', 'pr', 'chirps', 'Evap_tavg', 'aet', 'def', 'SWE_inst']
        lag_months = [1, 3, 6]
        
        # Find indices (avoid creating lags of lags)
        lag_indices = []
        lag_names = []
        for i, name in enumerate(feature_names):
            if any(candidate in name for candidate in lag_candidates):
                if not any(suffix in name for suffix in ['_lag', '_anom']):
                    lag_indices.append(i)
                    lag_names.append(name)
        
        if not lag_indices:
            return temporal_stack, feature_names
            
        print(f"   📍 Creating lags for {len(lag_indices)} features, {len(lag_months)} time periods")
        
        n_times = temporal_stack.shape[0]
        new_features = []
        new_names = []
        
        for feat_idx, feat_name in zip(lag_indices, lag_names):
            for lag in lag_months:
                if lag < n_times:
                    # Create lagged version
                    lagged_data = np.zeros_like(temporal_stack[:, feat_idx, :, :])
                    lagged_data[lag:] = temporal_stack[:-lag, feat_idx, :, :]
                    lagged_data[:lag] = np.nan  # Fill initial periods with NaN
                    
                    new_features.append(lagged_data)
                    new_names.append(f"{feat_name}_lag{lag}")
        
        if new_features:
            new_stack = np.stack(new_features, axis=1)
            temporal_stack = np.concatenate([temporal_stack, new_stack], axis=1)
            feature_names.extend(new_names)
        
        return temporal_stack, feature_names
    
    def _add_spatial_lag_features(self, temporal_stack: np.ndarray, feature_names: list):
        """Add spatial lag features - groundwater flows spatially."""
        from scipy import ndimage
        
        spatial_candidates = ['tmean', 'pr', 'chirps', 'Evap_tavg', 'aet', 'SWE_inst']
        
        # Find indices (focus on key variables, avoid over-processing)
        spatial_indices = []
        spatial_names = []
        for i, name in enumerate(feature_names):
            if any(candidate in name for candidate in spatial_candidates):
                if not any(suffix in name for suffix in ['_lag3', '_lag6', '_spatial']):
                    spatial_indices.append(i)
                    spatial_names.append(name)
        
        if not spatial_indices:
            return temporal_stack, feature_names
            
        print(f"   📍 Creating spatial lags for {len(spatial_indices)} features")
        
        # 3x3 averaging kernel
        kernel = np.ones((3, 3)) / 9
        
        new_features = []
        new_names = []
        
        for feat_idx, feat_name in zip(spatial_indices, spatial_names):
            # Apply spatial smoothing for each time step
            smoothed_data = np.zeros_like(temporal_stack[:, feat_idx, :, :])
            
            for t in range(temporal_stack.shape[0]):
                # Apply convolution with proper boundary handling
                smoothed_data[t] = ndimage.convolve(
                    temporal_stack[t, feat_idx, :, :], 
                    kernel, 
                    mode='nearest'
                )
            
            new_features.append(smoothed_data)
            new_names.append(f"{feat_name}_spatial")
        
        if new_features:
            new_stack = np.stack(new_features, axis=1)
            temporal_stack = np.concatenate([temporal_stack, new_stack], axis=1)
            feature_names.extend(new_names)
        
        return temporal_stack, feature_names
    
    def _add_precipitation_features(self, temporal_stack: np.ndarray, feature_names: list):
        """Add precipitation accumulation features."""
        precip_candidates = ['pr', 'chirps']
        
        # Find precipitation indices
        precip_indices = []
        precip_names = []
        for i, name in enumerate(feature_names):
            if any(candidate in name for candidate in precip_candidates):
                if not any(suffix in name for suffix in ['_lag', '_anom', '_spatial', '_accum']):
                    precip_indices.append(i)
                    precip_names.append(name)
        
        if not precip_indices:
            return temporal_stack, feature_names
            
        print(f"   📍 Creating accumulation features for {len(precip_indices)} precipitation variables")
        
        accumulation_periods = [3, 6, 12]  # months
        new_features = []
        new_names = []
        
        for feat_idx, feat_name in zip(precip_indices, precip_names):
            for period in accumulation_periods:
                if period <= temporal_stack.shape[0]:
                    # Calculate rolling sum
                    accum_data = np.zeros_like(temporal_stack[:, feat_idx, :, :])
                    
                    for t in range(temporal_stack.shape[0]):
                        start_idx = max(0, t - period + 1)
                        accum_data[t] = np.nansum(temporal_stack[start_idx:t+1, feat_idx, :, :], axis=0)
                    
                    new_features.append(accum_data)
                    new_names.append(f"{feat_name}_accum{period}m")
        
        if new_features:
            new_stack = np.stack(new_features, axis=1)
            temporal_stack = np.concatenate([temporal_stack, new_stack], axis=1)
            feature_names.extend(new_names)
        
        return temporal_stack, feature_names


def main():
    """Command-line interface for feature aggregation."""
    import argparse
    from src_new_approach.utils_downscaling import load_config, create_output_directories
    
    parser = argparse.ArgumentParser(
        description="Aggregate fine features to GRACE resolution"
    )
    parser.add_argument('--config', 
                       default='src_new_approach/config_coarse_to_fine.yaml',
                       help='Configuration file')
    parser.add_argument('--features-fine', 
                       help='Input fine features (overrides config)')
    parser.add_argument('--output', 
                       help='Output path (overrides config)')
    parser.add_argument('--grace-mask',
                       help='GRACE mask NetCDF file (optional)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    create_output_directories(config)
    
    # Get paths
    features_fine_path = args.features_fine or config['paths']['feature_stack_fine']
    output_path = args.output or config['paths']['feature_stack_coarse']
    
    # Initialize aggregator
    aggregator = FeatureAggregator(config)
    
    # Load fine features
    fine_ds = aggregator.load_fine_features(features_fine_path)
    
    # Load GRACE mask if provided
    grace_mask = None
    if args.grace_mask:
        print(f"\n📂 Loading GRACE mask from: {args.grace_mask}")
        grace_ds = xr.open_dataset(args.grace_mask)
        # Use first time step as mask template
        grace_mask = grace_ds['tws_anomaly'].isel(time=0)
        print(f"   GRACE mask shape: {grace_mask.shape}")
    
    # Aggregate features
    coarse_ds = aggregator.aggregate_feature_stack(fine_ds, grace_mask)
    
    # Save
    aggregator.save_coarse_features(coarse_ds, output_path)
    
    print("\n✅ Feature aggregation complete!")
    print(f"   Coarse features: {output_path}")


if __name__ == "__main__":
    main()

