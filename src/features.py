#!/usr/bin/env python3
"""
Enhanced features.py with scientific resampling - Compatible with existing pipeline.

This version:
1. Produces the exact same output structure as original
2. Uses scientifically correct resampling internally  
3. Works as drop-in replacement
4. Adds metadata about methods used
5. Optionally handles categorical encoding (can be disabled for compatibility)
"""

import os
import numpy as np
import rioxarray
import xarray as xr
from tqdm import tqdm
from src.utils import parse_grace_months
import rasterio.enums
import warnings
warnings.filterwarnings('ignore')

# Configuration flag for backward compatibility
ENABLE_CATEGORICAL_ENCODING = False  # Set to True to enable one-hot encoding
SCIENTIFIC_RESAMPLING = True  # Set to False to use original bilinear for everything


def get_resampling_method(data_name):
    """Get appropriate resampling method for data type."""
    if not SCIENTIFIC_RESAMPLING:
        # Compatibility mode - use defaults from original
        if "landscan" in data_name.lower():
            return rasterio.enums.Resampling.sum
        else:
            return rasterio.enums.Resampling.bilinear
    
    # Scientific mode - use appropriate methods
    data_name_lower = data_name.lower()
    
    # Categorical data
    if any(cat in data_name_lower for cat in ['land_cover', 'modis', 'lc_type']):
        return rasterio.enums.Resampling.nearest
    
    # Population/count data
    elif any(pop in data_name_lower for pop in ['population', 'landscan']):
        return rasterio.enums.Resampling.sum
    
    # Elevation
    elif any(elev in data_name_lower for elev in ['elevation', 'dem', 'srtm']):
        return rasterio.enums.Resampling.cubic
    
    # Precipitation (use average to approximate mass conservation)
    elif any(precip in data_name_lower for precip in ['precipitation', 'chirps', 'pr']):
        return rasterio.enums.Resampling.average
    
    # Default for continuous variables
    else:
        return rasterio.enums.Resampling.bilinear


def get_valid_grace_months(grace_dir):
    return parse_grace_months(grace_dir)


def get_chirps_reference_raster(chirps_dir):
    """Use CHIRPS as reference - maintains compatibility."""
    chirps_files = sorted(os.listdir(chirps_dir))
    chirps_path = os.path.join(chirps_dir, chirps_files[0])
    reference_raster = rioxarray.open_rasterio(chirps_path, masked=True).squeeze()
    return reference_raster


def get_grace_reference_raster(grace_dir, valid_months):
    """Original function for compatibility."""
    grace_files = sorted(os.listdir(grace_dir))
    if len(grace_files) == 0:
        raise ValueError(f"No GRACE files found in {grace_dir}")
    grace_path = os.path.join(grace_dir, grace_files[0])
    reference_raster = rioxarray.open_rasterio(grace_path, masked=True).squeeze()
    print(f"✅ Using GRACE reference shape: {reference_raster.shape}")
    return reference_raster


def load_feature_array(folder_path, months, suffix=".tif", filename_style="numeric", reference_raster=None):
    """Load feature array with scientific resampling while maintaining compatibility."""
    arr_list = []
    
    # Determine data type from folder path
    folder_name = os.path.basename(folder_path).lower()
    parent_name = os.path.basename(os.path.dirname(folder_path)).lower()
    
    # Get appropriate resampling method
    resample_method = get_resampling_method(f"{parent_name}_{folder_name}")
    
    if SCIENTIFIC_RESAMPLING and resample_method != rasterio.enums.Resampling.bilinear:
        print(f"  📊 {folder_name}: using {resample_method.name} resampling")

    if filename_style == "numeric":
        all_files = sorted(os.listdir(folder_path), key=lambda x: int(os.path.splitext(x)[0]))
        if len(all_files) < len(months):
            raise ValueError(f"Not enough files in {folder_path}: found {len(all_files)}, expected {len(months)}")
        for idx, m in enumerate(months):
            fname = f"{idx}{suffix}"
            fpath = os.path.join(folder_path, fname)
            if os.path.exists(fpath):
                raster = rioxarray.open_rasterio(fpath, masked=True).squeeze()
                if reference_raster is not None:
                    raster = raster.rio.reproject_match(reference_raster, resampling=resample_method)
                    
                    # For categorical data, ensure integer values
                    if resample_method == rasterio.enums.Resampling.nearest:
                        raster.values = np.round(raster.values).astype(raster.values.dtype)
                        
                arr_list.append(raster.values)
            else:
                raise FileNotFoundError(f"Missing: {fname} for month {m} in {folder_path}")

    elif filename_style == "yyyymm":
        for m in months:
            fname = f"{m.replace('-', '')}{suffix}"
            fpath = os.path.join(folder_path, fname)
            if os.path.exists(fpath):
                raster = rioxarray.open_rasterio(fpath, masked=True).squeeze()
                if reference_raster is not None:
                    raster = raster.rio.reproject_match(reference_raster, resampling=resample_method)
                    
                    # For categorical data, ensure integer values
                    if resample_method == rasterio.enums.Resampling.nearest:
                        raster.values = np.round(raster.values).astype(raster.values.dtype)
                        
                arr_list.append(raster.values)
            else:
                raise FileNotFoundError(f"Missing: {fname} for month {m} in {folder_path}")

    else:
        raise ValueError("Unknown filename_style")

    shapes = [a.shape for a in arr_list]
    if len(set(shapes)) != 1:
        raise ValueError(f"Mismatched array shapes in {folder_path}: {shapes}")

    return np.stack(arr_list, axis=0), raster


def load_static_features(static_dirs, reference_raster):
    """Load static features with scientific resampling."""
    static_arrays = []
    static_names = []
    categorical_info = {}  # Track categorical features

    for name, path in static_dirs.items():
        try:
            raster = rioxarray.open_rasterio(path, masked=True).squeeze()
            if reference_raster is not None:
                # Get appropriate resampling method
                resample_method = get_resampling_method(name)
                
                if SCIENTIFIC_RESAMPLING:
                    # Special handling for population to preserve totals
                    if "landscan" in name.lower():
                        # First resample with sum to preserve total population
                        raster = raster.rio.reproject_match(
                            reference_raster,
                            resampling=rasterio.enums.Resampling.sum
                        )
                    else:
                        # Standard resampling for other data types
                        raster = raster.rio.reproject_match(reference_raster, resampling=resample_method)
                        
                        # For categorical data, ensure integer values
                        if resample_method == rasterio.enums.Resampling.nearest:
                            raster.values = np.round(raster.values).astype(int)
                            # Track categorical features for potential encoding
                            unique_vals = np.unique(raster.values[~np.isnan(raster.values)])
                            categorical_info[name] = unique_vals
                else:
                    # Original behavior
                    raster = raster.rio.reproject_match(reference_raster)
                    
            static_arrays.append(raster.values)
            static_names.append(name)
        except Exception as e:
            print(f"❌ Static {name}: {e}")

    if len(static_arrays) > 0:
        static_stack = np.stack(static_arrays, axis=0)
        
        # Optional: One-hot encode categorical features
        if ENABLE_CATEGORICAL_ENCODING and categorical_info:
            print("🔢 One-hot encoding categorical features...")
            encoded_arrays = []
            encoded_names = []
            
            for i, (arr, name) in enumerate(zip(static_stack, static_names)):
                if name in categorical_info:
                    # One-hot encode this feature
                    unique_vals = categorical_info[name]
                    print(f"  Encoding {name}: {len(unique_vals)} classes")
                    
                    for val in unique_vals:
                        mask = (arr == val).astype(float)
                        encoded_arrays.append(mask)
                        encoded_names.append(f"{name}_class_{int(val)}")
                else:
                    # Keep as-is
                    encoded_arrays.append(arr)
                    encoded_names.append(name)
            
            static_stack = np.stack(encoded_arrays, axis=0)
            static_names = encoded_names
            
        return static_stack, static_names
    else:
        return None, []


def main():
    """Main function - maintains exact compatibility with original pipeline."""
    grace_dir = "data/raw/grace"
    valid_months = get_valid_grace_months(grace_dir)
    print(f"✅ Using {len(valid_months)} months aligned with GRACE")

    # Use CHIRPS as reference (maintains compatibility)
    reference_raster = get_chirps_reference_raster("data/raw/chirps")
    
    # Alternative: use GRACE reference
    # reference_raster = get_grace_reference_raster(grace_dir, valid_months)

    base_dir = "data/raw"
    datasets = {
        "gldas": [
            "Evap_tavg",
            "SWE_inst",
            "SoilMoi0_10cm_inst",
            "SoilMoi10_40cm_inst",
            "SoilMoi40_100cm_inst",
            "SoilMoi100_200cm_inst"
        ],
        "chirps": [""],
        "terraclimate": ["aet", "def", "pr", "tmmn", "tmmx"]
    }

    feature_arrays = []
    aligned_datasets = []
    feature_names = []

    # Process time-varying features
    for group, names in datasets.items():
        for name in tqdm(names, desc=f"📦 Stacking {group}"):
            folder = os.path.join(base_dir, group, name)
            try:
                if group in ["gldas", "chirps"]:
                    arr, _ = load_feature_array(folder, valid_months, filename_style="numeric", reference_raster=reference_raster)
                elif group == "terraclimate":
                    arr, _ = load_feature_array(folder, valid_months, filename_style="yyyymm", reference_raster=reference_raster)
                else:
                    print(f"ℹ️ Skipping {group}/{name} - will process as static feature if available")
                    continue
                feature_arrays.append(arr)
                aligned_datasets.append(name if name else group)
                feature_names.extend([f"{name if name else group}_{m}" for m in valid_months])
            except Exception as e:
                print(f"❌ {name}: {e}")

    # Print shapes for verification
    for i, (arr, name) in enumerate(zip(feature_arrays, aligned_datasets)):
        print(f"📐 {name}: shape = {arr.shape}")

    if len(feature_arrays) == 0:
        print("❌ No features aligned, exiting.")
        return

    try:
        temporal_stack = np.stack(feature_arrays, axis=1)  # shape: (time, feature, lat, lon)
    except Exception as e:
        print(f"❌ Stack failed: {e}")
        return

    print(f"✅ Final stacked shape: {temporal_stack.shape}")

    # Load static features
    static_files = {
        "modis_land_cover": os.path.join(base_dir, "modis_land_cover", "2003_01_01.tif"),
        "usgs_dem": os.path.join(base_dir, "usgs_dem", "srtm_dem.tif"),
        "landscan_population": os.path.join(base_dir, "landscan", "2003.tif"),
    }
    
    # Add soil properties
    for depth in ["0cm", "10cm", "30cm", "60cm", "100cm", "200cm"]:
        for var in ["sand", "clay"]:  # Note: original only has sand and clay, not silt
            name = f"{var}_{depth}"
            path = os.path.join(base_dir, "openlandmap", f"{name}.tif")
            static_files[name] = path

    static_stack, static_names = load_static_features(static_files, reference_raster)

    # Create feature names - MUST match original behavior (12 features only)
    assert len(feature_arrays) == 12
    feature_labels = [name if name else "chirps" for name in aligned_datasets]

    # Build coordinates - exactly as original
    coords = {
        "time": valid_months,
        "feature": feature_labels,  # Use feature_labels, not feature_names
        "lat": reference_raster.y.values,
        "lon": reference_raster.x.values,
    }

    data_vars = {
        "features": (["time", "feature", "lat", "lon"], temporal_stack)
    }

    if static_stack is not None:
        data_vars["static_features"] = (["static_feature", "lat", "lon"], static_stack)
        coords["static_feature"] = static_names

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    
    # Add metadata about resampling methods used
    if SCIENTIFIC_RESAMPLING:
        # Convert dictionary to string for netCDF compatibility
        resampling_methods = {
            'categorical_data': 'nearest neighbor',
            'population_data': 'sum preserving',
            'elevation_data': 'cubic interpolation',
            'precipitation_data': 'area-weighted average',
            'continuous_data': 'bilinear interpolation',
            'categorical_encoding': 'enabled' if ENABLE_CATEGORICAL_ENCODING else 'disabled'
        }
        # Store as formatted string that can be parsed later if needed
        ds.attrs['resampling_info'] = '; '.join([f'{k}: {v}' for k, v in resampling_methods.items()])
    
    # Add any additional metadata
    ds.attrs['creation_date'] = str(np.datetime64('now'))
    ds.attrs['grace_months'] = len(valid_months)

    # Save to exact same location as original
    os.makedirs("data/processed", exist_ok=True)
    ds.to_netcdf("data/processed/feature_stack.nc")
    print("✅ Saved feature_stack.nc")


if __name__ == "__main__":
    main()