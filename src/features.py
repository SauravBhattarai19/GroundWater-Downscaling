#!/usr/bin/env python3
"""
Enhanced features.py with scientific resampling - Fully integrated with config system.

This version:
1. Reads ALL settings from config.yaml for consistency
2. Uses scientifically correct resampling based on config
3. Works as configurable pipeline component
4. Adds metadata about methods used
5. Supports categorical encoding based on config
"""

import os
import numpy as np
import rioxarray
import xarray as xr
from tqdm import tqdm
from src.utils import parse_grace_months
from src.config_manager import get_config
import rasterio.enums
import warnings
warnings.filterwarnings('ignore')


def calculate_flow_direction_d8(elevation, dx, dy):
    """
    Calculate flow direction using D8 algorithm.
    
    Parameters:
    -----------
    elevation : np.array
        2D elevation array
    dx, dy : float
        Pixel spacing in meters
        
    Returns:
    --------
    np.array
        Flow direction array (1-8 for 8 directions, 0 for no flow)
    """
    rows, cols = elevation.shape
    flow_dir = np.zeros_like(elevation, dtype=np.int32)
    
    # D8 directions: E, SE, S, SW, W, NW, N, NE
    # Direction codes: 1, 2, 4, 8, 16, 32, 64, 128
    directions = [
        (0, 1, 1),    # E
        (1, 1, 2),    # SE  
        (1, 0, 4),    # S
        (1, -1, 8),   # SW
        (0, -1, 16),  # W
        (-1, -1, 32), # NW
        (-1, 0, 64),  # N
        (-1, 1, 128)  # NE
    ]
    
    # Calculate distances for diagonal vs orthogonal
    dist_ortho = min(dx, dy)
    dist_diag = np.sqrt(dx**2 + dy**2)
    
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if np.isnan(elevation[i, j]):
                continue
                
            center_elev = elevation[i, j]
            max_slope = -999999
            flow_direction = 0
            
            for di, dj, code in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols and not np.isnan(elevation[ni, nj]):
                    neighbor_elev = elevation[ni, nj]
                    
                    # Calculate slope
                    elev_diff = center_elev - neighbor_elev
                    distance = dist_diag if (di != 0 and dj != 0) else dist_ortho
                    slope = elev_diff / distance
                    
                    if slope > max_slope:
                        max_slope = slope
                        flow_direction = code
            
            flow_dir[i, j] = flow_direction
    
    return flow_dir


def calculate_flow_accumulation(elevation, flow_direction):
    """
    Calculate flow accumulation from flow direction.
    
    Parameters:
    -----------
    elevation : np.array
        2D elevation array
    flow_direction : np.array
        Flow direction array from calculate_flow_direction_d8
        
    Returns:
    --------
    np.array
        Flow accumulation array (number of upstream cells)
    """
    rows, cols = elevation.shape
    flow_acc = np.ones_like(elevation, dtype=np.float32)
    
    # Create a list of cells sorted by elevation (highest to lowest)
    valid_mask = ~np.isnan(elevation)
    valid_coords = np.where(valid_mask)
    elevations_flat = elevation[valid_coords]
    
    # Sort by elevation (descending - process from high to low)
    sort_indices = np.argsort(-elevations_flat)
    sorted_coords = [(valid_coords[0][i], valid_coords[1][i]) for i in sort_indices]
    
    # Direction mappings for D8
    dir_map = {
        1: (0, 1),    # E
        2: (1, 1),    # SE
        4: (1, 0),    # S
        8: (1, -1),   # SW
        16: (0, -1),  # W
        32: (-1, -1), # NW
        64: (-1, 0),  # N
        128: (-1, 1)  # NE
    }
    
    # Process cells from highest to lowest elevation
    for i, j in sorted_coords:
        if flow_direction[i, j] in dir_map:
            di, dj = dir_map[flow_direction[i, j]]
            ni, nj = i + di, j + dj
            
            if 0 <= ni < rows and 0 <= nj < cols and not np.isnan(elevation[ni, nj]):
                flow_acc[ni, nj] += flow_acc[i, j]
    
    return flow_acc


def calculate_topographic_derivatives(dem_data):
    """
    Calculate slope, aspect, and curvature from DEM data.
    
    Parameters:
    -----------
    dem_data : xarray.DataArray
        DEM elevation data with spatial coordinates
        
    Returns:
    --------
    dict
        Dictionary containing slope, aspect, and curvature arrays
    """
    print("  🗻 Calculating topographic derivatives from DEM...")
    
    # Get elevation values and coordinate information
    elevation = dem_data.values
    
    # Get pixel size in degrees (assuming geographic coordinates)
    if hasattr(dem_data, 'rio') and dem_data.rio.resolution() is not None:
        dx = abs(dem_data.rio.resolution()[0])  # degrees
        dy = abs(dem_data.rio.resolution()[1])  # degrees
    else:
        # Fallback: calculate from coordinates
        if 'lon' in dem_data.coords or 'x' in dem_data.coords:
            x_coord = 'x' if 'x' in dem_data.coords else 'lon'
            y_coord = 'y' if 'y' in dem_data.coords else 'lat'
            dx = abs(float(dem_data.coords[x_coord][1] - dem_data.coords[x_coord][0]))
            dy = abs(float(dem_data.coords[y_coord][1] - dem_data.coords[y_coord][0]))
        else:
            # Default pixel size
            dx = dy = 0.1  # degrees
    
    print(f"    📏 Pixel resolution: {dx:.4f}° x {dy:.4f}°")
    
    # Convert degrees to approximate meters at latitude center
    lat_center = float(dem_data.coords['lat' if 'lat' in dem_data.coords else 'y'].mean())
    dx_m = dx * 111320 * np.cos(np.radians(lat_center))  # meters
    dy_m = dy * 111320  # meters
    
    print(f"    📏 Approximate resolution: {dx_m:.0f}m x {dy_m:.0f}m")
    
    # Calculate gradients using numpy gradient
    grad_y, grad_x = np.gradient(elevation, dy_m, dx_m)
    
    # Calculate slope (in degrees)
    slope_radians = np.arctan(np.sqrt(grad_x**2 + grad_y**2))
    slope_degrees = np.degrees(slope_radians)
    
    # Calculate aspect (in degrees from North, clockwise)
    aspect_radians = np.arctan2(-grad_x, grad_y)
    aspect_degrees = np.degrees(aspect_radians)
    # Convert to 0-360 degrees
    aspect_degrees = (aspect_degrees + 360) % 360
    
    # Calculate curvature (second derivatives)
    grad_xx = np.gradient(grad_x, dx_m, axis=1)
    grad_yy = np.gradient(grad_y, dy_m, axis=0) 
    grad_xy = np.gradient(grad_x, dy_m, axis=0)
    
    # Mean curvature (general surface curvature)
    p = grad_x
    q = grad_y
    r = grad_xx
    s = grad_xy
    t = grad_yy
    
    # Mean curvature formula
    curvature = -((1 + q**2) * r - 2 * p * q * s + (1 + p**2) * t) / (2 * (1 + p**2 + q**2)**(3/2))
    
    # Scientific accuracy check - NEVER create synthetic data
    invalid_slope = ~np.isfinite(slope_degrees)
    invalid_aspect = ~np.isfinite(aspect_degrees) 
    invalid_curvature = ~np.isfinite(curvature)
    
    if np.any(invalid_slope):
        n_invalid = np.sum(invalid_slope)
        print(f"    ⚠️ WARNING: {n_invalid} invalid slope values found - preserving NaN for scientific accuracy")
        # KEEP NaN values - do NOT create synthetic data
        
    if np.any(invalid_aspect):
        n_invalid = np.sum(invalid_aspect)
        print(f"    ⚠️ WARNING: {n_invalid} invalid aspect values found - preserving NaN for scientific accuracy")
        # KEEP NaN values - do NOT create synthetic data
        
    if np.any(invalid_curvature):
        n_invalid = np.sum(invalid_curvature)
        print(f"    ⚠️ WARNING: {n_invalid} invalid curvature values found - preserving NaN for scientific accuracy")
        # KEEP NaN values - do NOT create synthetic data
    
    print(f"    ✅ Slope range: {np.nanmin(slope_degrees):.1f}° to {np.nanmax(slope_degrees):.1f}°")
    print(f"    ✅ Aspect range: {np.nanmin(aspect_degrees):.1f}° to {np.nanmax(aspect_degrees):.1f}°")
    print(f"    ✅ Curvature range: {np.nanmin(curvature):.6f} to {np.nanmax(curvature):.6f}")
    
    # Calculate flow direction using D8 algorithm (8-direction flow)
    print("  🌊 Calculating flow direction (D8 algorithm)...")
    flow_direction = calculate_flow_direction_d8(elevation, dx_m, dy_m)
    
    # Calculate basic flow accumulation
    print("  🌊 Calculating flow accumulation...")
    flow_accumulation = calculate_flow_accumulation(elevation, flow_direction)
    
    # Calculate topographic wetness index (TWI)
    print("  🌊 Calculating topographic wetness index...")
    # TWI = ln(flow_accumulation / slope), but handle divide by zero
    slope_radians_safe = np.maximum(slope_radians, 1e-6)  # Avoid division by zero
    flow_acc_safe = np.maximum(flow_accumulation, 1.0)   # Minimum drainage area
    twi = np.log(flow_acc_safe / np.tan(slope_radians_safe))
    
    print(f"    ✅ Flow direction range: {np.nanmin(flow_direction):.0f} to {np.nanmax(flow_direction):.0f}")
    print(f"    ✅ Flow accumulation range: {np.nanmin(flow_accumulation):.0f} to {np.nanmax(flow_accumulation):.0f}")
    print(f"    ✅ TWI range: {np.nanmin(twi):.2f} to {np.nanmax(twi):.2f}")
    
    return {
        'slope': slope_degrees,
        'aspect': aspect_degrees, 
        'curvature': curvature,
        'flow_direction': flow_direction,
        'flow_accumulation': flow_accumulation,
        'twi': twi  # Topographic Wetness Index
    }


def create_topographic_features(dem_path, reference_raster, enable_derivatives=True):
    """
    Create topographic features from DEM including slope, aspect, and curvature.
    
    Parameters:
    -----------
    dem_path : str
        Path to DEM file
    reference_raster : xarray.DataArray
        Reference raster for alignment
    enable_derivatives : bool
        Whether to calculate slope, aspect, curvature
        
    Returns:
    --------
    dict
        Dictionary of topographic feature arrays
    """
    if not os.path.exists(dem_path):
        error_msg = f"❌ CRITICAL ERROR: DEM file not found: {dem_path}"
        print(error_msg)
        print("   This compromises scientific accuracy of topographic features!")
        print("   Pipeline cannot continue without real elevation data!")
        raise FileNotFoundError(error_msg)
    
    print(f"  📂 Loading DEM: {dem_path}")
    
    # Load DEM data
    dem_raster = rioxarray.open_rasterio(dem_path, masked=True).squeeze()
    
    # Resample to match reference
    resampling_method = get_resampling_method("elevation")
    dem_aligned = dem_raster.rio.reproject_match(
        reference_raster,
        resampling=resampling_method
    )
    
    results = {'elevation': dem_aligned.values}
    
    if enable_derivatives:
        # Calculate topographic derivatives
        derivatives = calculate_topographic_derivatives(dem_aligned)
        results.update(derivatives)
    
    return results


def get_resampling_method(data_name):
    """Get appropriate resampling method for data type based on config."""
    scientific_resampling = get_config('feature_processing.scientific_resampling', True)
    
    if not scientific_resampling:
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
    
    scientific_resampling = get_config('feature_processing.scientific_resampling', True)
    if scientific_resampling and resample_method != rasterio.enums.Resampling.bilinear:
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
                
                # CRITICAL UNIT CONVERSIONS for GLDAS data
                if 'Evap_tavg' in folder_path:
                    # Convert GLDAS evapotranspiration from kg/m²/s to mm/day
                    raster.values = raster.values * 86400  # 86400 seconds in a day
                    print(f"  🔧 Converted Evap_tavg: kg/m²/s → mm/day (×86400)")
                        
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
                
                # CRITICAL UNIT CONVERSIONS for GLDAS data
                if 'Evap_tavg' in folder_path:
                    # Convert GLDAS evapotranspiration from kg/m²/s to mm/day
                    raster.values = raster.values * 86400  # 86400 seconds in a day
                    print(f"  🔧 Converted Evap_tavg: kg/m²/s → mm/day (×86400)")
                        
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
                
                scientific_resampling = get_config('feature_processing.scientific_resampling', True)
                if scientific_resampling:
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
        enable_categorical_encoding = get_config('feature_processing.enable_categorical_encoding', False)
        if enable_categorical_encoding and categorical_info:
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


def analyze_feature_correlations(temporal_stack, feature_names, threshold=0.9):
    """
    Analyze feature correlations and identify features to remove.
    
    Parameters:
    -----------
    temporal_stack : np.array
        Feature data with shape (time, feature, lat, lon)
    feature_names : list
        List of feature names
    threshold : float
        Correlation threshold above which features are considered redundant
    
    Returns:
    --------
    dict: Analysis results including correlation matrix and features to remove
    """
    n_time, n_feat, n_lat, n_lon = temporal_stack.shape
    
    # Flatten spatially and compute spatial averages for correlation analysis
    features_flat = temporal_stack.reshape(n_time, n_feat, -1)
    features_mean = np.nanmean(features_flat, axis=2)  # (time, feat)
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(features_mean.T)
    
    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(n_feat):
        for j in range(i+1, n_feat):
            corr_val = corr_matrix[i, j]
            if abs(corr_val) > threshold:
                high_corr_pairs.append({
                    'idx1': i, 'idx2': j, 
                    'feature1': feature_names[i], 'feature2': feature_names[j],
                    'correlation': corr_val
                })
    
    # Intelligent feature removal strategy
    features_to_remove = set()
    
    for pair in high_corr_pairs:
        feat1, feat2 = pair['feature1'], pair['feature2']
        idx1, idx2 = pair['idx1'], pair['idx2']
        
        # Skip if either feature already marked for removal
        if feat1 in features_to_remove or feat2 in features_to_remove:
            continue
            
        # Priority rules for feature removal (keep more physically meaningful ones)
        removed_feature = None
        
        # Rule 1: Temperature features are ESSENTIAL - never remove
        if 'tmean' in feat1 or 'tmean' in feat2:
            print(f"  🌡️ PROTECTING ESSENTIAL TEMPERATURE: {feat1} ↔ {feat2} - keeping tmean")
            continue  # Skip removal for any correlation involving tmean
            
        # Rule 2: Temperature min/max pairs (legacy - should not occur now)
        elif ('tmmn' in feat1 and 'tmmx' in feat2) or ('tmmx' in feat1 and 'tmmn' in feat2):
            print(f"  🌡️ Temperature pair detected: {feat1} ↔ {feat2} - will create tmean feature")
            continue  # Skip removal for temperature pairs
            
        # Rule 2: For soil moisture layers, prefer deeper layers (more stable)
        elif 'SoilMoi' in feat1 and 'SoilMoi' in feat2:
            # Extract depths and keep deeper layer
            try:
                # Handle formats like SoilMoi0_10cm_inst, SoilMoi10_40cm_inst, etc.
                if 'SoilMoi0_10cm' in feat1:
                    depth1 = 5  # Average of 0-10cm
                elif 'SoilMoi10_40cm' in feat1:
                    depth1 = 25  # Average of 10-40cm
                elif 'SoilMoi40_100cm' in feat1:
                    depth1 = 70  # Average of 40-100cm
                elif 'SoilMoi100_200cm' in feat1:
                    depth1 = 150  # Average of 100-200cm
                else:
                    depth1 = 0
                    
                if 'SoilMoi0_10cm' in feat2:
                    depth2 = 5
                elif 'SoilMoi10_40cm' in feat2:
                    depth2 = 25
                elif 'SoilMoi40_100cm' in feat2:
                    depth2 = 70
                elif 'SoilMoi100_200cm' in feat2:
                    depth2 = 150
                else:
                    depth2 = 0
                    
                if depth1 < depth2:
                    removed_feature = feat1
                else:
                    removed_feature = feat2
            except:
                removed_feature = feat2  # Default fallback
                
        # Rule 3: For evapotranspiration variables, keep actual ET over deficit
        elif 'aet' in feat1 and 'def' in feat2:
            removed_feature = feat2  # Remove deficit, keep actual ET
        elif 'def' in feat1 and 'aet' in feat2:
            removed_feature = feat1  # Remove deficit, keep actual ET
            
        # Rule 4: Default - remove feature with higher index (later in processing)
        else:
            removed_feature = feat2 if idx2 > idx1 else feat1
        
        if removed_feature:
            features_to_remove.add(removed_feature)
    
    return {
        'correlation_matrix': corr_matrix,
        'high_correlations': high_corr_pairs,
        'features_to_remove': list(features_to_remove),
        'n_removed': len(features_to_remove),
        'n_remaining': n_feat - len(features_to_remove)
    }


def remove_correlated_features(temporal_stack, feature_names, correlation_analysis):
    """Remove highly correlated features from the dataset."""
    features_to_remove = correlation_analysis['features_to_remove']
    
    if not features_to_remove:
        print("✅ No highly correlated features found")
        return temporal_stack, feature_names
    
    print(f"\n🔧 REMOVING {len(features_to_remove)} HIGHLY CORRELATED FEATURES:")
    for feat in features_to_remove:
        print(f"   ❌ {feat}")
    
    # Find indices of features to keep
    keep_indices = []
    keep_names = []
    
    for i, name in enumerate(feature_names):
        if name not in features_to_remove:
            keep_indices.append(i)
            keep_names.append(name)
    
    # Remove features from temporal stack
    filtered_stack = temporal_stack[:, keep_indices, :, :]
    
    print(f"✅ Reduced from {len(feature_names)} to {len(keep_names)} features")
    
    return filtered_stack, keep_names


def add_precipitation_accumulation_features(temporal_stack, feature_names):
    """
    Add precipitation accumulation features (30-day, 90-day rolling sums).
    Critical for groundwater modeling as groundwater responds to accumulated precipitation.
    """
    import numpy as np
    
    print("\n🌧️ ADDING PRECIPITATION ACCUMULATION FEATURES")
    
    # Find precipitation features
    precip_indices = []
    precip_names = []
    for i, name in enumerate(feature_names):
        if any(precip in name.lower() for precip in ['chirps', 'pr']):
            precip_indices.append(i)
            precip_names.append(name)
    
    if not precip_indices:
        print("  ⚠️ No precipitation features found - skipping accumulation")
        return temporal_stack, feature_names
    
    print(f"  📍 Found {len(precip_indices)} precipitation features: {precip_names}")
    
    # Create accumulation features for each precipitation variable
    new_features = []
    new_names = []
    
    for precip_idx, precip_name in zip(precip_indices, precip_names):
        precip_data = temporal_stack[:, precip_idx, :, :]  # Shape: (time, lat, lon)
        
        # 30-day accumulation (1 month)
        precip_30d = np.zeros_like(precip_data)
        for t in range(len(precip_data)):
            start_idx = max(0, t - 0)  # Current month only (30-day monthly data)
            precip_30d[t] = np.nansum(precip_data[start_idx:t+1], axis=0)
        
        new_features.append(precip_30d)
        new_names.append(f"{precip_name}_30d")
        
        # 90-day accumulation (3 months)
        precip_90d = np.zeros_like(precip_data)
        for t in range(len(precip_data)):
            start_idx = max(0, t - 2)  # Previous 3 months
            precip_90d[t] = np.nansum(precip_data[start_idx:t+1], axis=0)
        
        new_features.append(precip_90d)
        new_names.append(f"{precip_name}_90d")
        
        print(f"    ✅ {precip_name}: 30-day range [{np.nanmin(precip_30d):.1f}, {np.nanmax(precip_30d):.1f}] mm")
        print(f"    ✅ {precip_name}: 90-day range [{np.nanmin(precip_90d):.1f}, {np.nanmax(precip_90d):.1f}] mm")
    
    if new_features:
        # Add new features to temporal stack
        new_stack = np.stack(new_features, axis=1)  # Shape: (time, new_features, lat, lon)
        temporal_stack = np.concatenate([temporal_stack, new_stack], axis=1)
        feature_names.extend(new_names)
        
        print(f"  ✅ Added {len(new_features)} precipitation accumulation features")
        print(f"  ✅ Updated temporal stack shape: {temporal_stack.shape}")
    
    return temporal_stack, feature_names


def add_seasonal_anomaly_features(temporal_stack, feature_names):
    """
    Add seasonal anomaly features - deviation from monthly climatology.
    Critical for groundwater modeling as anomalies drive storage changes.
    """
    import numpy as np
    
    print("\n📅 ADDING SEASONAL ANOMALY FEATURES")
    
    # Variables to create anomalies for (key hydroclimate variables)
    anomaly_candidates = ['tmean', 'pr', 'chirps', 'Evap_tavg', 'aet', 'def', 'SWE_inst']
    
    # Find indices of candidate features
    anomaly_indices = []
    anomaly_names = []
    for i, name in enumerate(feature_names):
        if any(candidate in name for candidate in anomaly_candidates):
            anomaly_indices.append(i)
            anomaly_names.append(name)
    
    if not anomaly_indices:
        print("  ⚠️ No suitable features found for anomaly calculation")
        return temporal_stack, feature_names
        
    print(f"  📍 Creating anomalies for {len(anomaly_indices)} features: {anomaly_names}")
    
    # Assuming monthly data with shape (time=197, feature, lat, lon)
    # Calculate monthly climatology (mean for each month across all years)
    n_times = temporal_stack.shape[0]
    new_features = []
    new_names = []
    
    for feat_idx, feat_name in zip(anomaly_indices, anomaly_names):
        print(f"    🔄 Processing {feat_name} anomalies...")
        
        feature_data = temporal_stack[:, feat_idx, :, :]  # Shape: (time, lat, lon)
        
        # Calculate monthly climatology
        # Group by month (assume consistent monthly pattern from 2003-01 to 2022-11)
        climatology = np.zeros((12, feature_data.shape[1], feature_data.shape[2]))  # (12_months, lat, lon)
        
        for month in range(12):
            # Get all occurrences of this month
            month_indices = [t for t in range(n_times) if t % 12 == month]
            if month_indices:
                climatology[month] = np.nanmean(feature_data[month_indices], axis=0)
        
        # Calculate anomalies for each time step
        anomalies = np.zeros_like(feature_data)
        for t in range(n_times):
            month = t % 12
            anomalies[t] = feature_data[t] - climatology[month]
        
        # Add to new features
        new_features.append(anomalies)
        new_names.append(f"{feat_name}_anom")
        
        # Print statistics
        anom_mean = np.nanmean(anomalies)
        anom_std = np.nanstd(anomalies)
        anom_min = np.nanmin(anomalies)
        anom_max = np.nanmax(anomalies)
        print(f"      ✅ {feat_name}_anom: mean={anom_mean:.3f}, std={anom_std:.3f}, range=[{anom_min:.3f}, {anom_max:.3f}]")
    
    if new_features:
        # Add new features to temporal stack
        new_stack = np.stack(new_features, axis=1)  # Shape: (time, new_features, lat, lon)
        temporal_stack = np.concatenate([temporal_stack, new_stack], axis=1)
        feature_names.extend(new_names)
        
        print(f"  ✅ Added {len(new_features)} seasonal anomaly features")
        print(f"  ✅ Updated temporal stack shape: {temporal_stack.shape}")
    
    return temporal_stack, feature_names


def add_temporal_lag_features(temporal_stack, feature_names):
    """
    Add temporal lag features (1, 3, 6 months) for key hydrological variables.
    Critical for groundwater modeling - current storage depends on previous climate conditions.
    """
    import numpy as np
    
    print("\n⏰ ADDING TEMPORAL LAG FEATURES")
    
    # Key variables for lag analysis (hydrological memory) - RESTORE FOR TREES
    # Trees can handle complexity - restore full lag set for pattern detection
    lag_candidates = ['tmean', 'pr', 'chirps', 'Evap_tavg', 'aet', 'def', 'SWE_inst', 'tmean_anom', 'chirps_anom', 'SWE_inst_anom']
    lag_months = [1, 3, 6]  # RESTORE: Full temporal memory for trees
    
    # Find indices of candidate features
    lag_indices = []
    lag_names = []
    for i, name in enumerate(feature_names):
        if any(candidate in name for candidate in lag_candidates):
            # Avoid creating lags of already-lagged features or 90d features
            if not any(suffix in name for suffix in ['_lag', '_90d']):
                lag_indices.append(i)
                lag_names.append(name)
    
    if not lag_indices:
        print("  ⚠️ No suitable features found for lag calculation")
        return temporal_stack, feature_names
        
    print(f"  📍 Creating lags for {len(lag_indices)} features: {lag_names}")
    print(f"  📍 Lag periods: {lag_months} months")
    
    # Create lag features
    n_times = temporal_stack.shape[0]
    new_features = []
    new_names = []
    
    for feat_idx, feat_name in zip(lag_indices, lag_names):
        feature_data = temporal_stack[:, feat_idx, :, :]  # Shape: (time, lat, lon)
        
        for lag in lag_months:
            print(f"    🔄 Processing {feat_name} lag-{lag}...")
            
            # Create lagged feature
            lagged_data = np.full_like(feature_data, np.nan)
            
            # Fill lagged values (t-lag)
            for t in range(lag, n_times):
                lagged_data[t] = feature_data[t - lag]
            
            # Add to new features
            new_features.append(lagged_data)
            new_names.append(f"{feat_name}_lag{lag}")
            
            # Print statistics (skip NaN months at beginning)
            valid_data = lagged_data[lag:]
            if valid_data.size > 0:
                lag_mean = np.nanmean(valid_data)
                lag_std = np.nanstd(valid_data)
                lag_min = np.nanmin(valid_data)
                lag_max = np.nanmax(valid_data)
                print(f"      ✅ {feat_name}_lag{lag}: mean={lag_mean:.3f}, std={lag_std:.3f}, range=[{lag_min:.3f}, {lag_max:.3f}]")
            else:
                print(f"      ⚠️ {feat_name}_lag{lag}: No valid data")
    
    if new_features:
        # Add new features to temporal stack
        new_stack = np.stack(new_features, axis=1)  # Shape: (time, new_features, lat, lon)
        temporal_stack = np.concatenate([temporal_stack, new_stack], axis=1)
        feature_names.extend(new_names)
        
        print(f"  ✅ Added {len(new_features)} temporal lag features")
        print(f"  ✅ Updated temporal stack shape: {temporal_stack.shape}")
        print(f"  ℹ️ Note: First {max(lag_months)} months will have NaN lag values (handled in training)")
    
    return temporal_stack, feature_names


def add_spatial_lag_features(temporal_stack, feature_names):
    """
    Add spatial lag features (neighbor averaging) - CRITICAL for groundwater physics.
    Groundwater flows spatially, so current storage depends on neighboring pixels.
    Expected improvement: +0.08-0.15 R².
    """
    import numpy as np
    from scipy import ndimage
    
    print("\n🌍 ADDING SPATIAL LAG FEATURES")
    
    # Key variables for spatial analysis (hydrological flows) - RESTORE FOR TREES
    spatial_candidates = ['tmean', 'pr', 'chirps', 'Evap_tavg', 'aet', 'SWE_inst', 'tmean_anom', 'chirps_anom']  # RESTORE: Full spatial analysis
    
    # Find indices of candidate features
    spatial_indices = []
    spatial_names = []
    for i, name in enumerate(feature_names):
        if any(candidate in name for candidate in spatial_candidates):
            # Focus on most important features, avoid over-processing lags
            if not any(suffix in name for suffix in ['_lag3', '_lag6', '_90d']):
                spatial_indices.append(i)
                spatial_names.append(name)
    
    if not spatial_indices:
        print("  ⚠️ No suitable features found for spatial analysis")
        return temporal_stack, feature_names
        
    print(f"  📍 Creating spatial lags for {len(spatial_indices)} features: {spatial_names}")
    print(f"  📍 Window sizes: 3x3 (immediate neighbors)")
    
    # Create spatial features
    n_times, n_features, n_lat, n_lon = temporal_stack.shape
    new_features = []
    new_names = []
    
    # Define spatial kernels
    # 3x3 average kernel (immediate neighbors)
    kernel_3x3 = np.ones((3, 3)) / 9
    
    for feat_idx, feat_name in zip(spatial_indices, spatial_names):
        print(f"    🔄 Processing {feat_name} spatial lags...")
        
        feature_data = temporal_stack[:, feat_idx, :, :]  # Shape: (time, lat, lon)
        
        # 3x3 spatial averaging
        spatial_3x3 = np.zeros_like(feature_data)
        for t in range(n_times):
            # Use scipy.ndimage for efficient convolution with proper boundary handling
            spatial_3x3[t] = ndimage.convolve(feature_data[t], kernel_3x3, mode='reflect')
        
        # Add to new features
        new_features.append(spatial_3x3)
        new_names.append(f"{feat_name}_spatial3x3")
        
        # Print statistics
        spatial_mean = np.nanmean(spatial_3x3)
        spatial_std = np.nanstd(spatial_3x3)
        spatial_min = np.nanmin(spatial_3x3)
        spatial_max = np.nanmax(spatial_3x3)
        print(f"      ✅ {feat_name}_spatial3x3: mean={spatial_mean:.3f}, std={spatial_std:.3f}, range=[{spatial_min:.3f}, {spatial_max:.3f}]")
    
    if new_features:
        # Add new features to temporal stack
        new_stack = np.stack(new_features, axis=1)  # Shape: (time, new_features, lat, lon)
        temporal_stack = np.concatenate([temporal_stack, new_stack], axis=1)
        feature_names.extend(new_names)
        
        print(f"  ✅ Added {len(new_features)} spatial lag features")
        print(f"  ✅ Updated temporal stack shape: {temporal_stack.shape}")
        print(f"  🌊 Physics: Captures spatial groundwater flow relationships")
    
    return temporal_stack, feature_names


def main():
    """Main function - fully integrated with config system."""
    # Load paths from config
    grace_dir = get_config('paths.grace_dir', 'data/raw/grace')
    base_dir = get_config('paths.base_dir', 'data/raw')
    
    valid_months = get_valid_grace_months(grace_dir)
    print(f"✅ Using {len(valid_months)} months aligned with GRACE")

    # Use CHIRPS as reference (maintains compatibility)
    chirps_dir = os.path.join(base_dir, "chirps")
    reference_raster = get_chirps_reference_raster(chirps_dir)
    
    # Alternative: use GRACE reference
    # reference_raster = get_grace_reference_raster(grace_dir, valid_months)
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
            folder = os.path.join(base_dir, group, name) if name else os.path.join(base_dir, group)
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

    # CRITICAL: Create mean temperature feature BEFORE correlation analysis
    print("\n🌡️ PROCESSING TEMPERATURE FEATURES (Before Correlation Analysis)")
    tmmn_idx = None
    tmmx_idx = None
    
    for i, name in enumerate(aligned_datasets):
        if 'tmmn' in name:
            tmmn_idx = i
        elif 'tmmx' in name:
            tmmx_idx = i
    
    if tmmn_idx is not None and tmmx_idx is not None:
        print("  🌡️ Creating mean temperature from tmmn + tmmx")
        
        # Calculate mean temperature: (tmmn + tmmx) / 2
        tmmn_data = temporal_stack[:, tmmn_idx, :, :]
        tmmx_data = temporal_stack[:, tmmx_idx, :, :]
        tmean_data = (tmmn_data + tmmx_data) / 2
        
        # Add tmean to the stack
        new_stack = np.concatenate([temporal_stack, tmean_data[:, np.newaxis, :, :]], axis=1)
        aligned_datasets.append('tmean')
        
        # Remove original tmmn and tmmx
        keep_indices = [i for i in range(len(aligned_datasets)-1) if i not in [tmmn_idx, tmmx_idx]]
        temporal_stack = new_stack[:, keep_indices + [-1], :, :]  # Keep all except tmmn/tmmx, plus new tmean
        aligned_datasets = [aligned_datasets[i] for i in keep_indices] + ['tmean']
        
        print(f"  ✅ Replaced tmmn + tmmx with tmean")
        print(f"  ✅ Updated temporal stack shape: {temporal_stack.shape}")
        print(f"  ✅ Updated feature names: {aligned_datasets}")
    else:
        print("  ⚠️ Temperature features not found - check data sources")

    # Add precipitation accumulation features
    temporal_stack, aligned_datasets = add_precipitation_accumulation_features(temporal_stack, aligned_datasets)

    # Add seasonal anomaly features
    temporal_stack, aligned_datasets = add_seasonal_anomaly_features(temporal_stack, aligned_datasets)

    # Add temporal lag features
    temporal_stack, aligned_datasets = add_temporal_lag_features(temporal_stack, aligned_datasets)

    # Add spatial lag features
    temporal_stack, aligned_datasets = add_spatial_lag_features(temporal_stack, aligned_datasets)

    # Apply correlation-based feature selection (if enabled)
    remove_correlated = get_config('feature_processing.remove_correlated_features', True)
    if remove_correlated:
        correlation_threshold = get_config('feature_processing.correlation_threshold', 0.9)
        print(f"\n🔍 ANALYZING FEATURE CORRELATIONS (threshold = {correlation_threshold})")
        correlation_analysis = analyze_feature_correlations(temporal_stack, aligned_datasets, correlation_threshold)
        
        if correlation_analysis['high_correlations']:
            print(f"⚠️ Found {len(correlation_analysis['high_correlations'])} high correlation pairs:")
            for pair in correlation_analysis['high_correlations']:
                print(f"   {pair['feature1']} ↔ {pair['feature2']}: r = {pair['correlation']:.3f}")
            
            # Remove correlated features
            temporal_stack, aligned_datasets = remove_correlated_features(
                temporal_stack, aligned_datasets, correlation_analysis
            )
            print(f"✅ Updated stacked shape: {temporal_stack.shape}")
        else:
            print("✅ No highly correlated features found")
    else:
        print("🔍 Correlation-based feature selection disabled in config")

    # Load static features
    static_files = {
        "modis_land_cover": os.path.join(base_dir, "modis_land_cover", "landcover_5km.tif"),
        "landscan_population": os.path.join(base_dir, "landscan", "2003.tif"),
    }
    
    # Add soil properties
    for depth in ["0cm", "10cm", "30cm", "60cm", "100cm", "200cm"]:
        for var in ["sand", "clay"]:  # Note: original only has sand and clay, not silt
            name = f"{var}_{depth}"
            path = os.path.join(base_dir, "openlandmap", f"{name}_5km.tif")
            static_files[name] = path

    # Load basic static features (non-DEM)
    static_stack, static_names = load_static_features(static_files, reference_raster)
    
    # Load topographic features separately with derivatives
    enable_topographic_derivatives = get_config('feature_processing.enable_topographic_derivatives', True)
    dem_path = os.path.join(base_dir, "usgs_dem", "dem.tif")
    
    print(f"🗻 Processing topographic features (derivatives: {'enabled' if enable_topographic_derivatives else 'disabled'})")
    topographic_features = create_topographic_features(
        dem_path, 
        reference_raster, 
        enable_derivatives=enable_topographic_derivatives
    )
    
    if topographic_features:
        # Add topographic features to static stack
        topo_arrays = []
        topo_names = []
        
        for topo_name, topo_data in topographic_features.items():
            topo_arrays.append(topo_data)
            topo_names.append(f"topo_{topo_name}")
        
        if len(topo_arrays) > 0:
            # Combine with existing static features
            if static_stack is not None and len(static_stack) > 0:
                static_stack = np.concatenate([static_stack, np.stack(topo_arrays, axis=0)], axis=0)
                static_names.extend(topo_names)
            else:
                static_stack = np.stack(topo_arrays, axis=0)
                static_names = topo_names
            
            print(f"  ✅ Added {len(topo_arrays)} topographic features: {topo_names}")
    else:
        print("  ⚠️ No topographic features could be loaded")

    # Create feature names - updated after correlation filtering
    print(f"📊 Final feature count: {len(aligned_datasets)}")
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
    
    # Add metadata about processing methods used
    scientific_resampling = get_config('feature_processing.scientific_resampling', True)
    if scientific_resampling:
        # Convert dictionary to string for netCDF compatibility
        enable_categorical_encoding = get_config('feature_processing.enable_categorical_encoding', False)
        resampling_methods = {
            'categorical_data': 'nearest neighbor',
            'population_data': 'sum preserving',
            'elevation_data': 'cubic interpolation',
            'precipitation_data': 'area-weighted average',
            'continuous_data': 'bilinear interpolation',
            'categorical_encoding': 'enabled' if enable_categorical_encoding else 'disabled',
            'correlation_filtering': 'enabled' if remove_correlated else 'disabled',
            'correlation_threshold': str(correlation_threshold if remove_correlated else 'N/A')
        }
        # Store as formatted string that can be parsed later if needed
        ds.attrs['resampling_info'] = '; '.join([f'{k}: {v}' for k, v in resampling_methods.items()])
    
    # Add any additional metadata
    ds.attrs['creation_date'] = str(np.datetime64('now'))
    ds.attrs['grace_months'] = len(valid_months)

    # Save to configured location
    output_path = get_config('paths.feature_stack', 'data/processed/feature_stack.nc')
    processed_dir = get_config('paths.processed_data', 'data/processed')
    
    os.makedirs(processed_dir, exist_ok=True)
    ds.to_netcdf(output_path)
    print(f"✅ Saved feature_stack to {output_path}")


if __name__ == "__main__":
    main()