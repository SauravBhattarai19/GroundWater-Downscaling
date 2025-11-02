# src/utils.py - COMPLETE Fixed version for Mascon Data

import ee
import os
import re
import rioxarray as rxr
import xarray as xr
import numpy as np
import pandas as pd
from scipy import ndimage
import rasterio.enums

# CORRECTED Configuration - now detects automatically
GRACE_DATA_TYPE = "mascon"  # Will be auto-detected
MASCON_RESOLUTION_KM = 55.66  # JPL Mascon native resolution

def detect_grace_data_type(grace_dir="data/raw/grace"):
    """
    Automatically detect if we're using Mascon or raw GRACE data.
    """
    try:
        grace_files = [f for f in os.listdir(grace_dir) if f.endswith('.tif')]
        if not grace_files:
            return "mascon"  # Default assumption
        
        # Load a sample file to check resolution
        sample_file = os.path.join(grace_dir, grace_files[0])
        sample = rxr.open_rasterio(sample_file, masked=True).squeeze()
        
        # Check resolution - Mascon is ~0.5 degrees, raw GRACE is ~3 degrees
        resolution = abs(float(sample.rio.resolution()[0]))
        
        if resolution < 1.0:  # Less than 1 degree = likely Mascon
            print(f"  🎯 Auto-detected: Mascon data (resolution: {resolution:.3f}°)")
            return "mascon"
        else:  # Greater than 1 degree = likely raw GRACE
            print(f"  🔬 Auto-detected: Raw GRACE data (resolution: {resolution:.3f}°)")
            return "raw_grace"
            
    except Exception as e:
        print(f"  ⚠️ Could not auto-detect GRACE type: {e}, assuming Mascon")
        return "mascon"

# Auto-detect on import
GRACE_DATA_TYPE = detect_grace_data_type()

def create_date_list(start="2003-01", end="2022-12"):
    """Create list of YYYY-MM dates."""
    from datetime import datetime, timedelta
    dates = []
    current = datetime.strptime(start, "%Y-%m")
    end_date = datetime.strptime(end, "%Y-%m")
    while current <= end_date:
        dates.append(current.strftime("%Y-%m"))
        current += timedelta(days=32)
        current = current.replace(day=1)
    return dates

def bbox_to_geometry(region):
    """Convert bounding box to Earth Engine geometry."""
    return ee.Geometry.Rectangle([
        region["lon_min"],
        region["lat_min"],
        region["lon_max"],
        region["lat_max"]
    ])

def reproject_match(src, match):
    """Reproject src to match CRS of reference raster."""
    return src.rio.reproject_match(match)

def resample_match(src, match):
    """Resample src to match resolution and shape of reference raster."""
    return src.rio.reproject_match(match)

def match_resolution(src, match):
    """Alias: Match CRS, resolution, and alignment."""
    return reproject_match(src, match)

def parse_grace_months(grace_dir=None):
    """Parse valid GRACE months from various filename formats."""
    if grace_dir is None:
        grace_dir = get_config('paths.grace_dir', 'data/raw/grace')
    """Parse valid GRACE months from various filename formats."""
    months = set()
    
    # Check for new numeric format (0.tif, 1.tif, 2.tif...) - like GLDAS/CHIRPS
    numeric_pattern = re.compile(r"(\d+)\.tif$")
    # Check for yyyymm format (200301.tif, 200302.tif...)
    monthly_pattern = re.compile(r"(\d{6})\.tif$")
    # Check for old irregular format (YYYYMMDD_YYYYMMDD.tif) for backward compatibility
    range_pattern = re.compile(r"(\d{8})_(\d{8})\.tif$")
    
    if not os.path.exists(grace_dir):
        print(f"⚠️ GRACE directory not found: {grace_dir}")
        return []
    
    files = os.listdir(grace_dir)
    numeric_files = []
    monthly_files = 0
    range_files = 0
    
    for fname in files:
        # Try new numeric format first (0.tif, 1.tif, 2.tif...)
        match = numeric_pattern.match(fname)
        if match and not monthly_pattern.match(fname):  # Exclude yyyymm which also matches numeric
            numeric_files.append(int(match.group(1)))
            continue
        
        # Try yyyymm format (200301.tif, 200302.tif...)
        match = monthly_pattern.match(fname)
        if match:
            yyyymm = match.group(1)
            # Convert YYYYMM to YYYY-MM format
            year = yyyymm[:4]
            month = yyyymm[4:6]
            months.add(f"{year}-{month}")
            monthly_files += 1
            continue
        
        # Fall back to old irregular format for backward compatibility
        match = range_pattern.match(fname)
        if match:
            start = pd.to_datetime(match.group(1), format="%Y%m%d")
            end = pd.to_datetime(match.group(2), format="%Y%m%d")
            # Include months between start and end (but pick mid-point)
            mid = start + (end - start) / 2
            months.add(mid.strftime("%Y-%m"))
            range_files += 1
    
    # Handle numeric files (convert indices to YYYY-MM format)
    if numeric_files:
        # Sort numeric files and convert to months starting from config start date
        numeric_files.sort()
        start_date = get_config('data_processing.start_date', '2003-01-01')
        base_date = pd.to_datetime(start_date)
        
        for file_idx in numeric_files:
            # Each file represents one month starting from 2003-01
            month_date = base_date + pd.DateOffset(months=file_idx)
            months.add(month_date.strftime("%Y-%m"))
        
        print(f"✅ Using new numeric GRACE format: {len(numeric_files)} files (2003-01 to {(base_date + pd.DateOffset(months=max(numeric_files))).strftime('%Y-%m')})")
    elif monthly_files > 0:
        print(f"✅ Using YYYYMM GRACE format: {monthly_files} files")
    elif range_files > 0:
        print(f"⚠️ Using legacy GRACE format: {range_files} files (consider re-downloading)")
    
    return sorted(months)

def load_timestamp_map(grace_dir=None):
    """Load grace-based timestamp map for masking/filtering."""
    if grace_dir is None:
        grace_dir = get_config('paths.grace_dir', 'data/raw/grace')
    """Load grace-based timestamp map for masking/filtering."""
    valid_months = parse_grace_months(grace_dir)
    return set(valid_months)

def resample_grace_scientifically(grace_data, reference_raster, method='auto'):
    """
    CORRECTED: Resample GRACE data with appropriate method for data type.
    
    Parameters:
    -----------
    grace_data : xarray.DataArray
        GRACE data (Mascon or raw)
    reference_raster : xarray.DataArray  
        Target resolution grid
    method : str
        'auto' : Automatically detect based on GRACE_DATA_TYPE
        'mascon' : For JPL Mascon data (NO additional smoothing)
        'raw_grace' : For raw spherical harmonic GRACE (needs smoothing)
        'gaussian' : Legacy name for raw_grace
        'conservative' : Area-weighted resampling
        'bilinear' : Simple bilinear interpolation
        
    Returns:
    --------
    xarray.DataArray
        Properly resampled GRACE data
    """
    global GRACE_DATA_TYPE
    
    if method == 'auto':
        method = GRACE_DATA_TYPE
    elif method == 'gaussian':  # Legacy compatibility
        method = 'raw_grace'
    
    if method == 'mascon':
        print("  🎯 Processing Mascon data (preserving JPL's advanced processing)")
        
        # Mascon data is ALREADY optimally processed with geophysical constraints
        # JPL explicitly states: "do not need to be de-correlated or smoothed"
        # Just resample directly to preserve all available information
        
        return grace_data.rio.reproject_match(
            reference_raster,
            resampling=rasterio.enums.Resampling.bilinear  # Preserves smooth variations
        )
    
    elif method == 'raw_grace':
        print("  🔬 Processing raw GRACE with smoothing (for spherical harmonic data)")
        
        # Original logic for raw GRACE spherical harmonic data
        GRACE_RESOLUTION = 3.0  # degrees (~300km)
        
        # First resample to target resolution
        grace_fine = grace_data.rio.reproject_match(
            reference_raster,
            resampling=rasterio.enums.Resampling.bilinear
        )
        
        # Apply Gaussian smoothing to represent GRACE's footprint
        target_res = abs(float(reference_raster.rio.resolution()[0]))
        sigma_pixels = GRACE_RESOLUTION / target_res / 2
        
        print(f"    Smoothing with sigma={sigma_pixels:.1f} pixels (~300km footprint)")
        
        smoothed = ndimage.gaussian_filter(
            np.nan_to_num(grace_fine.values, nan=0),
            sigma=sigma_pixels,
            mode='constant'
        )
        
        # Restore NaN mask
        smoothed[np.isnan(grace_fine.values)] = np.nan
        
        result = grace_fine.copy()
        result.values = smoothed
        return result
    
    elif method == 'conservative':
        print("  🔧 Using conservative (area-weighted) resampling")
        return grace_data.rio.reproject_match(
            reference_raster,
            resampling=rasterio.enums.Resampling.average
        )
    
    elif method == 'bilinear':
        print("  📊 Using simple bilinear resampling")
        return grace_data.rio.reproject_match(
            reference_raster,
            resampling=rasterio.enums.Resampling.bilinear
        )
    
    else:
        raise ValueError(f"Unknown GRACE resampling method: {method}")

def create_grace_weight_mask(reference_shape, reference_resolution, min_weight=0.2):
    """
    UPDATED: Create weight mask appropriate for detected data type.
    """
    global GRACE_DATA_TYPE
    
    if GRACE_DATA_TYPE == "mascon":
        # Mascon data has much better spatial representation
        # Create weights based on 55.66 km effective resolution
        mascon_spacing_km = MASCON_RESOLUTION_KM
        mascon_spacing_deg = mascon_spacing_km / 111  # Convert km to degrees
        
        spacing_pixels = mascon_spacing_deg / reference_resolution
        
        # Less aggressive weighting since Mascon is already well-constrained
        min_weight = 0.7  # Higher minimum weight
        
    else:
        # Original logic for raw GRACE
        GRACE_RESOLUTION = 3.0  # degrees
        spacing_pixels = GRACE_RESOLUTION / reference_resolution
        min_weight = 0.2  # Lower minimum weight due to uncertainty
    
    # Create grid of measurement centers
    n_lat, n_lon = reference_shape
    lat_centers = np.arange(spacing_pixels/2, n_lat, spacing_pixels)
    lon_centers = np.arange(spacing_pixels/2, n_lon, spacing_pixels)
    
    # Initialize weight mask
    weights = np.zeros((n_lat, n_lon))
    
    # Add Gaussian weight for each measurement center
    sigma = spacing_pixels / 4  # Adjusted for data type
    
    for lat_c in lat_centers:
        for lon_c in lon_centers:
            lat_grid, lon_grid = np.ogrid[:n_lat, :n_lon]
            dist_sq = (lat_grid - lat_c)**2 + (lon_grid - lon_c)**2
            gaussian = np.exp(-dist_sq / (2 * sigma**2))
            weights = np.maximum(weights, gaussian)
    
    # Normalize to [min_weight, 1]
    weights = weights * (1 - min_weight) + min_weight
    
    return weights

def validate_grace_values(grace_data, warn_threshold=None):
    """
    UPDATED: Validate GRACE values with appropriate thresholds for data type.
    """
    global GRACE_DATA_TYPE
    
    valid_data = grace_data[~np.isnan(grace_data)]
    if len(valid_data) == 0:
        return
    
    min_val, max_val = valid_data.min(), valid_data.max()
    
    # Adjust thresholds based on detected data type
    if GRACE_DATA_TYPE == "mascon":
        # Mascon data can have higher variability due to better resolution
        expected_range = "[-60, 60] cm"
        extreme_threshold = warn_threshold or 100
    else:
        # Raw GRACE typically has lower extremes due to smoothing
        expected_range = "[-40, 40] cm"  
        extreme_threshold = warn_threshold or 80
    
    if abs(min_val) > extreme_threshold or abs(max_val) > extreme_threshold:
        print(f"  ⚠️ GRACE values outside expected range: [{min_val:.1f}, {max_val:.1f}] cm")
        print(f"     Expected range for {GRACE_DATA_TYPE}: approximately {expected_range}")
        print(f"     Check units (should be cm water equivalent)")
    else:
        print(f"  ✅ GRACE values in reasonable range: [{min_val:.1f}, {max_val:.1f}] cm")

def get_data_type(path_or_name):
    """Determine data type from path or name for appropriate processing."""
    name_lower = str(path_or_name).lower()
    
    if any(cat in name_lower for cat in ['land_cover', 'modis', 'lc_type']):
        return 'categorical'
    elif any(pop in name_lower for pop in ['population', 'landscan']):
        return 'population'
    elif any(elev in name_lower for elev in ['elevation', 'dem', 'srtm']):
        return 'elevation'
    elif any(precip in name_lower for precip in ['precipitation', 'chirps', 'pr']):
        return 'precipitation'
    elif 'grace' in name_lower:
        return 'grace'
    else:
        return 'continuous'

def get_resampling_method(data_name):
    """Get appropriate resampling method for data type - UPDATED for integration."""
    data_type = get_data_type(data_name)
    
    # Map data types to resampling methods
    resampling_map = {
        'categorical': rasterio.enums.Resampling.nearest,
        'population': rasterio.enums.Resampling.sum,
        'elevation': rasterio.enums.Resampling.cubic,
        'precipitation': rasterio.enums.Resampling.average,
        'grace': rasterio.enums.Resampling.bilinear,  # Will be handled by resample_grace_scientifically
        'continuous': rasterio.enums.Resampling.bilinear
    }
    
    return resampling_map.get(data_type, rasterio.enums.Resampling.bilinear)

def print_resampling_summary():
    """Print summary of resampling methods being used."""
    global GRACE_DATA_TYPE
    
    print(f"\n🎯 GRACE Processing Configuration: {GRACE_DATA_TYPE.upper()}")
    print("="*50)
    
    if GRACE_DATA_TYPE == "mascon":
        print("  🎯 Using JPL Mascon RL06.3Mv04:")
        print(f"    - Native resolution: {get_config('grace_native_resolution_km', 55.66)} km")
        print("    - Already includes geophysical constraints")
        print("    - NO additional smoothing applied")
        print("    - Preserves all JPL processing benefits")
        print("    - Direct resampling with bilinear interpolation")
        
    elif GRACE_DATA_TYPE == "raw_grace":
        print("  🔬 Using raw GRACE spherical harmonic data:")
        print("    - Native resolution: ~300 km")
        print("    - Applying Gaussian smoothing")
        print("    - Representing instrument footprint")
        
    print("\n  Other data resampling:")
    print("    - Land cover: Nearest neighbor (preserves classes)")
    print("    - Population: Sum-preserving aggregation")
    print("    - Elevation: Cubic interpolation")
    print("    - Precipitation: Area-weighted average")
    print("    - Other continuous: Bilinear interpolation")

# For backward compatibility - keep the USE_SCIENTIFIC_GRACE flag
USE_SCIENTIFIC_GRACE = True