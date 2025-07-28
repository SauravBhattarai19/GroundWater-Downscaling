# src/utils.py - Enhanced version with scientific GRACE handling

import ee
import os
import re
import rioxarray as rxr
import xarray as xr
import numpy as np
import pandas as pd
from scipy import ndimage
import rasterio.enums

# Configuration flags
USE_SCIENTIFIC_GRACE = True  # Set to False for original behavior


def create_date_list(start="2003-01", end="2022-12"):
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
    return ee.Geometry.Rectangle([
        region["lon_min"],
        region["lat_min"],
        region["lon_max"],
        region["lat_max"]
    ])


def reproject_match(src, match):
    """Reproject src to match CRS of reference raster"""
    return src.rio.reproject_match(match)


def resample_match(src, match):
    """Resample src to match resolution and shape of reference raster"""
    return src.rio.reproject_match(match)


def match_resolution(src, match):
    """Alias: Match CRS, resolution, and alignment"""
    return reproject_match(src, match)


def parse_grace_months(grace_dir):
    """Parse valid GRACE months from filename ranges."""
    months = set()
    pattern = re.compile(r"(\d{8})_(\d{8})\.tif$")
    for fname in os.listdir(grace_dir):
        match = pattern.match(fname)
        if match:
            start = pd.to_datetime(match.group(1), format="%Y%m%d")
            end = pd.to_datetime(match.group(2), format="%Y%m%d")
            # Include months between start and end (but pick mid-month)
            mid = start + (end - start) / 2
            months.add(mid.strftime("%Y-%m"))
    return sorted(months)


def load_timestamp_map(grace_dir):
    """Load grace-based timestamp map for masking/filtering."""
    valid_months = parse_grace_months(grace_dir)
    return set(valid_months)


# ============= NEW SCIENTIFIC GRACE FUNCTIONS =============

def resample_grace_scientifically(grace_data, reference_raster, method='gaussian'):
    """
    Resample GRACE data with scientific integrity.
    
    Parameters:
    -----------
    grace_data : xarray.DataArray
        Original GRACE data (~50km resolution)
    reference_raster : xarray.DataArray  
        Target resolution grid
    method : str
        'gaussian' : Apply Gaussian smoothing to represent GRACE footprint
        'conservative' : Area-weighted to preserve total water mass
        'bilinear' : Original method (not recommended)
        
    Returns:
    --------
    xarray.DataArray
        Resampled GRACE data
    """
    if not USE_SCIENTIFIC_GRACE or method == 'bilinear':
        # Original behavior
        return grace_data.rio.reproject_match(
            reference_raster,
            resampling=rasterio.enums.Resampling.bilinear
        )
    
    # GRACE native resolution in degrees (approximately 3° or 300km)
    GRACE_RESOLUTION = 3.0
    
    if method == 'gaussian':
        print("  🔬 Using Gaussian smoothing for GRACE (scientifically accurate)")
        
        # First resample to target resolution
        grace_fine = grace_data.rio.reproject_match(
            reference_raster,
            resampling=rasterio.enums.Resampling.bilinear
        )
        
        # Calculate smoothing kernel size
        target_res = abs(float(reference_raster.rio.resolution()[0]))
        sigma_pixels = GRACE_RESOLUTION / target_res / 2  # Divide by 2 for smoother result
        
        print(f"    Smoothing with sigma={sigma_pixels:.1f} pixels (~{GRACE_RESOLUTION*111:.0f}km footprint)")
        
        # Apply Gaussian smoothing
        smoothed = ndimage.gaussian_filter(
            np.nan_to_num(grace_fine.values, nan=0),
            sigma=sigma_pixels,
            mode='constant'
        )
        
        # Restore NaN mask
        smoothed[np.isnan(grace_fine.values)] = np.nan
        
        # Create output array
        result = grace_fine.copy()
        result.values = smoothed
        
        return result
        
    elif method == 'conservative':
        print("  🔬 Using conservative resampling for GRACE (mass-preserving)")
        
        # Use average resampling as approximation of conservative
        return grace_data.rio.reproject_match(
            reference_raster,
            resampling=rasterio.enums.Resampling.average
        )
    
    else:
        raise ValueError(f"Unknown GRACE resampling method: {method}")


def create_grace_weight_mask(reference_shape, reference_resolution, min_weight=0.2):
    """
    Create weight mask for ML training based on GRACE measurement density.
    
    Areas between GRACE measurement centers have higher uncertainty
    and should have lower weight in model training.
    
    Parameters:
    -----------
    reference_shape : tuple
        (n_lat, n_lon) shape of target grid
    reference_resolution : float
        Resolution in degrees
    min_weight : float
        Minimum weight to assign (prevents zero weights)
        
    Returns:
    --------
    np.ndarray
        Weight mask same shape as reference
    """
    if not USE_SCIENTIFIC_GRACE:
        # Return uniform weights for compatibility
        return np.ones(reference_shape)
    
    GRACE_RESOLUTION = 3.0  # degrees
    
    # Calculate spacing in pixels
    grace_spacing_pixels = GRACE_RESOLUTION / reference_resolution
    
    # Create grid of GRACE "measurement centers"
    n_lat, n_lon = reference_shape
    lat_centers = np.arange(grace_spacing_pixels/2, n_lat, grace_spacing_pixels)
    lon_centers = np.arange(grace_spacing_pixels/2, n_lon, grace_spacing_pixels)
    
    # Initialize weight mask
    weights = np.zeros((n_lat, n_lon))
    
    # Add Gaussian weight for each measurement center
    sigma = grace_spacing_pixels / 3  # 3-sigma covers most of footprint
    
    for lat_c in lat_centers:
        for lon_c in lon_centers:
            lat_grid, lon_grid = np.ogrid[:n_lat, :n_lon]
            dist_sq = (lat_grid - lat_c)**2 + (lon_grid - lon_c)**2
            gaussian = np.exp(-dist_sq / (2 * sigma**2))
            weights = np.maximum(weights, gaussian)
    
    # Normalize to [min_weight, 1]
    weights = weights * (1 - min_weight) + min_weight
    
    return weights


def validate_grace_values(grace_data, warn_threshold=100):
    """
    Validate GRACE values are in reasonable range.
    
    Parameters:
    -----------
    grace_data : np.ndarray
        GRACE TWS anomalies
    warn_threshold : float
        Threshold for warning (cm)
    """
    valid_data = grace_data[~np.isnan(grace_data)]
    if len(valid_data) == 0:
        return
    
    min_val, max_val = valid_data.min(), valid_data.max()
    
    if abs(min_val) > warn_threshold or abs(max_val) > warn_threshold:
        print(f"  ⚠️ GRACE values outside expected range: [{min_val:.1f}, {max_val:.1f}] cm")
        print(f"     Expected range: approximately [-50, 50] cm")
        print(f"     Check units (should be cm water equivalent)")
    else:
        print(f"  ✅ GRACE values in reasonable range: [{min_val:.1f}, {max_val:.1f}] cm")


# ============= ENHANCED HELPER FUNCTIONS =============

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


def print_resampling_summary():
    """Print summary of resampling methods being used."""
    if USE_SCIENTIFIC_GRACE:
        print("\n🔬 Scientific Resampling Mode ENABLED")
        print("  - GRACE: Gaussian smoothing (300km footprint)")
        print("  - Land cover: Nearest neighbor (preserves classes)")
        print("  - Population: Sum-preserving aggregation")
        print("  - Elevation: Cubic interpolation")
        print("  - Precipitation: Area-weighted average")
        print("  - Other continuous: Bilinear interpolation")
    else:
        print("\n📊 Standard Resampling Mode")
        print("  - All data: Bilinear interpolation")
        print("  - Population: Sum-preserving aggregation")