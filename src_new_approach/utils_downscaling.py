"""
Utility functions for coarse-to-fine downscaling approach
"""

import numpy as np
import xarray as xr
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings('ignore')


def load_config(config_path: str = "src_new_approach/config_coarse_to_fine.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_config_value(config: Dict, key_path: str, default=None):
    """
    Get nested config value using dot notation.
    
    Example: get_config_value(config, 'paths.models', 'models')
    """
    keys = key_path.split('.')
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


def create_output_directories(config: Dict):
    """Create all output directories specified in config."""
    paths_to_create = [
        'paths.processed_data',
        'paths.models',
        'paths.results',
        'paths.figures',
        'paths.logs'
    ]
    
    for path_key in paths_to_create:
        path = get_config_value(config, path_key)
        if path:
            Path(path).mkdir(parents=True, exist_ok=True)
            print(f"✓ Created directory: {path}")


def get_aggregation_factor(config: Dict) -> int:
    """
    Calculate aggregation factor from resolutions.
    
    Returns factor for downsampling (e.g., 11 for 55km/5km)
    """
    grace_res = get_config_value(config, 'resolution.grace_native_deg', 0.5)
    fine_res = get_config_value(config, 'resolution.fine_resolution_deg', 0.05)
    factor = int(np.round(grace_res / fine_res))
    return factor


def coarsen_2d(data: np.ndarray, factor: int, method: str = 'mean') -> np.ndarray:
    """
    Coarsen 2D array by aggregation factor.
    
    Parameters:
    -----------
    data : np.ndarray
        2D array to coarsen
    factor : int
        Aggregation factor (e.g., 11 for 55km/5km)
    method : str
        Aggregation method: 'mean', 'sum', 'mode', 'median'
    
    Returns:
    --------
    np.ndarray
        Coarsened array
    """
    if data.ndim != 2:
        raise ValueError("Input must be 2D array")
    
    ny, nx = data.shape
    
    # Calculate new dimensions
    ny_coarse = ny // factor
    nx_coarse = nx // factor
    
    # Trim data to be evenly divisible
    data_trimmed = data[:ny_coarse*factor, :nx_coarse*factor]
    
    # Reshape for aggregation
    reshaped = data_trimmed.reshape(ny_coarse, factor, nx_coarse, factor)
    
    # Apply aggregation method
    if method == 'mean':
        result = np.nanmean(reshaped, axis=(1, 3))
    elif method == 'sum':
        result = np.nansum(reshaped, axis=(1, 3))
    elif method == 'median':
        result = np.nanmedian(reshaped, axis=(1, 3))
    elif method == 'mode':
        # For categorical data
        from scipy.stats import mode
        # Flatten the aggregation dimensions
        flat = reshaped.reshape(ny_coarse, nx_coarse, -1)
        result = np.zeros((ny_coarse, nx_coarse))
        for i in range(ny_coarse):
            for j in range(nx_coarse):
                valid_data = flat[i, j, ~np.isnan(flat[i, j, :])]
                if len(valid_data) > 0:
                    result[i, j] = mode(valid_data, keepdims=False)[0]
                else:
                    result[i, j] = np.nan
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
    
    return result


def refine_2d(data: np.ndarray, factor: int, method: str = 'bilinear') -> np.ndarray:
    """
    Refine (upsample) 2D array by interpolation.
    
    Parameters:
    -----------
    data : np.ndarray
        2D coarse array to refine
    factor : int
        Refinement factor (e.g., 11 for 55km→5km)
    method : str
        Interpolation method: 'bilinear', 'cubic', 'nearest'
    
    Returns:
    --------
    np.ndarray
        Refined array
    """
    if data.ndim != 2:
        raise ValueError("Input must be 2D array")
    
    ny_coarse, nx_coarse = data.shape
    ny_fine = ny_coarse * factor
    nx_fine = nx_coarse * factor
    
    # Create coordinate grids
    y_coarse = np.arange(ny_coarse)
    x_coarse = np.arange(nx_coarse)
    yy_coarse, xx_coarse = np.meshgrid(y_coarse, x_coarse, indexing='ij')
    
    # Fine grid coordinates
    y_fine = np.linspace(0, ny_coarse - 1, ny_fine)
    x_fine = np.linspace(0, nx_coarse - 1, nx_fine)
    yy_fine, xx_fine = np.meshgrid(y_fine, x_fine, indexing='ij')
    
    # Get valid (non-NaN) points
    valid_mask = ~np.isnan(data)
    if not np.any(valid_mask):
        return np.full((ny_fine, nx_fine), np.nan)
    
    points = np.column_stack([yy_coarse[valid_mask], xx_coarse[valid_mask]])
    values = data[valid_mask]
    
    # Interpolate
    if method == 'bilinear':
        result = griddata(points, values, (yy_fine, xx_fine), method='linear')
    elif method == 'cubic':
        result = griddata(points, values, (yy_fine, xx_fine), method='cubic')
    elif method == 'nearest':
        result = griddata(points, values, (yy_fine, xx_fine), method='nearest')
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    
    return result


def smooth_2d(data: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Apply Gaussian smoothing to 2D array (handles NaNs).
    
    Parameters:
    -----------
    data : np.ndarray
        2D array to smooth
    sigma : float
        Gaussian kernel sigma
    
    Returns:
    --------
    np.ndarray
        Smoothed array
    """
    # Handle NaNs
    valid_mask = ~np.isnan(data)
    
    if not np.any(valid_mask):
        return data
    
    # Fill NaNs with 0 for filtering
    data_filled = np.where(valid_mask, data, 0)
    
    # Apply Gaussian filter
    smoothed = gaussian_filter(data_filled, sigma=sigma)
    
    # Also filter the mask to normalize
    mask_filtered = gaussian_filter(valid_mask.astype(float), sigma=sigma)
    
    # Normalize and restore NaNs
    result = np.where(mask_filtered > 0, smoothed / mask_filtered, np.nan)
    
    return result


def clip_outliers(data: np.ndarray, n_std: float = 3.0) -> np.ndarray:
    """
    Clip outliers beyond n standard deviations.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data
    n_std : float
        Number of standard deviations
    
    Returns:
    --------
    np.ndarray
        Clipped data
    """
    valid_data = data[~np.isnan(data)]
    if len(valid_data) == 0:
        return data
    
    mean = np.mean(valid_data)
    std = np.std(valid_data)
    
    lower = mean - n_std * std
    upper = mean + n_std * std
    
    return np.clip(data, lower, upper)


def add_metadata_to_dataset(ds: xr.Dataset, config: Dict, step_name: str) -> xr.Dataset:
    """
    Add comprehensive metadata to xarray Dataset.
    
    Parameters:
    -----------
    ds : xr.Dataset
        Dataset to add metadata to
    config : Dict
        Configuration dictionary
    step_name : str
        Name of processing step (for tracking)
    
    Returns:
    --------
    xr.Dataset
        Dataset with metadata
    """
    from datetime import datetime
    import platform
    
    if not get_config_value(config, 'scientific.add_metadata', True):
        return ds
    
    # Add global attributes
    ds.attrs['title'] = 'GRACE Downscaled Groundwater Storage Anomalies'
    ds.attrs['institution'] = get_config_value(config, 'scientific.metadata.institution', 'ORISE')
    ds.attrs['source'] = get_config_value(config, 'scientific.metadata.source', 'GRACE + ML')
    ds.attrs['method'] = get_config_value(config, 'scientific.metadata.method', 'Coarse-to-fine downscaling')
    ds.attrs['processing_step'] = step_name
    ds.attrs['creation_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ds.attrs['python_version'] = platform.python_version()
    ds.attrs['random_seed'] = get_config_value(config, 'reproducibility.random_seed', 42)
    
    # Add coordinate metadata
    if 'lat' in ds.coords:
        ds['lat'].attrs['units'] = 'degrees_north'
        ds['lat'].attrs['long_name'] = 'Latitude'
    
    if 'lon' in ds.coords:
        ds['lon'].attrs['units'] = 'degrees_east'
        ds['lon'].attrs['long_name'] = 'Longitude'
    
    if 'time' in ds.coords:
        ds['time'].attrs['long_name'] = 'Time'
    
    return ds


def calculate_statistics(data: np.ndarray) -> Dict:
    """
    Calculate comprehensive statistics for data array.
    
    Returns dictionary with mean, std, min, max, median, etc.
    """
    valid_data = data[~np.isnan(data)]
    
    if len(valid_data) == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
            'median': np.nan,
            'q25': np.nan,
            'q75': np.nan,
            'n_valid': 0,
            'n_total': len(data.flatten()),
            'pct_valid': 0.0
        }
    
    return {
        'mean': float(np.mean(valid_data)),
        'std': float(np.std(valid_data)),
        'min': float(np.min(valid_data)),
        'max': float(np.max(valid_data)),
        'median': float(np.median(valid_data)),
        'q25': float(np.percentile(valid_data, 25)),
        'q75': float(np.percentile(valid_data, 75)),
        'n_valid': int(len(valid_data)),
        'n_total': int(len(data.flatten())),
        'pct_valid': float(100 * len(valid_data) / len(data.flatten()))
    }


def print_statistics(data: np.ndarray, name: str = "Data"):
    """Pretty print data statistics."""
    stats = calculate_statistics(data)
    print(f"\n📊 Statistics for {name}:")
    print(f"   Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
    print(f"   Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    print(f"   Median: {stats['median']:.4f} (Q25={stats['q25']:.4f}, Q75={stats['q75']:.4f})")
    print(f"   Valid: {stats['n_valid']:,} / {stats['n_total']:,} ({stats['pct_valid']:.1f}%)")


def match_spatial_grids(data_fine: xr.DataArray, data_coarse: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Ensure two DataArrays have compatible spatial grids.
    
    Interpolates fine to coarse grid if needed.
    """
    # Check if grids already match
    if (data_fine.lat.shape == data_coarse.lat.shape and 
        data_fine.lon.shape == data_coarse.lon.shape):
        return data_fine, data_coarse
    
    # Interpolate fine to coarse grid
    data_fine_regridded = data_fine.interp(
        lat=data_coarse.lat,
        lon=data_coarse.lon,
        method='linear'
    )
    
    return data_fine_regridded, data_coarse


def get_feature_aggregation_method(feature_name: str, config: Dict) -> str:
    """
    Determine aggregation method for a feature based on config rules.
    
    Parameters:
    -----------
    feature_name : str
        Name of the feature
    config : Dict
        Configuration dictionary
    
    Returns:
    --------
    str
        Aggregation method: 'mean', 'sum', 'mode'
    """
    import re
    
    agg_methods = get_config_value(config, 'feature_aggregation.aggregation_methods', {})
    
    # Check each method's patterns
    for method, patterns in agg_methods.items():
        for pattern in patterns:
            if re.match(pattern, feature_name):
                return method
    
    # Default to mean
    return 'mean'


def setup_logging(config: Dict, log_name: str = 'pipeline') -> 'logging.Logger':
    """Setup logging configuration."""
    import logging
    from datetime import datetime
    
    log_level = get_config_value(config, 'logging.level', 'INFO')
    log_format = get_config_value(config, 'logging.format', 
                                   '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    save_to_file = get_config_value(config, 'logging.save_to_file', True)
    
    # Create logger
    logger = logging.getLogger(log_name)
    logger.setLevel(getattr(logging, log_level))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if save_to_file:
        log_dir = Path(get_config_value(config, 'paths.logs', 'logs_coarse_to_fine'))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = log_dir / f'{log_name}_{timestamp}.log'
        
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(getattr(logging, log_level))
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_filename}")
    
    return logger


def save_checkpoint(data: Dict, checkpoint_path: str, logger=None):
    """Save intermediate results as checkpoint."""
    import pickle
    
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(data, f)
    
    if logger:
        logger.info(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(checkpoint_path: str, logger=None) -> Optional[Dict]:
    """Load checkpoint if it exists."""
    import pickle
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        if logger:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return None
    
    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)
    
    if logger:
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
    
    return data

