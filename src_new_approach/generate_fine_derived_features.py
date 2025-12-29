"""
Generate Derived Features at Fine (5km) Resolution

This script creates anomalies, temporal lags, spatial lags, and precipitation
accumulations at fine resolution to match the coarse feature set.

This enables scale-consistent neural network training with all 96 features.

Usage:
    python src_new_approach/generate_fine_derived_features.py
    
Output:
    processed_coarse_to_fine/feature_stack_all_5km.nc
"""

import numpy as np
import xarray as xr
import os
from pathlib import Path
from typing import List, Tuple
import warnings
from tqdm import tqdm
import gc

warnings.filterwarnings('ignore')

from src_new_approach.utils_downscaling import load_config, get_config_value


def add_seasonal_anomaly_features(temporal_stack: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Add seasonal anomaly features.
    
    For each climate variable, compute: value - monthly_climatology
    """
    anomaly_candidates = ['tmean', 'pr', 'chirps', 'Evap_tavg', 'aet', 'def', 'SWE_inst']
    
    # Find candidate features (avoid creating anomalies of anomalies)
    anomaly_indices = []
    anomaly_names = []
    for i, name in enumerate(feature_names):
        name_str = str(name)
        if any(candidate in name_str for candidate in anomaly_candidates):
            if not any(suffix in name_str for suffix in ['_anom', '_lag', '_spatial', '_accum']):
                anomaly_indices.append(i)
                anomaly_names.append(name_str)
    
    if not anomaly_indices:
        print("   No anomaly candidates found")
        return temporal_stack, feature_names
    
    print(f"   Creating anomalies for {len(anomaly_indices)} features: {anomaly_names[:3]}...")
    
    n_times = temporal_stack.shape[0]
    new_features = []
    new_names = []
    
    for feat_idx, feat_name in zip(anomaly_indices, anomaly_names):
        feature_data = temporal_stack[:, feat_idx, :, :]
        
        # Calculate monthly climatology (12 months)
        climatology = np.zeros((12, feature_data.shape[1], feature_data.shape[2]), dtype=np.float32)
        
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
        new_stack = np.stack(new_features, axis=1)
        temporal_stack = np.concatenate([temporal_stack, new_stack], axis=1)
        feature_names = list(feature_names) + new_names
    
    return temporal_stack, feature_names


def add_temporal_lag_features(temporal_stack: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Add temporal lag features (1, 3, 6 months).
    
    Groundwater responds to past climate conditions.
    """
    lag_candidates = ['tmean', 'pr', 'chirps', 'Evap_tavg', 'aet', 'def', 'SWE_inst']
    lag_months = [1, 3, 6]
    
    # Find candidates (avoid creating lags of lags)
    lag_indices = []
    lag_names = []
    for i, name in enumerate(feature_names):
        name_str = str(name)
        if any(candidate in name_str for candidate in lag_candidates):
            if not any(suffix in name_str for suffix in ['_lag', '_anom', '_spatial', '_accum']):
                lag_indices.append(i)
                lag_names.append(name_str)
    
    if not lag_indices:
        print("   No lag candidates found")
        return temporal_stack, feature_names
    
    print(f"   Creating lags for {len(lag_indices)} features × {len(lag_months)} periods...")
    
    n_times = temporal_stack.shape[0]
    new_features = []
    new_names = []
    
    for feat_idx, feat_name in zip(lag_indices, lag_names):
        for lag in lag_months:
            if lag < n_times:
                lagged_data = np.zeros_like(temporal_stack[:, feat_idx, :, :])
                lagged_data[lag:] = temporal_stack[:-lag, feat_idx, :, :]
                lagged_data[:lag] = np.nan  # NaN for initial periods
                
                new_features.append(lagged_data)
                new_names.append(f"{feat_name}_lag{lag}")
    
    if new_features:
        new_stack = np.stack(new_features, axis=1)
        temporal_stack = np.concatenate([temporal_stack, new_stack], axis=1)
        feature_names = list(feature_names) + new_names
    
    return temporal_stack, feature_names


def add_spatial_lag_features(temporal_stack: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Add spatial lag features (3x3 neighborhood mean).
    
    Captures regional effects beyond point values.
    Memory-efficient implementation using convolution.
    """
    from scipy.ndimage import uniform_filter
    
    spatial_candidates = ['tmean', 'pr', 'chirps', 'Evap_tavg', 'aet', 'def', 'SWE_inst']
    
    # Find candidates including anomalies (spatial of anomaly is useful)
    spatial_indices = []
    spatial_names = []
    for i, name in enumerate(feature_names):
        name_str = str(name)
        if any(candidate in name_str for candidate in spatial_candidates):
            # Include base and anomaly features, but not lags
            if '_lag' not in name_str and '_spatial' not in name_str and '_accum' not in name_str:
                spatial_indices.append(i)
                spatial_names.append(name_str)
    
    if not spatial_indices:
        print("   No spatial candidates found")
        return temporal_stack, feature_names
    
    print(f"   Creating spatial features for {len(spatial_indices)} features...")
    
    new_features = []
    new_names = []
    
    # Process each feature
    for feat_idx, feat_name in tqdm(zip(spatial_indices, spatial_names), 
                                      total=len(spatial_indices), 
                                      desc="   Spatial features"):
        feature_data = temporal_stack[:, feat_idx, :, :]
        
        # Apply 3x3 uniform filter (mean of neighbors)
        # Process time steps efficiently
        spatial_mean = np.zeros_like(feature_data)
        for t in range(feature_data.shape[0]):
            # uniform_filter with size=3 computes 3x3 mean
            spatial_mean[t] = uniform_filter(feature_data[t], size=3, mode='nearest')
        
        new_features.append(spatial_mean)
        new_names.append(f"{feat_name}_spatial")
    
    if new_features:
        new_stack = np.stack(new_features, axis=1)
        temporal_stack = np.concatenate([temporal_stack, new_stack], axis=1)
        feature_names = list(feature_names) + new_names
    
    return temporal_stack, feature_names


def add_precipitation_accumulation_features(temporal_stack: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Add precipitation accumulation features (3, 6, 12 month totals).
    
    Long-term precipitation patterns affect groundwater.
    """
    precip_candidates = ['pr', 'chirps']
    accumulation_periods = [3, 6, 12]
    
    # Find candidates
    precip_indices = []
    precip_names = []
    for i, name in enumerate(feature_names):
        name_str = str(name)
        if any(candidate in name_str for candidate in precip_candidates):
            if not any(suffix in name_str for suffix in ['_lag', '_anom', '_spatial', '_accum']):
                precip_indices.append(i)
                precip_names.append(name_str)
    
    if not precip_indices:
        print("   No precipitation candidates found")
        return temporal_stack, feature_names
    
    print(f"   Creating accumulations for {len(precip_indices)} precip variables × {len(accumulation_periods)} periods...")
    
    n_times = temporal_stack.shape[0]
    new_features = []
    new_names = []
    
    for feat_idx, feat_name in zip(precip_indices, precip_names):
        for period in accumulation_periods:
            if period <= n_times:
                # Calculate rolling sum
                accum_data = np.zeros_like(temporal_stack[:, feat_idx, :, :])
                
                for t in range(n_times):
                    start_idx = max(0, t - period + 1)
                    accum_data[t] = np.nansum(temporal_stack[start_idx:t+1, feat_idx, :, :], axis=0)
                
                new_features.append(accum_data)
                new_names.append(f"{feat_name}_accum{period}m")
    
    if new_features:
        new_stack = np.stack(new_features, axis=1)
        temporal_stack = np.concatenate([temporal_stack, new_stack], axis=1)
        feature_names = list(feature_names) + new_names
    
    return temporal_stack, feature_names


def generate_fine_derived_features(config: dict, output_path: str = None) -> xr.Dataset:
    """
    Generate all derived features at fine (5km) resolution.
    
    This matches the feature engineering done at coarse resolution.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    output_path : str, optional
        Output path (defaults to config path)
    
    Returns:
    --------
    xr.Dataset
        Enhanced fine features with all derived features
    """
    print("\n" + "="*70)
    print("🚀 GENERATING DERIVED FEATURES AT FINE (5km) RESOLUTION")
    print("="*70)
    
    # Get paths
    fine_features_path = get_config_value(config, 'paths.feature_stack_fine')
    if output_path is None:
        output_path = fine_features_path.replace('.nc', '_all.nc')
        output_path = output_path.replace('feature_stack_5km', 'feature_stack_all_5km')
    
    print(f"📂 Input: {fine_features_path}")
    print(f"📂 Output: {output_path}")
    
    # Check if output already exists
    if Path(output_path).exists():
        print(f"\n⚠️ Output file already exists: {output_path}")
        print("   Loading existing file...")
        return xr.open_dataset(output_path)
    
    # Load fine features
    print("\n📥 Loading fine features...")
    fine_ds = xr.open_dataset(fine_features_path)
    
    print(f"   Time steps: {len(fine_ds.time)}")
    print(f"   Spatial: {len(fine_ds.lat)} × {len(fine_ds.lon)}")
    
    # Get temporal and static features
    if 'features' in fine_ds and 'static_features' in fine_ds:
        temporal_data = fine_ds['features'].values  # (time, feature, lat, lon)
        static_data = fine_ds['static_features'].values  # (static_feature, lat, lon)
        temporal_names = [str(n) for n in fine_ds.feature.values]
        static_names = [str(n) for n in fine_ds.static_feature.values]
    else:
        raise ValueError("Expected 'features' and 'static_features' in dataset")
    
    n_times, n_features_orig, n_lat, n_lon = temporal_data.shape
    print(f"   Original temporal features: {n_features_orig}")
    print(f"   Static features: {len(static_names)}")
    
    # Estimate memory usage
    estimated_features = n_features_orig * 6  # rough estimate after derivation
    estimated_gb = (n_times * estimated_features * n_lat * n_lon * 4) / (1024**3)
    print(f"\n📊 Estimated memory: ~{estimated_gb:.1f} GB for temporal features")
    
    # Add derived features
    feature_names = temporal_names.copy()
    
    print("\n🔄 Step 1: Adding seasonal anomaly features...")
    temporal_data, feature_names = add_seasonal_anomaly_features(temporal_data, feature_names)
    gc.collect()
    
    print(f"\n🔄 Step 2: Adding temporal lag features...")
    temporal_data, feature_names = add_temporal_lag_features(temporal_data, feature_names)
    gc.collect()
    
    print(f"\n🔄 Step 3: Adding spatial lag features...")
    temporal_data, feature_names = add_spatial_lag_features(temporal_data, feature_names)
    gc.collect()
    
    print(f"\n🔄 Step 4: Adding precipitation accumulation features...")
    temporal_data, feature_names = add_precipitation_accumulation_features(temporal_data, feature_names)
    gc.collect()
    
    print(f"\n📊 Enhanced temporal features: {len(feature_names)} (was {n_features_orig})")
    print(f"   New features added: {len(feature_names) - n_features_orig}")
    
    # Create output dataset
    print("\n💾 Creating output dataset...")
    
    enhanced_ds = xr.Dataset(
        {
            'features': (['time', 'feature', 'lat', 'lon'], temporal_data.astype(np.float32)),
            'static_features': (['static_feature', 'lat', 'lon'], static_data.astype(np.float32))
        },
        coords={
            'time': fine_ds.time,
            'feature': feature_names,
            'lat': fine_ds.lat,
            'lon': fine_ds.lon,
            'static_feature': static_names
        }
    )
    
    # Copy attributes (convert booleans to strings for NetCDF compatibility)
    enhanced_ds.attrs = fine_ds.attrs.copy()
    enhanced_ds.attrs['derived_features_added'] = 'true'  # NetCDF doesn't support bool
    enhanced_ds.attrs['feature_engineering'] = 'anomalies,lags,spatial,accumulations'
    enhanced_ds.attrs['total_features'] = int(len(feature_names) + len(static_names))
    
    # Save
    print(f"\n💾 Saving to: {output_path}")
    
    # Create directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save with compression
    encoding = {
        'features': {'dtype': 'float32', 'zlib': True, 'complevel': 4},
        'static_features': {'dtype': 'float32', 'zlib': True, 'complevel': 4}
    }
    
    enhanced_ds.to_netcdf(output_path, encoding=encoding)
    
    # Verify
    saved_size = Path(output_path).stat().st_size / (1024**3)
    print(f"✅ Saved: {saved_size:.2f} GB")
    
    # Clean up
    fine_ds.close()
    
    return enhanced_ds


def main():
    """Main function to generate fine derived features."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate derived features at fine (5km) resolution"
    )
    parser.add_argument(
        '--config',
        default='src_new_approach/config_coarse_to_fine.yaml',
        help='Configuration file path'
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Output path (optional)'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Generate features
    enhanced_ds = generate_fine_derived_features(config, args.output)
    
    # Print summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    print(f"   Temporal features: {len(enhanced_ds.feature)}")
    print(f"   Static features: {len(enhanced_ds.static_feature)}")
    print(f"   Total features: {len(enhanced_ds.feature) + len(enhanced_ds.static_feature)}")
    print(f"   Time steps: {len(enhanced_ds.time)}")
    print(f"   Spatial: {len(enhanced_ds.lat)} × {len(enhanced_ds.lon)}")
    
    enhanced_ds.close()


if __name__ == "__main__":
    main()

