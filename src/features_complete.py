# features_complete.py - Creates features for ALL available months, not just GRACE months
"""
Enhanced feature creation that processes ALL months with auxiliary data.
This enables complete gap-filling by not limiting to GRACE availability.
"""

import os
import numpy as np
import rioxarray
import xarray as xr
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import rasterio.enums  # For resampling methods


def get_all_months(start_date="2003-01", end_date="2022-12"):
    """Generate all months in the study period."""
    dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    return [d.strftime('%Y-%m') for d in dates]


def get_chirps_reference_raster(chirps_dir):
    """Use CHIRPS as reference instead of GRACE"""
    chirps_files = sorted(os.listdir(chirps_dir))
    chirps_path = os.path.join(chirps_dir, chirps_files[0])
    reference_raster = rioxarray.open_rasterio(chirps_path, masked=True).squeeze()
    return reference_raster


def load_feature_array(folder_path, months, suffix=".tif", filename_style="numeric", reference_raster=None):
    """Load features for all specified months."""
    arr_list = []
    valid_months = []
    
    if filename_style == "numeric":
        # For indexed files (0.tif, 1.tif, etc.)
        all_files = sorted(os.listdir(folder_path), key=lambda x: int(os.path.splitext(x)[0]))
        
        # Map month to index (assuming files start from 2003-01)
        start_date = pd.to_datetime("2003-01")
        
        for month_str in months:
            month_date = pd.to_datetime(month_str)
            month_idx = (month_date.year - start_date.year) * 12 + (month_date.month - start_date.month)
            
            fname = f"{month_idx}{suffix}"
            fpath = os.path.join(folder_path, fname)
            
            if os.path.exists(fpath) and month_idx < len(all_files):
                try:
                    raster = rioxarray.open_rasterio(fpath, masked=True).squeeze()
                    if reference_raster is not None:
                        raster = raster.rio.reproject_match(reference_raster)
                    arr_list.append(raster.values)
                    valid_months.append(month_str)
                except Exception as e:
                    print(f"  ⚠️ Error loading {fname} for {month_str}: {e}")
            else:
                print(f"  ⚠️ Missing: {fname} for month {month_str}")
                
    elif filename_style == "yyyymm":
        # For date-named files (200301.tif, etc.)
        for month_str in months:
            fname = f"{month_str.replace('-', '')}{suffix}"
            fpath = os.path.join(folder_path, fname)
            
            if os.path.exists(fpath):
                try:
                    raster = rioxarray.open_rasterio(fpath, masked=True).squeeze()
                    if reference_raster is not None:
                        raster = raster.rio.reproject_match(reference_raster)
                    arr_list.append(raster.values)
                    valid_months.append(month_str)
                except Exception as e:
                    print(f"  ⚠️ Error loading {fname}: {e}")
            else:
                print(f"  ⚠️ Missing: {fname}")
    
    if len(arr_list) > 0:
        return np.stack(arr_list, axis=0), valid_months, raster
    else:
        return None, [], None


def load_static_features(static_dirs, reference_raster):
    """Load static features."""
    static_arrays = []
    static_names = []

    for name, path in static_dirs.items():
        try:
            raster = rioxarray.open_rasterio(path, masked=True).squeeze()
            if reference_raster is not None:
                if "landscan" in name.lower():
                    raster = raster.rio.reproject_match(
                        reference_raster,
                        resampling=rasterio.enums.Resampling.sum
                    )
                else:
                    raster = raster.rio.reproject_match(reference_raster)
            static_arrays.append(raster.values)
            static_names.append(name)
        except Exception as e:
            print(f"  ❌ Static {name}: {e}")

    if len(static_arrays) > 0:
        static_stack = np.stack(static_arrays, axis=0)
        return static_stack, static_names
    else:
        return None, []


def main():
    """Create features for ALL months with auxiliary data."""
    print("🚀 COMPLETE FEATURE CREATION")
    print("="*60)
    print("Creating features for ALL months with auxiliary data (not limited to GRACE)")
    
    # Get ALL months in study period
    all_months = get_all_months("2003-01", "2022-12")
    print(f"\n📅 Target period: 2003-01 to 2022-12 ({len(all_months)} months)")
    
    # Use CHIRPS as spatial reference (5km resolution, complete coverage)
    reference_raster = get_chirps_reference_raster("data/raw/chirps")
    print(f"✅ Using CHIRPS reference shape: {reference_raster.shape}")
    
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
    
    # Process each dataset
    all_features = {}
    feature_names = []
    
    for group, names in datasets.items():
        for name in names:
            folder = os.path.join(base_dir, group, name)
            feature_key = name if name else group
            
            print(f"\n📦 Processing {feature_key}...")
            
            try:
                if group in ["gldas", "chirps"]:
                    arr, valid_months, _ = load_feature_array(
                        folder, all_months, filename_style="numeric", 
                        reference_raster=reference_raster
                    )
                elif group == "terraclimate":
                    arr, valid_months, _ = load_feature_array(
                        folder, all_months, filename_style="yyyymm", 
                        reference_raster=reference_raster
                    )
                else:
                    continue
                
                if arr is not None:
                    all_features[feature_key] = {
                        'data': arr,
                        'months': valid_months
                    }
                    feature_names.append(feature_key)
                    print(f"  ✅ Loaded {len(valid_months)}/{len(all_months)} months")
                else:
                    print(f"  ❌ Failed to load data")
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
    
    # Find common valid months across all features
    if all_features:
        common_months = set(all_features[feature_names[0]]['months'])
        for fname in feature_names[1:]:
            common_months = common_months.intersection(set(all_features[fname]['months']))
        
        common_months = sorted(list(common_months))
        print(f"\n📊 Common months across all features: {len(common_months)}")
        
        if len(common_months) < len(all_months):
            missing = set(all_months) - set(common_months)
            print(f"⚠️  Missing {len(missing)} months due to incomplete auxiliary data")
            
            # Show missing periods
            missing_sorted = sorted(list(missing))
            if missing_sorted:
                print("\nMissing months:")
                current_gap = [missing_sorted[0]]
                
                for i in range(1, len(missing_sorted)):
                    if (pd.to_datetime(missing_sorted[i]) - pd.to_datetime(missing_sorted[i-1])).days < 35:
                        current_gap.append(missing_sorted[i])
                    else:
                        if len(current_gap) > 1:
                            print(f"  • {current_gap[0]} to {current_gap[-1]} ({len(current_gap)} months)")
                        else:
                            print(f"  • {current_gap[0]}")
                        current_gap = [missing_sorted[i]]
                
                if current_gap:
                    if len(current_gap) > 1:
                        print(f"  • {current_gap[0]} to {current_gap[-1]} ({len(current_gap)} months)")
                    else:
                        print(f"  • {current_gap[0]}")
    else:
        print("❌ No features loaded!")
        return
    
    # Create aligned feature arrays using common months
    print(f"\n🔄 Creating aligned feature stack for {len(common_months)} months...")
    
    feature_arrays = []
    aligned_datasets = []
    
    for fname in feature_names:
        feature_data = all_features[fname]
        
        # Get indices for common months
        month_indices = [feature_data['months'].index(m) for m in common_months 
                        if m in feature_data['months']]
        
        # Extract data for common months
        aligned_data = feature_data['data'][month_indices]
        feature_arrays.append(aligned_data)
        aligned_datasets.append(fname)
        
        print(f"  📐 {fname}: shape = {aligned_data.shape}")
    
    # Stack features
    try:
        temporal_stack = np.stack(feature_arrays, axis=1)  # shape: (time, feature, lat, lon)
        print(f"\n✅ Final stacked shape: {temporal_stack.shape}")
    except Exception as e:
        print(f"❌ Stack failed: {e}")
        return
    
    # Load static features
    static_files = {
        "modis_land_cover": os.path.join(base_dir, "modis_land_cover", "2003_01_01.tif"),
        "usgs_dem": os.path.join(base_dir, "usgs_dem", "srtm_dem.tif"),
        "landscan_population": os.path.join(base_dir, "landscan", "2003.tif"),
    }
    
    # Add OpenLandMap soil data
    for depth in ["0cm", "10cm", "30cm", "60cm", "100cm", "200cm"]:
        for var in ["sand", "clay"]:
            name = f"{var}_{depth}"
            path = os.path.join(base_dir, "openlandmap", f"{name}.tif")
            static_files[name] = path
    
    static_stack, static_names = load_static_features(static_files, reference_raster)
    
    # Create xarray dataset
    coords = {
        "time": common_months,
        "feature": aligned_datasets,
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
    
    # Add metadata
    ds.attrs = {
        "description": "Complete feature stack for GRACE downscaling (all available months)",
        "creation_date": datetime.now().isoformat(),
        "n_months": len(common_months),
        "n_features": len(aligned_datasets),
        "reference_crs": "EPSG:4326",
        "note": "Not limited to GRACE availability - includes all months with auxiliary data"
    }
    
    # Save
    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/feature_stack_complete.nc"
    ds.to_netcdf(output_path)
    print(f"\n✅ Saved complete feature stack to {output_path}")
    print(f"   Months: {len(common_months)} (was {206})")
    print(f"   Features: {len(aligned_datasets)}")
    print(f"   Spatial: {temporal_stack.shape[2]} x {temporal_stack.shape[3]}")
    
    # Compare with original
    original_path = "data/processed/feature_stack.nc"
    if os.path.exists(original_path):
        original = xr.open_dataset(original_path)
        print(f"\n📊 Comparison with original:")
        print(f"   Original: {len(original.time)} months")
        print(f"   Complete: {len(common_months)} months")
        print(f"   Gained: {len(common_months) - len(original.time)} months")


if __name__ == "__main__":
    main()