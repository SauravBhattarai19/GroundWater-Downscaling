import os
import numpy as np
import rioxarray
import xarray as xr
from tqdm import tqdm
from src.utils import parse_grace_months
import rasterio.enums  # For resampling methods


def get_valid_grace_months(grace_dir):
    return parse_grace_months(grace_dir)

def get_chirps_reference_raster(chirps_dir):
    """Use CHIRPS as reference instead of GRACE"""
    chirps_files = sorted(os.listdir(chirps_dir))
    chirps_path = os.path.join(chirps_dir, chirps_files[0])
    reference_raster = rioxarray.open_rasterio(chirps_path, masked=True).squeeze()  # 5km
    return reference_raster

def get_grace_reference_raster(grace_dir, valid_months):
    grace_files = sorted(os.listdir(grace_dir))
    if len(grace_files) == 0:
        raise ValueError(f"No GRACE files found in {grace_dir}")
    grace_path = os.path.join(grace_dir, grace_files[0])
    reference_raster = rioxarray.open_rasterio(grace_path, masked=True).squeeze()
    print(f"✅ Using GRACE reference shape: {reference_raster.shape}")
    return reference_raster


def load_feature_array(folder_path, months, suffix=".tif", filename_style="numeric", reference_raster=None):
    arr_list = []

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
                    raster = raster.rio.reproject_match(reference_raster)
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
                    raster = raster.rio.reproject_match(reference_raster)
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
    static_arrays = []
    static_names = []

    for name, path in static_dirs.items():
        try:
            raster = rioxarray.open_rasterio(path, masked=True).squeeze()
            if reference_raster is not None:
                # Use sum-based resampling for population data
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
            print(f"❌ Static {name}: {e}")

    if len(static_arrays) > 0:
        static_stack = np.stack(static_arrays, axis=0)  # shape: (feature, lat, lon)
        return static_stack, static_names
    else:
        return None, []


def main():
    grace_dir = "data/raw/grace"
    valid_months = get_valid_grace_months(grace_dir)
    print(f"✅ Using {len(valid_months)} months aligned with GRACE")

    reference_raster = get_chirps_reference_raster("data/raw/chirps")
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

    for group, names in datasets.items():
        for name in tqdm(names, desc=f"📦 Stacking {group}"):
            folder = os.path.join(base_dir, group, name)
            try:
                if group in ["gldas", "chirps"]:
                    arr, _ = load_feature_array(folder, valid_months, filename_style="numeric", reference_raster=reference_raster)
                elif group == "terraclimate":
                    arr, _ = load_feature_array(folder, valid_months, filename_style="yyyymm", reference_raster=reference_raster)
                # Note: Static features (modis_land_cover, usgs_dem, landscan, openlandmap) will be handled separately
                else:
                    print(f"ℹ️ Skipping {group}/{name} - will process as static feature if available")
                    continue
                feature_arrays.append(arr)
                aligned_datasets.append(name if name else group)
                feature_names.extend([f"{name if name else group}_{m}" for m in valid_months])
            except Exception as e:
                print(f"❌ {name}: {e}")

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
        "landscan_population": os.path.join(base_dir, "landscan", "2003.tif"),  # Using 2003 as reference year
    }
    for depth in ["0cm", "10cm", "30cm", "60cm", "100cm", "200cm"]:
        for var in ["sand", "clay"]:
            name = f"{var}_{depth}"
            path = os.path.join(base_dir, "openlandmap", f"{name}.tif")
            static_files[name] = path

    static_stack, static_names = load_static_features(static_files, reference_raster)

    # Create 12 feature names only (one per variable)
    assert len(feature_arrays) == 12
    feature_labels = [name if name else "chirps" for name in aligned_datasets]

    coords = {
        "time": valid_months,
        "feature": feature_labels,
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

    os.makedirs("data/processed", exist_ok=True)
    ds.to_netcdf("data/processed/feature_stack.nc")
    print("✅ Saved feature_stack.nc")


if __name__ == "__main__":
    main()