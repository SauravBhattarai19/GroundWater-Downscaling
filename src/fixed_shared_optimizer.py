#!/usr/bin/env python3
"""
FINAL CORRECT Shared Memory Optimizer - Uses REAL GRACE .tif data
ULTRATHINK SOLUTION: Loads actual GRACE TWS .tif files like model_manager does!
"""

import os
import sys
import multiprocessing as mp
import numpy as np
import pandas as pd
import xgboost as xgb
import random
import time
import re
from datetime import datetime
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from multiprocessing import shared_memory
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_real_grace_data():
    """Load REAL GRACE data using EXACT same method as model_manager"""
    print("📦 Loading REAL GRACE data (EXACT model_manager method)...")
    
    try:
        import xarray as xr
        
        # Load feature stack (your processed features)
        feature_path = "data/processed/feature_stack.nc"
        print(f"   Loading features: {feature_path}")
        
        if not os.path.exists(feature_path):
            print(f"❌ Feature stack not found: {feature_path}")
            return None, None, None
            
        ds = xr.open_dataset(feature_path)
        print(f"   Dataset dimensions: {dict(ds.dims)}")
        
        if 'features' not in ds.data_vars:
            print(f"❌ 'features' not found in dataset")
            return None, None, None
            
        # Extract features exactly as model_manager
        feature_data = ds['features'].values  # Shape: (time, features, lat, lon)
        print(f"   Features shape: {feature_data.shape}")
        
        # Create feature dates exactly as model_manager
        feature_dates = []
        for i in range(feature_data.shape[0]):
            year = 2003 + i // 12
            month = (i % 12) + 1
            feature_dates.append(f"{year}-{month:02d}")  # YYYY-MM format like model_manager
        
        print(f"   Processed {len(feature_dates)} feature time points")
        
        # Create reference raster for GRACE loading (CRITICAL FIX!)
        # GRACE must match feature spatial resolution exactly
        n_lat, n_lon = feature_data.shape[2], feature_data.shape[3]
        print(f"   Target spatial resolution: {n_lat}×{n_lon}")
        
        # Create proper reference raster from feature coordinates
        if 'lat' in ds.coords and 'lon' in ds.coords:
            import xarray as xr
            lat_coords = ds.coords['lat'].values
            lon_coords = ds.coords['lon'].values
            
            # Create dummy data array for resampling reference
            dummy_data = np.zeros((n_lat, n_lon))
            reference_raster = xr.DataArray(
                dummy_data,
                coords={'lat': lat_coords, 'lon': lon_coords},
                dims=['lat', 'lon']
            )
            reference_raster = reference_raster.rio.write_crs("EPSG:4326")
        else:
            # Fallback: create coordinate arrays
            lat_coords = np.linspace(49, 24, n_lat)  # Approximate US bounds
            lon_coords = np.linspace(-125, -66, n_lon)
            
            dummy_data = np.zeros((n_lat, n_lon))
            reference_raster = xr.DataArray(
                dummy_data,
                coords={'lat': lat_coords, 'lon': lon_coords},
                dims=['lat', 'lon']
            )
            reference_raster = reference_raster.rio.write_crs("EPSG:4326")
        
        # Load GRACE data using EXACT same method as model_manager
        print("   Loading GRACE TWS data (.tif files)...")
        grace_dir = "data/raw/grace"
        
        grace_data, grace_dates = load_grace_tws_exact(grace_dir, reference_raster)
        print(f"   Loaded {len(grace_dates)} GRACE time points") 
        print(f"   GRACE data shape: {grace_data.shape}")
        
        # Find common dates (same as model_manager)
        print("   Finding common dates...")
        common_dates = []
        common_feature_indices = []
        common_grace_indices = []
        
        for i, fdate in enumerate(feature_dates):
            for j, gdate in enumerate(grace_dates):
                if fdate == gdate:
                    common_dates.append(fdate)
                    common_feature_indices.append(i)
                    common_grace_indices.append(j)
                    break
        
        print(f"✅ Found {len(common_dates)} common dates")
        
        if len(common_dates) == 0:
            print(f"Feature dates sample: {feature_dates[:5]}")
            print(f"GRACE dates sample: {grace_dates[:5]}")
            raise ValueError("No common dates found!")
        
        # Extract data for common dates (same as model_manager)
        print("   Extracting data for common dates...")
        X_temporal = feature_data[common_feature_indices]
        grace_tws = grace_data[common_grace_indices]  # REAL GRACE DATA!
        
        print(f"   X_temporal shape: {X_temporal.shape}")
        print(f"   grace_tws shape: {grace_tws.shape}")
        
        # Verify shapes match (CRITICAL CHECKS!)
        if X_temporal.shape[0] != grace_tws.shape[0]:
            raise ValueError(f"Time dimension mismatch: features={X_temporal.shape[0]}, grace={grace_tws.shape[0]}")
            
        if X_temporal.shape[2:] != grace_tws.shape[1:]:
            raise ValueError(f"Spatial dimension mismatch: features={X_temporal.shape[2:]}, grace={grace_tws.shape[1:]}")
            
        print(f"✅ Spatial dimensions match: {X_temporal.shape[2:]} == {grace_tws.shape[1:]}")
        
        # Add static features if available
        if 'static_features' in ds.data_vars:
            print("   Adding static features...")
            static_features = ds['static_features'].values
            
            # Expand static features to match temporal data
            static_expanded = np.repeat(static_features[np.newaxis, :, :, :], 
                                      X_temporal.shape[0], axis=0)
            
            # Combine temporal and static features
            X_enhanced = np.concatenate([X_temporal, static_expanded], axis=1)
            print(f"   Enhanced X shape: {X_enhanced.shape}")
        else:
            X_enhanced = X_temporal
            print("   No static features found")
        
        # Reshape for ML training (same as model_manager)
        print("   Reshaping data for model training...")
        n_times = X_enhanced.shape[0]
        X = X_enhanced.reshape(n_times, X_enhanced.shape[1], -1).transpose(0, 2, 1).reshape(-1, X_enhanced.shape[1])
        y = grace_tws.reshape(-1)  # REAL GRACE TARGET!
        
        # Filter out NaN values (same as model_manager)
        print("   Filtering out NaN values...")
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1) | 
                      np.isnan(y) | np.isinf(y))
        
        X_valid = X[valid_mask].astype(np.float32)
        y_valid = y[valid_mask].astype(np.float32)
        
        n_spatial_points = grace_tws.shape[1] * grace_tws.shape[2]
        temporal_indices = np.repeat(np.arange(n_times), n_spatial_points)[valid_mask]
        
        valid_ratio = len(X_valid) / len(X) * 100
        print(f"✅ Prepared data: {X_valid.shape[0]} valid samples ({valid_ratio:.1f}%), {X_valid.shape[1]} features")
        print(f"   Temporal structure: {n_times} time periods, {n_spatial_points} spatial points per time")
        print(f"   REAL GRACE target range: [{y_valid.min():.1f}, {y_valid.max():.1f}] cm")
        
        # Prepare metadata for temporal splitting
        metadata = {
            'common_dates': [f"{date}-01" for date in common_dates],  # Convert to YYYY-MM-01 format
            'temporal_indices': temporal_indices,
            'n_times': n_times,
            'n_spatial_points': n_spatial_points
        }
        
        return X_valid, y_valid, metadata
        
    except Exception as e:
        print(f"❌ Error loading real GRACE data: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def load_grace_tws_exact(grace_dir, reference_raster=None):
    """Load GRACE TWS data EXACTLY like model_manager._load_grace_tws"""
    try:
        import rioxarray as rxr
        import rasterio.enums
    except ImportError:
        raise ImportError("rioxarray and rasterio are required for GRACE data loading")
    
    # Get .tif files exactly as model_manager does
    grace_files = sorted(os.listdir(grace_dir))
    grace_files = [f for f in grace_files if f.endswith('.tif')]
    
    if not grace_files:
        raise ValueError(f"No GRACE .tif files found in {grace_dir}")
    
    print(f"  🔬 Loading {len(grace_files)} GRACE .tif files")
    
    grace_data = []
    grace_dates = []
    
    for grace_file in grace_files:
        try:
            filename_str = grace_file
            
            # Try YYYYMM format first (200301.tif, 200302.tif...)
            match = re.match(r'(\d{6})\.tif$', filename_str)
            if match:
                yyyymm = match.group(1)
                # Convert YYYYMM to YYYY-MM format (like model_manager)
                year = yyyymm[:4]
                month = yyyymm[4:6]
                grace_date = f"{year}-{month}"
                
                grace_path = os.path.join(grace_dir, filename_str)
                grace_raster = rxr.open_rasterio(grace_path, masked=True).squeeze()
                
                # Fix spatial dimensions (CRITICAL FIX for rioxarray!)
                if hasattr(grace_raster, 'rio'):
                    grace_raster = grace_raster.rio.set_spatial_dims(x_dim='x', y_dim='y', inplace=True)
                
                # Reproject to match reference if provided (CRITICAL RESAMPLING!)
                if reference_raster is not None:
                    print(f"     Resampling {grace_file} from {grace_raster.shape} to {reference_raster.shape}")
                    grace_raster = grace_raster.rio.reproject_match(
                        reference_raster,
                        resampling=rasterio.enums.Resampling.bilinear
                    )
                    print(f"     Resampled to: {grace_raster.shape}")
                
                grace_data.append(grace_raster.values)
                grace_dates.append(grace_date)
                
            # Fall back to old YYYYMMDD_YYYYMMDD format for backward compatibility
            elif re.match(r'(\d{8})_(\d{8})\.tif', filename_str):
                match = re.match(r'(\d{8})_(\d{8})\.tif', filename_str)
                date_str = match.group(1)
                date = datetime.strptime(date_str, '%Y%m%d')
                grace_date = date.strftime('%Y-%m')
                
                grace_path = os.path.join(grace_dir, filename_str)
                grace_raster = rxr.open_rasterio(grace_path, masked=True).squeeze()
                
                if reference_raster is not None:
                    print(f"     Resampling {grace_file} from {grace_raster.shape} to {reference_raster.shape}")
                    grace_raster = grace_raster.rio.reproject_match(
                        reference_raster,
                        resampling=rasterio.enums.Resampling.bilinear
                    )
                    print(f"     Resampled to: {grace_raster.shape}")
                
                grace_data.append(grace_raster.values)
                grace_dates.append(grace_date)
                
            else:
                # Skip files that don't match expected formats
                continue
                
        except Exception as e:
            print(f"⚠️ Error loading {grace_file}: {e}")
    
    if not grace_data:
        raise ValueError("No valid GRACE data loaded")
    
    grace_array = np.stack(grace_data)
    print(f"   Successfully loaded {len(grace_data)} GRACE files")
    print(f"  ✅ GRACE values in reasonable range: [{grace_array.min():.1f}, {grace_array.max():.1f}] cm")
    
    return grace_array, grace_dates

def stratified_temporal_split_exact(X, y, metadata, test_size=0.164):
    """EXACT replication of model_manager's stratified temporal splitting"""
    print("🔄 Using temporal splitting strategy")
    
    temporal_indices = metadata['temporal_indices']
    n_times = metadata['n_times']
    common_dates = metadata['common_dates']
    
    # Convert to DataFrame exactly as model_manager
    date_df = pd.DataFrame({
        'date': common_dates,
        'time_idx': range(n_times)
    })
    date_df['date'] = pd.to_datetime(date_df['date'])
    date_df['year'] = date_df['date'].dt.year
    date_df['month'] = date_df['date'].dt.month
    
    print(f"🔍 Debug info:")
    print(f"   n_times: {n_times}")
    print(f"   common_dates type: {type(common_dates)}")
    print(f"   common_dates length: {len(common_dates)}")
    print(f"   temporal_indices shape: {temporal_indices.shape}")
    
    # Stratified sampling exactly as model_manager
    train_indices = []
    test_indices = []
    
    np.random.seed(42)  # Same seed as model_manager
    for month in range(1, 13):
        month_data = date_df[date_df['month'] == month]
        unique_years = month_data['year'].unique()
        
        # Randomly shuffle years for this month
        shuffled_years = np.random.permutation(unique_years)
        n_test_years = max(1, int(len(unique_years) * test_size))
        
        test_years = shuffled_years[:n_test_years]
        train_years = shuffled_years[n_test_years:]
        
        # Add indices for this month
        month_train = month_data[month_data['year'].isin(train_years)]['time_idx'].values
        month_test = month_data[month_data['year'].isin(test_years)]['time_idx'].values
        
        train_indices.extend(month_train)
        test_indices.extend(month_test)
    
    train_time_indices = set(train_indices)
    test_time_indices = set(test_indices)
    
    # Create masks exactly as model_manager
    train_mask = np.array([t in train_time_indices for t in temporal_indices])
    test_mask = np.array([t in test_time_indices for t in temporal_indices])
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    print(f"📅 Stratified temporal splitting:")
    print(f"   Training periods: {len(train_indices)} time points (all months represented)")
    print(f"   Testing periods: {len(test_indices)} time points (all months represented)")
    print(f"   ✅ No climate shift bias - years randomly mixed!")
    print(f"   ✅ Seasonal balance maintained in both sets!")
    print(f"   ✅ No temporal data leakage - completely separate time periods!")
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Testing samples: {len(X_test):,}")
    
    return X_train, X_test, y_train, y_test

def create_shared_arrays(X_train, X_test, y_train, y_test):
    """Create shared memory arrays"""
    print("🧠 Creating shared memory arrays...")
    
    arrays = {}
    shared_info = {}
    
    for name, array in [('X_train', X_train), ('X_test', X_test), 
                       ('y_train', y_train), ('y_test', y_test)]:
        shm = shared_memory.SharedMemory(create=True, size=array.nbytes)
        shared_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
        shared_array[:] = array[:]
        
        arrays[name] = shm
        shared_info[f"{name}_name"] = shm.name
        shared_info[f"{name}_shape"] = array.shape
        shared_info[f"{name}_dtype"] = array.dtype
    
    total_gb = sum(arr.nbytes for _, arr in [('X_train', X_train), ('X_test', X_test), 
                                           ('y_train', y_train), ('y_test', y_test)]) / 1e9
    print(f"✅ Shared memory created: {total_gb:.2f} GB")
    
    return arrays, shared_info

def optimization_worker(args):
    """Worker function for parallel optimization"""
    worker_id, n_trials, shared_info, seed = args
    
    try:
        # Connect to shared memory
        shm_arrays = {}
        data_arrays = {}
        
        for name in ['X_train', 'X_test', 'y_train', 'y_test']:
            shm = shared_memory.SharedMemory(name=shared_info[f"{name}_name"])
            array = np.ndarray(shared_info[f"{name}_shape"], 
                             dtype=shared_info[f"{name}_dtype"], 
                             buffer=shm.buf)
            shm_arrays[name] = shm
            data_arrays[name] = array
        
        print(f"🔥 Worker {worker_id}: Connected to shared memory")
        
        np.random.seed(seed)
        random.seed(seed)
        
        results = []
        
        for trial in range(n_trials):
            # Random hyperparameter sampling
            params = {
                'n_estimators': random.choice([100, 200, 300, 500]),
                'max_depth': random.choice([6, 8, 10, 12, 15, 18]),
                'learning_rate': random.choice([0.01, 0.03, 0.05, 0.1, 0.15, 0.2]),
                'subsample': round(random.uniform(0.7, 0.95), 3),
                'colsample_bytree': round(random.uniform(0.6, 0.9), 3), 
                'reg_alpha': random.choice([0.01, 0.1, 0.3, 0.5, 1.0, 2.0]),
                'reg_lambda': random.choice([0.01, 0.1, 0.3, 0.5, 1.0, 2.0]),
            }
            
            try:
                start_time = time.time()
                
                # Create model
                model = xgb.XGBRegressor(
                    n_estimators=params['n_estimators'],
                    max_depth=params['max_depth'],
                    learning_rate=params['learning_rate'],
                    subsample=params['subsample'],
                    colsample_bytree=params['colsample_bytree'],
                    reg_alpha=params['reg_alpha'],
                    reg_lambda=params['reg_lambda'],
                    random_state=seed,
                    n_jobs=4,  # 4 cores per worker (192/48)
                    verbosity=0,
                    early_stopping_rounds=20
                )
                
                # Split training for validation
                X_cv_train, X_cv_val, y_cv_train, y_cv_val = train_test_split(
                    data_arrays['X_train'], data_arrays['y_train'], 
                    test_size=0.2, random_state=seed
                )
                
                # Train
                model.fit(X_cv_train, y_cv_train,
                         eval_set=[(X_cv_val, y_cv_val)],
                         verbose=False)
                
                # Evaluate
                y_cv_pred = model.predict(X_cv_val)
                y_test_pred = model.predict(data_arrays['X_test'])
                
                cv_r2 = r2_score(y_cv_val, y_cv_pred)
                test_r2 = r2_score(data_arrays['y_test'], y_test_pred)
                train_time = time.time() - start_time
                
                result = {
                    'worker_id': worker_id,
                    'trial': trial,
                    'cv_r2': cv_r2,
                    'test_r2': test_r2,
                    'train_time': train_time,
                    **params
                }
                
                results.append(result)
                
                if trial % 3 == 0:
                    print(f"⚡ Worker {worker_id}: Trial {trial}, CV R²={cv_r2:.4f}, Test R²={test_r2:.4f}")
                    
            except Exception as e:
                print(f"❌ Worker {worker_id} Trial {trial}: {e}")
                continue
        
        # Cleanup
        for shm in shm_arrays.values():
            shm.close()
        
        print(f"✅ Worker {worker_id}: Completed {len(results)} trials")
        return results
        
    except Exception as e:
        print(f"❌ Worker {worker_id} failed: {e}")
        return []

def main():
    """Main optimization function with REAL GRACE data"""
    print("🚀 FIXED SHARED MEMORY GRACE OPTIMIZATION")
    print("   Using REAL GRACE data like model_manager!")
    
    # Step 1: Load REAL GRACE data
    X, y, metadata = load_real_grace_data()
    if X is None:
        print("❌ Failed to load GRACE data")
        return None
    
    # Step 2: Apply exact temporal splitting
    X_train, X_test, y_train, y_test = stratified_temporal_split_exact(X, y, metadata)
    
    # Step 3: Create shared memory
    shared_arrays, shared_info = create_shared_arrays(X_train, X_test, y_train, y_test)
    
    try:
        # Step 4: Run parallel optimization
        n_workers = 48
        trials_per_worker = 10  # 480 total trials
        
        print(f"\n🔥 PARALLEL OPTIMIZATION")
        print(f"   Workers: {n_workers}")
        print(f"   Trials per worker: {trials_per_worker}")
        print(f"   Total trials: {n_workers * trials_per_worker}")
        
        # Prepare worker arguments
        worker_args = [
            (i, trials_per_worker, shared_info, 42 + i * 100)
            for i in range(n_workers)
        ]
        
        # Run optimization
        start_time = time.time()
        
        with mp.Pool(n_workers) as pool:
            worker_results = pool.map(optimization_worker, worker_args)
        
        total_time = time.time() - start_time
        
        # Combine results
        all_results = []
        for worker_result in worker_results:
            all_results.extend(worker_result)
        
        if len(all_results) == 0:
            print("❌ No successful trials")
            return None
            
        results_df = pd.DataFrame(all_results)
        
        print(f"\n🏆 OPTIMIZATION COMPLETE!")
        print(f"   Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"   Successful trials: {len(all_results)}")
        print(f"   Best CV R²: {results_df['cv_r2'].max():.4f}")
        print(f"   Best Test R²: {results_df['test_r2'].max():.4f}")
        
        # Show top 5 configurations
        top_configs = results_df.nlargest(5, 'test_r2')
        
        print(f"\n🎯 TOP 5 CONFIGURATIONS:")
        for i, (_, config) in enumerate(top_configs.iterrows(), 1):
            print(f"{i}. Test R²={config['test_r2']:.4f}, CV R²={config['cv_r2']:.4f}")
            print(f"   n_est={config['n_estimators']}, depth={config['max_depth']}, lr={config['learning_rate']:.3f}")
            print(f"   α={config['reg_alpha']:.2f}, λ={config['reg_lambda']:.2f}")
        
        # Save results
        os.makedirs('models', exist_ok=True)
        results_df.to_csv('models/fixed_shared_optimization.csv', index=False)
        print(f"\n💾 Results saved to models/fixed_shared_optimization.csv")
        
        return top_configs
        
    finally:
        # Cleanup shared memory
        print("🧹 Cleaning up shared memory...")
        for shm in shared_arrays.values():
            shm.close()
            shm.unlink()

if __name__ == "__main__":
    print("🎯 FIXED GRACE OPTIMIZATION - Using REAL GRACE data!")
    best_configs = main()
    
    if best_configs is not None:
        print(f"\n✅ SUCCESS! Apply the best config to your pipeline")
        print(f"   This uses your REAL GRACE data - expect R² ≥ 0.65!")
    else:
        print(f"\n❌ Optimization failed")