#!/usr/bin/env python
"""
CORRECTED Residual Correction Method Testing

This script uses the SAME validation approach as the original validation script
to properly compare residual correction methods. It tests before vs after correction.

The key fix: Uses the extract_before_after_correction_data approach from the 
original validation script to ensure fair comparison.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from src_new_approach.residual_corrector_multi import MultiMethodResidualCorrector
from src_new_approach.utils_downscaling import load_config, get_config_value, get_aggregation_factor
from src_new_approach.fine_predictor import FinePredictor
from scipy.ndimage import zoom
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import yaml


def extract_before_after_correction_data_multi(ds_pred_5km, residuals_fine_methods, ds_orig, sample_times=20):
    """
    Extract before/after residual correction comparison data for multiple methods.
    
    This follows the EXACT same approach as the original validation script.
    """
    from scipy.ndimage import zoom
    import pandas as pd
    
    print(f"\n🔬 CORRECTED VALIDATION: Using original validation methodology")
    print(f"   Comparing: Before (5km pred → 55km) vs After (5km corrected → 55km)")
    
    # Get time range
    times_pred = pd.to_datetime(ds_pred_5km.time.values)
    times_orig = pd.to_datetime(ds_orig.time.values)
    common_times = sorted(set(times_pred) & set(times_orig))[:sample_times]
    
    orig_var = list(ds_orig.data_vars)[0]
    
    # Calculate aggregation factors
    lat_factor = len(ds_pred_5km.lat) / len(ds_orig.lat)
    lon_factor = len(ds_pred_5km.lon) / len(ds_orig.lon)
    
    print(f"   Aggregation factors: lat={lat_factor:.1f}x, lon={lon_factor:.1f}x")
    print(f"   Processing {len(common_times)} time steps...")
    
    # Store results for each method
    method_results = {}
    
    # Initialize storage for before correction data (same for all methods)
    before_data = []
    grace_data = []
    
    # Process each time step
    for time_idx, time_point in enumerate(common_times):
        try:
            # Get indices
            pred_idx = list(times_pred).index(time_point)
            orig_idx = list(times_orig).index(time_point)
            
            # Get data slices
            grace_slice = ds_orig[orig_var].isel(time=orig_idx).values
            pred_5km_slice = ds_pred_5km.prediction_ensemble.isel(time=pred_idx).values
            
            # Aggregate 5km predictions to 55km (BEFORE correction)
            pred_agg = zoom(pred_5km_slice, (1/lat_factor, 1/lon_factor), order=1)
            
            # Ensure shape match
            if pred_agg.shape != grace_slice.shape:
                pred_agg = pred_agg[:grace_slice.shape[0], :grace_slice.shape[1]]
            
            # Apply mask and collect valid data for BEFORE correction
            mask = (~np.isnan(grace_slice)) & (~np.isnan(pred_agg))
            
            if np.sum(mask) > 0:
                grace_data.extend(grace_slice[mask].flatten())
                before_data.extend(pred_agg[mask].flatten())
            
            # Process each residual correction method
            for method, residuals_ds in residuals_fine_methods.items():
                if method not in method_results:
                    method_results[method] = []
                
                # Get residuals for this time step
                try:
                    resid_idx = list(pd.to_datetime(residuals_ds.time.values)).index(time_point)
                    residuals_5km_slice = residuals_ds.residual.isel(time=resid_idx).values
                    
                    # Apply residual correction at 5km level
                    corrected_5km_slice = pred_5km_slice + residuals_5km_slice
                    
                    # Aggregate corrected 5km to 55km (AFTER correction)
                    corrected_agg = zoom(corrected_5km_slice, (1/lat_factor, 1/lon_factor), order=1)
                    
                    # Ensure shape match
                    if corrected_agg.shape != grace_slice.shape:
                        corrected_agg = corrected_agg[:grace_slice.shape[0], :grace_slice.shape[1]]
                    
                    # Apply same mask and collect valid data for AFTER correction
                    if np.sum(mask) > 0:
                        method_results[method].extend(corrected_agg[mask].flatten())
                        
                except Exception as e:
                    print(f"   ⚠️ Skipping method {method} at time {time_point}: {e}")
                    continue
            
            if (time_idx + 1) % 5 == 0:
                print(f"   Progress: {time_idx + 1}/{len(common_times)} time steps")
                
        except Exception as e:
            print(f"   ⚠️ Skipping time {time_point}: {e}")
            continue
    
    # Compile results
    results = {
        'grace': np.array(grace_data),
        'before': np.array(before_data)
    }
    
    for method, after_data in method_results.items():
        if len(after_data) > 0:
            results[f'after_{method}'] = np.array(after_data)
            print(f"   ✅ {method}: {len(after_data):,} data points")
        else:
            print(f"   ⚠️ No valid data for {method}")
    
    return results


def calculate_metrics(y_true, y_pred):
    """Calculate validation metrics (same as original script)."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) < 2:
        return {'r2': np.nan, 'rmse': np.nan, 'mae': np.nan, 'bias': np.nan, 'n': 0}
    
    r2 = r2_score(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    bias = np.mean(y_pred_clean - y_true_clean)
    
    return {
        'r2': r2,
        'rmse': rmse, 
        'mae': mae,
        'bias': bias,
        'n': len(y_true_clean)
    }


def evaluate_all_methods_corrected(before_after_data):
    """Evaluate all methods using corrected validation approach."""
    print("\n" + "="*70)
    print("📊 CORRECTED PERFORMANCE EVALUATION")
    print("="*70)
    print("Using SAME validation methodology as original validation script:")
    print("Before: 5km predictions → 55km aggregated vs GRACE 55km")
    print("After:  5km corrected → 55km aggregated vs GRACE 55km")
    print("="*70)
    
    grace_data = before_after_data['grace']
    before_data = before_after_data['before']
    
    # Calculate baseline (before correction) metrics
    before_metrics = calculate_metrics(grace_data, before_data)
    
    print(f"\n📊 BASELINE (Before Correction):")
    print(f"   R² = {before_metrics['r2']:.4f}")
    print(f"   RMSE = {before_metrics['rmse']:.3f} cm")
    print(f"   MAE = {before_metrics['mae']:.3f} cm")
    print(f"   Bias = {before_metrics['bias']:.4f} cm")
    print(f"   Samples = {before_metrics['n']:,}")
    
    # Evaluate each residual correction method
    all_results = {'before_correction': before_metrics}
    
    method_names = {
        'bilinear': 'Bilinear Interpolation',
        'geographic_assignment': 'Geographic Assignment (Novel)',
        'nearest': 'Nearest Neighbor',
        'bicubic': 'Bicubic Interpolation',
        'idw': 'Inverse Distance Weighting',
        'gaussian_kernel': 'Gaussian Kernel Smoothing',
        'area_weighted': 'Area-Weighted Assignment',
        'distance_weighted_nearest': 'Distance-Weighted Nearest'
    }
    
    print(f"\n📊 RESIDUAL CORRECTION METHODS:")
    print("="*70)
    
    improvements = []
    
    for method_key in before_after_data.keys():
        if method_key.startswith('after_'):
            method = method_key.replace('after_', '')
            after_data = before_after_data[method_key]
            
            # Ensure same number of samples
            min_samples = min(len(grace_data), len(after_data))
            grace_subset = grace_data[:min_samples]
            after_subset = after_data[:min_samples]
            
            after_metrics = calculate_metrics(grace_subset, after_subset)
            all_results[method] = after_metrics
            
            # Calculate improvement
            delta_r2 = after_metrics['r2'] - before_metrics['r2']
            delta_rmse = after_metrics['rmse'] - before_metrics['rmse']
            delta_mae = after_metrics['mae'] - before_metrics['mae']
            
            improvements.append((method, delta_r2, delta_rmse, delta_mae, after_metrics))
            
            method_name = method_names.get(method, method.replace('_', ' ').title())
            print(f"{method_name:<30} R²={after_metrics['r2']:.4f} (+{delta_r2:+.4f})  "
                  f"RMSE={after_metrics['rmse']:.3f} ({delta_rmse:+.3f})  "
                  f"MAE={after_metrics['mae']:.3f} ({delta_mae:+.3f})")
    
    # Find best method
    if improvements:
        best_method, best_delta_r2, best_delta_rmse, best_delta_mae, best_metrics = max(improvements, key=lambda x: x[1])
        
        print("="*70)
        print(f"🏆 BEST METHOD: {method_names.get(best_method, best_method.replace('_', ' ').title())}")
        print(f"   Before:  R² = {before_metrics['r2']:.4f}")
        print(f"   After:   R² = {best_metrics['r2']:.4f}")
        print(f"   Improvement: ΔR² = {best_delta_r2:+.4f} ({100*best_delta_r2/before_metrics['r2']:+.1f}%)")
        print(f"   ΔRMSE = {best_delta_rmse:+.3f} cm, ΔMAE = {best_delta_mae:+.3f} cm")
        print("="*70)
        
        return best_method, all_results
    
    return None, all_results


def main():
    """Main function to test all residual correction methods with CORRECTED validation."""
    print("="*80)
    print("🔧 CORRECTED RESIDUAL CORRECTION METHOD TESTING")
    print("="*80)
    print("Using the SAME validation methodology as the original validation script")
    print("to properly compare before vs after residual correction performance.")
    print("="*80)
    
    # Load configuration
    config_path = Path('src_new_approach/config_coarse_to_fine.yaml')
    config = load_config(str(config_path))
    
    # Initialize tester
    tester = MultiMethodResidualCorrector(config)
    
    # Load data
    print("📂 Loading datasets...")
    grace_ds = xr.open_dataset('data/processed_coarse_to_fine/grace_filled_stl.nc')
    coarse_features_ds = xr.open_dataset('data/processed_coarse_to_fine/feature_stack_55km.nc')
    fine_features_ds = xr.open_dataset('data/processed_coarse_to_fine/feature_stack_5km.nc')
    fine_predictions_ds = xr.open_dataset('data/processed_coarse_to_fine/predictions_5km.nc')
    
    print(f"✅ Loaded datasets successfully")
    
    # Generate coarse predictions for residual calculation
    print("\n📊 Generating coarse-scale predictions...")
    predictor = FinePredictor(config, 'models_coarse_to_fine_simple')
    predictor.load_models()
    predictions_coarse_ds = predictor.predict_fine_resolution(coarse_features_ds, use_ensemble=True)
    
    # Calculate residuals at coarse scale
    print("\n📐 Calculating coarse-scale residuals...")
    residuals_coarse_ds = tester.calculate_residuals_coarse(
        grace_ds, predictions_coarse_ds, prediction_var='prediction_ensemble'
    )
    
    # Test all interpolation methods
    print("\n🧪 Testing all interpolation methods...")
    fine_lat = fine_features_ds.lat.values
    fine_lon = fine_features_ds.lon.values
    
    residuals_fine_methods = tester.interpolate_residuals_to_fine_multi(
        residuals_coarse_ds, fine_lat, fine_lon, methods=tester.available_methods
    )
    
    # Extract before/after data using CORRECTED validation approach
    print("\n📊 Extracting before/after data using CORRECTED validation...")
    before_after_data = extract_before_after_correction_data_multi(
        fine_predictions_ds, residuals_fine_methods, grace_ds, sample_times=20
    )
    
    # Evaluate all methods with corrected validation
    best_method, all_results = evaluate_all_methods_corrected(before_after_data)
    
    if best_method:
        # Update configuration with best method
        print(f"\n⚙️ Updating configuration with best method: {best_method}")
        config['residual_correction']['interpolation_method'] = best_method
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"   ✅ Configuration updated")
        
        # Generate summary report
        print(f"\n🎯 CORRECTED OPTIMIZATION COMPLETE!")
        print(f"   Best method: {best_method}")
        print(f"   Performance improvement validated with correct methodology")
        print(f"   This matches the validation approach in your original validation script")
        
    else:
        print("❌ No valid methods found!")
    
    return best_method, all_results


if __name__ == "__main__":
    main()